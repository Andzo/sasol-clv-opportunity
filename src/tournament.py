"""Multi-cutoff out-of-fold tournament and stacking engine.

Orchestrates:
1. Rolling temporal out-of-fold matrix with a 3-month causal embargo between
   training cutoffs and each evaluation cutoff.
2. Multi-family model suite:
   - Direct regressors (Tweedie, Huber, MSE, Ridge, ExtraTrees)
   - Bagged and multi-quantile two-part hurdles
   - Basket x ticket decomposed regressors
   - Flat multiclass ensembles and 4-stage hierarchical opportunity classifier
3. Out-of-fold calibration:
   - Zero-intercept per-member linear debiasing
   - Cross-target non-negative ridge stacking
   - Customer Empirical Bayes (BLUP) residual adjustment
   - Physical hurdle exceedance and inactivity coupling
   - Prior-aligned expected-F1 decision solver via dual coordinate ascent
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd
import scipy.special as sp
from scipy.optimize import minimize

from src import calibration, metrics
from src.config import PipelineConfig
from src.dataset import Panel, Snapshot, SnapshotBuilder, stack
from src.hurdle import (
    BaggedTwoPartFuelHurdle,
    BaggedTwoPartNonfuelHurdle,
    MultiQuantileTwoPartHurdle,
    PhysicalBasketTicketRegressor,
)
from src.models import (
    ClassifierEnsemble,
    HierarchicalOpportunityClassifier,
    RegressorEnsemble,
    StoreAttachmentClassifier,
)

logger = logging.getLogger(__name__)


def build_level2_structural_features(
    X: pd.DataFrame,
    probs_base: np.ndarray,
    probs_hier: np.ndarray,
    pred_fuel: np.ndarray = None,
    pred_nf: np.ndarray = None,
    p_fuel_active: np.ndarray = None,
    p_store_active: np.ndarray = None,
    mu_fuel_pos: np.ndarray = None,
    sigma_fuel_pos: np.ndarray = None,
) -> pd.DataFrame:
    """Build level-2 structural meta-features combining model probabilities and regression outputs."""
    n = len(X)
    df = pd.DataFrame(index=X.index)

    for c in range(probs_base.shape[1]):
        df[f"p_base_{c}"] = probs_base[:, c]
        df[f"p_hier_{c}"] = probs_hier[:, c]

    p_avg = 0.5 * probs_base + 0.5 * probs_hier
    if p_avg.shape[1] > 16:
        df["meta_margin_fg_vs_stable"] = np.log((p_avg[:, 2] + 1e-6) / (p_avg[:, 16] + 1e-6))
        df["meta_margin_inact_vs_stable"] = np.log((p_avg[:, 4] + 1e-6) / (p_avg[:, 16] + 1e-6))
    else:
        df["meta_margin_fg_vs_stable"] = 0.0
        df["meta_margin_inact_vs_stable"] = 0.0
    df["meta_entropy"] = -np.sum(p_avg * np.log(np.clip(p_avg, 1e-9, 1.0)), axis=1)

    prior_q_cols = [
        "prior_q_was_eligible",
        "prior_q_is_inactive",
        "prior_q_is_stable",
        "prior_q_is_fuel_growth",
        "prior_q_is_retail_growth",
        "prior_q_is_adoption",
        "prior_q_winning_increment_log",
        "prior_q_n_qualifying_growth_cats",
        "prior_q_fuel_active",
        "prior_q_nf_active",
        "two_quarter_fuel_persistence",
    ]
    structural_pass_cols = [
        "m1_baskets", "m3_baskets", "d14_baskets", "d45_baskets", "active_months_m3",
        "days_since_last_purchase", "days_since_last_fuel", "days_since_last_nonfuel",
        "btyd_overall_p_alive", "btyd_fuel_p_alive", "btyd_nonfuel_p_alive",
        "store_attachment_rate", "m3_fuel_litres", "m3_fuel_rand", "m3_nonfuel_rand",
        "real_fuel_hurdle_litres", "real_fuel_hurdle_litres_price_adj",
        "active_m6_fuel", "ever_fuel",
        "rule6_exp_inc_fuel", "rule6_exp_inc_beverages", "rule6_exp_inc_food_beverage",
        "rule6_exp_inc_other_growth", "rule6_fuel_minus_best_retail_inc",
        "rule6_n_categories_clearing_hurdle", "rule6_n_unvisited_categories",
    ]
    for c in prior_q_cols + structural_pass_cols:
        if c in X.columns:
            df[c] = X[c].to_numpy()

    for cat in ("beverages", "food_beverage", "other"):
        col_m3 = f"m3spend_{cat}"
        col_m1 = f"m1spend_{cat}" if f"m1spend_{cat}" in X.columns else None
        col_q2 = f"q2spend_{cat}" if f"q2spend_{cat}" in X.columns else None
        if col_m3 in X.columns:
            m3_s = X[col_m3].to_numpy()
            h_cat = np.maximum(1.25 * m3_s, m3_s + 20.0)
            if col_m1:
                df[f"{cat}_m1_hurdle_ratio"] = (3.0 * X[col_m1].to_numpy() + 1.0) / (h_cat + 1.0)
            if col_q2:
                df[f"{cat}_q2_hurdle_ratio"] = (X[col_q2].to_numpy() + 1.0) / (h_cat + 1.0)

    if "survival_prob_empirical" in X.columns:
        df["survival_prob_empirical"] = X["survival_prob_empirical"].to_numpy()
    if "fuel_survival_prob" in X.columns:
        df["fuel_survival_prob"] = X["fuel_survival_prob"].to_numpy()
    if "max_other_subcat_expected_delta_rand" in X.columns:
        df["max_other_subcat_expected_delta_rand"] = X["max_other_subcat_expected_delta_rand"].to_numpy()
    if "max_other_subcat_hurdle_ratio" in X.columns:
        df["max_other_subcat_hurdle_ratio"] = X["max_other_subcat_hurdle_ratio"].to_numpy()

    return refresh_level2_predictions(
        df=df,
        pred_fuel=pred_fuel,
        pred_nf=pred_nf,
        p_fuel_active=p_fuel_active,
        p_store_active=p_store_active,
        mu_fuel_pos=mu_fuel_pos,
        sigma_fuel_pos=sigma_fuel_pos,
    )


def refresh_level2_predictions(
    df: pd.DataFrame,
    pred_fuel: np.ndarray,
    pred_nf: np.ndarray,
    p_fuel_active: np.ndarray,
    p_store_active: np.ndarray,
    mu_fuel_pos: np.ndarray,
    sigma_fuel_pos: np.ndarray,
) -> pd.DataFrame:
    """Recomputes all regression-dependent Level-2 columns from calibrated fuel and nonfuel predictions."""
    out = df.copy()
    n = len(out)
    out["pred_fuel"] = pred_fuel
    out["pred_nf"] = pred_nf
    out["p_fuel_active"] = p_fuel_active
    out["p_store_active"] = p_store_active
    out["mu_fuel_pos"] = mu_fuel_pos
    out["sigma_fuel_pos"] = sigma_fuel_pos

    litres_pred = np.expm1(np.minimum(pred_fuel * 2.352825386185113, 15.0))
    out["pred_litres"] = litres_pred

    m3_fuel_litres = out["m3_fuel_litres"].to_numpy() if "m3_fuel_litres" in out.columns else np.zeros(n)
    m3_fuel_rand = out["m3_fuel_rand"].to_numpy() if "m3_fuel_rand" in out.columns else np.zeros(n)
    ever_fuel = (out["active_m6_fuel"].to_numpy() > 0) if "active_m6_fuel" in out.columns else (
        (out["ever_fuel"].to_numpy() > 0) if "ever_fuel" in out.columns else np.ones(n, dtype=bool)
    )

    pump_price = np.clip(
        np.where(m3_fuel_litres > 0.1, m3_fuel_rand / np.maximum(m3_fuel_litres, 0.1), 23.50),
        20.5, 27.5,
    )
    if "real_fuel_hurdle_litres_price_adj" in out.columns:
        h_litres = out["real_fuel_hurdle_litres_price_adj"].to_numpy()
    elif "real_fuel_hurdle_litres" in out.columns:
        h_litres = out["real_fuel_hurdle_litres"].to_numpy()
    else:
        h_litres = np.maximum(1.25 * m3_fuel_litres, m3_fuel_litres + 20.0 / pump_price)

    out["fuel_hurdle_ratio"] = (litres_pred + 0.5) / (h_litres + 0.5)
    out["fuel_hurdle_margin_litres"] = litres_pred - h_litres

    z_hurdle = (1.0 / 2.352825386185113) * np.log1p(np.maximum(h_litres, 0.0))
    z_center = 0.65 * mu_fuel_pos + 0.35 * pred_fuel
    z_diff = (z_hurdle - z_center) / np.maximum(sigma_fuel_pos, 0.05)
    phi = 0.5 * (1.0 + sp.erf(z_diff / np.sqrt(2.0)))
    out["p_exceed_fuel"] = np.where(ever_fuel, p_fuel_active * (1.0 - phi), 0.0)
    out["joint_hurdle_inactivity"] = (1.0 - p_fuel_active) * (1.0 - p_store_active)

    fuel_future_rand = litres_pred * pump_price
    fuel_monetary_increment = np.maximum(0.0, fuel_future_rand - m3_fuel_rand)
    out["fuel_monetary_increment"] = fuel_monetary_increment

    nf_pred_rand = np.expm1(np.minimum(pred_nf * 2.4200904988927476, 15.0))
    m3_nf_rand = out["m3_nonfuel_rand"].to_numpy() if "m3_nonfuel_rand" in out.columns else np.zeros(n)
    nf_monetary_increment = np.maximum(0.0, nf_pred_rand - m3_nf_rand)
    out["nf_monetary_increment"] = nf_monetary_increment
    out["fuel_vs_nf_monetary_margin"] = fuel_monetary_increment - nf_monetary_increment

    return out.fillna(0.0).astype("float64")


def select_top_features_by_gain(
    X: pd.DataFrame,
    y: np.ndarray,
    sample_weight: np.ndarray | None = None,
    top_k: int = 240,
    seed: int = 42,
    n_jobs: int = -1,
) -> list[str]:
    """Selects top_k numeric features using LightGBM feature importance (gain)."""
    if X.shape[1] <= top_k:
        return list(X.columns)

    essential = [
        "m3_fuel_litres", "m3_fuel_rand", "m3_nonfuel_rand", "m1_fuel_litres",
        "m1_baskets", "m3_baskets", "d14_baskets", "d45_baskets", "days_since_last_purchase",
        "real_fuel_hurdle_litres", "real_fuel_hurdle_litres_price_adj",
        "active_m6_fuel", "prior_q_fuel_active", "prior_q_nf_active",
        "z_anchor_fuel_mean", "z_anchor_nf_mean", "lag1_clv_fuel", "lag1_clv_nonfuel",
        "proj_m1_clv_fuel", "proj_m1_clv_nonfuel",
    ]
    essential_in_X = [c for c in essential if c in X.columns]

    from lightgbm import LGBMRegressor
    numeric_cols = [c for c in X.columns if not c.startswith("cat_")]
    X_num = X[numeric_cols]

    clf = LGBMRegressor(
        n_estimators=120,
        learning_rate=0.06,
        num_leaves=31,
        random_state=seed,
        n_jobs=n_jobs,
        verbose=-1,
        importance_type="gain",
    )
    if sample_weight is not None:
        clf.fit(X_num, y, sample_weight=sample_weight)
    else:
        clf.fit(X_num, y)

    importances = clf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]

    selected_set = set(essential_in_X)
    for i in sorted_idx:
        if len(selected_set) >= top_k:
            break
        selected_set.add(numeric_cols[i])

    selected_cols = [c for c in numeric_cols if c in selected_set]
    cat_cols = [c for c in X.columns if c.startswith("cat_")]
    return selected_cols + cat_cols


def select_top_classification_features(
    X: pd.DataFrame,
    y: np.ndarray,
    sample_weight: np.ndarray | None = None,
    top_k: int = 220,
    seed: int = 42,
    n_jobs: int = -1,
    inact_idx: int = 4,
    stable_idx: int = 16,
    fg_idx: int = 2,
) -> list[str]:
    """Selects top_k numeric features for Opportunity classification using LightGBM gain.

    Computes split gains across the 3 core competition decision boundaries:
    1. Inactivity vs Active
    2. Fuel Growth vs Rest
    3. Retail vs Fuel/Stable
    Guarantees retention of essential RFM, hurdle momentum, and longitudinal anchors.
    """
    if X.shape[1] <= top_k:
        return list(X.columns)

    essential = [
        "days_since_last_purchase", "m1_baskets", "m3_baskets", "d14_baskets", "d45_baskets",
        "btyd_p_alive", "btyd_exp_transactions", "recency_fuel", "recency_nonfuel",
        "active_m6_fuel", "prior_q_fuel_active", "prior_q_nf_active",
        "hawkes_intensity_total", "burn_rate_total", "spend_m1_vs_m2", "spend_m2_vs_m3",
        "baskets_m1_vs_m2", "baskets_m2_vs_m3",
        "m1_fuel_litres", "m3_fuel_litres", "m3_fuel_rand", "m1_fuel_rand", "ever_fuel",
        "real_fuel_hurdle_litres", "real_fuel_hurdle_litres_price_adj",
        "rule6_exp_inc_fuel", "fuel_diff_litres_m1_m3",
        "m1_nonfuel_rand", "m3_nonfuel_rand", "unique_categories_m3", "unique_categories_lifetime",
        "rule6_exp_inc_beverages", "rule6_exp_inc_other", "rule6_n_categories_clearing_hurdle",
        "z_anchor_fuel_mean", "z_anchor_nf_mean", "lag1_clv_fuel", "lag1_clv_nonfuel",
        "proj_m1_clv_fuel", "proj_m1_clv_nonfuel",
        "cutoff_sin_month", "cutoff_cos_month", "is_summer_outcome_window",
    ]
    essential_in_X = [c for c in essential if c in X.columns]

    from lightgbm import LGBMClassifier
    numeric_cols = [c for c in X.columns if not c.startswith("cat_")]
    X_num = X[numeric_cols]
    y_arr = np.asarray(y, dtype=int)

    n_samples = len(X_num)
    if n_samples > 25000:
        rng = np.random.default_rng(seed)
        sub_idx = rng.choice(n_samples, size=25000, replace=False)
        X_fit = X_num.iloc[sub_idx]
        y_fit = y_arr[sub_idx]
        sw_fit = sample_weight[sub_idx] if sample_weight is not None else None
    else:
        X_fit = X_num
        y_fit = y_arr
        sw_fit = sample_weight

    total_importance = np.zeros(len(numeric_cols), dtype=np.float64)

    # 3 Fast binary models targeting macro competition partitions (60 trees each)
    binary_targets = [
        (y_fit == inact_idx).astype(int),
        (y_fit == fg_idx).astype(int),
        (~np.isin(y_fit, [inact_idx, stable_idx, fg_idx])).astype(int),
    ]

    for y_bin in binary_targets:
        clf = LGBMClassifier(
            n_estimators=60,
            learning_rate=0.08,
            num_leaves=31,
            random_state=seed,
            n_jobs=n_jobs,
            verbose=-1,
            importance_type="gain",
        )
        if sw_fit is not None:
            clf.fit(X_fit, y_bin, sample_weight=sw_fit)
        else:
            clf.fit(X_fit, y_bin)
        total_importance += clf.feature_importances_

    sorted_idx = np.argsort(total_importance)[::-1]

    selected_set = set(essential_in_X)
    for i in sorted_idx:
        if len(selected_set) >= top_k:
            break
        selected_set.add(numeric_cols[i])

    selected_cols = [c for c in numeric_cols if c in selected_set]
    cat_cols = [c for c in X.columns if c.startswith("cat_")]
    return selected_cols + cat_cols


@dataclass
class TournamentResult:
    """Out-of-fold predictions and ground truths pooled across all tournament folds."""

    pool_fuel_m: np.ndarray
    pool_nf_m: np.ndarray
    pool_probs_base: np.ndarray
    pool_probs_hier: np.ndarray
    pool_phys_fg: np.ndarray
    pool_phys_inact: np.ndarray
    pool_level2_X: pd.DataFrame
    pool_y_fuel: np.ndarray
    pool_y_nonfuel: np.ndarray
    pool_y_opp: np.ndarray
    pool_masks: np.ndarray
    pool_weights: np.ndarray
    pool_ids: list[str]
    pool_cutoffs: list[pd.Timestamp]
    fold_scores: dict[str, dict[str, float]]
    member_names: list[str]
    pool_p_fuel_act: np.ndarray = field(default_factory=lambda: np.zeros(0))
    pool_p_store_act: np.ndarray = field(default_factory=lambda: np.zeros(0))
    pool_mu_fuel_pos: np.ndarray = field(default_factory=lambda: np.zeros(0))
    pool_mu_nf_pos: np.ndarray = field(default_factory=lambda: np.zeros(0))
    hist_residuals_f: dict[str, list[tuple[pd.Timestamp, float]]] = field(default_factory=dict)
    hist_residuals_nf: dict[str, list[tuple[pd.Timestamp, float]]] = field(default_factory=dict)

    def filter_cutoffs(self, keep_cutoffs: Sequence[pd.Timestamp]) -> "TournamentResult":
        """Returns a subset of TournamentResult restricted strictly to specified cutoffs."""
        keep_set = set(pd.Timestamp(c) for c in keep_cutoffs)
        indices = [i for i, c in enumerate(self.pool_cutoffs) if pd.Timestamp(c) in keep_set]
        if not indices:
            return self
        idx_arr = np.array(indices)
        return TournamentResult(
            pool_fuel_m=self.pool_fuel_m[idx_arr],
            pool_nf_m=self.pool_nf_m[idx_arr],
            pool_probs_base=self.pool_probs_base[idx_arr],
            pool_probs_hier=self.pool_probs_hier[idx_arr],
            pool_phys_fg=self.pool_phys_fg[idx_arr],
            pool_phys_inact=self.pool_phys_inact[idx_arr],
            pool_level2_X=self.pool_level2_X.iloc[idx_arr].reset_index(drop=True),
            pool_y_fuel=self.pool_y_fuel[idx_arr],
            pool_y_nonfuel=self.pool_y_nonfuel[idx_arr],
            pool_y_opp=self.pool_y_opp[idx_arr],
            pool_masks=self.pool_masks[idx_arr],
            pool_weights=self.pool_weights[idx_arr],
            pool_ids=[self.pool_ids[i] for i in indices],
            pool_cutoffs=[self.pool_cutoffs[i] for i in indices],
            fold_scores={k: v for k, v in self.fold_scores.items() if any(c.strftime("%Y-%m-%d") in k for c in keep_set)},
            member_names=self.member_names,
            pool_p_fuel_act=self.pool_p_fuel_act[idx_arr] if len(self.pool_p_fuel_act) == len(self.pool_cutoffs) else self.pool_p_fuel_act,
            pool_p_store_act=self.pool_p_store_act[idx_arr] if len(self.pool_p_store_act) == len(self.pool_cutoffs) else self.pool_p_store_act,
            pool_mu_fuel_pos=self.pool_mu_fuel_pos[idx_arr] if len(self.pool_mu_fuel_pos) == len(self.pool_cutoffs) else self.pool_mu_fuel_pos,
            pool_mu_nf_pos=self.pool_mu_nf_pos[idx_arr] if len(self.pool_mu_nf_pos) == len(self.pool_cutoffs) else self.pool_mu_nf_pos,
            hist_residuals_f=self.hist_residuals_f,
            hist_residuals_nf=self.hist_residuals_nf,
        )


@dataclass
class CalibrationArtifacts:
    """Calibrated post-processing parameters fitted on the pooled out-of-fold matrix."""

    member_names: list[str]
    fuel_debias: list[tuple[float, float]]
    nf_debias: list[tuple[float, float]]
    fuel_blend_weights: np.ndarray
    nf_blend_weights: np.ndarray
    fuel_slope: float
    fuel_intercept: float
    nf_slope: float
    nf_intercept: float
    cust_history_f: dict[str, list[tuple[pd.Timestamp, float]]]
    cust_history_nf: dict[str, list[tuple[pd.Timestamp, float]]]
    blup_eta_fuel: float
    blup_eta_nf: float
    fuel_shrinkage_power: float
    nf_alpha: float
    nf_beta: float
    inact_threshold: float
    prob_blend_weights: tuple[float, float, float, float, float]
    temperature: float
    class_weights: np.ndarray
    target_priors: np.ndarray
    precision_weighted_target_priors: np.ndarray
    commuter_weights: tuple[float, float, float]
    prob_stacker: Any = None


def build_prob_stack_features(
    probs_base: np.ndarray,
    probs_hier: np.ndarray,
    probs_meta: np.ndarray | None,
    p_phys_fg: np.ndarray,
    p_phys_inact: np.ndarray,
) -> np.ndarray:
    """Feature matrix for the log-loss probability stacker (log-probabilities of each source + physics probabilities)."""
    sources = [probs_base, probs_hier] + ([probs_meta] if probs_meta is not None else [])
    logs = [np.log(np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0)) for p in sources]
    return np.column_stack(logs + [
        np.asarray(p_phys_fg, dtype=np.float64).reshape(-1, 1),
        np.asarray(p_phys_inact, dtype=np.float64).reshape(-1, 1),
    ])


class ProbStacker:
    """Multinomial logistic stacker that always returns probabilities over the full label space."""

    def __init__(self, n_classes: int, C: float = 0.3) -> None:
        self.n_classes = n_classes
        self.C = C
        self.pipe: Any = None
        self.classes_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "ProbStacker":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        self.pipe = make_pipeline(StandardScaler(), LogisticRegression(C=self.C, max_iter=400))
        fit_kw = {"logisticregression__sample_weight": sample_weight} if sample_weight is not None else {}
        self.pipe.fit(X, y, **fit_kw)
        self.classes_ = self.pipe.classes_
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = self.pipe.predict_proba(X)
        out = np.zeros((raw.shape[0], self.n_classes), dtype=np.float64)
        out[:, self.classes_] = raw
        return out


def fuse_opportunity_probabilities(
    cal: "CalibrationArtifacts",
    probs_base: np.ndarray,
    probs_hier: np.ndarray,
    probs_meta: np.ndarray | None,
    p_phys_fg: np.ndarray,
    p_phys_inact: np.ndarray,
    fg_idx: int,
    inact_idx: int,
) -> np.ndarray:
    """Single shared fusion path used identically in calibration, holdout evaluation and test inference."""
    if cal.prob_stacker is not None and probs_meta is not None:
        return cal.prob_stacker.predict_proba(
            build_prob_stack_features(probs_base, probs_hier, probs_meta, p_phys_fg, p_phys_inact)
        )
    wb, wh, _ws, b_fg, b_in = cal.prob_blend_weights
    fused = wb * probs_base + wh * probs_hier
    fused[:, fg_idx] += b_fg * p_phys_fg
    fused[:, inact_idx] += b_in * p_phys_inact
    if probs_meta is not None:
        fused = 0.65 * probs_meta + 0.35 * fused
    return fused


class TournamentEngine:
    """Executes the multi-fold tournament and trains final models."""

    def __init__(self, cfg: PipelineConfig, builder: SnapshotBuilder) -> None:
        self.cfg = cfg
        self.builder = builder
        self.label_cfg = cfg.label_config
        self.opp_labels = self.label_cfg["opportunity_labels"]
        self.n_classes = len(self.opp_labels)
        self.label_to_idx = {l: i for i, l in enumerate(self.opp_labels)}
        self.idx_to_label = {i: l for i, l in enumerate(self.opp_labels)}
        self.inact_idx = self.label_to_idx["Inactivity"]
        self.stable_idx = self.label_to_idx["Stable"]
        self.fg_idx = self.label_to_idx["Existing-category growth: Fuel"]

    def _get_fold_definitions(self) -> list[dict[str, Any]]:
        """Constructs the strictly causal forward rolling out-of-fold calibration pool.

        Each fold is strictly forward-looking with training cutoffs <= T_fold - 3 months (embargoed).
        Zero lookahead leakage and zero backward prediction on early censored snapshots.
        """
        names = {
            "2025-06-01": ("Winter", 1.0),
            "2025-07-01": ("MidWinter", 1.25),
            "2025-08-01": ("LateWinter", 1.25),
            "2025-09-01": ("Spring", 1.5),
        }
        folds: list[dict[str, Any]] = []
        for cut in self.cfg.tournament_cutoffs:
            eval_t = pd.Timestamp(cut)
            label, weight = names.get(str(eval_t.date()), (f"Fold", 1.0))
            folds.append({
                "name": f"{label}_{eval_t.date()}",
                "eval_cutoff": eval_t,
                "train_cutoffs": self.cfg.training_cutoffs(boundary=eval_t),
                "weight": weight,
            })
        return folds

    def _l1_cache_key(self) -> str:
        """Hash of everything that determines level-1 OOF outputs (config + model/feature source code)."""
        import hashlib
        from pathlib import Path
        m = self.cfg.models
        h = hashlib.sha1()
        h.update(repr((
            self.cfg.tournament_cutoffs, self.cfg.first_train_cutoff, self.cfg.cutoff_step_months,
            m.regression_members, m.classification_members, m.seed, m.n_seeds_reg, m.n_seeds_clf,
            m.hurdle_seeds,
        )).encode())
        src_dir = Path(__file__).resolve().parent
        for fname in ("models.py", "hurdle.py", "features.py", "tournament.py"):
            h.update((src_dir / fname).read_bytes())
        return h.hexdigest()[:16]

    def run_tournament(self, use_l1_cache: bool = False) -> TournamentResult:
        """Runs the OOF tournament; level-1 outputs are cached on disk keyed by config + model/feature source."""
        import pickle
        cache_path = self.cfg.paths.cache_dir / f"tournament_l1_{self._l1_cache_key()}.pkl"
        if use_l1_cache and cache_path.exists():
            logger.info("Loading cached level-1 tournament result: %s", cache_path.name)
            with open(cache_path, "rb") as fh:
                return pickle.load(fh)
        result = self._run_tournament_uncached()
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "wb") as fh:
                pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info("Cached level-1 tournament result: %s", cache_path.name)
        except Exception as exc:  # caching must never break a run
            logger.warning("Could not cache tournament result: %s", exc)
        return result

    def _run_tournament_uncached(self) -> TournamentResult:
        """Execute temporal folds to generate causal out-of-fold predictions."""
        folds = self._get_fold_definitions()
        logger.info("=== Starting Multi-Horizon Out-of-Fold Tournament (%d folds) ===", len(folds))

        pool_fuel_m_list = []
        pool_nf_m_list = []
        pool_probs_base_list = []
        pool_probs_hier_list = []
        pool_phys_fg_list = []
        pool_phys_inact_list = []
        pool_p_fuel_act_list = []
        pool_p_store_act_list = []
        pool_mu_fuel_pos_list = []
        pool_mu_nf_pos_list = []
        pool_level2_X_list = []
        pool_y_fuel_list = []
        pool_y_nf_list = []
        pool_y_opp_list = []
        pool_masks_list = []
        pool_weights_list = []
        pool_ids_list: list[str] = []
        pool_cutoffs_list: list[pd.Timestamp] = []
        fold_scores: dict[str, dict[str, float]] = {}

        hist_residuals_f: dict[str, list[tuple[pd.Timestamp, float]]] = {}
        hist_residuals_nf: dict[str, list[tuple[pd.Timestamp, float]]] = {}
        recorded_pairs: set[tuple[str, pd.Timestamp]] = set()

        member_names: list[str] = list(self.cfg.models.regression_members) + ["hurdle_bagged", "hurdle_quantile", "phys_basket"]

        for fold in folds:
            fold_name = fold["name"]
            eval_t = fold["eval_cutoff"]
            train_cutoffs = fold["train_cutoffs"]
            w_fold = fold["weight"]
            logger.info("--- Running Tournament Fold: %s (Weight: %.1fx, N_train=%d cutoffs) ---", fold_name, w_fold, len(train_cutoffs))

            # Build snapshots
            train_snaps = [self.builder.build(c, with_labels=True) for c in train_cutoffs]
            eval_snap = self.builder.build(eval_t, with_labels=True)
            train_panel = stack(train_snaps)

            for snap in train_snaps:
                c_ts = pd.Timestamp(snap.cutoff)
                z_f_anc = snap.features["z_anchor_fuel_mean"].to_numpy() if "z_anchor_fuel_mean" in snap.features.columns else np.zeros(len(snap.ids))
                z_nf_anc = snap.features["z_anchor_nf_mean"].to_numpy() if "z_anchor_nf_mean" in snap.features.columns else np.zeros(len(snap.ids))
                y_f_snap = snap.labels["CLV_fuel"].to_numpy()
                y_nf_snap = snap.labels["CLV_nonfuel"].to_numpy()
                slope_f_anc, _ = calibration.fit_linear_calibration(y_f_snap, z_f_anc, force_zero_intercept=True)
                slope_nf_anc, _ = calibration.fit_linear_calibration(y_nf_snap, z_nf_anc, force_zero_intercept=True)
                res_f_snap = y_f_snap - slope_f_anc * z_f_anc
                res_nf_snap = y_nf_snap - slope_nf_anc * z_nf_anc
                for cid, rf_val, rnf_val in zip(snap.ids, res_f_snap, res_nf_snap):
                    if (cid, c_ts) not in recorded_pairs:
                        recorded_pairs.add((cid, c_ts))
                        hist_residuals_f.setdefault(cid, []).append((c_ts, float(rf_val)))
                        hist_residuals_nf.setdefault(cid, []).append((c_ts, float(rnf_val)))

            # Seasonal sample weights for training panel:
            # Temporal exponential decay (lambda=0.10) x summer multiplier (2.2x), removing 15x n_snaps_cust distortion
            train_cutoffs_dt = pd.to_datetime(train_panel.cutoffs)
            eval_dt = pd.Timestamp(eval_t)
            months_delta = np.asarray((eval_dt.year - train_cutoffs_dt.year) * 12 + (eval_dt.month - train_cutoffs_dt.month), dtype=np.float32)
            temporal_decay = np.exp(-0.10 * np.maximum(months_delta, 0.0))
            is_summer_train = train_cutoffs_dt.month.isin([11, 12, 1])
            seasonal_mult = np.where(is_summer_train, 2.2, 1.0)
            sw = (temporal_decay * seasonal_mult).astype(np.float32)
            sw = sw / max(float(sw.mean()), 1e-6)

            # Target-specific feature pruning (top 240 numeric features per target, top 220 for opportunity)
            fuel_features = select_top_features_by_gain(
                train_panel.X,
                train_panel.labels["CLV_fuel"].to_numpy(),
                sample_weight=sw,
                top_k=240,
                seed=self.cfg.models.seed,
                n_jobs=self.cfg.models.n_jobs,
            )
            nf_features = select_top_features_by_gain(
                train_panel.X,
                train_panel.labels["CLV_nonfuel"].to_numpy(),
                sample_weight=sw,
                top_k=240,
                seed=self.cfg.models.seed,
                n_jobs=self.cfg.models.n_jobs,
            )
            opp_features = select_top_classification_features(
                train_panel.X,
                train_panel.labels["Opportunity"].map(self.label_to_idx).to_numpy(),
                sample_weight=sw,
                top_k=220,
                seed=self.cfg.models.seed,
                n_jobs=self.cfg.models.n_jobs,
                inact_idx=self.inact_idx,
                stable_idx=self.stable_idx,
                fg_idx=self.fg_idx,
            )

            # 1. Regressors (fitted on top-240 selected features with 3-seed bagging)
            fuel_model = RegressorEnsemble(
                members=self.cfg.models.regression_members,
                seed=self.cfg.models.seed,
                n_jobs=self.cfg.models.n_jobs,
                tweedie_power=1.25,
                use_gpu=self.cfg.models.use_gpu,
                n_seeds=self.cfg.models.n_seeds_reg,
            ).fit(train_panel.X[fuel_features], train_panel.labels["CLV_fuel"].to_numpy(), sample_weight=sw)

            nf_model = RegressorEnsemble(
                members=self.cfg.models.regression_members,
                seed=self.cfg.models.seed,
                n_jobs=self.cfg.models.n_jobs,
                tweedie_power=1.40,
                use_gpu=self.cfg.models.use_gpu,
                n_seeds=self.cfg.models.n_seeds_reg,
            ).fit(train_panel.X[nf_features], train_panel.labels["CLV_nonfuel"].to_numpy(), sample_weight=sw)

            # 2. Hurdle Models (5 Seeds + MultiQuantile on selected features)
            hurdle_fuel = BaggedTwoPartFuelHurdle(
                seeds=self.cfg.models.hurdle_seeds,
                huber_alpha=1.35,
                power_shrinkage=1.0,
                zero_gate_threshold=0.0,
                n_jobs=self.cfg.models.n_jobs,
                use_gpu=self.cfg.models.use_gpu,
            ).fit(train_panel.X[fuel_features], train_panel.labels["CLV_fuel"].to_numpy(), sample_weight=sw)

            hurdle_nf = BaggedTwoPartNonfuelHurdle(
                seeds=self.cfg.models.hurdle_seeds,
                huber_alpha=1.35,
                power_shrinkage=1.0,
                zero_gate_threshold=0.0,
                n_jobs=self.cfg.models.n_jobs,
                use_gpu=self.cfg.models.use_gpu,
            ).fit(train_panel.X[nf_features], train_panel.labels["CLV_nonfuel"].to_numpy(), sample_weight=sw)

            mq_fuel = MultiQuantileTwoPartHurdle(
                seed=self.cfg.models.seed,
                n_jobs=self.cfg.models.n_jobs,
                use_gpu=self.cfg.models.use_gpu,
            ).fit(train_panel.X[fuel_features], train_panel.labels["CLV_fuel"].to_numpy(), sample_weight=sw)

            mq_nf = MultiQuantileTwoPartHurdle(
                seed=self.cfg.models.seed,
                n_jobs=self.cfg.models.n_jobs,
                use_gpu=self.cfg.models.use_gpu,
                target_type="nonfuel",
            ).fit(train_panel.X[nf_features], train_panel.labels["CLV_nonfuel"].to_numpy(), sample_weight=sw)

            # 3. Physical Basket x Ticket Regressors
            phys_fuel = PhysicalBasketTicketRegressor(
                scale=2.352825386185113,
                seed=self.cfg.models.seed,
                n_jobs=self.cfg.models.n_jobs,
                use_gpu=self.cfg.models.use_gpu,
                target_type="fuel",
            ).fit(train_panel.X[fuel_features], train_panel.labels["CLV_fuel"].to_numpy(), sample_weight=sw)

            phys_nf = PhysicalBasketTicketRegressor(
                scale=2.4200904988927476,
                seed=self.cfg.models.seed,
                n_jobs=self.cfg.models.n_jobs,
                use_gpu=self.cfg.models.use_gpu,
                target_type="nonfuel",
            ).fit(train_panel.X[nf_features], train_panel.labels["CLV_nonfuel"].to_numpy(), sample_weight=sw)

            # 4. Classifiers (fitted on pruned top features)
            opp_weights = [0.40, 0.25, 0.35] if tuple(self.cfg.models.classification_members) == ("lgbm", "catboost", "xgb") else None
            opp_model = ClassifierEnsemble(
                members=self.cfg.models.classification_members,
                n_classes=self.n_classes,
                seed=self.cfg.models.seed,
                n_jobs=self.cfg.models.n_jobs,
                weights=opp_weights,
                use_gpu=self.cfg.models.use_gpu,
                n_seeds=self.cfg.models.n_seeds_clf,
            ).fit(train_panel.X[opp_features], train_panel.labels["Opportunity"].map(self.label_to_idx).to_numpy(), sample_weight=sw)

            hier_model = HierarchicalOpportunityClassifier(
                n_classes=self.n_classes,
                seed=self.cfg.models.seed,
                n_jobs=self.cfg.models.n_jobs,
                inact_idx=self.inact_idx,
                stable_idx=self.stable_idx,
                fg_idx=self.fg_idx,
                use_gpu=self.cfg.models.use_gpu,
            ).fit(train_panel.X[opp_features], train_panel.labels["Opportunity"].map(self.label_to_idx).to_numpy(), sample_weight=sw)

            store_model = StoreAttachmentClassifier(
                seed=self.cfg.models.seed,
                n_jobs=self.cfg.models.n_jobs,
                use_gpu=self.cfg.models.use_gpu,
            ).fit(train_panel.X[nf_features], train_panel.labels["CLV_nonfuel"].to_numpy(), sample_weight=sw)

            # Predict OOF instances on eval_snap
            pred_fuel_direct = fuel_model.predict_members(eval_snap.features[fuel_features])
            pred_fuel_hurdle = hurdle_fuel.predict(eval_snap.features[fuel_features]).reshape(-1, 1)
            pred_fuel_mq = mq_fuel.predict(eval_snap.features[fuel_features]).reshape(-1, 1)
            pred_fuel_phys = phys_fuel.predict(eval_snap.features[fuel_features]).reshape(-1, 1)
            fold_fuel_m = np.column_stack([pred_fuel_direct, pred_fuel_hurdle, pred_fuel_mq, pred_fuel_phys])

            pred_nf_direct = nf_model.predict_members(eval_snap.features[nf_features])
            pred_nf_hurdle = hurdle_nf.predict(eval_snap.features[nf_features]).reshape(-1, 1)
            pred_nf_mq = mq_nf.predict(eval_snap.features[nf_features]).reshape(-1, 1)
            pred_nf_phys = phys_nf.predict(eval_snap.features[nf_features]).reshape(-1, 1)
            fold_nf_m = np.column_stack([pred_nf_direct, pred_nf_hurdle, pred_nf_mq, pred_nf_phys])

            fold_probs_base = opp_model.predict_proba(eval_snap.features[opp_features])
            fold_probs_hier = hier_model.predict_proba(eval_snap.features[opp_features])

            # Hurdle CDF exceedance probabilities and level-2 stacking features
            p_fuel_active = hurdle_fuel.classifier.predict_proba(eval_snap.features[fuel_features])
            p_store_active = hurdle_nf.classifier.predict_proba(eval_snap.features[nf_features])
            mu_fuel_pos, sigma_fuel_pos = mq_fuel.predict_conditional_pos(eval_snap.features[fuel_features])
            mu_nf_pos, sigma_nf_pos = mq_nf.predict_conditional_pos(eval_snap.features[nf_features])

            pool_p_fuel_act_list.append(p_fuel_active)
            pool_p_store_act_list.append(p_store_active)
            pool_mu_fuel_pos_list.append(mu_fuel_pos)
            pool_mu_nf_pos_list.append(mu_nf_pos)

            p_phys_fg, p_phys_inact = calibration.compute_physical_hurdle_probabilities(
                X=eval_snap.features,
                hurdle_model=mq_fuel,
                final_fuel_pred=np.mean(pred_fuel_direct, axis=1),
                store_model=store_model,
                p_fuel=p_fuel_active,
                mu_pos=mu_fuel_pos,
                sigma_pos=sigma_fuel_pos,
            )

            fold_level2_X = build_level2_structural_features(
                X=eval_snap.features,
                probs_base=fold_probs_base,
                probs_hier=fold_probs_hier,
                pred_fuel=np.mean(pred_fuel_direct, axis=1),
                pred_nf=np.mean(pred_nf_direct, axis=1),
                p_fuel_active=p_fuel_active,
                p_store_active=p_store_active,
                mu_fuel_pos=mu_fuel_pos,
                sigma_fuel_pos=sigma_fuel_pos,
            )
            pool_level2_X_list.append(fold_level2_X)

            n_fold = len(eval_snap.ids)
            pool_fuel_m_list.append(fold_fuel_m)
            pool_nf_m_list.append(fold_nf_m)
            pool_probs_base_list.append(fold_probs_base)
            pool_probs_hier_list.append(fold_probs_hier)
            pool_phys_fg_list.append(p_phys_fg)
            pool_phys_inact_list.append(p_phys_inact)
            pool_y_fuel_list.append(eval_snap.labels["CLV_fuel"].to_numpy())
            pool_y_nf_list.append(eval_snap.labels["CLV_nonfuel"].to_numpy())
            pool_y_opp_list.append(eval_snap.labels["Opportunity"].map(self.label_to_idx).to_numpy())
            pool_masks_list.append(eval_snap.mask)
            pool_weights_list.append(np.full(n_fold, w_fold, dtype=np.float32))
            pool_ids_list.extend(list(eval_snap.ids))
            pool_cutoffs_list.extend([eval_t] * n_fold)

            # Evaluate standalone fold benchmark
            bench_f = np.mean(fold_fuel_m[:, :len(self.cfg.models.regression_members)], axis=1)
            bench_nf = np.mean(fold_nf_m[:, :len(self.cfg.models.regression_members)], axis=1)
            bench_probs = 0.5 * fold_probs_base + 0.5 * fold_probs_hier
            bench_opp = [self.idx_to_label[i] for i in np.argmax(calibration.apply_eligibility_mask(bench_probs, eval_snap.mask), axis=1)]
            scores = metrics.score(eval_snap.labels, bench_f, bench_nf, np.array(bench_opp))
            fold_scores[fold_name] = {
                "rmse_fuel": round(scores.rmse_fuel, 4),
                "rmse_nonfuel": round(scores.rmse_nonfuel, 4),
                "weighted_f1": round(scores.weighted_f1, 4),
                "leaderboard_composite": round(scores.leaderboard_composite(), 4),
            }
            logger.info(
                "Fold %s Standalone: Fuel RMSE=%.4f, NF RMSE=%.4f, Opp F1=%.4f, LB Score=%.4f",
                fold_name, scores.rmse_fuel, scores.rmse_nonfuel, scores.weighted_f1, scores.leaderboard_composite()
            )

        return TournamentResult(
            pool_fuel_m=np.vstack(pool_fuel_m_list),
            pool_nf_m=np.vstack(pool_nf_m_list),
            pool_probs_base=np.vstack(pool_probs_base_list),
            pool_probs_hier=np.vstack(pool_probs_hier_list),
            pool_phys_fg=np.concatenate(pool_phys_fg_list),
            pool_phys_inact=np.concatenate(pool_phys_inact_list),
            pool_level2_X=pd.concat(pool_level2_X_list, axis=0, ignore_index=True),
            pool_y_fuel=np.concatenate(pool_y_fuel_list),
            pool_y_nonfuel=np.concatenate(pool_y_nf_list),
            pool_y_opp=np.concatenate(pool_y_opp_list),
            pool_masks=np.vstack(pool_masks_list),
            pool_weights=np.concatenate(pool_weights_list),
            pool_ids=pool_ids_list,
            pool_cutoffs=pool_cutoffs_list,
            fold_scores=fold_scores,
            member_names=member_names,
            pool_p_fuel_act=np.concatenate(pool_p_fuel_act_list),
            pool_p_store_act=np.concatenate(pool_p_store_act_list),
            pool_mu_fuel_pos=np.concatenate(pool_mu_fuel_pos_list),
            pool_mu_nf_pos=np.concatenate(pool_mu_nf_pos_list),
            hist_residuals_f=hist_residuals_f,
            hist_residuals_nf=hist_residuals_nf,
        )

    def fit_calibration(self, tourney: TournamentResult) -> CalibrationArtifacts:
        """Calibrates debiasing, blend weights, BLUP, shrinkage, Level-2 stacker, and decision solver on pooled OOF."""
        logger.info("=== Fitting Calibration on Pooled Out-of-Fold Matrix (N=%d) ===", len(tourney.pool_y_fuel))

        # 1. Per-member linear debiasing
        fuel_debias = calibration.fit_member_debiasing(tourney.pool_fuel_m, tourney.pool_y_fuel)
        nf_debias = calibration.fit_member_debiasing(tourney.pool_nf_m, tourney.pool_y_nonfuel)
        debiased_fuel_m = calibration.apply_member_debiasing(tourney.pool_fuel_m, fuel_debias)
        debiased_nf_m = calibration.apply_member_debiasing(tourney.pool_nf_m, nf_debias)

        # 2. Cross-Target Regression Stacking via Non-Negative Ridge (dynamic hurdle indices)
        M_reg_pool = calibration.build_regression_meta_matrix(
            debiased_fuel_m=debiased_fuel_m,
            debiased_nf_m=debiased_nf_m,
            p_fuel_active=tourney.pool_p_fuel_act,
            p_store_active=tourney.pool_p_store_act,
            mu_fuel_pos=tourney.pool_mu_fuel_pos,
            mu_nf_pos=tourney.pool_mu_nf_pos,
        )
        n_m = debiased_fuel_m.shape[1]
        fuel_hurdle_idx = [n_m - 3, n_m - 2, 2 * n_m + 2]
        nf_hurdle_idx = [2 * n_m - 3, 2 * n_m - 2, 2 * n_m + 3]

        fuel_blend_weights = calibration.fit_nonnegative_ridge(
            M=M_reg_pool,
            y=tourney.pool_y_fuel,
            sample_weight=tourney.pool_weights,
            alpha=10.0,
            hurdle_indices=fuel_hurdle_idx,
            min_hurdle_weight=self.cfg.hurdle_floor_fuel,
        )
        nf_blend_weights = calibration.fit_nonnegative_ridge(
            M=M_reg_pool,
            y=tourney.pool_y_nonfuel,
            sample_weight=tourney.pool_weights,
            alpha=10.0,
            hurdle_indices=nf_hurdle_idx,
            min_hurdle_weight=self.cfg.hurdle_floor_nf,
        )
        logger.info("Fitted fuel cross-target blend weights (%d meta-features): %s", len(fuel_blend_weights), np.round(fuel_blend_weights, 3))
        logger.info("Fitted nonfuel cross-target blend weights (%d meta-features): %s", len(nf_blend_weights), np.round(nf_blend_weights, 3))

        # 3. Overall linear calibration (zero intercept)
        raw_fuel = M_reg_pool @ fuel_blend_weights
        fuel_slope, fuel_intercept = calibration.fit_linear_calibration(tourney.pool_y_fuel, raw_fuel, force_zero_intercept=True)
        raw_nf = M_reg_pool @ nf_blend_weights
        nf_slope, nf_intercept = calibration.fit_linear_calibration(tourney.pool_y_nonfuel, raw_nf, force_zero_intercept=True)
        fuel_cal_lin = calibration.apply_linear_calibration(raw_fuel, fuel_slope, fuel_intercept)
        nf_cal_lin = calibration.apply_linear_calibration(raw_nf, nf_slope, nf_intercept)

        # 4. Multi-Cutoff Customer BLUP fixed effects (merging historical non-overlapping snapshots + OOF)
        res_fuel = tourney.pool_y_fuel - fuel_cal_lin
        res_nf = tourney.pool_y_nonfuel - nf_cal_lin
        cust_history_f: dict[str, list[tuple[pd.Timestamp, float]]] = {
            cid: list(entries) for cid, entries in tourney.hist_residuals_f.items()
        }
        cust_history_nf: dict[str, list[tuple[pd.Timestamp, float]]] = {
            cid: list(entries) for cid, entries in tourney.hist_residuals_nf.items()
        }
        seen_pairs = {(cid, cut) for cid, entries in cust_history_f.items() for cut, _ in entries}
        for cid, cut, rf, rnf in zip(tourney.pool_ids, tourney.pool_cutoffs, res_fuel, res_nf):
            c_ts = pd.Timestamp(cut)
            if (cid, c_ts) not in seen_pairs:
                seen_pairs.add((cid, c_ts))
                cust_history_f.setdefault(cid, []).append((c_ts, float(rf)))
                cust_history_nf.setdefault(cid, []).append((c_ts, float(rnf)))

        blup_f_vec = np.zeros(len(tourney.pool_y_fuel))
        blup_nf_vec = np.zeros(len(tourney.pool_y_nonfuel))
        pool_ids_arr = np.asarray(tourney.pool_ids, dtype=object)
        pool_cut_arr = pd.to_datetime(pd.Series(tourney.pool_cutoffs)).to_numpy()
        for u_cut in np.unique(pool_cut_arr):
            idx_c = np.flatnonzero(pool_cut_arr == u_cut)
            u_ts = pd.Timestamp(u_cut)
            map_f = calibration.compute_customer_blup(cust_history_f, u_ts)
            map_nf = calibration.compute_customer_blup(cust_history_nf, u_ts)
            ids_c = pool_ids_arr[idx_c]
            blup_f_vec[idx_c] = [map_f.get(cid, 0.0) for cid in ids_c]
            blup_nf_vec[idx_c] = [map_nf.get(cid, 0.0) for cid in ids_c]

        act_mask_f = fuel_cal_lin > 0.05
        act_mask_nf = nf_cal_lin > 0.05
        corr_f = float(np.corrcoef(res_fuel[act_mask_f], blup_f_vec[act_mask_f])[0, 1]) if act_mask_f.sum() > 20 and np.std(blup_f_vec[act_mask_f]) > 1e-6 else 0.0
        corr_nf = float(np.corrcoef(res_nf[act_mask_nf], blup_nf_vec[act_mask_nf])[0, 1]) if act_mask_nf.sum() > 20 and np.std(blup_nf_vec[act_mask_nf]) > 1e-6 else 0.0
        blup_eta_f = calibration.fit_blup_shrinkage(tourney.pool_y_fuel, fuel_cal_lin, blup_f_vec, act_mask_f)
        blup_eta_nf = calibration.fit_blup_shrinkage(tourney.pool_y_nonfuel, nf_cal_lin, blup_nf_vec, act_mask_nf)
        logger.info(
            "Out-of-time residual autocorrelation: fuel_corr=%.4f (eta=%.3f), nonfuel_corr=%.4f (eta=%.3f)",
            corr_f, blup_eta_f, corr_nf, blup_eta_nf,
        )

        fuel_after_blup = calibration.apply_customer_blup(fuel_cal_lin, blup_f_vec, blup_eta_f, act_mask_f)
        nf_after_blup = calibration.apply_customer_blup(nf_cal_lin, blup_nf_vec, blup_eta_nf, act_mask_nf)

        # 5. Inactivity Floor Gate
        fuel_shrinkage, nf_alpha, nf_beta = 0.0, 0.0, 0.0
        f_shrunk = fuel_after_blup
        nf_shrunk = nf_after_blup
        inact_thresh = calibration.fit_inactivity_threshold(
            tourney.pool_y_fuel, f_shrunk,
            tourney.pool_y_nonfuel, nf_shrunk,
            tourney.pool_p_fuel_act,
        )
        m3_b_pool = tourney.pool_level2_X["m3_baskets"].to_numpy() if "m3_baskets" in tourney.pool_level2_X.columns else np.zeros(len(f_shrunk))
        cal_fuel_pred = calibration.apply_extreme_dead_gate(f_shrunk, tourney.pool_p_fuel_act, m3_b_pool, threshold=inact_thresh)
        cal_nf_pred = calibration.apply_extreme_dead_gate(nf_shrunk, tourney.pool_p_store_act, m3_b_pool, threshold=inact_thresh)
        logger.info("Fitted inactivity floor threshold: %.3f", inact_thresh)

        # 6. Refresh Level-2 Structural Features with Calibrated Regression Predictions
        cal_level2_X = refresh_level2_predictions(
            df=tourney.pool_level2_X,
            pred_fuel=cal_fuel_pred,
            pred_nf=cal_nf_pred,
            p_fuel_active=tourney.pool_p_fuel_act,
            p_store_active=tourney.pool_p_store_act,
            mu_fuel_pos=tourney.pool_mu_fuel_pos,
            sigma_fuel_pos=tourney.pool_level2_X["sigma_fuel_pos"].to_numpy(),
        )
        p_phys_fg_cal = cal_level2_X["p_exceed_fuel"].to_numpy()
        m1_b_pool = cal_level2_X["m1_baskets"].to_numpy() if "m1_baskets" in cal_level2_X.columns else np.zeros(len(cal_fuel_pred))
        p_phys_inact_cal = cal_level2_X["joint_hurdle_inactivity"].to_numpy() * (m1_b_pool == 0).astype(float)

        # 7. Base Opportunity Probability Fusion
        def sim_objective(weights: np.ndarray) -> float:
            w_base, w_hier, beta_fg, beta_in = weights
            fused = w_base * tourney.pool_probs_base + w_hier * tourney.pool_probs_hier
            fused[:, self.fg_idx] += beta_fg * p_phys_fg_cal
            fused[:, self.inact_idx] += beta_in * p_phys_inact_cal
            fused_masked = calibration.apply_eligibility_mask(fused, tourney.pool_masks)
            pred = np.argmax(fused_masked, axis=1)
            return -1.0 * calibration.weighted_f1(tourney.pool_y_opp, pred, self.n_classes)

        opt_blend = minimize(
            sim_objective,
            x0=[0.50, 0.50, 0.15, 0.10],
            bounds=[(0.1, 0.9), (0.1, 0.9), (0.0, 0.4), (0.0, 0.3)],
            method="Nelder-Mead",
            options={"maxiter": 100},
        )
        wb, wh, b_fg, b_in = opt_blend.x
        prob_blend = (float(wb), float(wh), 0.0, float(b_fg), float(b_in))

        fused_pool = wb * tourney.pool_probs_base + wh * tourney.pool_probs_hier
        fused_pool[:, self.fg_idx] += b_fg * p_phys_fg_cal
        fused_pool[:, self.inact_idx] += b_in * p_phys_inact_cal

        prob_stacker = None
        if self.cfg.prob_fusion == "stack":
            stack_X = build_prob_stack_features(
                tourney.pool_probs_base, tourney.pool_probs_hier, None, p_phys_fg_cal, p_phys_inact_cal,
            )
            prob_stacker = ProbStacker(self.n_classes).fit(stack_X, tourney.pool_y_opp, tourney.pool_weights)
            combined_pool = prob_stacker.predict_proba(stack_X)
            logger.info("Opportunity fusion: multinomial log-loss stacker (%d features)", stack_X.shape[1])
        else:
            combined_pool = fused_pool
        combined_pool_masked = calibration.apply_eligibility_mask(combined_pool, tourney.pool_masks)

        # Apply hyper-active weekly sanity gate consistently in calibration
        d_last_pool = cal_level2_X["days_since_last_purchase"].to_numpy() if "days_since_last_purchase" in cal_level2_X.columns else np.full(len(cal_level2_X), 999.0)
        hyper_active_pool = (m1_b_pool >= 4) & (d_last_pool <= 7)
        combined_pool_masked[hyper_active_pool, self.inact_idx] = 0.0
        combined_pool_masked = combined_pool_masked / np.maximum(combined_pool_masked.sum(axis=1, keepdims=True), 1e-12)

        # 8. Temperature and Class Weights (evaluated on forward cutoffs when >= 2 cutoffs exist)
        dom_classes = [self.inact_idx, self.stable_idx, self.fg_idx]
        cutoffs_arr = pd.to_datetime(pd.Series(tourney.pool_cutoffs)).to_numpy()
        unique_cuts = np.sort(np.unique(cutoffs_arr))
        cal_sub_mask = (cutoffs_arr >= unique_cuts[1]) if len(unique_cuts) >= 2 else np.ones(len(cutoffs_arr), dtype=bool)

        opt_temp, class_weights = calibration.fit_temperature_and_class_weights(
            combined_pool_masked[cal_sub_mask],
            tourney.pool_y_opp[cal_sub_mask],
            mask=tourney.pool_masks[cal_sub_mask],
            dominant_classes=dom_classes,
            n_trials=200,
            seed=self.cfg.models.seed,
        )
        scaled_pool = calibration.apply_temperature_scaling(combined_pool_masked, opt_temp)
        masked_pool = calibration.apply_eligibility_mask(scaled_pool, tourney.pool_masks)
        masked_pool[hyper_active_pool, self.inact_idx] = 0.0
        masked_pool = masked_pool / np.maximum(masked_pool.sum(axis=1, keepdims=True), 1e-12)

        # 9. Seasonally-Adjusted Empirical Prior & Plug-In Weights
        pi_empirical = np.bincount(tourney.pool_y_opp, minlength=self.n_classes).astype(np.float64)
        target_priors = pi_empirical / pi_empirical.sum()

        from sklearn.metrics import f1_score
        pred_baseline = calibration.decide(masked_pool, class_weights)
        oof_f1_per_class = f1_score(
            tourney.pool_y_opp,
            pred_baseline,
            average=None,
            labels=list(range(self.n_classes)),
            zero_division=0.0,
        )
        priors_prec_shrunk = calibration.compute_precision_weighted_target_priors(
            target_priors,
            oof_f1_per_class,
            macro_classes=(self.stable_idx, self.fg_idx, self.inact_idx),
        )

        n_sub = int(cal_sub_mask.sum())
        target_bounds_pool = {
            self.stable_idx: (int(1920 * n_sub / 5488), int(1990 * n_sub / 5488)),
            self.fg_idx: (int(1540 * n_sub / 5488), int(1615 * n_sub / 5488)),
            self.inact_idx: (int(1410 * n_sub / 5488), int(1470 * n_sub / 5488)),
        }
        opt_class_weights = calibration.optimize_expected_f1_plug_in(
            masked_pool[cal_sub_mask],
            target_priors=priors_prec_shrunk,
            init_weights=class_weights,
            target_bounds=target_bounds_pool,
            max_iter=100,
        )
        pred_before = calibration.decide(masked_pool[cal_sub_mask], class_weights)
        f1_before = calibration.weighted_f1(tourney.pool_y_opp[cal_sub_mask], pred_before, self.n_classes)
        pred_after = calibration.decide(masked_pool[cal_sub_mask], opt_class_weights)
        f1_after = calibration.weighted_f1(tourney.pool_y_opp[cal_sub_mask], pred_after, self.n_classes)
        logger.info("Forward OOF Sub-Pool Weighted F1: before plug-in=%.4f, after plug-in=%.4f", f1_before, f1_after)
        final_class_weights = opt_class_weights if f1_after >= f1_before else class_weights

        commuter_weights = (0.0, 0.0, 1.0)

        return CalibrationArtifacts(
            member_names=tourney.member_names,
            fuel_debias=fuel_debias,
            nf_debias=nf_debias,
            fuel_blend_weights=fuel_blend_weights,
            nf_blend_weights=nf_blend_weights,
            fuel_slope=fuel_slope,
            fuel_intercept=fuel_intercept,
            nf_slope=nf_slope,
            nf_intercept=nf_intercept,
            cust_history_f=cust_history_f,
            cust_history_nf=cust_history_nf,
            blup_eta_fuel=blup_eta_f,
            blup_eta_nf=blup_eta_nf,
            fuel_shrinkage_power=fuel_shrinkage,
            nf_alpha=nf_alpha,
            nf_beta=nf_beta,
            inact_threshold=inact_thresh,
            prob_blend_weights=prob_blend,
            temperature=opt_temp,
            class_weights=final_class_weights,
            target_priors=target_priors,
            precision_weighted_target_priors=priors_prec_shrunk,
            commuter_weights=commuter_weights,
            prob_stacker=prob_stacker,
        )

