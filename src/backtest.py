"""Rolling-origin backtesting harness for the Sasol CLV & Opportunity pipeline.

Provides:
1. Multi-fold rolling-origin out-of-time evaluation across historical cutoffs.
2. Seasonal Dec-Feb holiday benchmark fold (2024-12-01).
3. Both Local Proxy (zero baseline) and Real Leaderboard (persistence baseline) scoring.
4. Paired bootstrap significance testing (B=1,000) for sound statistical decision making.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src import calibration, metrics
from src.config import PipelineConfig
from src.dataset import SnapshotBuilder, stack
from src.hurdle import BaggedTwoPartFuelHurdle, BaggedTwoPartNonfuelHurdle
from src.models import (
    ClassifierEnsemble,
    HierarchicalOpportunityClassifier,
    RegressorEnsemble,
    StoreAttachmentClassifier,
)

logger = logging.getLogger(__name__)

# Official empirically verified Leaderboard persistence x_max
LB_X_MAX_FUEL: float = 0.740
LB_X_MAX_NONFUEL: float = 0.816


@dataclass(frozen=True)
class FoldResult:
    cutoff: str
    n_train: int
    n_eval: int
    scores: metrics.Scores
    proxy_score: float
    lb_score: float
    predictions: pd.DataFrame


@dataclass(frozen=True)
class BacktestSummary:
    folds: list[FoldResult]
    mean_proxy_score: float
    mean_lb_score: float
    mean_fuel_rmse: float
    mean_nonfuel_rmse: float
    mean_weighted_f1: float
    seasonal_fold_result: FoldResult | None = None


def train_and_calibrate_fold(
    cfg: PipelineConfig,
    builder: SnapshotBuilder,
    train_cutoffs: Sequence[pd.Timestamp],
    cal_cutoff: pd.Timestamp,
    eval_cutoff: pd.Timestamp,
    label_cfg: dict[str, Any],
    cache_dir: Path | None = None,
) -> FoldResult:
    """Trains strictly on train_cutoffs, calibrates on cal_cutoff, and evaluates on eval_cutoff.

    Executes the exact same model stack, hurdle architecture, and post-processing as src/pipeline.py.
    """
    opp_labels = label_cfg["opportunity_labels"]
    n_classes = len(opp_labels)
    label_to_idx = {l: i for i, l in enumerate(opp_labels)}
    idx_to_label = {i: l for i, l in enumerate(opp_labels)}
    inact_idx = label_to_idx["Inactivity"]
    stable_idx = label_to_idx["Stable"]
    fg_idx = label_to_idx["Existing-category growth: Fuel"]

    if cache_dir:
        eval_cache = cache_dir / f"eval_{eval_cutoff.date()}.parquet"
        if eval_cache.exists():
            logger.info("Loading cached backtest evaluation for %s...", eval_cutoff.date())
            pred_df = pd.read_parquet(eval_cache)
            labels_df = pd.DataFrame({
                "CLV_fuel": pred_df["true_fuel"],
                "CLV_nonfuel": pred_df["true_nonfuel"],
                "Opportunity": pred_df["true_opp"],
            }, index=pred_df.index)
            eval_scores = metrics.score(
                labels_df,
                pred_df["pred_fuel"].to_numpy(),
                pred_df["pred_nonfuel"].to_numpy(),
                pred_df["pred_opp"].to_numpy(),
            )
            eval_snap = builder.build(eval_cutoff, with_labels=True)
            base_dict = metrics.baseline_scores(eval_snap.features, eval_snap.labels, label_cfg)
            x_max_f_zero = base_dict["sample_submission_zero"]["rmse_fuel"]
            x_max_nf_zero = base_dict["sample_submission_zero"]["rmse_nonfuel"]
            proxy_score = eval_scores.composite(x_max_f_zero, x_max_nf_zero)
            lb_score = eval_scores.leaderboard_composite(LB_X_MAX_FUEL, LB_X_MAX_NONFUEL)
            return FoldResult(
                cutoff=str(eval_cutoff.date()),
                n_train=-1,
                n_eval=len(pred_df),
                scores=eval_scores,
                proxy_score=proxy_score,
                lb_score=lb_score,
                predictions=pred_df,
            )

    train_snaps = [builder.build(c, with_labels=True) for c in train_cutoffs]
    train_panel = stack(train_snaps)

    cal_snap = builder.build(cal_cutoff, with_labels=True)
    eval_snap = builder.build(eval_cutoff, with_labels=True)

    # Seasonal sample weighting
    train_cutoffs_dt = pd.to_datetime(train_panel.cutoffs)
    is_summer = train_cutoffs_dt.isin([
        pd.Timestamp("2024-11-01"), pd.Timestamp("2024-12-01"), pd.Timestamp("2025-01-01")
    ])
    sw = np.ones(len(train_panel.X), dtype=np.float32)
    sw[is_summer] = 2.0

    # 1. Regressors & Hurdle Models
    fuel_model = RegressorEnsemble(
        members=cfg.models.regression_members,
        seed=cfg.models.seed,
        n_jobs=cfg.models.n_jobs,
        tweedie_power=1.25,
        use_gpu=cfg.models.use_gpu,
    ).fit(train_panel.X, train_panel.labels["CLV_fuel"].to_numpy(), sample_weight=sw)

    nonfuel_model = RegressorEnsemble(
        members=cfg.models.regression_members,
        seed=cfg.models.seed,
        n_jobs=cfg.models.n_jobs,
        tweedie_power=1.40,
        use_gpu=cfg.models.use_gpu,
    ).fit(train_panel.X, train_panel.labels["CLV_nonfuel"].to_numpy(), sample_weight=sw)

    opp_weights = [0.40, 0.25, 0.35] if tuple(cfg.models.classification_members) == ("lgbm", "catboost", "xgb") else None
    opp_model = ClassifierEnsemble(
        members=cfg.models.classification_members,
        n_classes=n_classes,
        seed=cfg.models.seed,
        n_jobs=cfg.models.n_jobs,
        weights=opp_weights,
        use_gpu=cfg.models.use_gpu,
    ).fit(train_panel.X, train_panel.labels["Opportunity"].map(label_to_idx).to_numpy(), sample_weight=sw)

    hier_model = HierarchicalOpportunityClassifier(
        n_classes=n_classes,
        seed=cfg.models.seed,
        n_jobs=cfg.models.n_jobs,
        inact_idx=inact_idx,
        stable_idx=stable_idx,
        fg_idx=fg_idx,
        use_gpu=cfg.models.use_gpu,
    ).fit(train_panel.X, train_panel.labels["Opportunity"].map(label_to_idx).to_numpy(), sample_weight=sw)

    store_model = StoreAttachmentClassifier(
        seed=cfg.models.seed,
        n_jobs=cfg.models.n_jobs,
        use_gpu=cfg.models.use_gpu,
    ).fit(train_panel.X, train_panel.labels["CLV_nonfuel"].to_numpy(), sample_weight=sw)

    hurdle_fuel = BaggedTwoPartFuelHurdle(
        seeds=(cfg.models.seed, cfg.models.seed + 1, cfg.models.seed + 2),
        n_jobs=cfg.models.n_jobs,
        use_gpu=cfg.models.use_gpu,
    ).fit(train_panel.X, train_panel.labels["CLV_fuel"].to_numpy(), sample_weight=sw)

    hurdle_nf = BaggedTwoPartNonfuelHurdle(
        seeds=(cfg.models.seed, cfg.models.seed + 1, cfg.models.seed + 2),
        n_jobs=cfg.models.n_jobs,
        use_gpu=cfg.models.use_gpu,
    ).fit(train_panel.X, train_panel.labels["CLV_nonfuel"].to_numpy(), sample_weight=sw)

    # 2. Calibration
    cal_hurdle_fuel = hurdle_fuel.predict(cal_snap.features).reshape(-1, 1)
    cal_fuel_m = np.column_stack([fuel_model.predict_members(cal_snap.features), cal_hurdle_fuel])
    fuel_blend_weights = calibration.fit_blend_weights(cal_fuel_m, cal_snap.labels["CLV_fuel"].to_numpy())

    cal_hurdle_nf = hurdle_nf.predict(cal_snap.features).reshape(-1, 1)
    cal_nonfuel_m = np.column_stack([nonfuel_model.predict_members(cal_snap.features), cal_hurdle_nf])
    nonfuel_blend_weights = calibration.fit_blend_weights(cal_nonfuel_m, cal_snap.labels["CLV_nonfuel"].to_numpy())

    cal_fuel_raw_uncal = cal_fuel_m @ fuel_blend_weights
    fuel_slope, fuel_intercept = calibration.fit_linear_calibration(
        cal_snap.labels["CLV_fuel"].to_numpy(), cal_fuel_raw_uncal
    )
    cal_nonfuel_raw_uncal = cal_nonfuel_m @ nonfuel_blend_weights
    nonfuel_slope, nonfuel_intercept = calibration.fit_linear_calibration(
        cal_snap.labels["CLV_nonfuel"].to_numpy(), cal_nonfuel_raw_uncal
    )

    cal_probs_base = opp_model.predict_proba(cal_snap.features)
    cal_probs_hier = hier_model.predict_proba(cal_snap.features)
    cal_probs = 0.50 * cal_probs_base + 0.50 * cal_probs_hier
    cal_y = cal_snap.labels["Opportunity"].map(label_to_idx).to_numpy()
    dom_classes = [label_to_idx[l] for l in ("Inactivity", "Stable", "Existing-category growth: Fuel") if l in label_to_idx]
    cal_temp, class_weights = calibration.fit_temperature_and_class_weights(
        cal_probs, cal_y, mask=cal_snap.mask, dominant_classes=dom_classes, seed=cfg.models.seed
    )
    cal_probs_scaled = calibration.apply_temperature_scaling(cal_probs, cal_temp)
    cal_probs_masked = calibration.apply_eligibility_mask(cal_probs_scaled, cal_snap.mask)

    p_active_cal = 1.0 - cal_probs_masked[:, inact_idx]
    p_store_cal = store_model.predict_proba(cal_snap.features)

    cal_fuel_calibrated = calibration.apply_linear_calibration(cal_fuel_raw_uncal, fuel_slope, fuel_intercept)
    fuel_shrinkage_power = calibration.fit_shrinkage_power(
        cal_snap.labels["CLV_fuel"].to_numpy(), cal_fuel_calibrated, p_active_cal
    )
    cal_nonfuel_calibrated = calibration.apply_linear_calibration(cal_nonfuel_raw_uncal, nonfuel_slope, nonfuel_intercept)
    nonfuel_alpha, nonfuel_beta = calibration.fit_dual_shrinkage_powers(
        cal_snap.labels["CLV_nonfuel"].to_numpy(),
        cal_nonfuel_calibrated,
        p_active_cal,
        p_store_cal,
        base_active_power=fuel_shrinkage_power,
    )

    cal_f_shrunk = calibration.apply_shrinkage(cal_fuel_calibrated, p_active_cal, fuel_shrinkage_power)
    cal_nf_shrunk = calibration.apply_dual_shrinkage(cal_nonfuel_calibrated, p_active_cal, p_store_cal, nonfuel_alpha, nonfuel_beta)
    inact_threshold = calibration.fit_inactivity_threshold(
        cal_snap.labels["CLV_fuel"].to_numpy(), cal_f_shrunk,
        cal_snap.labels["CLV_nonfuel"].to_numpy(), cal_nf_shrunk,
        p_active_cal,
    )

    # 3. Evaluation on eval_snap
    eval_hurdle_fuel = hurdle_fuel.predict(eval_snap.features).reshape(-1, 1)
    eval_fuel_m = np.column_stack([fuel_model.predict_members(eval_snap.features), eval_hurdle_fuel])
    eval_fuel_raw = eval_fuel_m @ fuel_blend_weights

    eval_hurdle_nf = hurdle_nf.predict(eval_snap.features).reshape(-1, 1)
    eval_nonfuel_m = np.column_stack([nonfuel_model.predict_members(eval_snap.features), eval_hurdle_nf])
    eval_nonfuel_raw = eval_nonfuel_m @ nonfuel_blend_weights

    eval_fuel_calibrated = calibration.apply_linear_calibration(eval_fuel_raw, fuel_slope, fuel_intercept)
    eval_nonfuel_calibrated = calibration.apply_linear_calibration(eval_nonfuel_raw, nonfuel_slope, nonfuel_intercept)

    eval_probs_base = opp_model.predict_proba(eval_snap.features)
    eval_probs_hier = hier_model.predict_proba(eval_snap.features)
    eval_probs = 0.50 * eval_probs_base + 0.50 * eval_probs_hier
    eval_probs_scaled = calibration.apply_temperature_scaling(eval_probs, cal_temp)
    eval_probs_masked = calibration.apply_eligibility_mask(eval_probs_scaled, eval_snap.mask)

    eval_opp_dec = calibration.decide(eval_probs_masked, class_weights)
    eval_opp_labels = np.array([idx_to_label[i] for i in eval_opp_dec])

    p_active_eval = 1.0 - eval_probs_masked[:, inact_idx]
    p_store_eval = store_model.predict_proba(eval_snap.features)

    eval_f_pred = calibration.apply_shrinkage(eval_fuel_calibrated, p_active_eval, fuel_shrinkage_power)
    eval_f_pred = calibration.apply_inactivity_gate(eval_f_pred, p_active_eval, threshold=inact_threshold)

    eval_nf_pred = calibration.apply_dual_shrinkage(eval_nonfuel_calibrated, p_active_eval, p_store_eval, nonfuel_alpha, nonfuel_beta)
    eval_nf_pred = calibration.apply_inactivity_gate(eval_nf_pred, p_active_eval, threshold=inact_threshold)

    # Physical boundary arbitration / active veto
    m1_baskets = eval_snap.features["m1_baskets"].to_numpy() if "m1_baskets" in eval_snap.features.columns else np.zeros(len(eval_snap.features))
    days_last = eval_snap.features["days_since_last_purchase"].to_numpy() if "days_since_last_purchase" in eval_snap.features.columns else np.full(len(eval_snap.features), 999.0)
    m3_fuel_litres = eval_snap.features["m3_fuel_litres"].to_numpy() if "m3_fuel_litres" in eval_snap.features.columns else np.zeros(len(eval_snap.features))
    is_physical_growth = (eval_f_pred > np.log1p(np.maximum(1.25 * m3_fuel_litres, m3_fuel_litres + 1.0)) / 2.352825386185113)
    is_inact = eval_opp_labels == "Inactivity"
    active_veto = is_inact & ((m1_baskets > 0) | (days_last < 45))
    if active_veto.sum() > 0:
        for idx in np.where(active_veto)[0]:
            eval_opp_labels[idx] = "Existing-category growth: Fuel" if is_physical_growth[idx] else "Stable"

    eval_scores = metrics.score(eval_snap.labels, eval_f_pred, eval_nf_pred, eval_opp_labels)
    base_dict = metrics.baseline_scores(eval_snap.features, eval_snap.labels, label_cfg)
    x_max_f_zero = base_dict["sample_submission_zero"]["rmse_fuel"]
    x_max_nf_zero = base_dict["sample_submission_zero"]["rmse_nonfuel"]
    proxy_score = eval_scores.composite(x_max_f_zero, x_max_nf_zero)
    lb_score = eval_scores.leaderboard_composite(LB_X_MAX_FUEL, LB_X_MAX_NONFUEL)

    pred_df = pd.DataFrame({
        "pred_fuel": eval_f_pred,
        "pred_nonfuel": eval_nf_pred,
        "pred_opp": eval_opp_labels,
        "true_fuel": eval_snap.labels["CLV_fuel"].to_numpy(),
        "true_nonfuel": eval_snap.labels["CLV_nonfuel"].to_numpy(),
        "true_opp": eval_snap.labels["Opportunity"].to_numpy(),
    }, index=eval_snap.ids)

    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        pred_df.to_parquet(cache_dir / f"eval_{eval_cutoff.date()}.parquet")

    return FoldResult(
        cutoff=str(eval_cutoff.date()),
        n_train=len(train_panel.X),
        n_eval=len(eval_snap.features),
        scores=eval_scores,
        proxy_score=proxy_score,
        lb_score=lb_score,
        predictions=pred_df,
    )


def evaluate_fold(
    cfg: PipelineConfig,
    builder: SnapshotBuilder,
    eval_cutoff: pd.Timestamp,
    cal_cutoff: pd.Timestamp,
    label_cfg: dict[str, Any],
    cache_dir: Path | None = None,
) -> FoldResult:
    """Trains strictly before cal_cutoff, calibrates on cal_cutoff, and evaluates on eval_cutoff."""
    train_cutoffs = cfg.training_cutoffs(boundary=cal_cutoff)
    if not train_cutoffs:
        import dataclasses
        cfg = dataclasses.replace(cfg, first_train_cutoff="2024-06-01")
        train_cutoffs = cfg.training_cutoffs(boundary=cal_cutoff)
    return train_and_calibrate_fold(
        cfg=cfg,
        builder=builder,
        train_cutoffs=train_cutoffs,
        cal_cutoff=cal_cutoff,
        eval_cutoff=eval_cutoff,
        label_cfg=label_cfg,
        cache_dir=cache_dir,
    )


def evaluate_leave_summer_out(
    cfg: PipelineConfig,
    builder: SnapshotBuilder,
    summer_eval_cutoff: str | pd.Timestamp = "2024-12-01",
    cal_cutoff: str | pd.Timestamp = "2025-06-01",
    cache_dir: Path | None = None,
) -> FoldResult:
    """True Leave-Summer-Out cross-validation evaluating the 2024-12-01 summer cutoff.

    Trains on all pre-summer cutoffs (<= 2024-09-01) and all post-summer cutoffs (>= 2025-03-01)
    whose outcome windows end before 2025-12-01. This guarantees zero label leakage with the
    held-out summer outcome period [2024-12-01, 2025-03-01) while utilizing a rich 10+ snapshot panel.
    """
    eval_cutoff = pd.Timestamp(summer_eval_cutoff)
    cal_t = pd.Timestamp(cal_cutoff)
    label_cfg = cfg.label_config

    pre_summer_cutoffs = [
        c for c in cfg.training_cutoffs(boundary=pd.Timestamp("2024-12-01"))
        if c <= pd.Timestamp("2024-09-01")
    ]
    post_summer_cutoffs = [
        c for c in cfg.training_cutoffs(boundary=pd.Timestamp("2025-12-01"))
        if c >= pd.Timestamp("2025-03-01") and c != cal_t
    ]
    train_cutoffs = sorted(pre_summer_cutoffs + post_summer_cutoffs)
    logger.info("Leave-Summer-Out training cutoffs (N=%d): %s", len(train_cutoffs), [str(c.date()) for c in train_cutoffs])

    return train_and_calibrate_fold(
        cfg=cfg,
        builder=builder,
        train_cutoffs=train_cutoffs,
        cal_cutoff=cal_t,
        eval_cutoff=eval_cutoff,
        label_cfg=label_cfg,
        cache_dir=cache_dir,
    )


def run_backtest(
    cfg: PipelineConfig,
    builder: SnapshotBuilder,
    eval_cutoffs: Sequence[str | pd.Timestamp] = ("2025-06-01", "2025-07-01", "2025-08-01", "2025-09-01"),
    include_seasonal_fold: bool = True,
    cache_dir: Path | None = None,
) -> BacktestSummary:
    """Executes multi-fold rolling origin backtest across the specified historical cutoffs."""
    label_cfg = cfg.label_config
    results: list[FoldResult] = []

    for c in eval_cutoffs:
        eval_t = pd.Timestamp(c)
        cal_t = eval_t - pd.DateOffset(months=3)
        logger.info("Running backtest fold for eval cutoff: %s (cal cutoff: %s)...", eval_t.date(), cal_t.date())
        res = evaluate_fold(cfg, builder, eval_t, cal_t, label_cfg, cache_dir=cache_dir)
        results.append(res)
        logger.info(
            "Fold %s: Fuel RMSE=%.4f, NF RMSE=%.4f, F1=%.4f | Proxy=%.4f | LB Score=%.4f",
            res.cutoff, res.scores.rmse_fuel, res.scores.rmse_nonfuel, res.scores.weighted_f1,
            res.proxy_score, res.lb_score,
        )

    seasonal_res: FoldResult | None = None
    if include_seasonal_fold:
        logger.info("Running Leave-Summer-Out benchmark fold (eval: 2024-12-01, cal: 2025-06-01)...")
        seasonal_res = evaluate_leave_summer_out(cfg, builder, summer_eval_cutoff="2024-12-01", cal_cutoff="2025-06-01", cache_dir=cache_dir)
        logger.info(
            "Leave-Summer-Out Fold %s: Fuel RMSE=%.4f, NF RMSE=%.4f, F1=%.4f | Proxy=%.4f | LB Score=%.4f",
            seasonal_res.cutoff, seasonal_res.scores.rmse_fuel, seasonal_res.scores.rmse_nonfuel,
            seasonal_res.scores.weighted_f1, seasonal_res.proxy_score, seasonal_res.lb_score,
        )

    mean_proxy = float(np.mean([r.proxy_score for r in results]))
    mean_lb = float(np.mean([r.lb_score for r in results]))
    mean_f = float(np.mean([r.scores.rmse_fuel for r in results]))
    mean_nf = float(np.mean([r.scores.rmse_nonfuel for r in results]))
    mean_f1 = float(np.mean([r.scores.weighted_f1 for r in results]))

    return BacktestSummary(
        folds=results,
        mean_proxy_score=mean_proxy,
        mean_lb_score=mean_lb,
        mean_fuel_rmse=mean_f,
        mean_nonfuel_rmse=mean_nf,
        mean_weighted_f1=mean_f1,
        seasonal_fold_result=seasonal_res,
    )


def paired_bootstrap_test(
    preds_baseline: pd.DataFrame,
    preds_candidate: pd.DataFrame,
    n_replicates: int = 1000,
    seed: int = 42,
    x_max_fuel: float = LB_X_MAX_FUEL,
    x_max_nonfuel: float = LB_X_MAX_NONFUEL,
) -> dict[str, float]:
    """Paired customer bootstrap test to determine statistical significance of delta."""
    rng = np.random.default_rng(seed)
    n = len(preds_baseline)
    deltas = np.zeros(n_replicates)

    y_f = preds_baseline["true_fuel"].to_numpy()
    y_nf = preds_baseline["true_nonfuel"].to_numpy()
    y_opp = preds_baseline["true_opp"].to_numpy()

    b_f = preds_baseline["pred_fuel"].to_numpy()
    b_nf = preds_baseline["pred_nonfuel"].to_numpy()
    b_opp = preds_baseline["pred_opp"].to_numpy()

    c_f = preds_candidate["pred_fuel"].to_numpy()
    c_nf = preds_candidate["pred_nonfuel"].to_numpy()
    c_opp = preds_candidate["pred_opp"].to_numpy()

    for i in range(n_replicates):
        idx = rng.integers(0, n, size=n)
        # Baseline score
        sc_b = metrics.score(pd.DataFrame({"CLV_fuel": y_f[idx], "CLV_nonfuel": y_nf[idx], "Opportunity": y_opp[idx]}),
                             b_f[idx], b_nf[idx], b_opp[idx]).leaderboard_composite(x_max_fuel, x_max_nonfuel)
        # Candidate score
        sc_c = metrics.score(pd.DataFrame({"CLV_fuel": y_f[idx], "CLV_nonfuel": y_nf[idx], "Opportunity": y_opp[idx]}),
                             c_f[idx], c_nf[idx], c_opp[idx]).leaderboard_composite(x_max_fuel, x_max_nonfuel)
        deltas[i] = sc_c - sc_b

    mean_delta = float(np.mean(deltas))
    ci_lower = float(np.percentile(deltas, 2.5))
    ci_upper = float(np.percentile(deltas, 97.5))
    p_value = float(np.mean(deltas <= 0.0))

    return {
        "mean_delta": mean_delta,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_value,
    }
