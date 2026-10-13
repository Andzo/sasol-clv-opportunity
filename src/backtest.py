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
from src.models import (
    ClassifierEnsemble,
    HierarchicalOpportunityClassifier,
    RegressorEnsemble,
    StoreAttachmentClassifier,
)

logger = logging.getLogger(__name__)

# Official empirically verified Leaderboard persistence x_max
LB_X_MAX_FUEL: float = 0.69046
LB_X_MAX_NONFUEL: float = 0.92955


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


def evaluate_fold(
    cfg: PipelineConfig,
    builder: SnapshotBuilder,
    eval_cutoff: pd.Timestamp,
    cal_cutoff: pd.Timestamp,
    label_cfg: dict[str, Any],
    cache_dir: Path | None = None,
) -> FoldResult:
    """Trains strictly before cal_cutoff, calibrates on cal_cutoff, and evaluates on eval_cutoff."""
    opp_labels = label_cfg["opportunity_labels"]
    n_classes = len(opp_labels)
    label_to_idx = {l: i for i, l in enumerate(opp_labels)}
    idx_to_label = {i: l for i, l in enumerate(opp_labels)}

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

    train_cutoffs = cfg.training_cutoffs(boundary=cal_cutoff)
    if not train_cutoffs:
        import dataclasses
        cfg = dataclasses.replace(cfg, first_train_cutoff="2024-06-01")
        train_cutoffs = cfg.training_cutoffs(boundary=cal_cutoff)
    train_snaps = [builder.build(c, with_labels=True) for c in train_cutoffs]
    train_panel = stack(train_snaps)

    cal_snap = builder.build(cal_cutoff, with_labels=True)
    eval_snap = builder.build(eval_cutoff, with_labels=True)

    # 1. Regressors
    fuel_model = RegressorEnsemble(
        members=cfg.models.regression_members,
        seed=cfg.models.seed,
        n_jobs=cfg.models.n_jobs,
        tweedie_power=1.25,
        use_gpu=cfg.models.use_gpu,
    ).fit(train_panel.X, train_panel.labels["CLV_fuel"].to_numpy())

    nonfuel_model = RegressorEnsemble(
        members=cfg.models.regression_members,
        seed=cfg.models.seed,
        n_jobs=cfg.models.n_jobs,
        tweedie_power=1.40,
        use_gpu=cfg.models.use_gpu,
    ).fit(train_panel.X, train_panel.labels["CLV_nonfuel"].to_numpy())

    opp_weights = [0.40, 0.25, 0.35] if tuple(cfg.models.classification_members) == ("lgbm", "catboost", "xgb") else None
    opp_model = ClassifierEnsemble(
        members=cfg.models.classification_members,
        n_classes=n_classes,
        seed=cfg.models.seed,
        n_jobs=cfg.models.n_jobs,
        weights=opp_weights,
        use_gpu=cfg.models.use_gpu,
    ).fit(train_panel.X, train_panel.labels["Opportunity"].map(label_to_idx).to_numpy())

    hier_model = HierarchicalOpportunityClassifier(
        n_classes=n_classes,
        seed=cfg.models.seed,
        n_jobs=cfg.models.n_jobs,
        inact_idx=label_to_idx["Inactivity"],
        stable_idx=label_to_idx["Stable"],
        fg_idx=label_to_idx["Existing-category growth: Fuel"],
        use_gpu=cfg.models.use_gpu,
    ).fit(train_panel.X, train_panel.labels["Opportunity"].map(label_to_idx).to_numpy())

    store_model = StoreAttachmentClassifier(
        seed=cfg.models.seed,
        n_jobs=cfg.models.n_jobs,
        use_gpu=cfg.models.use_gpu,
    ).fit(train_panel.X, train_panel.labels["CLV_nonfuel"].to_numpy())

    # 2. Calibration
    cal_fuel_m = fuel_model.predict_members(cal_snap.features)
    fuel_blend_weights = calibration.fit_blend_weights(cal_fuel_m, cal_snap.labels["CLV_fuel"].to_numpy())
    fuel_model.set_weights(fuel_blend_weights)

    cal_nonfuel_m = nonfuel_model.predict_members(cal_snap.features)
    nonfuel_blend_weights = calibration.fit_blend_weights(cal_nonfuel_m, cal_snap.labels["CLV_nonfuel"].to_numpy())
    nonfuel_model.set_weights(nonfuel_blend_weights)

    cal_fuel_raw = fuel_model.predict(cal_snap.features)
    fuel_slope, fuel_intercept = calibration.fit_linear_calibration(
        cal_snap.labels["CLV_fuel"].to_numpy(), cal_fuel_raw
    )
    cal_fuel_cal = calibration.apply_linear_calibration(cal_fuel_raw, fuel_slope, fuel_intercept)

    cal_nf_raw = nonfuel_model.predict(cal_snap.features)
    nf_slope, nf_intercept = calibration.fit_linear_calibration(
        cal_snap.labels["CLV_nonfuel"].to_numpy(), cal_nf_raw
    )
    cal_nf_cal = calibration.apply_linear_calibration(cal_nf_raw, nf_slope, nf_intercept)

    cal_p_base = opp_model.predict_proba(cal_snap.features)
    cal_p_hier = hier_model.predict_proba(cal_snap.features)
    cal_p = 0.50 * cal_p_base + 0.50 * cal_p_hier
    cal_y = cal_snap.labels["Opportunity"].map(label_to_idx).to_numpy()
    dom_classes = [label_to_idx[l] for l in ("Inactivity", "Stable", "Existing-category growth: Fuel") if l in label_to_idx]
    cal_temp, class_weights = calibration.fit_temperature_and_class_weights(
        cal_p, cal_y, mask=cal_snap.mask, dominant_classes=dom_classes, seed=cfg.models.seed
    )
    cal_p_scaled = calibration.apply_temperature_scaling(cal_p, cal_temp)
    cal_p_masked = calibration.apply_eligibility_mask(cal_p_scaled, cal_snap.mask)

    inact_idx = label_to_idx["Inactivity"]
    p_act_cal = 1.0 - cal_p_masked[:, inact_idx]
    p_store_cal = store_model.predict_proba(cal_snap.features)

    fuel_shrink = calibration.fit_shrinkage_power(cal_snap.labels["CLV_fuel"].to_numpy(), cal_fuel_cal, p_act_cal)
    nf_alpha, nf_beta = calibration.fit_dual_shrinkage_powers(
        cal_snap.labels["CLV_nonfuel"].to_numpy(), cal_nf_cal, p_act_cal, p_store_cal, base_active_power=fuel_shrink
    )

    cal_f_shrunk = calibration.apply_shrinkage(cal_fuel_cal, p_act_cal, fuel_shrink)
    cal_nf_shrunk = calibration.apply_dual_shrinkage(cal_nf_cal, p_act_cal, p_store_cal, nf_alpha, nf_beta)
    inact_thresh = calibration.fit_inactivity_threshold(
        cal_snap.labels["CLV_fuel"].to_numpy(), cal_f_shrunk,
        cal_snap.labels["CLV_nonfuel"].to_numpy(), cal_nf_shrunk,
        p_act_cal,
    )

    # 3. Evaluation Scoring
    eval_f_raw = calibration.apply_linear_calibration(fuel_model.predict(eval_snap.features), fuel_slope, fuel_intercept)
    eval_nf_raw = calibration.apply_linear_calibration(nonfuel_model.predict(eval_snap.features), nf_slope, nf_intercept)
    eval_p = 0.50 * opp_model.predict_proba(eval_snap.features) + 0.50 * hier_model.predict_proba(eval_snap.features)
    eval_p_scaled = calibration.apply_temperature_scaling(eval_p, cal_temp)
    eval_p_masked = calibration.apply_eligibility_mask(eval_p_scaled, eval_snap.mask)
    eval_opp_dec = calibration.decide(eval_p_masked, class_weights)
    eval_opp_labels = np.array([idx_to_label[i] for i in eval_opp_dec])

    p_act_eval = 1.0 - eval_p_masked[:, inact_idx]
    p_store_eval = store_model.predict_proba(eval_snap.features)
    eval_f_pred = calibration.apply_shrinkage(eval_f_raw, p_act_eval, fuel_shrink)
    eval_f_pred = calibration.apply_inactivity_gate(eval_f_pred, p_act_eval, threshold=inact_thresh)
    eval_nf_pred = calibration.apply_dual_shrinkage(eval_nf_raw, p_act_eval, p_store_eval, nf_alpha, nf_beta)
    eval_nf_pred = calibration.apply_inactivity_gate(eval_nf_pred, p_act_eval, threshold=inact_thresh)

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
        s_eval = pd.Timestamp("2024-12-01")
        s_cal = pd.Timestamp("2024-09-01")
        import dataclasses
        cfg_seasonal = dataclasses.replace(cfg, first_train_cutoff="2024-06-01")
        logger.info("Running seasonal Dec-Feb benchmark fold: %s (cal: %s)...", s_eval.date(), s_cal.date())
        seasonal_res = evaluate_fold(cfg_seasonal, builder, s_eval, s_cal, label_cfg, cache_dir=cache_dir)
        logger.info(
            "Seasonal Fold %s: Fuel RMSE=%.4f, NF RMSE=%.4f, F1=%.4f | Proxy=%.4f | LB Score=%.4f",
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
                             b_f[idx], b_nf[idx], b_opp[idx]).composite(x_max_fuel, x_max_nonfuel)
        # Candidate score
        sc_c = metrics.score(pd.DataFrame({"CLV_fuel": y_f[idx], "CLV_nonfuel": y_nf[idx], "Opportunity": y_opp[idx]}),
                             c_f[idx], c_nf[idx], c_opp[idx]).composite(x_max_fuel, x_max_nonfuel)
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
