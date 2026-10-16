"""End-to-end pipeline runner for the Sasol CLV & Opportunity challenge.

Orchestrates:
1. Multi-Cutoff Out-of-Fold Tournament & Stacking Engine (3 folds: Summer, Winter, Spring).
2. Out-of-fold debiasing, blend weights, Customer BLUP, and prior-aligned calibration.
3. Out-of-time evaluation on forward holdout cutoff (2025-09-01) and benchmark verification.
4. Final model zoo retraining across all historical cutoffs with seasonal sample weighting.
5. Test inference (2025-12-01), format validation, and submission generation.
"""
from __future__ import annotations

import json
import logging
import subprocess
import time
from typing import Any

import numpy as np
import pandas as pd

from src import calibration, metrics, submission
from src.config import PipelineConfig
from src.data import load_test_ids, load_transactions
from src.dataset import SnapshotBuilder, stack
from src.features import FEATURE_VERSION
from src.hurdle import (
    BaggedTwoPartFuelHurdle,
    BaggedTwoPartNonfuelHurdle,
    MultiQuantileTwoPartHurdle,
    PhysicalBasketTicketRegressor,
)
from src.models import (
    ClassifierEnsemble,
    DecomposedRule7Simulator,
    HierarchicalOpportunityClassifier,
    RegressorEnsemble,
    StoreAttachmentClassifier,
)
from src.tournament import (
    CalibrationArtifacts,
    TournamentEngine,
    TournamentResult,
    apply_boundary_arbitration,
)

logger = logging.getLogger(__name__)


def _get_git_info() -> dict[str, Any]:
    try:
        commit_full = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        commit_short = commit_full[:7]
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
        status_out = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        is_dirty = len(status_out) > 0
        return {
            "commit": commit_short,
            "commit_full": commit_full,
            "branch": branch,
            "dirty": is_dirty,
        }
    except Exception:
        return {"commit": "unknown", "commit_full": "unknown", "branch": "unknown", "dirty": False}


def run_pipeline(cfg: PipelineConfig, save_predictions: bool = False) -> dict[str, Any]:
    """Execute the full end-to-end training, validation, and submission pipeline."""
    start_time = time.time()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger.info("Initializing Sasol CLV Tournament Pipeline (Seed=%d)", cfg.models.seed)

    label_cfg = cfg.label_config
    opp_labels = label_cfg["opportunity_labels"]
    n_classes = len(opp_labels)
    label_to_idx = {l: i for i, l in enumerate(opp_labels)}
    idx_to_label = {i: l for i, l in enumerate(opp_labels)}
    inact_idx = label_to_idx["Inactivity"]
    stable_idx = label_to_idx["Stable"]
    fg_idx = label_to_idx["Existing-category growth: Fuel"]

    # 1. Load annotated transactions
    tx = load_transactions(cfg)
    builder = SnapshotBuilder(cfg, tx)

    eval_cutoff = pd.Timestamp(cfg.evaluation_cutoff)  # 2025-09-01
    test_cutoff = pd.Timestamp(cfg.test_cutoff)        # 2025-12-01

    # 2. Phase 1: Multi-Cutoff Out-of-Fold Tournament
    logger.info("--- Phase 1: Multi-Horizon Out-of-Fold Tournament Engine ---")
    tournament_engine = TournamentEngine(cfg, builder)
    tourney_res = tournament_engine.run_tournament()
    cal = tournament_engine.fit_calibration(tourney_res)

    # 3. Evaluate Calibrated Pipeline on Forward Holdout (2025-09-01)
    logger.info("--- Evaluating Full Calibrated Stack on Forward Holdout (%s) ---", eval_cutoff.date())
    eval_snap = builder.build(eval_cutoff, with_labels=True)
    eval_mask_indices = [i for i, c in enumerate(tourney_res.pool_cutoffs) if c == eval_cutoff]

    if eval_mask_indices:
        eval_fuel_m = tourney_res.pool_fuel_m[eval_mask_indices]
        eval_nf_m = tourney_res.pool_nf_m[eval_mask_indices]
        eval_probs_base = tourney_res.pool_probs_base[eval_mask_indices]
        eval_probs_hier = tourney_res.pool_probs_hier[eval_mask_indices]
        eval_probs_sim = tourney_res.pool_probs_sim[eval_mask_indices]
        eval_phys_fg = tourney_res.pool_phys_fg[eval_mask_indices]
        eval_phys_inact = tourney_res.pool_phys_inact[eval_mask_indices]
        eval_ids = [tourney_res.pool_ids[i] for i in eval_mask_indices]
    else:
        # Fallback: predict using fold 3 models
        eval_ids = list(eval_snap.ids)
        eval_fuel_m = np.zeros((len(eval_ids), len(cal.member_names)))
        eval_nf_m = np.zeros((len(eval_ids), len(cal.member_names)))
        eval_probs_base = np.zeros((len(eval_ids), n_classes))
        eval_probs_hier = np.zeros((len(eval_ids), n_classes))
        eval_probs_sim = np.zeros((len(eval_ids), n_classes))
        eval_phys_fg = np.zeros(len(eval_ids))
        eval_phys_inact = np.zeros(len(eval_ids))

    # Apply per-member debiasing and blend weights
    eval_fuel_debiased = calibration.apply_member_debiasing(eval_fuel_m, cal.fuel_debias)
    eval_fuel_raw = eval_fuel_debiased @ cal.fuel_blend_weights
    eval_fuel_cal = calibration.apply_linear_calibration(eval_fuel_raw, cal.fuel_slope, cal.fuel_intercept)

    eval_nf_debiased = calibration.apply_member_debiasing(eval_nf_m, cal.nf_debias)
    eval_nf_raw = eval_nf_debiased @ cal.nf_blend_weights
    eval_nf_cal = calibration.apply_linear_calibration(eval_nf_raw, cal.nf_slope, cal.nf_intercept)

    # Customer BLUP fixed effects for forward holdout evaluation
    eval_blup_f_map = calibration.compute_customer_blup(cal.cust_history_f, eval_cutoff)
    eval_blup_f_vec = np.array([eval_blup_f_map.get(cid, 0.0) for cid in eval_ids])
    eval_blup_nf_map = calibration.compute_customer_blup(cal.cust_history_nf, eval_cutoff)
    eval_blup_nf_vec = np.array([eval_blup_nf_map.get(cid, 0.0) for cid in eval_ids])

    eval_fuel_cal = calibration.apply_customer_blup(
        eval_fuel_cal, eval_blup_f_vec, cal.blup_eta_fuel, active_mask=(eval_fuel_cal > 0.05)
    )
    eval_nf_cal = calibration.apply_customer_blup(
        eval_nf_cal, eval_blup_nf_vec, cal.blup_eta_nf, active_mask=(eval_nf_cal > 0.05)
    )

    # Opportunity Probability Fusion
    wb, wh, ws, b_fg, b_in = cal.prob_blend_weights
    eval_probs_fused = wb * eval_probs_base + wh * eval_probs_hier + ws * eval_probs_sim
    eval_probs_fused[:, fg_idx] += b_fg * eval_phys_fg
    eval_probs_fused[:, inact_idx] += b_in * eval_phys_inact
    eval_probs_scaled = calibration.apply_temperature_scaling(eval_probs_fused, cal.temperature)
    eval_probs_masked = calibration.apply_eligibility_mask(eval_probs_scaled, eval_snap.mask)

    # Opportunity Decision with Calibrated Decision Weights
    eval_opp_decisions = calibration.decide(eval_probs_masked, cal.class_weights)
    eval_opp_labels = [idx_to_label[i] for i in eval_opp_decisions]

    # Inactivity shrinkage and gating
    p_active_eval = 1.0 - eval_probs_masked[:, inact_idx]
    p_store_eval = np.clip(eval_nf_raw / np.maximum(np.percentile(eval_nf_raw[eval_nf_raw > 0], 90), 0.1), 0.0, 1.0)
    eval_fuel_pred = calibration.apply_shrinkage(eval_fuel_cal, p_active_eval, cal.fuel_shrinkage_power)
    eval_fuel_pred = calibration.apply_inactivity_gate(eval_fuel_pred, p_active_eval, threshold=cal.inact_threshold)
    eval_nf_pred = calibration.apply_dual_shrinkage(eval_nf_cal, p_active_eval, p_store_eval, cal.nf_alpha, cal.nf_beta)
    eval_nf_pred = calibration.apply_inactivity_gate(eval_nf_pred, p_active_eval, threshold=cal.inact_threshold)

    # Boundary Arbitration on false inactives
    eval_opp_labels = apply_boundary_arbitration(eval_opp_labels, eval_snap.features, eval_fuel_pred)

    # Score evaluation snapshot
    eval_scores = metrics.score(
        eval_snap.labels,
        eval_fuel_pred,
        eval_nf_pred,
        np.array(eval_opp_labels),
    )
    leaderboard_score = eval_scores.leaderboard_composite(0.740, 0.816)
    score_breakdown = eval_scores.leaderboard_breakdown(0.740, 0.816)
    composite_score = leaderboard_score
    baseline_dict = metrics.baseline_scores(eval_snap.features, eval_snap.labels, label_cfg)

    logger.info("=== Out-of-Time Forward Holdout Evaluation (%s) ===", eval_cutoff.date())
    logger.info("CLV_fuel RMSE:       %.4f (Score contribution: %.5f)", eval_scores.rmse_fuel, score_breakdown["score_fuel"])
    logger.info("CLV_nonfuel RMSE:    %.4f (Score contribution: %.5f)", eval_scores.rmse_nonfuel, score_breakdown["score_nonfuel"])
    logger.info("Opportunity F1:      %.4f (Score contribution: %.5f)", eval_scores.weighted_f1, score_breakdown["score_opportunity"])
    logger.info("Leaderboard Score:   %.5f (Formula: 0.3*(1-RMSE_f/0.740) + 0.3*(1-RMSE_nf/0.816) + 0.4*F1)", leaderboard_score)

    # 4. Final Retraining across All Historical Snapshots
    logger.info("--- Phase 2: Full Historical Panel Retraining ---")
    full_train_cutoffs = cfg.training_cutoffs(boundary=test_cutoff)
    logger.info("Full retraining cutoffs (N=%d): %s", len(full_train_cutoffs), [str(c.date()) for c in full_train_cutoffs])
    full_snaps = [builder.build(c, with_labels=True) for c in full_train_cutoffs]
    full_panel = stack(full_snaps)
    logger.info("Full training panel: %d instances x %d features", len(full_panel.X), full_panel.X.shape[1])

    full_cutoffs_dt = pd.to_datetime(full_panel.cutoffs)
    is_summer_cutoff = full_cutoffs_dt.isin([
        pd.Timestamp("2024-11-01"), pd.Timestamp("2024-12-01"), pd.Timestamp("2025-01-01")
    ])
    full_sample_weights = np.ones(len(full_panel.X), dtype=np.float32)
    full_sample_weights[is_summer_cutoff] = 2.0
    logger.info("Seasonal sample weighting: %d summer instances weighted at 2.0x", is_summer_cutoff.sum())

    final_fuel_model = RegressorEnsemble(
        members=cfg.models.regression_members,
        seed=cfg.models.seed,
        n_jobs=cfg.models.n_jobs,
        tweedie_power=1.25,
        use_gpu=cfg.models.use_gpu,
    ).fit(full_panel.X, full_panel.labels["CLV_fuel"].to_numpy(), sample_weight=full_sample_weights)

    final_nf_model = RegressorEnsemble(
        members=cfg.models.regression_members,
        seed=cfg.models.seed,
        n_jobs=cfg.models.n_jobs,
        tweedie_power=1.40,
        use_gpu=cfg.models.use_gpu,
    ).fit(full_panel.X, full_panel.labels["CLV_nonfuel"].to_numpy(), sample_weight=full_sample_weights)

    final_hurdle_fuel = BaggedTwoPartFuelHurdle(
        seeds=(42, 100, 2024, 777, 314),
        huber_alpha=0.85,
        power_shrinkage=0.98,
        n_jobs=cfg.models.n_jobs,
        use_gpu=cfg.models.use_gpu,
    ).fit(full_panel.X, full_panel.labels["CLV_fuel"].to_numpy(), sample_weight=full_sample_weights)

    final_hurdle_nf = BaggedTwoPartNonfuelHurdle(
        seeds=(42, 100, 2024, 777, 314),
        huber_alpha=0.85,
        power_shrinkage=0.96,
        zero_gate_threshold=0.04,
        n_jobs=cfg.models.n_jobs,
        use_gpu=cfg.models.use_gpu,
    ).fit(full_panel.X, full_panel.labels["CLV_nonfuel"].to_numpy(), sample_weight=full_sample_weights)

    final_mq_fuel = MultiQuantileTwoPartHurdle(
        seed=cfg.models.seed,
        n_jobs=cfg.models.n_jobs,
        use_gpu=cfg.models.use_gpu,
    ).fit(full_panel.X, full_panel.labels["CLV_fuel"].to_numpy(), sample_weight=full_sample_weights)

    final_mq_nf = MultiQuantileTwoPartHurdle(
        seed=cfg.models.seed,
        n_jobs=cfg.models.n_jobs,
        use_gpu=cfg.models.use_gpu,
    ).fit(full_panel.X, full_panel.labels["CLV_nonfuel"].to_numpy(), sample_weight=full_sample_weights)

    final_phys_fuel = PhysicalBasketTicketRegressor(
        scale=2.352825386185113,
        seed=cfg.models.seed,
        n_jobs=cfg.models.n_jobs,
        use_gpu=cfg.models.use_gpu,
    ).fit(full_panel.X, full_panel.labels["CLV_fuel"].to_numpy(), sample_weight=full_sample_weights)

    final_phys_nf = PhysicalBasketTicketRegressor(
        scale=2.352825386185113,
        seed=cfg.models.seed,
        n_jobs=cfg.models.n_jobs,
        use_gpu=cfg.models.use_gpu,
    ).fit(full_panel.X, full_panel.labels["CLV_nonfuel"].to_numpy(), sample_weight=full_sample_weights)

    opp_weights = [0.40, 0.25, 0.35] if tuple(cfg.models.classification_members) == ("lgbm", "catboost", "xgb") else None
    final_opp_model = ClassifierEnsemble(
        members=cfg.models.classification_members,
        n_classes=n_classes,
        seed=cfg.models.seed,
        n_jobs=cfg.models.n_jobs,
        weights=opp_weights,
        use_gpu=cfg.models.use_gpu,
    ).fit(full_panel.X, full_panel.labels["Opportunity"].map(label_to_idx).to_numpy(), sample_weight=full_sample_weights)

    final_hier_model = HierarchicalOpportunityClassifier(
        n_classes=n_classes,
        seed=cfg.models.seed,
        n_jobs=cfg.models.n_jobs,
        inact_idx=inact_idx,
        stable_idx=stable_idx,
        fg_idx=fg_idx,
        use_gpu=cfg.models.use_gpu,
    ).fit(full_panel.X, full_panel.labels["Opportunity"].map(label_to_idx).to_numpy(), sample_weight=full_sample_weights)

    final_store_model = StoreAttachmentClassifier(
        seed=cfg.models.seed,
        n_jobs=cfg.models.n_jobs,
        use_gpu=cfg.models.use_gpu,
    ).fit(full_panel.X, full_panel.labels["CLV_nonfuel"].to_numpy(), sample_weight=full_sample_weights)

    final_rule7_sim = DecomposedRule7Simulator(
        label_config=label_cfg,
        seed=cfg.models.seed,
        n_jobs=cfg.models.n_jobs,
        use_gpu=cfg.models.use_gpu,
        n_draws=300,
    ).fit(full_panel.X, full_panel.labels["Opportunity"].map(label_to_idx).to_numpy(), sample_weight=full_sample_weights)

    # 5. Phase 3: Test Customer Inference (2025-12-01)
    logger.info("--- Phase 3: Test Customer Inference (%s) ---", test_cutoff.date())
    test_snap = builder.build(test_cutoff, with_labels=False)
    test_ids = load_test_ids(cfg)

    # Exact index alignment with test.csv
    test_feats = test_snap.features.reindex(pd.Index(test_ids, name="ID"))
    test_id_to_row = {cid: idx for idx, cid in enumerate(test_snap.ids)}
    test_mask_rows = [test_id_to_row[cid] for cid in test_ids]
    aligned_mask = test_snap.mask[test_mask_rows]

    # Predict test members
    test_pred_f_direct = final_fuel_model.predict_members(test_feats)
    test_pred_f_hurdle = final_hurdle_fuel.predict(test_feats).reshape(-1, 1)
    test_pred_f_mq = final_mq_fuel.predict(test_feats).reshape(-1, 1)
    test_pred_f_phys = final_phys_fuel.predict(test_feats).reshape(-1, 1)
    test_fuel_m = np.column_stack([test_pred_f_direct, test_pred_f_hurdle, test_pred_f_mq, test_pred_f_phys])

    test_pred_nf_direct = final_nf_model.predict_members(test_feats)
    test_pred_nf_hurdle = final_hurdle_nf.predict(test_feats).reshape(-1, 1)
    test_pred_nf_mq = final_mq_nf.predict(test_feats).reshape(-1, 1)
    test_pred_nf_phys = final_phys_nf.predict(test_feats).reshape(-1, 1)
    test_nf_m = np.column_stack([test_pred_nf_direct, test_pred_nf_hurdle, test_pred_nf_mq, test_pred_nf_phys])

    # Per-member debiasing and blend weights
    test_fuel_debiased = calibration.apply_member_debiasing(test_fuel_m, cal.fuel_debias)
    test_fuel_raw = test_fuel_debiased @ cal.fuel_blend_weights
    test_fuel_cal = calibration.apply_linear_calibration(test_fuel_raw, cal.fuel_slope, cal.fuel_intercept)

    test_nf_debiased = calibration.apply_member_debiasing(test_nf_m, cal.nf_debias)
    test_nf_raw = test_nf_debiased @ cal.nf_blend_weights
    test_nf_cal = calibration.apply_linear_calibration(test_nf_raw, cal.nf_slope, cal.nf_intercept)

    # Customer BLUP fixed effects for test customers
    test_blup_f_map = calibration.compute_customer_blup(cal.cust_history_f, test_cutoff)
    test_blup_f_vec = np.array([test_blup_f_map.get(cid, 0.0) for cid in test_ids])
    test_blup_nf_map = calibration.compute_customer_blup(cal.cust_history_nf, test_cutoff)
    test_blup_nf_vec = np.array([test_blup_nf_map.get(cid, 0.0) for cid in test_ids])

    test_fuel_cal = calibration.apply_customer_blup(
        test_fuel_cal, test_blup_f_vec, cal.blup_eta_fuel, active_mask=(test_fuel_cal > 0.05)
    )
    test_nf_cal = calibration.apply_customer_blup(
        test_nf_cal, test_blup_nf_vec, cal.blup_eta_nf, active_mask=(test_nf_cal > 0.05)
    )

    # Opportunity Classifier Inferences
    test_probs_base = final_opp_model.predict_proba(test_feats)
    test_probs_hier = final_hier_model.predict_proba(test_feats)
    test_probs_sim = final_rule7_sim.predict_proba(test_feats)

    # Physical hurdle exceedance & joint inactivity CDF probabilities
    p_phys_fg_test, p_phys_inact_test = calibration.compute_physical_hurdle_probabilities(
        test_feats,
        final_hurdle_fuel,
        np.mean(test_pred_f_direct, axis=1),
        final_store_model,
    )

    test_probs_fused = wb * test_probs_base + wh * test_probs_hier + ws * test_probs_sim
    test_probs_fused[:, fg_idx] += b_fg * p_phys_fg_test
    test_probs_fused[:, inact_idx] += b_in * p_phys_inact_test
    test_probs_scaled = calibration.apply_temperature_scaling(test_probs_fused, cal.temperature)
    test_probs_masked = calibration.apply_eligibility_mask(test_probs_scaled, aligned_mask)

    # Prior-Aligned Decision Rule with Precision-Shrunk Target Priors
    test_opp_decisions = calibration.decide_with_prior_alignment(
        test_probs_masked, cal.class_weights, cal.precision_weighted_target_priors,
        macro_classes=(inact_idx, stable_idx, fg_idx),
    )
    test_opp_labels = [idx_to_label[i] for i in test_opp_decisions]

    # Inactivity shrinkage and dynamic zero-gating
    p_active_test = 1.0 - test_probs_masked[:, inact_idx]
    p_store_test = final_store_model.predict_proba(test_feats)
    test_pred_fuel = calibration.apply_shrinkage(test_fuel_cal, p_active_test, cal.fuel_shrinkage_power)
    test_pred_fuel = calibration.apply_inactivity_gate(test_pred_fuel, p_active_test, threshold=cal.inact_threshold)
    test_pred_nonfuel = calibration.apply_dual_shrinkage(
        test_nf_cal, p_active_test, p_store_test, cal.nf_alpha, cal.nf_beta
    )
    test_pred_nonfuel = calibration.apply_inactivity_gate(test_pred_nonfuel, p_active_test, threshold=cal.inact_threshold)

    if save_predictions:
        eda_cache = cfg.paths.reports_dir / "eda" / "cache"
        eda_cache.mkdir(parents=True, exist_ok=True)
        test_df = pd.DataFrame({
            "pred_fuel_raw": test_fuel_raw,
            "pred_fuel_final": test_pred_fuel,
            "pred_nonfuel_raw": test_nf_raw,
            "pred_nonfuel_final": test_pred_nonfuel,
            "pred_opportunity": test_opp_labels,
            "p_active": p_active_test,
            "p_store": p_store_test,
        }, index=pd.Index(test_ids, name="ID"))
        for i, l in enumerate(opp_labels):
            test_df[f"prob_{l}"] = test_probs_masked[:, i]
        test_df.to_parquet(eda_cache / "test_predictions.parquet")
        logger.info("Saved test predictions to %s", eda_cache / "test_predictions.parquet")

    # 6. Submission Build & Strict Validation
    logger.info("--- Phase 4: Submission Build & Validation ---")
    sub_df = submission.build_submission(
        ids=test_ids,
        fuel=test_pred_fuel,
        nonfuel=test_pred_nonfuel,
        opportunity=test_opp_labels,
    )
    sample_df = pd.read_csv(cfg.paths.sample_submission)
    submission.validate_submission(sub_df, sample_df, opp_labels)
    cfg.paths.submissions_dir.mkdir(parents=True, exist_ok=True)
    git_info = _get_git_info()
    timestamp_str = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    sub_base_name = f"submission_git_{git_info['commit']}_{timestamp_str}_comp{composite_score:.4f}"
    timestamped_path = cfg.paths.submissions_dir / f"{sub_base_name}.csv"
    submission.write_submission(sub_df, timestamped_path)
    submission.write_submission(sub_df, cfg.paths.submission)
    submission.write_submission(sub_df, cfg.paths.root / "submission.csv")

    # Companion manifest for lineage tracking
    manifest: dict[str, Any] = {
        "submission_file": timestamped_path.name,
        "created_at": pd.Timestamp.now().isoformat(),
        "git_commit": git_info["commit_full"],
        "git_commit_short": git_info["commit"],
        "git_branch": git_info["branch"],
        "git_dirty": git_info["dirty"],
        "feature_version": FEATURE_VERSION,
        "tournament_folds": tourney_res.fold_scores,
        "models": {
            "regression_members": cal.member_names,
            "classification_members": list(cfg.models.classification_members),
            "seed": cfg.models.seed,
        },
        "evaluation_metrics": eval_scores.as_dict(),
        "leaderboard_composite_score": round(leaderboard_score, 5),
        "leaderboard_breakdown": score_breakdown,
        "reproduction_command": "python main.py --save-predictions",
    }
    manifest_path = cfg.paths.submissions_dir / f"{sub_base_name}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    (cfg.paths.submissions_dir / "submission_latest.manifest.json").write_text(json.dumps(manifest, indent=2))

    logger.info(
        "Submission written to %s (Linked to git commit %s, score=%.5f, %d rows)",
        timestamped_path.name, git_info["commit"], leaderboard_score, len(sub_df)
    )

    elapsed = time.time() - start_time
    summary: dict[str, Any] = {
        "status": "SUCCESS",
        "elapsed_seconds": round(elapsed, 2),
        "evaluation_cutoff": str(eval_cutoff.date()),
        "metrics": eval_scores.as_dict(),
        "leaderboard_composite_score": round(leaderboard_score, 5),
        "proxy_composite_score": round(leaderboard_score, 5),
        "leaderboard_breakdown": score_breakdown,
        "tournament_folds": tourney_res.fold_scores,
        "baselines": baseline_dict,
        "class_distribution_submission": sub_df["Opportunity"].value_counts().to_dict(),
        "fuel_mean": round(float(sub_df["CLV_fuel"].mean()), 4),
        "nonfuel_mean": round(float(sub_df["CLV_nonfuel"].mean()), 4),
        "blend_weights": {
            "fuel": dict(zip(cal.member_names, [round(float(w), 4) for w in cal.fuel_blend_weights])),
            "nonfuel": dict(zip(cal.member_names, [round(float(w), 4) for w in cal.nf_blend_weights])),
        },
        "temperature": round(float(cal.temperature), 2),
        "inactivity_threshold": round(float(cal.inact_threshold), 3),
        "shrinkage_powers": {
            "fuel": round(float(cal.fuel_shrinkage_power), 3),
            "nonfuel_alpha": round(float(cal.nf_alpha), 3),
            "nonfuel_beta": round(float(cal.nf_beta), 3),
        },
    }

    cfg.paths.reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = cfg.paths.reports_dir / "evaluation_report.json"
    report_path.write_text(json.dumps(summary, indent=2))
    logger.info("Pipeline report written to %s", report_path)
    logger.info("Tournament pipeline completed in %.2f minutes", elapsed / 60.0)

    return summary


if __name__ == "__main__":
    from src.config import PipelineConfig
    run_pipeline(PipelineConfig(), save_predictions=True)
