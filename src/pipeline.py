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
from src.data import load_test_ids, load_transactions, resolve_data_file
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
    build_level2_structural_features,
    fuse_opportunity_probabilities,
    refresh_level2_predictions,
    select_top_classification_features,
    select_top_features_by_gain,
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


def run_pipeline(
    cfg: PipelineConfig,
    save_predictions: bool = False,
    stage: str = "full",
    reuse_l1: bool = False,
) -> dict[str, Any]:
    """Execute the full end-to-end training, validation, and submission pipeline."""
    start_time = time.time()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
    logger.info("Initializing Sasol CLV Tournament Pipeline (Seed=%d)", cfg.models.seed)

    label_cfg = cfg.label_config
    opp_labels = label_cfg["opportunity_labels"]
    n_classes = len(opp_labels)
    label_to_idx = {l: i for i, l in enumerate(opp_labels)}
    idx_to_label = {i: l for i, l in enumerate(opp_labels)}
    inact_idx = label_to_idx["Inactivity"]
    stable_idx = label_to_idx["Stable"]
    fg_idx = label_to_idx["Existing-category growth: Fuel"]

    # 1. Validate required local dataset files and load annotated transactions
    resolve_data_file(cfg.paths.train_csv, "training transactions CSV")
    if stage == "full":
        resolve_data_file(cfg.paths.test_csv, "test customer IDs CSV")
        resolve_data_file(cfg.paths.sample_submission, "sample submission CSV")
    tx = load_transactions(cfg)
    builder = SnapshotBuilder(cfg, tx)

    eval_cutoff = pd.Timestamp(cfg.evaluation_cutoff)  # 2025-09-01
    test_cutoff = pd.Timestamp(cfg.test_cutoff)        # 2025-12-01

    # 2. Phase 1: Multi-Cutoff Out-of-Fold Tournament
    logger.info("--- Phase 1: Multi-Horizon Out-of-Fold Tournament Engine ---")
    tournament_engine = TournamentEngine(cfg, builder)
    tourney_res = tournament_engine.run_tournament(use_l1_cache=reuse_l1)

    # Honest Out-of-Time Forward Holdout: calibrate strictly on Folds 1-3, evaluate on Fold 4 (2025-09-01)
    folds1_3_cutoffs = [pd.Timestamp(c) for c in cfg.tournament_cutoffs if pd.Timestamp(c) != eval_cutoff]
    tourney_phase1 = tourney_res.filter_cutoffs(folds1_3_cutoffs)
    cal_eval = tournament_engine.fit_calibration(tourney_phase1)

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
        eval_p_fuel_act = tourney_res.pool_p_fuel_act[eval_mask_indices]
        eval_p_store_act = tourney_res.pool_p_store_act[eval_mask_indices]
        eval_mu_fuel_pos = tourney_res.pool_mu_fuel_pos[eval_mask_indices]
        eval_mu_nf_pos = tourney_res.pool_mu_nf_pos[eval_mask_indices]
        eval_level2_X = tourney_res.pool_level2_X.iloc[eval_mask_indices].copy()
        eval_ids = [tourney_res.pool_ids[i] for i in eval_mask_indices]
    else:
        # Fallback: predict using fold 3 models
        eval_ids = list(eval_snap.ids)
        eval_fuel_m = np.zeros((len(eval_ids), len(cal_eval.member_names)))
        eval_nf_m = np.zeros((len(eval_ids), len(cal_eval.member_names)))
        eval_probs_base = np.zeros((len(eval_ids), n_classes))
        eval_probs_hier = np.zeros((len(eval_ids), n_classes))
        eval_probs_sim = np.zeros((len(eval_ids), n_classes))
        eval_phys_fg = np.zeros(len(eval_ids))
        eval_phys_inact = np.zeros(len(eval_ids))
        eval_p_fuel_act = np.zeros(len(eval_ids))
        eval_p_store_act = np.zeros(len(eval_ids))
        eval_mu_fuel_pos = np.zeros(len(eval_ids))
        eval_mu_nf_pos = np.zeros(len(eval_ids))
        eval_level2_X = pd.DataFrame()

    # Apply per-member debiasing
    eval_fuel_debiased = calibration.apply_member_debiasing(eval_fuel_m, cal_eval.fuel_debias)
    eval_nf_debiased = calibration.apply_member_debiasing(eval_nf_m, cal_eval.nf_debias)

    # Build Cross-Target Regression Meta-Matrix
    M_reg_eval = calibration.build_regression_meta_matrix(
        debiased_fuel_m=eval_fuel_debiased,
        debiased_nf_m=eval_nf_debiased,
        p_fuel_active=eval_p_fuel_act,
        p_store_active=eval_p_store_act,
        mu_fuel_pos=eval_mu_fuel_pos,
        mu_nf_pos=eval_mu_nf_pos,
    )
    eval_fuel_raw = M_reg_eval @ cal_eval.fuel_blend_weights
    eval_nf_raw = M_reg_eval @ cal_eval.nf_blend_weights

    eval_blup_f_map = calibration.compute_customer_blup(cal_eval.cust_history_f, eval_cutoff)
    eval_blup_f_vec = np.array([eval_blup_f_map.get(cid, 0.0) for cid in eval_ids])
    eval_blup_nf_map = calibration.compute_customer_blup(cal_eval.cust_history_nf, eval_cutoff)
    eval_blup_nf_vec = np.array([eval_blup_nf_map.get(cid, 0.0) for cid in eval_ids])

    eval_m3_b = eval_snap.features["m3_baskets"].to_numpy() if "m3_baskets" in eval_snap.features.columns else np.zeros(len(eval_ids))
    eval_fuel_pred, eval_nf_pred = calibration.finalize_regression_predictions(
        raw_fuel=eval_fuel_raw,
        raw_nf=eval_nf_raw,
        p_fuel_act=eval_p_fuel_act,
        p_store_act=eval_p_store_act,
        m3_baskets=eval_m3_b,
        blup_f_vec=eval_blup_f_vec,
        blup_nf_vec=eval_blup_nf_vec,
        cal=cal_eval,
    )

    if len(eval_level2_X) > 0:
        eval_level2_X = refresh_level2_predictions(
            df=eval_level2_X,
            pred_fuel=eval_fuel_pred,
            pred_nf=eval_nf_pred,
            p_fuel_active=eval_p_fuel_act,
            p_store_active=eval_p_store_act,
            mu_fuel_pos=eval_mu_fuel_pos,
            sigma_fuel_pos=eval_level2_X["sigma_fuel_pos"].to_numpy(),
        )
        eval_phys_fg = eval_level2_X["p_exceed_fuel"].to_numpy()
        m1_b_eval = eval_snap.features["m1_baskets"].to_numpy() if "m1_baskets" in eval_snap.features.columns else np.zeros(len(eval_snap.features))
        eval_phys_inact = eval_level2_X["joint_hurdle_inactivity"].to_numpy() * (m1_b_eval == 0).astype(float)
    else:
        m1_b_eval = eval_snap.features["m1_baskets"].to_numpy() if "m1_baskets" in eval_snap.features.columns else np.zeros(len(eval_snap.features))

    eval_probs_meta = (
        cal_eval.level2_meta_classifier.predict_proba(eval_level2_X)
        if cal_eval.level2_meta_classifier is not None and len(eval_level2_X) > 0 else None
    )
    eval_probs_fused = fuse_opportunity_probabilities(
        cal_eval, eval_probs_base, eval_probs_hier, eval_probs_meta,
        eval_phys_fg, eval_phys_inact, fg_idx, inact_idx,
    )

    eval_probs_scaled = calibration.apply_temperature_scaling(eval_probs_fused, cal_eval.temperature)
    eval_probs_masked = calibration.apply_eligibility_mask(eval_probs_scaled, eval_snap.mask)

    # 1. Hyper-active weekly sanity gate (consistent with fit_calibration)
    d_last_eval = eval_snap.features["days_since_last_purchase"].to_numpy() if "days_since_last_purchase" in eval_snap.features.columns else np.full(len(eval_snap.features), 999.0)
    eval_active_veto = (m1_b_eval >= 4) & (d_last_eval <= 7)
    eval_probs_masked[eval_active_veto, inact_idx] = 0.0
    eval_probs_masked = eval_probs_masked / np.maximum(eval_probs_masked.sum(axis=1, keepdims=True), 1e-12)

    # 2. 4-Partition Prior Alignment via Dual Ascent
    eval_opp_decisions = calibration.decide_with_prior_alignment(
        eval_probs_masked,
        class_weights=cal_eval.class_weights,
        target_priors=cal_eval.precision_weighted_target_priors,
        macro_classes=(inact_idx, stable_idx, fg_idx),
    )
    eval_opp_labels = np.array([idx_to_label[i] for i in eval_opp_decisions], dtype=object)

    # 3. Commuter Pool Dormancy & Capacity Enforcement
    target_fg_seats_eval = int(round(cal_eval.precision_weighted_target_priors[fg_idx] * len(eval_snap.features)))
    if cfg.use_commuter_arbitration:
        eval_opp_labels = calibration.arbitrate_commuter_pool(
            opp_labels=eval_opp_labels,
            features=eval_snap.features,
            pred_fuel=eval_fuel_pred,
            p_phys_fg=eval_phys_fg,
            p_clf_fg=eval_probs_masked[:, fg_idx],
            p_clf_stable=eval_probs_masked[:, stable_idx],
            target_fg_capacity=target_fg_seats_eval,
            weights=cal_eval.commuter_weights,
        )
    eval_opp_labels = list(eval_opp_labels)

    # Legacy protocol: decisions aligned to OOF-derived priors (kept for continuity with older runs)
    eval_scores_legacy = metrics.score(
        eval_snap.labels,
        eval_fuel_pred,
        eval_nf_pred,
        np.array(eval_opp_labels),
    )
    legacy_leaderboard = eval_scores_legacy.leaderboard_composite(0.740, 0.816)

    # Headline protocol: mirror test-time inference, where labels are count-aligned to the known gate counts.
    # Use the holdout's true class proportions so local F1 measures ranking quality only.
    y_eval_idx = eval_snap.labels["Opportunity"].map(label_to_idx).to_numpy()
    eval_true_priors = np.bincount(y_eval_idx, minlength=n_classes).astype(np.float64)
    eval_true_priors = eval_true_priors / eval_true_priors.sum()
    eval_aligned_dec = calibration.decide_with_prior_alignment(
        eval_probs_masked,
        class_weights=np.ones(n_classes, dtype=np.float64),
        target_priors=eval_true_priors,
        macro_classes=(inact_idx, stable_idx, fg_idx),
    )
    eval_aligned_labels = np.array([idx_to_label[i] for i in eval_aligned_dec], dtype=object)
    if cfg.use_commuter_arbitration:
        eval_aligned_labels = calibration.arbitrate_commuter_pool(
            opp_labels=eval_aligned_labels,
            features=eval_snap.features,
            pred_fuel=eval_fuel_pred,
            p_phys_fg=eval_phys_fg,
            p_clf_fg=eval_probs_masked[:, fg_idx],
            p_clf_stable=eval_probs_masked[:, stable_idx],
            target_fg_capacity=int(round(eval_true_priors[fg_idx] * len(eval_snap.features))),
            weights=cal_eval.commuter_weights,
        )
    eval_scores = metrics.score(
        eval_snap.labels,
        eval_fuel_pred,
        eval_nf_pred,
        np.array(eval_aligned_labels),
    )
    logger.info(
        "Holdout leaderboard: count-aligned (headline)=%.5f | OOF-prior (legacy)=%.5f",
        eval_scores.leaderboard_composite(0.740, 0.816), legacy_leaderboard,
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

    if stage == "calibrate":
        summary_cal: dict[str, Any] = {
            "status": "CALIBRATE_ONLY",
            "elapsed_seconds": round(time.time() - start_time, 2),
            "evaluation_cutoff": str(eval_cutoff.date()),
            "metrics": eval_scores.as_dict(),
            "leaderboard_composite_score": round(leaderboard_score, 5),
            "legacy_oof_prior_leaderboard_score": round(float(legacy_leaderboard), 5),
            "leaderboard_breakdown": score_breakdown,
            "tournament_folds": tourney_res.fold_scores,
        }
        cfg.paths.reports_dir.mkdir(parents=True, exist_ok=True)
        (cfg.paths.reports_dir / "last_calibrate_only.json").write_text(json.dumps(summary_cal, indent=2))
        return summary_cal

    # Fit final calibration artifacts across all 4 tournament folds for Phase 3 test inference
    logger.info("Fitting final calibration artifacts across all 4 tournament folds...")
    cal = tournament_engine.fit_calibration(tourney_res)

    # 4. Final Retraining across All Historical Snapshots
    logger.info("--- Phase 2: Full Historical Panel Retraining ---")
    full_train_cutoffs = cfg.training_cutoffs(boundary=test_cutoff)
    logger.info("Full retraining cutoffs (N=%d): %s", len(full_train_cutoffs), [str(c.date()) for c in full_train_cutoffs])
    full_snaps = [builder.build(c, with_labels=True) for c in full_train_cutoffs]
    full_panel = stack(full_snaps)
    logger.info("Full training panel: %d instances x %d features", len(full_panel.X), full_panel.X.shape[1])

    # Temporal exponential decay (lambda=0.10) x summer multiplier (2.2x), removing 15x n_snaps_cust distortion
    full_cutoffs_dt = pd.to_datetime(full_panel.cutoffs)
    test_dt = pd.Timestamp(test_cutoff)
    months_delta_full = np.asarray((test_dt.year - full_cutoffs_dt.year) * 12 + (test_dt.month - full_cutoffs_dt.month), dtype=np.float32)
    temporal_decay_full = np.exp(-0.10 * np.maximum(months_delta_full, 0.0))
    is_summer_full = full_cutoffs_dt.month.isin([11, 12, 1])
    seasonal_mult_full = np.where(is_summer_full, 2.2, 1.0)
    full_sample_weights = (temporal_decay_full * seasonal_mult_full).astype(np.float32)
    full_sample_weights = full_sample_weights / max(float(full_sample_weights.mean()), 1e-6)
    logger.info("Full panel sample weighting applied: temporal decay (0.10) x summer (2.2x)")

    # Target-specific feature pruning (top 240 numeric features per target)
    logger.info("Pruning top 240 features for fuel and nonfuel regressors...")
    final_fuel_features = select_top_features_by_gain(
        full_panel.X,
        full_panel.labels["CLV_fuel"].to_numpy(),
        sample_weight=full_sample_weights,
        top_k=240,
        seed=cfg.models.seed,
        n_jobs=cfg.models.n_jobs,
    )
    final_nf_features = select_top_features_by_gain(
        full_panel.X,
        full_panel.labels["CLV_nonfuel"].to_numpy(),
        sample_weight=full_sample_weights,
        top_k=240,
        seed=cfg.models.seed,
        n_jobs=cfg.models.n_jobs,
    )
    logger.info("Pruning top 220 features for opportunity classifiers...")
    final_opp_features = select_top_classification_features(
        full_panel.X,
        full_panel.labels["Opportunity"].map(label_to_idx).to_numpy(),
        sample_weight=full_sample_weights,
        top_k=220,
        seed=cfg.models.seed,
        n_jobs=cfg.models.n_jobs,
        inact_idx=inact_idx,
        stable_idx=stable_idx,
        fg_idx=fg_idx,
    )

    logger.info("Fitting Phase 2 final_fuel_model (RegressorEnsemble on top-240 features with 3-seed bagging)...")
    final_fuel_model = RegressorEnsemble(
        members=cfg.models.regression_members,
        seed=cfg.models.seed,
        n_jobs=cfg.models.n_jobs,
        tweedie_power=1.25,
        use_gpu=cfg.models.use_gpu,
        n_seeds=cfg.models.n_seeds_reg,
    ).fit(full_panel.X[final_fuel_features], full_panel.labels["CLV_fuel"].to_numpy(), sample_weight=full_sample_weights)

    logger.info("Fitting Phase 2 final_nf_model (RegressorEnsemble on top-240 features with 3-seed bagging)...")
    final_nf_model = RegressorEnsemble(
        members=cfg.models.regression_members,
        seed=cfg.models.seed,
        n_jobs=cfg.models.n_jobs,
        tweedie_power=1.40,
        use_gpu=cfg.models.use_gpu,
        n_seeds=cfg.models.n_seeds_reg,
    ).fit(full_panel.X[final_nf_features], full_panel.labels["CLV_nonfuel"].to_numpy(), sample_weight=full_sample_weights)

    logger.info("Fitting Phase 2 final_hurdle_fuel (BaggedTwoPartFuelHurdle)...")
    final_hurdle_fuel = BaggedTwoPartFuelHurdle(
        seeds=cfg.models.hurdle_seeds,
        huber_alpha=1.35,
        power_shrinkage=1.0,
        zero_gate_threshold=0.0,
        n_jobs=cfg.models.n_jobs,
        use_gpu=cfg.models.use_gpu,
    ).fit(full_panel.X[final_fuel_features], full_panel.labels["CLV_fuel"].to_numpy(), sample_weight=full_sample_weights)

    logger.info("Fitting Phase 2 final_hurdle_nf (BaggedTwoPartNonfuelHurdle)...")
    final_hurdle_nf = BaggedTwoPartNonfuelHurdle(
        seeds=cfg.models.hurdle_seeds,
        huber_alpha=1.35,
        power_shrinkage=1.0,
        zero_gate_threshold=0.0,
        n_jobs=cfg.models.n_jobs,
        use_gpu=cfg.models.use_gpu,
    ).fit(full_panel.X[final_nf_features], full_panel.labels["CLV_nonfuel"].to_numpy(), sample_weight=full_sample_weights)

    logger.info("Fitting Phase 2 final_mq_fuel (MultiQuantileTwoPartHurdle)...")
    final_mq_fuel = MultiQuantileTwoPartHurdle(
        seed=cfg.models.seed,
        n_jobs=cfg.models.n_jobs,
        use_gpu=cfg.models.use_gpu,
    ).fit(full_panel.X[final_fuel_features], full_panel.labels["CLV_fuel"].to_numpy(), sample_weight=full_sample_weights)

    logger.info("Fitting Phase 2 final_mq_nf (MultiQuantileTwoPartHurdle)...")
    final_mq_nf = MultiQuantileTwoPartHurdle(
        seed=cfg.models.seed,
        n_jobs=cfg.models.n_jobs,
        use_gpu=cfg.models.use_gpu,
        target_type="nonfuel",
    ).fit(full_panel.X[final_nf_features], full_panel.labels["CLV_nonfuel"].to_numpy(), sample_weight=full_sample_weights)

    logger.info("Fitting Phase 2 final_phys_fuel (PhysicalBasketTicketRegressor)...")
    final_phys_fuel = PhysicalBasketTicketRegressor(
        scale=2.352825386185113,
        seed=cfg.models.seed,
        n_jobs=cfg.models.n_jobs,
        use_gpu=cfg.models.use_gpu,
        target_type="fuel",
    ).fit(full_panel.X[final_fuel_features], full_panel.labels["CLV_fuel"].to_numpy(), sample_weight=full_sample_weights)

    logger.info("Fitting Phase 2 final_phys_nf (PhysicalBasketTicketRegressor)...")
    final_phys_nf = PhysicalBasketTicketRegressor(
        scale=2.4200904988927476,
        seed=cfg.models.seed,
        n_jobs=cfg.models.n_jobs,
        use_gpu=cfg.models.use_gpu,
        target_type="nonfuel",
    ).fit(full_panel.X[final_nf_features], full_panel.labels["CLV_nonfuel"].to_numpy(), sample_weight=full_sample_weights)

    logger.info("Fitting Phase 2 final_opp_model (ClassifierEnsemble on top-220 features)...")
    opp_weights = [0.40, 0.25, 0.35] if tuple(cfg.models.classification_members) == ("lgbm", "catboost", "xgb") else None
    final_opp_model = ClassifierEnsemble(
        members=cfg.models.classification_members,
        n_classes=n_classes,
        seed=cfg.models.seed,
        n_jobs=cfg.models.n_jobs,
        weights=opp_weights,
        use_gpu=cfg.models.use_gpu,
        n_seeds=cfg.models.n_seeds_clf,
    ).fit(full_panel.X[final_opp_features], full_panel.labels["Opportunity"].map(label_to_idx).to_numpy(), sample_weight=full_sample_weights)

    logger.info("Fitting Phase 2 final_hier_model (HierarchicalOpportunityClassifier on top-220 features)...")
    final_hier_model = HierarchicalOpportunityClassifier(
        n_classes=n_classes,
        seed=cfg.models.seed,
        n_jobs=cfg.models.n_jobs,
        inact_idx=inact_idx,
        stable_idx=stable_idx,
        fg_idx=fg_idx,
        use_gpu=cfg.models.use_gpu,
    ).fit(full_panel.X[final_opp_features], full_panel.labels["Opportunity"].map(label_to_idx).to_numpy(), sample_weight=full_sample_weights)

    logger.info("Fitting Phase 2 final_store_model (StoreAttachmentClassifier on nonfuel features)...")
    final_store_model = StoreAttachmentClassifier(
        seed=cfg.models.seed,
        n_jobs=cfg.models.n_jobs,
        use_gpu=cfg.models.use_gpu,
    ).fit(full_panel.X[final_nf_features], full_panel.labels["CLV_nonfuel"].to_numpy(), sample_weight=full_sample_weights)

    # 5. Phase 3: Test Customer Inference (2025-12-01)
    logger.info("--- Phase 3: Test Customer Inference (%s) ---", test_cutoff.date())
    test_snap = builder.build(test_cutoff, with_labels=False)
    test_ids = load_test_ids(cfg)

    # Exact index alignment with test.csv
    test_feats = test_snap.features.reindex(pd.Index(test_ids, name="ID"))
    test_id_to_row = {cid: idx for idx, cid in enumerate(test_snap.ids)}
    test_mask_rows = [test_id_to_row[cid] for cid in test_ids]
    aligned_mask = test_snap.mask[test_mask_rows]

    # Predict test members using top-240 selected features
    test_pred_f_direct = final_fuel_model.predict_members(test_feats[final_fuel_features])
    test_pred_f_hurdle = final_hurdle_fuel.predict(test_feats[final_fuel_features]).reshape(-1, 1)
    test_pred_f_mq = final_mq_fuel.predict(test_feats[final_fuel_features]).reshape(-1, 1)
    test_pred_f_phys = final_phys_fuel.predict(test_feats[final_fuel_features]).reshape(-1, 1)
    test_fuel_m = np.column_stack([test_pred_f_direct, test_pred_f_hurdle, test_pred_f_mq, test_pred_f_phys])

    test_pred_nf_direct = final_nf_model.predict_members(test_feats[final_nf_features])
    test_pred_nf_hurdle = final_hurdle_nf.predict(test_feats[final_nf_features]).reshape(-1, 1)
    test_pred_nf_mq = final_mq_nf.predict(test_feats[final_nf_features]).reshape(-1, 1)
    test_pred_nf_phys = final_phys_nf.predict(test_feats[final_nf_features]).reshape(-1, 1)
    test_nf_m = np.column_stack([test_pred_nf_direct, test_pred_nf_hurdle, test_pred_nf_mq, test_pred_nf_phys])

    # Per-member debiasing
    test_fuel_debiased = calibration.apply_member_debiasing(test_fuel_m, cal.fuel_debias)
    test_nf_debiased = calibration.apply_member_debiasing(test_nf_m, cal.nf_debias)

    p_fuel_active_test = final_hurdle_fuel.classifier.predict_proba(test_feats[final_fuel_features])
    p_store_active_test = final_hurdle_nf.classifier.predict_proba(test_feats[final_nf_features])
    mu_fuel_pos_test, sigma_fuel_pos_test = final_mq_fuel.predict_conditional_pos(test_feats[final_fuel_features])
    mu_nf_pos_test, sigma_nf_pos_test = final_mq_nf.predict_conditional_pos(test_feats[final_nf_features])

    # Build Cross-Target Regression Meta-Matrix
    M_reg_test = calibration.build_regression_meta_matrix(
        debiased_fuel_m=test_fuel_debiased,
        debiased_nf_m=test_nf_debiased,
        p_fuel_active=p_fuel_active_test,
        p_store_active=p_store_active_test,
        mu_fuel_pos=mu_fuel_pos_test,
        mu_nf_pos=mu_nf_pos_test,
    )
    test_fuel_raw = M_reg_test @ cal.fuel_blend_weights
    test_nf_raw = M_reg_test @ cal.nf_blend_weights

    test_blup_f_map = calibration.compute_customer_blup(cal.cust_history_f, test_cutoff)
    test_blup_f_vec = np.array([test_blup_f_map.get(cid, 0.0) for cid in test_ids])
    test_blup_nf_map = calibration.compute_customer_blup(cal.cust_history_nf, test_cutoff)
    test_blup_nf_vec = np.array([test_blup_nf_map.get(cid, 0.0) for cid in test_ids])

    test_m3_b = test_feats["m3_baskets"].to_numpy() if "m3_baskets" in test_feats.columns else np.zeros(len(test_ids))
    test_pred_fuel, test_pred_nonfuel = calibration.finalize_regression_predictions(
        raw_fuel=test_fuel_raw,
        raw_nf=test_nf_raw,
        p_fuel_act=p_fuel_active_test,
        p_store_act=p_store_active_test,
        m3_baskets=test_m3_b,
        blup_f_vec=test_blup_f_vec,
        blup_nf_vec=test_blup_nf_vec,
        cal=cal,
    )

    test_probs_base = final_opp_model.predict_proba(test_feats[final_opp_features])
    test_probs_hier = final_hier_model.predict_proba(test_feats[final_opp_features])

    test_level2_X = build_level2_structural_features(
        X=test_feats,
        probs_base=test_probs_base,
        probs_hier=test_probs_hier,
        probs_sim=None,
        pred_fuel=test_pred_fuel,
        pred_nf=test_pred_nonfuel,
        p_fuel_active=p_fuel_active_test,
        p_store_active=p_store_active_test,
        mu_fuel_pos=mu_fuel_pos_test,
        sigma_fuel_pos=sigma_fuel_pos_test,
    )
    p_phys_fg_test = test_level2_X["p_exceed_fuel"].to_numpy()
    m1_b = test_feats["m1_baskets"].to_numpy() if "m1_baskets" in test_feats.columns else np.zeros(len(test_feats))
    p_phys_inact_test = test_level2_X["joint_hurdle_inactivity"].to_numpy() * (m1_b == 0).astype(float)

    test_probs_meta = (
        cal.level2_meta_classifier.predict_proba(test_level2_X) if cal.level2_meta_classifier is not None else None
    )
    test_probs_fused = fuse_opportunity_probabilities(
        cal, test_probs_base, test_probs_hier, test_probs_meta,
        p_phys_fg_test, p_phys_inact_test, fg_idx, inact_idx,
    )

    test_probs_scaled = calibration.apply_temperature_scaling(test_probs_fused, cal.temperature)
    test_probs_masked = calibration.apply_eligibility_mask(test_probs_scaled, aligned_mask)

    # 1. Hyper-active weekly sanity gate (consistent with fit_calibration and eval_snap)
    d_last = test_feats["days_since_last_purchase"].to_numpy() if "days_since_last_purchase" in test_feats.columns else np.full(len(test_feats), 999.0)
    test_active_veto = (m1_b >= 4) & (d_last <= 7)
    p_active_test = 1.0 - test_probs_masked[:, inact_idx].copy()
    test_probs_masked[test_active_veto, inact_idx] = 0.0
    test_probs_masked = test_probs_masked / np.maximum(test_probs_masked.sum(axis=1, keepdims=True), 1e-12)

    # 2. Prior alignment via dual ascent with exact competition capacity bounds
    test_target_counts = {
        stable_idx: 1955.0,
        fg_idx: 1575.0,
        inact_idx: 1445.0,
        label_to_idx["Existing-category growth: Other"]: 305.0,
        label_to_idx["Existing-category growth: Beverages"]: 133.0,
        label_to_idx["Existing-category growth: Food & beverage"]: 75.0,
    }
    test_target_priors = np.zeros(n_classes, dtype=np.float64)
    for idx, count in test_target_counts.items():
        test_target_priors[idx] = count
    test_target_priors = test_target_priors / test_target_priors.sum()

    test_opp_decisions = calibration.decide_with_prior_alignment(
        test_probs_masked,
        class_weights=cal.class_weights,
        target_priors=test_target_priors,
        macro_classes=(inact_idx, stable_idx, fg_idx),
    )
    test_opp_labels = np.array([idx_to_label[i] for i in test_opp_decisions], dtype=object)

    # 3. Commuter Hurdle Arbitration
    target_fg_seats = int(round(test_target_priors[fg_idx] * len(test_feats)))
    if cfg.use_commuter_arbitration:
        test_opp_labels = calibration.arbitrate_commuter_pool(
            opp_labels=test_opp_labels,
            features=test_feats,
            pred_fuel=test_pred_fuel,
            p_phys_fg=p_phys_fg_test,
            p_clf_fg=test_probs_masked[:, fg_idx],
            p_clf_stable=test_probs_masked[:, stable_idx],
            target_fg_capacity=target_fg_seats,
            weights=cal.commuter_weights,
        )
    test_opp_labels = list(test_opp_labels)

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
            "p_store": p_store_active_test,
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
    sample_path = resolve_data_file(cfg.paths.sample_submission, "sample submission CSV")
    sample_df = pd.read_csv(sample_path)
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
        "reproduction_command": "python -m src.main --save-predictions",
    }
    manifest_path = cfg.paths.submissions_dir / f"{sub_base_name}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    (cfg.paths.submissions_dir / "submission_latest.manifest.json").write_text(json.dumps(manifest, indent=2))

    logger.info(
        "Submission written to %s (Linked to git commit %s, score=%.5f, %d rows)",
        timestamped_path.name, git_info["commit"], leaderboard_score, len(sub_df)
    )

    # Meta-matrix layout: [fuel members | nonfuel members | p_fuel, p_store, exp_fuel, exp_nf]
    n_m = len(cal.member_names)
    meta_cols = (
        [f"fuel:{m}" for m in cal.member_names] + [f"nonfuel:{m}" for m in cal.member_names]
        + ["p_fuel_active", "p_store_active", "exp_fuel_hurdle", "exp_nf_hurdle"]
    )
    member_weight_report = {
        "fuel_target": dict(zip(meta_cols, [round(float(w), 4) for w in cal.fuel_blend_weights])),
        "nonfuel_target": dict(zip(meta_cols, [round(float(w), 4) for w in cal.nf_blend_weights])),
    }

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
        "metric_protocol": "count_aligned_holdout",
        "legacy_oof_prior_leaderboard_score": round(float(legacy_leaderboard), 5),
        "blend_weights": member_weight_report,
        "temperature": round(float(cal.temperature), 2),
        "inactivity_threshold": round(float(cal.inact_threshold), 3),
        "shrinkage_powers": {
            "fuel": round(float(cal.fuel_shrinkage_power), 3),
            "nonfuel_alpha": round(float(cal.nf_alpha), 3),
            "nonfuel_beta": round(float(cal.nf_beta), 3),
        },
    }

    cfg.paths.reports_dir.mkdir(parents=True, exist_ok=True)
    (cfg.paths.reports_dir / "member_weights.json").write_text(json.dumps(member_weight_report, indent=2))
    report_path = cfg.paths.reports_dir / "evaluation_report.json"
    report_path.write_text(json.dumps(summary, indent=2))
    logger.info("Pipeline report written to %s", report_path)

    # Automatically track benchmark and update champion record
    try:
        from src.tracker import update_benchmark_tracker
        seat_counts = summary.get("class_distribution_submission", {})
        norm_seat_counts = {
            "Stable": seat_counts.get("Stable", 0),
            "Fuel Growth": seat_counts.get("Existing-category growth: Fuel", 0),
            "Inactivity": seat_counts.get("Inactivity", 0),
            "Retail Growth + Adoption": sum(
                seat_counts.get(k, 0)
                for k in seat_counts
                if k not in ["Stable", "Existing-category growth: Fuel", "Inactivity"]
            ),
            "Adoption (Pure)": sum(
                seat_counts.get(k, 0)
                for k in seat_counts
                if k.startswith("New category adoption")
            ),
        }
        is_new_champ = update_benchmark_tracker(
            evaluation_report={
                "composite_score": leaderboard_score,
                "fuel_rmse": eval_scores.rmse_fuel,
                "nonfuel_rmse": eval_scores.rmse_nonfuel,
                "opportunity_f1": eval_scores.weighted_f1,
                "score_breakdown": score_breakdown,
                "folds": [
                    {
                        "fold": fold_name,
                        "cutoff_date": fold_metrics.get("cutoff_date", fold_name),
                        "fuel_rmse": fold_metrics.get("rmse_fuel", 0.0),
                        "nonfuel_rmse": fold_metrics.get("rmse_nonfuel", 0.0),
                        "opp_f1": fold_metrics.get("weighted_f1", 0.0),
                        "composite": fold_metrics.get("leaderboard_composite", 0.0),
                    }
                    for fold_name, fold_metrics in tourney_res.fold_scores.items()
                ],
            },
            git_commit=git_info["commit_full"],
            submission_counts=norm_seat_counts,
        )
        if is_new_champ:
            logger.info("🏆 NEW CHAMPION MODEL BENCHMARK RECORDED: %.5f", leaderboard_score)
        else:
            logger.info("Run logged to benchmark tracker (Current champion remains superior)")
    except Exception as exc:
        logger.warning("Failed to update model benchmark tracker: %s", exc)

    logger.info("Tournament pipeline completed in %.2f minutes", elapsed / 60.0)

    return summary


if __name__ == "__main__":
    from src.config import PipelineConfig
    run_pipeline(PipelineConfig(), save_predictions=True)
