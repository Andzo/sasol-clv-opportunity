"""End-to-end pipeline runner for the Sasol CLV & Opportunity challenge.

Orchestrates:
1. Transaction loading and snapshot assembly with content-addressed caching.
2. Temporal cross-validation with an embargo to avoid label leakage.
3. Model training (LightGBM, CatBoost, XGBoost) for regression and classification.
4. Post-processing: feasibility masking, NNLS blend weights, coordinate-ascent class weights.
5. Out-of-time evaluation on holdout cutoff (2025-09-01).
6. Final model retraining, test inference, format validation, and submission generation.
"""
from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src import calibration, dataset, metrics, submission
from src.config import PipelineConfig
from src.data import load_test_ids, load_transactions
from src.dataset import SnapshotBuilder, stack
from src.features import FEATURE_VERSION
from src.models import ClassifierEnsemble, RegressorEnsemble, StoreAttachmentClassifier

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
    logger.info("Initializing Sasol CLV Pipeline (Seed=%d)", cfg.models.seed)

    label_cfg = cfg.label_config
    opp_labels = label_cfg["opportunity_labels"]
    n_classes = len(opp_labels)
    label_to_idx = {l: i for i, l in enumerate(opp_labels)}
    idx_to_label = {i: l for i, l in enumerate(opp_labels)}

    # 1. Load annotated transactions
    tx = load_transactions(cfg)
    builder = SnapshotBuilder(cfg, tx)

    # 2. Assemble snapshots
    logger.info("Building snapshots with strict embargoes...")
    eval_cutoff = pd.Timestamp(cfg.evaluation_cutoff)
    cal_cutoff = pd.Timestamp(cfg.calibration_cutoff)
    test_cutoff = pd.Timestamp(cfg.test_cutoff)

    # Training snapshots for validation model (must finish strictly before calibration/evaluation)
    train_cutoffs = cfg.training_cutoffs(boundary=cal_cutoff)
    logger.info("Training cutoffs for validation: %s", [str(c.date()) for c in train_cutoffs])
    train_snaps = [builder.build(c, with_labels=True) for c in train_cutoffs]
    cal_snap = builder.build(cal_cutoff, with_labels=True)
    eval_snap = builder.build(eval_cutoff, with_labels=True)

    # Stack training panel
    train_panel = stack(train_snaps)
    feature_cols = list(train_panel.X.columns)
    logger.info("Stacked training panel: %d instances x %d features", len(train_panel.X), len(feature_cols))

    # 3. Validation Model Training
    logger.info("--- Phase 1: Out-of-Time Model Validation ---")
    fuel_model = RegressorEnsemble(
        members=cfg.models.regression_members,
        seed=cfg.models.seed,
        n_jobs=cfg.models.n_jobs
    ).fit(train_panel.X, train_panel.labels["CLV_fuel"].to_numpy())

    nonfuel_model = RegressorEnsemble(
        members=cfg.models.regression_members,
        seed=cfg.models.seed,
        n_jobs=cfg.models.n_jobs
    ).fit(train_panel.X, train_panel.labels["CLV_nonfuel"].to_numpy())

    opp_weights = [0.40, 0.25, 0.35] if tuple(cfg.models.classification_members) == ("lgbm", "catboost", "xgb") else None
    opp_model = ClassifierEnsemble(
        members=cfg.models.classification_members,
        n_classes=n_classes,
        seed=cfg.models.seed,
        n_jobs=cfg.models.n_jobs,
        weights=opp_weights,
    ).fit(train_panel.X, train_panel.labels["Opportunity"].map(label_to_idx).to_numpy())

    logger.info("Training StoreAttachmentClassifier for nonfuel zero-inflation hurdle...")
    store_model = StoreAttachmentClassifier(
        seed=cfg.models.seed,
        n_jobs=cfg.models.n_jobs
    ).fit(train_panel.X, train_panel.labels["CLV_nonfuel"].to_numpy())

    # Fit blend weights on calibration fold
    cal_fuel_m = fuel_model.predict_members(cal_snap.features)
    fuel_blend_weights = calibration.fit_blend_weights(cal_fuel_m, cal_snap.labels["CLV_fuel"].to_numpy())
    fuel_model.set_weights(fuel_blend_weights)
    logger.info("Fuel blend weights: %s", np.round(fuel_blend_weights, 3))

    cal_nonfuel_m = nonfuel_model.predict_members(cal_snap.features)
    nonfuel_blend_weights = calibration.fit_blend_weights(cal_nonfuel_m, cal_snap.labels["CLV_nonfuel"].to_numpy())
    nonfuel_model.set_weights(nonfuel_blend_weights)
    logger.info("Nonfuel blend weights: %s", np.round(nonfuel_blend_weights, 3))

    # Fit temperature and class weights jointly on calibration fold with dominant class protection
    cal_probs = opp_model.predict_proba(cal_snap.features)
    cal_y = cal_snap.labels["Opportunity"].map(label_to_idx).to_numpy()
    dom_classes = [label_to_idx[l] for l in ("Inactivity", "Stable", "Existing-category growth: Fuel") if l in label_to_idx]
    cal_temp, class_weights = calibration.fit_temperature_and_class_weights(
        cal_probs, cal_y, mask=cal_snap.mask, dominant_classes=dom_classes, seed=cfg.models.seed
    )
    cal_probs_scaled = calibration.apply_temperature_scaling(cal_probs, cal_temp)
    cal_probs_masked = calibration.apply_eligibility_mask(cal_probs_scaled, cal_snap.mask)
    logger.info("Fitted temperature T=%.2f, class weights (max=%.2f, min=%.2f)", cal_temp, class_weights.max(), class_weights.min())

    # Fit soft active shrinkage (fuel) and dual soft hurdle shrinkage (nonfuel) on calibration fold
    inact_idx = label_to_idx["Inactivity"]
    p_active_cal = 1.0 - cal_probs_masked[:, inact_idx]
    p_store_cal = store_model.predict_proba(cal_snap.features)

    fuel_shrinkage_power = calibration.fit_shrinkage_power(
        cal_snap.labels["CLV_fuel"].to_numpy(), fuel_model.predict(cal_snap.features), p_active_cal
    )
    nonfuel_alpha, nonfuel_beta = calibration.fit_dual_shrinkage_powers(
        cal_snap.labels["CLV_nonfuel"].to_numpy(),
        nonfuel_model.predict(cal_snap.features),
        p_active_cal,
        p_store_cal,
        base_active_power=fuel_shrinkage_power,
    )
    # Fit dynamic inactivity gating threshold on calibration fold
    cal_f_shrunk = calibration.apply_shrinkage(fuel_model.predict(cal_snap.features), p_active_cal, fuel_shrinkage_power)
    cal_nf_shrunk = calibration.apply_dual_shrinkage(nonfuel_model.predict(cal_snap.features), p_active_cal, p_store_cal, nonfuel_alpha, nonfuel_beta)
    inact_threshold = calibration.fit_inactivity_threshold(
        cal_snap.labels["CLV_fuel"].to_numpy(), cal_f_shrunk,
        cal_snap.labels["CLV_nonfuel"].to_numpy(), cal_nf_shrunk,
        p_active_cal,
    )
    logger.info(
        "Fitted shrinkage powers: fuel (alpha=%.2f); nonfuel (alpha=%.2f, beta=%.2f); inact_thresh=%.3f",
        fuel_shrinkage_power,
        nonfuel_alpha,
        nonfuel_beta,
        inact_threshold,
    )

    # Score on Evaluation Snapshot (2025-09-01)
    eval_fuel_raw = fuel_model.predict(eval_snap.features)
    eval_nonfuel_raw = nonfuel_model.predict(eval_snap.features)
    eval_probs = opp_model.predict_proba(eval_snap.features)
    eval_probs_scaled = calibration.apply_temperature_scaling(eval_probs, cal_temp)
    eval_probs_masked = calibration.apply_eligibility_mask(eval_probs_scaled, eval_snap.mask)
    eval_opp_decisions = calibration.decide(eval_probs_masked, class_weights)
    eval_opp_labels = [idx_to_label[i] for i in eval_opp_decisions]

    p_active_eval = 1.0 - eval_probs_masked[:, inact_idx]
    p_store_eval = store_model.predict_proba(eval_snap.features)
    eval_fuel_pred = calibration.apply_shrinkage(eval_fuel_raw, p_active_eval, fuel_shrinkage_power)
    eval_fuel_pred = calibration.apply_inactivity_gate(eval_fuel_pred, p_active_eval, threshold=inact_threshold)
    eval_nonfuel_pred = calibration.apply_dual_shrinkage(
        eval_nonfuel_raw, p_active_eval, p_store_eval, nonfuel_alpha, nonfuel_beta
    )
    eval_nonfuel_pred = calibration.apply_inactivity_gate(eval_nonfuel_pred, p_active_eval, threshold=inact_threshold)

    eval_scores = metrics.score(
        eval_snap.labels,
        eval_fuel_pred,
        eval_nonfuel_pred,
        np.array(eval_opp_labels)
    )
    baseline_dict = metrics.baseline_scores(eval_snap.features, eval_snap.labels, label_cfg)
    x_max_fuel = baseline_dict["sample_submission_zero"]["rmse_fuel"]
    x_max_nonfuel = baseline_dict["sample_submission_zero"]["rmse_nonfuel"]
    composite_score = eval_scores.composite(x_max_fuel, x_max_nonfuel)

    logger.info("=== Out-of-Time Evaluation (%s) ===", eval_cutoff.date())
    logger.info("CLV_fuel RMSE:    %.4f (SampleSub baseline: %.4f)", eval_scores.rmse_fuel, x_max_fuel)
    logger.info("CLV_nonfuel RMSE: %.4f (SampleSub baseline: %.4f)", eval_scores.rmse_nonfuel, x_max_nonfuel)
    logger.info("Proxy Composite:  %.4f", composite_score)

    if save_predictions:
        eda_cache = cfg.paths.reports_dir / "eda" / "cache"
        eda_cache.mkdir(parents=True, exist_ok=True)
        eval_df = pd.DataFrame({
            "pred_fuel_raw": eval_fuel_raw,
            "pred_fuel_final": eval_fuel_pred,
            "pred_nonfuel_raw": eval_nonfuel_raw,
            "pred_nonfuel_final": eval_nonfuel_pred,
            "pred_opportunity": eval_opp_labels,
            "p_active": p_active_eval,
            "p_store": p_store_eval,
            "true_fuel": eval_snap.labels["CLV_fuel"].to_numpy(),
            "true_nonfuel": eval_snap.labels["CLV_nonfuel"].to_numpy(),
            "true_opportunity": eval_snap.labels["Opportunity"].to_numpy(),
        }, index=eval_snap.ids)
        for i, l in enumerate(opp_labels):
            eval_df[f"prob_{l}"] = eval_probs_masked[:, i]
        eval_df.to_parquet(eda_cache / "eval_predictions.parquet")
        logger.info("Saved evaluation predictions to %s", eda_cache / "eval_predictions.parquet")

    # 4. Final Retraining across All Historical Snapshots
    logger.info("--- Phase 2: Full Historical Retraining ---")
    full_train_cutoffs = cfg.training_cutoffs(boundary=test_cutoff)
    logger.info("Full retraining cutoffs: %s", [str(c.date()) for c in full_train_cutoffs])
    full_snaps = [builder.build(c, with_labels=True) for c in full_train_cutoffs]
    full_panel = stack(full_snaps)
    logger.info("Full panel size: %d instances", len(full_panel.X))

    final_fuel_model = RegressorEnsemble(
        members=cfg.models.regression_members,
        seed=cfg.models.seed,
        n_jobs=cfg.models.n_jobs
    ).fit(full_panel.X, full_panel.labels["CLV_fuel"].to_numpy())
    final_fuel_model.set_weights(fuel_blend_weights)

    final_nonfuel_model = RegressorEnsemble(
        members=cfg.models.regression_members,
        seed=cfg.models.seed,
        n_jobs=cfg.models.n_jobs
    ).fit(full_panel.X, full_panel.labels["CLV_nonfuel"].to_numpy())
    final_nonfuel_model.set_weights(nonfuel_blend_weights)

    final_opp_model = ClassifierEnsemble(
        members=cfg.models.classification_members,
        n_classes=n_classes,
        seed=cfg.models.seed,
        n_jobs=cfg.models.n_jobs,
        weights=opp_weights,
    ).fit(full_panel.X, full_panel.labels["Opportunity"].map(label_to_idx).to_numpy())

    logger.info("Retraining final StoreAttachmentClassifier on full historical panel...")
    final_store_model = StoreAttachmentClassifier(
        seed=cfg.models.seed,
        n_jobs=cfg.models.n_jobs
    ).fit(full_panel.X, full_panel.labels["CLV_nonfuel"].to_numpy())

    # Honest Out-of-Fold Multi-Cutoff Pooled Calibration (2025-06-01 + 2025-09-01)
    logger.info("Calibrating final models on honest out-of-fold pooled snapshots (%s + %s)...", cal_cutoff.date(), eval_cutoff.date())
    pool_snaps = [cal_snap, eval_snap]
    pool_masks = np.vstack([s.mask for s in pool_snaps])
    pool_y_fuel = np.concatenate([s.labels["CLV_fuel"].to_numpy() for s in pool_snaps])
    pool_y_nonfuel = np.concatenate([s.labels["CLV_nonfuel"].to_numpy() for s in pool_snaps])
    pool_y_opp = np.concatenate([s.labels["Opportunity"].map(label_to_idx).to_numpy() for s in pool_snaps])

    # Retain honest out-of-fold calibrated blend weights (fitting NNLS on in-sample pool leads to degenerate [1,0,0])
    logger.info("Retained out-of-fold fuel blend weights: %s", np.round(fuel_blend_weights, 3))
    logger.info("Retained out-of-fold nonfuel blend weights: %s", np.round(nonfuel_blend_weights, 3))

    # Pool honest out-of-fold predictions from Phase 1 models across cal_snap and eval_snap (10,199 instances)
    # (Phase 1 models were trained strictly before cal_cutoff, so their predictions on both cal and eval are 100% out-of-fold)
    oof_pool_probs = np.vstack([cal_probs, eval_probs])
    final_temp, final_class_weights = calibration.fit_temperature_and_class_weights(
        oof_pool_probs, pool_y_opp, mask=pool_masks, dominant_classes=dom_classes, seed=cfg.models.seed
    )
    oof_pool_probs_scaled = calibration.apply_temperature_scaling(oof_pool_probs, final_temp)
    oof_pool_probs_masked = calibration.apply_eligibility_mask(oof_pool_probs_scaled, pool_masks)
    logger.info(
        "Final retrained T=%.2f, class weights on %d honest out-of-fold pooled instances (max=%.2f, min=%.2f)",
        final_temp, len(pool_y_opp), final_class_weights.max(), final_class_weights.min()
    )

    # Shrinkage powers on honest out-of-fold pooled predictions
    pool_p_active = 1.0 - oof_pool_probs_masked[:, inact_idx]
    pool_p_store = np.concatenate([p_store_cal, p_store_eval])
    oof_pool_fuel = np.concatenate([fuel_model.predict(cal_snap.features), eval_fuel_raw])
    oof_pool_nonfuel = np.concatenate([nonfuel_model.predict(cal_snap.features), eval_nonfuel_raw])

    final_fuel_shrinkage = calibration.fit_shrinkage_power(
        pool_y_fuel, oof_pool_fuel, pool_p_active
    )
    final_nf_alpha, final_nf_beta = calibration.fit_dual_shrinkage_powers(
        pool_y_nonfuel, oof_pool_nonfuel, pool_p_active, pool_p_store,
        base_active_power=final_fuel_shrinkage
    )
    # Fit final dynamic inactivity gating threshold on honest out-of-fold pooled predictions
    pool_f_shrunk = calibration.apply_shrinkage(oof_pool_fuel, pool_p_active, final_fuel_shrinkage)
    pool_nf_shrunk = calibration.apply_dual_shrinkage(oof_pool_nonfuel, pool_p_active, pool_p_store, final_nf_alpha, final_nf_beta)
    final_inact_thresh = calibration.fit_inactivity_threshold(
        pool_y_fuel, pool_f_shrunk,
        pool_y_nonfuel, pool_nf_shrunk,
        pool_p_active,
    )
    logger.info(
        "Final retrained shrinkage powers: fuel (alpha=%.2f); nonfuel (alpha=%.2f, beta=%.2f); inact_thresh=%.3f",
        final_fuel_shrinkage, final_nf_alpha, final_nf_beta, final_inact_thresh
    )

    # 5. Test Inference (2025-12-01)
    logger.info("--- Phase 3: Test Customer Inference (%s) ---", test_cutoff.date())
    test_snap = builder.build(test_cutoff, with_labels=False)
    test_ids = load_test_ids(cfg)

    # Ensure exact index alignment with test.csv
    test_feats = test_snap.features.reindex(pd.Index(test_ids, name="ID"))
    # Reindex mask
    test_id_to_row = {cid: idx for idx, cid in enumerate(test_snap.ids)}
    test_mask_rows = [test_id_to_row[cid] for cid in test_ids]
    aligned_mask = test_snap.mask[test_mask_rows]

    test_pred_fuel_raw = final_fuel_model.predict(test_feats)
    test_pred_nonfuel_raw = final_nonfuel_model.predict(test_feats)
    test_probs = final_opp_model.predict_proba(test_feats)
    test_probs_scaled = calibration.apply_temperature_scaling(test_probs, final_temp)
    test_probs_masked = calibration.apply_eligibility_mask(test_probs_scaled, aligned_mask)
    test_opp_decisions = calibration.decide(test_probs_masked, final_class_weights)
    test_opp_labels = [idx_to_label[i] for i in test_opp_decisions]

    p_active_test = 1.0 - test_probs_masked[:, inact_idx]
    p_store_test = final_store_model.predict_proba(test_feats)
    test_pred_fuel = calibration.apply_shrinkage(test_pred_fuel_raw, p_active_test, final_fuel_shrinkage)
    test_pred_fuel = calibration.apply_inactivity_gate(test_pred_fuel, p_active_test, threshold=final_inact_thresh)
    test_pred_nonfuel = calibration.apply_dual_shrinkage(
        test_pred_nonfuel_raw, p_active_test, p_store_test, final_nf_alpha, final_nf_beta
    )
    test_pred_nonfuel = calibration.apply_inactivity_gate(test_pred_nonfuel, p_active_test, threshold=final_inact_thresh)

    if save_predictions:
        eda_cache = cfg.paths.reports_dir / "eda" / "cache"
        eda_cache.mkdir(parents=True, exist_ok=True)
        test_df = pd.DataFrame({
            "pred_fuel_raw": test_pred_fuel_raw,
            "pred_fuel_final": test_pred_fuel,
            "pred_nonfuel_raw": test_pred_nonfuel_raw,
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
        opportunity=test_opp_labels
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

    # Companion manifest for 100% reproducibility and lineage tracking
    manifest: dict[str, Any] = {
        "submission_file": timestamped_path.name,
        "created_at": pd.Timestamp.now().isoformat(),
        "git_commit": git_info["commit_full"],
        "git_commit_short": git_info["commit"],
        "git_branch": git_info["branch"],
        "git_dirty": git_info["dirty"],
        "feature_version": FEATURE_VERSION,
        "models": {
            "regression_members": list(cfg.models.regression_members),
            "classification_members": list(cfg.models.classification_members),
            "seed": cfg.models.seed,
        },
        "evaluation_metrics": eval_scores.as_dict(),
        "proxy_composite_score": round(composite_score, 4),
        "reproduction_command": f"git checkout {git_info['commit']} && python main.py --save-predictions",
    }
    manifest_path = cfg.paths.submissions_dir / f"{sub_base_name}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    (cfg.paths.submissions_dir / "submission_latest.manifest.json").write_text(json.dumps(manifest, indent=2))

    logger.info(
        "Submission written to %s (Linked to git commit %s, comp=%.4f, %d rows)",
        timestamped_path.name, git_info["commit"], composite_score, len(sub_df)
    )

    # Save evaluation summary
    elapsed = time.time() - start_time
    summary: dict[str, Any] = {
        "status": "SUCCESS",
        "elapsed_seconds": round(elapsed, 2),
        "evaluation_cutoff": str(eval_cutoff.date()),
        "metrics": eval_scores.as_dict(),
        "proxy_composite_score": round(composite_score, 4),
        "baselines": baseline_dict,
        "class_distribution_submission": sub_df["Opportunity"].value_counts().to_dict(),
        "fuel_mean": round(float(sub_df["CLV_fuel"].mean()), 4),
        "nonfuel_mean": round(float(sub_df["CLV_nonfuel"].mean()), 4),
        "blend_weights": {
            "fuel": dict(zip(cfg.models.regression_members, [round(float(w), 4) for w in fuel_blend_weights])),
            "nonfuel": dict(zip(cfg.models.regression_members, [round(float(w), 4) for w in nonfuel_blend_weights])),
        },
        "temperature": {
            "validation": round(float(cal_temp), 2),
            "final": round(float(final_temp), 2),
        },
        "inactivity_threshold": {
            "validation": round(float(inact_threshold), 3),
            "final": round(float(final_inact_thresh), 3),
        },
        "shrinkage_powers": {
            "fuel": round(float(final_fuel_shrinkage), 3),
            "nonfuel_alpha": round(float(final_nf_alpha), 3),
            "nonfuel_beta": round(float(final_nf_beta), 3),
            "validation_fuel": round(float(fuel_shrinkage_power), 3),
            "validation_nonfuel_alpha": round(float(nonfuel_alpha), 3),
            "validation_nonfuel_beta": round(float(nonfuel_beta), 3),
        },
    }

    cfg.paths.reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = cfg.paths.reports_dir / "evaluation_report.json"
    report_path.write_text(json.dumps(summary, indent=2))
    logger.info("Pipeline report written to %s", report_path)
    logger.info("Completed in %.2f minutes", elapsed / 60.0)

    return summary
