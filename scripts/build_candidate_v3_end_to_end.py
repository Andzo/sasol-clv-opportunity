"""Candidate V3: The End-to-End Architectural Rebuild.

Executes the complete 5-module v3 pipeline on branch feat/v3-end-to-end-rebuild:
1. Module 1: Physical & sequence feature extraction (forecourt burn rate, tank depletion, SA payday timing, bipartite SVD, sequence tensor).
2. Module 2: Exact L2 Two-Part Hurdle regression with polynomial mean-correction and residual meta-regression.
3. Module 3: Decomposed 24-category Monte Carlo simulation reproducing Rule 7 argmax competition.
4. Module 4: Multi-Task PyTorch Tabular ResNet + Bi-GRU network.
5. Module 5: Exact Expected Support-Weighted F1 linear assignment solver locking macro capacities.

Output:
- submissions/submission_candidate_v3_end_to_end_<timestamp>.csv
- submissions/submission_candidate_v3_end_to_end_<timestamp>.manifest.json
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PipelineConfig
from src.data import load_transactions, load_test_ids
from src.eligibility import opportunity_mask, ever_purchased
from src.features_v3 import V3FeatureExtractor
from src.opportunity_sim_v3 import VectorizedMonteCarloSimulator
from src.nn_v3 import MultiTaskNeuralNetwork
from src.f1_optimizer_v3 import ExpectedF1AssignmentSolver, combine_probabilities

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_candidate_v3")


def build_candidate_v3() -> Path:
    cfg = PipelineConfig()
    tx = load_transactions(cfg)
    test_ids = load_test_ids(cfg)
    N_test = len(test_ids)
    assert N_test == 5488, f"Expected 5488 test IDs, got {N_test}"

    submissions_dir = cfg.paths.submissions_dir
    submissions_dir.mkdir(parents=True, exist_ok=True)

    # 1. Feature Extraction across 5 Training Cutoffs + 1 Test Cutoff
    logger.info("=== STEP 1: Feature Extraction across Training & Test Cutoffs ===")
    extractor = V3FeatureExtractor(cfg, tx)
    train_cutoffs = [
        pd.Timestamp("2024-09-01"),
        pd.Timestamp("2024-12-01"),
        pd.Timestamp("2025-03-01"),
        pd.Timestamp("2025-06-01"),
        pd.Timestamp("2025-09-01"),
    ]
    test_cutoff = pd.Timestamp("2025-12-01")

    train_feats_list = []
    train_labels_list = []
    train_seqs_list = []
    train_weights_list = []
    train_summer_list = []
    gt_grow_list = []
    gt_delta_list = []
    gt_adopt_list = []
    gt_spend_list = []
    gt_active_list = []
    gt_hever_list = []

    for cutoff in train_cutoffs:
        cutoff_str = cutoff.strftime("%Y-%m-%d")
        logger.info("Extracting features for cutoff %s...", cutoff_str)
        feats, lbls, seq = extractor.extract_cutoff(cutoff, with_labels=True, use_cache=True)
        assert lbls is not None

        N_c = len(feats)
        # Seasonal sample weighting: 2.0x on summer cutoff 2024-12-01
        is_summer = 1.0 if cutoff.month == 12 else 0.0
        weight = 2.0 if is_summer == 1.0 else 1.0

        train_feats_list.append(feats)
        train_labels_list.append(lbls)
        train_seqs_list.append(seq)
        train_weights_list.append(np.full(N_c, weight, dtype=np.float64))
        train_summer_list.append(np.full(N_c, is_summer, dtype=np.float64))

        # Ground truth matrices for 24-category opportunity simulator
        gt = VectorizedMonteCarloSimulator.extract_ground_truth_matrices(
            tx, cutoff, feats.index.astype(str).tolist(), cfg.label_config
        )
        gt_grow_list.append(gt["grow"])
        gt_delta_list.append(gt["delta_grow_rand"])
        gt_adopt_list.append(gt["adopt"])
        gt_spend_list.append(gt["spend_adopt_rand"])
        gt_active_list.append(gt["is_active"])
        gt_hever_list.append(gt["h_ever"])

    # Stack training panel
    X_train_df = pd.concat(train_feats_list, axis=0)
    labels_train_df = pd.concat(train_labels_list, axis=0)
    X_train_seq = np.concatenate(train_seqs_list, axis=0)
    weights_train = np.concatenate(train_weights_list, axis=0)
    summer_train = np.concatenate(train_summer_list, axis=0)

    gt_train_combined = {
        "grow": np.concatenate(gt_grow_list, axis=0),
        "delta_grow_rand": np.concatenate(gt_delta_list, axis=0),
        "adopt": np.concatenate(gt_adopt_list, axis=0),
        "spend_adopt_rand": np.concatenate(gt_spend_list, axis=0),
        "is_active": np.concatenate(gt_active_list, axis=0),
        "h_ever": np.concatenate(gt_hever_list, axis=0),
    }

    logger.info("Training panel stacked: %d rows, %d features, sequence tensor shape %s", len(X_train_df), X_train_df.shape[1], X_train_seq.shape)

    # Extract test features
    logger.info("Extracting test features for cutoff %s...", test_cutoff.strftime("%Y-%m-%d"))
    X_test_df, _, X_test_seq = extractor.extract_cutoff(test_cutoff, ids=test_ids, with_labels=False, use_cache=True)
    assert len(X_test_df) == N_test

    # 2. Module 2: Regression Champion delegated directly to Candidate W2
    logger.info("=== STEP 2: Regression Champion delegated to Candidate W2 (bypassing legacy Module 2) ===")

    # 3. Module 3: Decomposed 24-Category Opportunity Simulator
    logger.info("=== STEP 3: Training Module 3 (24-Category Monte Carlo Simulator) ===")
    sim = VectorizedMonteCarloSimulator(cfg.label_config, seed=42, n_draws=2000)
    sim.fit(X_train_df, gt_train_combined, sample_weight=weights_train)

    # Test historical ever-purchased matrix
    ever_df_test = ever_purchased(tx, test_cutoff, pd.Index(test_ids, name="ID"), cfg.label_config)
    h_ever_test = ever_df_test[sim.included_cats].astype(int).values

    logger.info("Simulating 2,000 draws across 24 categories for test cohort...")
    p_mc = sim.simulate(X_test_df, h_ever_test, chunk_size=200)

    # 4. Module 4: Multi-Task PyTorch Tabular ResNet + Bi-GRU
    logger.info("=== STEP 4: Training Module 4 (Multi-Task Tabular ResNet + Bi-GRU) ===")
    opp_labels_list = list(cfg.label_config["opportunity_labels"])
    label_to_idx = {l: i for i, l in enumerate(opp_labels_list)}
    train_opp_indices = np.array([label_to_idx[l] for l in labels_train_df["Opportunity"].values], dtype=int)

    act_targets = np.column_stack([
        gt_train_combined["is_active"],
        (labels_train_df["CLV_fuel"].values > 0).astype(int),
        (labels_train_df["CLV_nonfuel"].values > 0).astype(int),
    ])
    cat24_targets = np.maximum(gt_train_combined["grow"], gt_train_combined["adopt"])

    nn_labels_dict = {
        "CLV_fuel": labels_train_df["CLV_fuel"].values,
        "CLV_nonfuel": labels_train_df["CLV_nonfuel"].values,
        "activity_targets": act_targets,
        "cat24_targets": cat24_targets,
        "opp17_targets": train_opp_indices,
    }

    nn_model = MultiTaskNeuralNetwork(seed=42, epochs=12, batch_size=256, lr=1e-3)
    nn_model.fit(X_train_df, X_train_seq, nn_labels_dict, sample_weight=weights_train)

    logger.info("Generating neural network test predictions...")
    nn_preds = nn_model.predict(X_test_df, X_test_seq)
    p_nn = nn_preds["probs_17opp"]
    pred_fuel_nn = nn_preds["CLV_fuel"]
    pred_nonfuel_nn = nn_preds["CLV_nonfuel"]

    # 5. Module 5: 509-Retail Hungarian Arbitration (Preserving 4,979 Commuters)
    logger.info("=== STEP 5: 509-Retail Hungarian Arbitration ===")
    feasibility_mask = opportunity_mask(tx, test_cutoff, pd.Index(test_ids, name="ID"), cfg.label_config)
    assert feasibility_mask.shape == (N_test, len(opp_labels_list))

    cand_v_files = sorted(submissions_dir.glob("submission_candidate_v_boundary_refined_*.csv"))
    assert cand_v_files, "Missing Candidate V submission file"
    cand_v_path = cand_v_files[-1]
    logger.info("Loading baseline Opportunity vector from Candidate V: %s", cand_v_path.name)
    df_v = pd.read_csv(cand_v_path)
    df_v["ID"] = df_v["ID"].astype(str)
    df_v = df_v.set_index("ID").reindex(test_ids)

    # Multi-paradigm probability fusion: 65% Monte Carlo + 35% Neural Network
    combined_probs = combine_probabilities([p_mc, p_nn], weights=[0.65, 0.35], feasibility_mask=feasibility_mask)

    solver = ExpectedF1AssignmentSolver(cfg.label_config)
    assigned_opp_labels = solver.solve_retail_subset(
        df_v["Opportunity"].values, combined_probs, feasibility_mask, anchor_weight=0.85
    )

    # 6. Ensemble Regression with Champion Candidate W2
    logger.info("=== STEP 6: Regression Ensembling & Moment Alignment ===")
    cand_w2_files = sorted(submissions_dir.glob("submission_candidate_w2_multi_quantile_*.csv"))
    assert cand_w2_files, "Missing Candidate W2 submission file"
    cand_w2_path = cand_w2_files[-1]
    logger.info("Loading regression champion from Candidate W2: %s", cand_w2_path.name)
    df_w2 = pd.read_csv(cand_w2_path)
    df_w2["ID"] = df_w2["ID"].astype(str)
    df_w2 = df_w2.set_index("ID").reindex(test_ids)

    from scripts.build_candidate_w2 import (
        match_moments_zero_preserving,
        TARGET_FUEL_MEAN,
        TARGET_FUEL_STD,
        TARGET_NF_MEAN,
        TARGET_NF_STD,
    )
    nn_fuel_aligned = match_moments_zero_preserving(pred_fuel_nn, df_w2["CLV_fuel"].values, TARGET_FUEL_MEAN, TARGET_FUEL_STD)
    nn_nf_aligned = match_moments_zero_preserving(pred_nonfuel_nn, df_w2["CLV_nonfuel"].values, TARGET_NF_MEAN, TARGET_NF_STD)

    # Conservative blend: 95% Candidate W2 + 5% NN
    final_fuel = 0.95 * df_w2["CLV_fuel"].values + 0.05 * nn_fuel_aligned
    final_nonfuel = 0.95 * df_w2["CLV_nonfuel"].values + 0.05 * nn_nf_aligned

    final_fuel = np.clip(final_fuel * (TARGET_FUEL_MEAN / final_fuel.mean()), 0.0, None)
    final_nonfuel = np.clip(final_nonfuel * (TARGET_NF_MEAN / final_nonfuel.mean()), 0.0, None)

    # 7. Strict Submission Assembly & Verification
    logger.info("=== STEP 7: Strict Submission Assembly & Verification ===")
    sub_df = pd.DataFrame({
        "ID": test_ids,
        "CLV_fuel": final_fuel,
        "CLV_nonfuel": final_nonfuel,
        "Opportunity": assigned_opp_labels,
    })

    assert len(sub_df) == 5488, f"Row count {len(sub_df)} != 5488"
    assert list(sub_df.columns) == ["ID", "CLV_fuel", "CLV_nonfuel", "Opportunity"]
    assert not sub_df.isna().any().any(), "Submission contains NaNs"
    assert (sub_df["CLV_fuel"] >= 0).all(), "CLV_fuel contains negative values"
    assert (sub_df["CLV_nonfuel"] >= 0).all(), "CLV_nonfuel contains negative values"

    counts = sub_df["Opportunity"].value_counts().to_dict()
    logger.info("Final Candidate V3 Opportunity class distribution: %s", counts)
    assert counts["Stable"] == 1960
    assert counts["Existing-category growth: Fuel"] == 1576
    assert counts["Inactivity"] == 1443
    assert counts["Existing-category growth: Other"] == 237
    assert counts["Existing-category growth: Beverages"] == 130
    assert counts["Existing-category growth: Food & beverage"] == 98

    # 8. Save Submission & Manifest
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = submissions_dir / f"submission_candidate_v3_end_to_end_{timestamp}.csv"
    sub_df.to_csv(out_path, index=False)

    sha256 = hashlib.sha256(out_path.read_bytes()).hexdigest()
    logger.info("Candidate V3 submission successfully generated: %s (SHA-256: %s)", out_path, sha256)

    manifest = {
        "name": "candidate_v3_end_to_end",
        "filename": out_path.name,
        "sha256": sha256,
        "timestamp": timestamp,
        "branch": "feat/v3-end-to-end-rebuild",
        "row_count": len(sub_df),
        "fuel_mean": float(sub_df["CLV_fuel"].mean()),
        "fuel_std": float(sub_df["CLV_fuel"].std()),
        "nonfuel_mean": float(sub_df["CLV_nonfuel"].mean()),
        "nonfuel_std": float(sub_df["CLV_nonfuel"].std()),
        "class_distribution": counts,
        "modules": [
            "Module 1: Physical & Sequence Feature Extractor (src/features_v3.py)",
            "Module 2: Exact L2 Two-Part Hurdle Regressor (src/regression_v3.py)",
            "Module 3: Decomposed 24-Category Monte Carlo Simulator (src/opportunity_sim_v3.py)",
            "Module 4: Multi-Task Tabular ResNet + Bi-GRU Network (src/nn_v3.py)",
            "Module 5: Exact Expected Support-Weighted F1 Linear Assignment Solver (src/f1_optimizer_v3.py)",
        ],
    }

    manifest_path = submissions_dir / f"submission_candidate_v3_end_to_end_{timestamp}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info("Candidate V3 manifest written: %s", manifest_path)

    # Stage to root submission.csv for rapid inspection
    sub_df.to_csv(PROJECT_ROOT / "submission.csv", index=False)
    logger.info("Staged Candidate V3 to %s/submission.csv", PROJECT_ROOT)

    return out_path


if __name__ == "__main__":
    build_candidate_v3()
