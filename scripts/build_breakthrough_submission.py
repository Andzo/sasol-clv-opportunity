"""Builds and validates the breakthrough submissions targeted at > 0.330 and > 0.34283.

Constructs:
1. Candidate A: Pure summer-calibrated super-ensemble (LGBM + CatBoost GPU + XGBoost GPU + Ridge).
2. Candidate B: Multi-checkpoint variance-reduced regression blend + summer-calibrated Opportunity.

Both candidates strictly follow competition constraints, use only provided datasets,
and format all 5,488 test customer IDs in exact order.
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_breakthrough")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def main() -> None:
    test_csv = PROJECT_ROOT / "test.csv"
    test_df = pd.read_csv(test_csv)
    test_ids_str = test_df["ID"].astype(str).tolist()
    n_test = len(test_ids_str)
    assert n_test == 5488, f"Expected 5488 test IDs, got {n_test}"

    # 1. Load cached test predictions from latest pipeline run
    test_parquet = PROJECT_ROOT / "reports" / "eda" / "cache" / "test_predictions.parquet"
    assert test_parquet.exists(), f"Missing test parquet: {test_parquet}"
    df_test = pd.read_parquet(test_parquet).reindex(test_ids_str)

    # 2. Load historical best submission (0.319 benchmark)
    past_best_csv = PROJECT_ROOT / "submissions" / "submission_20261012_173021_comp0.5253.csv"
    assert past_best_csv.exists(), f"Missing past best submission: {past_best_csv}"
    s_past = pd.read_csv(past_best_csv)
    s_past["ID"] = s_past["ID"].astype(str)
    s_past = s_past.set_index("ID").reindex(test_ids_str)

    # 3. Summer-Calibrated Opportunity Probabilities & Optimal Decoding
    prob_cols = [c for c in df_test.columns if c.startswith("prob_")]
    classes = [c.replace("prob_", "") for c in prob_cols]
    probs = df_test[prob_cols].to_numpy()

    inact_idx = classes.index("Inactivity")
    stable_idx = classes.index("Stable")
    fg_idx = classes.index("Existing-category growth: Fuel")
    other_idx = classes.index("Existing-category growth: Other")
    bev_idx = classes.index("Existing-category growth: Beverages")
    fb_idx = classes.index("Existing-category growth: Food & beverage")

    # Summer-calibrated class decision weights
    w = np.ones(len(classes), dtype=float)
    w[inact_idx] = 0.88
    w[fg_idx] = 1.32
    w[stable_idx] = 1.00
    w[other_idx] = 1.15
    w[bev_idx] = 1.65
    w[fb_idx] = 1.55

    pred_opp_summer = np.array(classes)[np.argmax(probs * w, axis=1)]

    # 4. Construct Candidate A: Pure Summer-Calibrated Super-Ensemble
    sub_a = pd.DataFrame({
        "ID": test_df["ID"],
        "CLV_fuel": df_test["pred_fuel_final"].to_numpy(),
        "CLV_nonfuel": df_test["pred_nonfuel_final"].to_numpy(),
        "Opportunity": pred_opp_summer,
    })

    # 5. Construct Candidate B: Variance-Reduced Regression Blend (50/50) + Summer-Calibrated Opportunity
    pred_fuel_blend = 0.50 * df_test["pred_fuel_final"].to_numpy() + 0.50 * s_past["CLV_fuel"].to_numpy()
    pred_nonfuel_blend = 0.50 * df_test["pred_nonfuel_final"].to_numpy() + 0.50 * s_past["CLV_nonfuel"].to_numpy()

    sub_b = pd.DataFrame({
        "ID": test_df["ID"],
        "CLV_fuel": pred_fuel_blend,
        "CLV_nonfuel": pred_nonfuel_blend,
        "Opportunity": pred_opp_summer,
    })

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub_dir = PROJECT_ROOT / "submissions"
    sub_dir.mkdir(parents=True, exist_ok=True)

    for name, df_sub in [("candidate_a_summer_calibrated", sub_a), ("candidate_b_super_blend", sub_b)]:
        # Validate format
        assert len(df_sub) == 5488, f"Invalid row count: {len(df_sub)}"
        assert list(df_sub.columns) == ["ID", "CLV_fuel", "CLV_nonfuel", "Opportunity"], f"Invalid columns: {df_sub.columns}"
        assert not df_sub.isna().any().any(), "Submission contains NaN values"
        assert (df_sub["CLV_fuel"] >= 0).all(), "CLV_fuel contains negative values"
        assert (df_sub["CLV_nonfuel"] >= 0).all(), "CLV_nonfuel contains negative values"

        out_path = sub_dir / f"submission_{name}_{timestamp}.csv"
        df_sub.to_csv(out_path, index=False)
        sha256 = hashlib.sha256(out_path.read_bytes()).hexdigest()

        # Manifest
        manifest = {
            "name": name,
            "filename": out_path.name,
            "sha256": sha256,
            "timestamp": timestamp,
            "row_count": len(df_sub),
            "clv_fuel_mean": float(df_sub["CLV_fuel"].mean()),
            "clv_nonfuel_mean": float(df_sub["CLV_nonfuel"].mean()),
            "class_distribution": df_sub["Opportunity"].value_counts().to_dict(),
            "target_zindi_threshold": "> 0.330 (Top 10) / > 0.34283 (Rank #1)",
        }
        manifest_path = sub_dir / f"submission_{name}_{timestamp}.manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))

        logger.info("Created %s (%d rows, SHA256: %s)", out_path.name, len(df_sub), sha256[:10])
        logger.info("Class distribution: %s", manifest["class_distribution"])

    # Update active submission.csv to Candidate B (the strongest variance-reduced model)
    sub_b.to_csv(PROJECT_ROOT / "submission.csv", index=False)
    sub_b.to_csv(sub_dir / "submission_latest.csv", index=False)
    logger.info("Updated active root submission.csv to Candidate B (Super Blend).")

if __name__ == "__main__":
    main()
