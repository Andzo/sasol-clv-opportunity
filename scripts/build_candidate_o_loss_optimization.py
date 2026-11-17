"""Build Candidate O: Huber Loss Fine-Tuning and Optimal Hurdle Blending.

Executes Lever 4:
1. Trains BaggedTwoPartFuelHurdle (seeds=[42, 100, 2024, 777, 314], huber_alpha=0.85, power_shrinkage=0.98)
   and BaggedTwoPartNonfuelHurdle (seeds=[42, 100, 2024, 777, 314], huber_alpha=0.85, power_shrinkage=0.96, zero_gate_threshold=0.04)
   on the full stacked historical panel through 2025-09-01.
2. Incurs test inference for CLV Fuel and Nonfuel on test.csv (2025-12-01).
3. Blends tuned bagged hurdle predictions with Candidate N baseline (Fuel: w=0.25, Nonfuel: w=0.30).
4. Pairs with Candidate N Cost-Sensitive Optimal Transport LP Opportunity (Inactivity = 1,444).
5. Strictly validates format and updates root submission.csv.
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import resolve_data_file

from src.hurdle import BaggedTwoPartFuelHurdle, BaggedTwoPartNonfuelHurdle

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_candidate_o")


def main() -> None:
    test_csv = resolve_data_file(PROJECT_ROOT / "test.csv", "test customer IDs CSV")
    test_df = pd.read_csv(test_csv)
    test_ids = [str(x) for x in test_df["ID"]]
    n_test = len(test_ids)
    assert n_test == 5488, f"Expected 5488 test IDs, got {n_test}"

    data_cache = PROJECT_ROOT / "data_cache"

    # 1. Load Candidate N (baseline for regression + Cost-Sensitive LP Opportunity)
    cand_n_csv = PROJECT_ROOT / "submissions" / "submission_candidate_n_cost_sensitive_lp_20261014_001330.csv"
    assert cand_n_csv.exists(), f"Missing Candidate N: {cand_n_csv}"
    s_cand_n = pd.read_csv(cand_n_csv)
    s_cand_n["ID"] = s_cand_n["ID"].astype(str)
    s_cand_n = s_cand_n.set_index("ID").reindex(test_ids)

    # 2. Build full stacked training panel across all 5 cutoffs
    logger.info("Assembling full historical training panel across all 5 cutoffs...")
    train_cuts = ["2024-09-01", "2024-12-01", "2025-03-01", "2025-06-01", "2025-09-01"]
    X_train_list, y_fuel_list, y_nf_list = [], [], []

    for c in train_cuts:
        f_path = data_cache / f"feats_{c}.parquet"
        l_path = data_cache / f"labels_{c}.parquet"
        assert f_path.exists() and l_path.exists(), f"Missing cache for {c}"
        f_df = pd.read_parquet(f_path)
        f_df["ID"] = f_df["ID"].astype(str)
        f_df = f_df.set_index("ID")
        l_df = pd.read_parquet(l_path)
        l_df.index = f_df.index

        drop_cols = [col for col in f_df.columns if col in ("ID", "opp", "Opportunity") or f_df[col].dtype == object]
        X_train_list.append(f_df.drop(columns=drop_cols))
        y_fuel_list.append(l_df["CLV_fuel"].to_numpy())
        y_nf_list.append(l_df["CLV_nonfuel"].to_numpy())

    X_train = pd.concat(X_train_list, axis=0)
    y_fuel_train = np.concatenate(y_fuel_list)
    y_nf_train = np.concatenate(y_nf_list)
    logger.info("Stacked historical panel: N=%d instances x %d features", len(X_train), X_train.shape[1])

    # 3. Fit Bagged Two-Part Fuel Hurdle with tuned Huber alpha=0.85 and shrinkage=0.98
    seeds = (42, 100, 2024, 777, 314)
    logger.info("Training Tuned BaggedTwoPartFuelHurdle (alpha=0.85, shrinkage=0.98) across 5 seeds: %s...", seeds)
    t0 = time.time()
    bag_fuel = BaggedTwoPartFuelHurdle(
        seeds=seeds,
        n_jobs=-1,
        use_gpu=False,
        huber_alpha=0.85,
        power_shrinkage=0.98,
        zero_gate_threshold=0.0,
    )
    bag_fuel.fit(X_train, y_fuel_train)
    logger.info("Tuned Bagged Fuel Hurdle fitted in %.2fs", time.time() - t0)

    # 4. Fit Bagged Two-Part Nonfuel Hurdle with tuned Huber alpha=0.85, shrinkage=0.96, gate=0.04
    logger.info("Training Tuned BaggedTwoPartNonfuelHurdle (alpha=0.85, shrinkage=0.96, gate=0.04) across 5 seeds: %s...", seeds)
    t0 = time.time()
    bag_nf = BaggedTwoPartNonfuelHurdle(
        seeds=seeds,
        n_jobs=-1,
        use_gpu=False,
        huber_alpha=0.85,
        power_shrinkage=0.96,
        zero_gate_threshold=0.04,
    )
    bag_nf.fit(X_train, y_nf_train)
    logger.info("Tuned Bagged Nonfuel Hurdle fitted in %.2fs", time.time() - t0)

    # 5. Load test features (2025-12-01)
    test_feats_path = data_cache / "feats_2025-12-01.parquet"
    assert test_feats_path.exists(), f"Missing test features cache: {test_feats_path}"
    test_feats_df = pd.read_parquet(test_feats_path)
    test_feats_df["ID"] = test_feats_df["ID"].astype(str)
    test_feats_df = test_feats_df.set_index("ID").reindex(test_ids)
    drop_test_cols = [col for col in test_feats_df.columns if col in ("ID", "opp", "Opportunity") or test_feats_df[col].dtype == object]
    X_test = test_feats_df.drop(columns=drop_test_cols)

    logger.info("Generating test predictions on N=%d test customers...", len(X_test))
    hurdle_fuel_test = bag_fuel.predict(X_test)
    hurdle_nf_test = bag_nf.predict(X_test)

    # 6. Variance-Reduction Blending with Candidate N Baseline
    # Empirical out-of-time optima: Fuel w=0.25, Nonfuel w=0.30
    cand_n_fuel = s_cand_n["CLV_fuel"].to_numpy()
    cand_n_nf = s_cand_n["CLV_nonfuel"].to_numpy()

    w_fuel = 0.25
    w_nf = 0.30
    pred_fuel_blend = (1.0 - w_fuel) * cand_n_fuel + w_fuel * hurdle_fuel_test
    pred_nf_blend = (1.0 - w_nf) * cand_n_nf + w_nf * hurdle_nf_test

    logger.info(
        "CLV Fuel: Cand N mean=%.4f, Hurdle mean=%.4f -> Blend (w=%.2f) mean=%.4f (MAE diff=%.4f)",
        cand_n_fuel.mean(), hurdle_fuel_test.mean(), w_fuel, pred_fuel_blend.mean(),
        np.mean(np.abs(pred_fuel_blend - cand_n_fuel)),
    )
    logger.info(
        "CLV Nonfuel: Cand N mean=%.4f, Hurdle mean=%.4f -> Blend (w=%.2f) mean=%.4f (MAE diff=%.4f)",
        cand_n_nf.mean(), hurdle_nf_test.mean(), w_nf, pred_nf_blend.mean(),
        np.mean(np.abs(pred_nf_blend - cand_n_nf)),
    )

    # 7. Assemble Candidate O (Pairing with Candidate N Cost-Sensitive LP Opportunity)
    df_sub = pd.DataFrame({
        "ID": test_df["ID"],
        "CLV_fuel": pred_fuel_blend,
        "CLV_nonfuel": pred_nf_blend,
        "Opportunity": s_cand_n["Opportunity"].to_numpy(),
    })

    # Validate format strictly
    assert len(df_sub) == 5488, f"Invalid row count: {len(df_sub)}"
    assert list(df_sub.columns) == ["ID", "CLV_fuel", "CLV_nonfuel", "Opportunity"], f"Invalid columns: {df_sub.columns}"
    assert not df_sub.isna().any().any(), "Submission contains NaN values"
    assert (df_sub["CLV_fuel"] >= 0).all(), "Negative fuel values"
    assert (df_sub["CLV_nonfuel"] >= 0).all(), "Negative nonfuel values"

    counts = df_sub["Opportunity"].value_counts()
    inact_cnt = counts.get("Inactivity", 0)
    assert 1442 <= inact_cnt <= 1445, f"Inactivity count {inact_cnt} out of bounds [1442, 1445]"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub_dir = PROJECT_ROOT / "submissions"
    sub_dir.mkdir(parents=True, exist_ok=True)
    out_path = sub_dir / f"submission_candidate_o_loss_optimization_{timestamp}.csv"
    df_sub.to_csv(out_path, index=False)

    sha256 = hashlib.sha256(out_path.read_bytes()).hexdigest()
    manifest = {
        "name": "candidate_o_loss_optimization",
        "filename": out_path.name,
        "sha256": sha256,
        "timestamp": timestamp,
        "row_count": len(df_sub),
        "clv_fuel_mean": float(df_sub["CLV_fuel"].mean()),
        "clv_nonfuel_mean": float(df_sub["CLV_nonfuel"].mean()),
        "clv_fuel_rmse_target": "< 0.58950",
        "clv_nonfuel_rmse_target": "< 0.72000",
        "opportunity_f1_target": 0.54580,
        "composite_score_projected": ">= 0.31510",
        "class_distribution": counts.to_dict(),
        "scientific_basis": [
            "Lever 4: Huber loss fine-tuning (alpha=0.85) on positive spenders compressing heavy-tail dispersion",
            "Calibrated power shrinkage (gamma=0.96 nonfuel, gamma=0.98 fuel) and zero-gating (tau=0.04)",
            "Analytical OLS variance-reduction blending with Candidate N baseline (w_f=0.25, w_nf=0.30)",
            "Pairing with Lever 2 Cost-Sensitive Optimal Transport LP Opportunity (Inactivity locked to 1,444)",
        ],
    }
    manifest_path = sub_dir / f"{out_path.stem}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # Update active root submission.csv and submission_latest.csv
    df_sub.to_csv(PROJECT_ROOT / "submission.csv", index=False)
    df_sub.to_csv(sub_dir / "submission_latest.csv", index=False)
    sub_dir.joinpath("submission_latest.manifest.json").write_text(json.dumps(manifest, indent=2))

    logger.info("Successfully generated Candidate O (%s, SHA256: %s)", out_path.name, sha256)
    logger.info("Root submission.csv and submission_latest.csv updated.")


if __name__ == "__main__":
    main()
