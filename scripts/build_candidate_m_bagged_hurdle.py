"""Build Candidate M: Multi-Seed Bagged Two-Part Hurdle Regressors.

Executes Lever 3:
1. Trains BaggedTwoPartFuelHurdle and BaggedTwoPartNonfuelHurdle across 5 random seeds
   ([42, 100, 2024, 777, 314]) and dual architectures (LightGBM Huber + CatBoost MAE)
   on the full historical panel through 2025-09-01.
2. Incurs test inference for CLV Fuel and Nonfuel on test.csv (2025-12-01).
3. Blends 90% verified record peak regression baselines with 10% variance-reduced bagged hurdles.
4. Pairs with Candidate L decoupled Opportunity predictions (locking Inactivity to 1,443).
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
logger = logging.getLogger("build_candidate_m")


def main() -> None:
    test_csv = resolve_data_file(PROJECT_ROOT / "test.csv", "test customer IDs CSV")
    test_df = pd.read_csv(test_csv)
    test_ids = [str(x) for x in test_df["ID"]]
    n_test = len(test_ids)
    assert n_test == 5488, f"Expected 5488 test IDs, got {n_test}"

    data_cache = PROJECT_ROOT / "data_cache"

    # 1. Load Candidate L (baseline for regression + peak decoupled Opportunity)
    cand_l_csv = PROJECT_ROOT / "submissions" / "submission_candidate_l_boundary_decoupled_20261013_235742.csv"
    assert cand_l_csv.exists(), f"Missing Candidate L: {cand_l_csv}"
    s_cand_l = pd.read_csv(cand_l_csv)
    s_cand_l["ID"] = s_cand_l["ID"].astype(str)
    s_cand_l = s_cand_l.set_index("ID").reindex(test_ids)

    # 2. Build full stacked training panel through 2025-09-01
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

    # 3. Fit Bagged Two-Part Fuel Hurdle
    seeds = (42, 100, 2024, 777, 314)
    logger.info("Training BaggedTwoPartFuelHurdle across 5 seeds: %s...", seeds)
    t0 = time.time()
    bag_fuel = BaggedTwoPartFuelHurdle(seeds=seeds, n_jobs=-1, use_gpu=False)
    bag_fuel.fit(X_train, y_fuel_train)
    logger.info("BaggedTwoPartFuelHurdle fitted in %.2f seconds.", time.time() - t0)

    # 4. Fit Bagged Two-Part Nonfuel Hurdle
    logger.info("Training BaggedTwoPartNonfuelHurdle across 5 seeds: %s...", seeds)
    t0 = time.time()
    bag_nf = BaggedTwoPartNonfuelHurdle(seeds=seeds, n_jobs=-1, use_gpu=False)
    bag_nf.fit(X_train, y_nf_train)
    logger.info("BaggedTwoPartNonfuelHurdle fitted in %.2f seconds.", time.time() - t0)

    # 5. Test Customer Inference (2025-12-01)
    f_test = pd.read_parquet(data_cache / "feats_2025-12-01.parquet")
    f_test["ID"] = f_test["ID"].astype(str)
    f_test = f_test.set_index("ID").reindex(test_ids)
    drop_test = [col for col in f_test.columns if col in ("ID", "opp", "Opportunity") or f_test[col].dtype == object]
    X_test = f_test.drop(columns=drop_test)

    hurdle_fuel_test = bag_fuel.predict(X_test)
    hurdle_nf_test = bag_nf.predict(X_test)

    # 6. Variance-Reduction Blending (90% Verified Peak Baseline + 10% Bagged Hurdle)
    cand_l_fuel = s_cand_l["CLV_fuel"].to_numpy()
    cand_l_nf = s_cand_l["CLV_nonfuel"].to_numpy()

    pred_fuel_blend = 0.90 * cand_l_fuel + 0.10 * hurdle_fuel_test
    pred_nf_blend = 0.90 * cand_l_nf + 0.10 * hurdle_nf_test

    logger.info(
        "CLV Fuel: Cand L mean=%.4f, Hurdle mean=%.4f -> Blend mean=%.4f (MAE diff=%.4f)",
        cand_l_fuel.mean(), hurdle_fuel_test.mean(), pred_fuel_blend.mean(),
        np.mean(np.abs(pred_fuel_blend - cand_l_fuel)),
    )
    logger.info(
        "CLV Nonfuel: Cand L mean=%.4f, Hurdle mean=%.4f -> Blend mean=%.4f (MAE diff=%.4f)",
        cand_l_nf.mean(), hurdle_nf_test.mean(), pred_nf_blend.mean(),
        np.mean(np.abs(pred_nf_blend - cand_l_nf)),
    )

    # 7. Assemble Candidate M
    df_sub = pd.DataFrame({
        "ID": test_df["ID"],
        "CLV_fuel": pred_fuel_blend,
        "CLV_nonfuel": pred_nf_blend,
        "Opportunity": s_cand_l["Opportunity"].to_numpy(),
    })

    # Validate format
    assert len(df_sub) == 5488, f"Invalid row count: {len(df_sub)}"
    assert list(df_sub.columns) == ["ID", "CLV_fuel", "CLV_nonfuel", "Opportunity"], f"Invalid columns: {df_sub.columns}"
    assert not df_sub.isna().any().any(), "Submission contains NaN values"
    assert (df_sub["CLV_fuel"] >= 0).all(), "Negative fuel values"
    assert (df_sub["CLV_nonfuel"] >= 0).all(), "Negative nonfuel values"

    counts = df_sub["Opportunity"].value_counts()
    inact_cnt = counts.get("Inactivity", 0)
    assert 1440 <= inact_cnt <= 1445, f"Inactivity count {inact_cnt} out of bounds [1440, 1445]"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub_dir = PROJECT_ROOT / "submissions"
    sub_dir.mkdir(parents=True, exist_ok=True)
    out_path = sub_dir / f"submission_candidate_m_bagged_hurdle_{timestamp}.csv"
    df_sub.to_csv(out_path, index=False)

    sha256 = hashlib.sha256(out_path.read_bytes()).hexdigest()
    manifest = {
        "name": "candidate_m_bagged_hurdle",
        "filename": out_path.name,
        "sha256": sha256,
        "timestamp": timestamp,
        "row_count": len(df_sub),
        "clv_fuel_rmse_target": "< 0.59000",
        "clv_nonfuel_rmse_target": "< 0.72200",
        "opportunity_f1_verified": 0.540531031,
        "composite_score_projected": "> 0.31250",
        "class_distribution": counts.to_dict(),
        "scientific_basis": [
            "Lever 3: Multi-Seed Bagged Two-Part Hurdle Regressors across 5 seeds [42, 100, 2024, 777, 314]",
            "Dual-model positive spend intensity ensemble: 55% LightGBM Huber + 45% CatBoost MAE",
            "Variance-reduction blend: 90% verified peak baseline + 10% bagged hurdle predictions",
            "Pairing with Lever 1 decoupled Opportunity predictions from Candidate L (1,443 Inactivity lock-in)",
        ],
    }
    manifest_path = sub_dir / f"{out_path.stem}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # Update active root submission.csv and submission_latest.csv
    df_sub.to_csv(PROJECT_ROOT / "submission.csv", index=False)
    df_sub.to_csv(sub_dir / "submission_latest.csv", index=False)
    sub_dir.joinpath("submission_latest.manifest.json").write_text(json.dumps(manifest, indent=2))

    logger.info("Successfully generated Candidate M (%s, SHA256: %s)", out_path.name, sha256[:10])
    logger.info("Root submission.csv and submission_latest.csv updated.")


if __name__ == "__main__":
    main()
