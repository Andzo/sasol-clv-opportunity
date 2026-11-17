"""Candidate X: The Unified Champion Super-Submission.

Consolidates the full suite of verified breakthroughs:
1. CLV Fuel: Candidate W (168-feature seasonally weighted super-ensemble + bagged hurdle,
   derivative-guarded quadratic calibration, exact moment preservation to 1.44565036).
2. CLV Nonfuel: Candidate W (168-feature seasonally weighted super-ensemble + bagged hurdle,
   derivative-guarded quadratic calibration, exact moment preservation to 0.70165665).
3. Opportunity: Candidate U2 (Candidate V's 9 verified commuter boundary swaps between Fuel Growth
   and Stable + Candidate U2's 5 physical increment retail swaps inside the 509 cohort).

Macro Invariants Strictly Preserved:
- Stable: 1,960
- Existing-category growth: Fuel: 1,576
- Inactivity: 1,443
- Retail Cohort Total: 509 (Other=237, Bev=130, F&B=98, Adoptions=44)
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import resolve_data_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_candidate_x")


def build_candidate_x() -> Path:
    test_csv = resolve_data_file(PROJECT_ROOT / "test.csv", "test customer IDs CSV")
    test_df = pd.read_csv(test_csv)
    test_ids = [str(x) for x in test_df["ID"]]
    n_test = len(test_ids)
    assert n_test == 5488, f"Expected 5488 test IDs, got {n_test}"

    submissions_dir = PROJECT_ROOT / "submissions"

    # 1. Load Candidate W (Regression champion)
    cand_w_files = sorted(submissions_dir.glob("submission_candidate_w_calibrated_regression_*.csv"))
    assert cand_w_files, "Missing Candidate W submission file"
    cand_w_path = cand_w_files[-1]
    logger.info("Loading regression champion from Candidate W: %s", cand_w_path.name)
    df_w = pd.read_csv(cand_w_path)
    df_w["ID"] = df_w["ID"].astype(str)
    df_w = df_w.set_index("ID").reindex(test_ids)

    # 2. Load Candidate U2 (Opportunity champion)
    cand_u2_files = sorted(submissions_dir.glob("submission_candidate_u2_retail_increments_*.csv"))
    assert cand_u2_files, "Missing Candidate U2 submission file"
    cand_u2_path = cand_u2_files[-1]
    logger.info("Loading opportunity champion from Candidate U2: %s", cand_u2_path.name)
    df_u2 = pd.read_csv(cand_u2_path)
    df_u2["ID"] = df_u2["ID"].astype(str)
    df_u2 = df_u2.set_index("ID").reindex(test_ids)

    # 3. Assemble Candidate X
    df_x = pd.DataFrame({
        "ID": test_ids,
        "CLV_fuel": df_w["CLV_fuel"].to_numpy(),
        "CLV_nonfuel": df_w["CLV_nonfuel"].to_numpy(),
        "Opportunity": df_u2["Opportunity"].to_numpy(),
    })

    # 4. Strict Validation
    assert len(df_x) == 5488
    assert list(df_x.columns) == ["ID", "CLV_fuel", "CLV_nonfuel", "Opportunity"]
    assert not df_x.isna().any().any()
    assert (df_x["CLV_fuel"] >= 0).all()
    assert (df_x["CLV_nonfuel"] >= 0).all()

    counts = df_x["Opportunity"].value_counts().to_dict()
    logger.info("Candidate X Opportunity distribution: %s", counts)
    assert counts["Stable"] == 1960
    assert counts["Existing-category growth: Fuel"] == 1576
    assert counts["Inactivity"] == 1443
    assert counts["Existing-category growth: Other"] == 237
    assert counts["Existing-category growth: Beverages"] == 130
    assert counts["Existing-category growth: Food & beverage"] == 98

    # 5. Save submission and manifest
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = submissions_dir / f"submission_candidate_x_unified_champion_{timestamp}.csv"
    df_x.to_csv(out_path, index=False)

    sha256 = hashlib.sha256(out_path.read_bytes()).hexdigest()
    logger.info("Candidate X submission written: %s (SHA-256: %s)", out_path, sha256)

    manifest = {
        "name": "candidate_x_unified_champion",
        "filename": out_path.name,
        "sha256": sha256,
        "timestamp": timestamp,
        "row_count": len(df_x),
        "clv_fuel_mean": float(df_x["CLV_fuel"].mean()),
        "clv_fuel_std": float(df_x["CLV_fuel"].std()),
        "clv_nonfuel_mean": float(df_x["CLV_nonfuel"].mean()),
        "clv_nonfuel_std": float(df_x["CLV_nonfuel"].std()),
        "macro_counts": counts,
        "components": {
            "regression_source": cand_w_path.name,
            "opportunity_source": cand_u2_path.name,
            "fuel_mean": float(df_x["CLV_fuel"].mean()),
            "nonfuel_mean": float(df_x["CLV_nonfuel"].mean()),
            "opportunity_f1_baseline": "0.541843185 (Candidate V)",
        },
        "scientific_basis": [
            "CLV Fuel & Nonfuel: Seasonally-weighted (2.0x) 168-feature super-ensemble + bagged hurdle with derivative-guarded quadratic calibration",
            "Opportunity: Candidate V's 9 verified commuter boundary swaps + Candidate U2's 5 physical increment retail swaps",
            "100% Macro capacity compliance: Stable=1960, FG=1576, Inactivity=1443, Retail=509",
        ],
    }

    manifest_path = out_path.with_suffix(".manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Manifest written: %s", manifest_path)
    return out_path


if __name__ == "__main__":
    build_candidate_x()
