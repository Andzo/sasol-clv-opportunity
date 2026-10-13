"""Build Candidate J: Fuel Hurdle Breakthrough.

Combines:
1. CLV Fuel: 90% verified Candidate C/I (0.590668 RMSE) + 10% Two-Part Fuel Hurdle
   architecture (P(Fuel > 0 | X) * E[Y | X, Y > 0]), targeting drop below 0.5900.
2. CLV Nonfuel: Verified record best from Candidate H (0.72220868 RMSE).
3. Opportunity: Verified peak Opportunity F1 from Candidate D (0.540531031).

Target: Beat Candidate I (0.311234500) and accelerate trajectory toward > 0.330.
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
from src.data import load_transactions
from src.dataset import SnapshotBuilder, stack
from src.models import StoreAttachmentClassifier, make_regressor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_candidate_j")


def main() -> None:
    test_csv = PROJECT_ROOT / "test.csv"
    test_df = pd.read_csv(test_csv)
    test_ids = [str(x) for x in test_df["ID"]]
    n_test = len(test_ids)
    assert n_test == 5488, f"Expected 5488 test IDs, got {n_test}"

    # 1. Load Candidate I (locks in Nonfuel=0.722209 and Opp=0.540531)
    cand_i_csv = PROJECT_ROOT / "submissions" / "submission_candidate_i_peak_verified_20261013_194106.csv"
    assert cand_i_csv.exists(), f"Missing Candidate I: {cand_i_csv}"
    s_cand_i = pd.read_csv(cand_i_csv)
    s_cand_i["ID"] = s_cand_i["ID"].astype(str)
    s_cand_i = s_cand_i.set_index("ID").reindex(test_ids)

    # 2. Train Two-Part Fuel Hurdle on full historical panel
    logger.info("Training Two-Part Fuel Hurdle on full historical panel...")
    cfg = PipelineConfig()
    tx = load_transactions(cfg)
    builder = SnapshotBuilder(cfg, tx)

    full_train_cutoffs = cfg.training_cutoffs(boundary=pd.Timestamp(cfg.test_cutoff))
    full_snaps = [builder.build(c, with_labels=True) for c in full_train_cutoffs]
    full_panel = stack(full_snaps)

    y_fuel_train = full_panel.labels["CLV_fuel"].to_numpy()
    pos_mask = y_fuel_train > 0

    # Binary Fuel activity classifier
    clf_fuel = StoreAttachmentClassifier(seed=42, n_jobs=-1, use_gpu=True)
    clf_fuel.fit(full_panel.X, pos_mask.astype(int))

    # Positive spend intensity regressor
    reg_fuel_pos = make_regressor("lgbm_huber", seed=42, n_jobs=-1, use_gpu=True)
    reg_fuel_pos.fit(full_panel.X.iloc[pos_mask], y_fuel_train[pos_mask])

    test_snap = builder.build(pd.Timestamp(cfg.test_cutoff), with_labels=False)
    test_feats = test_snap.features.reindex(pd.Index(test_ids, name="ID"))

    p_fuel_active = clf_fuel.predict_proba(test_feats)
    p_fuel_amt = np.clip(reg_fuel_pos.predict(test_feats), 0.0, None)
    hurdle_fuel_test = p_fuel_active * p_fuel_amt

    # 3. Blend 90% Candidate I Fuel + 10% Two-Part Fuel Hurdle
    cand_i_fuel = s_cand_i["CLV_fuel"].to_numpy()
    pred_fuel_blend = 0.90 * cand_i_fuel + 0.10 * hurdle_fuel_test
    logger.info(
        "Fuel blend: Cand I mean=%.4f, Hurdle mean=%.4f -> Blend mean=%.4f (MAE diff=%.4f)",
        cand_i_fuel.mean(), hurdle_fuel_test.mean(), pred_fuel_blend.mean(),
        np.mean(np.abs(pred_fuel_blend - cand_i_fuel))
    )

    df_sub = pd.DataFrame({
        "ID": test_df["ID"],
        "CLV_fuel": pred_fuel_blend,
        "CLV_nonfuel": s_cand_i["CLV_nonfuel"].to_numpy(),
        "Opportunity": s_cand_i["Opportunity"].to_numpy(),
    })

    # Validate format
    assert len(df_sub) == 5488, f"Invalid row count: {len(df_sub)}"
    assert list(df_sub.columns) == ["ID", "CLV_fuel", "CLV_nonfuel", "Opportunity"]
    assert not df_sub.isna().any().any(), "Submission contains NaN values"
    assert (df_sub["CLV_fuel"] >= 0).all(), "Negative fuel values"
    assert (df_sub["CLV_nonfuel"] >= 0).all(), "Negative nonfuel values"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub_dir = PROJECT_ROOT / "submissions"
    out_path = sub_dir / f"submission_candidate_j_fuel_hurdle_{timestamp}.csv"
    df_sub.to_csv(out_path, index=False)

    sha256 = hashlib.sha256(out_path.read_bytes()).hexdigest()
    manifest = {
        "name": "candidate_j_fuel_hurdle",
        "filename": out_path.name,
        "sha256": sha256,
        "timestamp": timestamp,
        "row_count": len(df_sub),
        "clv_fuel_mean": float(df_sub["CLV_fuel"].mean()),
        "clv_nonfuel_mean": float(df_sub["CLV_nonfuel"].mean()),
        "opportunity_f1_baseline": 0.540531031,
        "class_distribution": df_sub["Opportunity"].value_counts().to_dict(),
        "scientific_innovations": [
            "CLV Fuel: Two-Part Fuel Hurdle (P(Fuel > 0 | X) * E[Y | X, Y > 0]) blended at 10% with verified best Candidate C/I (0.590668 RMSE)",
            "CLV Nonfuel: Strictly preserved Candidate H verified best Two-Part Nonfuel Hurdle blend (0.72220868 RMSE)",
            "Opportunity: Strictly preserved Candidate D verified peak Opportunity F1 (0.540531031)",
        ],
    }
    manifest_path = sub_dir / f"{out_path.stem}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    logger.info("Created %s (%d rows, SHA256: %s)", out_path.name, len(df_sub), sha256[:10])

    # Update root submission.csv and submission_latest.csv
    df_sub.to_csv(PROJECT_ROOT / "submission.csv", index=False)
    df_sub.to_csv(sub_dir / "submission_latest.csv", index=False)
    logger.info("Successfully updated root submission.csv and submission_latest.csv to Candidate J.")


if __name__ == "__main__":
    main()
