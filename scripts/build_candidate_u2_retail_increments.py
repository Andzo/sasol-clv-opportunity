"""Candidate U2: Physical Increment Cost Matrix on the 509 Retail Pool.

Incorporates Rule 5 & Rule 6 physical monetary increment physics:
- Uses category-specific historical spend rates (m6_spend / 6.0) rather than global basket value
- Evaluates raw monetary increment: delta_R = R_future - m3_spend
- Evaluates qualification hurdle: R_future >= max(1.25 * m3_spend, m3_spend + 20)
- Scores categories by expected log monetary increment:
    S_{i, c} = I(qualifies) * log1p(max(0, R_future - m3_spend))
- Solves 509x509 Hungarian optimal transport matching, preserving exact retail seat counts:
    Other Growth: 237, Beverages Growth: 130, Food & beverage Growth: 98, Adoptions: 44.
- Built on top of Candidate V, preserving the 9 verified commuter swaps and Candidate V regression.
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PipelineConfig
from src.data import load_transactions
from src.dataset import SnapshotBuilder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_candidate_u2")


def build_candidate_u2(w_blend: float = 0.85, seeds: list[int] = [42, 100, 2026]) -> Path:
    test_csv = resolve_data_file(PROJECT_ROOT / "test.csv", "test customer IDs CSV")
    test_df = pd.read_csv(test_csv)
    test_ids = [str(x) for x in test_df["ID"]]
    n_test = len(test_ids)
    assert n_test == 5488, f"Expected 5488 test IDs, got {n_test}"

    # 1. Load active champion: Candidate V
    cand_v_path = PROJECT_ROOT / "submissions/submission_candidate_v_boundary_refined_20261014_195540.csv"
    assert cand_v_path.exists(), f"Missing Candidate V: {cand_v_path}"
    df_v = pd.read_csv(cand_v_path)
    df_v["ID"] = df_v["ID"].astype(str)
    df_v = df_v.set_index("ID").reindex(test_ids).reset_index()

    logger.info("Loaded baseline Candidate V (Zindi Leaderboard Champion: 0.312139112)")

    # 2. Extract Retail Cohort (N=509)
    retail_mask = ~df_v["Opportunity"].isin(["Stable", "Existing-category growth: Fuel", "Inactivity"])
    n_retail = retail_mask.sum()
    assert n_retail == 509, f"Expected 509 retail customers, got {n_retail}"

    sub_retail = df_v[retail_mask].copy()
    retail_ids = sub_retail["ID"].tolist()
    retail_classes = sub_retail["Opportunity"].value_counts().index.tolist()
    target_counts = sub_retail["Opportunity"].value_counts().to_dict()

    class_map = {c: i for i, c in enumerate(retail_classes)}
    inv_map = {i: c for c, i in class_map.items()}
    K = len(retail_classes)

    # 3. Build multi-cutoff retail training set
    cfg = PipelineConfig()
    tx = load_transactions(cfg)
    builder = SnapshotBuilder(cfg, tx)

    train_cutoffs = [
        pd.Timestamp("2024-09-01"),
        pd.Timestamp("2024-12-01"),
        pd.Timestamp("2025-03-01"),
        pd.Timestamp("2025-06-01"),
        pd.Timestamp("2025-09-01"),
    ]
    train_snaps = [builder.build(c, with_labels=True) for c in train_cutoffs]

    X_train_list, y_train_list = [], []
    for s in train_snaps:
        m = s.labels["Opportunity"].isin(retail_classes)
        X_train_list.append(s.features.loc[m])
        y_train_list.append(s.labels.loc[m, "Opportunity"])

    X_train = pd.concat(X_train_list, axis=0)
    y_train = pd.concat(y_train_list, axis=0)

    # 4. Load test snapshot features for test retail cohort
    test_snap = builder.build(pd.Timestamp("2025-12-01"), with_labels=False)
    test_feats_retail = test_snap.features.reindex(retail_ids)

    # 5. Multi-seed LightGBM ensemble
    probs_ensemble = np.zeros((n_retail, K), dtype=np.float64)
    for seed in seeds:
        clf = lgb.LGBMClassifier(
            n_estimators=150,
            learning_rate=0.03,
            num_leaves=15,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            verbose=-1,
        )
        clf.fit(X_train, y_train.map(class_map))
        probs_ensemble += clf.predict_proba(test_feats_retail)
    probs_ensemble /= len(seeds)

    # 6. Rule 5 & 7 Feasibility Mask Matrix
    feasibility_mask = np.ones((n_retail, K), dtype=bool)
    for i, rid in enumerate(retail_ids):
        row = test_feats_retail.loc[rid]
        for cl_name, k_idx in class_map.items():
            if cl_name == "Existing-category growth: Beverages":
                if row.get("ever_beverages", 0) == 0:
                    feasibility_mask[i, k_idx] = False
            elif cl_name == "Existing-category growth: Food & beverage":
                if row.get("ever_food_beverage", 0) == 0:
                    feasibility_mask[i, k_idx] = False
            elif cl_name == "Existing-category growth: Other":
                other_growth_cats = [
                    "ever_automotive_products", "ever_car_wash", "ever_coffee", "ever_confectionary",
                    "ever_gas", "ever_general_merchandise", "ever_groceries", "ever_ice_cream",
                    "ever_lubricants", "ever_perishables", "ever_seattle", "ever_snacking",
                    "ever_tobacco", "ever_braai_ice", "ever_breads_rolls"
                ]
                has_any = any(row.get(col, 0) == 1 for col in other_growth_cats if col in row)
                if not has_any:
                    feasibility_mask[i, k_idx] = False
            elif cl_name == "New category adoption: Beverages":
                if row.get("ever_beverages", 0) == 1:
                    feasibility_mask[i, k_idx] = False
            elif cl_name == "New category adoption: Food & beverage":
                if row.get("ever_food_beverage", 0) == 1:
                    feasibility_mask[i, k_idx] = False
            elif cl_name == "New category adoption: Confectionary":
                if row.get("ever_confectionary", 0) == 1:
                    feasibility_mask[i, k_idx] = False
            elif cl_name == "New category adoption: Lubricants":
                if row.get("ever_lubricants", 0) == 1:
                    feasibility_mask[i, k_idx] = False

    # 7. Extract M1 and M6 category-specific spend for retail customers
    cutoff = pd.Timestamp("2025-12-01")
    m1_start = cutoff - pd.DateOffset(months=1)
    m6_start = cutoff - pd.DateOffset(months=6)

    tx_retail = tx[
        (tx["ID"].astype(str).isin(set(retail_ids))) &
        (tx["_time"] >= m6_start) &
        (tx["_time"] < cutoff) &
        (tx["is_qualifying"] == True)
    ]
    tx_m1 = tx_retail[tx_retail["_time"] >= m1_start]

    piv_m1 = tx_m1.pivot_table(index="ID", columns="ItemCategoryLevel2", values="amount", aggfunc="sum", fill_value=0.0).reindex(index=retail_ids, fill_value=0.0)
    piv_m6 = tx_retail.pivot_table(index="ID", columns="ItemCategoryLevel2", values="amount", aggfunc="sum", fill_value=0.0).reindex(index=retail_ids, fill_value=0.0)

    def compute_cat_log_increment(cat_name: str, feat_suffix: str) -> np.ndarray:
        m1_sp = piv_m1[cat_name].to_numpy() if cat_name in piv_m1 else np.zeros(n_retail)
        m6_sp = piv_m6[cat_name].to_numpy() if cat_name in piv_m6 else np.zeros(n_retail)
        m3_sp = test_feats_retail[f"m3spend_{feat_suffix}"].fillna(0).to_numpy() if f"m3spend_{feat_suffix}" in test_feats_retail else np.zeros(n_retail)
        
        cat_monthly_rate = m6_sp / 6.0
        r_future = 0.55 * (3.0 * m1_sp) + 0.45 * (3.0 * cat_monthly_rate)
        
        qualifies = (r_future >= np.maximum(1.25 * m3_sp, m3_sp + 20.0))
        raw_increment = np.maximum(0.0, r_future - m3_sp)
        score = np.where(qualifies, np.log1p(raw_increment), 0.0)
        return score

    score_bev = compute_cat_log_increment("Beverages", "beverages")
    score_fb = compute_cat_log_increment("Food & beverage", "food_beverage")

    other_cats = [
        ("Automotive Products", "automotive_products"),
        ("Braai & Ice", "braai_ice"),
        ("Breads & rolls", "breads_rolls"),
        ("Car Wash", "car_wash"),
        ("Coffee", "coffee"),
        ("Confectionary", "confectionary"),
        ("Gas", "gas"),
        ("General merchandise", "general_merchandise"),
        ("Groceries", "groceries"),
        ("Ice-cream", "ice_cream"),
        ("Lubricants", "lubricants"),
        ("Perishables", "perishables"),
        ("Seattle", "seattle"),
        ("Snacking", "snacking"),
        ("Tobacco", "tobacco")
    ]
    other_scores = []
    for cat_name, feat_suffix in other_cats:
        sc = compute_cat_log_increment(cat_name, feat_suffix)
        ever = test_feats_retail.get(f"ever_{feat_suffix}", pd.Series(0, index=retail_ids)).fillna(0).to_numpy()
        sc = np.where(ever == 1, sc, 0.0)
        other_scores.append(sc)
    score_other = np.max(np.stack(other_scores, axis=0), axis=0)

    S_matrix = np.zeros((n_retail, K))
    for cl_name, k_idx in class_map.items():
        if cl_name == "Existing-category growth: Other":
            S_matrix[:, k_idx] = score_other
        elif cl_name == "Existing-category growth: Beverages":
            S_matrix[:, k_idx] = score_bev
        elif cl_name == "Existing-category growth: Food & beverage":
            S_matrix[:, k_idx] = score_fb

    # 8. Anchor Blending & Hungarian Matching
    anchor_labels = sub_retail["Opportunity"].tolist()
    anchor_indices = [class_map[l] for l in anchor_labels]
    anchor_probs = np.zeros_like(probs_ensemble)
    for i, idx in enumerate(anchor_indices):
        anchor_probs[i, idx] = 1.0

    P_blend = (1.0 - w_blend) * anchor_probs + w_blend * probs_ensemble

    seat_classes = []
    for c in retail_classes:
        seat_classes.extend([class_map[c]] * target_counts[c])
    seat_classes = np.array(seat_classes)

    cost_matrix = -np.log(P_blend[:, seat_classes] + 1e-6)
    for i in range(n_retail):
        for j, seat_k in enumerate(seat_classes):
            cost_matrix[i, j] += -0.15 * np.tanh(S_matrix[i, seat_k] / 5.0)
            if not feasibility_mask[i, seat_k]:
                cost_matrix[i, j] += 1e9

    logger.info("Solving 509x509 Hungarian matching with physical increment cost matrix...")
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matched_preds = [inv_map[seat_classes[c]] for c in col_ind]

    matched_counts = pd.Series(matched_preds).value_counts().to_dict()
    logger.info("Matched Retail Distribution: %s", matched_counts)
    for c, target_n in target_counts.items():
        assert matched_counts.get(c, 0) == target_n, f"Capacity mismatch for {c}"

    diff_mask = np.array(anchor_labels) != np.array(matched_preds)
    n_swapped = diff_mask.sum()
    logger.info("Candidate U2 Hungarian matching swapped %d / %d retail customers relative to Candidate V", n_swapped, n_retail)

    swapped_df = pd.DataFrame({
        "ID": np.array(retail_ids)[diff_mask],
        "Candidate_V_Label": np.array(anchor_labels)[diff_mask],
        "Candidate_U2_Label": np.array(matched_preds)[diff_mask],
    })
    logger.info("Swap transitions in Retail Cohort:\n%s", swapped_df.to_string())

    # 9. Apply retail updates to Candidate V base
    opp_final = df_v["Opportunity"].copy()
    opp_final.loc[retail_mask] = matched_preds

    counts = opp_final.value_counts().to_dict()
    logger.info("Final Opportunity Distribution in Candidate U2: %s", counts)
    assert counts["Stable"] == 1960
    assert counts["Existing-category growth: Fuel"] == 1576
    assert counts["Inactivity"] == 1443
    assert counts["Existing-category growth: Other"] == 237
    assert counts["Existing-category growth: Beverages"] == 130
    assert counts["Existing-category growth: Food & beverage"] == 98

    # 10. Assemble submission dataframe
    df_u2 = pd.DataFrame({
        "ID": test_ids,
        "CLV_fuel": df_v["CLV_fuel"].to_numpy(),
        "CLV_nonfuel": df_v["CLV_nonfuel"].to_numpy(),
        "Opportunity": opp_final.to_numpy(),
    })

    assert len(df_u2) == 5488
    assert list(df_u2.columns) == ["ID", "CLV_fuel", "CLV_nonfuel", "Opportunity"]
    assert not df_u2.isna().any().any()
    assert (df_u2["CLV_fuel"] >= 0).all()
    assert (df_u2["CLV_nonfuel"] >= 0).all()

    # 11. Save submission and manifest
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submissions_dir = PROJECT_ROOT / "submissions"
    submissions_dir.mkdir(parents=True, exist_ok=True)
    out_path = submissions_dir / f"submission_candidate_u2_retail_increments_{timestamp}.csv"
    df_u2.to_csv(out_path, index=False)

    sha256 = hashlib.sha256(out_path.read_bytes()).hexdigest()
    logger.info("Candidate U2 submission written: %s (SHA-256: %s)", out_path, sha256)

    manifest = {
        "name": "candidate_u2_retail_increments",
        "filename": out_path.name,
        "sha256": sha256,
        "timestamp": timestamp,
        "row_count": len(df_u2),
        "clv_fuel_mean": float(df_u2["CLV_fuel"].mean()),
        "clv_nonfuel_mean": float(df_u2["CLV_nonfuel"].mean()),
        "macro_counts": {
            "Stable": 1960,
            "Existing-category growth: Fuel": 1576,
            "Inactivity": 1443,
            "Retail Cohort Total": 509,
        },
        "swaps_performed_in_retail": int(n_swapped),
        "swapped_customers": swapped_df.to_dict(orient="records"),
        "verified_zindi_components": {
            "clv_fuel_rmse": 0.590265219,
            "clv_nonfuel_rmse": 0.721619948,
            "opportunity_f1_candidate_v": 0.541843185,
        },
        "scientific_basis": [
            "Category-specific historical monthly spend rate (m6_spend / 6.0) replacing global avg_basket",
            "Rule 5 & Rule 6 physical qualification hurdle: R_future >= max(1.25 * m3_spend, m3_spend + 20)",
            "Expected log monetary increment score S_{i, c} rewarding genuine high-increment categories",
            "509x509 Hungarian optimal transport matching preserving exact retail seat counts",
            "100% preservation of Candidate V's 9 verified commuter swaps (Stable=1960, FG=1576)",
            "100% preservation of Candidate V's verified regression outputs (0.590265 fuel, 0.721620 nonfuel)",
        ],
    }

    manifest_path = out_path.with_suffix(".manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Manifest written: %s", manifest_path)
    return out_path


if __name__ == "__main__":
    build_candidate_u2()
