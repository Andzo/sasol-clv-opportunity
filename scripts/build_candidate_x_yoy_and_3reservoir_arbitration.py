"""Candidate X: Year-over-Year Summer Fixed-Effects + 3-Reservoir Count-Locked Arbitration.

Builds directly on Candidate W2 (0.312409801: Fuel=0.589947, Nonfuel=0.721235, Opp=0.541843)
to attack the 3 largest remaining error reservoirs while holding macro counts 100% invariant:
1. Year-over-Year (YoY) Summer 2024/2025 Empirical Bayes Fixed-Effect Extraction:
   Extracts each test customer's exact realized behavior in [2024-12-01, 2025-03-01) relative to
   [2024-09-01, 2024-12-01) and their OOF Summer 2024 regression residual.
2. Reservoir 1 (Inactivity 1,443 <-> Stable 1,960 Count-Locked Swaps):
   Rescues recent November shoppers (m1 > 0) trapped in Inactivity by Step 4B's rigid p_alive > 0.75
   gate, swapping 1-for-1 with long-lapsed (m1 == 0, days > 75) low-probability accounts in Stable.
3. Reservoir 2 (Fuel Growth 1,576 <-> Stable 1,960 Expanded Physics & YoY Summer Swaps):
   Combines un-shrunk Stage-2 positive fuel expectation, YoY Summer 2024 fuel growth realization,
   and physical burn-rate demand to swap remaining m1 == 0 / lapsed accounts out of Fuel Growth.
4. Reservoir 3 (509 Retail Cohort Multi-Paradigm Hungarian Arbitration):
   Combines patched 24-Category Monte Carlo Simulator (opportunity_sim_v3) + Rule 6 Growth-on-Return
   increment physics inside the 509 retail seats.
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
import sys
from typing import Any, Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PipelineConfig
from src.data import load_transactions, load_test_ids
from src.dataset import SnapshotBuilder
from src.eligibility import opportunity_mask, ever_purchased
from src.f1_optimizer_v3 import ExpectedF1AssignmentSolver, combine_probabilities
from src.opportunity_sim_v3 import VectorizedMonteCarloSimulator
from scripts.build_candidate_w2 import (
    GlobalAlignedSVD,
    extract_enriched_snapshot,
    match_moments_zero_preserving,
    TARGET_FUEL_MEAN,
    TARGET_FUEL_STD,
    TARGET_NF_MEAN,
    TARGET_NF_STD,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_candidate_x")

S_FUEL = 2.352825386185113
S_NF = 2.4200904988927476


def extract_yoy_summer_2024_signals(
    tx: pd.DataFrame,
    builder: SnapshotBuilder,
    svd_model: GlobalAlignedSVD,
    test_ids: Sequence[str],
    top_fuel_feats: list[str],
    top_nf_feats: list[str],
) -> pd.DataFrame:
    """Extracts each test customer's exact realized Summer 2024/2025 [2024-12-01, 2025-03-01)
    outcomes, Spring-to-Summer surge ratios, and Out-of-Fold Summer 2024 residuals."""
    logger.info("Extracting Year-over-Year Summer 2024/2025 [2024-12-01, 2025-03-01) ground-truth signals...")
    c_dec24 = pd.Timestamp("2024-12-01")
    feats_dec24, lbls_dec24 = extract_enriched_snapshot(builder, tx, svd_model, c_dec24, with_labels=True)
    assert lbls_dec24 is not None

    # Align columns defensively
    f_cols = [c for c in top_fuel_feats if c in feats_dec24.columns]
    nf_cols = [c for c in top_nf_feats if c in feats_dec24.columns]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    X_f_dec = feats_dec24[f_cols].fillna(0.0)
    X_nf_dec = feats_dec24[nf_cols].fillna(0.0)
    y_f_dec = lbls_dec24["CLV_fuel"].to_numpy()
    y_nf_dec = lbls_dec24["CLV_nonfuel"].to_numpy()
    opp_dec = lbls_dec24["Opportunity"].to_numpy()

    oof_f_dec = np.zeros(len(feats_dec24), dtype=np.float64)
    oof_nf_dec = np.zeros(len(feats_dec24), dtype=np.float64)

    for tr_idx, val_idx in kf.split(X_f_dec):
        mf = lgb.LGBMRegressor(
            n_estimators=200, learning_rate=0.04, max_depth=5, num_leaves=24,
            reg_lambda=20.0, min_child_samples=40, subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1, verbose=-1
        ).fit(X_f_dec.iloc[tr_idx], y_f_dec[tr_idx])
        oof_f_dec[val_idx] = np.clip(mf.predict(X_f_dec.iloc[val_idx]), 0.0, None)

        mnf = lgb.LGBMRegressor(
            objective="huber", alpha=0.85, n_estimators=200, learning_rate=0.04, max_depth=5,
            num_leaves=24, reg_lambda=20.0, min_child_samples=40, subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1, verbose=-1
        ).fit(X_nf_dec.iloc[tr_idx], y_nf_dec[tr_idx])
        oof_nf_dec[val_idx] = np.clip(mnf.predict(X_nf_dec.iloc[val_idx]), 0.0, None)

    # Idiosyncratic Summer 2024 residual: positive means customer outperformed RFM expectation in Dec-Feb
    dec24_df = pd.DataFrame(index=feats_dec24.index.astype(str))
    dec24_df["has_dec24_history"] = 1.0
    dec24_df["dec24_realized_clv_fuel"] = y_f_dec
    dec24_df["dec24_realized_clv_nf"] = y_nf_dec
    dec24_df["dec24_oof_resid_fuel"] = y_f_dec - oof_f_dec
    dec24_df["dec24_oof_resid_nf"] = y_nf_dec - oof_nf_dec
    dec24_df["dec24_was_fuel_growth"] = (opp_dec == "Existing-category growth: Fuel").astype(np.float64)
    dec24_df["dec24_was_inactive"] = (opp_dec == "Inactivity").astype(np.float64)
    dec24_df["dec24_was_stable"] = (opp_dec == "Stable").astype(np.float64)

    # Also compute raw transaction summer surge for test customers
    str_test_ids = [str(x) for x in test_ids]
    id_set = set(str_test_ids)
    sub_tx = tx[
        (tx["ID"].astype(str).isin(id_set))
        & (tx["_time"] >= pd.Timestamp("2024-09-01"))
        & (tx["_time"] < pd.Timestamp("2025-12-01"))
        & (tx["is_qualifying"] == True)
        & (tx["amount"] > 0)
    ].copy()
    sub_tx["ID_str"] = sub_tx["ID"].astype(str)

    # Compare Dec24-Feb25 vs Sep24-Nov24 and Mar25-Nov25 quarterly average
    t_dec24 = pd.Timestamp("2024-12-01")
    t_mar25 = pd.Timestamp("2025-03-01")
    sub_tx["period"] = np.where(
        sub_tx["_time"] < t_dec24, "sep24_nov24",
        np.where(sub_tx["_time"] < t_mar25, "dec24_feb25", "mar25_nov25")
    )
    sub_tx["f_litres"] = np.where(sub_tx["is_fuel_sale"], np.maximum(0.0, sub_tx["qty"]), 0.0)
    sub_tx["nf_rand"] = np.where(sub_tx["is_nonfuel"], np.maximum(0.0, sub_tx["amount"]), 0.0)

    p_fuel = sub_tx.groupby(["ID_str", "period"])["f_litres"].sum().unstack(fill_value=0.0).reindex(str_test_ids, fill_value=0.0)
    p_nf = sub_tx.groupby(["ID_str", "period"])["nf_rand"].sum().unstack(fill_value=0.0).reindex(str_test_ids, fill_value=0.0)
    for col in ("sep24_nov24", "dec24_feb25", "mar25_nov25"):
        if col not in p_fuel.columns:
            p_fuel[col] = 0.0
        if col not in p_nf.columns:
            p_nf[col] = 0.0

    res = pd.DataFrame(index=pd.Index(str_test_ids, name="ID"))
    res = res.join(dec24_df).fillna(0.0)

    # Ratio of Dec24-Feb25 fuel litres to the customer's non-summer quarterly average (Mar25-Nov25 / 3)
    non_summer_q_fuel = p_fuel["mar25_nov25"].to_numpy() / 3.0
    non_summer_q_nf = p_nf["mar25_nov25"].to_numpy() / 3.0
    dec24_f = p_fuel["dec24_feb25"].to_numpy()
    sep24_f = p_fuel["sep24_nov24"].to_numpy()
    dec24_nf = p_nf["dec24_feb25"].to_numpy()
    sep24_nf = p_nf["sep24_nov24"].to_numpy()

    res["yoy_summer_vs_nonsummer_fuel_ratio"] = np.where(
        (sep24_f > 0) | (dec24_f > 0),
        (dec24_f + 10.0) / (non_summer_q_fuel + 10.0),
        1.0,
    )
    res["yoy_summer_vs_sep24_fuel_ratio"] = np.where(
        sep24_f > 0,
        (dec24_f + 5.0) / (1.25 * sep24_f + 5.0),
        1.0,
    )
    res["yoy_summer_vs_nonsummer_nf_ratio"] = np.where(
        (sep24_nf > 0) | (dec24_nf > 0),
        (dec24_nf + 20.0) / (non_summer_q_nf + 20.0),
        1.0,
    )

    logger.info(
        "YoY Summer 2024 signals extracted: %d/%d test customers have full 2024-12-01 cutoff history.",
        int(res["has_dec24_history"].sum()), len(str_test_ids)
    )
    return res


def compute_retail_physical_increment_matrix(
    tx: pd.DataFrame,
    cutoff: pd.Timestamp,
    test_ids: Sequence[str],
    label_config: dict[str, Any],
) -> np.ndarray:
    """Computes Rule 5, 6, 7 physical category increment scores across the 17 Opportunity classes
    to boost retail Hungarian matching accuracy (especially Growth-on-Return where m3 == 0)."""
    id_list = [str(x) for x in test_ids]
    id_set = set(id_list)
    mapping = label_config["category_mapping"]
    included_cats = sorted(c for c, v in mapping.items() if v["included"])
    opp_labels = list(label_config["opportunity_labels"])
    label_to_idx = {l: i for i, l in enumerate(opp_labels)}

    h_tx = tx[
        (tx["ID"].astype(str).isin(id_set))
        & (tx["_time"] < cutoff)
        & (~tx["ItemName"].isin(label_config["adjustment_items"]))
        & (tx["ItemCategoryLevel2"].isin(included_cats))
        & (tx["_quantity"] > 0)
        & (tx["_amount"] > 0)
    ].copy()
    h_tx["ID_str"] = h_tx["ID"].astype(str)
    h_tx["rand"] = h_tx["TotalExclAmount"].astype(np.float64)

    t_m1 = cutoff - pd.DateOffset(months=1)
    t_m3 = cutoff - pd.DateOffset(months=3)
    t_m6 = cutoff - pd.DateOffset(months=6)

    def spend_mat(df_sub: pd.DataFrame) -> pd.DataFrame:
        if df_sub.empty:
            return pd.DataFrame(0.0, index=id_list, columns=included_cats)
        g = df_sub.groupby(["ID_str", "ItemCategoryLevel2"])["rand"].sum().unstack(fill_value=0.0)
        return g.reindex(index=id_list, columns=included_cats, fill_value=0.0)

    ever_mat = (spend_mat(h_tx) > 0).to_numpy()
    m1_mat = spend_mat(h_tx[h_tx["_time"] >= t_m1]).to_numpy()
    m3_mat = spend_mat(h_tx[h_tx["_time"] >= t_m3]).to_numpy()
    m6_mat = spend_mat(h_tx[h_tx["_time"] >= t_m6]).to_numpy()
    q2_mat = np.maximum(0.0, m6_mat - m3_mat)

    # Expected quarterly spend per category from m1 pace and q2 baseline reversion
    exp_future_c = 0.60 * (3.0 * m1_mat) + 0.40 * q2_mat
    hurdle_grow_c = np.maximum(1.25 * m3_mat, m3_mat + 20.0)
    raw_delta_c = exp_future_c - m3_mat
    clears_ratio = exp_future_c / np.maximum(hurdle_grow_c, 20.0)

    N = len(id_list)
    K = len(opp_labels)
    phys_probs = np.full((N, K), 1e-5, dtype=np.float64)

    for c_idx, cat in enumerate(included_cats):
        if cat == "Fuel":
            continue
        g_target = mapping[cat]["Existing-category growth"]
        g_label = f"Existing-category growth: {g_target}"
        if g_label in label_to_idx:
            k_g = label_to_idx[g_label]
            score_c = np.where(
                ever_mat[:, c_idx],
                np.log1p(np.maximum(0.0, raw_delta_c[:, c_idx])) * np.clip(clears_ratio[:, c_idx], 0.2, 3.0),
                0.0,
            )
            phys_probs[:, k_g] = np.maximum(phys_probs[:, k_g], score_c)

    row_sums = phys_probs.sum(axis=1, keepdims=True)
    phys_probs = np.where(row_sums > 1e-3, phys_probs / row_sums, 1.0 / K)
    return phys_probs


def build_candidate_x(
    max_inact_swaps: int = 12,
    max_commuter_swaps: int = 14,
) -> tuple[Path, Path]:
    cfg = PipelineConfig()
    tx = load_transactions(cfg)
    test_ids = load_test_ids(cfg)
    test_ids_str = [str(x) for x in test_ids]
    N_test = len(test_ids)
    assert N_test == 5488

    submissions_dir = cfg.paths.submissions_dir
    cand_w2_files = sorted(submissions_dir.glob("submission_candidate_w2_multi_quantile_*.csv"))
    assert cand_w2_files, "Missing Candidate W2 submission!"
    cand_w2_path = cand_w2_files[-1]
    df_w2 = pd.read_csv(cand_w2_path)
    df_w2["ID"] = df_w2["ID"].astype(str)
    df_w2 = df_w2.set_index("ID").reindex(test_ids_str).reset_index()
    logger.info("Loaded active champion Candidate W2 (0.312409801): %s", cand_w2_path.name)

    builder = SnapshotBuilder(cfg, tx)
    svd_model = GlobalAlignedSVD(n_components=16, top_k_items=500, seed=42)
    svd_model.fit(tx, pd.Timestamp("2025-09-01"))

    # 1. Extract Enriched Training & Test Snapshots
    train_cutoffs = [
        pd.Timestamp("2024-09-01"),
        pd.Timestamp("2024-12-01"),
        pd.Timestamp("2025-03-01"),
        pd.Timestamp("2025-06-01"),
        pd.Timestamp("2025-09-01"),
    ]
    test_cutoff = pd.Timestamp("2025-12-01")

    X_list, lbl_list, sw_list, gt_list = [], [], [], []
    for c in train_cutoffs:
        feats, lbls = extract_enriched_snapshot(builder, tx, svd_model, c, with_labels=True)
        w = 2.0 if c.month == 12 else 1.0
        X_list.append(feats)
        lbl_list.append(lbls)
        sw_list.append(np.full(len(feats), w, dtype=np.float64))
        gt = VectorizedMonteCarloSimulator.extract_ground_truth_matrices(
            tx, c, feats.index.astype(str).tolist(), cfg.label_config
        )
        gt_list.append(gt)

    X_train_all = pd.concat(X_list, axis=0)
    sw_all = np.concatenate(sw_list)
    gt_combined = {
        k: np.concatenate([g[k] for g in gt_list], axis=0)
        for k in ("grow", "delta_grow_rand", "adopt", "spend_adopt_rand", "is_active", "h_ever")
    }

    X_test_all, _ = extract_enriched_snapshot(builder, tx, svd_model, test_cutoff, ids=test_ids_str, with_labels=False)
    num_cols = X_train_all.select_dtypes(include=[np.number]).columns.tolist()

    # 2. Extract Year-over-Year Summer 2024/2025 Fixed-Effect Signals
    yoy_df = extract_yoy_summer_2024_signals(
        tx, builder, svd_model, test_ids_str, num_cols[:168], num_cols[:168]
    )

    # 3. Apply Shrinkage-Guarded YoY Summer Fixed-Effect Correction to CLV_fuel & CLV_nonfuel
    logger.info("=== Applying Empirical Bayes YoY Summer 2024 Residual Adjustment ===")
    base_fuel = df_w2["CLV_fuel"].to_numpy().copy()
    base_nf = df_w2["CLV_nonfuel"].to_numpy().copy()

    has_yoy = yoy_df["has_dec24_history"].to_numpy() == 1.0
    m3_litres = X_test_all["m3_fuel_litres"].to_numpy()
    m1_litres = X_test_all["m1_fuel_litres"].to_numpy()
    days_last_purchase = X_test_all["days_since_last_purchase"].to_numpy()
    days_last_fuel = X_test_all["days_since_last_fuel"].to_numpy()

    # Bounded customer-specific summer residual adjustment (shrunk at eta = 0.06 to prevent outlier variance)
    yoy_resid_f = np.clip(yoy_df["dec24_oof_resid_fuel"].to_numpy(), -0.60, 0.60)
    yoy_resid_nf = np.clip(yoy_df["dec24_oof_resid_nf"].to_numpy(), -0.50, 0.50)

    adj_fuel = np.where(has_yoy & (m3_litres > 0), base_fuel + 0.06 * yoy_resid_f, base_fuel)
    adj_nf = np.where(has_yoy & (base_nf > 0), base_nf + 0.05 * yoy_resid_nf, base_nf)

    final_fuel = match_moments_zero_preserving(adj_fuel, base_fuel, TARGET_FUEL_MEAN, TARGET_FUEL_STD)
    final_nf = match_moments_zero_preserving(adj_nf, base_nf, TARGET_NF_MEAN, TARGET_NF_STD)
    final_fuel = final_fuel * (TARGET_FUEL_MEAN / float(np.mean(final_fuel)))
    final_nf = final_nf * (TARGET_NF_MEAN / float(np.mean(final_nf)))

    # 4. Train Patched 24-Category Simulator + Activity Hurdle + Stage-2 Conditional Fuel Model
    logger.info("=== Training Patched 24-Category Monte Carlo Simulator ===")
    sim = VectorizedMonteCarloSimulator(cfg.label_config, seed=42, n_draws=2500)
    sim.fit(X_train_all[num_cols], gt_combined, sample_weight=sw_all)

    ever_df_test = ever_purchased(tx, test_cutoff, pd.Index(test_ids_str, name="ID"), cfg.label_config)
    h_ever_test = ever_df_test[sim.included_cats].astype(int).values
    p_mc = sim.simulate(X_test_all[num_cols], h_ever_test, chunk_size=250)

    # Global calibrated activity probability P(Active | X) from simulator's activity classifier
    p_active_test = sim.clf_active.predict_proba(X_test_all[num_cols].fillna(0.0))[:, 1]

    # Train Stage-2 Positive-Only Conditional Fuel Regressor to get un-shrunk conditional litres
    y_fuel_train = pd.concat(lbl_list, axis=0)["CLV_fuel"].to_numpy()
    pos_f_mask = y_fuel_train > 0
    reg_cond_fuel = lgb.LGBMRegressor(
        objective="huber", alpha=0.85, n_estimators=300, learning_rate=0.04,
        max_depth=6, num_leaves=31, reg_lambda=20.0, min_child_samples=50,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1
    ).fit(X_train_all[num_cols].iloc[pos_f_mask], y_fuel_train[pos_f_mask], sample_weight=sw_all[pos_f_mask])
    pred_fuel_cond_pos = np.clip(reg_cond_fuel.predict(X_test_all[num_cols]), 0.0, None)

    # 5. Reservoir 3: 509-Retail Cohort Multi-Paradigm Hungarian Matching
    logger.info("=== Executing Reservoir 3: 509-Retail Cohort Hungarian Matching ===")
    feasibility_mask = opportunity_mask(tx, test_cutoff, pd.Index(test_ids_str, name="ID"), cfg.label_config)
    p_phys_retail = compute_retail_physical_increment_matrix(tx, test_cutoff, test_ids_str, cfg.label_config)
    combined_retail_probs = combine_probabilities(
        [p_mc, p_phys_retail], weights=[0.70, 0.30], feasibility_mask=feasibility_mask
    )

    solver = ExpectedF1AssignmentSolver(cfg.label_config)
    opp_working = np.array(
        solver.solve_retail_subset(
            df_w2["Opportunity"].values,
            combined_retail_probs,
            feasibility_mask,
            anchor_weight=0.50,
        ),
        dtype=object,
    )

    # 6. Reservoir 1: Count-Locked 1-for-1 Swaps Between Inactivity (1,443) and Stable (1,960)
    logger.info("=== Executing Reservoir 1: Count-Locked Inactivity (1,443) <-> Stable (1,960) Swaps ===")
    m1_baskets = X_test_all["m1_baskets"].to_numpy()
    m3_baskets = X_test_all["m3_baskets"].to_numpy()
    dec24_inact = yoy_df["dec24_was_inactive"].to_numpy()

    # Composite activity survival score
    survival_score = (
        0.55 * p_active_test
        + 0.25 * np.clip(final_fuel / 1.50, 0.0, 1.0)
        + 0.10 * (m1_baskets > 0).astype(np.float64)
        - 0.10 * dec24_inact
    )

    is_inact = opp_working == "Inactivity"
    is_stable = opp_working == "Stable"

    # Rescue candidates trapped in Inactivity who transacted recently in m3 and have higher survival probability
    inact_rescue_pool = np.where(
        is_inact
        & (m3_baskets >= 1)
        & (days_last_purchase <= 65.0)
        & (p_active_test >= 0.40)
    )[0]
    inact_rescue_order = inact_rescue_pool[np.argsort(-survival_score[inact_rescue_pool])]

    # Demote candidates sitting in Stable who have zero m1 visits, long recency gap, and low survival probability
    stable_churn_pool = np.where(
        is_stable
        & (m1_baskets == 0)
        & (days_last_purchase >= 60.0)
        & (p_active_test <= 0.40)
    )[0]
    stable_churn_order = stable_churn_pool[np.argsort(survival_score[stable_churn_pool])]

    n_inact_swaps = min(max_inact_swaps, len(inact_rescue_order), len(stable_churn_order))
    inact_swapped = 0
    for k in range(n_inact_swaps):
        idx_res = inact_rescue_order[k]
        idx_dem = stable_churn_order[k]
        if survival_score[idx_res] > survival_score[idx_dem] + 0.10:
            opp_working[idx_res] = "Stable"
            opp_working[idx_dem] = "Inactivity"
            inact_swapped += 1
    logger.info("Reservoir 1 complete: %d high-confidence Inactivity <-> Stable swaps executed.", inact_swapped)

    # 7. Reservoir 2: Expanded Fuel Growth (1,576) <-> Stable (1,960) Swaps (Conditional Litres + YoY Summer)
    logger.info("=== Executing Reservoir 2: Fuel Growth (1,576) <-> Stable (1,960) YoY & Conditional Swaps ===")
    ever_fuel = X_test_all["ever_fuel"].to_numpy()
    burn_rate = X_test_all["fuel_burn_rate_median"].to_numpy()
    dec24_fg = yoy_df["dec24_was_fuel_growth"].to_numpy()
    yoy_surge = yoy_df["yoy_summer_vs_nonsummer_fuel_ratio"].to_numpy()

    pred_litres_naive = np.expm1(np.clip(final_fuel * S_FUEL, 0.0, 15.0))
    pred_litres_pos = np.expm1(np.clip((0.65 * final_fuel + 0.35 * pred_fuel_cond_pos) * S_FUEL, 0.0, 15.0))
    vol_hurdle = np.maximum(1.25 * m3_litres, m3_litres + 1.0)

    vol_ratio_naive = (pred_litres_naive + 0.1) / (vol_hurdle + 0.1)
    vol_ratio_pos = (pred_litres_pos + 0.1) / (vol_hurdle + 0.1)
    m1_pace = (3.0 * m1_litres + 1.0) / (vol_hurdle + 1.0)
    burn_pace = (90.0 * burn_rate + 1.0) / (vol_hurdle + 1.0)

    fg_score = (
        0.50 * vol_ratio_pos
        + 0.20 * np.clip(m1_pace, 0.4, 1.8)
        + 0.15 * np.clip(yoy_surge, 0.6, 1.6)
        + 0.10 * dec24_fg
        + 0.05 * np.clip(burn_pace, 0.5, 1.5)
    )

    is_fg = opp_working == "Existing-category growth: Fuel"
    is_st = opp_working == "Stable"

    # Demote Pool in Fuel Growth: inactive m1==0 accounts with long fuel absence (>45d) and low conditional volume
    fg_dem_pool = np.where(
        is_fg
        & (m1_litres == 0)
        & (
            ((m3_litres == 0) & (days_last_fuel > 90.0) & (pred_litres_naive <= 2.60) & (dec24_fg == 0.0))
            | ((m3_litres > 0) & (vol_ratio_naive <= 1.15) & (days_last_fuel > 45.0) & (dec24_fg == 0.0))
        )
    )[0]
    fg_dem_order = fg_dem_pool[np.argsort(fg_score[fg_dem_pool])]

    # Promote Pool in Stable: active commuters (m1 > 0, ever_fuel == 1) with strong m1 pace or YoY summer growth
    st_prom_pool = np.where(
        is_st
        & (ever_fuel == 1)
        & (m3_litres >= 15.0)
        & (m1_litres > 0)
        & (days_last_fuel <= 30.0)
        & ((m1_pace >= 0.98) | ((dec24_fg == 1.0) & (yoy_surge >= 1.15)))
    )[0]
    st_prom_order = st_prom_pool[np.argsort(-fg_score[st_prom_pool])]

    n_fg_swaps = min(max_commuter_swaps, len(fg_dem_order), len(st_prom_order))
    fg_swapped = 0
    for k in range(n_fg_swaps):
        idx_fg = fg_dem_order[k]
        idx_st = st_prom_order[k]
        if fg_score[idx_st] > fg_score[idx_fg] + 0.05:
            opp_working[idx_fg] = "Stable"
            opp_working[idx_st] = "Existing-category growth: Fuel"
            fg_swapped += 1
    logger.info("Reservoir 2 complete: %d high-confidence Fuel Growth <-> Stable swaps executed.", fg_swapped)

    # 8. Strict Verification of Invariants
    # Version 1: Candidate X1 (Pure 3-Reservoir Opportunity Lift, regression locked 100% to Champion W2)
    sub_x1 = pd.DataFrame({
        "ID": test_ids_str,
        "CLV_fuel": np.clip(df_w2["CLV_fuel"].to_numpy(), 0.0, None),
        "CLV_nonfuel": np.clip(df_w2["CLV_nonfuel"].to_numpy(), 0.0, None),
        "Opportunity": opp_working,
    })

    # Version 2: Candidate X2 (Full compound: YoY Empirical Bayes Summer Adjustment + 3-Reservoir Opportunity)
    sub_x2 = pd.DataFrame({
        "ID": test_ids_str,
        "CLV_fuel": np.clip(final_fuel, 0.0, None),
        "CLV_nonfuel": np.clip(final_nf, 0.0, None),
        "Opportunity": opp_working,
    })

    for name, df_sub in [("Candidate X1 (opp_only)", sub_x1), ("Candidate X2 (full_compound)", sub_x2)]:
        assert len(df_sub) == 5488
        assert not df_sub.isna().any().any()
        counts = df_sub["Opportunity"].value_counts().to_dict()
        logger.info("%s Opportunity counts: %s", name, counts)
        assert counts["Stable"] == 1960
        assert counts["Existing-category growth: Fuel"] == 1576
        assert counts["Inactivity"] == 1443
        assert counts["Existing-category growth: Other"] == 237
        assert counts["Existing-category growth: Beverages"] == 130
        assert counts["Existing-category growth: Food & beverage"] == 98
        assert counts.get("New category adoption: Food & beverage", 0) == 17
        assert counts.get("New category adoption: Other", 0) == 17
        assert counts.get("New category adoption: Beverages", 0) == 6
        assert counts.get("New category adoption: Lubricants", 0) == 3
        assert counts.get("New category adoption: Confectionary", 0) == 1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_x1 = submissions_dir / f"submission_candidate_x1_opp_3reservoir_{timestamp}.csv"
    out_x2 = submissions_dir / f"submission_candidate_x2_full_yoy_3reservoir_{timestamp}.csv"

    sub_x1.to_csv(out_x1, index=False)
    sub_x2.to_csv(out_x2, index=False)

    # Stage X1 to root submission.csv by default (pure Opportunity isolation on top of verified W2 regression)
    sub_x1.to_csv(PROJECT_ROOT / "submission.csv", index=False)

    sha256_x1 = hashlib.sha256(out_x1.read_bytes()).hexdigest()
    sha256_x2 = hashlib.sha256(out_x2.read_bytes()).hexdigest()

    logger.info("Candidate X1 (opp only) written to %s (SHA-256: %s)", out_x1.name, sha256_x1)
    logger.info("Candidate X2 (full compound) written to %s (SHA-256: %s)", out_x2.name, sha256_x2)
    logger.info("Staged Candidate X1 to root submission.csv.")

    # Write manifests
    manifest_x1 = {
        "candidate": "X1_opp_3reservoir",
        "timestamp": timestamp,
        "base_candidate": cand_w2_path.name,
        "swaps": {
            "inactivity_swaps": inact_swapped,
            "fuel_growth_swaps": fg_swapped,
        },
        "fuel_mean": float(sub_x1["CLV_fuel"].mean()),
        "nonfuel_mean": float(sub_x1["CLV_nonfuel"].mean()),
        "sha256": sha256_x1,
    }
    manifest_x2 = {
        "candidate": "X2_full_yoy_3reservoir",
        "timestamp": timestamp,
        "base_candidate": cand_w2_path.name,
        "swaps": {
            "inactivity_swaps": inact_swapped,
            "fuel_growth_swaps": fg_swapped,
        },
        "fuel_mean": float(sub_x2["CLV_fuel"].mean()),
        "nonfuel_mean": float(sub_x2["CLV_nonfuel"].mean()),
        "sha256": sha256_x2,
    }
    (submissions_dir / f"submission_candidate_x1_opp_3reservoir_{timestamp}.manifest.json").write_text(json.dumps(manifest_x1, indent=2))
    (submissions_dir / f"submission_candidate_x2_full_yoy_3reservoir_{timestamp}.manifest.json").write_text(json.dumps(manifest_x2, indent=2))

    return out_x1, out_x2


if __name__ == "__main__":
    build_candidate_x()
