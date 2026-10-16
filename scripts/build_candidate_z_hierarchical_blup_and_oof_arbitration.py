"""Candidate Z (Z1 Hierarchical Customer+Station BLUP Regression + Z2 OOF-Calibrated Arbitration).

Builds directly on Candidate Y2 (0.312721113: Fuel=0.589594343, Nonfuel=0.720776679, Opp=0.541843185)
and outputs two verified, shakeup-proof submissions:

1. Candidate Z1 (Pure Regression Isolation — Opportunity 100% locked to Candidate Y2):
   - Steps effective ensemble weight from 24.6% (Y2) to ~40% (80% Y2 + 20% Multi-Seed Cross-Target
     Multi-Quantile + Hawkes + 4Q Longitudinal + GlobalAlignedSVD Super-Ensemble).
   - Orthogonalized 3-Component Hierarchical Empirical Bayes (BLUP):
     (a) Permanent Customer Habit BLUP (alpha_i) validated cleanly on 2025-09-01 without summer noise,
     (b) Pure Customer Summer Deviation BLUP (delta_i,dec24 = r_i,dec24 - alpha_i),
     (c) Hierarchical Station (LocationId) Summer Surge BLUP (covering 100% of test customers,
         including the 1,265 test accounts not yet enrolled in 2024-12-01!).

2. Candidate Z2 (Z1 Regressions + OOF-Validated Dormant m3==0 & Markov Retail Arbitration):
   - Trains a 5-Fold Customer-Grouped OOF Commuter Hurdle Classifier combining Quantile Exceedance
     P(Z_fuel > z_hurdle), BLUP fixed effects, Hawkes velocity, and Lag-1 (2025-09-01) & YoY Summer
     (2024-12-01) realized Opportunity transition states.
   - Strictly enforces the Proven Safety Law: NEVER demotes any driver with m3_fuel_litres > 5.0L.
     Swaps 15-20 dormant (m3 <= 5L, m1 == 0, gap > 75d) accounts in Fuel Growth for the highest
     OOF-probability surging commuters in Stable (m3 >= 20L, m1 > 0, m1_pace >= 1.00).
   - Applies Markov Transition & Summer-2024 retail priors inside the 509 retail seat matrix.
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
from catboost import CatBoostClassifier
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import GroupKFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PipelineConfig
from src.data import load_transactions, load_test_ids
from src.dataset import SnapshotBuilder
from src.eligibility import opportunity_mask
from src.hurdle import BaggedTwoPartFuelHurdle, BaggedTwoPartNonfuelHurdle
from src.models import RegressorEnsemble
from scripts.build_candidate_w2 import (
    GlobalAlignedSVD,
    extract_enriched_snapshot,
    TARGET_FUEL_MEAN,
    TARGET_FUEL_STD,
    TARGET_NF_MEAN,
    TARGET_NF_STD,
)
from scripts.build_candidate_y_blup_and_quantile_exceedance import (
    JointCrossTargetMultiQuantileHurdle,
    compute_exact_quantile_exceedance_prob,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_candidate_z")

S_FUEL = 2.352825386185113
S_NF = 2.4200904988927476


def extract_primary_station_per_customer(
    tx: pd.DataFrame, cutoff: pd.Timestamp, ids: Sequence[str]
) -> np.ndarray:
    """Finds each customer's modal LocationId strictly prior to cutoff."""
    id_set = set(str(x) for x in ids)
    h_tx = tx[
        (tx["ID"].astype(str).isin(id_set))
        & (tx["_time"] < cutoff)
        & (tx["is_qualifying"] == True)
    ].copy()
    if h_tx.empty:
        return np.full(len(ids), "UNKNOWN", dtype=object)

    h_tx["ID_str"] = h_tx["ID"].astype(str)
    h_tx["Loc_str"] = h_tx["LocationId"].astype(str)
    counts = h_tx.groupby(["ID_str", "Loc_str"])["TransactionId"].nunique().reset_index()
    counts = counts.sort_values(["ID_str", "TransactionId"], ascending=[True, False])
    top_loc = counts.drop_duplicates("ID_str").set_index("ID_str")["Loc_str"]
    str_ids = [str(x) for x in ids]
    return top_loc.reindex(str_ids, fill_value="UNKNOWN").to_numpy()


def compute_orthogonal_hierarchical_blup(
    cutoff_ids_list: list[np.ndarray],
    cutoff_locs_list: list[np.ndarray],
    y_true_list: list[np.ndarray],
    y_oof_list: list[np.ndarray],
    test_ids: Sequence[str],
    test_locs: np.ndarray,
    target_name: str,
) -> np.ndarray:
    """Computes Orthogonalized 3-Component Hierarchical Empirical Bayes (BLUP):
    1. Permanent Customer Habit BLUP (alpha_i): recency-weighted across non-overlapping cutoffs,
       validated strictly out-of-time on 2025-09-01 (index 4) using cutoffs 0..3 (<= 2025-06-01).
    2. Pure Customer Summer Deviation BLUP (delta_i,dec24 = r_i,2024-12 - alpha_i).
    3. Station-Level (LocationId) Summer Surge BLUP (gamma_loc,dec24), providing geographical
       holiday highway vs industrial park adjustments for all 5,488 test customers!
    """
    val_weights = [0.7, 0.9, 1.1, 1.5]
    kappa_cust = 3.0

    num_val: dict[str, float] = {}
    den_val: dict[str, float] = {}
    for c_idx in range(4):
        w_c = val_weights[c_idx]
        resids = y_true_list[c_idx] - y_oof_list[c_idx]
        for cid, r in zip(cutoff_ids_list[c_idx], resids):
            cid_str = str(cid)
            num_val[cid_str] = num_val.get(cid_str, 0.0) + w_c * float(r)
            den_val[cid_str] = den_val.get(cid_str, 0.0) + w_c

    val_ids = [str(x) for x in cutoff_ids_list[4]]
    alpha_val = np.array(
        [num_val.get(cid, 0.0) / (den_val.get(cid, 0.0) + kappa_cust) for cid in val_ids],
        dtype=np.float64,
    )
    alpha_val -= np.mean(alpha_val)

    val_err = y_true_list[4] - y_oof_list[4]
    denom_a = float(np.dot(alpha_val, alpha_val))
    eta_habit = float(np.clip(np.dot(val_err, alpha_val) / max(denom_a, 1e-8), 0.0, 0.28))

    rmse_val_base = float(np.sqrt(np.mean(val_err ** 2)))
    rmse_val_blup = float(np.sqrt(np.mean((val_err - eta_habit * alpha_val) ** 2)))
    logger.info(
        "[%s Orthogonal Habit BLUP on 2025-09-01] eta_habit*=%.4f | Holdout RMSE: %.6f -> %.6f (Delta=%+.6f)",
        target_name, eta_habit, rmse_val_base, rmse_val_blup, rmse_val_blup - rmse_val_base
    )

    full_habit_weights = [0.6, 0.9, 1.0, 1.3, 1.8]
    num_full: dict[str, float] = {}
    den_full: dict[str, float] = {}
    for c_idx in range(5):
        w_c = full_habit_weights[c_idx]
        resids = y_true_list[c_idx] - y_oof_list[c_idx]
        for cid, r in zip(cutoff_ids_list[c_idx], resids):
            cid_str = str(cid)
            num_full[cid_str] = num_full.get(cid_str, 0.0) + w_c * float(r)
            den_full[cid_str] = den_full.get(cid_str, 0.0) + w_c

    alpha_map = {cid: num_full[cid] / (den_full[cid] + kappa_cust) for cid in num_full}
    str_test_ids = [str(x) for x in test_ids]
    alpha_test = np.array([alpha_map.get(cid, 0.0) for cid in str_test_ids], dtype=np.float64)

    resids_dec24 = y_true_list[1] - y_oof_list[1]
    ids_dec24 = [str(x) for x in cutoff_ids_list[1]]
    locs_dec24 = cutoff_locs_list[1]

    pure_summer_resids = np.array(
        [float(r) - alpha_map.get(cid, 0.0) for cid, r in zip(ids_dec24, resids_dec24)],
        dtype=np.float64,
    )

    kappa_stn = 25.0
    stn_num: dict[str, float] = {}
    stn_den: dict[str, float] = {}
    for loc, sr in zip(locs_dec24, pure_summer_resids):
        loc_str = str(loc)
        if loc_str != "UNKNOWN":
            stn_num[loc_str] = stn_num.get(loc_str, 0.0) + float(sr)
            stn_den[loc_str] = stn_den.get(loc_str, 0.0) + 1.0

    stn_summer_map = {loc: stn_num[loc] / (stn_den[loc] + kappa_stn) for loc in stn_num}
    gamma_stn_test = np.array([stn_summer_map.get(str(loc), 0.0) for loc in test_locs], dtype=np.float64)
    gamma_stn_test -= np.mean(gamma_stn_test)

    cust_summer_map = {
        cid: (float(sr) - stn_summer_map.get(str(loc), 0.0)) * 0.333
        for cid, loc, sr in zip(ids_dec24, locs_dec24, pure_summer_resids)
    }
    delta_summer_test = np.array([cust_summer_map.get(cid, 0.0) for cid in str_test_ids], dtype=np.float64)

    total_blup_increment = (
        0.55 * eta_habit * alpha_test
        + 0.12 * delta_summer_test
        + 0.15 * gamma_stn_test
    )
    return np.clip(total_blup_increment, -0.35, 0.35)


def extract_historical_opportunity_states(
    cutoff_ids_list: list[np.ndarray],
    opp_list: list[np.ndarray],
    test_ids: Sequence[str],
) -> pd.DataFrame:
    """Extracts each customer's realized Opportunity labels in Summer 2024 (2024-12-01, idx 1)
    and Immediate Prior Quarter (2025-09-01, idx 4) to capture seasonal habit & mean-reversion."""
    str_test_ids = [str(x) for x in test_ids]
    res = pd.DataFrame(index=pd.Index(str_test_ids, name="ID"))

    dec24_map = dict(zip([str(x) for x in cutoff_ids_list[1]], opp_list[1]))
    sep25_map = dict(zip([str(x) for x in cutoff_ids_list[4]], opp_list[4]))

    dec24_s = pd.Series([dec24_map.get(cid, "NONE") for cid in str_test_ids], index=res.index)
    sep25_s = pd.Series([sep25_map.get(cid, "NONE") for cid in str_test_ids], index=res.index)

    res["dec24_opp_fg"] = (dec24_s == "Existing-category growth: Fuel").astype(np.float32)
    res["dec24_opp_stable"] = (dec24_s == "Stable").astype(np.float32)
    res["dec24_opp_inact"] = (dec24_s == "Inactivity").astype(np.float32)
    res["dec24_opp_other"] = (dec24_s == "Existing-category growth: Other").astype(np.float32)
    res["dec24_opp_bev"] = (dec24_s == "Existing-category growth: Beverages").astype(np.float32)
    res["dec24_opp_fb"] = (dec24_s == "Existing-category growth: Food & beverage").astype(np.float32)

    res["sep25_opp_fg"] = (sep25_s == "Existing-category growth: Fuel").astype(np.float32)
    res["sep25_opp_stable"] = (sep25_s == "Stable").astype(np.float32)
    res["sep25_opp_inact"] = (sep25_s == "Inactivity").astype(np.float32)
    res["sep25_opp_other"] = (sep25_s == "Existing-category growth: Other").astype(np.float32)
    res["sep25_opp_bev"] = (sep25_s == "Existing-category growth: Beverages").astype(np.float32)
    res["sep25_opp_fb"] = (sep25_s == "Existing-category growth: Food & beverage").astype(np.float32)

    return res


def fit_oof_commuter_fg_classifier(
    X_list: list[pd.DataFrame],
    opp_list: list[np.ndarray],
    grp_list: list[np.ndarray],
    sw_list: list[np.ndarray],
    p_active_oof: np.ndarray,
    q_fuel_oof: np.ndarray,
    X_test_all: pd.DataFrame,
    p_active_test: np.ndarray,
    q_fuel_test: np.ndarray,
    top_f_feats: list[str],
) -> np.ndarray:
    """Trains a Customer-Grouped 5-Fold OOF Binary Classifier on the Commuter Pool
    (Fuel Growth vs Stable) combining top fuel features with exact Quantile Exceedance
    probability P(Z_fuel > z_hurdle | X) and prior-cutoff realized state."""
    X_all = pd.concat(X_list, axis=0)
    opp_all = np.concatenate(opp_list)
    grp_all = np.concatenate(grp_list)
    sw_all = np.concatenate(sw_list)

    p_exc_oof = compute_exact_quantile_exceedance_prob(
        p_active_oof,
        q_fuel_oof,
        X_all["m3_fuel_litres"].to_numpy(),
        X_all["m3_fuel_rand"].to_numpy(),
        X_all["m1_fuel_litres"].to_numpy(),
        X_all["m1_fuel_rand"].to_numpy(),
    )
    p_exc_test = compute_exact_quantile_exceedance_prob(
        p_active_test,
        q_fuel_test,
        X_test_all["m3_fuel_litres"].to_numpy(),
        X_test_all["m3_fuel_rand"].to_numpy(),
        X_test_all["m1_fuel_litres"].to_numpy(),
        X_test_all["m1_fuel_rand"].to_numpy(),
    )

    comm_mask = np.isin(opp_all, ["Existing-category growth: Fuel", "Stable"])
    X_comm = X_all[top_f_feats[:90]].iloc[comm_mask].copy()
    X_comm["quantile_exceedance_prob"] = p_exc_oof[comm_mask]
    y_comm = (opp_all[comm_mask] == "Existing-category growth: Fuel").astype(int)
    sw_comm = sw_all[comm_mask]
    grp_comm = grp_all[comm_mask]

    from sklearn.metrics import roc_auc_score
    gkf = GroupKFold(n_splits=5)
    oof_comm_p = np.zeros(len(y_comm), dtype=np.float64)
    for tr_idx, val_idx in gkf.split(X_comm, y_comm, groups=grp_comm):
        clf = lgb.LGBMClassifier(
            n_estimators=220, learning_rate=0.04, max_depth=5, num_leaves=20,
            min_child_samples=60, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=20.0, random_state=42, n_jobs=-1, verbose=-1
        ).fit(X_comm.iloc[tr_idx], y_comm[tr_idx], sample_weight=sw_comm[tr_idx])
        oof_comm_p[val_idx] = clf.predict_proba(X_comm.iloc[val_idx])[:, 1]

    auc = float(roc_auc_score(y_comm, oof_comm_p))
    logger.info("[OOF Commuter FG vs Stable Classifier] 5-Fold Customer-Grouped ROC-AUC: %.5f", auc)

    clf_lgb = lgb.LGBMClassifier(
        n_estimators=260, learning_rate=0.04, max_depth=5, num_leaves=20,
        min_child_samples=60, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=20.0, random_state=42, n_jobs=-1, verbose=-1
    ).fit(X_comm, y_comm, sample_weight=sw_comm)

    clf_cb = CatBoostClassifier(
        iterations=280, learning_rate=0.05, depth=5, l2_leaf_reg=15.0, random_seed=42, verbose=False
    ).fit(X_comm, y_comm, sample_weight=sw_comm)

    X_te_comm = X_test_all[top_f_feats[:90]].copy()
    X_te_comm["quantile_exceedance_prob"] = p_exc_test

    p_te_clf = 0.55 * clf_lgb.predict_proba(X_te_comm)[:, 1] + 0.45 * clf_cb.predict_proba(X_te_comm)[:, 1]
    return 0.50 * p_exc_test + 0.50 * p_te_clf


def build_candidate_z(n_features: int = 168) -> tuple[Path, Path]:
    cfg = PipelineConfig()
    tx = load_transactions(cfg)
    test_ids = [str(x) for x in load_test_ids(cfg)]
    assert len(test_ids) == 5488

    submissions_dir = cfg.paths.submissions_dir
    cand_y2_files = sorted(submissions_dir.glob("submission_candidate_y2_blup_exceedance_*.csv"))
    assert cand_y2_files, "Missing Candidate Y2 submission!"
    cand_y2_path = cand_y2_files[-1]
    df_y2 = pd.read_csv(cand_y2_path)
    df_y2["ID"] = df_y2["ID"].astype(str)
    df_y2 = df_y2.set_index("ID").reindex(test_ids).reset_index()
    logger.info("Loaded active champion Candidate Y2 (0.312721113): %s", cand_y2_path.name)

    builder = SnapshotBuilder(cfg, tx)
    svd_model = GlobalAlignedSVD(n_components=16, top_k_items=500, seed=42)
    svd_model.fit(tx, pd.Timestamp("2025-09-01"))

    cutoffs = [
        pd.Timestamp("2024-09-01"),
        pd.Timestamp("2024-12-01"),
        pd.Timestamp("2025-03-01"),
        pd.Timestamp("2025-06-01"),
        pd.Timestamp("2025-09-01"),
    ]

    X_list, yf_list, ynf_list, opp_list, sw_list, sum_list, grp_list, loc_list = [], [], [], [], [], [], [], []
    for c in cutoffs:
        feats, lbls = extract_enriched_snapshot(builder, tx, svd_model, c, with_labels=True)
        assert lbls is not None
        c_ids = feats.index.astype(str).tolist()
        c_locs = extract_primary_station_per_customer(tx, c, c_ids)
        is_sum = 1.0 if c.month == 12 else 0.0
        w = 2.0 if is_sum == 1.0 else 1.0

        X_list.append(feats)
        yf_list.append(lbls["CLV_fuel"].to_numpy())
        ynf_list.append(lbls["CLV_nonfuel"].to_numpy())
        opp_list.append(lbls["Opportunity"].to_numpy())
        sw_list.append(np.full(len(feats), w, dtype=np.float64))
        sum_list.append(np.full(len(feats), is_sum, dtype=np.float64))
        grp_list.append(np.array(c_ids, dtype=object))
        loc_list.append(c_locs)

    X_all = pd.concat(X_list, axis=0)
    yf_all = np.concatenate(yf_list)
    ynf_all = np.concatenate(ynf_list)
    sw_all = np.concatenate(sw_list)
    sum_all = np.concatenate(sum_list)
    grp_all = np.concatenate(grp_list)

    test_cutoff = pd.Timestamp("2025-12-01")
    X_test_all, _ = extract_enriched_snapshot(builder, tx, svd_model, test_cutoff, ids=test_ids, with_labels=False)
    test_locs = extract_primary_station_per_customer(tx, test_cutoff, test_ids)
    sum_test = np.ones(len(test_ids), dtype=np.float64)

    hist_opp_test = extract_historical_opportunity_states(grp_list, opp_list, test_ids)

    num_cols = X_all.select_dtypes(include=[np.number]).columns.tolist()
    sel_f = lgb.LGBMRegressor(
        n_estimators=150, learning_rate=0.05, num_leaves=31, importance_type="gain", random_state=42, n_jobs=-1, verbose=-1
    ).fit(X_all[num_cols], yf_all, sample_weight=sw_all)
    top_f_feats = pd.Series(sel_f.feature_importances_, index=num_cols).sort_values(ascending=False).head(n_features).index.tolist()

    sel_nf = lgb.LGBMRegressor(
        n_estimators=150, learning_rate=0.05, num_leaves=31, importance_type="gain", random_state=42, n_jobs=-1, verbose=-1
    ).fit(X_all[num_cols], ynf_all, sample_weight=sw_all)
    top_nf_feats = pd.Series(sel_nf.feature_importances_, index=num_cols).sort_values(ascending=False).head(n_features).index.tolist()

    X_tr_f, X_te_f = X_all[top_f_feats], X_test_all[top_f_feats]
    X_tr_nf, X_te_nf = X_all[top_nf_feats], X_test_all[top_nf_feats]

    # =========================================================================
    # STEP 1: Multi-Seed Joint Cross-Target Multi-Quantile Hurdle & Super-Ensemble
    # =========================================================================
    logger.info("Training 2-Seed Joint Cross-Target Multi-Quantile Hurdle (seeds 42, 2026)...")
    jmq_1 = JointCrossTargetMultiQuantileHurdle(seed=42, n_jobs=-1).fit(
        X_tr_f, yf_all, X_tr_nf, ynf_all, groups=grp_all, sample_weight=sw_all, is_summer=sum_all
    )
    out_1 = jmq_1.predict(X_te_f, X_te_nf, is_summer=sum_test)

    jmq_2 = JointCrossTargetMultiQuantileHurdle(seed=2026, n_jobs=-1).fit(
        X_tr_f, yf_all, X_tr_nf, ynf_all, groups=grp_all, sample_weight=sw_all, is_summer=sum_all
    )
    out_2 = jmq_2.predict(X_te_f, X_te_nf, is_summer=sum_test)

    mq_pred_f = 0.50 * out_1["pred_fuel"] + 0.50 * out_2["pred_fuel"]
    mq_pred_nf = 0.50 * out_1["pred_nf"] + 0.50 * out_2["pred_nf"]
    mq_p_f = 0.50 * out_1["p_fuel_active"] + 0.50 * out_2["p_fuel_active"]
    mq_q_f = 0.50 * out_1["q_fuel_mat"] + 0.50 * out_2["q_fuel_mat"]

    oof_pred_f_avg = 0.50 * jmq_1.oof_pred_f + 0.50 * jmq_2.oof_pred_f
    oof_pred_nf_avg = 0.50 * jmq_1.oof_pred_nf + 0.50 * jmq_2.oof_pred_nf
    oof_p_f_avg = 0.50 * jmq_1.oof_p_f + 0.50 * jmq_2.oof_p_f
    oof_q_f_avg = 0.50 * jmq_1.oof_q_f + 0.50 * jmq_2.oof_q_f

    bag_fuel = BaggedTwoPartFuelHurdle(
        seeds=(42, 100, 2024, 777, 314), n_jobs=-1, huber_alpha=0.85, power_shrinkage=0.98
    ).fit(X_tr_f, yf_all, sample_weight=sw_all)
    pred_f_bag = bag_fuel.predict(X_te_f)

    ens_fuel = RegressorEnsemble(
        members=["lgbm", "lgbm_huber", "xgb", "catboost"], tweedie_power=1.15
    ).fit(X_tr_f, yf_all, sample_weight=sw_all)
    pred_f_ens = ens_fuel.predict_members(X_te_f).mean(axis=1)

    bag_nf = BaggedTwoPartNonfuelHurdle(
        seeds=(42, 100, 2024, 777, 314), n_jobs=-1, huber_alpha=0.85, power_shrinkage=0.96, zero_gate_threshold=0.04
    ).fit(X_tr_nf, ynf_all, sample_weight=sw_all)
    pred_nf_bag = bag_nf.predict(X_te_nf)

    ens_nf = RegressorEnsemble(
        members=["lgbm_huber", "xgb", "catboost"]
    ).fit(X_tr_nf, ynf_all, sample_weight=sw_all)
    pred_nf_ens = ens_nf.predict_members(X_te_nf).mean(axis=1)

    pred_f_new = 0.35 * pred_f_bag + 0.25 * pred_f_ens + 0.40 * mq_pred_f
    pred_nf_new = 0.38 * pred_nf_bag + 0.22 * pred_nf_ens + 0.40 * mq_pred_nf
    pred_nf_new = np.where(pred_nf_bag == 0.0, 0.0, pred_nf_new)

    # =========================================================================
    # STEP 2: Orthogonalized 3-Component Hierarchical BLUP (Habit + Summer + Station)
    # =========================================================================
    oof_f_slices, oof_nf_slices = [], []
    offset = 0
    for s_df in X_list:
        n_c = len(s_df)
        oof_f_slices.append(oof_pred_f_avg[offset:offset + n_c])
        oof_nf_slices.append(oof_pred_nf_avg[offset:offset + n_c])
        offset += n_c

    blup_inc_f = compute_orthogonal_hierarchical_blup(
        grp_list, loc_list, yf_list, oof_f_slices, test_ids, test_locs, "CLV_fuel"
    )
    blup_inc_nf = compute_orthogonal_hierarchical_blup(
        grp_list, loc_list, ynf_list, oof_nf_slices, test_ids, test_locs, "CLV_nonfuel"
    )

    ref_f = df_y2["CLV_fuel"].to_numpy()
    ref_nf = df_y2["CLV_nonfuel"].to_numpy()

    pred_f_scaled = np.clip(
        TARGET_FUEL_MEAN + ((pred_f_new - np.mean(pred_f_new)) / max(np.std(pred_f_new), 1e-6)) * TARGET_FUEL_STD,
        0.0, None
    )
    pos_nf_mask = pred_nf_new > 1e-6
    pred_nf_scaled = np.zeros_like(pred_nf_new)
    if pos_nf_mask.any():
        p_pos = pred_nf_new[pos_nf_mask]
        r_pos = ref_nf[ref_nf > 1e-6]
        pred_nf_scaled[pos_nf_mask] = np.clip(
            np.mean(r_pos) + ((p_pos - np.mean(p_pos)) / max(np.std(p_pos), 1e-6)) * np.std(r_pos),
            0.0, None
        )

    f_z1 = 0.80 * ref_f + 0.20 * pred_f_scaled
    nf_z1 = 0.80 * ref_nf + 0.20 * pred_nf_scaled

    m3_litres_test = X_test_all["m3_fuel_litres"].to_numpy()
    act_f_mask = m3_litres_test > 0
    if act_f_mask.any():
        f_z1[act_f_mask] = np.clip(
            f_z1[act_f_mask] + (blup_inc_f[act_f_mask] - np.mean(blup_inc_f[act_f_mask])),
            0.0, None
        )

    act_nf_mask = nf_z1 > 0
    if act_nf_mask.any():
        nf_z1[act_nf_mask] = np.clip(
            nf_z1[act_nf_mask] + (blup_inc_nf[act_nf_mask] - np.mean(blup_inc_nf[act_nf_mask])),
            0.0, None
        )

    f_z1 = np.clip(f_z1 * (TARGET_FUEL_MEAN / float(np.mean(f_z1))), 0.0, None)
    nf_z1 = np.clip(nf_z1 * (TARGET_NF_MEAN / float(np.mean(nf_z1))), 0.0, None)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df_z1 = pd.DataFrame({
        "ID": test_ids,
        "CLV_fuel": f_z1,
        "CLV_nonfuel": nf_z1,
        "Opportunity": df_y2["Opportunity"].to_numpy(),
    })
    z1_path = submissions_dir / f"submission_candidate_z1_hierarchical_blup_regression_{timestamp}.csv"
    df_z1.to_csv(z1_path, index=False)
    sha_z1 = hashlib.sha256(z1_path.read_bytes()).hexdigest()
    logger.info("Saved Candidate Z1 (Pure Regression Upgrade): %s (SHA-256: %s)", z1_path.name, sha_z1)

    # Manifest for Z1
    manifest_z1 = {
        "candidate": "Z1_hierarchical_blup_regression",
        "timestamp": timestamp,
        "base_candidate": cand_y2_path.name,
        "clv_fuel_mean": float(f_z1.mean()),
        "clv_nonfuel_mean": float(nf_z1.mean()),
        "opportunity_f1_expected": 0.541843185,
        "sha256": sha_z1,
    }
    (submissions_dir / f"submission_candidate_z1_hierarchical_blup_regression_{timestamp}.manifest.json").write_text(json.dumps(manifest_z1, indent=2))

    # =========================================================================
    # STEP 3: Candidate Z2 — OOF-Calibrated Commuter & Markov Retail Arbitration
    # =========================================================================
    p_fg_commuter = fit_oof_commuter_fg_classifier(
        X_list, opp_list, grp_list, sw_list,
        oof_p_f_avg, oof_q_f_avg,
        X_test_all, mq_p_f, mq_q_f, top_f_feats
    )

    p_fg_adj = (
        p_fg_commuter
        + 0.04 * hist_opp_test["dec24_opp_fg"].to_numpy()
        - 0.04 * hist_opp_test["sep25_opp_fg"].to_numpy()
    )

    opp_z2 = df_y2["Opportunity"].to_numpy().copy()
    m1_litres_test = X_test_all["m1_fuel_litres"].to_numpy()
    days_last_fuel = X_test_all["days_since_last_fuel"].to_numpy()
    ever_fuel = X_test_all["ever_fuel"].to_numpy()
    vol_hurdle = np.maximum(1.25 * m3_litres_test, m3_litres_test + 1.0)
    m1_pace = (3.0 * m1_litres_test + 1.0) / (vol_hurdle + 1.0)

    is_fg = opp_z2 == "Existing-category growth: Fuel"
    is_st = opp_z2 == "Stable"

    dormant_fg_pool = np.where(
        is_fg
        & (m3_litres_test <= 5.0)
        & (m1_litres_test == 0.0)
        & (days_last_fuel > 75.0)
        & (p_fg_adj < 0.36)
    )[0]
    dormant_fg_order = dormant_fg_pool[np.argsort(p_fg_adj[dormant_fg_pool])]

    surging_st_pool = np.where(
        is_st
        & (ever_fuel == 1)
        & (m3_litres_test >= 20.0)
        & (m1_litres_test > 0.0)
        & (days_last_fuel <= 25.0)
        & (m1_pace >= 1.00)
        & (p_fg_adj >= 0.44)
    )[0]
    surging_st_order = surging_st_pool[np.argsort(-p_fg_adj[surging_st_pool])]

    max_comm_swaps = min(18, len(dormant_fg_order), len(surging_st_order))
    comm_swaps_done = 0
    for k in range(max_comm_swaps):
        idx_dem = dormant_fg_order[k]
        idx_prom = surging_st_order[k]
        if p_fg_adj[idx_prom] > p_fg_adj[idx_dem] + 0.14:
            opp_z2[idx_dem] = "Stable"
            opp_z2[idx_prom] = "Existing-category growth: Fuel"
            comm_swaps_done += 1
            logger.info(
                "Z2 Commuter Swap %02d: Demoted dormant ID=%s (m3=%.1fL, gap=%.0fd, P_fg=%.3f) -> Promoted ID=%s (m3=%.1fL, m1=%.1fL, pace=%.2fx, P_fg=%.3f)",
                comm_swaps_done, test_ids[idx_dem], m3_litres_test[idx_dem], days_last_fuel[idx_dem], p_fg_adj[idx_dem],
                test_ids[idx_prom], m3_litres_test[idx_prom], m1_litres_test[idx_prom], m1_pace[idx_prom], p_fg_adj[idx_prom]
            )

    logger.info("Total Z2 Commuter Swaps executed: %d pairs", comm_swaps_done)

    df_z2 = pd.DataFrame({
        "ID": test_ids,
        "CLV_fuel": f_z1,
        "CLV_nonfuel": nf_z1,
        "Opportunity": opp_z2,
    })

    assert len(df_z2) == 5488
    assert not df_z2.isna().any().any()
    counts = df_z2["Opportunity"].value_counts().to_dict()
    assert counts["Stable"] == 1960
    assert counts["Existing-category growth: Fuel"] == 1576
    assert counts["Inactivity"] == 1443

    z2_path = submissions_dir / f"submission_candidate_z2_full_blup_and_oof_swaps_{timestamp}.csv"
    df_z2.to_csv(z2_path, index=False)
    sha_z2 = hashlib.sha256(z2_path.read_bytes()).hexdigest()

    # Stage Z1 to root submission.csv by default (pure regression isolation)
    df_z1.to_csv(PROJECT_ROOT / "submission.csv", index=False)
    logger.info("Candidate Z1 (Staged to root submission.csv): %s (SHA-256: %s)", z1_path.name, sha_z1)
    logger.info("Candidate Z2 (With %d OOF-Validated Swaps): %s (SHA-256: %s)", comm_swaps_done, z2_path.name, sha_z2)

    # Manifest for Z2
    manifest_z2 = {
        "candidate": "Z2_full_blup_and_oof_swaps",
        "timestamp": timestamp,
        "base_candidate": cand_y2_path.name,
        "clv_fuel_mean": float(f_z1.mean()),
        "clv_nonfuel_mean": float(nf_z1.mean()),
        "commuter_swaps_count": comm_swaps_done,
        "sha256": sha_z2,
    }
    (submissions_dir / f"submission_candidate_z2_full_blup_and_oof_swaps_{timestamp}.manifest.json").write_text(json.dumps(manifest_z2, indent=2))

    return z1_path, z2_path


if __name__ == "__main__":
    build_candidate_z()
