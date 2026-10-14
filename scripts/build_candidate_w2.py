"""Candidate W2: Leak-Free OOF Multi-Quantile Hurdle + Target-Specific Pruned Super-Ensemble.

Upgrades Candidate W and Module 2 (regression_v3.py) by:
1. Extracting true 4-quarter longitudinal baselines, Hawkes exponential decay kernels,
   refueling physics, payday liquidity timing, station priors, and time-aligned global SVD.
2. Target-Specific Gain Pruning: Selects 168 top features for CLV_fuel and 168 top features
   for CLV_nonfuel separately via LightGBM information gain.
3. 3-Pillar Regression Architecture per Target (with 2.0x weight on 2024-12-01):
   - Pillar A: 5-seed BaggedTwoPartHurdle (Huber alpha=0.85 + CatBoost MAE)
   - Pillar B: Loss-Diverse RegressorEnsemble (LGBM Tweedie, LGBM Huber, XGB, CatBoost)
   - Pillar C: GroupKFold OOF Multi-Quantile Quadrature + L2 Hurdle with RidgeCV Meta-Stacker
4. Zero-Preserving Moment & Quantile Alignment to Candidate N (Fuel) and Candidate O (Nonfuel).
5. Locks Opportunity 100% to Candidate V's verified 0.541843185 peak.
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
import sys
import time
from typing import Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PipelineConfig
from src.data import load_transactions
from src.dataset import SnapshotBuilder
from src.features_v3 import (
    compute_payday_timing,
    compute_refueling_physics,
    compute_station_attachment_priors,
)
from src.hurdle import BaggedTwoPartFuelHurdle, BaggedTwoPartNonfuelHurdle
from src.models import RegressorEnsemble

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_candidate_w2")

TARGET_FUEL_MEAN = 1.445650359224524
TARGET_FUEL_STD = 0.89858823
TARGET_NF_MEAN = 0.7016566530602794
TARGET_NF_STD = 0.68971049


def compute_hawkes_and_longitudinal_features(
    tx: pd.DataFrame, cutoff: pd.Timestamp, ids: Sequence[str]
) -> pd.DataFrame:
    """Computes continuous-time Hawkes exponential decay kernels and true 4-quarter
    longitudinal demeaned & lag-1 realized hurdle features directly from transactions."""
    id_set = set(ids)
    h_tx = tx[
        (tx["ID"].astype(str).isin(id_set))
        & (tx["_time"] < cutoff)
        & (tx["is_qualifying"] == True)
        & (tx["amount"] > 0)
    ].copy()

    res = pd.DataFrame(index=pd.Index(ids, name="ID"))
    if h_tx.empty:
        return res

    h_tx["ID_str"] = h_tx["ID"].astype(str)
    days_ago = (cutoff - h_tx["_time"]).dt.total_seconds() / 86400.0
    h_tx["days_ago"] = days_ago
    fuel_qty = np.where(h_tx["is_fuel_sale"], np.maximum(0.0, h_tx["qty"]), 0.0)
    nf_rand = np.where(h_tx["is_nonfuel"], np.maximum(0.0, h_tx["amount"]), 0.0)

    # 1. Multi-Scale Hawkes Exponential Kernels (half-lives: 14d, 45d, 135d)
    for hl in (14.0, 45.0, 135.0):
        decay = np.exp(-days_ago / hl)
        h_tx[f"hk_fuel_{int(hl)}"] = fuel_qty * decay
        h_tx[f"hk_nf_{int(hl)}"] = nf_rand * decay

    hk_cols = [f"hk_fuel_{hl}" for hl in (14, 45, 135)] + [f"hk_nf_{hl}" for hl in (14, 45, 135)]
    hk_agg = h_tx.groupby("ID_str")[hk_cols].sum()
    res = res.join(hk_agg).fillna(0.0)

    res["hawkes_fuel_accel_14_45"] = (3.21 * res["hk_fuel_14"] + 1.0) / (res["hk_fuel_45"] + 1.0)
    res["hawkes_fuel_accel_45_135"] = (3.00 * res["hk_fuel_45"] + 1.0) / (res["hk_fuel_135"] + 1.0)
    res["hawkes_nf_accel_14_45"] = (3.21 * res["hk_nf_14"] + 1.0) / (res["hk_nf_45"] + 1.0)
    res["hawkes_nf_accel_45_135"] = (3.00 * res["hk_nf_45"] + 1.0) / (res["hk_nf_135"] + 1.0)

    # 2. True 4-Quarter Longitudinal History: q0=[T-90d, T), q1=[T-180d, T-90d), q2=[T-270d, T-180d), q3=[T-360d, T-270d)
    h_tx["q_idx"] = np.floor(days_ago / 90.0).astype(int)
    h_sub = h_tx[h_tx["q_idx"].isin([0, 1, 2, 3])].copy()
    h_sub["f_qty"] = np.where(h_sub["is_fuel_sale"], np.maximum(0.0, h_sub["qty"]), 0.0)
    h_sub["nf_amt"] = np.where(h_sub["is_nonfuel"], np.maximum(0.0, h_sub["amount"]), 0.0)

    q_fuel = h_sub.groupby(["ID_str", "q_idx"])["f_qty"].sum().unstack(fill_value=0.0)
    q_nf = h_sub.groupby(["ID_str", "q_idx"])["nf_amt"].sum().unstack(fill_value=0.0)
    q_bask = h_sub.groupby(["ID_str", "q_idx"])["TransactionId"].nunique().unstack(fill_value=0.0)

    for q in (0, 1, 2, 3):
        if q not in q_fuel.columns:
            q_fuel[q] = 0.0
        if q not in q_nf.columns:
            q_nf[q] = 0.0
        if q not in q_bask.columns:
            q_bask[q] = 0.0

    q_fuel = q_fuel.reindex(ids, fill_value=0.0)
    q_nf = q_nf.reindex(ids, fill_value=0.0)
    q_bask = q_bask.reindex(ids, fill_value=0.0)

    # Empirical Bayes / Credibility-Weighted Customer Baseline across prior quarters q1, q2, q3
    active_prior_q = (q_bask[[1, 2, 3]] > 0).sum(axis=1)
    prior_fuel_mean = q_fuel[[1, 2, 3]].sum(axis=1) / np.maximum(1.0, active_prior_q)
    prior_fuel_max = q_fuel[[1, 2, 3]].max(axis=1)
    prior_nf_mean = q_nf[[1, 2, 3]].sum(axis=1) / np.maximum(1.0, active_prior_q)
    prior_nf_max = q_nf[[1, 2, 3]].max(axis=1)

    res["within_demeaned_m3_fuel_litres"] = q_fuel[0] - prior_fuel_mean
    res["within_ratio_m3_to_prior_max_fuel"] = (q_fuel[0] + 1.0) / (prior_fuel_max + 1.0)
    res["within_demeaned_m3_nonfuel_rand"] = q_nf[0] - prior_nf_mean
    res["within_ratio_m3_to_prior_max_nf"] = (q_nf[0] + 5.0) / (prior_nf_max + 5.0)

    # Realized growth outcomes in [T-3m, T) relative to [T-6m, T-3m) (Mean-Reversion & Habit Persistence)
    res["lag1_realized_clv_fuel"] = np.log1p(np.maximum(0.0, q_fuel[0])) / 2.352825386185113
    res["lag1_realized_clv_nf"] = np.log1p(np.maximum(0.0, q_nf[0])) / 2.4200904988927476
    res["lag2_realized_clv_fuel"] = np.log1p(np.maximum(0.0, q_fuel[1])) / 2.352825386185113
    res["lag2_realized_clv_nf"] = np.log1p(np.maximum(0.0, q_nf[1])) / 2.4200904988927476
    res["lag1_realized_fuel_growth_ratio"] = (q_fuel[0] + 1.0) / (1.25 * q_fuel[1] + 1.0)
    res["lag1_cleared_fuel_hurdle"] = (
        (q_fuel[0] > 1.25 * q_fuel[1]) & ((q_fuel[0] - q_fuel[1]) >= 1.0)
    ).astype(np.float32)

    # Left-censoring / newly eligible customer tenure adjustment
    first_tx_time = h_tx.groupby("ID_str")["_time"].min().reindex(ids)
    tenure_days = ((cutoff - first_tx_time).dt.total_seconds() / 86400.0).fillna(0.0)
    res["is_left_censored_90d"] = (tenure_days < 90.0).astype(np.float32)
    eff_days_m3 = np.clip(tenure_days, 7.0, 90.0)
    res["annualized_m3_fuel_litres"] = q_fuel[0] * (90.0 / eff_days_m3)
    res["annualized_m3_nf_rand"] = q_nf[0] * (90.0 / eff_days_m3)

    return res.fillna(0.0)


class GlobalAlignedSVD:
    """Fits a single fixed vocabulary and SVD basis on historical transactions prior to 2025-09-01
    so that latent item embeddings have 100% consistent coordinate axes across all cutoffs."""

    def __init__(self, n_components: int = 16, top_k_items: int = 500, seed: int = 42) -> None:
        self.n_components = n_components
        self.top_k_items = top_k_items
        self.seed = seed
        self.item_map: dict[Any, int] = {}
        self.svd: TruncatedSVD | None = None

    def fit(self, tx: pd.DataFrame, anchor_cutoff: pd.Timestamp) -> "GlobalAlignedSVD":
        h_tx = tx[(tx["_time"] < anchor_cutoff) & (tx["is_qualifying"] == True) & (tx["amount"] > 0)]
        top_items = sorted(h_tx["StockItemId"].dropna().value_counts().head(self.top_k_items).index.tolist())
        self.item_map = {item: i for i, item in enumerate(top_items)}

        sub_tx = h_tx[h_tx["StockItemId"].isin(self.item_map)].copy()
        sub_tx["ID_str"] = sub_tx["ID"].astype(str)
        agg = sub_tx.groupby(["ID_str", "StockItemId"])["amount"].sum().reset_index()

        unique_custs = sorted(agg["ID_str"].unique())
        cust_map = {cid: i for i, cid in enumerate(unique_custs)}

        r_idx = agg["ID_str"].map(cust_map).to_numpy()
        c_idx = agg["StockItemId"].map(self.item_map).to_numpy()
        data = np.log1p(np.maximum(0.0, agg["amount"].to_numpy()))

        mat = csr_matrix((data, (r_idx, c_idx)), shape=(len(unique_custs), len(self.item_map)))
        self.svd = TruncatedSVD(n_components=self.n_components, random_state=self.seed)
        self.svd.fit(mat)
        return self

    def transform(self, tx: pd.DataFrame, cutoff: pd.Timestamp, ids: Sequence[str]) -> pd.DataFrame:
        cols = [f"svd_aligned_{k}" for k in range(self.n_components)]
        if self.svd is None or not self.item_map:
            return pd.DataFrame(0.0, index=pd.Index(ids, name="ID"), columns=cols)

        id_list = list(ids)
        cust_map = {cid: i for i, cid in enumerate(id_list)}
        h_tx = tx[
            (tx["_time"] < cutoff)
            & (tx["is_qualifying"] == True)
            & (tx["amount"] > 0)
            & (tx["StockItemId"].isin(self.item_map))
            & (tx["ID"].astype(str).isin(cust_map))
        ].copy()

        if h_tx.empty:
            return pd.DataFrame(0.0, index=pd.Index(ids, name="ID"), columns=cols)

        h_tx["ID_str"] = h_tx["ID"].astype(str)
        agg = h_tx.groupby(["ID_str", "StockItemId"])["amount"].sum().reset_index()

        r_idx = agg["ID_str"].map(cust_map).to_numpy()
        c_idx = agg["StockItemId"].map(self.item_map).to_numpy()
        data = np.log1p(np.maximum(0.0, agg["amount"].to_numpy()))

        mat = csr_matrix((data, (r_idx, c_idx)), shape=(len(id_list), len(self.item_map)))
        emb = self.svd.transform(mat)
        return pd.DataFrame(emb, index=pd.Index(ids, name="ID"), columns=cols)


class OutOfFoldMultiQuantileHurdle:
    """Leak-Free Customer-Grouped OOF Hurdle with Multi-Quantile Quadrature for Stage 2."""

    def __init__(self, target_name: str = "CLV_fuel", seed: int = 42, n_jobs: int = -1) -> None:
        self.target_name = target_name
        self.seed = seed
        self.n_jobs = n_jobs
        self.quantiles = (0.10, 0.25, 0.50, 0.75, 0.90)
        self.q_weights = np.array([0.15, 0.225, 0.25, 0.225, 0.15], dtype=np.float64)

        self.clf_lgb: lgb.LGBMClassifier | None = None
        self.clf_cb: CatBoostClassifier | None = None
        self.calibrator: IsotonicRegression | None = None

        self.reg_l2_lgb: lgb.LGBMRegressor | None = None
        self.reg_l2_cb: CatBoostRegressor | None = None
        self.reg_quantiles: list[lgb.LGBMRegressor] = []
        self.reg_direct: lgb.LGBMRegressor | None = None
        self.meta_reg: RidgeCV | None = None
        self.high_cap: float = 3.5

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        groups: np.ndarray,
        sample_weight: np.ndarray,
        is_summer: np.ndarray,
    ) -> "OutOfFoldMultiQuantileHurdle":
        X_df = X.select_dtypes(include=[np.number]).fillna(0.0)
        y_arr = np.asarray(y, dtype=np.float64)
        y_bin = (y_arr > 0).astype(int)
        N = len(y_arr)

        pos_all = y_arr > 0
        self.high_cap = float(np.percentile(y_arr[pos_all], 99.5))

        gkf = GroupKFold(n_splits=5)
        oof_p_raw = np.zeros(N, dtype=np.float64)
        oof_mu_pos = np.zeros(N, dtype=np.float64)
        oof_y_direct = np.zeros(N, dtype=np.float64)

        logger.info("[%s] Generating Customer-Grouped 5-Fold OOF predictions...", self.target_name)
        for tr_idx, val_idx in gkf.split(X_df, y_arr, groups=groups):
            # 1. Stage 1 Activity Classifier (OOF)
            clf_f = lgb.LGBMClassifier(
                n_estimators=250, learning_rate=0.05, max_depth=6, num_leaves=31,
                min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
                random_state=self.seed, n_jobs=self.n_jobs, verbose=-1,
            )
            clf_f.fit(X_df.iloc[tr_idx], y_bin[tr_idx], sample_weight=sample_weight[tr_idx])
            oof_p_raw[val_idx] = clf_f.predict_proba(X_df.iloc[val_idx])[:, 1]

            # 2. Stage 2 Positive Spenders: Winsorized L2 + 5-Quantile Quadrature (OOF)
            tr_pos = tr_idx[y_arr[tr_idx] > 0]
            reg_l2 = lgb.LGBMRegressor(
                n_estimators=300, learning_rate=0.04, max_depth=6, num_leaves=31,
                reg_lambda=20.0, min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
                random_state=self.seed, n_jobs=self.n_jobs, verbose=-1,
            )
            reg_l2.fit(
                X_df.iloc[tr_pos],
                np.clip(y_arr[tr_pos], 0.0, self.high_cap),
                sample_weight=sample_weight[tr_pos],
            )
            pred_l2_val = np.clip(reg_l2.predict(X_df.iloc[val_idx]), 0.0, None)

            pred_q_val = np.zeros(len(val_idx), dtype=np.float64)
            for q, qw in zip(self.quantiles, self.q_weights):
                reg_q = lgb.LGBMRegressor(
                    objective="quantile", alpha=q, n_estimators=200, learning_rate=0.05,
                    max_depth=5, num_leaves=24, min_child_samples=50, subsample=0.8,
                    colsample_bytree=0.8, random_state=self.seed, n_jobs=self.n_jobs, verbose=-1,
                )
                reg_q.fit(X_df.iloc[tr_pos], y_arr[tr_pos], sample_weight=sample_weight[tr_pos])
                pred_q_val += qw * np.clip(reg_q.predict(X_df.iloc[val_idx]), 0.0, None)

            oof_mu_pos[val_idx] = 0.50 * pred_l2_val + 0.50 * pred_q_val

            # 3. Direct Single-Stage L2 Regressor (OOF)
            reg_dir = lgb.LGBMRegressor(
                n_estimators=250, learning_rate=0.05, max_depth=6, num_leaves=31,
                reg_lambda=25.0, min_child_samples=60, subsample=0.8, colsample_bytree=0.8,
                random_state=self.seed, n_jobs=self.n_jobs, verbose=-1,
            )
            reg_dir.fit(X_df.iloc[tr_idx], y_arr[tr_idx], sample_weight=sample_weight[tr_idx])
            oof_y_direct[val_idx] = np.clip(reg_dir.predict(X_df.iloc[val_idx]), 0.0, None)

        # Fit Isotonic Calibrator & RidgeCV Meta-Regressor strictly on OOF predictions
        self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        oof_p_cal = self.calibrator.fit_transform(oof_p_raw, y_bin, sample_weight=sample_weight)

        oof_meta_X = self._build_meta_matrix(oof_p_cal, oof_mu_pos, oof_y_direct, is_summer)
        self.meta_reg = RidgeCV(alphas=np.logspace(-1, 4, 25), cv=5)
        self.meta_reg.fit(oof_meta_X, y_arr, sample_weight=sample_weight)
        oof_pred = np.clip(self.meta_reg.predict(oof_meta_X), 0.0, None)
        oof_rmse = float(np.sqrt(np.mean((y_arr - oof_pred) ** 2)))
        logger.info("[%s] Leak-Free OOF RMSE: %.6f (Meta alpha=%.2f)", self.target_name, oof_rmse, self.meta_reg.alpha_)

        # Fit final full-panel models for test inference
        self.clf_lgb = lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6, num_leaves=31,
            min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
            random_state=self.seed, n_jobs=self.n_jobs, verbose=-1,
        ).fit(X_df, y_bin, sample_weight=sample_weight)

        self.clf_cb = CatBoostClassifier(
            iterations=300, learning_rate=0.06, depth=6, random_seed=self.seed, verbose=False
        ).fit(X_df, y_bin, sample_weight=sample_weight)

        X_pos = X_df.iloc[pos_all]
        y_pos = y_arr[pos_all]
        sw_pos = sample_weight[pos_all]

        self.reg_l2_lgb = lgb.LGBMRegressor(
            n_estimators=350, learning_rate=0.04, max_depth=6, num_leaves=31,
            reg_lambda=20.0, min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
            random_state=self.seed, n_jobs=self.n_jobs, verbose=-1,
        ).fit(X_pos, np.clip(y_pos, 0.0, self.high_cap), sample_weight=sw_pos)

        self.reg_l2_cb = CatBoostRegressor(
            iterations=350, learning_rate=0.05, depth=6, l2_leaf_reg=15.0,
            random_seed=self.seed, verbose=False,
        ).fit(X_pos, np.clip(y_pos, 0.0, self.high_cap), sample_weight=sw_pos)

        self.reg_quantiles = []
        for q in self.quantiles:
            rq = lgb.LGBMRegressor(
                objective="quantile", alpha=q, n_estimators=250, learning_rate=0.05,
                max_depth=5, num_leaves=24, min_child_samples=50, subsample=0.8,
                colsample_bytree=0.8, random_state=self.seed, n_jobs=self.n_jobs, verbose=-1,
            ).fit(X_pos, y_pos, sample_weight=sw_pos)
            self.reg_quantiles.append(rq)

        self.reg_direct = lgb.LGBMRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=6, num_leaves=31,
            reg_lambda=25.0, min_child_samples=60, subsample=0.8, colsample_bytree=0.8,
            random_state=self.seed, n_jobs=self.n_jobs, verbose=-1,
        ).fit(X_df, y_arr, sample_weight=sample_weight)

        return self

    @staticmethod
    def _build_meta_matrix(
        p_cal: np.ndarray, mu_pos: np.ndarray, y_direct: np.ndarray, is_summer: np.ndarray
    ) -> np.ndarray:
        y_hurdle = p_cal * mu_pos
        return np.column_stack([
            p_cal,
            mu_pos,
            y_hurdle,
            y_direct,
            y_hurdle * is_summer,
            y_direct * is_summer,
        ])

    def predict(self, X: pd.DataFrame, is_summer: np.ndarray) -> np.ndarray:
        X_df = X.select_dtypes(include=[np.number]).fillna(0.0)
        p_raw = 0.55 * self.clf_lgb.predict_proba(X_df)[:, 1] + 0.45 * self.clf_cb.predict_proba(X_df)[:, 1]
        p_cal = self.calibrator.predict(p_raw)

        mu_l2 = 0.55 * np.clip(self.reg_l2_lgb.predict(X_df), 0.0, None) + 0.45 * np.clip(self.reg_l2_cb.predict(X_df), 0.0, None)
        mu_q = np.zeros(len(X_df), dtype=np.float64)
        for rq, qw in zip(self.reg_quantiles, self.q_weights):
            mu_q += qw * np.clip(rq.predict(X_df), 0.0, None)

        mu_pos = 0.50 * mu_l2 + 0.50 * mu_q
        y_dir = np.clip(self.reg_direct.predict(X_df), 0.0, None)

        meta_X = self._build_meta_matrix(p_cal, mu_pos, y_dir, is_summer)
        return np.clip(self.meta_reg.predict(meta_X), 0.0, None)


def match_moments_zero_preserving(
    y_pred: np.ndarray,
    ref_pred: np.ndarray,
    target_mean: float,
    target_std: float,
    zero_gate_quantile: float = 0.0,
) -> np.ndarray:
    """Aligns mean and standard deviation to verified champion moments while preserving
    exact 0.0 values in zero-gated nonfuel predictions."""
    pred = np.copy(y_pred)
    if zero_gate_quantile > 0.0:
        thresh = np.quantile(pred, zero_gate_quantile)
        pred = np.where(pred <= thresh, 0.0, pred)

    # Rank-preserving empirical distribution mapping onto the verified champion reference
    ranks = np.argsort(np.argsort(pred))
    ref_sorted = np.sort(ref_pred)
    quantile_mapped = ref_sorted[ranks]

    # Blend 75% linear mean-scaled prediction with 25% empirical quantile-matched prediction
    pos_mask = pred > 1e-6
    scaled = np.zeros_like(pred)
    if pos_mask.any():
        p_pos = pred[pos_mask]
        r_pos = ref_pred[ref_pred > 1e-6]
        r_pos_mean = float(np.mean(r_pos))
        r_pos_std = float(np.std(r_pos))
        p_pos_std = float(np.std(p_pos))
        p_pos_mean = float(np.mean(p_pos))
        scaled_pos = r_pos_mean + ((p_pos - p_pos_mean) / max(p_pos_std, 1e-6)) * r_pos_std
        scaled[pos_mask] = np.clip(scaled_pos, 0.0, None)

    out = 0.75 * scaled + 0.25 * quantile_mapped
    out = np.where(pos_mask, out, 0.0)
    out = out * (target_mean / max(float(np.mean(out)), 1e-8))
    return np.clip(out, 0.0, None)


def extract_enriched_snapshot(
    builder: SnapshotBuilder,
    tx: pd.DataFrame,
    svd_model: GlobalAlignedSVD,
    cutoff: pd.Timestamp,
    ids: Sequence[str] | None = None,
    with_labels: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    snap = builder.build(cutoff, with_labels=with_labels)
    if ids is not None:
        base_feats = snap.features.reindex(ids)
        labels = snap.labels.reindex(ids) if with_labels and snap.labels is not None else None
        active_ids = list(ids)
    else:
        base_feats = snap.features
        labels = snap.labels if with_labels else None
        active_ids = base_feats.index.astype(str).tolist()

    df_burn = compute_refueling_physics(tx, cutoff, active_ids)
    df_payday = compute_payday_timing(tx, cutoff, active_ids)
    df_stn = compute_station_attachment_priors(tx, cutoff, active_ids)
    df_hk = compute_hawkes_and_longitudinal_features(tx, cutoff, active_ids)
    df_svd = svd_model.transform(tx, cutoff, active_ids)

    full_feats = pd.concat([base_feats, df_burn, df_payday, df_stn, df_hk, df_svd], axis=1)
    # Remove duplicate column names if any overlap with base_feats
    full_feats = full_feats.loc[:, ~full_feats.columns.duplicated()]

    rec_cols = [
        c for c in full_feats.columns
        if "recency_" in c or "days_since_" in c or "gap_" in c or c.startswith("days_")
    ]
    full_feats[rec_cols] = full_feats[rec_cols].fillna(999.0)
    if "fuel_price_m3" in full_feats.columns:
        full_feats["fuel_price_m3"] = full_feats["fuel_price_m3"].fillna(23.50)
    full_feats = full_feats.fillna(0.0)
    return full_feats, labels


def build_candidate_w2(blend_w: float = 0.15, n_features: int = 168) -> Path:
    test_csv = PROJECT_ROOT / "test.csv"
    test_df = pd.read_csv(test_csv)
    test_ids = [str(x) for x in test_df["ID"]]
    assert len(test_ids) == 5488

    # 1. Load active champion: Candidate V (0.312139112)
    cand_v_files = sorted((PROJECT_ROOT / "submissions").glob("submission_candidate_v_boundary_refined_*.csv"))
    assert cand_v_files, "No Candidate V submission found in submissions/!"
    cand_v_path = cand_v_files[-1]
    df_v = pd.read_csv(cand_v_path)
    df_v["ID"] = df_v["ID"].astype(str)
    df_v = df_v.set_index("ID").reindex(test_ids).reset_index()
    logger.info("Loaded Champion Candidate V: %s", cand_v_path.name)

    cfg = PipelineConfig()
    tx = load_transactions(cfg)
    builder = SnapshotBuilder(cfg, tx)

    # 2. Fit Global Time-Aligned SVD on historical transactions prior to 2025-09-01
    logger.info("Fitting GlobalAlignedSVD vocabulary & basis...")
    svd_model = GlobalAlignedSVD(n_components=16, top_k_items=500, seed=42)
    svd_model.fit(tx, pd.Timestamp("2025-09-01"))

    cutoffs = [
        pd.Timestamp("2024-09-01"),
        pd.Timestamp("2024-12-01"),
        pd.Timestamp("2025-03-01"),
        pd.Timestamp("2025-06-01"),
        pd.Timestamp("2025-09-01"),
    ]

    X_list, y_fuel_list, y_nf_list, sw_list, summer_list, grp_list = [], [], [], [], [], []
    for c in cutoffs:
        logger.info("Building enriched snapshot for %s...", c.strftime("%Y-%m-%d"))
        feats, lbls = extract_enriched_snapshot(builder, tx, svd_model, c, with_labels=True)
        is_sum = 1.0 if c.month == 12 else 0.0
        w = 2.0 if is_sum == 1.0 else 1.0

        X_list.append(feats)
        y_fuel_list.append(lbls["CLV_fuel"].to_numpy())
        y_nf_list.append(lbls["CLV_nonfuel"].to_numpy())
        sw_list.append(np.full(len(feats), w, dtype=np.float64))
        summer_list.append(np.full(len(feats), is_sum, dtype=np.float64))
        grp_list.append(feats.index.astype(str).to_numpy())

    X_all = pd.concat(X_list, axis=0)
    y_fuel_all = np.concatenate(y_fuel_list)
    y_nf_all = np.concatenate(y_nf_list)
    sw_all = np.concatenate(sw_list)
    summer_all = np.concatenate(summer_list)
    groups_all = np.concatenate(grp_list)

    logger.info("Extracting enriched test snapshot for 2025-12-01...")
    X_test_all, _ = extract_enriched_snapshot(
        builder, tx, svd_model, pd.Timestamp("2025-12-01"), ids=test_ids, with_labels=False
    )
    summer_test = np.ones(len(test_ids), dtype=np.float64)

    # 3. Target-Specific Gain Feature Pruning (168 Fuel Features & 168 Nonfuel Features)
    num_cols = X_all.select_dtypes(include=[np.number]).columns.tolist()
    logger.info("Selecting target-specific top %d features by LightGBM gain from %d candidates...", n_features, len(num_cols))

    sel_f = lgb.LGBMRegressor(
        n_estimators=150, learning_rate=0.05, num_leaves=31, importance_type="gain", random_state=42, n_jobs=-1, verbose=-1
    ).fit(X_all[num_cols], y_fuel_all, sample_weight=sw_all)
    top_fuel_feats = pd.Series(sel_f.feature_importances_, index=num_cols).sort_values(ascending=False).head(n_features).index.tolist()

    sel_nf = lgb.LGBMRegressor(
        n_estimators=150, learning_rate=0.05, num_leaves=31, importance_type="gain", random_state=42, n_jobs=-1, verbose=-1
    ).fit(X_all[num_cols], y_nf_all, sample_weight=sw_all)
    top_nf_feats = pd.Series(sel_nf.feature_importances_, index=num_cols).sort_values(ascending=False).head(n_features).index.tolist()

    logger.info("Top 8 Fuel features: %s", top_fuel_feats[:8])
    logger.info("Top 8 Nonfuel features: %s", top_nf_feats[:8])

    X_train_f, X_test_f = X_all[top_fuel_feats], X_test_all[top_fuel_feats]
    X_train_nf, X_test_nf = X_all[top_nf_feats], X_test_all[top_nf_feats]

    # 4. Train 3-Pillar Fuel Super-Ensemble
    logger.info("=== Training 3-Pillar Fuel Super-Ensemble ===")
    bag_fuel = BaggedTwoPartFuelHurdle(
        seeds=(42, 100, 2024, 777, 314), n_jobs=-1, huber_alpha=0.85, power_shrinkage=0.98
    ).fit(X_train_f, y_fuel_all, sample_weight=sw_all)
    pred_fuel_bag = bag_fuel.predict(X_test_f)

    ens_fuel = RegressorEnsemble(
        members=["lgbm", "lgbm_huber", "xgb", "catboost"], tweedie_power=1.15
    ).fit(X_train_f, y_fuel_all, sample_weight=sw_all)
    pred_fuel_ens = ens_fuel.predict_members(X_test_f).mean(axis=1)

    mq_fuel = OutOfFoldMultiQuantileHurdle(target_name="CLV_fuel", seed=42, n_jobs=-1).fit(
        X_train_f, y_fuel_all, groups=groups_all, sample_weight=sw_all, is_summer=summer_all
    )
    pred_fuel_mq = mq_fuel.predict(X_test_f, is_summer=summer_test)

    pred_fuel_raw = 0.40 * pred_fuel_bag + 0.35 * pred_fuel_ens + 0.25 * pred_fuel_mq

    # 5. Train 3-Pillar Nonfuel Super-Ensemble
    logger.info("=== Training 3-Pillar Nonfuel Super-Ensemble ===")
    bag_nf = BaggedTwoPartNonfuelHurdle(
        seeds=(42, 100, 2024, 777, 314), n_jobs=-1, huber_alpha=0.85, power_shrinkage=0.96, zero_gate_threshold=0.04
    ).fit(X_train_nf, y_nf_all, sample_weight=sw_all)
    pred_nf_bag = bag_nf.predict(X_test_nf)

    ens_nf = RegressorEnsemble(
        members=["lgbm_huber", "xgb", "catboost"]
    ).fit(X_train_nf, y_nf_all, sample_weight=sw_all)
    pred_nf_ens = ens_nf.predict_members(X_test_nf).mean(axis=1)

    mq_nf = OutOfFoldMultiQuantileHurdle(target_name="CLV_nonfuel", seed=42, n_jobs=-1).fit(
        X_train_nf, y_nf_all, groups=groups_all, sample_weight=sw_all, is_summer=summer_all
    )
    pred_nf_mq = mq_nf.predict(X_test_nf, is_summer=summer_test)

    pred_nf_raw = 0.45 * pred_nf_bag + 0.30 * pred_nf_ens + 0.25 * pred_nf_mq
    # Preserve exact zero-gating where BaggedTwoPartNonfuelHurdle gated to 0
    pred_nf_raw = np.where(pred_nf_bag == 0.0, 0.0, pred_nf_raw)

    # 6. Zero-Preserving Moment Alignment & Analytical Blend with Candidate V
    cand_v_fuel = df_v["CLV_fuel"].to_numpy()
    cand_v_nf = df_v["CLV_nonfuel"].to_numpy()

    pred_fuel_aligned = match_moments_zero_preserving(
        pred_fuel_raw, cand_v_fuel, TARGET_FUEL_MEAN, TARGET_FUEL_STD
    )
    pred_nf_aligned = match_moments_zero_preserving(
        pred_nf_raw, cand_v_nf, TARGET_NF_MEAN, TARGET_NF_STD
    )

    corr_f = float(np.corrcoef(cand_v_fuel, pred_fuel_aligned)[0, 1])
    corr_nf = float(np.corrcoef(cand_v_nf, pred_nf_aligned)[0, 1])
    logger.info("Correlation with Candidate V: Fuel rho=%.4f, Nonfuel rho=%.4f", corr_f, corr_nf)

    pred_fuel_final = (1.0 - blend_w) * cand_v_fuel + blend_w * pred_fuel_aligned
    pred_nf_final = (1.0 - blend_w) * cand_v_nf + blend_w * pred_nf_aligned

    pred_fuel_final = pred_fuel_final * (TARGET_FUEL_MEAN / float(np.mean(pred_fuel_final)))
    pred_nf_final = pred_nf_final * (TARGET_NF_MEAN / float(np.mean(pred_nf_final)))

    # 7. Assemble & Verify Submission
    df_w2 = pd.DataFrame({
        "ID": test_ids,
        "CLV_fuel": np.clip(pred_fuel_final, 0.0, None),
        "CLV_nonfuel": np.clip(pred_nf_final, 0.0, None),
        "Opportunity": df_v["Opportunity"].to_numpy(),
    })

    assert len(df_w2) == 5488
    assert not df_w2.isna().any().any()
    counts = df_w2["Opportunity"].value_counts().to_dict()
    assert counts["Stable"] == 1960
    assert counts["Existing-category growth: Fuel"] == 1576
    assert counts["Inactivity"] == 1443

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = PROJECT_ROOT / "submissions" / f"submission_candidate_w2_multi_quantile_{timestamp}.csv"
    df_w2.to_csv(out_path, index=False)
    df_w2.to_csv(PROJECT_ROOT / "submission.csv", index=False)

    sha256 = hashlib.sha256(out_path.read_bytes()).hexdigest()
    logger.info("Candidate W2 saved: %s (SHA-256: %s)", out_path, sha256)
    return out_path


if __name__ == "__main__":
    build_candidate_w2(blend_w=0.15, n_features=168)
