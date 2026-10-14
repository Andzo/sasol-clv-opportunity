"""Module 1: Physical & Sequence Feature Extractor for v3 Architecture.

Extracts:
1. Forecourt Refueling Physics & Vehicle Burn-Rate Dynamics
2. South African Payday & Intra-Month Liquidity Timing
3. Bipartite Graph TruncatedSVD Embeddings (Customer x Products) & Station Priors
4. Within-Customer Longitudinal Demeaning & Lagged Realized Outcomes
5. 3D Bi-Weekly Sequence Tensor (N, 26, 8) for Neural Sequence Models
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

from src.config import PipelineConfig
from src.dataset import SnapshotBuilder

logger = logging.getLogger("features_v3")


def compute_refueling_physics(tx: pd.DataFrame, cutoff: pd.Timestamp, ids: Sequence[str]) -> pd.DataFrame:
    """Computes inter-fill burn rates, tank capacity, completeness, and depletion states."""
    id_set = set(ids)
    fuel_tx = tx[
        (tx["ID"].astype(str).isin(id_set))
        & (tx["is_fuel_sale"] == True)
        & (tx["_time"] < cutoff)
        & (tx["qty"] > 0)
    ].copy()

    fuel_tx = fuel_tx.sort_values(["ID", "_time"])
    fuel_tx["prev_time"] = fuel_tx.groupby("ID")["_time"].shift(1)
    fuel_tx["dt_days"] = (fuel_tx["_time"] - fuel_tx["prev_time"]).dt.total_seconds() / 86400.0

    # Calculate burn rates for fills with valid interval (> 0.5 days and < 60 days)
    valid_burn = fuel_tx[(fuel_tx["dt_days"] >= 0.5) & (fuel_tx["dt_days"] <= 60.0)].copy()
    valid_burn["burn_rate"] = valid_burn["qty"] / valid_burn["dt_days"]

    burn_agg = valid_burn.groupby("ID")["burn_rate"].agg(
        fuel_burn_rate_median="median",
        fuel_burn_rate_mean="mean",
    )

    # General fill aggregates
    fuel_tx["is_round_50"] = (fuel_tx["amount"] % 50.0 < 0.05) | (fuel_tx["amount"] % 50.0 > 49.95)
    gen_agg = fuel_tx.groupby("ID").agg(
        tank_capacity_proxy=("qty", "max"),
        mean_fill_litres=("qty", "mean"),
        last_fill_time=("_time", "max"),
        round_rand_fill_share=("is_round_50", "mean"),
    )
    gen_agg["fill_completeness_ratio"] = gen_agg["mean_fill_litres"] / np.maximum(gen_agg["tank_capacity_proxy"], 1e-3)
    gen_agg["days_since_last_fuel_sale"] = (cutoff - gen_agg["last_fill_time"]).dt.total_seconds() / 86400.0

    res = pd.DataFrame(index=pd.Index(ids, name="ID"))
    res = res.join(burn_agg).join(gen_agg.drop(columns=["last_fill_time", "mean_fill_litres"]))

    # Fill defaults for customers with no/sparse fuel transactions
    global_median_burn = float(res["fuel_burn_rate_median"].median()) if not res["fuel_burn_rate_median"].dropna().empty else 1.5
    res["fuel_burn_rate_median"] = res["fuel_burn_rate_median"].fillna(global_median_burn)
    res["fuel_burn_rate_mean"] = res["fuel_burn_rate_mean"].fillna(global_median_burn)
    res["tank_capacity_proxy"] = res["tank_capacity_proxy"].fillna(45.0)
    res["fill_completeness_ratio"] = res["fill_completeness_ratio"].fillna(0.70)
    res["round_rand_fill_share"] = res["round_rand_fill_share"].fillna(0.0)
    res["days_since_last_fuel_sale"] = res["days_since_last_fuel_sale"].fillna(999.0)

    # Expected litres needed at T
    res["expected_litres_needed_at_cutoff"] = np.clip(
        res["days_since_last_fuel_sale"] * res["fuel_burn_rate_median"], 0.0, 2.5 * res["tank_capacity_proxy"]
    )
    return res


def compute_payday_timing(tx: pd.DataFrame, cutoff: pd.Timestamp, ids: Sequence[str]) -> pd.DataFrame:
    """Computes South African salary/wage cycle timing (25th-2nd and 15th) spend shares."""
    id_set = set(ids)
    start_6m = cutoff - pd.DateOffset(months=6)

    sub_tx = tx[
        (tx["ID"].astype(str).isin(id_set))
        & (tx["_time"] >= start_6m)
        & (tx["_time"] < cutoff)
        & (tx["is_qualifying"] == True)
        & (tx["amount"] > 0)
    ].copy()

    day = sub_tx["_time"].dt.day
    sub_tx["is_payday_window"] = (day >= 25) | (day <= 2)
    sub_tx["is_midmonth_window"] = (day >= 14) & (day <= 16)
    sub_tx["is_friday_pm"] = (sub_tx["_time"].dt.dayofweek == 4) & (sub_tx["_time"].dt.hour >= 12)

    agg = sub_tx.groupby("ID").agg(
        total_spend=("amount", "sum"),
        payday_spend=("amount", lambda s: s[sub_tx.loc[s.index, "is_payday_window"]].sum()),
        midmonth_spend=("amount", lambda s: s[sub_tx.loc[s.index, "is_midmonth_window"]].sum()),
        friday_pm_spend=("amount", lambda s: s[sub_tx.loc[s.index, "is_friday_pm"]].sum()),
    )

    res = pd.DataFrame(index=pd.Index(ids, name="ID"))
    res = res.join(agg)

    res["payday_window_spend_share"] = (res["payday_spend"] / np.maximum(res["total_spend"], 1e-3)).fillna(0.0)
    res["midmonth_window_spend_share"] = (res["midmonth_spend"] / np.maximum(res["total_spend"], 1e-3)).fillna(0.0)
    res["friday_pm_spend_share"] = (res["friday_pm_spend"] / np.maximum(res["total_spend"], 1e-3)).fillna(0.0)

    # Days from last payday fill
    payday_fuel = sub_tx[sub_tx["is_fuel_sale"] & sub_tx["is_payday_window"]].groupby("ID")["_time"].max()
    res["days_from_last_payday_fill"] = ((cutoff - payday_fuel).dt.total_seconds() / 86400.0).reindex(ids).fillna(999.0)

    return res[["payday_window_spend_share", "midmonth_window_spend_share", "friday_pm_spend_share", "days_from_last_payday_fill"]]


def compute_bipartite_svd(tx: pd.DataFrame, cutoff: pd.Timestamp, ids: Sequence[str], n_components: int = 16) -> pd.DataFrame:
    """Computes 16-component TruncatedSVD embeddings across top StockItemIds and ItemCategoryLevel3."""
    h_tx = tx[(tx["_time"] < cutoff) & (tx["is_qualifying"] == True) & (tx["amount"] > 0)].copy()

    # Top 500 stock items
    top_items = set(h_tx["StockItemId"].value_counts().head(500).index)
    sub_tx = h_tx[h_tx["StockItemId"].isin(top_items) & h_tx["ID"].astype(str).isin(set(ids))]

    if sub_tx.empty:
        cols = [f"svd_item_{k}" for k in range(n_components)]
        return pd.DataFrame(0.0, index=pd.Index(ids, name="ID"), columns=cols)

    cust_list = list(ids)
    cust_map = {cid: i for i, cid in enumerate(cust_list)}
    sorted_items = sorted(top_items)
    item_map = {item: i for i, item in enumerate(sorted_items)}

    r_idx = sub_tx["ID"].astype(str).map(cust_map).values
    c_idx = sub_tx["StockItemId"].map(item_map).values
    data = np.log1p(np.maximum(0.0, sub_tx["amount"].values))

    mat = csr_matrix((data, (r_idx, c_idx)), shape=(len(cust_list), len(sorted_items)))
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    emb = svd.fit_transform(mat)

    cols = [f"svd_item_{k}" for k in range(n_components)]
    return pd.DataFrame(emb, index=pd.Index(ids, name="ID"), columns=cols)


def compute_station_attachment_priors(tx: pd.DataFrame, cutoff: pd.Timestamp, ids: Sequence[str]) -> pd.DataFrame:
    """Computes station footprint category attachment vectors weighted by customer visit share."""
    h_tx = tx[(tx["_time"] < cutoff) & (tx["is_qualifying"] == True)].copy()

    key_cats = ["Coffee", "Car Wash", "Krispy Kreme", "Braai & Ice", "Beverages", "Food & beverage", "Tobacco"]
    stn_cat = h_tx.groupby(["LocationId", "ItemCategoryLevel2"])["amount"].sum().unstack(fill_value=0.0)
    for c in key_cats:
        if c not in stn_cat:
            stn_cat[c] = 0.0
    stn_total = stn_cat.sum(axis=1)
    stn_prior = stn_cat[key_cats].div(np.maximum(stn_total, 1.0), axis=0)

    # Customer station visit shares in last 6 months
    sub_6m = h_tx[h_tx["_time"] >= cutoff - pd.DateOffset(months=6)]
    cust_stn = sub_6m.groupby(["ID", "LocationId"])["TransactionId"].nunique().unstack(fill_value=0.0)
    cust_total = cust_stn.sum(axis=1)
    cust_stn_share = cust_stn.div(np.maximum(cust_total, 1.0), axis=0)

    common_stns = cust_stn_share.columns.intersection(stn_prior.index)
    cust_attachment = cust_stn_share[common_stns].dot(stn_prior.loc[common_stns])
    cust_attachment.columns = [f"stn_prior_{c.lower().replace(' ', '_')}" for c in key_cats]

    res = pd.DataFrame(index=pd.Index(ids, name="ID"))
    res = res.join(cust_attachment).fillna(0.0)
    return res


def compute_within_customer_demeaned_features(
    base_features: pd.DataFrame,
    historical_snapshots_features: list[pd.DataFrame],
) -> pd.DataFrame:
    """Computes expanding longitudinal mean and within-customer demeaned features."""
    all_snaps = historical_snapshots_features + [base_features]
    key_metrics = ["m3_fuel_litres", "m3_nonfuel_rand", "m3_baskets"]

    res = pd.DataFrame(index=base_features.index)
    for col in key_metrics:
        if col in base_features.columns:
            stacked_col = pd.concat([df[[col]] for df in all_snaps if col in df.columns], axis=1)
            expanding_mean = stacked_col.mean(axis=1)
            res[f"within_demeaned_{col}"] = base_features[col] - expanding_mean
            res[f"expanding_mean_{col}"] = expanding_mean
    return res.fillna(0.0)


def build_sequence_tensor(
    tx: pd.DataFrame,
    cutoff: pd.Timestamp,
    ids: Sequence[str],
    n_steps: int = 26,
    step_days: int = 14,
) -> np.ndarray:
    """Constructs a 3D sequence tensor (N_customers, 26, 8) covering 364 days before cutoff."""
    id_map = {cid: i for i, cid in enumerate(ids)}
    total_days = n_steps * step_days
    start_date = cutoff - pd.Timedelta(days=total_days)

    sub = tx[
        (tx["_time"] >= start_date)
        & (tx["_time"] < cutoff)
        & (tx["ID"].astype(str).isin(id_map))
    ].copy()

    days_before = (cutoff - sub["_time"]).dt.total_seconds() / 86400.0
    bin_idx = (n_steps - 1) - np.clip(np.floor(days_before / float(step_days)).astype(int), 0, n_steps - 1)
    sub["bin"] = bin_idx
    sub["cust_idx"] = sub["ID"].astype(str).map(id_map)

    tensor = np.zeros((len(ids), n_steps, 8), dtype=np.float32)
    if sub.empty:
        return tensor

    sub["fuel_qty"] = np.where(sub["is_fuel_sale"], sub["qty"], 0.0)
    sub["fuel_rand"] = np.where(sub["is_fuel_sale"], sub["amount"], 0.0)
    sub["nonfuel_rand"] = np.where(sub["is_nonfuel"], sub["amount"], 0.0)
    sub["is_weekend"] = sub["_time"].dt.dayofweek >= 5
    hr = sub["_time"].dt.hour
    sub["is_commuter_hr"] = ((hr >= 6) & (hr <= 9)) | ((hr >= 16) & (hr <= 19))

    agg = sub.groupby(["cust_idx", "bin"]).agg(
        fuel_qty=("fuel_qty", "sum"),
        fuel_rand=("fuel_rand", "sum"),
        nonfuel_rand=("nonfuel_rand", "sum"),
        n_baskets=("TransactionId", "nunique"),
        n_categories=("ItemCategoryLevel2", "nunique"),
        discount=("modifier_amount", "sum"),
        weekend_cnt=("is_weekend", "sum"),
        commuter_cnt=("is_commuter_hr", "sum"),
        total_lines=("TransactionId", "count"),
    ).reset_index()

    c_idx = agg["cust_idx"].values
    b_idx = agg["bin"].values
    tensor[c_idx, b_idx, 0] = np.log1p(np.maximum(0.0, agg["fuel_qty"].values))
    tensor[c_idx, b_idx, 1] = np.log1p(np.maximum(0.0, agg["fuel_rand"].values))
    tensor[c_idx, b_idx, 2] = np.log1p(np.maximum(0.0, agg["nonfuel_rand"].values))
    tensor[c_idx, b_idx, 3] = np.log1p(agg["n_baskets"].values)
    tensor[c_idx, b_idx, 4] = agg["n_categories"].values
    tensor[c_idx, b_idx, 5] = np.log1p(np.maximum(0.0, agg["discount"].values))
    tensor[c_idx, b_idx, 6] = agg["weekend_cnt"].values / np.maximum(1, agg["total_lines"].values)
    tensor[c_idx, b_idx, 7] = agg["commuter_cnt"].values / np.maximum(1, agg["total_lines"].values)

    return tensor


class V3FeatureExtractor:
    """Orchestrates comprehensive v3 feature extraction."""

    def __init__(self, cfg: PipelineConfig, tx: pd.DataFrame) -> None:
        self.cfg = cfg
        self.tx = tx
        self.builder = SnapshotBuilder(cfg, tx)

    def extract_cutoff(
        self,
        cutoff: pd.Timestamp,
        ids: Sequence[str] | None = None,
        with_labels: bool = True,
        historical_snapshots: list[pd.DataFrame] | None = None,
        use_cache: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame | None, np.ndarray]:
        """Extracts combined flat feature matrix, labels (if with_labels=True), and sequence tensor with disk caching."""
        cutoff_str = cutoff.strftime("%Y-%m-%d")
        cache_dir = self.cfg.paths.cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        feat_cache = cache_dir / f"v3_features_{cutoff_str}.parquet"
        lbl_cache = cache_dir / f"v3_labels_{cutoff_str}.parquet"
        seq_cache = cache_dir / f"v3_seq_{cutoff_str}.npy"

        if use_cache and feat_cache.exists() and seq_cache.exists() and (not with_labels or lbl_cache.exists()):
            logger.info("Loading cached v3 snapshot for cutoff %s...", cutoff_str)
            v3_features = pd.read_parquet(feat_cache)
            labels = pd.read_parquet(lbl_cache) if with_labels and lbl_cache.exists() else None
            seq_tensor = np.load(seq_cache)

            if ids is not None:
                id_list = list(ids)
                cur_ids = v3_features.index.astype(str).tolist()
                id_to_idx = {cid: idx for idx, cid in enumerate(cur_ids)}
                idx_sel = [id_to_idx[cid] for cid in id_list if cid in id_to_idx]
                v3_features = v3_features.reindex(id_list).fillna(0.0)
                if labels is not None:
                    labels = labels.reindex(id_list)
                if len(idx_sel) == len(id_list):
                    seq_tensor = seq_tensor[idx_sel]
                else:
                    seq_tensor = build_sequence_tensor(self.tx, cutoff, id_list, n_steps=26, step_days=14)
            return v3_features, labels, seq_tensor

        logger.info("Extracting base snapshot for cutoff %s...", cutoff_str)
        snap = self.builder.build(cutoff, with_labels=with_labels)
        if ids is not None:
            features = snap.features.reindex(ids)
            labels = snap.labels.reindex(ids) if with_labels and snap.labels is not None else None
            active_ids = list(ids)
        else:
            features = snap.features
            labels = snap.labels if with_labels else None
            active_ids = features.index.astype(str).tolist()

        logger.info("Computing refueling physics...")
        df_burn = compute_refueling_physics(self.tx, cutoff, active_ids)

        logger.info("Computing payday liquidity timing...")
        df_payday = compute_payday_timing(self.tx, cutoff, active_ids)

        logger.info("Computing bipartite SVD embeddings...")
        df_svd = compute_bipartite_svd(self.tx, cutoff, active_ids, n_components=16)

        logger.info("Computing station attachment priors...")
        df_stn = compute_station_attachment_priors(self.tx, cutoff, active_ids)

        logger.info("Computing within-customer demeaning...")
        df_within = compute_within_customer_demeaned_features(features, historical_snapshots or [])

        # Concatenate flat feature blocks
        v3_features = pd.concat([features, df_burn, df_payday, df_svd, df_stn, df_within], axis=1)

        # Impute recency and base NaNs cleanly
        rec_cols = [c for c in v3_features.columns if 'recency_' in c or 'days_since_' in c or 'gap_' in c or c.startswith('days_')]
        v3_features[rec_cols] = v3_features[rec_cols].fillna(999.0)
        if 'fuel_price_m3' in v3_features.columns:
            v3_features['fuel_price_m3'] = v3_features['fuel_price_m3'].fillna(23.50)
        v3_features = v3_features.fillna(0.0)

        logger.info("Building 3D sequence tensor (N, 26, 8)...")
        seq_tensor = build_sequence_tensor(self.tx, cutoff, active_ids, n_steps=26, step_days=14)

        if use_cache and ids is None:
            v3_features.to_parquet(feat_cache)
            if labels is not None:
                labels.to_parquet(lbl_cache)
            np.save(seq_cache, seq_tensor)
            logger.info("Cached v3 snapshot to %s", cache_dir)

        logger.info(
            "Cutoff %s complete: flat features shape %s, sequence tensor shape %s",
            cutoff_str, v3_features.shape, seq_tensor.shape
        )
        return v3_features, labels, seq_tensor
