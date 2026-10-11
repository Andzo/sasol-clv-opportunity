"""Point-in-time customer feature engineering.

Guarantees
----------
* **No leakage**: features for cutoff ``t`` use only rows with ``_time < t``.
* **Label parity**: window bounds use calendar months (``pd.DateOffset``), so the
  ``m3`` window equals the "previous quarter" baseline of the growth rule and
  ``ever_<cat>`` equals the rule's historical-purchase test.
* **Stationarity**: lifetime quantities are expressed as per-month rates over the
  customer's observed span (absolute all-time sums grow mechanically with the
  snapshot date and push the test snapshot outside the training range).
* **Honest censoring**: a window starting before the first observed transaction
  is reported as ``NaN`` (unknown), never as ``0`` (inactive).
"""
from __future__ import annotations

import re
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.data import included_categories

FEATURE_VERSION = "2.1.0"
DAY = pd.Timedelta(days=1)
DAYS_PER_MONTH = 30.4375
TENURE_CAP_DAYS = 365.0

#: name -> (start_months_ago, end_months_ago); the window is [t - start, t - end).
WINDOWS: dict[str, tuple[int, int]] = {
    "m1": (1, 0),
    "m3": (3, 0),    # == label "previous quarter" baseline
    "m6": (6, 0),
    "q2": (6, 3),    # quarter before the previous quarter
    "yoy": (12, 9),  # same calendar season as the outcome window, one year earlier
}
TREND_MONTHS = 6


def safe_name(text: str) -> str:
    """Model-safe snake_case identifier for a category / item name."""
    return re.sub(r"[^0-9a-zA-Z]+", "_", text).strip("_").lower()


def _months_before(t: pd.Timestamp, months: int) -> pd.Timestamp:
    return t - pd.DateOffset(months=months)


def _days_since(t: pd.Timestamp, times: pd.Series) -> pd.Series:
    times_dt = pd.to_datetime(times, errors="coerce")
    return (t - times_dt) / DAY


def _window_block(hist: pd.DataFrame, ids: pd.Index, t: pd.Timestamp, data_start: pd.Timestamp) -> pd.DataFrame:
    """Activity / spend / volume aggregates for each calendar window."""
    frames = []
    for name, (start, end) in WINDOWS.items():
        lo, hi = _months_before(t, start), _months_before(t, end)
        w = hist[(hist["_time"] >= lo) & (hist["_time"] < hi)]
        q, s = w[w["is_qualifying"]], w[w["in_scope"]]
        block = pd.DataFrame({
            f"{name}_baskets": q.groupby("ID")["TransactionId"].nunique(),
            f"{name}_fuel_baskets": q[q["is_fuel_cat"]].groupby("ID")["TransactionId"].nunique(),
            f"{name}_nonfuel_baskets": q[q["is_nonfuel"]].groupby("ID")["TransactionId"].nunique(),
            f"{name}_fuel_litres": s[s["is_fuel_sale"]].groupby("ID")["qty"].sum(),
            f"{name}_fuel_rand": s[s["is_fuel_cat"]].groupby("ID")["amount"].sum(),
            f"{name}_nonfuel_rand": s[s["is_nonfuel"]].groupby("ID")["amount"].sum(),
            f"{name}_n_categories": q.groupby("ID")["ItemCategoryLevel2"].nunique(),
        }).reindex(ids).astype("float64").fillna(0.0)
        if lo < data_start:  # window not fully observed -> unknown, not zero
            block.loc[:, :] = np.nan
        frames.append(block)
    return pd.concat(frames, axis=1)


def _derived_block(win: pd.DataFrame) -> pd.DataFrame:
    """Momentum ratios and unit economics derived from the window aggregates."""
    out: dict[str, pd.Series] = {}
    for qty in ("baskets", "fuel_litres", "fuel_rand", "nonfuel_rand"):
        out[f"mom_m3_q2_{qty}"] = (win[f"m3_{qty}"] + 1.0) / (win[f"q2_{qty}"] + 1.0)
        out[f"mom_m3_yoy_{qty}"] = (win[f"m3_{qty}"] + 1.0) / (win[f"yoy_{qty}"] + 1.0)
    out["delta_m3_q2_fuel_rand"] = win["m3_fuel_rand"] - win["q2_fuel_rand"]
    out["delta_m3_q2_nonfuel_rand"] = win["m3_nonfuel_rand"] - win["q2_nonfuel_rand"]
    out["m1_share_of_m3_baskets"] = win["m1_baskets"] / win["m3_baskets"].where(win["m3_baskets"] > 0)
    out["fuel_price_m3"] = win["m3_fuel_rand"] / win["m3_fuel_litres"].where(win["m3_fuel_litres"] > 0)
    spend = win["m3_fuel_rand"] + win["m3_nonfuel_rand"]
    out["basket_value_m3"] = spend / win["m3_baskets"].where(win["m3_baskets"] > 0)
    out["fuel_share_m3"] = win["m3_fuel_rand"] / spend.where(spend > 0)
    out["fuel_m1_m3_ratio"] = (win["m1_fuel_rand"] + 1.0) / (win["m3_fuel_rand"] / 3.0 + 1.0)
    out["fuel_litres_m1_m3_ratio"] = (win["m1_fuel_litres"] + 1.0) / (win["m3_fuel_litres"] / 3.0 + 1.0)
    out["nonfuel_m1_m3_ratio"] = (win["m1_nonfuel_rand"] + 1.0) / (win["m3_nonfuel_rand"] / 3.0 + 1.0)
    out["fuel_reversion_ratio"] = (win["m3_fuel_rand"] / 3.0 + 1.0) / (win["m6_fuel_rand"] / 6.0 + 1.0)
    out["nonfuel_reversion_ratio"] = (win["m3_nonfuel_rand"] / 3.0 + 1.0) / (win["m6_nonfuel_rand"] / 6.0 + 1.0)
    out["fuel_growth_hurdle_rand"] = np.maximum(20.0, 0.25 * win["m3_fuel_rand"])
    out["fuel_hurdle_ratio"] = out["fuel_growth_hurdle_rand"] / (win["m3_fuel_rand"] + 1.0)
    out["store_attachment_rate"] = win["m3_nonfuel_baskets"] / (win["m3_baskets"] + 1.0)
    out["forecourt_only"] = (win["m3_nonfuel_baskets"] == 0).astype("float64")
    out["has_visited_m1"] = (win["m1_baskets"] > 0).astype("float64")
    out["has_visited_fuel_m1"] = (win["m1_fuel_baskets"] > 0).astype("float64")
    return pd.DataFrame(out)


def _recency_block(
    hist: pd.DataFrame, ids: pd.Index, t: pd.Timestamp, data_start: pd.Timestamp
) -> tuple[pd.DataFrame, pd.Series]:
    """Recency, regularity and lifetime *rates*; also returns observed months per customer."""
    q = hist[hist["is_qualifying"]]
    s = hist[hist["in_scope"]]
    baskets = (
        q.groupby(["ID", "TransactionId"])["_time"].min().reset_index()
        .sort_values(["ID", "_time"], kind="mergesort")
    )
    baskets["gap"] = baskets.groupby("ID")["_time"].diff() / DAY
    g = baskets.groupby("ID")
    first_any = hist.groupby("ID")["_time"].min().reindex(ids)
    observed_months = (_days_since(t, first_any) / DAYS_PER_MONTH).clip(lower=1.0)

    out = pd.DataFrame(index=ids)
    out["days_since_last_purchase"] = _days_since(t, g["_time"].max())
    out["days_since_last_any_txn"] = _days_since(t, hist.groupby("ID")["_time"].max())
    out["days_since_last_fuel"] = _days_since(t, q[q["is_fuel_cat"]].groupby("ID")["_time"].max())
    out["days_since_last_nonfuel"] = _days_since(t, q[q["is_nonfuel"]].groupby("ID")["_time"].max())
    out["days_since_first_seen_capped"] = _days_since(t, first_any).clip(upper=TENURE_CAP_DAYS)
    out["left_censored"] = (first_any < data_start + 30 * DAY).astype("float64")
    out["gap_mean"] = g["gap"].mean()
    out["gap_median"] = g["gap"].median()
    out["gap_std"] = g["gap"].std()
    out["gap_last"] = g["gap"].last()
    out["overdue_ratio"] = out["days_since_last_purchase"] / out["gap_median"].clip(lower=1.0)
    out["exp_churn_hazard"] = 1.0 - np.exp(-out["days_since_last_purchase"] / out["gap_mean"].clip(lower=1.0))
    out["fuel_overdue_ratio"] = out["days_since_last_fuel"] / out["gap_median"].clip(lower=1.0)
    out["fuel_dormant_in_m3"] = (
        (q[q["is_fuel_cat"]].groupby("ID").size().reindex(ids).fillna(0) > 0) & (out["days_since_last_fuel"] > 90)
    ).astype("float64")
    out["baskets_per_month_life"] = g.size().reindex(ids).fillna(0) / observed_months
    out["fuel_litres_per_month_life"] = (
        s[s["is_fuel_sale"]].groupby("ID")["qty"].sum().reindex(ids).fillna(0) / observed_months
    )
    out["nonfuel_rand_per_month_life"] = (
        s[s["is_nonfuel"]].groupby("ID")["amount"].sum().reindex(ids).fillna(0) / observed_months
    )
    return out, observed_months


def _trend_block(hist: pd.DataFrame, ids: pd.Index, t: pd.Timestamp, data_start: pd.Timestamp) -> pd.DataFrame:
    """Monthly qualifying-basket counts, spend & volumes for the last 6 months plus OLS slopes."""
    q = hist[hist["is_qualifying"]]
    s = hist[hist["in_scope"]]
    out = pd.DataFrame(index=ids)
    for lag in range(1, TREND_MONTHS + 1):
        lo, hi = _months_before(t, lag), _months_before(t, lag - 1)
        m_q = q[(q["_time"] >= lo) & (q["_time"] < hi)]
        m_s = s[(s["_time"] >= lo) & (s["_time"] < hi)]
        b_col = m_q.groupby("ID")["TransactionId"].nunique().reindex(ids).fillna(0.0)
        f_rand = m_s[m_s["is_fuel_cat"]].groupby("ID")["amount"].sum().reindex(ids).fillna(0.0)
        f_lit = m_s[m_s["is_fuel_sale"]].groupby("ID")["qty"].sum().reindex(ids).fillna(0.0)
        nf_rand = m_s[m_s["is_nonfuel"]].groupby("ID")["amount"].sum().reindex(ids).fillna(0.0)
        out[f"baskets_lag{lag}"] = np.nan if lo < data_start else b_col
        out[f"fuel_rand_lag{lag}"] = np.nan if lo < data_start else f_rand
        out[f"fuel_litres_lag{lag}"] = np.nan if lo < data_start else f_lit
        out[f"nonfuel_rand_lag{lag}"] = np.nan if lo < data_start else nf_rand

    x = np.arange(TREND_MONTHS, dtype="float64") - (TREND_MONTHS - 1) / 2.0
    denom = float((x**2).sum())
    for prefix in ("baskets", "fuel_rand", "fuel_litres", "nonfuel_rand"):
        lags = out[[f"{prefix}_lag{k}" for k in range(TREND_MONTHS, 0, -1)]].to_numpy(dtype="float64")
        out[f"{prefix}_trend_slope"] = ((lags - lags.mean(axis=1, keepdims=True)) * x).sum(axis=1) / denom
        out.loc[np.isnan(lags).any(axis=1), f"{prefix}_trend_slope"] = np.nan

    out["active_months_6"] = (out[[f"baskets_lag{k}" for k in range(1, TREND_MONTHS + 1)]].to_numpy() > 0).sum(axis=1).astype("float64")
    out.loc[out[[f"baskets_lag{k}" for k in range(1, TREND_MONTHS + 1)]].isna().any(axis=1), "active_months_6"] = np.nan
    return out


def _category_block(
    hist: pd.DataFrame, ids: pd.Index, t: pd.Timestamp, cats: list[str], observed_months: pd.Series
) -> pd.DataFrame:
    """Per-category history: ever-bought, previous-quarter spend/baskets, recency, rate."""
    q = hist[hist["is_qualifying"]]
    s = hist[hist["in_scope"]]
    m3_lo = _months_before(t, 3)
    keys = ["ID", "ItemCategoryLevel2"]

    def pivot(series: pd.Series, fill: float | None) -> pd.DataFrame:
        frame = series.unstack().reindex(index=ids, columns=cats)
        return frame if fill is None else frame.fillna(fill)

    life_baskets = pivot(q.groupby(keys)["TransactionId"].nunique(), 0.0)
    last_time = pivot(q.groupby(keys)["_time"].max(), None)
    m3_spend = pivot(s[s["_time"] >= m3_lo].groupby(keys)["amount"].sum(), 0.0)
    m3_baskets = pivot(q[q["_time"] >= m3_lo].groupby(keys)["TransactionId"].nunique(), 0.0)

    parts = {
        "ever": life_baskets.gt(0).astype("float64"),
        "rate": life_baskets.div(observed_months, axis=0),
        "recency": last_time.apply(lambda col: _days_since(t, col)).astype("float64"),
        "m3spend": m3_spend,
        "m3baskets": m3_baskets,
    }
    frames = []
    for prefix, frame in parts.items():
        frame = frame.copy()
        frame.columns = [f"{prefix}_{safe_name(c)}" for c in cats]
        frames.append(frame)
    block = pd.concat(frames, axis=1)
    block["n_categories_ever"] = parts["ever"].sum(axis=1)
    return block


def _behaviour_block(
    hist: pd.DataFrame, ids: pd.Index, t: pd.Timestamp, label_config: Mapping[str, Any]
) -> pd.DataFrame:
    """Store loyalty, timing habits, cross-shopping, discounts, returns, excluded spend."""
    w = hist[hist["_time"] >= _months_before(t, 6)]
    core = w[~w["is_adjustment"]]
    b = core.groupby(["ID", "TransactionId"]).agg(
        time=("_time", "min"), loc=("LocationId", "first"),
        has_fuel=("is_fuel_cat", "any"), has_nonfuel=("is_nonfuel", "any"),
    ).reset_index()
    b["weekend"] = (b["time"].dt.dayofweek >= 5).astype("float64")
    b["hour"] = b["time"].dt.hour + b["time"].dt.minute / 60.0
    b["cross"] = (b["has_fuel"] & b["has_nonfuel"]).astype("float64")
    g = b.groupby("ID")

    out = pd.DataFrame(index=ids)
    out["m6_distinct_locations"] = g["loc"].nunique()
    out["m6_top_location_share"] = b.groupby(["ID", "loc"]).size().groupby(level=0).max() / g.size()
    out["m6_weekend_share"] = g["weekend"].mean()
    out["m6_mean_hour"] = g["hour"].mean()
    out["m6_std_hour"] = g["hour"].std()
    out["m6_cross_share"] = g["cross"].mean()
    out["m6_return_lines"] = core[core["_quantity"] < 0].groupby("ID").size()
    excluded = core[~core["is_included"]]
    out["m6_excluded_baskets"] = excluded.groupby("ID")["TransactionId"].nunique()
    out["m6_excluded_rand"] = excluded.groupby("ID")["amount"].sum()
    for item in label_config["adjustment_items"]:
        tag = safe_name(item)
        out[f"ever_adj_{tag}"] = hist[hist["ItemName"] == item].groupby("ID").size().gt(0).astype("float64")
        out[f"m6_adj_{tag}_lines"] = w[w["ItemName"] == item].groupby("ID").size()
    count_cols = [c for c in out.columns if not c.startswith(("m6_mean_hour", "m6_std_hour", "m6_top", "m6_weekend", "m6_cross"))]
    out[count_cols] = out[count_cols].fillna(0.0)
    return out


def build_features(
    tx: pd.DataFrame,
    cutoff: pd.Timestamp | str,
    ids: pd.Index,
    label_config: Mapping[str, Any],
    data_start: pd.Timestamp,
) -> pd.DataFrame:
    """Build the full feature matrix for ``ids`` at ``cutoff``.

    Args:
        tx: annotated transactions (see :func:`src.data.annotate`).
        cutoff: snapshot timestamp; only rows strictly before it are used.
        ids: customers to featurise (index of the returned frame, in order).
        label_config: official label configuration.
        data_start: first observed transaction timestamp (censoring boundary).

    Returns:
        ``float64`` DataFrame indexed by ``ID`` with a deterministic column order.
    """
    t = pd.Timestamp(cutoff)
    ids = pd.Index(ids, name="ID").astype(str)
    hist = tx[(tx["_time"] < t) & tx["ID"].isin(ids)]
    cats = included_categories(label_config)

    win = _window_block(hist, ids, t, data_start)
    recency, observed_months = _recency_block(hist, ids, t, data_start)
    blocks = [
        win,
        _derived_block(win),
        recency,
        _trend_block(hist, ids, t, data_start),
        _category_block(hist, ids, t, cats, observed_months),
        _behaviour_block(hist, ids, t, label_config),
    ]
    feats = pd.concat(blocks, axis=1)
    feats["cutoff_month_sin"] = np.sin(2 * np.pi * t.month / 12.0)
    feats["cutoff_month_cos"] = np.cos(2 * np.pi * t.month / 12.0)
    feats.index.name = "ID"
    return feats.astype("float64")
