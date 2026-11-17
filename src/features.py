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
import scipy.special as sp
from scipy.optimize import minimize

from src.data import included_categories

FEATURE_VERSION = "5.0.0"
DAY = pd.Timedelta(days=1)
DAYS_PER_MONTH = 30.4375
TENURE_CAP_DAYS = 180.0

CAT_STRUCTURAL_FEATURES: tuple[str, ...] = (
    "cat_primary_location_id",
    "cat_top_nonfuel_level2",
    "cat_top_nonfuel_level3",
    "cat_payday_commuter_archetype",
)

NONFUEL_LEVEL2_CATEGORIES: tuple[str, ...] = (
    "Automotive Products", "Beverages", "Braai & Ice", "Breads & rolls", "Car Wash",
    "Carrier Bags", "Coffee", "Confectionary", "Food & beverage", "Frozen food",
    "Gas", "General merchandise", "Groceries", "Ice-cream", "Krispy Kreme",
    "Lubricants", "Miscellaneous", "Perishables", "Print media", "Promotional items",
    "Seattle", "Snacking", "Tobacco", "Unknown", "Value added services",
)
NONFUEL_L2_MAP: dict[str, int] = {c: i + 1 for i, c in enumerate(NONFUEL_LEVEL2_CATEGORIES)}

NONFUEL_LEVEL3_CATEGORIES: tuple[str, ...] = (
    "Alternate snacking", "BI Confectionary", "BI Meals", "Batteries", "Bill Payments",
    "Biltong", "Biscuits", "Bought in_BI", "Braai", "Breads & rolls",
    "Car Wash", "Carrier Bags", "Cellphone Accessories", "Chips", "Chocolate",
    "Cigarettes", "Coffee", "Coffee Packaging", "Coffee Recipe Items", "Dairy and eggs",
    "Dairy and eggs - Butter", "Dairy and eggs - Cheese", "Dairy and eggs - Eggs", "Dairy drinks", "E-Cigarettes & Vape",
    "Edible groceries", "Edible groceries - Canned Food", "Edible groceries - Infant Edible", "Edible groceries - Pharmaceutical Edible", "Electronics",
    "Empties & crates", "Energy", "Entertainment", "Events & ticketing", "FPG Biltong",
    "FPG Bread & rolls", "FPG Chicken full & pieces", "FPG Confectionary", "FPG Meals", "FPG Pies",
    "FPG Sandwiches & Subs", "FPG Savoury", "Flowers", "Food & beverage consumables", "Freshly Prepaired",
    "Freshly prepared_FPG", "Frozen meals", "Frozen meat", "Fruit & veg", "Gaming",
    "Gas", "General merchandise", "Gums & rolls", "Hardware", "Health & energy",
    "Hot Chocolate", "Ice", "Ice Tea & Coffee", "Ice-Cream", "Juice",
    "Krispy Kreme", "Lotto", "Lubricants - Additives", "Lubricants - Lubricants", "Lubricants - Other",
    "Milk - Fresh Cream", "Milk - Fresh Milk", "Milk - Long Life", "Milk - Other Milk Products", "Miscellaneous",
    "Non-edible groceries", "Non-edible groceries - Infant Non-Edible", "Non-edible groceries - Personal Care", "Non-edible groceries - Pet", "Nougat",
    "Novelty Sweets", "Occasions", "Other VAS", "Prepacks", "Prepaid",
    "Prepared Beverages", "Print media", "Processed meats", "Promotional items", "Promotions",
    "Seattle", "Soft drinks", "Speciality", "Sport drinks", "Stationery",
    "Sugar based confectionary", "Sunglasses", "Tea", "Tobacco products", "Toys",
    "Unknown", "Vehicle Products", "Water", "Yoghurt",
)
NONFUEL_L3_MAP: dict[str, int] = {c: i + 1 for i, c in enumerate(NONFUEL_LEVEL3_CATEGORIES)}


#: name -> (start_months_ago, end_months_ago); the window is [t - start, t - end).
WINDOWS: dict[str, tuple[int, int]] = {
    "m1": (1, 0),
    "m3": (3, 0),    # == label "previous quarter" baseline
    "m6": (6, 0),
    "q2": (6, 3),    # quarter before the previous quarter
    "yoy_q": (12, 9), # same quarter last year (year-over-year baseline)
}
TREND_MONTHS = 6

DEAD_CATEGORIES: set[str] = set()
ACTIVE_ADJUSTMENT_ITEMS: tuple[str, ...] = ("TOTAL/ROUNDING DISCO",)


def safe_name(text: str) -> str:
    """Model-safe snake_case identifier for a category / item name."""
    return re.sub(r"[^0-9a-zA-Z]+", "_", text).strip("_").lower()


def _months_before(t: pd.Timestamp, months: int) -> pd.Timestamp:
    return t - pd.DateOffset(months=months)


def _days_since(t: pd.Timestamp, times: pd.Series) -> pd.Series:
    times_dt = pd.to_datetime(times, errors="coerce")
    return (t - times_dt) / DAY


def _window_block(hist: pd.DataFrame, ids: pd.Index, t: pd.Timestamp, data_start: pd.Timestamp) -> pd.DataFrame:
    """Activity / spend / volume aggregates for each calendar window plus d14/d45 sub-windows."""
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
        if lo < data_start:  # window not fully observed -> unknown (NaN), not zero
            block.loc[:, :] = np.nan
        frames.append(block)

    for days_back, tag in [(14, "d14"), (45, "d45")]:
        lo_d = t - pd.Timedelta(days=days_back)
        w_d = hist[(hist["_time"] >= lo_d) & (hist["_time"] < t)]
        q_d, s_d = w_d[w_d["is_qualifying"]], w_d[w_d["in_scope"]]
        d_block = pd.DataFrame({
            f"{tag}_baskets": q_d.groupby("ID")["TransactionId"].nunique(),
            f"{tag}_fuel_litres": s_d[s_d["is_fuel_sale"]].groupby("ID")["qty"].sum(),
            f"{tag}_fuel_rand": s_d[s_d["is_fuel_cat"]].groupby("ID")["amount"].sum(),
            f"{tag}_nonfuel_rand": s_d[s_d["is_nonfuel"]].groupby("ID")["amount"].sum(),
        }).reindex(ids).astype("float64").fillna(0.0)
        if lo_d < data_start:
            d_block.loc[:, :] = np.nan
        frames.append(d_block)

    return pd.concat(frames, axis=1)


def _derived_block(win: pd.DataFrame) -> pd.DataFrame:
    """Momentum ratios and unit economics derived from the window aggregates."""
    out: dict[str, pd.Series] = {}
    for qty in ("baskets", "fuel_litres", "nonfuel_rand"):
        out[f"mom_m3_q2_{qty}"] = (win[f"m3_{qty}"] + 1.0) / (win[f"q2_{qty}"] + 1.0)
    out["delta_m3_q2_fuel_litres"] = win["m3_fuel_litres"] - win["q2_fuel_litres"]
    out["delta_m3_q2_fuel_rand"] = win["m3_fuel_rand"] - win["q2_fuel_rand"]
    out["delta_m3_q2_nonfuel_rand"] = win["m3_nonfuel_rand"] - win["q2_nonfuel_rand"]
    out["m1_share_of_m3_baskets"] = win["m1_baskets"] / win["m3_baskets"].where(win["m3_baskets"] > 0)
    out["fuel_price_m3"] = win["m3_fuel_rand"] / win["m3_fuel_litres"].where(win["m3_fuel_litres"] > 0)
    vol_weighted_price = (win["m3_fuel_rand"].sum() / win["m3_fuel_litres"].sum()) if win["m3_fuel_litres"].sum() > 0 else 23.0
    out["rel_fuel_price_m3"] = (out["fuel_price_m3"] / vol_weighted_price).fillna(1.0)
    spend = win["m3_fuel_rand"] + win["m3_nonfuel_rand"]
    out["basket_value_m3"] = spend / win["m3_baskets"].where(win["m3_baskets"] > 0)
    out["fuel_share_m3"] = win["m3_fuel_rand"] / spend.where(spend > 0)

    # Pure physical litres behavioural momentum (insulated from pump price changes)
    out["fuel_litres_m1_m3_ratio"] = (win["m1_fuel_litres"] + 1.0) / (win["m3_fuel_litres"] / 3.0 + 1.0)
    out["fuel_litres_reversion_ratio"] = (win["m3_fuel_litres"] / 3.0 + 1.0) / (win["m6_fuel_litres"] / 6.0 + 1.0)
    out["fuel_d14_m1_litres_ratio"] = (win["d14_fuel_litres"] * (30.4375 / 14.0) + 1.0) / (win["m1_fuel_litres"] + 1.0)
    out["nonfuel_d14_m1_rand_ratio"] = (win["d14_nonfuel_rand"] * (30.4375 / 14.0) + 5.0) / (win["m1_nonfuel_rand"] + 5.0)

    # Nominal Rand ratios relative to the historical growth hurdle
    out["fuel_m1_m3_ratio"] = (win["m1_fuel_rand"] + 1.0) / (win["m3_fuel_rand"] / 3.0 + 1.0)
    out["nonfuel_m1_m3_ratio"] = (win["m1_nonfuel_rand"] + 1.0) / (win["m3_nonfuel_rand"] / 3.0 + 1.0)
    out["fuel_reversion_ratio"] = (win["m3_fuel_rand"] / 3.0 + 1.0) / (win["m6_fuel_rand"] / 6.0 + 1.0)
    out["nonfuel_reversion_ratio"] = (win["m3_nonfuel_rand"] / 3.0 + 1.0) / (win["m6_nonfuel_rand"] / 6.0 + 1.0)

    # Physical category spend hurdles and projected distance
    H_fuel = np.maximum(win["m3_fuel_rand"] + 20.0, 1.25 * win["m3_fuel_rand"])
    H_nonfuel = np.maximum(win["m3_nonfuel_rand"] + 20.0, 1.25 * win["m3_nonfuel_rand"])
    fuel_growth_hurdle = np.maximum(20.0, 0.25 * win["m3_fuel_rand"])
    out["fuel_hurdle_ratio"] = fuel_growth_hurdle / (win["m3_fuel_rand"] + 1.0)
    out["hurdle_growth_required_rand"] = fuel_growth_hurdle
    out["hurdle_fuel_rand_dist"] = 3.0 * win["m1_fuel_rand"] - H_fuel
    out["fuel_proj_m1_hurdle_ratio"] = (3.0 * win["m1_fuel_rand"] + 1.0) / (H_fuel + 1.0)
    out["fuel_proj_q2_hurdle_dist"] = win["q2_fuel_rand"] - H_fuel
    out["fuel_proj_m6_hurdle_dist"] = 0.5 * win["m6_fuel_rand"] - H_fuel
    out["nonfuel_proj_m1_hurdle_dist"] = 3.0 * win["m1_nonfuel_rand"] - H_nonfuel
    out["nonfuel_proj_q2_hurdle_dist"] = win["q2_nonfuel_rand"] - H_nonfuel
    out["nonfuel_proj_m6_hurdle_dist"] = 0.5 * win["m6_nonfuel_rand"] - H_nonfuel

    # Fuel price inflation deflators
    tot_m1_rand = float(win["m1_fuel_rand"].sum()) if "m1_fuel_rand" in win.columns else 0.0
    tot_m1_litres = float(win["m1_fuel_litres"].sum()) if "m1_fuel_litres" in win.columns else 0.0
    tot_m3_rand = float(win["m3_fuel_rand"].sum()) if "m3_fuel_rand" in win.columns else 0.0
    tot_m3_litres = float(win["m3_fuel_litres"].sum()) if "m3_fuel_litres" in win.columns else 0.0

    p_m1 = (tot_m1_rand / tot_m1_litres) if tot_m1_litres > 0 else 23.50
    p_m3 = (tot_m3_rand / tot_m3_litres) if tot_m3_litres > 0 else 23.50
    out["network_fuel_price_m1_m3_ratio"] = pd.Series(p_m1 / max(p_m3, 1e-4), index=win.index)

    st_price = win["station_pump_price_m3"] if "station_pump_price_m3" in win.columns else p_m3
    pump_price_clamped = out["fuel_price_m3"].clip(20.0, 27.0).fillna(st_price).fillna(p_m3)
    out["real_fuel_hurdle_litres"] = np.maximum(
        1.25 * win["m3_fuel_litres"],
        win["m3_fuel_litres"] + 20.0 / pump_price_clamped,
    )
    price_ratio_m3_m1 = float(np.clip(p_m3 / max(p_m1, 1e-4), 0.80, 1.25))
    out["real_fuel_hurdle_litres_price_adj"] = np.maximum(
        1.25 * win["m3_fuel_litres"] * price_ratio_m3_m1,
        win["m3_fuel_litres"] * price_ratio_m3_m1 + 20.0 / max(p_m1, 15.0),
    )

    out["store_attachment_rate"] = win["m3_nonfuel_baskets"] / (win["m3_baskets"] + 1.0)
    out["has_visited_m1"] = (win["m1_baskets"] > 0).astype("float64")

    if "yoy_q_baskets" in win.columns:
        for qty in ("baskets", "fuel_litres", "nonfuel_rand", "fuel_rand"):
            if f"yoy_q_{qty}" in win.columns and f"m3_{qty}" in win.columns:
                c = 10.0 if "rand" in qty else 1.0
                out[f"yoy_to_m3_ratio_{qty}"] = (win[f"yoy_q_{qty}"] + c) / (win[f"m3_{qty}"] + c)
                out[f"mom_m3_yoy_{qty}"] = (win[f"m3_{qty}"] + 1.0) / (win[f"yoy_q_{qty}"] + 1.0)
        if "yoy_q_fuel_rand" in win.columns:
            out["delta_m3_yoy_fuel_rand"] = win["m3_fuel_rand"] - win["yoy_q_fuel_rand"]
            out["delta_m3_yoy_nonfuel_rand"] = win["m3_nonfuel_rand"] - win["yoy_q_nonfuel_rand"]
            yoy_spend = win["yoy_q_fuel_rand"] + win["yoy_q_nonfuel_rand"]
            yoy_fuel_share = win["yoy_q_fuel_rand"] / yoy_spend.where(yoy_spend > 0)
            out["fuel_yoy_share_shift"] = (out["fuel_share_m3"] - yoy_fuel_share).fillna(0.0)
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
    out["days_since_last_fuel"] = _days_since(t, q[q["is_fuel_cat"]].groupby("ID")["_time"].max())
    out["days_since_last_nonfuel"] = _days_since(t, q[q["is_nonfuel"]].groupby("ID")["_time"].max())
    out["days_since_first_seen_capped"] = _days_since(t, first_any).clip(upper=TENURE_CAP_DAYS)
    out["gap_mean"] = g["gap"].mean()
    out["gap_median"] = g["gap"].median()
    out["gap_std"] = g["gap"].std()
    out["gap_last"] = g["gap"].last()
    out["overdue_ratio"] = out["days_since_last_purchase"] / out["gap_median"].clip(lower=1.0)
    out["exp_churn_hazard"] = 1.0 - np.exp(-out["days_since_last_purchase"] / out["gap_mean"].clip(lower=1.0))
    gap_med_safe = out["gap_median"].clip(lower=3.0)
    out["survival_prob_empirical"] = np.exp(- (out["days_since_last_purchase"] / gap_med_safe) ** 1.5)
    out["fuel_survival_prob"] = np.exp(- (out["days_since_last_fuel"] / gap_med_safe) ** 1.5)
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
    for prefix in ("baskets", "fuel_litres", "nonfuel_rand"):
        lags = out[[f"{prefix}_lag{k}" for k in range(TREND_MONTHS, 0, -1)]].to_numpy(dtype="float64")
        out[f"{prefix}_trend_slope"] = ((lags - lags.mean(axis=1, keepdims=True)) * x).sum(axis=1) / denom
        out.loc[np.isnan(lags).any(axis=1), f"{prefix}_trend_slope"] = np.nan

    out["active_months_6"] = (out[[f"baskets_lag{k}" for k in range(1, TREND_MONTHS + 1)]].to_numpy() > 0).sum(axis=1).astype("float64")
    out.loc[out[[f"baskets_lag{k}" for k in range(1, TREND_MONTHS + 1)]].isna().any(axis=1), "active_months_6"] = np.nan

    # Intra-quarter acceleration (Month -1 vs Month -2 vs Month -3)
    out["baskets_m1_vs_m2"] = (out["baskets_lag1"] + 0.5) / (out["baskets_lag2"] + 0.5)
    out["baskets_m2_vs_m3"] = (out["baskets_lag2"] + 0.5) / (out["baskets_lag3"] + 0.5)
    out["fuel_litres_m1_vs_m2"] = (out["fuel_litres_lag1"] + 1.0) / (out["fuel_litres_lag2"] + 1.0)
    out["fuel_litres_m2_vs_m3"] = (out["fuel_litres_lag2"] + 1.0) / (out["fuel_litres_lag3"] + 1.0)
    out["nonfuel_rand_m1_vs_m2"] = (out["nonfuel_rand_lag1"] + 5.0) / (out["nonfuel_rand_lag2"] + 5.0)
    out["nonfuel_rand_m2_vs_m3"] = (out["nonfuel_rand_lag2"] + 5.0) / (out["nonfuel_rand_lag3"] + 5.0)
    out["active_months_m3"] = (
        (out["baskets_lag1"] > 0).astype("float64")
        + (out["baskets_lag2"] > 0).astype("float64")
        + (out["baskets_lag3"] > 0).astype("float64")
    )

    # Drop lag1 duplicates (captured by m1_*) and older fuel_rand_lag4..6, keeping fuel_rand_lag2..3
    drop_cols = [f"{p}_lag1" for p in ("baskets", "fuel_rand", "fuel_litres", "nonfuel_rand")]
    drop_cols += [f"fuel_rand_lag{k}" for k in range(4, TREND_MONTHS + 1)]
    out = out.drop(columns=[c for c in drop_cols if c in out.columns])
    return out


def _category_block(
    hist: pd.DataFrame, ids: pd.Index, t: pd.Timestamp, cats: list[str], observed_months: pd.Series
) -> pd.DataFrame:
    """Per-category history: bounded m6/m3 activity, spend, recency, and rate."""
    q = hist[hist["is_qualifying"]]
    s = hist[hist["in_scope"]]
    m6_lo = _months_before(t, 6)
    m3_lo = _months_before(t, 3)
    m1_lo = _months_before(t, 1)
    q2_lo = _months_before(t, 6)
    q2_hi = _months_before(t, 3)
    keys = ["ID", "ItemCategoryLevel2"]

    def pivot(series: pd.Series, fill: float | None) -> pd.DataFrame:
        frame = series.unstack().reindex(index=ids, columns=cats)
        return frame if fill is None else frame.fillna(fill)

    m6_baskets = pivot(q[q["_time"] >= m6_lo].groupby(keys)["TransactionId"].nunique(), 0.0)
    m3_baskets = pivot(q[q["_time"] >= m3_lo].groupby(keys)["TransactionId"].nunique(), 0.0)
    m6_spend = pivot(s[s["_time"] >= m6_lo].groupby(keys)["amount"].sum(), 0.0)
    m3_spend = pivot(s[s["_time"] >= m3_lo].groupby(keys)["amount"].sum(), 0.0)
    m1_spend = pivot(s[s["_time"] >= m1_lo].groupby(keys)["amount"].sum(), 0.0)
    q2_spend = pivot(s[(s["_time"] >= q2_lo) & (s["_time"] < q2_hi)].groupby(keys)["amount"].sum(), 0.0)
    last_time = pivot(q.groupby(keys)["_time"].max(), None)

    active_m6 = m6_baskets.gt(0).astype("float64")
    active_m3 = m3_baskets.gt(0).astype("float64")
    # Rate expressed over bounded 6-month window min(observed_months, 6.0)
    rate_m6 = m6_baskets.div(observed_months.clip(upper=6.0), axis=0)
    recency = last_time.apply(lambda col: _days_since(t, col)).clip(upper=180.0).astype("float64")

    life_baskets = pivot(q.groupby(keys)["TransactionId"].nunique(), 0.0)
    ever = life_baskets.gt(0).astype("float64")

    parts = {
        "ever": ever,
        "active_m6": active_m6,
        "active_m3": active_m3,
        "rate": rate_m6,
        "recency": recency,
        "m1spend": m1_spend,
        "m3spend": m3_spend,
        "q2spend": q2_spend,
        "m3baskets": m3_baskets,
    }
    frames = []
    for prefix, frame in parts.items():
        frame = frame.copy()
        frame.columns = [f"{prefix}_{safe_name(c)}" for c in cats]
        frames.append(frame)
    block = pd.concat(frames, axis=1)
    block["n_categories_m6"] = active_m6.sum(axis=1)
    block["n_categories_m3"] = active_m3.sum(axis=1)
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

    # Station Location Target Profiling
    loc_baskets = core.groupby("LocationId")["TransactionId"].nunique()
    loc_fuel_litres = core[core["is_fuel_sale"]].groupby("LocationId")["qty"].sum()
    loc_fuel_baskets = core[core["is_fuel_cat"]].groupby("LocationId")["TransactionId"].nunique()
    loc_nonfuel_baskets = core[core["is_nonfuel"]].groupby("LocationId")["TransactionId"].nunique()
    loc_weekend_baskets = core[core["_time"].dt.dayofweek >= 5].groupby("LocationId")["TransactionId"].nunique()

    loc_mean_litres = (loc_fuel_litres / loc_fuel_baskets.clip(lower=1)).fillna(21.0)
    loc_store_attach = (loc_nonfuel_baskets / loc_baskets.clip(lower=1)).fillna(0.10)
    loc_weekend_share = (loc_weekend_baskets / loc_baskets.clip(lower=1)).fillna(0.25)
    loc_log_baskets = np.log1p(loc_baskets.astype("float64"))

    cust_top_loc = b.groupby(["ID", "loc"]).size().reset_index(name="cnt").sort_values(["ID", "cnt"], ascending=[True, False]).groupby("ID")["loc"].first()

    med_litres = float(loc_mean_litres.median()) if len(loc_mean_litres) > 0 else 21.0
    med_attach = float(loc_store_attach.median()) if len(loc_store_attach) > 0 else 0.10
    med_weekend = float(loc_weekend_share.median()) if len(loc_weekend_share) > 0 else 0.25
    med_log_vol = float(loc_log_baskets.median()) if len(loc_log_baskets) > 0 else 5.0

    out["m6_top_loc_mean_litres"] = cust_top_loc.map(loc_mean_litres).reindex(ids).fillna(med_litres)
    out["m6_top_loc_store_attach"] = cust_top_loc.map(loc_store_attach).reindex(ids).fillna(med_attach)
    out["m6_top_loc_weekend_share"] = cust_top_loc.map(loc_weekend_share).reindex(ids).fillna(med_weekend)
    out["m6_top_loc_log_volume"] = cust_top_loc.map(loc_log_baskets).reindex(ids).fillna(med_log_vol)

    out["m6_weekend_share"] = g["weekend"].mean()
    out["m6_mean_hour"] = g["hour"].mean()
    out["m6_std_hour"] = g["hour"].std()
    out["m6_cross_share"] = g["cross"].mean()
    items_per_basket = core.groupby(["ID", "TransactionId"]).size()
    out["m6_mean_basket_items"] = items_per_basket.groupby("ID").mean().reindex(ids).fillna(1.0)

    # Fuel fill-up volume and variance per transaction
    fuel_lines = core[core["is_fuel_sale"]]
    if len(fuel_lines) > 0:
        fuel_baskets = fuel_lines.groupby(["ID", "TransactionId"])["qty"].sum()
        g_fq = fuel_baskets.groupby("ID")
        out["m6_fuel_mean_litres"] = g_fq.mean().reindex(ids).fillna(med_litres)
        out["m6_fuel_std_litres"] = g_fq.std().reindex(ids).fillna(0.0)
    else:
        out["m6_fuel_mean_litres"] = med_litres
        out["m6_fuel_std_litres"] = 0.0

    # Commuter peak-hour share and day-of-week entropy
    if len(b) > 0:
        is_commuter = (b["time"].dt.dayofweek < 5) & b["time"].dt.hour.isin([6, 7, 8, 16, 17, 18])
        out["m6_commuter_share"] = is_commuter.groupby(b["ID"]).mean().reindex(ids).fillna(0.25)
        dow_counts = b.groupby(["ID", b["time"].dt.dayofweek]).size().unstack(fill_value=0).reindex(ids, fill_value=0)
        dow_p = dow_counts.div(dow_counts.sum(axis=1).clip(lower=1), axis=0)
        out["m6_dow_entropy"] = -(dow_p * np.log2(dow_p + 1e-6)).sum(axis=1).fillna(2.0)
    else:
        out["m6_commuter_share"] = 0.25
        out["m6_dow_entropy"] = 2.0

    # Promotional discounts & modifier savings
    mod_col = "modifier_amount" if "modifier_amount" in w.columns else "TotalModifierAmount"
    if mod_col in w.columns:
        w_mod = pd.to_numeric(w[mod_col], errors="coerce").fillna(0.0)
        hist_mod = pd.to_numeric(hist[mod_col], errors="coerce").fillna(0.0)
    else:
        w_mod = pd.Series(0.0, index=w.index)
        hist_mod = pd.Series(0.0, index=hist.index)
    has_disc = w_mod > 0.0
    w_disc = w[has_disc]
    disc_lines = w_disc.groupby("ID").size().reindex(ids).fillna(0.0)
    tot_lines = w.groupby("ID").size().reindex(ids).fillna(1.0)
    out["m6_discount_lines"] = disc_lines
    out["m6_discount_line_share"] = (disc_lines / tot_lines.clip(lower=1.0)).fillna(0.0)
    out["m6_discount_savings_rand"] = pd.to_numeric(w_disc[mod_col], errors="coerce").groupby(w_disc["ID"]).sum().reindex(ids).fillna(0.0)
    out["ever_discount_shopper"] = (
        hist[hist_mod > 0.0].groupby("ID").size().reindex(ids).fillna(0).gt(0).astype("float64")
    )
    excluded = core[~core["is_included"]]
    out["m6_excluded_baskets"] = excluded.groupby("ID")["TransactionId"].nunique()
    out["m6_excluded_rand"] = excluded.groupby("ID")["amount"].sum()
    for item in ACTIVE_ADJUSTMENT_ITEMS:
        tag = safe_name(item)
        out[f"ever_adj_{tag}"] = hist[hist["ItemName"] == item].groupby("ID").size().gt(0).astype("float64")
        out[f"m6_adj_{tag}_lines"] = w[w["ItemName"] == item].groupby("ID").size()
    count_cols = [c for c in out.columns if not c.startswith(("m6_mean_hour", "m6_std_hour", "m6_top", "m6_weekend", "m6_cross", "m6_fuel_mean", "m6_commuter", "m6_dow"))]
    out[count_cols] = out[count_cols].fillna(0.0)
    return out


def _extract_rfm(
    sub_df: pd.DataFrame, ids: pd.Index, t: pd.Timestamp
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Point-in-time RFM extraction for a transaction subset."""
    n = len(ids)
    if sub_df.empty:
        return np.zeros(n), np.zeros(n), np.ones(n), np.zeros(n), np.zeros(n, dtype=bool)

    baskets = sub_df.groupby(["ID", "TransactionId"]).agg(
        time=("_time", "min"),
        amount=("amount", "sum"),
    ).reset_index().sort_values(["ID", "time"])

    first_time = baskets.groupby("ID")["time"].min().reindex(ids)
    last_time = baskets.groupby("ID")["time"].max().reindex(ids)
    n_baskets = baskets.groupby("ID").size().reindex(ids).fillna(0)

    has_baskets = (n_baskets > 0).to_numpy(dtype=bool)
    x = (n_baskets - 1).clip(lower=0).to_numpy(dtype=float)
    t_x = np.where(has_baskets, (last_time - first_time) / DAY / DAYS_PER_MONTH, 0.0)
    T = np.where(has_baskets, (t - first_time) / DAY / DAYS_PER_MONTH, 1.0)
    T = np.clip(T, 0.01, None)

    total_spend = baskets.groupby("ID")["amount"].sum().reindex(ids).fillna(0)
    first_spend = baskets.groupby("ID").first()["amount"].reindex(ids).fillna(0)
    repeat_spend = (total_spend - first_spend).to_numpy(dtype=float)
    m_x = np.where(x > 0, repeat_spend / np.maximum(x, 1.0), 0.0)
    return x, t_x, T, m_x, has_baskets


def _fit_bgnbd(
    x: np.ndarray, t_x: np.ndarray, T: np.ndarray, penalizer: float = 0.001
) -> tuple[float, float, float, float]:
    """Fit BG/NBD parameters (r, alpha, a, b) via log-space L-BFGS-B optimization."""
    mask = T > 0
    if mask.sum() < 20:
        return 1.0, 1.0, 1.0, 10.0
    x_sub, tx_sub, T_sub = x[mask], t_x[mask], T[mask]

    def neg_ll(log_params: np.ndarray) -> float:
        r, alpha, a, b = np.exp(log_params)
        ln_A1 = sp.gammaln(r + x_sub) - sp.gammaln(r) + r * np.log(alpha)
        ln_B1 = sp.betaln(a, b + x_sub) - sp.betaln(a, b) - (r + x_sub) * np.log(alpha + T_sub)
        has_repeat = x_sub > 0
        ln_B2 = np.zeros_like(x_sub)
        if np.any(has_repeat):
            ln_B2[has_repeat] = (
                sp.betaln(a + 1, b + x_sub[has_repeat] - 1)
                - sp.betaln(a, b)
                - (r + x_sub[has_repeat]) * np.log(alpha + tx_sub[has_repeat])
            )
        max_B = np.maximum(ln_B1, np.where(has_repeat, ln_B2, -np.inf))
        sum_exp = np.exp(ln_B1 - max_B) + np.where(has_repeat, np.exp(ln_B2 - max_B), 0.0)
        log_sum = max_B + np.log(np.maximum(sum_exp, 1e-15))
        ll = ln_A1 + log_sum
        pen = penalizer * (r**2 + alpha**2 + a**2 + b**2)
        return float(-np.sum(ll) + pen)

    try:
        res = minimize(
            neg_ll,
            [0.0, 0.0, 0.0, 1.0],
            method="L-BFGS-B",
            bounds=[(-4.0, 4.0)] * 4,
            options={"maxiter": 80, "ftol": 1e-5},
        )
        params = np.exp(res.x)
        return float(params[0]), float(params[1]), float(params[2]), float(params[3])
    except Exception:
        return 1.0, 1.0, 1.0, 10.0


def _predict_bgnbd_stats(
    r: float,
    alpha: float,
    a: float,
    b: float,
    x: np.ndarray,
    t_x: np.ndarray,
    T: np.ndarray,
    has_baskets: np.ndarray,
    t_future: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute P(Alive), latent velocity lambda, and expected repeat purchases."""
    log_div = (r + x) * np.log((alpha + T) / (alpha + t_x)) + np.log(
        a / (b + np.maximum(x, 1.0) - 1.0)
    )
    # For x == 0 (single-transaction cohort), dropout is not conditioned on repeat timing;
    # apply Pareto-II survival decay (alpha / (alpha + T)) ** r instead of hardcoded 1.0
    survival_x0 = (alpha / (alpha + T)) ** r
    p_alive = np.where(x == 0, survival_x0, sp.expit(-log_div))
    p_alive = np.where(has_baskets, np.clip(p_alive, 0.0, 1.0), 0.0)
    lam = np.where(has_baskets, (r + x) / (alpha + T), 0.0)
    exp_purchases = p_alive * lam * t_future
    return p_alive, lam, exp_purchases


def _fit_gamma_gamma(x: np.ndarray, m_x: np.ndarray) -> tuple[float, float, float]:
    """Fit Gamma-Gamma monetary model parameters (p, q, v)."""
    mask = (x > 0) & (m_x > 0)
    if mask.sum() < 20:
        return 1.0, 2.5, 200.0
    x_sub, m_sub = x[mask], m_x[mask]

    def gg_neg_ll(params: np.ndarray) -> float:
        p, q, v = params
        ll = (
            sp.gammaln(p * x_sub + q)
            - sp.gammaln(p * x_sub)
            - sp.gammaln(q)
            + q * np.log(v)
            + (p * x_sub - 1.0) * np.log(m_sub)
            + p * x_sub * np.log(x_sub)
            - (p * x_sub + q) * np.log(x_sub * m_sub + v)
        )
        return float(-np.sum(ll))

    try:
        res = minimize(
            gg_neg_ll,
            [1.0, 2.5, 200.0],
            bounds=[(0.01, 20.0), (1.01, 20.0), (0.1, 2000.0)],
            method="L-BFGS-B",
            options={"maxiter": 60, "ftol": 1e-5},
        )
        return float(res.x[0]), float(res.x[1]), float(res.x[2])
    except Exception:
        return 1.0, 2.5, 200.0


def _predict_gamma_gamma_spend(
    p: float, q: float, v: float, x: np.ndarray, m_x: np.ndarray, has_baskets: np.ndarray
) -> np.ndarray:
    """Compute empirical Bayes expected spend per basket E[M]."""
    pop_mean = (v * p) / max(q - 1.0, 1e-4)
    weight = (p * x) / (p * x + max(q - 1.0, 1e-4))
    exp_spend = weight * m_x + (1.0 - weight) * pop_mean
    return np.where(has_baskets, exp_spend, 0.0)


def _btyd_block(hist: pd.DataFrame, ids: pd.Index, t: pd.Timestamp) -> pd.DataFrame:
    """Buy-Till-You-Die (BG/NBD & Gamma-Gamma) probabilistic customer base priors."""
    # 1. Overall qualifying transactions
    q = hist[hist["is_qualifying"]]
    x_all, tx_all, T_all, m_all, has_all = _extract_rfm(q, ids, t)
    r_all, a_all, a_p_all, b_all = _fit_bgnbd(x_all, tx_all, T_all)
    p_alive_all, lam_all, exp_b_all = _predict_bgnbd_stats(
        r_all, a_all, a_p_all, b_all, x_all, tx_all, T_all, has_all
    )

    # 2. Fuel forecourt transactions
    q_fuel = q[q["is_fuel_cat"]]
    x_f, tx_f, T_f, _, has_f = _extract_rfm(q_fuel, ids, t)
    r_f, a_f, a_p_f, b_f = _fit_bgnbd(x_f, tx_f, T_f)
    p_alive_f, _, exp_b_f = _predict_bgnbd_stats(r_f, a_f, a_p_f, b_f, x_f, tx_f, T_f, has_f)

    # 3. Nonfuel (store) transactions
    q_nf = q[q["is_nonfuel"]]
    x_nf, tx_nf, T_nf, _, has_nf = _extract_rfm(q_nf, ids, t)
    r_nf, a_nf, a_p_nf, b_nf = _fit_bgnbd(x_nf, tx_nf, T_nf)
    p_alive_nf, _, exp_b_nf = _predict_bgnbd_stats(r_nf, a_nf, a_p_nf, b_nf, x_nf, tx_nf, T_nf, has_nf)

    # 4. Gamma-Gamma spend model
    p_gg, q_gg, v_gg = _fit_gamma_gamma(x_all, m_all)
    exp_basket_spend = _predict_gamma_gamma_spend(p_gg, q_gg, v_gg, x_all, m_all, has_all)
    exp_clv_m3 = exp_b_all * exp_basket_spend

    out = pd.DataFrame(index=ids)
    out["btyd_overall_p_alive"] = p_alive_all
    out["btyd_overall_lambda"] = lam_all
    out["btyd_overall_exp_baskets_m3"] = exp_b_all
    out["btyd_fuel_p_alive"] = p_alive_f
    out["btyd_fuel_exp_baskets_m3"] = exp_b_f
    out["btyd_nonfuel_p_alive"] = p_alive_nf
    out["btyd_nonfuel_exp_baskets_m3"] = exp_b_nf
    out["btyd_exp_monetary_spend"] = exp_basket_spend
    out["btyd_exp_clv_spend_m3"] = exp_clv_m3
    return out


def _hurdle_physics_block(feats: pd.DataFrame) -> pd.DataFrame:
    """Statistical hurdle physics, tail exceedance Z-scores, and category competition features."""
    out = pd.DataFrame(index=feats.index)

    # 1. Fuel Lags & Volatility
    f_lags = feats[["m1_fuel_litres", "fuel_litres_lag2", "fuel_litres_lag3", "fuel_litres_lag4", "fuel_litres_lag5", "fuel_litres_lag6"]]
    f_std = f_lags.std(axis=1).fillna(0.0)
    f_mean = f_lags.mean(axis=1).fillna(0.0)
    f_cv = (f_std / (f_mean + 1e-4)).fillna(0.0)

    m3_f_rand = feats["m3_fuel_rand"]
    q2_f_rand = feats["q2_fuel_rand"]
    m6_f_rand = feats["m6_fuel_rand"]
    life_f = feats["fuel_litres_per_month_life"]
    rel_p = feats.get("rel_fuel_price_m3", pd.Series(1.0, index=feats.index))

    m3_f_lit = feats["m3_fuel_litres"]
    h_f_lit = np.maximum(m3_f_lit + 1.0, 1.25 * m3_f_lit)
    z_f = (h_f_lit - f_mean * 3.0) / (f_std * np.sqrt(3.0) + 1.0)

    f_hurdle = np.maximum(m3_f_rand + 20.0, 1.25 * m3_f_rand)

    out["fuel_volatility_cv"] = f_cv
    out["fuel_monthly_std"] = f_std
    out["fuel_hurdle_zscore"] = z_f
    out["fuel_hurdle_rand"] = f_hurdle
    out["fuel_attainability_q2"] = (q2_f_rand + 1.0) / (f_hurdle + 1.0)
    out["fuel_attainability_life"] = (life_f * 3.0 * rel_p * 23.0 + 1.0) / (f_hurdle + 1.0)
    out["fuel_rebound_ratio"] = (m6_f_rand / 2.0 + 1.0) / (m3_f_rand + 1.0)
    out["fuel_headroom_rand"] = f_hurdle - m3_f_rand

    # 2. Nonfuel Lags & Volatility
    nf_lags = feats[["m1_nonfuel_rand", "nonfuel_rand_lag2", "nonfuel_rand_lag3", "nonfuel_rand_lag4", "nonfuel_rand_lag5", "nonfuel_rand_lag6"]]
    nf_std = nf_lags.std(axis=1).fillna(0.0)
    nf_mean = nf_lags.mean(axis=1).fillna(0.0)
    nf_cv = (nf_std / (nf_mean + 1.0)).fillna(0.0)

    m3_nf_rand = feats["m3_nonfuel_rand"]
    q2_nf_rand = feats["q2_nonfuel_rand"]
    m6_nf_rand = feats["m6_nonfuel_rand"]
    life_nf = feats["nonfuel_rand_per_month_life"]

    nf_hurdle = np.maximum(m3_nf_rand + 20.0, 1.25 * m3_nf_rand)
    z_nf = (nf_hurdle - nf_mean * 3.0) / (nf_std * np.sqrt(3.0) + 1.0)

    out["nonfuel_volatility_cv"] = nf_cv
    out["nonfuel_monthly_std"] = nf_std
    out["nonfuel_hurdle_zscore"] = z_nf
    out["nonfuel_hurdle_rand"] = nf_hurdle
    out["nonfuel_attainability_q2"] = (q2_nf_rand + 1.0) / (nf_hurdle + 1.0)
    out["nonfuel_attainability_life"] = (life_nf * 3.0 + 1.0) / (nf_hurdle + 1.0)
    out["nonfuel_rebound_ratio"] = (m6_nf_rand / 2.0 + 1.0) / (m3_nf_rand + 1.0)
    out["nonfuel_headroom_rand"] = nf_hurdle - m3_nf_rand

    # 3. Major Retail Hurdles (Beverages & Food & beverage)
    m3_bev = feats["m3spend_beverages"] if "m3spend_beverages" in feats.columns else pd.Series(0.0, index=feats.index)
    H_bev = np.maximum(m3_bev + 20.0, 1.25 * m3_bev)
    out["bev_hurdle_rand"] = H_bev
    out["bev_headroom_rand"] = H_bev - m3_bev

    m3_food = feats["m3spend_food_beverage"] if "m3spend_food_beverage" in feats.columns else pd.Series(0.0, index=feats.index)
    H_food = np.maximum(m3_food + 20.0, 1.25 * m3_food)
    out["food_hurdle_rand"] = H_food
    out["food_headroom_rand"] = H_food - m3_food

    # 4. Relative Z-score spread between fuel and non-fuel hurdles
    out["z_diff_fuel_nonfuel"] = z_f - z_nf

    # 5. Category Competition Across All 24 Included Spend Categories
    other_cats = [
        "Automotive Products", "Braai & Ice", "Breads & rolls", "Car Wash", "Coffee",
        "Confectionary", "Frozen food", "Gas", "General merchandise", "Groceries",
        "Ice-cream", "Krispy Kreme", "Lubricants", "Miscellaneous", "NA",
        "Perishables", "Promotional items", "Seattle", "Snacking", "Tobacco", "Unknown",
    ]
    all_24_cats = ["Fuel", "Beverages", "Food & beverage"] + other_cats

    max_delta_rand = np.zeros(len(feats), dtype=np.float64)
    max_hurdle_ratio = np.zeros(len(feats), dtype=np.float64)
    n_clearing = np.zeros(len(feats), dtype=np.float64)
    n_visited = np.zeros(len(feats), dtype=np.float64)
    cat_increments: dict[str, np.ndarray] = {}

    for c in all_24_cats:
        c_safe = safe_name(c)
        col_m3 = f"m3spend_{c_safe}"
        if col_m3 in feats.columns:
            m3_c = feats[col_m3].fillna(0.0).to_numpy(dtype=np.float64)
            col_m1 = f"m1spend_{c_safe}"
            m1_c = feats[col_m1].fillna(0.0).to_numpy(dtype=np.float64) if col_m1 in feats.columns else np.zeros(len(feats))
            col_q2 = f"q2spend_{c_safe}"
            q2_c = feats[col_q2].fillna(3.0 * pd.Series(m1_c, index=feats.index)).to_numpy(dtype=np.float64) if col_q2 in feats.columns else (3.0 * m1_c)
            col_ever = f"ever_{c_safe}"
            ever_c = (feats[col_ever].fillna(0.0).to_numpy() > 0) if col_ever in feats.columns else (m3_c > 0)

            H_c = np.maximum(1.25 * m3_c, m3_c + 20.0)
            proj_spend = np.maximum(3.0 * m1_c, 0.5 * (3.0 * m1_c + q2_c))
            delta_rand = np.where(ever_c & (proj_spend >= H_c), proj_spend - m3_c, 0.0)
            ratio_c = np.where(ever_c, (3.0 * m1_c + 1.0) / (H_c + 1.0), 0.0)

            cat_increments[c] = delta_rand
            n_clearing += (ever_c & (proj_spend >= H_c)).astype(np.float64)
            n_visited += ever_c.astype(np.float64)

            if c in other_cats:
                max_delta_rand = np.maximum(max_delta_rand, delta_rand)
                max_hurdle_ratio = np.maximum(max_hurdle_ratio, ratio_c)
        else:
            cat_increments[c] = np.zeros(len(feats), dtype=np.float64)

    out["max_other_subcat_expected_delta_rand"] = max_delta_rand
    out["max_other_subcat_hurdle_ratio"] = max_hurdle_ratio
    out["rule6_exp_inc_fuel"] = cat_increments["Fuel"]
    out["rule6_exp_inc_beverages"] = cat_increments["Beverages"]
    out["rule6_exp_inc_food_beverage"] = cat_increments["Food & beverage"]
    out["rule6_exp_inc_other_growth"] = max_delta_rand
    best_retail_inc = np.maximum.reduce([
        cat_increments["Beverages"],
        cat_increments["Food & beverage"],
        max_delta_rand,
    ])
    out["rule6_fuel_minus_best_retail_inc"] = cat_increments["Fuel"] - best_retail_inc
    out["rule6_n_categories_clearing_hurdle"] = n_clearing
    out["rule6_n_unvisited_categories"] = 24.0 - n_visited

    if "is_summer_outcome_window" in feats.columns and "station_summer_fuel_index" in feats.columns:
        is_sum = feats["is_summer_outcome_window"].to_numpy() > 0.5
        out["seasonal_expected_fuel_litres"] = feats["m3_fuel_litres"].to_numpy() * np.where(
            is_sum, feats["station_summer_fuel_index"].to_numpy(), 1.0
        )
        out["seasonal_expected_nf_rand"] = feats["m3_nonfuel_rand"].to_numpy() * np.where(
            is_sum, feats["station_summer_nf_index"].to_numpy(), 1.0
        )

    return out


def _longitudinal_panel_block(win: pd.DataFrame, recency: pd.DataFrame, ids: pd.Index) -> pd.DataFrame:
    """Longitudinal lagged outcomes, target-space anchors, tenure censoring, and hurdle clearing dynamics."""
    out = pd.DataFrame(index=ids)
    s_f = 2.352825386185113
    s_nf = 2.4200904988927476

    out["lag1_clv_fuel"] = np.log1p(win["m3_fuel_litres"].clip(lower=0.0)) / s_f
    out["lag1_clv_nonfuel"] = np.log1p(win["m3_nonfuel_rand"].clip(lower=0.0)) / s_nf
    out["lag2_clv_fuel"] = np.log1p(win["q2_fuel_litres"].clip(lower=0.0)) / s_f
    out["lag2_clv_nonfuel"] = np.log1p(win["q2_nonfuel_rand"].clip(lower=0.0)) / s_nf
    out["delta_lag_clv_fuel"] = out["lag1_clv_fuel"] - out["lag2_clv_fuel"]
    out["delta_lag_clv_nonfuel"] = out["lag1_clv_nonfuel"] - out["lag2_clv_nonfuel"]

    out["proj_m1_clv_fuel"] = np.log1p(3.0 * win["m1_fuel_litres"].clip(lower=0.0)) / s_f
    out["proj_m1_clv_nonfuel"] = np.log1p(3.0 * win["m1_nonfuel_rand"].clip(lower=0.0)) / s_nf
    out["proj_d45_clv_fuel"] = np.log1p(2.0 * win["d45_fuel_litres"].clip(lower=0.0)) / s_f
    out["proj_d45_clv_nonfuel"] = np.log1p(2.0 * win["d45_nonfuel_rand"].clip(lower=0.0)) / s_nf
    out["proj_m6_clv_fuel"] = np.log1p(0.5 * win["m6_fuel_litres"].clip(lower=0.0)) / s_f
    out["proj_m6_clv_nonfuel"] = np.log1p(0.5 * win["m6_nonfuel_rand"].clip(lower=0.0)) / s_nf

    l2_f_fill = out["lag2_clv_fuel"].fillna(out["lag1_clv_fuel"])
    m6_f_fill = out["proj_m6_clv_fuel"].fillna(out["lag1_clv_fuel"])
    out["z_anchor_fuel_mean"] = (
        0.45 * out["lag1_clv_fuel"] + 0.25 * out["proj_m1_clv_fuel"] + 0.15 * l2_f_fill + 0.15 * m6_f_fill
    )
    z_f_mat = np.column_stack([
        out["lag1_clv_fuel"].to_numpy(),
        out["proj_m1_clv_fuel"].to_numpy(),
        l2_f_fill.to_numpy(),
        m6_f_fill.to_numpy(),
    ])
    out["z_anchor_fuel_std"] = np.nanstd(z_f_mat, axis=1)
    out["blup_empirical_innov_fuel"] = out["lag1_clv_fuel"] - 0.75 * l2_f_fill

    l2_nf_fill = out["lag2_clv_nonfuel"].fillna(out["lag1_clv_nonfuel"])
    m6_nf_fill = out["proj_m6_clv_nonfuel"].fillna(out["lag1_clv_nonfuel"])
    out["z_anchor_nf_mean"] = (
        0.45 * out["lag1_clv_nonfuel"] + 0.25 * out["proj_m1_clv_nonfuel"] + 0.15 * l2_nf_fill + 0.15 * m6_nf_fill
    )
    z_nf_mat = np.column_stack([
        out["lag1_clv_nonfuel"].to_numpy(),
        out["proj_m1_clv_nonfuel"].to_numpy(),
        l2_nf_fill.to_numpy(),
        m6_nf_fill.to_numpy(),
    ])
    out["z_anchor_nf_std"] = np.nanstd(z_nf_mat, axis=1)
    out["blup_empirical_innov_nf"] = out["lag1_clv_nonfuel"] - 0.70 * l2_nf_fill

    tenure = recency["days_since_first_seen_capped"]
    out["tenure_days_capped_90"] = np.minimum(tenure, 90.0)
    annual_mult = 90.0 / np.maximum(np.minimum(tenure, 90.0), 7.0)
    out["m3_annualized_rate_fuel"] = win["m3_fuel_litres"] * annual_mult
    out["m3_annualized_rate_nonfuel"] = win["m3_nonfuel_rand"] * annual_mult

    cleared = (
        (win["m3_fuel_litres"] > 1.25 * win["q2_fuel_litres"])
        & ((win["m3_fuel_litres"] - win["q2_fuel_litres"]) >= 1.0)
    ).astype("float64")
    dip_f = (win["m3_fuel_litres"] + 5.0) / (np.maximum(win["q2_fuel_litres"], 5.0) + 5.0)
    dip_nf = (win["m3_nonfuel_rand"] + 50.0) / (np.maximum(win["q2_nonfuel_rand"], 50.0) + 50.0)

    q2_nan_f = win["q2_fuel_litres"].isna()
    q2_nan_nf = win["q2_nonfuel_rand"].isna()
    cleared = cleared.where(~q2_nan_f, np.nan)
    dip_f = dip_f.where(~q2_nan_f, np.nan)
    dip_nf = dip_nf.where(~q2_nan_nf, np.nan)

    out["cleared_lag_fuel_growth"] = cleared
    out["dip_ratio_fuel"] = dip_f
    out["dip_ratio_nonfuel"] = dip_nf
    return out


def _granular_basket_block(hist: pd.DataFrame, ids: pd.Index, t: pd.Timestamp) -> pd.DataFrame:
    """Forecourt SKU habituality, fill properties, and basket structure."""
    out = pd.DataFrame(index=ids)
    # 1. Fuel fill properties in m6
    fuel_tx = hist[hist["is_fuel_sale"] & (hist["_time"] >= t - pd.DateOffset(months=6))]
    out["max_single_fill_litres"] = fuel_tx.groupby("ID")["qty"].max().reindex(ids).fillna(0.0)
    if len(fuel_tx) > 0 and "ID" in fuel_tx.columns:
        is_round = ((fuel_tx["_amount"] % 5000 == 0) | (fuel_tx["_amount"] % 10000 == 0)).astype(float)
        round_share = (
            fuel_tx.assign(is_round=is_round)
            .groupby("ID")["is_round"]
            .mean()
            .reindex(ids)
            .fillna(0.0)
        )
    else:
        round_share = pd.Series(0.0, index=ids)
    out["full_tank_fill_share"] = 1.0 - round_share

    # 2. SKU Repeat rate on non-fuel merchandise
    if "StockItemId" in hist.columns:
        store_tx = hist[hist["is_nonfuel"] & hist["is_qualifying"] & (hist["_time"] >= t - pd.DateOffset(months=6))]
        unique_skus = store_tx.groupby("ID")["StockItemId"].nunique().reindex(ids).fillna(0.0)
        total_skus = store_tx.groupby("ID")["StockItemId"].count().reindex(ids).fillna(0.0)
        out["sku_repeat_rate"] = np.where(total_skus > 1, 1.0 - (unique_skus / total_skus), 0.0)
    else:
        out["sku_repeat_rate"] = 0.0

    # 3. Line sequence max per basket
    if "LineItemSequence" in hist.columns and len(hist) > 0 and "ID" in hist.columns:
        seq_num = pd.to_numeric(hist["LineItemSequence"], errors="coerce").fillna(1.0)
        baskets_max_seq = hist.assign(seq_num=seq_num).groupby(["ID", "TransactionId"])["seq_num"].max().reset_index()
        out["mean_line_sequence_max"] = baskets_max_seq.groupby("ID")["seq_num"].mean().reindex(ids).fillna(1.0)
        is_multi = (baskets_max_seq["seq_num"] > 1).astype("float64")
        out["multi_item_basket_share"] = (
            baskets_max_seq.assign(is_multi=is_multi)
            .groupby("ID")["is_multi"]
            .mean()
            .reindex(ids)
            .fillna(0.0)
        )
    else:
        out["mean_line_sequence_max"] = 1.0
        out["multi_item_basket_share"] = 0.0

    # 4. Total wallet share: in-scope spend vs non-adjustment spend in consistent units
    if len(hist) > 0 and "ID" in hist.columns:
        in_scope_sum = hist[hist["in_scope"]].groupby("ID")["amount"].sum().reindex(ids).fillna(0.0)
        non_adj_sum = hist[~hist["is_adjustment"]].groupby("ID")["amount"].sum().reindex(ids).fillna(0.0)
        out["wallet_in_scope_share"] = np.where(
            non_adj_sum > 0,
            np.clip(in_scope_sum / non_adj_sum, 0.0, 1.0),
            1.0,
        )
    else:
        out["wallet_in_scope_share"] = 1.0

    return out


def _refueling_burn_and_payday_block(hist: pd.DataFrame, ids: pd.Index, t: pd.Timestamp) -> pd.DataFrame:
    """Inter-fill burn rates, tank depletion proxies, and South African payday liquidity shares."""
    out = pd.DataFrame(index=ids)

    # 1. Refueling physics in past 6 months
    fuel_tx = hist[hist["is_fuel_sale"] & (hist["_time"] >= t - pd.DateOffset(months=6))]
    if len(fuel_tx) > 0 and "ID" in fuel_tx.columns:
        fuel_times = fuel_tx.groupby("ID")["_time"].agg(["min", "max", "count"])
        time_span_days = (fuel_times["max"] - fuel_times["min"]).dt.total_seconds() / 86400.0
        tot_litres = fuel_tx.groupby("ID")["qty"].sum()
        burn_rate = (tot_litres / np.maximum(time_span_days, 7.0)).reindex(ids).fillna(0.0)
        days_since_fuel = ((t - fuel_times["max"]).dt.total_seconds() / 86400.0).reindex(ids).fillna(999.0)
        cycle_days = (time_span_days / np.maximum(fuel_times["count"] - 1, 1)).reindex(ids).fillna(14.0).clip(3.0, 60.0)
        depletion = np.clip(days_since_fuel / cycle_days, 0.0, 3.0)
        max_fill = fuel_tx.groupby("ID")["qty"].max().reindex(ids).fillna(0.0)
        p90_fill = fuel_tx.groupby("ID")["qty"].quantile(0.90).reindex(ids).fillna(0.0)
    else:
        burn_rate = pd.Series(0.0, index=ids)
        depletion = pd.Series(0.0, index=ids)
        max_fill = pd.Series(0.0, index=ids)
        p90_fill = pd.Series(0.0, index=ids)

    out["refuel_burn_rate_daily"] = burn_rate
    out["tank_depletion_state"] = depletion
    out["tank_max_fill_proxy"] = max_fill
    out["tank_p90_fill_proxy"] = p90_fill

    # 2. South African payday window liquidity shares (25th-2nd and 15th)
    m3_tx = hist[hist["_time"] >= t - pd.DateOffset(months=3)]
    if len(m3_tx) > 0 and "ID" in m3_tx.columns:
        day = m3_tx["_time"].dt.day
        is_payday = (day >= 25) | (day <= 2) | ((day >= 14) & (day <= 16))
        payday_spend = m3_tx[is_payday].groupby("ID")["amount"].sum().reindex(ids).fillna(0.0)
        tot_spend_m3 = m3_tx.groupby("ID")["amount"].sum().reindex(ids).fillna(0.0)
        out["payday_spend_share_m3"] = np.where(tot_spend_m3 > 0, payday_spend / tot_spend_m3, 0.0)

        payday_baskets = m3_tx[is_payday].groupby("ID")["TransactionId"].nunique().reindex(ids).fillna(0.0)
        tot_baskets_m3 = m3_tx.groupby("ID")["TransactionId"].nunique().reindex(ids).fillna(0.0)
        out["payday_basket_share_m3"] = np.where(tot_baskets_m3 > 0, payday_baskets / tot_baskets_m3, 0.0)

        fuel_payday = m3_tx[is_payday & m3_tx["is_fuel_cat"]].groupby("ID")["amount"].sum().reindex(ids).fillna(0.0)
        fuel_tot = m3_tx[m3_tx["is_fuel_cat"]].groupby("ID")["amount"].sum().reindex(ids).fillna(0.0)
        out["payday_fuel_spend_share_m3"] = np.where(fuel_tot > 0, fuel_payday / fuel_tot, 0.0)
    else:
        out["payday_spend_share_m3"] = 0.0
        out["payday_basket_share_m3"] = 0.0
        out["payday_fuel_spend_share_m3"] = 0.0

    return out


def _hawkes_decay_block(hist: pd.DataFrame, ids: pd.Index, t: pd.Timestamp) -> pd.DataFrame:
    """Multi-scale continuous-time exponential decay intensities and acceleration ratios."""
    out = pd.DataFrame(index=ids)
    w_m6 = hist[hist["_time"] >= t - pd.DateOffset(months=6)]
    if len(w_m6) == 0 or "ID" not in w_m6.columns:
        for tau in (7, 14, 45, 135):
            out[f"hawkes_intensity_{tau}d"] = 0.0
            out[f"hawkes_fuel_{tau}d"] = 0.0
            out[f"hawkes_nonfuel_{tau}d"] = 0.0
        out["hawkes_ratio_7_45"] = 0.0
        out["hawkes_ratio_14_135"] = 0.0
        out["hawkes_fuel_accel"] = 0.0
        out["hawkes_nonfuel_accel"] = 0.0
        return out

    dt_days = (t - w_m6["_time"]).dt.total_seconds() / 86400.0
    intensities: dict[int, pd.Series] = {}
    f_intensities: dict[int, pd.Series] = {}
    nf_intensities: dict[int, pd.Series] = {}

    for tau in (7, 14, 45, 135):
        w = np.exp(-dt_days / tau)
        w_df = w_m6.assign(w=w)
        int_all = w_df.groupby("ID")["w"].sum().reindex(ids).fillna(0.0)
        int_f = w_df[w_df["is_fuel_sale"]].assign(wf=w[w_df["is_fuel_sale"]]).groupby("ID")["wf"].sum().reindex(ids).fillna(0.0) if w_df["is_fuel_sale"].any() else pd.Series(0.0, index=ids)
        int_nf = w_df[w_df["is_nonfuel"]].assign(wnf=w[w_df["is_nonfuel"]]).groupby("ID")["wnf"].sum().reindex(ids).fillna(0.0) if w_df["is_nonfuel"].any() else pd.Series(0.0, index=ids)

        out[f"hawkes_intensity_{tau}d"] = int_all
        out[f"hawkes_fuel_{tau}d"] = int_f
        out[f"hawkes_nonfuel_{tau}d"] = int_nf
        intensities[tau] = int_all
        f_intensities[tau] = int_f
        nf_intensities[tau] = int_nf

    out["hawkes_ratio_7_45"] = intensities[7] / (intensities[45] + 1e-4)
    out["hawkes_ratio_14_135"] = intensities[14] / (intensities[135] + 1e-4)
    out["hawkes_fuel_accel"] = f_intensities[7] / (f_intensities[45] + 1e-4)
    out["hawkes_nonfuel_accel"] = nf_intensities[7] / (nf_intensities[45] + 1e-4)
    return out


GLOBAL_TOP_35_LEVEL3: tuple[str, ...] = (
    "Petrol", "Diesel", "Soft drinks", "Energy", "Water", "FPG Pies", "Chips", "Chocolate",
    "Breads & rolls", "Gums & rolls", "Cigarettes", "Unknown", "FPG Meals", "Lubricants - Lubricants",
    "Edible groceries - Pharmaceutical Edible", "Sport drinks", "Juice", "Ice-Cream", "Coffee",
    "Prepacks", "FPG Confectionary", "Biscuits", "Seattle", "NA", "FPG Sandwiches & Subs",
    "Dairy drinks", "Alternate snacking", "Health & energy", "Sugar based confectionary",
    "Freshly prepared_FPG", "Ice", "Edible groceries", "FPG Chicken full & pieces",
    "Tobacco products", "Bought in_BI",
)


def _level3_and_svd_block(hist: pd.DataFrame, ids: pd.Index, t: pd.Timestamp) -> pd.DataFrame:
    """Granular ItemCategoryLevel3 routine concentration (HHI) and spend summary statistics."""
    out = pd.DataFrame(index=ids)
    w_m6 = hist[(hist["_time"] >= t - pd.DateOffset(months=6)) & hist["in_scope"]]
    if len(w_m6) == 0 or "ItemCategoryLevel3" not in w_m6.columns or "ID" not in w_m6.columns:
        for cat in GLOBAL_TOP_35_LEVEL3:
            out[f"l3_share_{safe_name(cat)}"] = 0.0
        out["level3_hhi_concentration"] = 0.0
        out["l3_active_count"] = 0.0
        out["l3_top1_share"] = 0.0
        out["l3_top3_share"] = 0.0
        return out

    spend_matrix = w_m6.groupby(["ID", "ItemCategoryLevel3"])["amount"].sum().unstack(fill_value=0.0).reindex(ids).fillna(0.0)
    tot_spend = spend_matrix.sum(axis=1).replace(0.0, 1.0)
    shares = spend_matrix.div(tot_spend, axis=0)

    for cat in GLOBAL_TOP_35_LEVEL3:
        clean_cat = safe_name(cat)
        if cat in shares.columns:
            out[f"l3_share_{clean_cat}"] = shares[cat]
        else:
            out[f"l3_share_{clean_cat}"] = 0.0

    out["level3_hhi_concentration"] = (shares ** 2).sum(axis=1)
    out["l3_active_count"] = (spend_matrix > 0).sum(axis=1).astype("float64")
    sorted_shares = np.sort(shares.to_numpy(), axis=1)[:, ::-1]
    out["l3_top1_share"] = sorted_shares[:, 0].astype("float64")
    out["l3_top3_share"] = sorted_shares[:, :min(3, sorted_shares.shape[1])].sum(axis=1).astype("float64")

    return out


def _station_seasonal_block(hist: pd.DataFrame, ids: pd.Index, t: pd.Timestamp) -> pd.DataFrame:
    """Station-Level Historical Seasonal Multipliers and Behavioral Elasticity."""
    out = pd.DataFrame(index=ids)
    if "LocationId" not in hist.columns or len(hist) == 0:
        out["station_summer_fuel_index"] = 1.0
        out["station_summer_nf_index"] = 1.0
        out["station_summer_fg_rate"] = 0.5
        out["station_weekend_to_weekday_ratio"] = 1.0
        out["station_payday_ratio"] = 1.0
        out["station_large_fill_share"] = 0.15
        out["station_pump_price_m3"] = 23.50
        out["cat_primary_location_id"] = np.zeros(len(ids), dtype=np.int64)
        out["cat_top_nonfuel_level2"] = np.zeros(len(ids), dtype=np.int64)
        out["cat_top_nonfuel_level3"] = np.zeros(len(ids), dtype=np.int64)
        return out

    q = hist[hist["is_qualifying"]]
    if len(q) > 0:
        loc_counts = q.groupby(["ID", "LocationId"]).size().reset_index(name="count")
        primary_loc = (
            loc_counts.sort_values(["ID", "count"], ascending=[True, False])
            .drop_duplicates(subset=["ID"])
            .set_index("ID")["LocationId"]
            .reindex(ids)
        )
    else:
        primary_loc = pd.Series(np.nan, index=ids)

    # 1. Structural categorical: cat_primary_location_id
    loc_clean = pd.to_numeric(primary_loc, errors="coerce").fillna(0).astype("int64")
    out["cat_primary_location_id"] = (loc_clean % 100000).astype("int64")

    # 2. Structural categorical: cat_top_nonfuel_level2
    if "ItemCategoryLevel2" in hist.columns:
        nf_tx = hist[hist["is_nonfuel"] & hist["ItemCategoryLevel2"].notna()]
        if len(nf_tx) > 0:
            l2_counts = nf_tx.groupby(["ID", "ItemCategoryLevel2"]).size().reset_index(name="count")
            top_l2 = (
                l2_counts.sort_values(["ID", "count"], ascending=[True, False])
                .drop_duplicates(subset=["ID"])
                .set_index("ID")["ItemCategoryLevel2"]
                .reindex(ids)
            )
            out["cat_top_nonfuel_level2"] = top_l2.map(NONFUEL_L2_MAP).fillna(0).astype("int64")
        else:
            out["cat_top_nonfuel_level2"] = np.zeros(len(ids), dtype=np.int64)
    else:
        out["cat_top_nonfuel_level2"] = np.zeros(len(ids), dtype=np.int64)

    # 3. Structural categorical: cat_top_nonfuel_level3
    if "ItemCategoryLevel3" in hist.columns:
        nf_l3_tx = hist[hist["is_nonfuel"] & hist["ItemCategoryLevel3"].notna()]
        if len(nf_l3_tx) > 0:
            l3_counts = nf_l3_tx.groupby(["ID", "ItemCategoryLevel3"]).size().reset_index(name="count")
            top_l3 = (
                l3_counts.sort_values(["ID", "count"], ascending=[True, False])
                .drop_duplicates(subset=["ID"])
                .set_index("ID")["ItemCategoryLevel3"]
                .reindex(ids)
            )
            out["cat_top_nonfuel_level3"] = top_l3.map(NONFUEL_L3_MAP).fillna(0).astype("int64")
        else:
            out["cat_top_nonfuel_level3"] = np.zeros(len(ids), dtype=np.int64)
    else:
        out["cat_top_nonfuel_level3"] = np.zeros(len(ids), dtype=np.int64)

    # Station-level behavioral indicators derived solely from historical visits before cutoff
    fuel_tx = hist[hist["is_fuel_cat"]]
    if len(fuel_tx) > 0:
        is_weekend = fuel_tx["_time"].dt.weekday >= 5
        wknd_spend = fuel_tx[is_weekend].groupby("LocationId")["amount"].sum() / 2.0
        wkdy_spend = fuel_tx[~is_weekend].groupby("LocationId")["amount"].sum() / 5.0
        weekend_ratio = (wknd_spend / wkdy_spend.replace(0.0, np.nan)).clip(0.2, 5.0).fillna(1.0)
        out["station_weekend_to_weekday_ratio"] = primary_loc.map(weekend_ratio).fillna(1.0).astype("float64")
    else:
        out["station_weekend_to_weekday_ratio"] = 1.0

    is_payday = hist["_time"].dt.day >= 25
    payday_spend = hist[is_payday].groupby("LocationId")["amount"].sum() / 7.0
    normal_spend = hist[~is_payday].groupby("LocationId")["amount"].sum() / 23.5
    payday_ratio = (payday_spend / normal_spend.replace(0.0, np.nan)).clip(0.5, 4.0).fillna(1.0)
    out["station_payday_ratio"] = primary_loc.map(payday_ratio).fillna(1.0).astype("float64")

    fuel_sales = hist[hist["is_fuel_sale"] & (hist["qty"] > 0)]
    if len(fuel_sales) > 0:
        large_fills = fuel_sales[fuel_sales["qty"] > 45.0].groupby("LocationId")["TransactionId"].nunique()
        all_fills = fuel_sales.groupby("LocationId")["TransactionId"].nunique()
        large_share = (large_fills / all_fills.replace(0, np.nan)).clip(0.0, 1.0).fillna(0.15)
        out["station_large_fill_share"] = primary_loc.map(large_share).fillna(0.15).astype("float64")
    else:
        out["station_large_fill_share"] = 0.15

    is_summer = hist["_time"].dt.month.isin([12, 1, 2])
    summer_tx = hist[is_summer]
    nonsummer_tx = hist[~is_summer]
    summer_months = max(summer_tx["_time"].dt.to_period("M").nunique(), 1)
    nonsummer_months = max(nonsummer_tx["_time"].dt.to_period("M").nunique(), 1)

    if len(summer_tx) > 0 and len(nonsummer_tx) > 0:
        f_summer = summer_tx[summer_tx["is_fuel_cat"]].groupby("LocationId")["amount"].sum() / summer_months
        f_nonsummer = nonsummer_tx[nonsummer_tx["is_fuel_cat"]].groupby("LocationId")["amount"].sum() / nonsummer_months
        f_idx = (f_summer / f_nonsummer.replace(0.0, np.nan)).clip(0.5, 2.5).fillna(1.0)

        nf_summer = summer_tx[summer_tx["is_nonfuel"]].groupby("LocationId")["amount"].sum() / summer_months
        nf_nonsummer = nonsummer_tx[nonsummer_tx["is_nonfuel"]].groupby("LocationId")["amount"].sum() / nonsummer_months
        nf_idx = (nf_summer / nf_nonsummer.replace(0.0, np.nan)).clip(0.5, 2.5).fillna(1.0)

        fg_tx = summer_tx[summer_tx["is_fuel_cat"]].groupby("LocationId")["TransactionId"].nunique()
        tot_tx = summer_tx[summer_tx["is_qualifying"]].groupby("LocationId")["TransactionId"].nunique()
        fg_rate = (fg_tx / tot_tx.replace(0.0, np.nan)).clip(0.0, 1.0).fillna(0.5)
    else:
        f_idx = pd.Series(1.0, index=primary_loc.dropna().unique())
        nf_idx = pd.Series(1.0, index=primary_loc.dropna().unique())
        fg_rate = pd.Series(0.5, index=primary_loc.dropna().unique())

    # Station-level recent fuel pump price (empirical median per LocationId in m3)
    m3_fuel = hist[(hist["_time"] >= t - pd.DateOffset(months=3)) & hist["is_fuel_sale"] & (hist["qty"] > 2.0) & (hist["amount"] > 50.0)]
    if len(m3_fuel) > 0:
        st_price = (m3_fuel["amount"] / m3_fuel["qty"]).groupby(m3_fuel["LocationId"]).median().clip(19.0, 28.0)
        out["station_pump_price_m3"] = primary_loc.map(st_price).fillna(23.50).astype("float64")
    else:
        out["station_pump_price_m3"] = 23.50

    out["station_summer_fuel_index"] = primary_loc.map(f_idx).fillna(1.0).astype("float64")
    out["station_summer_nf_index"] = primary_loc.map(nf_idx).fillna(1.0).astype("float64")
    out["station_summer_fg_rate"] = primary_loc.map(fg_rate).fillna(0.5).astype("float64")

    angle = 2.0 * np.pi * (t.month - 1) / 12.0
    out["cutoff_sin_month"] = float(np.sin(angle))
    out["cutoff_cos_month"] = float(np.cos(angle))
    out["is_summer_outcome_window"] = float(1.0 if t.month in (11, 12, 1) else 0.0)
    return out


def _mundlak_block(
    hist: pd.DataFrame, win: pd.DataFrame, ids: pd.Index, observed_months: pd.Series
) -> pd.DataFrame:
    """Mundlak (1978) within-between panel econometric decomposition."""
    s = hist[hist["in_scope"]]
    out = pd.DataFrame(index=ids)

    fuel_litres_life = s[s["is_fuel_sale"]].groupby("ID")["qty"].sum().reindex(ids).fillna(0.0)
    nf_rand_life = s[s["is_nonfuel"]].groupby("ID")["amount"].sum().reindex(ids).fillna(0.0)

    obs_quarters = (observed_months / 3.0).clip(lower=1.0)
    mean_q_fuel = fuel_litres_life / obs_quarters
    mean_q_nf = nf_rand_life / obs_quarters

    pop_fuel = float(mean_q_fuel[mean_q_fuel > 0].median()) if (mean_q_fuel > 0).any() else 25.0
    pop_nf = float(mean_q_nf[mean_q_nf > 0].median()) if (mean_q_nf > 0).any() else 150.0

    # Bühlmann-Straub credibility shrinkage factor kappa = 3.0
    cred = obs_quarters / (obs_quarters + 3.0)

    perm_fuel = cred * mean_q_fuel + (1.0 - cred) * pop_fuel
    perm_nf = cred * mean_q_nf + (1.0 - cred) * pop_nf

    out["mundlak_perm_fuel_litres"] = perm_fuel.astype("float64")
    out["mundlak_perm_nf_rand"] = perm_nf.astype("float64")

    out["mundlak_shock_fuel_litres"] = (win["m3_fuel_litres"] - perm_fuel).astype("float64")
    out["mundlak_shock_nf_rand"] = (win["m3_nonfuel_rand"] - perm_nf).astype("float64")

    out["mundlak_shock_ratio_fuel"] = ((win["m3_fuel_litres"] + 1.0) / (perm_fuel + 1.0)).astype("float64")
    out["mundlak_shock_ratio_nf"] = ((win["m3_nonfuel_rand"] + 1.0) / (perm_nf + 1.0)).astype("float64")
    return out


def _prior_quarter_official_label_block(
    hist: pd.DataFrame,
    win: pd.DataFrame,
    ids: pd.Index,
    t: pd.Timestamp,
    label_config: Mapping[str, Any],
    data_start: pd.Timestamp,
) -> pd.DataFrame:
    """Official Prior-Quarter Label & Markov State-Transition Features (EDA Section 18.2).

    At cutoff t, the outcome window [t - 3m, t) (m3) and preceding baseline [t - 6m, t - 3m) (q2)
    are strictly observed and leakage-safe whenever t - 3m >= data_start.
    """
    out = pd.DataFrame(index=ids)
    t_prior = t - pd.DateOffset(months=3)
    min_tx = int(label_config.get("minimum_historical_transactions", 3))

    # Continuous two-quarter persistence from win
    m3_f = win["m3_fuel_litres"] if "m3_fuel_litres" in win.columns else pd.Series(0.0, index=ids)
    q2_f = win["q2_fuel_litres"] if "q2_fuel_litres" in win.columns else pd.Series(0.0, index=ids)

    out["q2_fuel_active"] = (q2_f > 0).astype("float64")
    out["two_quarter_fuel_persistence"] = ((m3_f > 0).astype("float64") + (q2_f > 0).astype("float64")).astype("float64")

    # If t_prior is before data_start, no prior quarter exists
    if t_prior <= data_start or len(hist) == 0:
        out["prior_q_was_eligible"] = 0.0
        out["prior_q_is_inactive"] = np.nan
        out["prior_q_is_stable"] = np.nan
        out["prior_q_is_fuel_growth"] = np.nan
        out["prior_q_is_retail_growth"] = np.nan
        out["prior_q_is_adoption"] = np.nan
        out["prior_q_winning_increment_log"] = np.nan
        out["prior_q_n_qualifying_growth_cats"] = np.nan
        out["prior_q_fuel_active"] = np.nan
        out["prior_q_nf_active"] = np.nan
        return out

    # Determine eligibility at t_prior: >= min_tx qualifying transactions strictly before t_prior
    h_before_prior = hist[hist["_time"] < t_prior]
    if len(h_before_prior) > 0:
        q_before = h_before_prior[h_before_prior["is_qualifying"]]
        n_prior_tx = q_before.groupby("ID")["TransactionId"].nunique()
        eligible_prior = set(n_prior_tx[n_prior_tx >= min_tx].index)
    else:
        eligible_prior = set()

    is_elig = pd.Series([1.0 if cid in eligible_prior else 0.0 for cid in ids], index=ids, dtype="float64")
    out["prior_q_was_eligible"] = is_elig

    # Reconstruct realized label at t_prior using official rules
    try:
        from src import label_rules
        raw = label_rules.raw_labels(hist, str(t_prior.date()), label_config).set_index("ID")
    except Exception:
        raw = pd.DataFrame()

    # Reconstruct growth category count across all in-scope categories
    p = hist[~hist["ItemName"].isin(label_config.get("adjustment_items", []))]
    h_sub = p[p["_time"] < t_prior]
    recent = h_sub[h_sub["_time"] >= t_prior - pd.DateOffset(months=3)]
    future = p[(p["_time"] >= t_prior) & (p["_time"] < t)]
    mapping = label_config.get("category_mapping", {})
    cats = sorted(c for c, v in mapping.items() if v.get("included", False))

    if len(future) > 0 and len(cats) > 0:
        def _cat_matrix(frame: pd.DataFrame, count: bool = False) -> pd.DataFrame:
            frame_sub = frame[frame["ItemCategoryLevel2"].isin(cats)]
            if count:
                frame_sub = frame_sub[(frame_sub["_quantity"] > 0) & (frame_sub["_amount"] > 0)]
            g = frame_sub.groupby(["ID", "ItemCategoryLevel2"])
            s = g["TransactionId"].nunique() if count else g["_amount"].sum()
            return s.unstack(fill_value=0).reindex(index=ids, columns=cats, fill_value=0).astype("int64")

        historical_cnt = _cat_matrix(h_sub, True)
        future_cnt = _cat_matrix(future, True)
        prev_spd = _cat_matrix(recent)
        fut_spd = _cat_matrix(future)
        delta_spd = fut_spd - prev_spd
        SCALE = 1000000
        min_rand_scaled = int(label_config.get("growth_min_rand", 20.0) * SCALE)
        rel_growth = label_config.get("growth_min_relative", 0.25)

        grow = historical_cnt.gt(0) & future_cnt.ge(label_config.get("growth_min_purchases", 1))
        grow &= delta_spd.ge(min_rand_scaled)
        if rel_growth is not None:
            grow &= fut_spd.gt(prev_spd.clip(lower=0) * (1 + rel_growth))
        n_growth_cats = grow.sum(axis=1)
    else:
        n_growth_cats = pd.Series(0, index=ids)

    if len(raw) > 0:
        raw_reindexed = raw.reindex(ids)
        opp = raw_reindexed["Opportunity"]
        is_inact = (opp == "Inactivity").astype("float64")
        is_stable = (opp == "Stable").astype("float64")
        is_fg = (opp == "Existing-category growth: Fuel").astype("float64")
        is_rg = (opp.str.startswith("Existing-category growth") & (opp != "Existing-category growth: Fuel")).astype("float64")
        is_adopt = opp.str.startswith("New category adoption").astype("float64")
        inc_log = np.log1p(np.maximum(raw_reindexed["winning_increment_rand"].fillna(0.0), 0.0))
        f_act = (raw_reindexed["fuel_litres"].fillna(0.0) > 0).astype("float64")
        nf_act = (raw_reindexed["nonfuel_rand"].fillna(0.0) > 0).astype("float64")
    else:
        is_inact = pd.Series(np.nan, index=ids)
        is_stable = pd.Series(np.nan, index=ids)
        is_fg = pd.Series(np.nan, index=ids)
        is_rg = pd.Series(np.nan, index=ids)
        is_adopt = pd.Series(np.nan, index=ids)
        inc_log = pd.Series(np.nan, index=ids)
        f_act = pd.Series(np.nan, index=ids)
        nf_act = pd.Series(np.nan, index=ids)

    non_elig_mask = is_elig == 0.0
    is_inact[non_elig_mask] = np.nan
    is_stable[non_elig_mask] = np.nan
    is_fg[non_elig_mask] = np.nan
    is_rg[non_elig_mask] = np.nan
    is_adopt[non_elig_mask] = np.nan
    inc_log[non_elig_mask] = np.nan
    f_act[non_elig_mask] = np.nan
    nf_act[non_elig_mask] = np.nan
    n_growth = n_growth_cats.astype("float64")
    n_growth[non_elig_mask] = np.nan

    out["prior_q_is_inactive"] = is_inact
    out["prior_q_is_stable"] = is_stable
    out["prior_q_is_fuel_growth"] = is_fg
    out["prior_q_is_retail_growth"] = is_rg
    out["prior_q_is_adoption"] = is_adopt
    out["prior_q_winning_increment_log"] = inc_log
    out["prior_q_n_qualifying_growth_cats"] = n_growth
    out["prior_q_fuel_active"] = f_act
    out["prior_q_nf_active"] = nf_act

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
    cats = [c for c in included_categories(label_config) if c not in DEAD_CATEGORIES]

    win = _window_block(hist, ids, t, data_start)
    recency, observed_months = _recency_block(hist, ids, t, data_start)
    blocks = [
        win,
        _derived_block(win),
        recency,
        _trend_block(hist, ids, t, data_start),
        _category_block(hist, ids, t, cats, observed_months),
        _behaviour_block(hist, ids, t, label_config),
        _btyd_block(hist, ids, t),
        _longitudinal_panel_block(win, recency, ids),
        _granular_basket_block(hist, ids, t),
        _refueling_burn_and_payday_block(hist, ids, t),
        _hawkes_decay_block(hist, ids, t),
        _level3_and_svd_block(hist, ids, t),
        _station_seasonal_block(hist, ids, t),
        _mundlak_block(hist, win, ids, observed_months),
        _prior_quarter_official_label_block(hist, win, ids, t, label_config, data_start),
    ]
    base_feats = pd.concat(blocks, axis=1)
    hurdle_block = _hurdle_physics_block(base_feats)
    # 4. Structural categorical: cat_payday_commuter_archetype (4-state interaction of is_commuter x is_payday_shopper)
    commuter_col = base_feats["m6_commuter_share"] if "m6_commuter_share" in base_feats.columns else np.zeros(len(base_feats))
    payday_col = base_feats["payday_spend_share_m3"] if "payday_spend_share_m3" in base_feats.columns else np.zeros(len(base_feats))
    is_commuter = (commuter_col >= 0.40).astype(int)
    is_payday = (payday_col >= 0.35).astype(int)
    archetype_df = pd.DataFrame(
        {"cat_payday_commuter_archetype": (2 * is_commuter + is_payday).astype("int64")},
        index=ids,
    )

    feats = pd.concat([base_feats, hurdle_block, archetype_df], axis=1)

    for c in CAT_STRUCTURAL_FEATURES:
        if c in feats.columns:
            feats[c] = feats[c].fillna(0).astype("int64")
    num_cols = [c for c in feats.columns if c not in CAT_STRUCTURAL_FEATURES]
    feats[num_cols] = feats[num_cols].astype("float64")

    feats.index.name = "ID"
    return feats

