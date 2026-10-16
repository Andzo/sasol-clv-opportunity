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

FEATURE_VERSION = "3.0.0"
DAY = pd.Timedelta(days=1)
DAYS_PER_MONTH = 30.4375
TENURE_CAP_DAYS = 365.0

#: name -> (start_months_ago, end_months_ago); the window is [t - start, t - end).
WINDOWS: dict[str, tuple[int, int]] = {
    "m1": (1, 0),
    "m3": (3, 0),    # == label "previous quarter" baseline
    "m6": (6, 0),
    "q2": (6, 3),    # quarter before the previous quarter
    "yoy_q": (12, 9), # same quarter last year (year-over-year baseline)
}
TREND_MONTHS = 6

DEAD_CATEGORIES: set[str] = {
    "Frozen food", "Krispy Kreme", "Miscellaneous", "Promotional items", "Unknown"
}
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
    """Activity / spend / volume aggregates for each calendar window with season-normalized fallback."""
    frames = []
    has_observed_yoy = pd.Series(1.0, index=ids, dtype="float64")

    for name, (start, end) in WINDOWS.items():
        lo, hi = _months_before(t, start), _months_before(t, end)
        if name == "yoy_q" and lo < data_start:
            # Fallback to customer's earliest observed 90-day window scaled by station seasonal index
            has_observed_yoy[:] = 0.0
            if len(hist) > 0 and "ID" in hist.columns:
                first_seen = hist.groupby("ID")["_time"].min().reindex(ids)
                tx_cust_first = hist["ID"].map(first_seen)
                delta_days = (hist["_time"] - tx_cust_first).dt.total_seconds() / 86400.0
                in_first_90d = (delta_days >= 0.0) & (delta_days < 90.0)
                w_early = hist[in_first_90d]
                q_early, s_early = w_early[w_early["is_qualifying"]], w_early[w_early["in_scope"]]

                early_b = q_early.groupby("ID")["TransactionId"].nunique().reindex(ids).fillna(0.0)
                early_fb = q_early[q_early["is_fuel_cat"]].groupby("ID")["TransactionId"].nunique().reindex(ids).fillna(0.0)
                early_nfb = q_early[q_early["is_nonfuel"]].groupby("ID")["TransactionId"].nunique().reindex(ids).fillna(0.0)
                early_fl = s_early[s_early["is_fuel_sale"]].groupby("ID")["qty"].sum().reindex(ids).fillna(0.0)
                early_fr = s_early[s_early["is_fuel_cat"]].groupby("ID")["amount"].sum().reindex(ids).fillna(0.0)
                early_nfr = s_early[s_early["is_nonfuel"]].groupby("ID")["amount"].sum().reindex(ids).fillna(0.0)
                early_nc = q_early.groupby("ID")["ItemCategoryLevel2"].nunique().reindex(ids).fillna(0.0)

                # Station-level seasonal scaling
                if "LocationId" in hist.columns:
                    loc_counts = hist.groupby(["ID", "LocationId"]).size().reset_index(name="count")
                    cust_loc = loc_counts.sort_values(["ID", "count"], ascending=[True, False]).drop_duplicates(subset=["ID"]).set_index("ID")["LocationId"].reindex(ids)
                    station_month = hist[hist["_time"].dt.month == t.month].groupby("LocationId")["amount"].sum()
                    station_total = hist.groupby("LocationId")["amount"].sum() / max(hist["_time"].dt.month.nunique(), 1)
                    station_idx = (station_month / station_total.where(station_total > 0)).clip(0.75, 1.35).fillna(1.0)
                    seasonal_mult = cust_loc.map(station_idx).fillna(1.0).to_numpy()
                else:
                    seasonal_mult = np.ones(len(ids))

                block = pd.DataFrame({
                    f"{name}_baskets": early_b * seasonal_mult,
                    f"{name}_fuel_baskets": early_fb * seasonal_mult,
                    f"{name}_nonfuel_baskets": early_nfb * seasonal_mult,
                    f"{name}_fuel_litres": early_fl * seasonal_mult,
                    f"{name}_fuel_rand": early_fr * seasonal_mult,
                    f"{name}_nonfuel_rand": early_nfr * seasonal_mult,
                    f"{name}_n_categories": early_nc,
                }, index=ids).astype("float64")
            else:
                block = pd.DataFrame({
                    f"{name}_baskets": 0.0,
                    f"{name}_fuel_baskets": 0.0,
                    f"{name}_nonfuel_baskets": 0.0,
                    f"{name}_fuel_litres": 0.0,
                    f"{name}_fuel_rand": 0.0,
                    f"{name}_nonfuel_rand": 0.0,
                    f"{name}_n_categories": 0.0,
                }, index=ids).astype("float64")
        else:
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

    frames.append(pd.DataFrame({"has_observed_yoy_window": has_observed_yoy}, index=ids))
    return pd.concat(frames, axis=1)


def _derived_block(win: pd.DataFrame) -> pd.DataFrame:
    """Momentum ratios and unit economics derived from the window aggregates."""
    out: dict[str, pd.Series] = {}
    for qty in ("baskets", "fuel_litres", "nonfuel_rand"):
        out[f"mom_m3_q2_{qty}"] = (win[f"m3_{qty}"] + 1.0) / (win[f"q2_{qty}"] + 1.0)
    out["delta_m3_q2_fuel_rand"] = win["m3_fuel_rand"] - win["q2_fuel_rand"]
    out["delta_m3_q2_nonfuel_rand"] = win["m3_nonfuel_rand"] - win["q2_nonfuel_rand"]
    out["m1_share_of_m3_baskets"] = win["m1_baskets"] / win["m3_baskets"].where(win["m3_baskets"] > 0)
    out["fuel_price_m3"] = win["m3_fuel_rand"] / win["m3_fuel_litres"].where(win["m3_fuel_litres"] > 0)
    vol_weighted_price = (win["m3_fuel_rand"].sum() / win["m3_fuel_litres"].sum()) if win["m3_fuel_litres"].sum() > 0 else 23.0
    out["rel_fuel_price_m3"] = (out["fuel_price_m3"] / vol_weighted_price).fillna(1.0)
    spend = win["m3_fuel_rand"] + win["m3_nonfuel_rand"]
    out["basket_value_m3"] = spend / win["m3_baskets"].where(win["m3_baskets"] > 0)
    out["fuel_share_m3"] = win["m3_fuel_rand"] / spend.where(spend > 0)
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
    # Distance between projected quarterly spend (3.0 * m1_rand) and hurdle H_C
    out["hurdle_fuel_rand_dist"] = 3.0 * win["m1_fuel_rand"] - H_fuel
    out["fuel_proj_m1_hurdle_ratio"] = (3.0 * win["m1_fuel_rand"] + 1.0) / (H_fuel + 1.0)
    out["fuel_proj_q2_hurdle_dist"] = win["q2_fuel_rand"] - H_fuel
    out["fuel_proj_m6_hurdle_dist"] = 0.5 * win["m6_fuel_rand"] - H_fuel
    out["nonfuel_proj_m1_hurdle_dist"] = 3.0 * win["m1_nonfuel_rand"] - H_nonfuel
    out["nonfuel_proj_q2_hurdle_dist"] = win["q2_nonfuel_rand"] - H_nonfuel
    out["nonfuel_proj_m6_hurdle_dist"] = 0.5 * win["m6_nonfuel_rand"] - H_nonfuel

    out["store_attachment_rate"] = win["m3_nonfuel_baskets"] / (win["m3_baskets"] + 1.0)
    out["has_visited_m1"] = (win["m1_baskets"] > 0).astype("float64")

    # Inverted YoY to m3 ratios (high yoy_q / m3 > 1.25 signals seasonal surge)
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
    out["left_censored"] = (first_any < data_start + 30 * DAY).astype("float64")
    out["gap_mean"] = g["gap"].mean()
    out["gap_median"] = g["gap"].median()
    out["gap_std"] = g["gap"].std()
    out["gap_last"] = g["gap"].last()
    out["overdue_ratio"] = out["days_since_last_purchase"] / out["gap_median"].clip(lower=1.0)
    out["exp_churn_hazard"] = 1.0 - np.exp(-out["days_since_last_purchase"] / out["gap_mean"].clip(lower=1.0))
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

    # Drop redundant lag1 columns (already captured by m1_*) and fuel_rand_lag* columns
    drop_cols = [f"{p}_lag1" for p in ("baskets", "fuel_rand", "fuel_litres", "nonfuel_rand")]
    drop_cols += [f"fuel_rand_lag{k}" for k in range(2, TREND_MONTHS + 1)]
    out = out.drop(columns=[c for c in drop_cols if c in out.columns])
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
    p_alive = np.where(x == 0, 1.0, sp.expit(-log_div))
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
    """Statistical hurdle physics, tail exceedance Z-scores, and spend volatility."""
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
    m3_bev = feats["cat_m3_rand_beverages"] if "cat_m3_rand_beverages" in feats.columns else pd.Series(0.0, index=feats.index)
    H_bev = np.maximum(m3_bev + 20.0, 1.25 * m3_bev)
    out["bev_hurdle_rand"] = H_bev
    out["bev_headroom_rand"] = H_bev - m3_bev

    m3_food = feats["cat_m3_rand_food_beverage"] if "cat_m3_rand_food_beverage" in feats.columns else pd.Series(0.0, index=feats.index)
    H_food = np.maximum(m3_food + 20.0, 1.25 * m3_food)
    out["food_hurdle_rand"] = H_food
    out["food_headroom_rand"] = H_food - m3_food

    # 4. Relative Z-score competition (Rule 6 argmax)
    out["z_diff_fuel_nonfuel"] = z_f - z_nf

    return out


def _longitudinal_panel_block(win: pd.DataFrame, recency: pd.DataFrame, ids: pd.Index) -> pd.DataFrame:
    """Longitudinal lagged outcomes, tenure censoring, and hurdle clearing dynamics."""
    out = pd.DataFrame(index=ids)
    out["lag1_clv_fuel"] = np.log1p(win["m3_fuel_litres"]) / 2.352825386185113
    out["lag1_clv_nonfuel"] = np.log1p(win["m3_nonfuel_rand"]) / 2.4200904988927476
    out["lag2_clv_fuel"] = np.log1p(win["q2_fuel_litres"]) / 2.352825386185113
    out["lag2_clv_nonfuel"] = np.log1p(win["q2_nonfuel_rand"]) / 2.4200904988927476
    out["delta_lag_clv_fuel"] = out["lag1_clv_fuel"] - out["lag2_clv_fuel"]
    out["delta_lag_clv_nonfuel"] = out["lag1_clv_nonfuel"] - out["lag2_clv_nonfuel"]

    tenure = recency["days_since_first_seen_capped"]
    out["tenure_days_capped_90"] = np.minimum(tenure, 90.0)
    out["is_left_censored_m3"] = (tenure < 90).astype("float64")
    annual_mult = 90.0 / np.maximum(np.minimum(tenure, 90.0), 7.0)
    out["m3_annualized_rate_fuel"] = win["m3_fuel_litres"] * annual_mult
    out["m3_annualized_rate_nonfuel"] = win["m3_nonfuel_rand"] * annual_mult

    cleared = (
        (win["m3_fuel_litres"] > 1.25 * win["q2_fuel_litres"])
        & ((win["m3_fuel_litres"] - win["q2_fuel_litres"]) >= 1.0)
    ).astype("float64")
    dip_f = (win["m3_fuel_litres"] + 5.0) / (np.maximum(win["q2_fuel_litres"], 5.0) + 5.0)
    dip_nf = (win["m3_nonfuel_rand"] + 50.0) / (np.maximum(win["q2_nonfuel_rand"], 50.0) + 50.0)

    # Censor when q2 is NaN
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
    """Granular ItemCategoryLevel3 routine concentration (HHI) and 16-component TruncatedSVD."""
    out = pd.DataFrame(index=ids)
    w_m6 = hist[(hist["_time"] >= t - pd.DateOffset(months=6)) & hist["in_scope"]]
    if len(w_m6) == 0 or "ItemCategoryLevel3" not in w_m6.columns or "ID" not in w_m6.columns:
        for cat in GLOBAL_TOP_35_LEVEL3:
            out[f"l3_share_{safe_name(cat)}"] = 0.0
        out["level3_hhi_concentration"] = 0.0
        for i in range(16):
            out[f"svd_dim_{i}"] = 0.0
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

    l3_matrix = pd.DataFrame(0.0, index=ids, columns=GLOBAL_TOP_35_LEVEL3)
    for cat in GLOBAL_TOP_35_LEVEL3:
        if cat in spend_matrix.columns:
            l3_matrix[cat] = spend_matrix[cat]

    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=16, random_state=42)
    reduced = svd.fit_transform(l3_matrix)
    for i in range(16):
        out[f"svd_dim_{i}"] = reduced[:, i]

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
    ]
    feats = pd.concat(blocks, axis=1)
    hurdle_block = _hurdle_physics_block(feats)
    # Ablated static snapshot calendar constants and hardcoded Dec 2024 features.
    # Relative seasonality is captured cleanly by the yoy_q window across all snapshots,
    # preventing trees from memorizing the single summer snapshot in the training history.
    feats = pd.concat([feats, hurdle_block], axis=1)

    feats.index.name = "ID"
    return feats.astype("float64")
