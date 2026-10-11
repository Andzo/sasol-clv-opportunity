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

FEATURE_VERSION = "2.4.0"
DAY = pd.Timedelta(days=1)
DAYS_PER_MONTH = 30.4375
TENURE_CAP_DAYS = 365.0

#: name -> (start_months_ago, end_months_ago); the window is [t - start, t - end).
WINDOWS: dict[str, tuple[int, int]] = {
    "m1": (1, 0),
    "m3": (3, 0),    # == label "previous quarter" baseline
    "m6": (6, 0),
    "q2": (6, 3),    # quarter before the previous quarter
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
    fuel_growth_hurdle = np.maximum(20.0, 0.25 * win["m3_fuel_rand"])
    out["fuel_hurdle_ratio"] = fuel_growth_hurdle / (win["m3_fuel_rand"] + 1.0)
    out["store_attachment_rate"] = win["m3_nonfuel_baskets"] / (win["m3_baskets"] + 1.0)
    out["has_visited_m1"] = (win["m1_baskets"] > 0).astype("float64")
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
    excluded = core[~core["is_included"]]
    out["m6_excluded_baskets"] = excluded.groupby("ID")["TransactionId"].nunique()
    out["m6_excluded_rand"] = excluded.groupby("ID")["amount"].sum()
    for item in ACTIVE_ADJUSTMENT_ITEMS:
        tag = safe_name(item)
        out[f"ever_adj_{tag}"] = hist[hist["ItemName"] == item].groupby("ID").size().gt(0).astype("float64")
        out[f"m6_adj_{tag}_lines"] = w[w["ItemName"] == item].groupby("ID").size()
    count_cols = [c for c in out.columns if not c.startswith(("m6_mean_hour", "m6_std_hour", "m6_top", "m6_weekend", "m6_cross"))]
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
    ]
    feats = pd.concat(blocks, axis=1)
    feats["cutoff_month_sin"] = np.sin(2 * np.pi * t.month / 12.0)
    feats["cutoff_month_cos"] = np.cos(2 * np.pi * t.month / 12.0)
    feats["is_holiday_quarter"] = float(t.month in (11, 12, 1))
    feats["cutoff_quarter"] = float(t.quarter)

    dec24_lo, dec24_hi = pd.Timestamp("2024-12-01"), pd.Timestamp("2025-01-01")
    if t >= dec24_hi:
        dec24_tx = hist[(hist["_time"] >= dec24_lo) & (hist["_time"] < dec24_hi) & hist["is_qualifying"]]
        dec24_spend = dec24_tx.groupby("ID")["amount"].sum().reindex(ids).fillna(0.0)
        feats["hist_december_active"] = (dec24_spend > 0).astype("float64")
        feats["hist_december_spend_log"] = np.log1p(dec24_spend.astype("float64"))
    else:
        feats["hist_december_active"] = 0.0
        feats["hist_december_spend_log"] = 0.0

    feats.index.name = "ID"
    return feats.astype("float64")
