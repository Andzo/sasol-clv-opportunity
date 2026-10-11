"""Section 1: Raw data audit (D01 - D10)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.viz.style import PALETTE, apply_style


def plot_d01_daily_activity(tx: pd.DataFrame) -> tuple[go.Figure, dict[str, float]]:
    """D01: Daily transaction volume and active customer counts over time."""
    daily = tx.groupby(tx["_time"].dt.date).agg(
        transactions=("TransactionId", "nunique"),
        customers=("ID", "nunique"),
        total_amount=("amount", "sum"),
    ).reset_index()
    daily["_time"] = pd.to_datetime(daily["_time"])
    daily["tx_7d"] = daily["transactions"].rolling(7, min_periods=1).mean()
    daily["cust_7d"] = daily["customers"].rolling(7, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily["_time"], y=daily["transactions"], mode="lines", name="Daily Transactions", line=dict(color="#B0C4DE", width=1)))
    fig.add_trace(go.Scatter(x=daily["_time"], y=daily["tx_7d"], mode="lines", name="7-Day Avg Transactions", line=dict(color=PALETTE["fuel"], width=2.5)))
    fig.add_trace(go.Scatter(x=daily["_time"], y=daily["cust_7d"], mode="lines", name="7-Day Avg Active Customers", line=dict(color=PALETTE["accent"], width=2.5), yaxis="y2"))

    fig.update_layout(
        title="D01: Daily Transaction Volume and Customer Footfall (Interactive)",
        xaxis_title="Date",
        yaxis=dict(title="Transactions / Day"),
        yaxis2=dict(title="Active Customers / Day", overlaying="y", side="right"),
        hovermode="x unified",
        template="plotly_white",
        legend=dict(x=0.01, y=0.99),
    )
    findings = {
        "start_date": str(daily["_time"].min().date()),
        "end_date": str(daily["_time"].max().date()),
        "mean_daily_transactions": float(daily["transactions"].mean()),
        "max_daily_transactions": float(daily["transactions"].max()),
        "mean_daily_customers": float(daily["customers"].mean()),
    }
    return fig, findings


def plot_d02_category_shares(tx: pd.DataFrame) -> tuple[plt.Figure, dict[str, float]]:
    """D02: Category share of line items vs share of gross rand spend."""
    apply_style()
    cat_summary = tx.groupby("ItemCategoryLevel2").agg(
        lines=("TransactionId", "count"),
        spend=("amount", "sum"),
    ).sort_values("lines", ascending=False).head(12)

    cat_summary["line_share"] = cat_summary["lines"] / tx["TransactionId"].count()
    cat_summary["spend_share"] = cat_summary["spend"] / tx["amount"].sum()

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(cat_summary))
    w = 0.35
    ax.bar(x - w/2, cat_summary["line_share"] * 100, width=w, label="% Line Items", color=PALETTE["fuel"])
    ax.bar(x + w/2, cat_summary["spend_share"] * 100, width=w, label="% Rand Spend", color=PALETTE["nonfuel"])
    ax.set_xticks(x)
    ax.set_xticklabels(cat_summary.index, rotation=35, ha="right")
    ax.set_ylabel("Share (%)")
    ax.set_title("D02: Line Item Share vs Gross Rand Share by Category (Top 12)")
    ax.legend()
    plt.tight_layout()

    findings = {
        "fuel_line_share_pct": float(cat_summary.loc["Fuel", "line_share"] * 100) if "Fuel" in cat_summary.index else 0.0,
        "fuel_spend_share_pct": float(cat_summary.loc["Fuel", "spend_share"] * 100) if "Fuel" in cat_summary.index else 0.0,
    }
    return fig, findings


def plot_d04_fuel_fill_modes(tx: pd.DataFrame) -> tuple[plt.Figure, dict[str, float]]:
    """D04: Distribution of fuel litres per fill and mode identification."""
    apply_style()
    fuel_sales = tx[tx["is_fuel_sale"] & (tx["qty"] > 0)]["qty"].clip(upper=150)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    n, bins, patches = ax.hist(fuel_sales, bins=60, color=PALETTE["fuel"], edgecolor="white", alpha=0.85)
    ax.set_xlabel("Fuel Litres per Transaction")
    ax.set_ylabel("Frequency (Transactions)")
    ax.set_title("D04: Fuel Volume Distribution per Transaction (Litres)")
    ax.axvline(fuel_sales.median(), color=PALETTE["nonfuel"], linestyle="--", label=f"Median ({fuel_sales.median():.1f} L)")
    ax.legend()
    plt.tight_layout()

    findings = {
        "fuel_fill_median_litres": float(fuel_sales.median()),
        "fuel_fill_p90_litres": float(fuel_sales.quantile(0.90)),
        "fuel_fill_mode_approx": float(bins[np.argmax(n)]),
    }
    return fig, findings


def plot_d05_fuel_price_trend(tx: pd.DataFrame) -> tuple[plt.Figure, dict[str, float]]:
    """D05: Implied historical fuel price (Rand/Litre) month by month."""
    apply_style()
    fuel_tx = tx[tx["is_fuel_sale"] & (tx["qty"] > 0) & (tx["amount"] > 0)].copy()
    fuel_tx["month"] = fuel_tx["_time"].dt.to_period("M").dt.to_timestamp()
    monthly_price = fuel_tx.groupby("month").apply(
        lambda g: g["amount"].sum() / g["qty"].sum()
    ).reset_index(name="implied_price")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(monthly_price["month"], monthly_price["implied_price"], marker="o", color=PALETTE["fuel"], linewidth=2)
    ax.set_ylabel("Implied Price (ZAR / Litre)")
    ax.set_title("D05: Monthly Historical Implied Fuel Price (Pre-Tax Rand/Litre)")
    plt.xticks(rotation=30)
    plt.tight_layout()

    findings = {
        "min_monthly_price": float(monthly_price["implied_price"].min()),
        "max_monthly_price": float(monthly_price["implied_price"].max()),
        "price_variation_pct": float(100 * (monthly_price["implied_price"].max() - monthly_price["implied_price"].min()) / monthly_price["implied_price"].min()),
    }
    return fig, findings


def plot_d08_hour_weekday_heatmap(tx: pd.DataFrame) -> tuple[go.Figure, dict[str, float]]:
    """D08: Interactive shopping rhythm heatmap across weekday and hour."""
    tx_valid = tx[tx["is_qualifying"]].copy()
    tx_valid["hour"] = tx_valid["_time"].dt.hour
    tx_valid["day"] = tx_valid["_time"].dt.day_name()
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    cross = pd.crosstab(tx_valid["day"], tx_valid["hour"]).reindex(days)
    fig = px.imshow(
        cross,
        labels=dict(x="Hour of Day", y="Day of Week", color="Baskets"),
        x=cross.columns,
        y=cross.index,
        color_continuous_scale="Blues",
        title="D08: Qualifying Customer Visits by Day and Hour (Interactive)",
    )
    fig.update_layout(template="plotly_white")
    findings = {
        "peak_hour": int(tx_valid["hour"].value_counts().idxmax()),
        "weekend_share_pct": float(100 * tx_valid["_time"].dt.dayofweek.isin([5, 6]).mean()),
    }
    return fig, findings
