"""Section 2: Customer cohorts & lifetime dynamics (C01 - C08)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.dataset import Snapshot
from src.viz.style import PALETTE, apply_style


def plot_c01_cohort_growth(snapshots: list[Snapshot]) -> tuple[plt.Figure, dict[str, float]]:
    """C01: Evolution of eligible customer count per snapshot stacked by tenure."""
    apply_style()
    records = []
    for s in snapshots:
        tenure = s.features["days_since_first_seen_capped"]
        t_bucket = pd.cut(tenure, bins=[0, 60, 180, 365, 9999], labels=["<2m", "2-6m", "6-12m", "12m+"])
        counts = t_bucket.value_counts()
        for b, cnt in counts.items():
            records.append({"cutoff": str(s.cutoff.date()), "tenure": b, "count": cnt})

    df = pd.DataFrame(records)
    pivot = df.pivot(index="cutoff", columns="tenure", values="count").fillna(0)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    pivot.plot(kind="bar", stacked=True, ax=ax, colormap="viridis", edgecolor="none")
    ax.set_ylabel("Eligible Customer Count")
    ax.set_title("C01: Customer Cohort Growth & Tenure Composition by Snapshot")
    plt.xticks(rotation=35, ha="right")
    ax.legend(title="Tenure")
    plt.tight_layout()

    findings = {
        "first_snapshot_count": float(pivot.iloc[0].sum()),
        "last_snapshot_count": float(pivot.iloc[-1].sum()),
        "cohort_growth_factor": float(pivot.iloc[-1].sum() / pivot.iloc[0].sum()),
    }
    return fig, findings


def plot_c02_customer_recurrence(snapshots: list[Snapshot]) -> tuple[plt.Figure, dict[str, float]]:
    """C02: Distribution of snapshot appearances per customer across history."""
    apply_style()
    from collections import Counter
    all_ids = []
    for s in snapshots:
        all_ids.extend(s.ids.tolist())

    counts = Counter(all_ids)
    num_snaps = pd.Series(list(counts.values()))

    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.hist(num_snaps, bins=np.arange(0.5, len(snapshots) + 1.5, 1), color=PALETTE["fuel"], edgecolor="white", rwidth=0.85)
    ax.set_xlabel("Number of Historical Snapshots Appeared In")
    ax.set_ylabel("Customer Count")
    ax.set_title("C02: Multi-Cohort Customer Recurrence Distribution")
    ax.set_xticks(range(1, len(snapshots) + 1))
    plt.tight_layout()

    findings = {
        "pct_customers_in_all_snaps": float(100 * (num_snaps == len(snapshots)).mean()),
        "pct_customers_in_single_snap": float(100 * (num_snaps == 1).mean()),
        "median_snapshot_appearances": float(num_snaps.median()),
    }
    return fig, findings


def plot_c04_eligibility_floor_proximity(snapshots: list[Snapshot]) -> tuple[plt.Figure, dict[str, float]]:
    """C04: Historical basket count distribution emphasizing the 3-basket eligibility floor."""
    apply_style()
    eval_snap = snapshots[-1]
    life_baskets = eval_snap.features["baskets_per_month_life"] * (eval_snap.features["days_since_first_seen_capped"] / 30.4375).clip(lower=1.0)
    life_baskets = life_baskets.clip(upper=40)

    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.hist(life_baskets, bins=35, color=PALETTE["accent"], edgecolor="white", alpha=0.9)
    ax.axvline(3.0, color="#D55E00", linestyle="--", linewidth=2, label="Eligibility Floor (3 baskets)")
    ax.set_xlabel("Estimated Lifetime Qualifying Baskets")
    ax.set_ylabel("Customer Count")
    ax.set_title(f"C04: Proximity to Eligibility Floor ({eval_snap.cutoff.date()})")
    ax.legend()
    plt.tight_layout()

    findings = {
        "pct_near_floor_3_to_5_baskets": float(100 * ((life_baskets >= 3) & (life_baskets <= 5)).mean()),
    }
    return fig, findings


def plot_c05_gap_survival_curve(eval_snap: Snapshot) -> tuple[plt.Figure, dict[str, float]]:
    """C05: Empirical non-parametric Kaplan-Meier survival curve of purchase gaps."""
    apply_style()
    gaps = eval_snap.features["gap_median"].dropna().clip(1, 120).to_numpy()
    gaps_sorted = np.sort(gaps)
    surv = 1.0 - np.arange(len(gaps_sorted)) / len(gaps_sorted)

    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.step(gaps_sorted, surv, where="post", color=PALETTE["fuel"], linewidth=2)
    ax.axvline(30, color="gray", linestyle=":", label="30 Days (1 Month)")
    ax.axvline(60, color="gray", linestyle="--", label="60 Days (2 Months)")
    ax.axvline(90, color="red", linestyle="-", label="90 Days (Quarter Boundary)")
    ax.set_xlabel("Days Between Consecutive Baskets")
    ax.set_ylabel("Survival Probability P(Gap > t)")
    ax.set_title("C05: Non-Parametric Survival Distribution of Inter-Purchase Intervals")
    ax.legend()
    plt.tight_layout()

    findings = {
        "p_gap_greater_30d": float(np.interp(30, gaps_sorted, surv)),
        "p_gap_greater_60d": float(np.interp(60, gaps_sorted, surv)),
        "p_gap_greater_90d": float(np.interp(90, gaps_sorted, surv)),
    }
    return fig, findings
