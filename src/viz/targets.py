"""Section 3: Targets & label mechanics (T01 - T14)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.dataset import Snapshot
from src.viz.style import OPPORTUNITY_COLORS, PALETTE, apply_style


def plot_t01_target_distributions(eval_snap: Snapshot) -> tuple[plt.Figure, dict[str, float]]:
    """T01: Raw quantities vs normalized targets with zero inflation annotation."""
    apply_style()
    labels = eval_snap.labels
    if labels is None:
        raise ValueError("Snapshot must have labels")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # Raw fuel litres
    axes[0, 0].hist(labels["fuel_litres"], bins=40, color=PALETTE["fuel"], edgecolor="white", alpha=0.85)
    axes[0, 0].set_title("Raw Fuel Volume (Litres in Q)")
    axes[0, 0].set_ylabel("Count")

    # Transformed CLV_fuel
    axes[0, 1].hist(labels["CLV_fuel"], bins=40, color=PALETTE["fuel"], edgecolor="white", alpha=0.85)
    axes[0, 1].set_title("Transformed Target CLV_fuel = ln(1+litres)/2.35")
    zero_fuel_pct = 100 * (labels["CLV_fuel"] == 0).mean()
    axes[0, 1].annotate(f"Zeros: {zero_fuel_pct:.1f}%", xy=(0, len(labels)*0.25), xytext=(0.5, len(labels)*0.3),
                        arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=4))

    # Raw nonfuel rand
    axes[1, 0].hist(labels["nonfuel_rand"].clip(upper=2500), bins=40, color=PALETTE["nonfuel"], edgecolor="white", alpha=0.85)
    axes[1, 0].set_title("Raw Non-fuel Spend (ZAR in Q, clipped at R2500)")
    axes[1, 0].set_ylabel("Count")

    # Transformed CLV_nonfuel
    axes[1, 1].hist(labels["CLV_nonfuel"], bins=40, color=PALETTE["nonfuel"], edgecolor="white", alpha=0.85)
    axes[1, 1].set_title("Transformed Target CLV_nonfuel = ln(1+rand)/2.42")
    zero_nf_pct = 100 * (labels["CLV_nonfuel"] == 0).mean()
    axes[1, 1].annotate(f"Zeros: {zero_nf_pct:.1f}%", xy=(0, len(labels)*0.55), xytext=(0.5, len(labels)*0.6),
                        arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=4))

    plt.tight_layout()
    findings = {
        "pct_zero_fuel": float(zero_fuel_pct),
        "pct_zero_nonfuel": float(zero_nf_pct),
        "fuel_p95_litres": float(labels["fuel_litres"].quantile(0.95)),
        "nonfuel_p95_rand": float(labels["nonfuel_rand"].quantile(0.95)),
    }
    return fig, findings


def plot_t03_class_frequencies(snapshots: list[Snapshot]) -> tuple[plt.Figure, dict[str, float]]:
    """T03: Opportunity class frequency breakdown across all historical snapshots."""
    apply_style()
    records = []
    for s in snapshots:
        if s.labels is not None:
            vc = s.labels["Opportunity"].value_counts(normalize=True)
            for cls_name, share in vc.items():
                records.append({"cutoff": str(s.cutoff.date()), "class": cls_name, "share": share * 100})

    df = pd.DataFrame(records)
    pivot = df.pivot(index="cutoff", columns="class", values="share").fillna(0)

    # Reorder columns by average share
    ordered_cols = pivot.mean().sort_values(ascending=False).index
    pivot = pivot[ordered_cols]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = [OPPORTUNITY_COLORS.get(col, "#AAAAAA") for col in ordered_cols]
    pivot.plot(kind="bar", stacked=True, ax=ax, color=colors, edgecolor="none")
    ax.set_ylabel("Share of Customers (%)")
    ax.set_title("T03: Opportunity Class Proportions Across Historical Snapshots")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7.5)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()

    top_3_share = float(pivot[ordered_cols[:3]].iloc[-1].sum())
    findings = {
        "top_3_classes_share_pct": top_3_share,
        "inactivity_share_pct": float(pivot.get("Inactivity", pd.Series(0)).iloc[-1]),
        "fuel_growth_share_pct": float(pivot.get("Existing-category growth: Fuel", pd.Series(0)).iloc[-1]),
        "stable_share_pct": float(pivot.get("Stable", pd.Series(0)).iloc[-1]),
    }
    return fig, findings


def plot_t04_label_transitions(snapshots: list[Snapshot]) -> tuple[plt.Figure, dict[str, float]]:
    """T04: Quarter-over-quarter label transition matrix for recurring customers."""
    apply_style()
    # Pair consecutive snapshots separated by 3 months
    pairs = []
    for i in range(len(snapshots) - 1):
        s1, s2 = snapshots[i], snapshots[i+1]
        if (s2.cutoff - s1.cutoff).days in range(85, 95) and s1.labels is not None and s2.labels is not None:
            common = s1.ids.intersection(s2.ids)
            if len(common) > 100:
                p_df = pd.DataFrame({
                    "prev_label": s1.labels.loc[common, "Opportunity"],
                    "curr_label": s2.labels.loc[common, "Opportunity"],
                })
                pairs.append(p_df)

    if not pairs:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No 3-month snapshot pairs available", ha="center")
        return fig, {}

    all_pairs = pd.concat(pairs, axis=0)
    top_labels = ["Inactivity", "Existing-category growth: Fuel", "Stable", "Existing-category growth: Other", "Existing-category growth: Beverages"]
    filtered = all_pairs[all_pairs["prev_label"].isin(top_labels) & all_pairs["curr_label"].isin(top_labels)]

    ct = pd.crosstab(filtered["prev_label"].to_numpy(), filtered["curr_label"].to_numpy(), normalize="index").reindex(index=top_labels, columns=top_labels)

    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(ct.to_numpy(), cmap="Blues")
    ax.set_xticks(range(len(top_labels)))
    ax.set_yticks(range(len(top_labels)))
    ax.set_xticklabels([l.replace("Existing-category growth: ", "Growth: ") for l in top_labels], rotation=35, ha="right")
    ax.set_yticklabels([l.replace("Existing-category growth: ", "Growth: ") for l in top_labels])
    ax.set_xlabel("Quarter t Outcome Label")
    ax.set_ylabel("Quarter t-1 Outcome Label")
    ax.set_title("T04: Quarter-over-Quarter Opportunity Transition Probability")

    for i in range(len(top_labels)):
        for j in range(len(top_labels)):
            val = ct.iloc[i, j]
            ax.text(j, i, f"{val:.1%}", ha="center", va="center", color="white" if val > 0.4 else "black", fontsize=8.5)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    findings = {
        "inactivity_persistence": float(ct.loc["Inactivity", "Inactivity"]),
        "growth_to_stable_reversion": float(ct.loc["Existing-category growth: Fuel", "Stable"]),
        "stable_to_growth_rate": float(ct.loc["Stable", "Existing-category growth: Fuel"]),
    }
    return fig, findings


def plot_t05_growth_rule_geometry(eval_snap: Snapshot) -> tuple[go.Figure, dict[str, float]]:
    """T05: Interactive visualization of Rule 5 growth hurdle boundary."""
    labels = eval_snap.labels
    features = eval_snap.features
    if labels is None:
        raise ValueError("Snapshot must have labels")

    prev_spend = features["m3_fuel_rand"].reindex(eval_snap.ids).fillna(0.0)
    # Estimate outcome fuel spend from fuel litres * fuel price
    price = features["fuel_price_m3"].fillna(23.0).clip(15, 35)
    future_spend = labels["fuel_litres"] * price

    df = pd.DataFrame({
        "ID": eval_snap.ids,
        "prev_fuel_rand": prev_spend,
        "future_fuel_rand": future_spend,
        "Opportunity": labels["Opportunity"],
    })
    # Sample 1200 points for crisp interactive rendering
    sample_df = df.sample(min(1200, len(df)), random_state=42).copy()

    fig = px.scatter(
        sample_df,
        x="prev_fuel_rand",
        y="future_fuel_rand",
        color="Opportunity",
        color_discrete_map=OPPORTUNITY_COLORS,
        hover_data=["ID", "prev_fuel_rand", "future_fuel_rand"],
        title="T05: Growth Hurdle Geometry (Prev Quarter vs Future Fuel Spend, Interactive)",
        labels={"prev_fuel_rand": "Previous Quarter Fuel Spend (ZAR)", "future_fuel_rand": "Future Quarter Fuel Spend (ZAR)"},
        opacity=0.75,
    )
    # Draw growth boundaries: y >= x + 20 and y >= 1.25 * x
    max_val = min(sample_df["prev_fuel_rand"].quantile(0.98), 8000)
    x_line = np.linspace(0, max_val, 100)
    fig.add_trace(go.Scatter(x=x_line, y=1.25 * x_line, mode="lines", name="+25% Hurdle Boundary", line=dict(color="black", dash="dash", width=2)))
    fig.add_trace(go.Scatter(x=x_line, y=x_line + 20, mode="lines", name="+R20 Hurdle Boundary", line=dict(color="red", dash="dot", width=1.5)))

    fig.update_layout(template="plotly_white", legend=dict(x=0.01, y=0.99))
    findings = {
        "median_prev_spend": float(prev_spend.median()),
        "pct_zero_prev_spend": float(100 * (prev_spend == 0).mean()),
    }
    return fig, findings


def plot_t08_winner_decomposition(eval_snap: Snapshot) -> tuple[plt.Figure, dict[str, float]]:
    """T08: Distribution of winning source categories that trigger growth or adoption."""
    apply_style()
    labels = eval_snap.labels
    if labels is None or "winning_source_category" not in labels.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "Winning category data unavailable", ha="center")
        return fig, {}

    winners = labels[labels["winning_source_category"] != ""]["winning_source_category"].value_counts().head(12)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(winners.index, winners.values, color=PALETTE["fuel"], edgecolor="white")
    ax.set_ylabel("Winning Opportunity Count")
    ax.set_title("T08: Winning Source Categories Generating Growth & Adoption")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()

    fuel_wins = float(winners.get("Fuel", 0))
    findings = {
        "fuel_winning_share_pct": float(100 * fuel_wins / winners.sum()) if winners.sum() > 0 else 0.0,
    }
    return fig, findings
