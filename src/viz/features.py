"""Section 5: Feature diagnostics & mutual information (F01 - F10)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from src.dataset import Snapshot
from src.viz.style import PALETTE, apply_style


def plot_f01_correlation_clustermap(snap: Snapshot, n_top: int = 25) -> tuple[plt.Figure, dict[str, float]]:
    """F01: Clustered Spearman rank correlation heatmap among key engineered features."""
    apply_style()
    # Select non-constant numerical features with variance
    vars_ = snap.features.var()
    top_cols = vars_.sort_values(ascending=False).head(n_top).index
    corr = snap.features[top_cols].corr(method="spearman").fillna(0.0)

    fig, ax = plt.subplots(figsize=(10, 8.5))
    im = ax.imshow(corr.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(top_cols)))
    ax.set_yticks(range(len(top_cols)))
    ax.set_xticklabels(top_cols, rotation=60, ha="right", fontsize=8)
    ax.set_yticklabels(top_cols, fontsize=8)
    ax.set_title(f"F01: Spearman Rank Correlation Matrix (Top {n_top} Dynamic Features)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    triu_idx = np.triu_indices_from(corr, k=1)
    high_corr_count = int((corr.to_numpy()[triu_idx] > 0.85).sum())

    findings = {
        "pairs_with_spearman_gt_0.85": high_corr_count,
        "max_offdiag_correlation": float(np.max(np.abs(corr.to_numpy()[triu_idx]))),
    }
    return fig, findings


def plot_f02_mutual_information(snap: Snapshot) -> tuple[plt.Figure, dict[str, str]]:
    """F02: Mutual Information ranking of features against each challenge target."""
    apply_style()
    if snap.labels is None:
        raise ValueError("Snapshot must have labels")

    features = snap.features.fillna(0.0)
    sample_idx = np.random.RandomState(42).choice(len(features), min(1500, len(features)), replace=False)
    X_sub = features.iloc[sample_idx]
    y_fuel = snap.labels["CLV_fuel"].iloc[sample_idx]
    y_nf = snap.labels["CLV_nonfuel"].iloc[sample_idx]
    y_opp = snap.labels["Opportunity"].astype("category").cat.codes.iloc[sample_idx]

    mi_f = pd.Series(mutual_info_regression(X_sub, y_fuel, random_state=42), index=X_sub.columns).sort_values(ascending=False).head(8)
    mi_nf = pd.Series(mutual_info_regression(X_sub, y_nf, random_state=42), index=X_sub.columns).sort_values(ascending=False).head(8)
    mi_opp = pd.Series(mutual_info_classif(X_sub, y_opp, random_state=42), index=X_sub.columns).sort_values(ascending=False).head(8)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    axes[0].barh(mi_f.index[::-1], mi_f.values[::-1], color=PALETTE["fuel"])
    axes[0].set_title("Top MI: Fuel CLV")
    axes[0].set_xlabel("Mutual Information")

    axes[1].barh(mi_nf.index[::-1], mi_nf.values[::-1], color=PALETTE["nonfuel"])
    axes[1].set_title("Top MI: Non-Fuel CLV")
    axes[1].set_xlabel("Mutual Information")

    axes[2].barh(mi_opp.index[::-1], mi_opp.values[::-1], color=PALETTE["accent"])
    axes[2].set_title("Top MI: Opportunity Label")
    axes[2].set_xlabel("Mutual Information")

    plt.tight_layout()
    findings = {
        "top_fuel_feature": str(mi_f.index[0]),
        "top_nonfuel_feature": str(mi_nf.index[0]),
        "top_opportunity_feature": str(mi_opp.index[0]),
    }
    return fig, findings


def plot_f04_class_conditional_distributions(snap: Snapshot) -> tuple[plt.Figure, dict[str, float]]:
    """F04: Class-conditional distributions separating Inactivity, Fuel Growth, and Stable."""
    apply_style()
    if snap.labels is None:
        raise ValueError("Snapshot must have labels")

    df = pd.DataFrame({
        "days_since_last_purchase": snap.features["days_since_last_purchase"].clip(upper=120),
        "fuel_m1_m3_ratio": snap.features["fuel_m1_m3_ratio"].clip(upper=3),
        "Opportunity": snap.labels["Opportunity"],
    })
    target_classes = ["Inactivity", "Existing-category growth: Fuel", "Stable"]
    sub_df = df[df["Opportunity"].isin(target_classes)]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for cls in target_classes:
        subset = sub_df[sub_df["Opportunity"] == cls]
        axes[0].hist(subset["days_since_last_purchase"], bins=25, alpha=0.5, density=True, label=cls.replace("Existing-category growth: ", ""))
        axes[1].hist(subset["fuel_m1_m3_ratio"], bins=25, alpha=0.5, density=True, label=cls.replace("Existing-category growth: ", ""))

    axes[0].set_title("Days Since Last Basket")
    axes[0].set_xlabel("Days")
    axes[0].legend()

    axes[1].set_title("Fuel M1 / M3 Momentum Ratio")
    axes[1].set_xlabel("Ratio")
    axes[1].legend()

    plt.tight_layout()
    findings = {
        "inactivity_median_days_since_last": float(sub_df[sub_df["Opportunity"] == "Inactivity"]["days_since_last_purchase"].median()),
        "growth_median_days_since_last": float(sub_df[sub_df["Opportunity"] == "Existing-category growth: Fuel"]["days_since_last_purchase"].median()),
    }
    return fig, findings
