"""Section 6: Customer segmentation & site network effects (G01 - G07)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.dataset import Snapshot
from src.viz.style import PALETTE, apply_style


def plot_g01_rfm_clustering(snap: Snapshot) -> tuple[px.scatter, dict[str, Any]]:
    """G01: Interactive customer archetypes via RFM KMeans & PCA embedding."""
    feats = snap.features[["days_since_last_purchase", "m3_baskets", "m3_fuel_rand", "m3_nonfuel_rand"]].fillna(0.0)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(np.log1p(feats))

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(scaled)

    cluster_names = {
        0: "Forecourt Regulars",
        1: "Convenience Cross-Shoppers",
        2: "Lapsed / Infrequent",
        3: "High-Volume Core",
    }
    df = pd.DataFrame({
        "ID": snap.ids,
        "PCA_1": coords[:, 0],
        "PCA_2": coords[:, 1],
        "Cluster": [cluster_names.get(c, f"Segment {c}") for c in cluster_labels],
        "m3_fuel_rand": feats["m3_fuel_rand"],
        "m3_nonfuel_rand": feats["m3_nonfuel_rand"],
        "m3_baskets": feats["m3_baskets"],
    })
    sample_df = df.sample(min(1200, len(df)), random_state=42)

    fig = px.scatter(
        sample_df,
        x="PCA_1",
        y="PCA_2",
        color="Cluster",
        hover_data=["ID", "m3_fuel_rand", "m3_nonfuel_rand", "m3_baskets"],
        title="G01: Customer Behavioral Archetypes (PCA 2D Embedding, Interactive)",
        opacity=0.8,
        template="plotly_white",
    )

    findings = {
        "pca_variance_explained": [float(v) for v in pca.explained_variance_ratio_],
        "cluster_distribution": df["Cluster"].value_counts().to_dict(),
    }
    return fig, findings


def plot_g04_site_effects_icc(tx: pd.DataFrame) -> tuple[plt.Figure, dict[str, float]]:
    """G04: Site variation and Intraclass Correlation Coefficient (ICC) across Sasol forecourts."""
    apply_style()
    valid_tx = tx[tx["is_fuel_sale"] & (tx["qty"] > 0)].copy()
    site_counts = valid_tx.groupby("LocationId")["TransactionId"].count()
    top_sites = site_counts[site_counts >= 100].index

    subset = valid_tx[valid_tx["LocationId"].isin(top_sites)]
    site_means = subset.groupby("LocationId")["qty"].mean().sort_values()

    # Approximate one-way ANOVA ICC
    grand_mean = subset["qty"].mean()
    ss_total = ((subset["qty"] - grand_mean) ** 2).sum()
    site_grp = subset.groupby("LocationId")["qty"]
    ss_between = (site_grp.count() * (site_grp.mean() - grand_mean) ** 2).sum()
    eta_squared = float(ss_between / ss_total) if ss_total > 0 else 0.0

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.hist(site_means, bins=30, color=PALETTE["fuel"], edgecolor="white")
    ax.set_xlabel("Mean Fuel Transaction Volume per Site (Litres)")
    ax.set_ylabel("Number of Retail Sites")
    ax.set_title(f"G04: Site Network Heterogeneity (Eta-Squared ICC = {eta_squared:.3f})")
    plt.tight_layout()

    findings = {
        "site_heterogeneity_icc": eta_squared,
        "num_analyzed_sites": int(len(top_sites)),
        "min_site_mean_litres": float(site_means.min()),
        "max_site_mean_litres": float(site_means.max()),
    }
    return fig, findings
