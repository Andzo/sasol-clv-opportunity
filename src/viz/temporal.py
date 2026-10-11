"""Section 4: Seasonality, temporal shift & adversarial validation (S01 - S10)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

from src.dataset import Snapshot
from src.viz.style import PALETTE, apply_style


def compute_psi(train_series: pd.Series, test_series: pd.Series, num_bins: int = 10) -> float:
    """Compute Population Stability Index (PSI) between training and test feature distributions."""
    t_clean = train_series.dropna()
    te_clean = test_series.dropna()
    if len(t_clean) < 50 or len(te_clean) < 50:
        return 0.0

    quantiles = np.linspace(0, 1, num_bins + 1)
    bins = np.percentile(t_clean, quantiles * 100)
    bins[0] = -np.inf
    bins[-1] = np.inf
    bins = np.unique(bins)
    if len(bins) < 3:
        return 0.0

    train_counts = np.histogram(t_clean, bins=bins)[0] + 1e-4
    test_counts = np.histogram(te_clean, bins=bins)[0] + 1e-4

    train_pct = train_counts / train_counts.sum()
    test_pct = test_counts / test_counts.sum()

    psi = np.sum((test_pct - train_pct) * np.log(test_pct / train_pct))
    return float(psi)


def plot_s02_feature_psi_heatmap(train_snap: Snapshot, test_snap: Snapshot) -> tuple[plt.Figure, dict[str, float]]:
    """S02: Population Stability Index (PSI) across core feature groups."""
    apply_style()
    feature_candidates = [
        "m3_fuel_litres", "m3_fuel_rand", "m3_nonfuel_rand", "m3_baskets",
        "m1_baskets", "days_since_last_purchase", "gap_median",
        "fuel_m1_m3_ratio", "nonfuel_m1_m3_ratio", "store_attachment_rate",
        "baskets_per_month_life", "fuel_price_m3", "exp_churn_hazard"
    ]
    present = [f for f in feature_candidates if f in train_snap.features.columns and f in test_snap.features.columns]

    psi_values = {}
    for col in present:
        psi_values[col] = compute_psi(train_snap.features[col], test_snap.features[col])

    psi_s = pd.Series(psi_values).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    colors = ["#D55E00" if v > 0.25 else ("#E69F00" if v > 0.10 else PALETTE["fuel"]) for v in psi_s.values]
    ax.barh(psi_s.index[::-1], psi_s.values[::-1], color=colors[::-1], edgecolor="none")
    ax.axvline(0.10, color="gray", linestyle="--", label="Moderate Shift (PSI=0.10)")
    ax.axvline(0.25, color="red", linestyle=":", label="Significant Shift (PSI=0.25)")
    ax.set_xlabel("Population Stability Index (PSI)")
    ax.set_title(f"S02: Feature Drift Analysis (Train {train_snap.cutoff.date()} vs Test {test_snap.cutoff.date()})")
    ax.legend()
    plt.tight_layout()

    findings = {
        "max_feature_psi": float(psi_s.max()) if not psi_s.empty else 0.0,
        "features_above_0.10": int((psi_s > 0.10).sum()),
        "features_above_0.25": int((psi_s > 0.25).sum()),
    }
    return fig, findings


def plot_s03_adversarial_validation(train_snap: Snapshot, test_snap: Snapshot) -> tuple[plt.Figure, dict[str, float]]:
    """S03: Adversarial validation: LightGBM predicting Train (0) vs Test (1)."""
    apply_style()
    common_cols = [c for c in train_snap.features.columns if c in test_snap.features.columns and not c.startswith("cutoff_")]
    X_train = train_snap.features[common_cols].copy()
    X_test = test_snap.features[common_cols].copy()

    X = pd.concat([X_train, X_test], axis=0)
    y = np.concatenate([np.zeros(len(X_train)), np.ones(len(X_test))])

    # Simple 2-fold cross-validation
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X))

    for tr_idx, val_idx in kf.split(X):
        clf = LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42, verbose=-1, n_jobs=-1)
        clf.fit(X.iloc[tr_idx], y[tr_idx])
        oof_preds[val_idx] = clf.predict_proba(X.iloc[val_idx])[:, 1]

    auc = roc_auc_score(y, oof_preds)

    # Train full model for feature importances
    full_clf = LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42, verbose=-1, n_jobs=-1)
    full_clf.fit(X, y)
    imp = pd.Series(full_clf.feature_importances_, index=common_cols).sort_values(ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(imp.index[::-1], imp.values[::-1], color=PALETTE["accent"])
    ax.set_xlabel("Feature Importance (Split Gain)")
    ax.set_title(f"S03: Adversarial Validation Feature Drift (ROC-AUC = {auc:.3f})")
    plt.tight_layout()

    findings = {
        "adversarial_roc_auc": float(auc),
        "top_drifting_feature": str(imp.index[0]) if not imp.empty else "none",
    }
    return fig, findings


def plot_s08_rolling_origin_metrics(rolling_evals: list[dict[str, float]]) -> tuple[plt.Figure, dict[str, float]]:
    """S08: Rolling-origin evaluation metrics across multiple quarterly cutoffs."""
    apply_style()
    df = pd.DataFrame(rolling_evals)
    if df.empty or "cutoff" not in df.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "Rolling holdout evaluation in progress", ha="center")
        return fig, {}

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    x = range(len(df))
    ax.plot(x, df["composite_score"], marker="o", color="#0072B2", linewidth=2.5, label="Composite Proxy Score")
    ax.plot(x, df["weighted_f1"], marker="s", color="#009E73", linewidth=2, label="Opportunity Weighted F1")
    ax.plot(x, df["norm_fuel"], marker="^", color="#E69F00", linewidth=1.5, linestyle="--", label="Norm Fuel Score")
    ax.plot(x, df["norm_nonfuel"], marker="v", color="#CC79A7", linewidth=1.5, linestyle="--", label="Norm Non-Fuel Score")

    ax.set_xticks(x)
    ax.set_xticklabels(df["cutoff"], rotation=25)
    ax.set_ylabel("Score (Higher is Better)")
    ax.set_title("S08: Rolling-Origin Multi-Cutoff Evaluation (Temporal Stability)")
    ax.set_ylim(0.2, 0.8)
    ax.legend(loc="lower left")
    plt.tight_layout()

    findings = {
        "mean_rolling_composite": float(df["composite_score"].mean()),
        "std_rolling_composite": float(df["composite_score"].std()),
        "min_rolling_composite": float(df["composite_score"].min()),
        "max_rolling_composite": float(df["composite_score"].max()),
    }
    return fig, findings
