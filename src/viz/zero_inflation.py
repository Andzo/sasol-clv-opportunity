"""Section 8: Zero-inflation, hurdle modeling, and error decomposition diagnostics (Z01 - Z04)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score

from src.dataset import Snapshot
from src.viz.style import PALETTE, apply_style


def plot_z01_zero_mass_decomposition(snapshots: list[Snapshot]) -> tuple[plt.Figure, dict[str, float]]:
    """Z01: Historical evolution of four customer spending states across cutoffs."""
    apply_style()
    records = []
    for s in snapshots:
        if s.labels is None:
            continue
        l = s.labels
        n = len(l)
        both_z = ((l["CLV_fuel"] == 0) & (l["CLV_nonfuel"] == 0)).sum()
        fuel_only = ((l["CLV_fuel"] > 0) & (l["CLV_nonfuel"] == 0)).sum()
        nf_only = ((l["CLV_fuel"] == 0) & (l["CLV_nonfuel"] > 0)).sum()
        dual = ((l["CLV_fuel"] > 0) & (l["CLV_nonfuel"] > 0)).sum()

        records.append({
            "cutoff": str(s.cutoff.date()),
            "Full Inactivity (0 Fuel, 0 Store)": both_z / n * 100,
            "Forecourt Only (Fuel, No Store)": fuel_only / n * 100,
            "Dual Spender (Fuel + Store)": dual / n * 100,
            "Store Only (No Fuel, Store)": nf_only / n * 100,
        })

    df = pd.DataFrame(records).set_index("cutoff")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    df.plot(kind="bar", stacked=True, ax=ax, color=colors, edgecolor="white", width=0.65)
    ax.set_ylabel("Share of Customer Cohort (%)")
    ax.set_title("Z01: Customer Shopping Decomposition Across Historical Snapshots")
    ax.set_ylim(0, 100)
    plt.xticks(rotation=30, ha="right")
    ax.legend(title="Shopping State", bbox_to_anchor=(1.02, 1), loc="upper left")

    # Annotate latest snapshot
    latest = df.iloc[-1]
    y_accum = 0.0
    for col in df.columns:
        val = latest[col]
        if val > 3.0:
            ax.text(len(df) - 1, y_accum + val / 2, f"{val:.1f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=9)
        y_accum += val

    plt.tight_layout()

    findings = {
        "latest_inactivity_pct": float(latest["Full Inactivity (0 Fuel, 0 Store)"]),
        "latest_forecourt_only_pct": float(latest["Forecourt Only (Fuel, No Store)"]),
        "latest_dual_spender_pct": float(latest["Dual Spender (Fuel + Store)"]),
        "latest_store_only_pct": float(latest["Store Only (No Fuel, Store)"]),
        "forecourt_regular_ratio": float(latest["Forecourt Only (Fuel, No Store)"] / (latest["Forecourt Only (Fuel, No Store)"] + latest["Dual Spender (Fuel + Store)"])),
    }
    return fig, findings


def plot_z02_hurdle_calibration(y_true_pos: np.ndarray, prob_pos: np.ndarray) -> tuple[plt.Figure, dict[str, float]]:
    """Z02: Reliability diagram and Brier score for non-fuel hurdle classifier."""
    apply_style()
    prob_clean = np.clip(prob_pos, 0.0, 1.0)
    prob_true, prob_pred = calibration_curve(y_true_pos, prob_clean, n_bins=10, strategy="uniform")

    auc = float(roc_auc_score(y_true_pos, prob_clean))
    brier = float(brier_score_loss(y_true_pos, prob_clean))
    ece = float(np.mean(np.abs(prob_true - prob_pred)))

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(prob_pred, prob_true, marker="o", linewidth=2.5, color=PALETTE["fuel"], label=f"Hurdle Model (ECE = {ece:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect Calibration")
    ax.set_xlabel("Mean Predicted Probability P(Non-Fuel > 0)")
    ax.set_ylabel("Observed Fraction of Non-Fuel Spenders")
    ax.set_title(f"Z02: Hurdle Classifier Reliability Diagram (ROC-AUC = {auc:.3f}, Brier = {brier:.3f})")
    ax.legend(loc="upper left")
    plt.tight_layout()

    findings = {
        "hurdle_roc_auc": auc,
        "hurdle_brier_score": brier,
        "hurdle_expected_calibration_error": ece,
    }
    return fig, findings


def plot_z03_decomposed_error_boxplots(method_residuals: dict[str, dict[str, np.ndarray]]) -> tuple[plt.Figure, dict[str, Any]]:
    """Z03: Decomposed residual distribution on true zeros vs positive spenders across models."""
    apply_style()
    methods = list(method_residuals.keys())
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    zero_data = [method_residuals[m]["zero_residuals"] for m in methods]
    pos_data = [method_residuals[m]["pos_residuals"] for m in methods]

    # Zero residuals boxplot
    bp0 = axes[0].boxplot(zero_data, tick_labels=[m.replace(" ", "\n") for m in methods], patch_artist=True, showfliers=False)
    for box in bp0["boxes"]:
        box.set(facecolor="#4C72B0", alpha=0.7)
    axes[0].set_ylabel("Absolute Error |y_true - y_pred|")
    axes[0].set_title("Errors on True Zeros (y = 0)\nLower is Better (False Positive Bleed)")

    # Pos residuals boxplot
    bp1 = axes[1].boxplot(pos_data, tick_labels=[m.replace(" ", "\n") for m in methods], patch_artist=True, showfliers=False)
    for box in bp1["boxes"]:
        box.set(facecolor="#55A868", alpha=0.7)
    axes[1].set_ylabel("Absolute Error |y_true - y_pred|")
    axes[1].set_title("Errors on Positive Spenders (y > 0)\nLower is Better (Underprediction Penalty)")

    plt.tight_layout()

    findings = {
        m: {
            "median_err_zeros": float(np.median(method_residuals[m]["zero_residuals"])),
            "median_err_pos": float(np.median(method_residuals[m]["pos_residuals"])),
            "mean_err_zeros": float(np.mean(method_residuals[m]["zero_residuals"])),
            "mean_err_pos": float(np.mean(method_residuals[m]["pos_residuals"])),
        }
        for m in methods
    }
    return fig, findings


def plot_z04_recombination_pareto_curve(pareto_df: pd.DataFrame) -> tuple[go.Figure, dict[str, float]]:
    """Z04: Interactive pareto curve exploring the shrinkage exponent gamma trade-off."""
    fig = px.scatter(
        pareto_df,
        x="rmse_zeros",
        y="rmse_pos",
        color="composite_score",
        size="marker_size",
        hover_data=["gamma", "rmse_total", "composite_score"],
        color_continuous_scale="Viridis",
        title="Z04: Recombination Shrinkage Pareto Front: Error on Zeros vs Positive Spenders (Interactive)",
        labels={
            "rmse_zeros": "RMSE on True Zeros (Penalizes False Bleed)",
            "rmse_pos": "RMSE on Positive Spenders (Penalizes Underprediction)",
            "composite_score": "Composite Score",
        },
        template="plotly_white",
    )
    fig.update_layout(coloraxis_colorbar=dict(title="Proxy Score"))

    best_row = pareto_df.loc[pareto_df["composite_score"].idxmax()]
    fig.add_trace(go.Scatter(
        x=[best_row["rmse_zeros"]],
        y=[best_row["rmse_pos"]],
        mode="markers+text",
        marker=dict(color="red", size=14, symbol="star"),
        text=[f"Optimal gamma={best_row['gamma']:.2f}"],
        textposition="top right",
        name="Optimal Hurdle Exponent",
    ))

    findings = {
        "optimal_gamma": float(best_row["gamma"]),
        "optimal_composite_score": float(best_row["composite_score"]),
        "optimal_rmse_zeros": float(best_row["rmse_zeros"]),
        "optimal_rmse_pos": float(best_row["rmse_pos"]),
    }
    return fig, findings
