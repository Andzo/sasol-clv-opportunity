"""Section 7: Model diagnostics, calibration, & bootstrap confidence intervals (M01 - M14)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, f1_score

from src.dataset import Snapshot
from src.metrics import normalized_rmse
from src.viz.style import OPPORTUNITY_COLORS, PALETTE, apply_style


def plot_m01_predicted_vs_actual(eval_preds: pd.DataFrame) -> tuple[plt.Figure, dict[str, float]]:
    """M01: Predicted vs Actual scatter hexbin for Fuel and Non-fuel CLV."""
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Fuel CLV
    hb1 = axes[0].hexbin(eval_preds["true_fuel"], eval_preds["pred_fuel_final"], gridsize=35, cmap="Blues", mincnt=1)
    axes[0].plot([0, 3.5], [0, 3.5], "r--", linewidth=1.5, label="Perfect 1:1 Line")
    axes[0].set_xlabel("Actual Transformed Fuel CLV")
    axes[0].set_ylabel("Predicted Fuel CLV")
    axes[0].set_title("Fuel CLV: Predicted vs Actual")
    axes[0].legend()
    plt.colorbar(hb1, ax=axes[0], fraction=0.046, pad=0.04)

    # Nonfuel CLV
    hb2 = axes[1].hexbin(eval_preds["true_nonfuel"], eval_preds["pred_nonfuel_final"], gridsize=35, cmap="Oranges", mincnt=1)
    axes[1].plot([0, 3.5], [0, 3.5], "r--", linewidth=1.5, label="Perfect 1:1 Line")
    axes[1].set_xlabel("Actual Transformed Non-fuel CLV")
    axes[1].set_ylabel("Predicted Non-fuel CLV")
    axes[1].set_title("Non-fuel CLV: Predicted vs Actual")
    axes[1].legend()
    plt.colorbar(hb2, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    fuel_rmse = float(np.sqrt(np.mean((eval_preds["true_fuel"] - eval_preds["pred_fuel_final"]) ** 2)))
    nf_rmse = float(np.sqrt(np.mean((eval_preds["true_nonfuel"] - eval_preds["pred_nonfuel_final"]) ** 2)))
    findings = {
        "fuel_eval_rmse": fuel_rmse,
        "nonfuel_eval_rmse": nf_rmse,
    }
    return fig, findings


def plot_m03_p_active_calibration(eval_preds: pd.DataFrame) -> tuple[plt.Figure, dict[str, float]]:
    """M03: Reliability calibration curve for predicted p_active probability."""
    apply_style()
    y_true_active = (eval_preds["true_opportunity"] != "Inactivity").astype(int)
    prob_active = eval_preds["p_active"].clip(0.0, 1.0)

    prob_true, prob_pred = calibration_curve(y_true_active, prob_active, n_bins=10)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(prob_pred, prob_true, marker="o", linewidth=2, color=PALETTE["fuel"], label="Model p_active")
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Actives (Observed)")
    ax.set_title("M03: Reliability Diagram for p_active Churn Predictor")
    ax.legend()
    plt.tight_layout()

    ece = float(np.mean(np.abs(prob_true - prob_pred)))
    findings = {
        "expected_calibration_error_ece": ece,
    }
    return fig, findings


def plot_m04_confusion_matrix(eval_preds: pd.DataFrame, opp_labels: list[str]) -> tuple[plt.Figure, dict[str, float]]:
    """M04: Row-normalized confusion matrix across top Opportunity classes."""
    apply_style()
    top_5 = ["Inactivity", "Existing-category growth: Fuel", "Stable", "Existing-category growth: Other", "Existing-category growth: Beverages"]
    sub = eval_preds[eval_preds["true_opportunity"].isin(top_5)].copy()
    cm = confusion_matrix(sub["true_opportunity"], sub["pred_opportunity"], labels=top_5)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(top_5)))
    ax.set_yticks(range(len(top_5)))
    short_labels = [l.replace("Existing-category growth: ", "Growth: ") for l in top_5]
    ax.set_xticklabels(short_labels, rotation=35, ha="right", fontsize=8.5)
    ax.set_yticklabels(short_labels, fontsize=8.5)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Ground-Truth Label")
    ax.set_title("M04: Normalized Confusion Matrix on Dominant Classes")

    for i in range(len(top_5)):
        for j in range(len(top_5)):
            val = cm_norm[i, j]
            ax.text(j, i, f"{val:.1%}\n({cm[i, j]})", ha="center", va="center", color="white" if val > 0.4 else "black", fontsize=8)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    findings = {
        "inactivity_recall": float(cm_norm[0, 0]),
        "fuel_growth_recall": float(cm_norm[1, 1]),
        "stable_recall": float(cm_norm[2, 2]),
    }
    return fig, findings


def plot_m06_bootstrap_uncertainty(eval_preds: pd.DataFrame, n_bootstraps: int = 1000) -> tuple[plt.Figure, dict[str, float]]:
    """M06: Bootstrap sampling (1,000 iterations) of composite score addressing single holdout risk."""
    apply_style()
    x_max_f = 1.8344871663064635
    x_max_nf = 1.2240635969972369
    n = len(eval_preds)

    rng = np.random.RandomState(42)
    scores = []
    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, size=n)
        sample = eval_preds.iloc[idx]
        f_rmse = float(np.sqrt(np.mean((sample["true_fuel"] - sample["pred_fuel_final"]) ** 2)))
        nf_rmse = float(np.sqrt(np.mean((sample["true_nonfuel"] - sample["pred_nonfuel_final"]) ** 2)))
        f1 = float(f1_score(sample["true_opportunity"], sample["pred_opportunity"], average="weighted", zero_division=0))
        comp = (
            0.3 * normalized_rmse(f_rmse, x_max_f)
            + 0.3 * normalized_rmse(nf_rmse, x_max_nf)
            + 0.4 * f1
        )
        scores.append(comp)

    scores_arr = np.array(scores)
    ci_low = float(np.percentile(scores_arr, 2.5))
    ci_high = float(np.percentile(scores_arr, 97.5))
    mean_val = float(np.mean(scores_arr))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(scores_arr, bins=35, color=PALETTE["fuel"], edgecolor="white", alpha=0.85)
    ax.axvline(mean_val, color="black", linestyle="-", linewidth=2, label=f"Mean ({mean_val:.4f})")
    ax.axvline(ci_low, color="#D55E00", linestyle="--", linewidth=1.5, label=f"95% CI Lower ({ci_low:.4f})")
    ax.axvline(ci_high, color="#D55E00", linestyle="--", linewidth=1.5, label=f"95% CI Upper ({ci_high:.4f})")
    ax.set_xlabel("Composite Proxy Score")
    ax.set_ylabel("Bootstrap Replicates")
    ax.set_title(f"M06: Empirical Uncertainty (1,000 Bootstraps, Holdout Risk Assessment)")
    ax.legend()
    plt.tight_layout()

    findings = {
        "bootstrap_mean_score": mean_val,
        "bootstrap_ci_95_low": ci_low,
        "bootstrap_ci_95_high": ci_high,
        "bootstrap_ci_width": ci_high - ci_low,
        "bootstrap_std": float(np.std(scores_arr)),
    }
    return fig, findings


def plot_m13_oracle_upper_bounds(eval_preds: pd.DataFrame) -> tuple[plt.Figure, dict[str, float]]:
    """M13: Oracle simulation: what would the composite score become if components were perfected?"""
    apply_style()
    x_max_f = 1.8344871663064635
    x_max_nf = 1.2240635969972369

    base_f_rmse = float(np.sqrt(np.mean((eval_preds["true_fuel"] - eval_preds["pred_fuel_final"]) ** 2)))
    base_nf_rmse = float(np.sqrt(np.mean((eval_preds["true_nonfuel"] - eval_preds["pred_nonfuel_final"]) ** 2)))
    base_f1 = float(f1_score(eval_preds["true_opportunity"], eval_preds["pred_opportunity"], average="weighted", zero_division=0))
    base_comp = 0.3 * normalized_rmse(base_f_rmse, x_max_f) + 0.3 * normalized_rmse(base_nf_rmse, x_max_nf) + 0.4 * base_f1

    # Scenario 1: Perfect Inactivity classification (all Inactivity correctly classified)
    sc1_opp = eval_preds["pred_opportunity"].copy()
    sc1_opp[eval_preds["true_opportunity"] == "Inactivity"] = "Inactivity"
    sc1_f1 = float(f1_score(eval_preds["true_opportunity"], sc1_opp, average="weighted", zero_division=0))
    sc1_comp = 0.3 * normalized_rmse(base_f_rmse, x_max_f) + 0.3 * normalized_rmse(base_nf_rmse, x_max_nf) + 0.4 * sc1_f1

    # Scenario 2: Perfect Fuel Growth vs Stable distinction
    sc2_opp = eval_preds["pred_opportunity"].copy()
    mask_fs = eval_preds["true_opportunity"].isin(["Existing-category growth: Fuel", "Stable"])
    sc2_opp[mask_fs] = eval_preds["true_opportunity"][mask_fs]
    sc2_f1 = float(f1_score(eval_preds["true_opportunity"], sc2_opp, average="weighted", zero_division=0))
    sc2_comp = 0.3 * normalized_rmse(base_f_rmse, x_max_f) + 0.3 * normalized_rmse(base_nf_rmse, x_max_nf) + 0.4 * sc2_f1

    # Scenario 3: Perfect Nonfuel zero detection (nonfuel RMSE drops to 0.45)
    sc3_comp = 0.3 * normalized_rmse(base_f_rmse, x_max_f) + 0.3 * normalized_rmse(0.45, x_max_nf) + 0.4 * base_f1

    scenarios = {
        "Current Baseline": base_comp,
        "+ Perfect Inactivity": sc1_comp,
        "+ Perfect Fuel vs Stable": sc2_comp,
        "+ Low-Error Non-fuel (0.45)": sc3_comp,
    }

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(scenarios.keys(), scenarios.values(), color=[PALETTE["dark"], PALETTE["fuel"], PALETTE["accent"], PALETTE["nonfuel"]])
    ax.axhline(0.70, color="red", linestyle="--", label="Target Score 0.70")
    ax.set_ylabel("Composite Proxy Score")
    ax.set_title("M13: Theoretical Upper Bounds from Component Improvements")
    ax.set_ylim(0.4, 0.75)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.005, f"{h:.4f}", ha="center", va="bottom", fontsize=8.5)
    ax.legend()
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    findings = scenarios
    return fig, findings
