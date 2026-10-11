"""Unit tests for visualization, diagnostics, and registry suite."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from src.dataset import Snapshot
from src.viz.cohorts import plot_c05_gap_survival_curve
from src.viz.models import plot_m01_predicted_vs_actual, plot_m06_bootstrap_uncertainty
from src.viz.raw import plot_d01_daily_activity, plot_d02_category_shares, plot_d04_fuel_fill_modes
from src.viz.registry import VisualRegistry
from src.viz.targets import plot_t01_target_distributions
from src.viz.temporal import compute_psi


@pytest.fixture
def dummy_tx() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=100, freq="12h")
    return pd.DataFrame({
        "TransactionId": [f"T{i}" for i in range(100)],
        "ID": [f"C{i % 10}" for i in range(100)],
        "_time": dates,
        "amount": np.random.uniform(50, 800, size=100),
        "qty": np.random.uniform(5, 50, size=100),
        "is_fuel_sale": [i % 2 == 0 for i in range(100)],
        "is_qualifying": [True] * 100,
        "ItemCategoryLevel2": ["Fuel" if i % 2 == 0 else "Beverages" for i in range(100)],
        "LocationId": [f"L{i % 3}" for i in range(100)],
    })


@pytest.fixture
def dummy_snapshot() -> Snapshot:
    n = 50
    ids = pd.Index([f"C{i}" for i in range(n)], name="ID")
    feats = pd.DataFrame({
        "gap_median": np.random.uniform(5, 45, size=n),
        "days_since_first_seen_capped": np.random.uniform(60, 365, size=n),
        "baskets_per_month_life": np.random.uniform(1, 8, size=n),
        "days_since_last_purchase": np.random.uniform(1, 60, size=n),
        "m3_fuel_litres": np.random.uniform(20, 200, size=n),
        "m3_fuel_rand": np.random.uniform(400, 4500, size=n),
        "m3_nonfuel_rand": np.random.uniform(0, 500, size=n),
        "m3_baskets": np.random.randint(1, 15, size=n),
        "fuel_price_m3": np.random.uniform(22, 25, size=n),
    }, index=ids)
    mask = np.ones((n, 9), dtype=bool)
    labels = pd.DataFrame({
        "fuel_litres": np.random.uniform(0, 150, size=n),
        "CLV_fuel": np.random.uniform(0, 2.5, size=n),
        "nonfuel_rand": np.random.uniform(0, 800, size=n),
        "CLV_nonfuel": np.random.uniform(0, 2.0, size=n),
        "Opportunity": ["Stable" if i % 2 == 0 else "Inactivity" for i in range(n)],
        "winning_source_category": ["Fuel" if i % 2 == 0 else "" for i in range(n)],
        "winning_increment_rand": [50.0 if i % 2 == 0 else 0.0 for i in range(n)],
    }, index=ids)
    return Snapshot(pd.Timestamp("2025-09-01"), feats, mask, labels)


@pytest.fixture
def dummy_predictions() -> pd.DataFrame:
    n = 60
    ids = pd.Index([f"C{i}" for i in range(n)], name="ID")
    return pd.DataFrame({
        "pred_fuel_raw": np.random.uniform(0.5, 2.0, size=n),
        "pred_fuel_final": np.random.uniform(0.5, 2.0, size=n),
        "pred_nonfuel_raw": np.random.uniform(0.2, 1.5, size=n),
        "pred_nonfuel_final": np.random.uniform(0.2, 1.5, size=n),
        "pred_opportunity": ["Stable" if i % 2 == 0 else "Inactivity" for i in range(n)],
        "p_active": np.random.uniform(0.1, 0.9, size=n),
        "true_fuel": np.random.uniform(0.5, 2.0, size=n),
        "true_nonfuel": np.random.uniform(0.2, 1.5, size=n),
        "true_opportunity": ["Stable" if i % 3 == 0 else ("Inactivity" if i % 3 == 1 else "Existing-category growth: Fuel") for i in range(n)],
    }, index=ids)


def test_registry_lifecycle(tmp_path: Path):
    reg = VisualRegistry(tmp_path)

    @reg.register("TEST_STATIC", "sec1", "Test Static Figure", is_interactive=False)
    def dummy_static():
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [4, 5, 6])
        return fig, {"slope": 1.0}

    @reg.register("TEST_INTERACTIVE", "sec2", "Test Interactive Figure", is_interactive=True)
    def dummy_interactive():
        fig = go.Figure(data=[go.Scatter(x=[1, 2], y=[3, 4])])
        return fig, {"point_count": 2}

    fig1, f1 = reg.run_and_save("TEST_STATIC")
    assert f1["slope"] == 1.0
    assert (tmp_path / "sec1" / "TEST_STATIC.png").exists()

    fig2, f2 = reg.run_and_save("TEST_INTERACTIVE")
    assert f2["point_count"] == 2
    assert (tmp_path / "sec2" / "TEST_INTERACTIVE.html").exists()

    manifest_file = reg.save_manifest()
    assert manifest_file.exists()
    data = json.loads(manifest_file.read_text())
    assert len(data) == 2
    assert data[0]["fig_id"] == "TEST_STATIC"
    assert data[1]["fig_id"] == "TEST_INTERACTIVE"


def test_compute_psi():
    s1 = pd.Series(np.random.normal(0, 1, 1000))
    s2 = pd.Series(np.random.normal(0, 1, 1000))
    psi_stable = compute_psi(s1, s2)
    assert 0.0 <= psi_stable < 0.15

    s3 = pd.Series(np.random.normal(3, 1, 1000))
    psi_shifted = compute_psi(s1, s3)
    assert psi_shifted > 0.5


def test_raw_plots(dummy_tx: pd.DataFrame):
    fig1, f1 = plot_d01_daily_activity(dummy_tx)
    assert isinstance(fig1, go.Figure)
    assert "mean_daily_transactions" in f1

    fig2, f2 = plot_d02_category_shares(dummy_tx)
    assert isinstance(fig2, plt.Figure)
    assert "fuel_line_share_pct" in f2
    plt.close(fig2)

    fig4, f4 = plot_d04_fuel_fill_modes(dummy_tx)
    assert isinstance(fig4, plt.Figure)
    assert "fuel_fill_median_litres" in f4
    plt.close(fig4)


def test_cohort_and_target_plots(dummy_snapshot: Snapshot):
    fig_c5, fc5 = plot_c05_gap_survival_curve(dummy_snapshot)
    assert isinstance(fig_c5, plt.Figure)
    assert "p_gap_greater_30d" in fc5
    plt.close(fig_c5)

    fig_t1, ft1 = plot_t01_target_distributions(dummy_snapshot)
    assert isinstance(fig_t1, plt.Figure)
    assert "pct_zero_fuel" in ft1
    plt.close(fig_t1)


def test_model_plots(dummy_predictions: pd.DataFrame):
    fig_m1, fm1 = plot_m01_predicted_vs_actual(dummy_predictions)
    assert isinstance(fig_m1, plt.Figure)
    assert "fuel_eval_rmse" in fm1
    plt.close(fig_m1)

    fig_m6, fm6 = plot_m06_bootstrap_uncertainty(dummy_predictions, n_bootstraps=50)
    assert isinstance(fig_m6, plt.Figure)
    assert "bootstrap_ci_95_low" in fm6
    assert "bootstrap_ci_95_high" in fm6
    assert fm6["bootstrap_ci_95_low"] <= fm6["bootstrap_ci_95_high"]
    plt.close(fig_m6)


def test_zero_inflation_plots(dummy_snapshot: Snapshot):
    from src.viz.zero_inflation import (
        plot_z01_zero_mass_decomposition,
        plot_z02_hurdle_calibration,
        plot_z03_decomposed_error_boxplots,
        plot_z04_recombination_pareto_curve,
    )

    fig_z1, f_z1 = plot_z01_zero_mass_decomposition([dummy_snapshot])
    assert isinstance(fig_z1, plt.Figure)
    assert "latest_inactivity_pct" in f_z1
    plt.close(fig_z1)

    y_true = np.array([0, 1, 0, 1, 1, 0])
    prob = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3])
    fig_z2, f_z2 = plot_z02_hurdle_calibration(y_true, prob)
    assert isinstance(fig_z2, plt.Figure)
    assert "hurdle_roc_auc" in f_z2
    plt.close(fig_z2)

    residuals = {
        "Base": {"zero_residuals": np.array([0.1, 0.2]), "pos_residuals": np.array([0.3, 0.4])},
        "Hurdle": {"zero_residuals": np.array([0.05, 0.1]), "pos_residuals": np.array([0.25, 0.35])},
    }
    fig_z3, f_z3 = plot_z03_decomposed_error_boxplots(residuals)
    assert isinstance(fig_z3, plt.Figure)
    assert "Base" in f_z3
    plt.close(fig_z3)

    pareto_df = pd.DataFrame({
        "gamma": [0.8, 1.0, 1.2],
        "rmse_total": [0.76, 0.75, 0.755],
        "rmse_zeros": [0.60, 0.58, 0.56],
        "rmse_pos": [0.98, 1.00, 1.02],
        "composite_score": [0.518, 0.520, 0.519],
        "marker_size": [8, 8, 8],
    })
    fig_z4, f_z4 = plot_z04_recombination_pareto_curve(pareto_df)
    assert isinstance(fig_z4, go.Figure)
    assert "optimal_gamma" in f_z4

