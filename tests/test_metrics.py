"""Unit tests for metrics module."""
import numpy as np
import pandas as pd
from src.metrics import normalized_rmse, rmse, score


def test_rmse():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    assert rmse(y_true, y_pred) == 0.0

    y_pred2 = np.array([2.0, 3.0, 4.0])
    assert np.isclose(rmse(y_true, y_pred2), 1.0)


def test_normalized_rmse():
    assert normalized_rmse(0.0, 2.0) == 1.0
    assert normalized_rmse(1.0, 2.0) == 0.5
    assert normalized_rmse(2.5, 2.0) == 0.0  # clipped at 0


def test_score():
    labels = pd.DataFrame({
        "CLV_fuel": [1.0, 2.0],
        "CLV_nonfuel": [0.5, 1.5],
        "Opportunity": ["Stable", "Inactivity"],
    })
    fuel_pred = np.array([1.0, 2.0])
    nonfuel_pred = np.array([0.5, 1.5])
    opp_pred = np.array(["Stable", "Inactivity"])

    sc = score(labels, fuel_pred, nonfuel_pred, opp_pred)
    assert sc.rmse_fuel == 0.0
    assert sc.rmse_nonfuel == 0.0
    assert sc.weighted_f1 == 1.0
    assert sc.composite(1.0, 1.0) == 1.0
    assert np.isclose(sc.leaderboard_composite(), 1.0)
    bd = sc.leaderboard_breakdown()
    assert np.isclose(bd["total_score"], 1.0)
    assert np.isclose(bd["score_fuel"], 0.3)
    assert np.isclose(bd["score_nonfuel"], 0.3)
    assert np.isclose(bd["score_opportunity"], 0.4)

