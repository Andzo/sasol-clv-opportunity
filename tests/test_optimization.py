"""Unit tests for the Cost-Sensitive Optimal Transport Linear Program module."""
import numpy as np
import pandas as pd
import pytest

from src.optimization import build_cost_sensitive_matrix, solve_optimal_transport_lp


def test_build_cost_sensitive_matrix():
    np.random.seed(42)
    N, K = 50, 4
    P = np.random.uniform(0.1, 0.9, (N, K))
    P = P / P.sum(axis=1, keepdims=True)
    support = np.array([20.0, 15.0, 10.0, 5.0])
    classes = ["Inactivity", "Existing-category growth: Fuel", "Stable", "Other"]

    feats = pd.DataFrame({
        "days_since_last_txn": [10.0] * 25 + [80.0] * 25,
        "txns_30d": [2.0] * 25 + [0.0] * 25,
        "m1_baskets": [1.0] * 25 + [0.0] * 25,
    })

    C = build_cost_sensitive_matrix(
        P, support, features_df=feats, classes=classes, gamma=0.6, active_penalty=50.0
    )

    assert C.shape == (N, K)
    assert not np.isnan(C).any()

    # Verify active penalty applied to Inactivity (class index 0) for first 25 customers
    assert (C[:25, 0] > C[25:, 0]).all()


def test_solve_optimal_transport_lp():
    np.random.seed(42)
    N, K = 30, 3
    # Synthetic cost matrix
    C = np.random.uniform(0.1, 5.0, (N, K))

    # Bounds: sum(L_k) <= N <= sum(U_k)
    L_k = np.array([8, 8, 8])
    U_k = np.array([12, 12, 12])

    X_opt = solve_optimal_transport_lp(C, lower_bounds=L_k, upper_bounds=U_k)

    assert X_opt.shape == (N, K)
    # Every customer assigned to exactly one class
    assert np.allclose(X_opt.sum(axis=1), 1.0)
    # Total assignments satisfy bounds
    col_sums = X_opt.sum(axis=0)
    assert (col_sums >= L_k - 1e-5).all()
    assert (col_sums <= U_k + 1e-5).all()
    # Integer solution (0 or 1)
    assert np.all(np.isclose(X_opt, 0.0) | np.isclose(X_opt, 1.0))
