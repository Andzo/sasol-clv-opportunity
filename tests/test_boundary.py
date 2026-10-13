"""Unit tests for the specialized binary boundary classifier module."""
import numpy as np
import pandas as pd
import pytest

from src.boundary import build_boundary_features, BinaryBoundaryClassifier


def test_build_boundary_features_synthetic():
    df = pd.DataFrame({
        "fuel_litres_90d": [100.0, 50.0, 0.0],
        "fuel_litres_30d": [40.0, 10.0, 0.0],
        "fuel_litres_180d": [180.0, 120.0, 0.0],
        "cat_q1_spend_Fuel": [2300.0, 1150.0, 0.0],
        "fuel_txns_30d": [4.0, 1.0, 0.0],
        "fuel_txns_90d": [10.0, 3.0, 0.0],
    })
    pred_fuel = np.array([1.2, 0.8, 0.0])

    aug = build_boundary_features(df, pred_fuel=pred_fuel)

    assert "bnd_hurdle_25pct_rand" in aug.columns
    assert "bnd_litres_m1_m3_accel" in aug.columns
    assert "bnd_fuel_freq_accel" in aug.columns
    assert "bnd_fuel_reversion_score" in aug.columns
    assert "bnd_total_spend_m1_m3_accel" in aug.columns

    # Verify no NaNs introduced
    assert not aug.isna().any().any()

    # Check non-empty without pred_fuel
    aug_no_pred = build_boundary_features(df, pred_fuel=None)
    assert "bnd_hurdle_to_spend_ratio" in aug_no_pred.columns
    assert not aug_no_pred.isna().any().any()


def test_binary_boundary_classifier_fit_predict():
    np.random.seed(42)
    n = 60
    X = pd.DataFrame({
        "feat1": np.random.randn(n),
        "feat2": np.random.randn(n) * 2.0,
        "fuel_litres_90d": np.random.uniform(10, 200, n),
        "fuel_litres_30d": np.random.uniform(5, 70, n),
        "fuel_litres_180d": np.random.uniform(20, 400, n),
        "cat_q1_spend_Fuel": np.random.uniform(200, 5000, n),
        "fuel_txns_30d": np.random.randint(1, 6, n),
        "fuel_txns_90d": np.random.randint(2, 15, n),
    })
    pred_fuel = np.random.uniform(0.5, 2.0, n)
    X_aug = build_boundary_features(X, pred_fuel)

    # Synthetic binary label: 1 = FG, 0 = Stable
    y = ((X["feat1"] + 0.5 * X_aug["bnd_litres_m1_m3_accel"]) > 1.0).astype(int).to_numpy()

    clf = BinaryBoundaryClassifier(seed=42, n_jobs=1)
    # Fast test with small iterations
    clf.clf_lgb.set_params(n_estimators=10)
    clf.clf_cb.set_params(iterations=10)
    clf.clf_xgb.set_params(n_estimators=10)

    clf.fit(X_aug, y)
    probs = clf.predict_proba(X_aug)

    assert probs.shape == (n, 2)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()
