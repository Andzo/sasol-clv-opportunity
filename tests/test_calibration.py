"""Unit tests for calibration and post-processing algorithms."""
import numpy as np
from src.calibration import (
    apply_eligibility_mask,
    decide,
    fit_blend_weights,
    fit_class_weights,
    weighted_f1,
)


def test_apply_eligibility_mask():
    probs = np.array([
        [0.4, 0.4, 0.2],
        [0.1, 0.8, 0.1],
    ])
    mask = np.array([
        [True, False, True],  # class 1 masked
        [False, False, False], # all masked -> fallback uniform
    ])
    masked = apply_eligibility_mask(probs, mask)
    
    # Row 0: class 1 zeroed, remaining renormalized
    assert masked[0, 1] == 0.0
    assert np.isclose(masked[0, 0], 0.4 / 0.6)
    assert np.isclose(masked[0, 2], 0.2 / 0.6)
    assert np.isclose(masked[0].sum(), 1.0)

    # Row 1: fallback uniform over mask (all false -> uniform)
    assert np.isclose(masked[1].sum(), 1.0)


def test_fit_class_weights_improves_or_maintains_f1():
    rng = np.random.default_rng(42)
    n_samples, n_classes = 500, 5
    probs = rng.dirichlet(np.ones(n_classes), size=n_samples)
    y_true = rng.integers(0, n_classes, size=n_samples)

    init_preds = decide(probs)
    init_f1 = weighted_f1(y_true, init_preds, n_classes)

    weights = fit_class_weights(probs, y_true, n_passes=2)
    cal_preds = decide(probs, weights)
    cal_f1 = weighted_f1(y_true, cal_preds, n_classes)

    assert cal_f1 >= init_f1 - 1e-9


def test_fit_blend_weights():
    # Synthetic member predictions
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    member_preds = np.column_stack([
        y_true + 0.1,
        y_true - 0.2,
        np.zeros(4)
    ])
    w = fit_blend_weights(member_preds, y_true)
    assert len(w) == 3
    assert np.isclose(w.sum(), 1.0)
    assert (w >= 0).all()
    # Member 2 (zeros) should receive near-zero weight
    assert w[2] < 0.1


def test_shrinkage_calibration():
    from src.calibration import fit_shrinkage_power, apply_shrinkage

    y_true = np.array([2.0, 1.5, 0.0, 0.0, 0.0])
    y_pred = np.array([2.1, 1.4, 0.8, 0.7, 0.9])
    p_active = np.array([0.95, 0.90, 0.10, 0.05, 0.15])

    power = fit_shrinkage_power(y_true, y_pred, p_active)
    assert power > 0.0  # Should learn positive shrinkage to penalize inactive false positives

    shrunk = apply_shrinkage(y_pred, p_active, power)
    init_rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    shrunk_rmse = np.sqrt(np.mean((y_true - shrunk) ** 2))
    assert shrunk_rmse < init_rmse


def test_dual_shrinkage_calibration():
    from src.calibration import fit_dual_shrinkage_powers, apply_dual_shrinkage

    y_true = np.array([2.0, 1.5, 0.0, 0.0, 0.0])
    y_pred = np.array([2.1, 1.4, 0.8, 0.7, 0.9])
    p_active = np.array([0.95, 0.90, 0.10, 0.05, 0.15])
    p_store = np.array([0.90, 0.85, 0.15, 0.10, 0.20])

    alpha, beta = fit_dual_shrinkage_powers(y_true, y_pred, p_active, p_store)
    assert alpha >= 0.0
    assert beta >= 0.0
    assert (alpha + beta) > 0.0

    shrunk = apply_dual_shrinkage(y_pred, p_active, p_store, alpha, beta)
    init_rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    shrunk_rmse = np.sqrt(np.mean((y_true - shrunk) ** 2))
    assert shrunk_rmse < init_rmse

    # Anchored alpha
    a_anchored, b_anchored = fit_dual_shrinkage_powers(
        y_true, y_pred, p_active, p_store, base_active_power=0.15
    )
    assert np.isclose(a_anchored, 0.15)
    assert b_anchored >= 0.0



def test_store_attachment_classifier():
    import pandas as pd
    from src.models import StoreAttachmentClassifier

    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(100, 5)), columns=[f"f{i}" for i in range(5)])
    # Continuous nonfuel labels with zeros
    y = np.clip(rng.normal(loc=0.0, scale=1.0, size=100), 0.0, None)

    clf = StoreAttachmentClassifier(seed=42, n_jobs=1)
    clf.fit(X, y)
    probs = clf.predict_proba(X)

    assert probs.shape == (100,)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_inactivity_gate():
    from src.calibration import apply_inactivity_gate

    y_pred = np.array([2.0, 1.5, 0.8, 0.4])
    p_active = np.array([0.9, 0.5, 0.04, 0.01])

    gated = apply_inactivity_gate(y_pred, p_active, threshold=0.05)
    assert np.isclose(gated[0], 2.0)
    assert np.isclose(gated[1], 1.5)
    assert gated[2] == 0.0
    assert gated[3] == 0.0


def test_fit_class_weights_with_dominant_classes():
    from src.calibration import fit_class_weights, decide, weighted_f1

    rng = np.random.default_rng(42)
    n_samples, n_classes = 400, 5
    probs = rng.dirichlet(np.ones(n_classes), size=n_samples)
    y_true = rng.integers(0, n_classes, size=n_samples)

    weights = fit_class_weights(probs, y_true, dominant_classes=[0, 1])
    assert len(weights) == n_classes
    assert (weights > 0).all()
    # Should improve or maintain F1
    base_f1 = weighted_f1(y_true, decide(probs), n_classes)
    cal_f1 = weighted_f1(y_true, decide(probs, weights), n_classes)
    assert cal_f1 >= base_f1 - 1e-9



