"""Unit tests for model factories, ClassifierEnsemble, and StoreAttachmentClassifier."""
import numpy as np
import pandas as pd
import pytest

from src.models import (
    ClassifierEnsemble,
    HierarchicalOpportunityClassifier,
    RegressorEnsemble,
    StoreAttachmentClassifier,
    make_classifier,
    make_regressor,
)


def test_classifier_ensemble_weighting_and_simplex():
    rng = np.random.default_rng(42)
    n_samples, n_features, n_classes = 60, 4, 3
    X = pd.DataFrame(rng.normal(size=(n_samples, n_features)), columns=[f"f{i}" for i in range(n_features)])
    y = rng.integers(0, n_classes, size=n_samples)

    # LGBM + XGB ensemble with non-uniform weights
    ens = ClassifierEnsemble(
        members=["lgbm", "xgb"],
        n_classes=n_classes,
        seed=42,
        n_jobs=1,
        weights=[0.3, 0.7],
    )
    assert np.isclose(ens.weights.sum(), 1.0)
    assert np.isclose(ens.weights[0], 0.3)
    assert np.isclose(ens.weights[1], 0.7)

    ens.fit(X, y)
    probs = ens.predict_proba(X)
    assert probs.shape == (n_samples, n_classes)
    assert (probs >= 0.0).all()
    assert np.allclose(probs.sum(axis=1), 1.0)

    # Test set_weights dynamically
    ens.set_weights([0.6, 0.4])
    assert np.isclose(ens.weights[0], 0.6)
    probs2 = ens.predict_proba(X)
    assert np.allclose(probs2.sum(axis=1), 1.0)


def test_store_attachment_ensemble_probabilities():
    rng = np.random.default_rng(42)
    n_samples, n_features = 60, 4
    X = pd.DataFrame(rng.normal(size=(n_samples, n_features)), columns=[f"f{i}" for i in range(n_features)])
    y = rng.choice([0.0, 150.0], size=n_samples)

    store_clf = StoreAttachmentClassifier(seed=42, n_jobs=1)
    store_clf.fit(X, y)
    p_store = store_clf.predict_proba(X)

    assert p_store.shape == (n_samples,)
    assert (p_store >= 0.0).all()
    assert (p_store <= 1.0).all()


def test_regressor_ensemble_blending():
    rng = np.random.default_rng(42)
    n_samples, n_features = 60, 4
    X = pd.DataFrame(rng.normal(size=(n_samples, n_features)), columns=[f"f{i}" for i in range(n_features)])
    y = np.clip(rng.normal(loc=1.5, scale=0.5, size=n_samples), 0.0, None)

    ens = RegressorEnsemble(members=["lgbm", "xgb"], seed=42, n_jobs=1)
    ens.fit(X, y)
    ens.set_weights([0.2, 0.8])
    preds = ens.predict(X)

    assert preds.shape == (n_samples,)
    assert (preds >= 0.0).all()


def test_hierarchical_opportunity_classifier():
    rng = np.random.default_rng(42)
    n_samples, n_features, n_classes = 80, 5, 5
    X = pd.DataFrame(rng.normal(size=(n_samples, n_features)), columns=[f"f{i}" for i in range(n_features)])
    X["m3_baskets"] = rng.integers(0, 5, size=n_samples)
    y = rng.integers(0, n_classes, size=n_samples)
    # Ensure all classes are represented
    y[:n_classes] = np.arange(n_classes)

    clf = HierarchicalOpportunityClassifier(
        n_classes=n_classes,
        seed=42,
        n_jobs=1,
        inact_idx=0,
        stable_idx=1,
        fg_idx=2,
    )
    clf.fit(X, y)
    probs = clf.predict_proba(X)

    assert probs.shape == (n_samples, n_classes)
    assert (probs >= 0.0).all()
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_make_regressor_diverse_members():
    rng = np.random.default_rng(42)
    n_samples, n_features = 40, 4
    X = pd.DataFrame(rng.normal(size=(n_samples, n_features)), columns=[f"f{i}" for i in range(n_features)])
    y = np.clip(rng.normal(loc=1.5, scale=0.5, size=n_samples), 0.0, None)

    for member in ["ridge", "extratrees"]:
        reg = make_regressor(member, seed=42, n_jobs=1)
        reg.fit(X, y)
        preds = reg.predict(X)
        assert preds.shape == (n_samples,)


def test_models_with_gpu_flag():
    rng = np.random.default_rng(42)
    n_samples, n_features = 40, 4
    X = pd.DataFrame(rng.normal(size=(n_samples, n_features)), columns=[f"f{i}" for i in range(n_features)])
    y = np.clip(rng.normal(loc=1.5, scale=0.5, size=n_samples), 0.0, None)

    # Verify that make_regressor accepts use_gpu
    reg = make_regressor("xgb", seed=42, n_jobs=1, use_gpu=True)
    reg.fit(X, y)
    preds = reg.predict(X)
    assert preds.shape == (n_samples,)

