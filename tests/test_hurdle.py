"""Unit tests for Two-Part Hurdle models and Multi-Seed Bagged Hurdle models."""
import numpy as np
import pandas as pd
import pytest

from src.hurdle import (
    TwoPartFuelHurdle,
    TwoPartNonfuelHurdle,
    BaggedTwoPartFuelHurdle,
    BaggedTwoPartNonfuelHurdle,
)


@pytest.fixture
def synthetic_data():
    np.random.seed(42)
    n = 60
    X = pd.DataFrame({
        "feat1": np.random.randn(n),
        "feat2": np.random.randn(n) * 2.0,
        "txns_90d": np.random.randint(0, 10, n),
        "fuel_litres_90d": np.random.uniform(0, 150, n),
    })
    # Zero-inflated target (50% zeros)
    is_active = (X["feat1"] > 0).astype(int)
    amount = np.maximum(0.0, X["feat2"] + 2.0)
    y = (is_active * amount).to_numpy()
    return X, y


def test_two_part_hurdle_models(synthetic_data):
    X, y = synthetic_data
    h_fuel = TwoPartFuelHurdle(seed=42, n_jobs=1)
    h_fuel.fit(X, y)
    preds_f = h_fuel.predict(X)

    assert len(preds_f) == len(X)
    assert not np.isnan(preds_f).any()
    assert (preds_f >= 0.0).all()

    h_nf = TwoPartNonfuelHurdle(seed=42, n_jobs=1)
    h_nf.fit(X, y)
    preds_nf = h_nf.predict(X)

    assert len(preds_nf) == len(X)
    assert not np.isnan(preds_nf).any()
    assert (preds_nf >= 0.0).all()


def test_bagged_two_part_hurdle_models(synthetic_data):
    X, y = synthetic_data
    # Fast test with 2 seeds
    seeds = (42, 100)

    bag_fuel = BaggedTwoPartFuelHurdle(seeds=seeds, n_jobs=1)
    # Fast parameters
    for reg in bag_fuel.regressors_lgb:
        reg.set_params(n_estimators=10)
    for reg in bag_fuel.regressors_cb:
        reg.set_params(iterations=10)
    bag_fuel.fit(X, y)
    preds_f = bag_fuel.predict(X)

    assert len(preds_f) == len(X)
    assert not np.isnan(preds_f).any()
    assert (preds_f >= 0.0).all()

    bag_nf = BaggedTwoPartNonfuelHurdle(seeds=seeds, n_jobs=1)
    for reg in bag_nf.regressors_lgb:
        reg.set_params(n_estimators=10)
    for reg in bag_nf.regressors_cb:
        reg.set_params(iterations=10)
    bag_nf.fit(X, y)
    preds_nf = bag_nf.predict(X)

    assert len(preds_nf) == len(X)
    assert not np.isnan(preds_nf).any()
    assert (preds_nf >= 0.0).all()


def test_physical_basket_ticket_regressor():
    from src.hurdle import PhysicalBasketTicketRegressor

    rng = np.random.default_rng(42)
    n = 80
    X = pd.DataFrame({
        "m3_baskets": rng.poisson(3, size=n),
        "days_since_last_purchase": rng.uniform(1, 90, size=n),
        "f1": rng.normal(size=n),
    })
    # Target: log1p(litres / 10)
    litres = rng.exponential(scale=50, size=n) * (X["m3_baskets"] > 0)
    y = np.log1p(litres / 10.0)

    model = PhysicalBasketTicketRegressor(seed=42, n_jobs=1)
    model.basket_regressor.set_params(n_estimators=10)
    model.ticket_regressor.set_params(n_estimators=10)
    model.fit(X, y)
    preds = model.predict(X)

    assert len(preds) == n
    assert not np.isnan(preds).any()
    assert (preds >= 0.0).all()


def test_multi_quantile_two_part_hurdle(synthetic_data):
    from src.hurdle import MultiQuantileTwoPartHurdle

    X, y = synthetic_data
    mq = MultiQuantileTwoPartHurdle(seed=42, n_jobs=1)
    for q in mq.lgb_quantiles:
        q.set_params(n_estimators=10)
    for q in mq.cb_quantiles:
        q.set_params(iterations=10)
    mq.l2_regressor.set_params(n_estimators=10)
    mq.fit(X, y)
    preds = mq.predict(X)

    assert len(preds) == len(X)
    assert not np.isnan(preds).any()
    assert (preds >= 0.0).all()


