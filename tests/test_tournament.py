import numpy as np
import pandas as pd
import pytest

from src.config import PipelineConfig
from src.tournament import TournamentEngine


def test_fold_definitions():
    cfg = PipelineConfig()
    engine = TournamentEngine(cfg, builder=None)
    folds = engine._get_fold_definitions()
    assert len(folds) == 4
    names = [f["name"] for f in folds]
    assert "Winter_2025-06-01" in names
    assert "MidWinter_2025-07-01" in names
    assert "LateWinter_2025-08-01" in names
    assert "Spring_2025-09-01" in names

    # All folds must strictly obey causal forward embargo (training cutoffs <= eval - 3 months)
    for f in folds:
        eval_t = f["eval_cutoff"]
        for c in f["train_cutoffs"]:
            assert c <= eval_t - pd.DateOffset(months=3)


def test_build_level2_structural_features():
    from src.tournament import build_level2_structural_features
    n = 10
    X = pd.DataFrame({
        "m3_fuel_litres": np.full(n, 10.0),
        "m3_fuel_rand": np.full(n, 235.0),
        "active_m6_fuel": np.ones(n),
        "survival_prob_empirical": np.full(n, 0.8),
        "fuel_survival_prob": np.full(n, 0.7),
        "m3spend_beverages": np.full(n, 50.0),
        "m1spend_beverages": np.full(n, 20.0),
    })
    probs_base = np.full((n, 8), 0.125)
    probs_hier = np.full((n, 8), 0.125)
    pred_fuel = np.full(n, 1.2)
    pred_nf = np.full(n, 0.8)
    p_fuel_active = np.full(n, 0.9)
    p_store_active = np.full(n, 0.6)
    mu_fuel_pos = np.full(n, 1.5)
    sigma_fuel_pos = np.full(n, 0.3)

    level2_X = build_level2_structural_features(
        X, probs_base, probs_hier,
        pred_fuel, pred_nf, p_fuel_active, p_store_active,
        mu_fuel_pos, sigma_fuel_pos,
    )
    assert isinstance(level2_X, pd.DataFrame)
    assert "fuel_hurdle_ratio" in level2_X.columns
    assert "p_exceed_fuel" in level2_X.columns
    assert "joint_hurdle_inactivity" in level2_X.columns
    assert "fuel_monetary_increment" in level2_X.columns
    assert "fuel_vs_nf_monetary_margin" in level2_X.columns
    assert not level2_X.isna().any().any()

