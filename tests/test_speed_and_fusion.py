"""Tests for fast-mode config, boosting schedule scaling, probability stacker and shared fusion path."""
import numpy as np

from src.config import ModelConfig, PipelineConfig
from src.models import GB_SPEED, _gb
from src.tournament import (
    ProbStacker,
    build_prob_stack_features,
    fuse_opportunity_probabilities,
)


def test_fast_model_config_is_single_seed():
    m = ModelConfig.fast(seed=7)
    assert m.n_seeds_reg == 1 and m.n_seeds_clf == 1 and m.hurdle_seeds == (42,) and m.seed == 7


def test_gb_schedule_preserves_total_shrinkage():
    p = _gb(650, 0.018)
    assert p["n_estimators"] < 650
    assert abs(p["n_estimators"] * p["learning_rate"] - 650 * 0.018) / (650 * 0.018) < 0.02
    assert GB_SPEED > 1.0


def test_default_tournament_cutoffs_end_with_holdout():
    cfg = PipelineConfig()
    assert cfg.tournament_cutoffs[-1] == cfg.evaluation_cutoff
    assert "lgbm_mse" not in cfg.models.regression_members  # pruned dead members


def test_prob_stacker_returns_full_label_space():
    rng = np.random.default_rng(0)
    n, k = 300, 6
    y = rng.integers(0, 4, size=n)  # classes 4,5 never observed
    logits = rng.normal(size=(n, k))
    logits[np.arange(n), y] += 2.0
    probs = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    X = build_prob_stack_features(probs, probs, probs, rng.random(n), rng.random(n))
    st = ProbStacker(k).fit(X, y)
    out = st.predict_proba(X)
    assert out.shape == (n, k)
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-6)
    assert (out[:, 4:] == 0).all()
    assert (out.argmax(1) == y).mean() > 0.6


def test_fusion_blend_matches_legacy_formula():
    from types import SimpleNamespace
    n, k = 5, 4
    rng = np.random.default_rng(1)
    base = rng.dirichlet(np.ones(k), n)
    hier = rng.dirichlet(np.ones(k), n)
    meta = rng.dirichlet(np.ones(k), n)
    pf, pi = rng.random(n), rng.random(n)
    cal = SimpleNamespace(prob_stacker=None, prob_blend_weights=(0.5, 0.5, 0.0, 0.1, 0.2))
    got = fuse_opportunity_probabilities(cal, base.copy(), hier.copy(), meta, pf, pi, fg_idx=1, inact_idx=0)
    fused = 0.5 * base + 0.5 * hier
    fused[:, 1] += 0.1 * pf
    fused[:, 0] += 0.2 * pi
    np.testing.assert_allclose(got, 0.65 * meta + 0.35 * fused)


def test_select_top_classification_features():
    import pandas as pd
    from src.tournament import select_top_classification_features
    rng = np.random.default_rng(42)
    n = 100
    data = {
        "m3_fuel_litres": rng.uniform(0, 50, n),
        "days_since_last_purchase": rng.uniform(1, 90, n),
        "z_anchor_fuel_mean": rng.normal(0, 1, n),
        "cat_cluster": ["A" if i % 2 == 0 else "B" for i in range(n)],
    }
    for i in range(25):
        data[f"noise_feat_{i}"] = rng.normal(0, 1, n)
    X = pd.DataFrame(data)
    y = rng.choice([4, 16, 2, 0, 1], size=n)

    selected = select_top_classification_features(X, y, top_k=10, seed=42)
    # Essential features must be kept
    assert "m3_fuel_litres" in selected
    assert "days_since_last_purchase" in selected
    assert "z_anchor_fuel_mean" in selected
    # Categorical features must be kept
    assert "cat_cluster" in selected
    # Size bounded
    assert len(selected) <= 10 + 1  # 10 numeric + 1 cat

