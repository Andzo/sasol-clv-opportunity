"""Unit tests for src/backtest.py module."""
import numpy as np
import pandas as pd
from src.backtest import paired_bootstrap_test

def test_paired_bootstrap_test():
    n = 100
    rng = np.random.default_rng(42)
    y_f = np.ones(n)
    y_nf = np.ones(n)
    y_opp = np.array(["Stable"] * n)

    # Baseline predictions (exact match)
    df_base = pd.DataFrame({
        "pred_fuel": y_f,
        "pred_nonfuel": y_nf,
        "pred_opp": y_opp,
        "true_fuel": y_f,
        "true_nonfuel": y_nf,
        "true_opp": y_opp,
    })

    # Candidate predictions (slightly noisy)
    df_cand = pd.DataFrame({
        "pred_fuel": y_f + rng.normal(0, 0.2, size=n),
        "pred_nonfuel": y_nf + rng.normal(0, 0.2, size=n),
        "pred_opp": y_opp,
        "true_fuel": y_f,
        "true_nonfuel": y_nf,
        "true_opp": y_opp,
    })

    res = paired_bootstrap_test(df_base, df_cand, n_replicates=50, seed=42)
    assert "mean_delta" in res
    assert "ci_lower" in res
    assert "ci_upper" in res
    assert "p_value" in res
    # Candidate is worse than baseline, so delta should be negative
    assert res["mean_delta"] < 0.0


def test_evaluate_leave_summer_out_import():
    from src.backtest import evaluate_leave_summer_out
    assert callable(evaluate_leave_summer_out)

