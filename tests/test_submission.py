"""Unit tests for submission validation and generation."""
import numpy as np
import pandas as pd
import pytest
from src.submission import build_submission, validate_submission


def test_build_and_validate_submission():
    sample = pd.DataFrame({
        "ID": ["1", "2", "3"],
        "CLV_fuel": [0, 0, 0],
        "CLV_nonfuel": [0, 0, 0],
        "Opportunity": ["Stable", "Stable", "Stable"],
    })
    allowed = ["Stable", "Inactivity", "Existing-category growth: Fuel"]

    sub = build_submission(
        ids=["1", "2", "3"],
        fuel=np.array([1.2, 0.5, 0.0]),
        nonfuel=np.array([0.0, 1.0, 2.0]),
        opportunity=["Stable", "Inactivity", "Existing-category growth: Fuel"]
    )

    # Should pass without error
    validate_submission(sub, sample, allowed)

    # Negative values should fail
    sub_neg = sub.copy()
    sub_neg.loc[0, "CLV_fuel"] = -0.1
    with pytest.raises(ValueError, match="contains negative values"):
        validate_submission(sub_neg, sample, allowed)

    # Invalid opportunity label should fail
    sub_bad_label = sub.copy()
    sub_bad_label.loc[0, "Opportunity"] = "Unknown Label"
    with pytest.raises(ValueError, match="Invalid Opportunity labels"):
        validate_submission(sub_bad_label, sample, allowed)

    # Missing ID should fail
    sub_wrong_ids = sub.copy()
    sub_wrong_ids.loc[0, "ID"] = "999"
    with pytest.raises(ValueError, match="Submission IDs differ"):
        validate_submission(sub_wrong_ids, sample, allowed)
