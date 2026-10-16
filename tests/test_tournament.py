import numpy as np
import pandas as pd
import pytest

from src.config import PipelineConfig
from src.tournament import (
    TournamentEngine,
    TournamentResult,
    apply_boundary_arbitration,
)


def test_fold_definitions():
    cfg = PipelineConfig()
    engine = TournamentEngine(cfg, builder=None)
    folds = engine._get_fold_definitions()
    assert len(folds) == 3
    names = [f["name"] for f in folds]
    assert "Summer_2024-12-01" in names
    assert "Winter_2025-06-01" in names
    assert "Spring_2025-09-01" in names

    # Fold 1 (Summer) weight must be 2.0
    f1 = [f for f in folds if "Summer" in f["name"]][0]
    assert f1["weight"] == 2.0
    # No outcome overlap with summer outcome window [2024-12-01, 2025-03-01)
    for c in f1["train_cutoffs"]:
        assert not (pd.Timestamp("2024-10-01") <= c < pd.Timestamp("2025-03-01"))


def test_boundary_arbitration():
    opp_labels = ["Inactivity", "Stable", "Inactivity"]
    features = pd.DataFrame({
        "m1_baskets": [2, 0, 0],
        "days_since_last_purchase": [10.0, 60.0, 90.0],
        "m3_fuel_litres": [10.0, 0.0, 0.0],
    })
    pred_fuel = np.array([1.5, 0.0, 0.0])

    arbitrated = apply_boundary_arbitration(opp_labels, features, pred_fuel)
    # Row 0: was Inactivity but has m1_baskets > 0 and days_last < 45 -> should be recovered
    assert arbitrated[0] in ("Existing-category growth: Fuel", "Stable")
    assert arbitrated[1] == "Stable"
    assert arbitrated[2] == "Inactivity"
