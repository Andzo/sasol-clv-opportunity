"""Challenge metrics and naive reference baselines.

Zindi normalises each RMSE as ``max(0, 1 - RMSE / x_max)`` with ``x_max`` taken from
a starter submission that is not public. We therefore report raw metrics as the
primary signal and a *proxy* composite using the all-zero ``SampleSubmission`` RMSE
as ``x_max``. Lower RMSE / higher F1 strictly improves the true leaderboard score
regardless of the exact ``x_max``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

WEIGHTS = {"CLV_fuel": 0.3, "CLV_nonfuel": 0.3, "Opportunity": 0.4}


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def normalized_rmse(value: float, x_max: float) -> float:
    """Zindi-style normalisation to [0, 1], higher is better."""
    return max(0.0, 1.0 - value / x_max)


@dataclass(frozen=True)
class Scores:
    rmse_fuel: float
    rmse_nonfuel: float
    weighted_f1: float
    macro_f1: float = 0.0

    def composite(self, x_max_fuel: float, x_max_nonfuel: float) -> float:
        return (
            WEIGHTS["CLV_fuel"] * normalized_rmse(self.rmse_fuel, x_max_fuel)
            + WEIGHTS["CLV_nonfuel"] * normalized_rmse(self.rmse_nonfuel, x_max_nonfuel)
            + WEIGHTS["Opportunity"] * self.weighted_f1
        )

    def leaderboard_composite(self, x_max_fuel: float = 0.740, x_max_nonfuel: float = 0.816) -> float:
        """Zindi composite score normalized by competition reference scales."""
        return self.composite(x_max_fuel, x_max_nonfuel)

    def leaderboard_breakdown(self, x_max_fuel: float = 0.740, x_max_nonfuel: float = 0.816) -> dict[str, float]:
        """Component score contributions summing to leaderboard_composite."""
        s_fuel = WEIGHTS["CLV_fuel"] * normalized_rmse(self.rmse_fuel, x_max_fuel)
        s_nonfuel = WEIGHTS["CLV_nonfuel"] * normalized_rmse(self.rmse_nonfuel, x_max_nonfuel)
        s_opp = WEIGHTS["Opportunity"] * self.weighted_f1
        return {
            "score_fuel": float(s_fuel),
            "score_nonfuel": float(s_nonfuel),
            "score_opportunity": float(s_opp),
            "total_score": float(s_fuel + s_nonfuel + s_opp),
        }

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def score(labels: pd.DataFrame, fuel: np.ndarray, nonfuel: np.ndarray, opportunity: np.ndarray) -> Scores:
    """Score predictions (string Opportunity labels) against a labels frame."""
    return Scores(
        rmse_fuel=rmse(labels["CLV_fuel"], fuel),
        rmse_nonfuel=rmse(labels["CLV_nonfuel"], nonfuel),
        weighted_f1=float(f1_score(labels["Opportunity"], opportunity, average="weighted", zero_division=0)),
        macro_f1=float(f1_score(labels["Opportunity"], opportunity, average="macro", zero_division=0)),
    )


def baseline_scores(features: pd.DataFrame, labels: pd.DataFrame, label_config: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    """Reference points: all-zero sample submission, persistence of last quarter, majority class."""
    s_f = label_config["normalization"]["CLV_fuel"]
    s_n = label_config["normalization"]["CLV_nonfuel"]
    n = len(labels)
    majority = labels["Opportunity"].value_counts().idxmax()
    zero = score(labels, np.zeros(n), np.zeros(n), np.full(n, "Stable"))
    persistence = score(
        labels,
        np.log1p(features["m3_fuel_litres"].clip(lower=0).to_numpy()) / s_f,
        np.log1p(features["m3_nonfuel_rand"].clip(lower=0).to_numpy()) / s_n,
        np.full(n, majority),
    )
    return {"sample_submission_zero": zero.as_dict(), "persistence_last_quarter": persistence.as_dict(),
            "majority_class": majority}
