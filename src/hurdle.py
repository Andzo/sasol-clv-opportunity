"""Two-part hurdle model for zero-inflated target prediction.

Deconstructs zero-inflated customer spend into:
1. P(Y > 0 | X): binary store attachment classification.
2. E[Y | X, Y > 0]: conditional spend regression trained exclusively on positive instances.

The expected unconditional target is E[Y | X] = P(Y > 0 | X) * E[Y | X, Y > 0].
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.models import StoreAttachmentClassifier, make_regressor


class TwoPartNonfuelHurdle:
    """Two-part hurdle model for nonfuel customer lifetime value."""

    def __init__(self, seed: int = 42, n_jobs: int = -1, use_gpu: bool = False) -> None:
        self.seed = seed
        self.n_jobs = n_jobs
        self.use_gpu = use_gpu
        self.classifier = StoreAttachmentClassifier(seed=seed, n_jobs=n_jobs, use_gpu=use_gpu)
        self.regressor = make_regressor("lgbm_huber", seed=seed, n_jobs=n_jobs, use_gpu=use_gpu)

    def fit(self, X: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "TwoPartNonfuelHurdle":
        y_arr = np.asarray(y, dtype="float64")
        pos_mask = y_arr > 0
        self.classifier.fit(X, y_arr)
        X_pos = X.iloc[pos_mask] if isinstance(X, pd.DataFrame) else X[pos_mask]
        y_pos = y_arr[pos_mask]
        if sample_weight is not None:
            sw_pos = np.asarray(sample_weight, dtype="float64")[pos_mask]
            self.regressor.fit(X_pos, y_pos, sample_weight=sw_pos)
        else:
            self.regressor.fit(X_pos, y_pos)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        p_store = self.classifier.predict_proba(X)
        pred_amount = np.clip(self.regressor.predict(X), 0.0, None)
        return p_store * pred_amount
