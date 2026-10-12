"""Gradient-boosting model factories and ensembles.

Fixes vs. the legacy implementation:
* Class probabilities are scattered into the full label space using the classes
  actually seen in training (legacy right-padding silently shifted labels when a
  rare class was absent from a training fold).
* LightGBM bagging is actually enabled (``subsample`` is ignored unless
  ``subsample_freq > 0``).
* CatBoost no longer writes ``catboost_info/`` into the working directory.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor


def make_regressor(name: str, seed: int, n_jobs: int) -> Any:
    """Squared-error regressor (the metric is RMSE on the transformed target)."""
    if name == "lgbm":
        return LGBMRegressor(
            n_estimators=650, learning_rate=0.018, num_leaves=28, min_child_samples=55,
            subsample=0.75, subsample_freq=1, colsample_bytree=0.50, reg_lambda=12.0,
            random_state=seed, n_jobs=n_jobs, deterministic=True, force_col_wise=True, verbose=-1,
        )
    if name == "catboost":
        return CatBoostRegressor(
            iterations=550, learning_rate=0.035, depth=5, l2_leaf_reg=10.0, loss_function="RMSE",
            random_seed=seed, thread_count=n_jobs, verbose=0, allow_writing_files=False,
        )
    if name == "xgb":
        return XGBRegressor(
            n_estimators=550, learning_rate=0.018, max_depth=4, min_child_weight=6.5,
            subsample=0.75, colsample_bytree=0.50, reg_lambda=14.0, tree_method="hist",
            random_state=seed, n_jobs=n_jobs,
        )
    raise ValueError(f"Unknown regressor {name!r}")


def make_classifier(name: str, seed: int, n_jobs: int) -> Any:
    """Multiclass probabilistic classifier."""
    if name == "lgbm":
        return LGBMClassifier(
            objective="multiclass", n_estimators=450, learning_rate=0.018, num_leaves=30,
            min_child_samples=51, subsample=0.77, subsample_freq=1, colsample_bytree=0.58,
            reg_lambda=10.0, random_state=seed, n_jobs=n_jobs, deterministic=True,
            force_col_wise=True, verbose=-1,
        )
    if name == "catboost":
        return CatBoostClassifier(
            iterations=500, learning_rate=0.045, depth=6, l2_leaf_reg=6.0, loss_function="MultiClass",
            random_seed=seed, thread_count=n_jobs, verbose=0, allow_writing_files=False,
        )
    if name == "xgb":
        return XGBClassifier(
            objective="multi:softprob", n_estimators=350, learning_rate=0.031, max_depth=5,
            min_child_weight=4.74, subsample=0.75, colsample_bytree=0.68, reg_lambda=9.2,
            tree_method="hist", random_state=seed, n_jobs=n_jobs,
        )
    raise ValueError(f"Unknown classifier {name!r}")


class RegressorEnsemble:
    """Weighted average of non-negative regressor predictions."""

    def __init__(self, members: Sequence[str], seed: int = 42, n_jobs: int = -1) -> None:
        self.members = tuple(members)
        self.models = [make_regressor(m, seed, n_jobs) for m in self.members]
        self.weights = np.full(len(self.members), 1.0 / len(self.members))

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "RegressorEnsemble":
        for model in self.models:
            model.fit(X, y)
        return self

    def predict_members(self, X: pd.DataFrame) -> np.ndarray:
        """(n_samples, n_members) matrix of clipped member predictions."""
        return np.column_stack([np.clip(m.predict(X), 0.0, None) for m in self.models])

    def set_weights(self, weights: np.ndarray) -> None:
        weights = np.asarray(weights, dtype="float64")
        if weights.shape != (len(self.members),) or weights.sum() <= 0:
            raise ValueError("Blend weights must be non-negative, one per member, with positive sum")
        self.weights = weights / weights.sum()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict_members(X) @ self.weights


class ClassifierEnsemble:
    """Probability-averaging ensemble over a fixed label space ``0..n_classes-1``."""

    def __init__(
        self,
        members: Sequence[str],
        n_classes: int,
        seed: int = 42,
        n_jobs: int = -1,
        weights: Sequence[float] | None = None,
    ) -> None:
        self.members = tuple(members)
        self.n_classes = n_classes
        self.models = [make_classifier(m, seed, n_jobs) for m in self.members]
        self.classes_: np.ndarray | None = None
        if weights is not None:
            self.set_weights(weights)
        else:
            self.weights = np.full(len(self.members), 1.0 / len(self.members))

    def set_weights(self, weights: Sequence[float]) -> None:
        w = np.asarray(weights, dtype="float64")
        if w.shape != (len(self.members),) or w.sum() <= 0 or (w < 0).any():
            raise ValueError("Classifier weights must be non-negative, one per member, with positive sum")
        self.weights = w / w.sum()

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "ClassifierEnsemble":
        y = np.asarray(y, dtype=int)
        if y.min() < 0 or y.max() >= self.n_classes:
            raise ValueError("Class indices out of range")
        self.classes_ = np.unique(y)
        encoded = np.searchsorted(self.classes_, y)  # contiguous 0..P-1 (required by XGBoost)
        for model in self.models:
            model.fit(X, encoded)
        return self

    def predict_member_proba(self, X: pd.DataFrame) -> list[np.ndarray]:
        """Per-member probabilities aligned to the full label space."""
        if self.classes_ is None:
            raise RuntimeError("Ensemble is not fitted")
        out = []
        for model in self.models:
            p = np.asarray(model.predict_proba(X), dtype="float64")
            full = np.zeros((len(X), self.n_classes))
            full[:, self.classes_] = p
            out.append(full)
        return out

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        member_probs = self.predict_member_proba(X)
        probs = np.zeros_like(member_probs[0])
        for w, p in zip(self.weights, member_probs):
            probs += w * p
        return probs / probs.sum(axis=1, keepdims=True)


def lgbm_importance(model: Any, feature_names: Sequence[str]) -> pd.Series:
    """Gain importance of a fitted LightGBM model (empty if not LightGBM)."""
    booster = getattr(model, "booster_", None)
    if booster is None:
        return pd.Series(dtype="float64")
    return pd.Series(booster.feature_importance("gain"), index=list(feature_names)).sort_values(ascending=False)


class StoreAttachmentClassifier:
    """Predicts binary convenience store visitation P(CLV_nonfuel > 0 | X) using a diverse GBDT ensemble."""

    def __init__(self, seed: int = 42, n_jobs: int = -1) -> None:
        self.lgbm = LGBMClassifier(
            objective="binary",
            n_estimators=400,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.6,
            reg_lambda=5.0,
            random_state=seed,
            n_jobs=n_jobs,
            deterministic=True,
            force_col_wise=True,
            verbose=-1,
        )
        self.catboost = CatBoostClassifier(
            iterations=400,
            learning_rate=0.04,
            depth=6,
            loss_function="Logloss",
            random_seed=seed,
            thread_count=n_jobs,
            verbose=0,
            allow_writing_files=False,
        )
        self.xgb = XGBClassifier(
            objective="binary:logistic",
            n_estimators=350,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.6,
            tree_method="hist",
            random_state=seed,
            n_jobs=n_jobs,
        )

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "StoreAttachmentClassifier":
        y_binary = (np.asarray(y) > 0).astype(int)
        self.lgbm.fit(X, y_binary)
        self.catboost.fit(X, y_binary)
        self.xgb.fit(X, y_binary)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p_lgbm = np.asarray(self.lgbm.predict_proba(X)[:, 1], dtype="float64")
        p_cb = np.asarray(self.catboost.predict_proba(X)[:, 1], dtype="float64")
        p_xgb = np.asarray(self.xgb.predict_proba(X)[:, 1], dtype="float64")
        return (p_lgbm + p_cb + p_xgb) / 3.0

