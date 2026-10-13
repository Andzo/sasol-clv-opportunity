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


def make_regressor(name: str, seed: int, n_jobs: int, tweedie_power: float = 1.15, use_gpu: bool = False) -> Any:
    """Compound Poisson-Gamma (Tweedie), Huber, and squared-error regressors."""
    if name == "lgbm":
        return LGBMRegressor(
            objective="tweedie", tweedie_variance_power=tweedie_power,
            n_estimators=650, learning_rate=0.018, num_leaves=28, min_child_samples=55,
            subsample=0.75, subsample_freq=1, colsample_bytree=0.50, reg_lambda=12.0,
            random_state=seed, n_jobs=n_jobs, deterministic=True, force_col_wise=True, verbose=-1,
        )
    if name == "lgbm_huber":
        return LGBMRegressor(
            objective="huber", alpha=0.90,
            n_estimators=650, learning_rate=0.018, num_leaves=28, min_child_samples=55,
            subsample=0.75, subsample_freq=1, colsample_bytree=0.50, reg_lambda=12.0,
            random_state=seed, n_jobs=n_jobs, deterministic=True, force_col_wise=True, verbose=-1,
        )
    if name == "lgbm_mse":
        return LGBMRegressor(
            objective="regression",
            n_estimators=650, learning_rate=0.018, num_leaves=28, min_child_samples=55,
            subsample=0.75, subsample_freq=1, colsample_bytree=0.50, reg_lambda=12.0,
            random_state=seed, n_jobs=n_jobs, deterministic=True, force_col_wise=True, verbose=-1,
        )
    if name == "catboost":
        cb_kwargs: dict[str, Any] = {
            "iterations": 550, "learning_rate": 0.035, "depth": 5, "l2_leaf_reg": 10.0, "loss_function": "RMSE",
            "random_seed": seed, "verbose": 0, "allow_writing_files": False,
        }
        if use_gpu:
            cb_kwargs["task_type"] = "GPU"
        else:
            cb_kwargs["thread_count"] = n_jobs
        return CatBoostRegressor(**cb_kwargs)
    if name == "xgb":
        xgb_kwargs: dict[str, Any] = {
            "objective": "reg:tweedie", "tweedie_variance_power": tweedie_power,
            "n_estimators": 550, "learning_rate": 0.018, "max_depth": 4, "min_child_weight": 6.5,
            "subsample": 0.75, "colsample_bytree": 0.50, "reg_lambda": 14.0, "tree_method": "hist",
            "random_state": seed, "n_jobs": n_jobs,
        }
        if use_gpu:
            xgb_kwargs["device"] = "cuda"
        return XGBRegressor(**xgb_kwargs)
    if name == "xgb_huber":
        xgb_kwargs: dict[str, Any] = {
            "objective": "reg:pseudohubererror",
            "n_estimators": 550, "learning_rate": 0.018, "max_depth": 4, "min_child_weight": 6.5,
            "subsample": 0.75, "colsample_bytree": 0.50, "reg_lambda": 14.0, "tree_method": "hist",
            "random_state": seed, "n_jobs": n_jobs,
        }
        if use_gpu:
            xgb_kwargs["device"] = "cuda"
        return XGBRegressor(**xgb_kwargs)
    if name == "xgb_mse":
        xgb_kwargs: dict[str, Any] = {
            "objective": "reg:squarederror",
            "n_estimators": 550, "learning_rate": 0.018, "max_depth": 4, "min_child_weight": 6.5,
            "subsample": 0.75, "colsample_bytree": 0.50, "reg_lambda": 14.0, "tree_method": "hist",
            "random_state": seed, "n_jobs": n_jobs,
        }
        if use_gpu:
            xgb_kwargs["device"] = "cuda"
        return XGBRegressor(**xgb_kwargs)
    if name == "ridge":
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=100.0, random_state=seed)),
        ])
    if name == "extratrees":
        from sklearn.impute import SimpleImputer
        from sklearn.ensemble import ExtraTreesRegressor
        from sklearn.pipeline import Pipeline
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("extratrees", ExtraTreesRegressor(
                n_estimators=100, max_depth=15, min_samples_leaf=5,
                max_features="sqrt", random_state=seed, n_jobs=n_jobs,
            )),
        ])
    raise ValueError(f"Unknown regressor {name!r}")


def make_classifier(name: str, seed: int, n_jobs: int, use_gpu: bool = False) -> Any:
    """Multiclass probabilistic classifier."""
    if name == "lgbm":
        return LGBMClassifier(
            objective="multiclass", n_estimators=450, learning_rate=0.018, num_leaves=30,
            min_child_samples=51, subsample=0.77, subsample_freq=1, colsample_bytree=0.58,
            reg_lambda=10.0, random_state=seed, n_jobs=n_jobs, deterministic=True,
            force_col_wise=True, verbose=-1,
        )
    if name == "catboost":
        cb_kwargs: dict[str, Any] = {
            "iterations": 500, "learning_rate": 0.045, "depth": 6, "l2_leaf_reg": 6.0, "loss_function": "MultiClass",
            "random_seed": seed, "verbose": 0, "allow_writing_files": False,
        }
        if use_gpu:
            cb_kwargs["task_type"] = "GPU"
        else:
            cb_kwargs["thread_count"] = n_jobs
        return CatBoostClassifier(**cb_kwargs)
    if name == "xgb":
        xgb_kwargs: dict[str, Any] = {
            "objective": "multi:softprob", "n_estimators": 350, "learning_rate": 0.031, "max_depth": 5,
            "min_child_weight": 4.74, "subsample": 0.75, "colsample_bytree": 0.68, "reg_lambda": 9.2,
            "tree_method": "hist", "random_state": seed, "n_jobs": n_jobs,
        }
        if use_gpu:
            xgb_kwargs["device"] = "cuda"
        return XGBClassifier(**xgb_kwargs)
    raise ValueError(f"Unknown classifier {name!r}")


class RegressorEnsemble:
    """Weighted average of non-negative regressor predictions."""

    def __init__(
        self,
        members: Sequence[str],
        seed: int = 42,
        n_jobs: int = -1,
        tweedie_power: float = 1.15,
        use_gpu: bool = False,
    ) -> None:
        self.members = tuple(members)
        self.tweedie_power = tweedie_power
        self.models = [
            make_regressor(m, seed, n_jobs, tweedie_power=tweedie_power, use_gpu=use_gpu)
            for m in self.members
        ]
        self.weights = np.full(len(self.members), 1.0 / len(self.members))

    def fit(self, X: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "RegressorEnsemble":
        for model in self.models:
            if sample_weight is not None:
                try:
                    model.fit(X, y, sample_weight=sample_weight)
                    continue
                except (TypeError, ValueError):
                    pass
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
        use_gpu: bool = False,
    ) -> None:
        self.members = tuple(members)
        self.n_classes = n_classes
        self.models = [make_classifier(m, seed, n_jobs, use_gpu=use_gpu) for m in self.members]
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

    def fit(self, X: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "ClassifierEnsemble":
        y = np.asarray(y, dtype=int)
        if y.min() < 0 or y.max() >= self.n_classes:
            raise ValueError("Class indices out of range")
        self.classes_ = np.unique(y)
        encoded = np.searchsorted(self.classes_, y)  # contiguous 0..P-1 (required by XGBoost)
        for model in self.models:
            if sample_weight is not None:
                try:
                    model.fit(X, encoded, sample_weight=sample_weight)
                    continue
                except (TypeError, ValueError):
                    pass
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

    def __init__(self, seed: int = 42, n_jobs: int = -1, use_gpu: bool = False) -> None:
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
        cb_kwargs: dict[str, Any] = {
            "iterations": 400,
            "learning_rate": 0.04,
            "depth": 6,
            "loss_function": "Logloss",
            "random_seed": seed,
            "verbose": 0,
            "allow_writing_files": False,
        }
        if use_gpu:
            cb_kwargs["task_type"] = "GPU"
        else:
            cb_kwargs["thread_count"] = n_jobs
        self.catboost = CatBoostClassifier(**cb_kwargs)

        xgb_kwargs: dict[str, Any] = {
            "objective": "binary:logistic",
            "n_estimators": 350,
            "learning_rate": 0.03,
            "max_depth": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.6,
            "tree_method": "hist",
            "random_state": seed,
            "n_jobs": n_jobs,
        }
        if use_gpu:
            xgb_kwargs["device"] = "cuda"
        self.xgb = XGBClassifier(**xgb_kwargs)

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


class HierarchicalOpportunityClassifier:
    """Multi-tier hierarchical opportunity ensemble.

    Deconstructs the 17-class Opportunity problem into 4 specialized stages:
    1. Inactivity Gate: P(Inactivity | X) using LightGBM + CatBoost
    2. Active Retail Gate: P(Retail | X, Active) using LightGBM + CatBoost
    3. Pairwise Boundary: P(Fuel Growth | X, Fuel or Stable) using LightGBM + CatBoost
    4. Retail Multiclass: P(c | X, Retail) over secondary/minor retail classes using LightGBM
    """

    def __init__(
        self,
        n_classes: int = 17,
        seed: int = 42,
        n_jobs: int = -1,
        inact_idx: int = 4,
        stable_idx: int = 16,
        fg_idx: int = 2,
        use_gpu: bool = False,
    ) -> None:
        self.n_classes = n_classes
        self.seed = seed
        self.n_jobs = n_jobs
        self.inact_idx = inact_idx
        self.stable_idx = stable_idx
        self.fg_idx = fg_idx
        self.trio_indices = [inact_idx, stable_idx, fg_idx]
        self.retail_indices = [i for i in range(n_classes) if i not in self.trio_indices]
        self.retail_idx_to_local = {idx: i for i, idx in enumerate(self.retail_indices)}

        self.clf_inact_lgb = LGBMClassifier(
            n_estimators=350, learning_rate=0.02, num_leaves=31, colsample_bytree=0.6,
            reg_lambda=8.0, random_state=seed, n_jobs=n_jobs, verbose=-1,
        )
        cb_kwargs_inact: dict[str, Any] = {
            "iterations": 400, "learning_rate": 0.03, "depth": 6, "l2_leaf_reg": 6.0,
            "loss_function": "Logloss", "random_seed": seed, "verbose": 0, "allow_writing_files": False,
        }
        if use_gpu:
            cb_kwargs_inact["task_type"] = "GPU"
        else:
            cb_kwargs_inact["thread_count"] = n_jobs
        self.clf_inact_cb = CatBoostClassifier(**cb_kwargs_inact)

        xgb_kwargs_inact: dict[str, Any] = {
            "objective": "binary:logistic", "n_estimators": 350, "learning_rate": 0.02,
            "max_depth": 5, "min_child_weight": 5.0, "subsample": 0.8, "colsample_bytree": 0.6,
            "reg_lambda": 8.0, "tree_method": "hist", "random_state": seed, "n_jobs": n_jobs,
        }
        if use_gpu:
            xgb_kwargs_inact["device"] = "cuda"
        self.clf_inact_xgb = XGBClassifier(**xgb_kwargs_inact)

        self.clf_ret_lgb = LGBMClassifier(
            n_estimators=300, learning_rate=0.02, num_leaves=31, colsample_bytree=0.6,
            reg_lambda=8.0, random_state=seed, n_jobs=n_jobs, verbose=-1,
        )
        cb_kwargs_ret: dict[str, Any] = {
            "iterations": 350, "learning_rate": 0.03, "depth": 6, "l2_leaf_reg": 6.0,
            "loss_function": "Logloss", "random_seed": seed, "verbose": 0, "allow_writing_files": False,
        }
        if use_gpu:
            cb_kwargs_ret["task_type"] = "GPU"
        else:
            cb_kwargs_ret["thread_count"] = n_jobs
        self.clf_ret_cb = CatBoostClassifier(**cb_kwargs_ret)

        xgb_kwargs_ret: dict[str, Any] = {
            "objective": "binary:logistic", "n_estimators": 300, "learning_rate": 0.02,
            "max_depth": 5, "min_child_weight": 5.0, "subsample": 0.8, "colsample_bytree": 0.6,
            "reg_lambda": 8.0, "tree_method": "hist", "random_state": seed, "n_jobs": n_jobs,
        }
        if use_gpu:
            xgb_kwargs_ret["device"] = "cuda"
        self.clf_ret_xgb = XGBClassifier(**xgb_kwargs_ret)

        self.clf_fg_lgb = LGBMClassifier(
            n_estimators=350, learning_rate=0.02, num_leaves=31, colsample_bytree=0.6,
            reg_lambda=8.0, random_state=seed, n_jobs=n_jobs, verbose=-1,
        )
        cb_kwargs_fg: dict[str, Any] = {
            "iterations": 350, "learning_rate": 0.03, "depth": 6, "l2_leaf_reg": 6.0,
            "loss_function": "Logloss", "random_seed": seed, "verbose": 0, "allow_writing_files": False,
        }
        if use_gpu:
            cb_kwargs_fg["task_type"] = "GPU"
        else:
            cb_kwargs_fg["thread_count"] = n_jobs
        self.clf_fg_cb = CatBoostClassifier(**cb_kwargs_fg)

        xgb_kwargs_fg: dict[str, Any] = {
            "objective": "binary:logistic", "n_estimators": 350, "learning_rate": 0.02,
            "max_depth": 5, "min_child_weight": 5.0, "subsample": 0.8, "colsample_bytree": 0.6,
            "reg_lambda": 8.0, "tree_method": "hist", "random_state": seed, "n_jobs": n_jobs,
        }
        if use_gpu:
            xgb_kwargs_fg["device"] = "cuda"
        self.clf_fg_xgb = XGBClassifier(**xgb_kwargs_fg)

        self.clf_ret_multi = LGBMClassifier(
            objective="multiclass", num_class=len(self.retail_indices),
            n_estimators=300, learning_rate=0.02, num_leaves=28, colsample_bytree=0.6,
            reg_lambda=8.0, random_state=seed, n_jobs=n_jobs, verbose=-1,
        )

    def fit(self, X: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "HierarchicalOpportunityClassifier":
        y_arr = np.asarray(y, dtype=int)
        y_inact = (y_arr == self.inact_idx).astype(int)
        if sample_weight is not None:
            sw = np.asarray(sample_weight, dtype=np.float32)
            self.clf_inact_lgb.fit(X, y_inact, sample_weight=sw)
            self.clf_inact_cb.fit(X, y_inact, sample_weight=sw)
            self.clf_inact_xgb.fit(X, y_inact, sample_weight=sw)

            m_act = y_arr != self.inact_idx
            y_ret = np.isin(y_arr[m_act], self.retail_indices).astype(int)
            self.clf_ret_lgb.fit(X[m_act], y_ret, sample_weight=sw[m_act])
            self.clf_ret_cb.fit(X[m_act], y_ret, sample_weight=sw[m_act])
            self.clf_ret_xgb.fit(X[m_act], y_ret, sample_weight=sw[m_act])

            m_fs = np.isin(y_arr, [self.stable_idx, self.fg_idx])
            y_fg = (y_arr[m_fs] == self.fg_idx).astype(int)
            self.clf_fg_lgb.fit(X[m_fs], y_fg, sample_weight=sw[m_fs])
            self.clf_fg_cb.fit(X[m_fs], y_fg, sample_weight=sw[m_fs])
            self.clf_fg_xgb.fit(X[m_fs], y_fg, sample_weight=sw[m_fs])

            m_ret = np.isin(y_arr, self.retail_indices)
            y_ret_multi = np.array([self.retail_idx_to_local[yi] for yi in y_arr[m_ret]])
            self.clf_ret_multi.fit(X[m_ret], y_ret_multi, sample_weight=sw[m_ret])
        else:
            self.clf_inact_lgb.fit(X, y_inact)
            self.clf_inact_cb.fit(X, y_inact)
            self.clf_inact_xgb.fit(X, y_inact)

            m_act = y_arr != self.inact_idx
            y_ret = np.isin(y_arr[m_act], self.retail_indices).astype(int)
            self.clf_ret_lgb.fit(X[m_act], y_ret)
            self.clf_ret_cb.fit(X[m_act], y_ret)
            self.clf_ret_xgb.fit(X[m_act], y_ret)

            m_fs = np.isin(y_arr, [self.stable_idx, self.fg_idx])
            y_fg = (y_arr[m_fs] == self.fg_idx).astype(int)
            self.clf_fg_lgb.fit(X[m_fs], y_fg)
            self.clf_fg_cb.fit(X[m_fs], y_fg)
            self.clf_fg_xgb.fit(X[m_fs], y_fg)

            m_ret = np.isin(y_arr, self.retail_indices)
            y_ret_multi = np.array([self.retail_idx_to_local[yi] for yi in y_arr[m_ret]])
            self.clf_ret_multi.fit(X[m_ret], y_ret_multi)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p_inact = (
            0.40 * self.clf_inact_lgb.predict_proba(X)[:, 1]
            + 0.30 * self.clf_inact_cb.predict_proba(X)[:, 1]
            + 0.30 * self.clf_inact_xgb.predict_proba(X)[:, 1]
        )
        p_ret = (
            0.40 * self.clf_ret_lgb.predict_proba(X)[:, 1]
            + 0.30 * self.clf_ret_cb.predict_proba(X)[:, 1]
            + 0.30 * self.clf_ret_xgb.predict_proba(X)[:, 1]
        )
        p_fg = (
            0.40 * self.clf_fg_lgb.predict_proba(X)[:, 1]
            + 0.30 * self.clf_fg_cb.predict_proba(X)[:, 1]
            + 0.30 * self.clf_fg_xgb.predict_proba(X)[:, 1]
        )
        p_ret_multi = self.clf_ret_multi.predict_proba(X)

        n = len(X)
        p_i = np.clip(p_inact, 1e-6, 1.0 - 1e-6)
        p_a = 1.0 - p_i
        p_r = np.clip(p_ret, 1e-6, 1.0 - 1e-6)
        p_f = np.clip(p_fg, 1e-6, 1.0 - 1e-6)

        if "m3_baskets" in X.columns:
            is_dormant = X["m3_baskets"].to_numpy() == 0
            p_f = p_f.copy()
            p_f[is_dormant] = np.maximum(p_f[is_dormant], 0.98)

        joint = np.zeros((n, self.n_classes), dtype="float64")
        joint[:, self.inact_idx] = p_i
        p_commuter = p_a * (1.0 - p_r)
        joint[:, self.fg_idx] = p_commuter * p_f
        joint[:, self.stable_idx] = p_commuter * (1.0 - p_f)

        p_retail_total = p_a * p_r
        for local_i, global_i in enumerate(self.retail_indices):
            joint[:, global_i] = p_retail_total * p_ret_multi[:, local_i]

        return joint / joint.sum(axis=1, keepdims=True)

