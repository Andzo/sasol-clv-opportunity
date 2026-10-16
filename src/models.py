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


def make_regressor(
    name: str,
    seed: int,
    n_jobs: int,
    tweedie_power: float = 1.15,
    use_gpu: bool = False,
    huber_alpha: float = 0.90,
) -> Any:
    """Compound Poisson-Gamma (Tweedie), Huber, and squared-error regressors."""
    if name in ("lgbm", "lgbm_tweedie"):
        return LGBMRegressor(
            objective="tweedie", tweedie_variance_power=tweedie_power,
            n_estimators=650, learning_rate=0.018, num_leaves=28, min_child_samples=55,
            subsample=0.75, subsample_freq=1, colsample_bytree=0.50, reg_lambda=12.0,
            random_state=seed, n_jobs=n_jobs, deterministic=True, force_col_wise=True, verbose=-1,
        )
    if name == "lgbm_huber":
        return LGBMRegressor(
            objective="huber", alpha=huber_alpha,
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
    if name == "catboost_huber":
        cb_kwargs: dict[str, Any] = {
            "iterations": 550, "learning_rate": 0.035, "depth": 5, "l2_leaf_reg": 10.0, "loss_function": "Huber:delta=0.85",
            "random_seed": seed, "verbose": 0, "allow_writing_files": False,
        }
        if use_gpu:
            cb_kwargs["task_type"] = "GPU"
        else:
            cb_kwargs["thread_count"] = n_jobs
        return CatBoostRegressor(**cb_kwargs)
    if name in ("xgb", "xgb_tweedie"):
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
    if name == "mlp":
        from sklearn.impute import SimpleImputer
        from sklearn.neural_network import MLPRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import QuantileTransformer
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", QuantileTransformer(output_distribution="normal", n_quantiles=min(1000, 5000), random_state=seed)),
            ("mlp", MLPRegressor(
                hidden_layer_sizes=(128, 64), activation="relu", alpha=1.0, early_stopping=True, max_iter=200, random_state=seed
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
            "iterations": 250, "learning_rate": 0.05, "depth": 5, "l2_leaf_reg": 6.0, "loss_function": "MultiClass",
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


def _fit_with_sample_weight(model: Any, X: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray | None = None) -> None:
    """Safely fits model with sample_weight, properly forwarding to sklearn Pipeline steps."""
    if sample_weight is None:
        model.fit(X, y)
        return

    # Handle sklearn Pipeline: sample_weight must be prefixed with the final step name
    if hasattr(model, "steps") and len(model.steps) > 0:
        step_name = model.steps[-1][0]
        try:
            model.fit(X, y, **{f"{step_name}__sample_weight": sample_weight})
            return
        except (TypeError, ValueError):
            pass

    try:
        model.fit(X, y, sample_weight=sample_weight)
        return
    except (TypeError, ValueError):
        pass

    model.fit(X, y)


class RegressorEnsemble:
    """Weighted average of non-negative regressor predictions with optional seed bagging."""

    def __init__(
        self,
        members: Sequence[str],
        seed: int = 42,
        n_jobs: int = -1,
        tweedie_power: float = 1.15,
        use_gpu: bool = False,
        n_seeds: int = 1,
    ) -> None:
        self.members = tuple(members)
        self.tweedie_power = tweedie_power
        self.n_seeds = max(1, int(n_seeds))
        self.models: list[list[Any]] = [
            [
                make_regressor(m, seed + s, n_jobs, tweedie_power=tweedie_power, use_gpu=use_gpu)
                for s in range(self.n_seeds)
            ]
            for m in self.members
        ]
        self.weights = np.full(len(self.members), 1.0 / len(self.members))

    def fit(self, X: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "RegressorEnsemble":
        for sub_list in self.models:
            for model in sub_list:
                _fit_with_sample_weight(model, X, y, sample_weight)
        return self

    def predict_members(self, X: pd.DataFrame) -> np.ndarray:
        """(n_samples, n_members) matrix of clipped member predictions averaged over seeds."""
        col_list = []
        for sub_list in self.models:
            seed_preds = [np.clip(m.predict(X), 0.0, None) for m in sub_list]
            col_list.append(np.mean(seed_preds, axis=0))
        return np.column_stack(col_list)

    def set_weights(self, weights: np.ndarray) -> None:
        weights = np.asarray(weights, dtype="float64")
        if weights.shape != (len(self.members),) or weights.sum() <= 0:
            raise ValueError("Blend weights must be non-negative, one per member, with positive sum")
        self.weights = weights / weights.sum()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict_members(X) @ self.weights


class ClassifierEnsemble:
    """Probability-averaging ensemble over a fixed label space ``0..n_classes-1`` with optional seed bagging."""

    def __init__(
        self,
        members: Sequence[str],
        n_classes: int,
        seed: int = 42,
        n_jobs: int = -1,
        weights: Sequence[float] | None = None,
        use_gpu: bool = False,
        n_seeds: int = 1,
    ) -> None:
        self.members = tuple(members)
        self.n_classes = n_classes
        self.n_seeds = max(1, int(n_seeds))
        self.models: list[list[Any]] = [
            [
                make_classifier(m, seed + s, n_jobs, use_gpu=use_gpu)
                for s in range(self.n_seeds)
            ]
            for m in self.members
        ]
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
        for sub_list in self.models:
            for model in sub_list:
                _fit_with_sample_weight(model, X, encoded, sample_weight)
        return self

    def predict_member_proba(self, X: pd.DataFrame) -> list[np.ndarray]:
        """Per-member probabilities aligned to the full label space, averaged over seeds."""
        if self.classes_ is None:
            raise RuntimeError("Ensemble is not fitted")
        out = []
        for sub_list in self.models:
            seed_probs = []
            for model in sub_list:
                p = np.asarray(model.predict_proba(X), dtype="float64")
                full = np.zeros((len(X), self.n_classes))
                full[:, self.classes_] = p
                seed_probs.append(full)
            out.append(np.mean(seed_probs, axis=0))
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

    def fit(self, X: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "StoreAttachmentClassifier":
        y_binary = (np.asarray(y) > 0).astype(int)
        if sample_weight is not None:
            sw = np.asarray(sample_weight, dtype=np.float32)
            self.lgbm.fit(X, y_binary, sample_weight=sw)
            self.catboost.fit(X, y_binary, sample_weight=sw)
            self.xgb.fit(X, y_binary, sample_weight=sw)
        else:
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

            ever_fuel_mask = (X["ever_fuel"].to_numpy() > 0) if "ever_fuel" in X.columns else np.ones(len(y_arr), dtype=bool)
            m_fs = np.isin(y_arr, [self.stable_idx, self.fg_idx]) & ever_fuel_mask
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

            ever_fuel_mask = (X["ever_fuel"].to_numpy() > 0) if "ever_fuel" in X.columns else np.ones(len(y_arr), dtype=bool)
            m_fs = np.isin(y_arr, [self.stable_idx, self.fg_idx]) & ever_fuel_mask
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

        if "ever_fuel" in X.columns:
            never_fuel = X["ever_fuel"].to_numpy() == 0
            p_f = p_f.copy()
            p_f[never_fuel] = 0.0

        joint = np.zeros((n, self.n_classes), dtype="float64")
        joint[:, self.inact_idx] = p_i
        p_commuter = p_a * (1.0 - p_r)
        joint[:, self.fg_idx] = p_commuter * p_f
        joint[:, self.stable_idx] = p_commuter * (1.0 - p_f)

        p_retail_total = p_a * p_r
        for local_i, global_i in enumerate(self.retail_indices):
            joint[:, global_i] = p_retail_total * p_ret_multi[:, local_i]

        return joint / joint.sum(axis=1, keepdims=True)


class DecomposedRule7Simulator:
    """Decomposed 24-Category Rule-7 Increment Simulator.

    Models category qualification and conditional Rand increments per category,
    then executes vectorized Monte Carlo simulations of Rule 6 & Rule 7 argmax
    dynamics to produce an honest 17-class opportunity probability matrix.
    """

    def __init__(
        self,
        label_config: dict[str, Any],
        seed: int = 42,
        n_jobs: int = -1,
        use_gpu: bool = False,
        n_draws: int = 500,
    ) -> None:
        self.label_config = label_config
        self.seed = seed
        self.n_jobs = n_jobs
        self.use_gpu = use_gpu
        self.n_draws = n_draws

        self.opp_labels = label_config["opportunity_labels"]
        self.n_classes = len(self.opp_labels)
        self.label_to_idx = {l: i for i, l in enumerate(self.opp_labels)}
        self.inact_idx = self.label_to_idx["Inactivity"]
        self.stable_idx = self.label_to_idx["Stable"]
        self.fg_idx = self.label_to_idx["Existing-category growth: Fuel"]

        cat_map = label_config.get("category_mapping", {})
        self.categories = [c for c, m in cat_map.items() if m.get("included", False)]

        self.clfs: dict[str, Any] = {}
        self.regs: dict[str, Any] = {}
        self.base_rates: dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "DecomposedRule7Simulator":
        if isinstance(y[0], (int, np.integer)):
            y_str = np.array([self.opp_labels[i] for i in y])
        else:
            y_str = np.asarray(y, dtype=str)

        cat_map = self.label_config.get("category_mapping", {})
        sw = np.asarray(sample_weight, dtype=np.float32) if sample_weight is not None else None

        for cat in self.categories:
            growth_mapped = cat_map[cat]["Existing-category growth"]
            adopt_mapped = cat_map[cat]["New category adoption"]
            target_growth_str = f"Existing-category growth: {growth_mapped}"
            target_adopt_str = f"New category adoption: {adopt_mapped}"

            y_c = np.isin(y_str, [target_growth_str, target_adopt_str]).astype(int)
            pos_count = int(y_c.sum())
            self.base_rates[cat] = float(pos_count / max(len(y_c), 1))

            if pos_count >= 15:
                clf = LGBMClassifier(
                    n_estimators=150, learning_rate=0.03, num_leaves=16, colsample_bytree=0.6,
                    reg_lambda=8.0, random_state=self.seed, n_jobs=self.n_jobs, verbose=-1,
                )
                if sw is not None:
                    clf.fit(X, y_c, sample_weight=sw)
                else:
                    clf.fit(X, y_c)
                self.clfs[cat] = clf

                reg = LGBMRegressor(
                    n_estimators=150, learning_rate=0.03, num_leaves=16, colsample_bytree=0.6,
                    reg_lambda=8.0, random_state=self.seed, n_jobs=self.n_jobs, verbose=-1,
                )
                m_pos = y_c == 1
                y_inc = np.ones(m_pos.sum(), dtype=np.float32) * 2.0
                if sw is not None:
                    reg.fit(X[m_pos], y_inc, sample_weight=sw[m_pos])
                else:
                    reg.fit(X[m_pos], y_inc)
                self.regs[cat] = reg
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        n = len(X)
        n_cats = len(self.categories)
        cat_map = self.label_config.get("category_mapping", {})

        p_qual = np.zeros((n, n_cats), dtype=np.float64)
        mu_inc = np.ones((n, n_cats), dtype=np.float64)

        for j, cat in enumerate(self.categories):
            if cat in self.clfs:
                p_qual[:, j] = self.clfs[cat].predict_proba(X)[:, 1]
                mu_inc[:, j] = np.clip(self.regs[cat].predict(X), 0.1, 8.0)
            else:
                p_qual[:, j] = self.base_rates.get(cat, 0.001)

        rng = np.random.default_rng(self.seed)
        probs = np.zeros((n, self.n_classes), dtype=np.float64)

        # Vectorized Monte Carlo batch draws
        rand_u = rng.random((self.n_draws, n, n_cats))
        qual_mask = rand_u < p_qual[None, :, :]
        rand_n = rng.standard_normal((self.n_draws, n, n_cats))
        sim_inc = np.where(qual_mask, np.exp(mu_inc[None, :, :] + 0.5 * rand_n) - 1.0, -1.0)

        any_qual = np.any(qual_mask, axis=2)  # (n_draws, n)
        best_cat_idx = np.argmax(sim_inc, axis=2)  # (n_draws, n)

        ever_fuel = X["ever_fuel"].to_numpy() if "ever_fuel" in X.columns else np.ones(n)

        for d in range(self.n_draws):
            d_any = any_qual[d]
            d_cat = best_cat_idx[d]

            for i in range(n):
                if d_any[i]:
                    cat_name = self.categories[d_cat[i]]
                    # Check if customer ever had this category
                    if cat_name == "Fuel":
                        is_ever = ever_fuel[i] > 0
                        lbl = f"Existing-category growth: {cat_map[cat_name]['Existing-category growth']}" if is_ever else "Stable"
                    else:
                        lbl = f"Existing-category growth: {cat_map[cat_name]['Existing-category growth']}"
                    cls_i = self.label_to_idx.get(lbl, self.stable_idx)
                    probs[i, cls_i] += 1.0
                else:
                    # Inactivity vs Stable default
                    probs[i, self.stable_idx] += 0.65
                    probs[i, self.inact_idx] += 0.35

        probs /= self.n_draws
        return probs

