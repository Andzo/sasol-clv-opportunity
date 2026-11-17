"""Model factories and ensemble wrappers for spend regression and opportunity classification.

Multiclass probability predictions are scattered into the full label index space
using ``classes_`` so rare-class omissions in early folds never misalign class
columns. LightGBM explicitly enables bagging via ``subsample_freq=1``, and
CatBoost disables sidecar directory writes.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor


GB_SPEED: float = 2.5  # Scale n_estimators down and learning_rate up by 2.5x, preserving total shrinkage


def _gb(n_estimators: int, learning_rate: float) -> dict[str, Any]:
    """Boosting schedule preserving total shrinkage (n_estimators * learning_rate)."""
    return {
        "n_estimators": max(50, int(round(n_estimators / GB_SPEED))),
        "learning_rate": float(learning_rate) * GB_SPEED,
    }


def _catboost_kwargs(
    *,
    iterations: int,
    learning_rate: float,
    depth: int,
    l2_leaf_reg: float,
    loss_function: str,
    seed: int,
    n_jobs: int,
    use_gpu: bool,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "iterations": iterations,
        "learning_rate": learning_rate,
        "depth": depth,
        "l2_leaf_reg": l2_leaf_reg,
        "loss_function": loss_function,
        "random_seed": seed,
        "verbose": 0,
        "allow_writing_files": False,
    }
    if use_gpu:
        kwargs["task_type"] = "GPU"
        kwargs["gpu_ram_part"] = 0.65
    else:
        kwargs["thread_count"] = n_jobs
    return kwargs


def _xgb_kwargs(*, seed: int, n_jobs: int, use_gpu: bool, **kwargs: Any) -> dict[str, Any]:
    params: dict[str, Any] = {
        "tree_method": "hist",
        "random_state": seed,
        "n_jobs": n_jobs,
        **kwargs,
    }
    if use_gpu:
        params["device"] = "cuda"
    return params


def make_regressor(
    name: str,
    seed: int,
    n_jobs: int,
    tweedie_power: float = 1.15,
    use_gpu: bool = False,
    huber_alpha: float = 1.35,
) -> Any:
    """Instantiate a regularized regressor across Tweedie, Huber, MSE, Ridge, or ExtraTrees families."""
    lgbm_base: dict[str, Any] = {
        **_gb(650, 0.018),
        "num_leaves": 28,
        "min_child_samples": 85,
        "subsample": 0.75,
        "subsample_freq": 1,
        "colsample_bytree": 0.50,
        "reg_lambda": 12.0,
        "random_state": seed,
        "n_jobs": n_jobs,
        "deterministic": True,
        "force_col_wise": True,
        "max_bin": 63,
        "verbose": -1,
    }
    if name in ("lgbm", "lgbm_tweedie"):
        return LGBMRegressor(objective="tweedie", tweedie_variance_power=tweedie_power, **lgbm_base)
    if name == "lgbm_huber":
        return LGBMRegressor(objective="huber", alpha=huber_alpha, **lgbm_base)
    if name == "lgbm_mse":
        return LGBMRegressor(objective="regression", **lgbm_base)
    if name == "lgbm_mse_deep":
        return LGBMRegressor(
            objective="regression",
            num_leaves=45,
            max_depth=6,
            min_child_samples=60,
            **_gb(750, 0.015),
            colsample_bytree=0.55,
            subsample=0.80,
            subsample_freq=1,
            reg_lambda=15.0,
            random_state=seed,
            n_jobs=n_jobs,
            deterministic=True,
            force_col_wise=True,
            max_bin=63,
            verbose=-1,
        )
    if name == "catboost":
        return CatBoostRegressor(
            **_catboost_kwargs(
                iterations=550,
                learning_rate=0.035,
                depth=5,
                l2_leaf_reg=10.0,
                loss_function="RMSE",
                seed=seed,
                n_jobs=n_jobs,
                use_gpu=use_gpu,
            )
        )
    if name == "catboost_huber":
        return CatBoostRegressor(
            **_catboost_kwargs(
                iterations=550,
                learning_rate=0.035,
                depth=5,
                l2_leaf_reg=10.0,
                loss_function=f"Huber:delta={huber_alpha}",
                seed=seed,
                n_jobs=n_jobs,
                use_gpu=use_gpu,
            )
        )
    xgb_base: dict[str, Any] = {
        **_gb(550, 0.018),
        "max_depth": 4,
        "min_child_weight": 6.5,
        "subsample": 0.75,
        "colsample_bytree": 0.50,
        "reg_lambda": 14.0,
    }
    if name in ("xgb", "xgb_tweedie"):
        return XGBRegressor(**_xgb_kwargs(objective="reg:tweedie", tweedie_variance_power=tweedie_power, seed=seed, n_jobs=n_jobs, use_gpu=use_gpu, **xgb_base))
    if name == "xgb_huber":
        return XGBRegressor(**_xgb_kwargs(objective="reg:pseudohubererror", seed=seed, n_jobs=n_jobs, use_gpu=use_gpu, **xgb_base))
    if name == "xgb_mse":
        return XGBRegressor(**_xgb_kwargs(objective="reg:squarederror", seed=seed, n_jobs=n_jobs, use_gpu=use_gpu, **xgb_base))
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
        from sklearn.ensemble import ExtraTreesRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline

        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            (
                "extratrees",
                ExtraTreesRegressor(
                    n_estimators=100,
                    max_depth=15,
                    min_samples_leaf=5,
                    max_features="sqrt",
                    random_state=seed,
                    n_jobs=n_jobs,
                ),
            ),
        ])
    raise ValueError(f"Unknown regressor {name!r}")


def make_classifier(name: str, seed: int, n_jobs: int, use_gpu: bool = False) -> Any:
    """Instantiate a multiclass probabilistic classifier."""
    if name == "lgbm":
        return LGBMClassifier(
            objective="multiclass",
            **_gb(450, 0.018),
            num_leaves=30,
            min_child_samples=85,
            subsample=0.77,
            subsample_freq=1,
            colsample_bytree=0.58,
            reg_lambda=10.0,
            random_state=seed,
            n_jobs=n_jobs,
            deterministic=True,
            force_col_wise=True,
            max_bin=63,
            verbose=-1,
        )
    if name == "catboost":
        return CatBoostClassifier(
            **_catboost_kwargs(
                iterations=250,
                learning_rate=0.05,
                depth=5,
                l2_leaf_reg=6.0,
                loss_function="MultiClass",
                seed=seed,
                n_jobs=n_jobs,
                use_gpu=use_gpu,
            )
        )
    if name == "xgb":
        return XGBClassifier(
            **_xgb_kwargs(
                objective="multi:softprob",
                **_gb(350, 0.031),
                max_depth=5,
                min_child_weight=4.74,
                subsample=0.75,
                colsample_bytree=0.68,
                reg_lambda=9.2,
                seed=seed,
                n_jobs=n_jobs,
                use_gpu=use_gpu,
            )
        )
    raise ValueError(f"Unknown classifier {name!r}")


CAT_STRUCTURAL_FEATURES: tuple[str, ...] = (
    "cat_primary_location_id",
    "cat_top_nonfuel_level2",
    "cat_top_nonfuel_level3",
    "cat_payday_commuter_archetype",
)


def _prepare_model_X(model: Any, X: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
    """Passes full X (with integer categorical IDs) to CatBoost, strips them for all other models."""
    if not isinstance(X, pd.DataFrame):
        return X
    if isinstance(model, (CatBoostRegressor, CatBoostClassifier)):
        return X
    cols_to_drop = [c for c in CAT_STRUCTURAL_FEATURES if c in X.columns]
    if cols_to_drop:
        return X.drop(columns=cols_to_drop)
    return X


def _fit_with_sample_weight(model: Any, X: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray | None = None) -> None:
    """Safely fits model with sample_weight, properly forwarding to sklearn Pipeline steps and CatBoost cat_features."""
    X_in = _prepare_model_X(model, X)
    fit_kwargs: dict[str, Any] = {}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight

    # Forward cat_features to CatBoost if present in X_in
    if isinstance(model, (CatBoostRegressor, CatBoostClassifier)) and isinstance(X_in, pd.DataFrame):
        cats_present = [c for c in CAT_STRUCTURAL_FEATURES if c in X_in.columns]
        if cats_present:
            fit_kwargs["cat_features"] = cats_present

    # Handle sklearn Pipeline: sample_weight must be prefixed with the final step name
    if hasattr(model, "steps") and len(model.steps) > 0:
        step_name = model.steps[-1][0]
        if sample_weight is not None:
            try:
                model.fit(X_in, y, **{f"{step_name}__sample_weight": sample_weight})
                return
            except (TypeError, ValueError):
                pass
        model.fit(X_in, y)
        return

    try:
        model.fit(X_in, y, **fit_kwargs)
        return
    except (TypeError, ValueError):
        pass

    if sample_weight is not None:
        try:
            model.fit(X_in, y, sample_weight=sample_weight)
            return
        except (TypeError, ValueError):
            pass

    model.fit(X_in, y)


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
            seed_preds = [np.clip(m.predict(_prepare_model_X(m, X)), 0.0, None) for m in sub_list]
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
        self.class_priors_: np.ndarray | None = None
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
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
        y = np.asarray(y, dtype=int)
        if y.min() < 0 or y.max() >= self.n_classes:
            raise ValueError("Class indices out of range")
        self.classes_ = np.unique(y)
        self.class_priors_ = np.bincount(y, minlength=self.n_classes) / float(len(y))
        encoded = np.searchsorted(self.classes_, y)  # contiguous 0..P-1 (required by XGBoost)
        for sub_list in self.models:
            for model in sub_list:
                _fit_with_sample_weight(model, X, encoded, sample_weight)
        return self

    def predict_member_proba(self, X: pd.DataFrame) -> list[np.ndarray]:
        """Per-member probabilities aligned to the full label space, averaged over seeds."""
        if self.classes_ is None:
            raise RuntimeError("Ensemble is not fitted")
        if isinstance(X, pd.DataFrame) and hasattr(self, "feature_names_in_"):
            if set(self.feature_names_in_).issubset(X.columns) and len(X.columns) != len(self.feature_names_in_):
                X = X[self.feature_names_in_]
        out = []
        for m_name, sub_list in zip(self.members, self.models):
            seed_probs = []
            for model in sub_list:
                p = np.asarray(model.predict_proba(_prepare_model_X(model, X)), dtype="float64")
                full = np.zeros((len(X), self.n_classes))
                full[:, self.classes_] = p
                seed_probs.append(full)
            avg_p = np.mean(seed_probs, axis=0)
            out.append(avg_p)
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
            cb_kwargs["gpu_ram_part"] = 0.65
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
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
        y_binary = (np.asarray(y) > 0).astype(int)
        X_lgb = _prepare_model_X(self.lgbm, X)
        X_cb = _prepare_model_X(self.catboost, X)
        X_xgb = _prepare_model_X(self.xgb, X)
        cb_kwargs: dict[str, Any] = {}
        if isinstance(X_cb, pd.DataFrame):
            cats = [c for c in CAT_STRUCTURAL_FEATURES if c in X_cb.columns]
            if cats:
                cb_kwargs["cat_features"] = cats
        if sample_weight is not None:
            sw = np.asarray(sample_weight, dtype=np.float32)
            self.lgbm.fit(X_lgb, y_binary, sample_weight=sw)
            self.catboost.fit(X_cb, y_binary, sample_weight=sw, **cb_kwargs)
            self.xgb.fit(X_xgb, y_binary, sample_weight=sw)
        else:
            self.lgbm.fit(X_lgb, y_binary)
            self.catboost.fit(X_cb, y_binary, **cb_kwargs)
            self.xgb.fit(X_xgb, y_binary)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if isinstance(X, pd.DataFrame) and hasattr(self, "feature_names_in_"):
            if set(self.feature_names_in_).issubset(X.columns) and len(X.columns) != len(self.feature_names_in_):
                X = X[self.feature_names_in_]
        X_lgb = _prepare_model_X(self.lgbm, X)
        X_cb = _prepare_model_X(self.catboost, X)
        X_xgb = _prepare_model_X(self.xgb, X)
        p_lgbm = np.asarray(self.lgbm.predict_proba(X_lgb)[:, 1], dtype="float64")
        p_cb = np.asarray(self.catboost.predict_proba(X_cb)[:, 1], dtype="float64")
        p_xgb = np.asarray(self.xgb.predict_proba(X_xgb)[:, 1], dtype="float64")
        return (p_lgbm + p_cb + p_xgb) / 3.0


class HierarchicalOpportunityClassifier:
    """Multi-tier hierarchical opportunity ensemble.

    Deconstructs the 17-class Opportunity problem into 4 stages:
    1. Inactivity gate: P(Inactivity | X) using LightGBM + CatBoost + XGBoost
    2. Active retail gate: P(Retail | X, Active) using LightGBM + CatBoost + XGBoost
    3. Commuter boundary: P(Fuel Growth | X, Fuel Growth or Stable) using LightGBM + CatBoost + XGBoost
    4. Retail multiclass: P(c | X, Retail) over retail categories using LightGBM
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

        def _make_stage(lgb_n: int, cb_iters: int, xgb_n: int) -> tuple[LGBMClassifier, CatBoostClassifier, XGBClassifier]:
            lgb = LGBMClassifier(
                n_estimators=lgb_n, learning_rate=0.02, num_leaves=31, colsample_bytree=0.6,
                reg_lambda=8.0, random_state=seed, n_jobs=n_jobs, verbose=-1,
            )
            cb = CatBoostClassifier(
                **_catboost_kwargs(
                    iterations=cb_iters, learning_rate=0.03, depth=6, l2_leaf_reg=6.0,
                    loss_function="Logloss", seed=seed, n_jobs=n_jobs, use_gpu=use_gpu,
                )
            )
            xgb = XGBClassifier(
                **_xgb_kwargs(
                    objective="binary:logistic", n_estimators=xgb_n, learning_rate=0.02,
                    max_depth=5, min_child_weight=5.0, subsample=0.8, colsample_bytree=0.6,
                    reg_lambda=8.0, seed=seed, n_jobs=n_jobs, use_gpu=use_gpu,
                )
            )
            return lgb, cb, xgb

        self.clf_inact_lgb, self.clf_inact_cb, self.clf_inact_xgb = _make_stage(350, 400, 350)
        self.clf_ret_lgb, self.clf_ret_cb, self.clf_ret_xgb = _make_stage(300, 350, 300)
        self.clf_fg_lgb, self.clf_fg_cb, self.clf_fg_xgb = _make_stage(350, 350, 350)

        self.clf_ret_multi = LGBMClassifier(
            objective="multiclass", num_class=len(self.retail_indices),
            n_estimators=300, learning_rate=0.02, num_leaves=28, colsample_bytree=0.6,
            reg_lambda=8.0, random_state=seed, n_jobs=n_jobs, verbose=-1,
        )
        self.temp_inact: float = 1.0
        self.temp_ret: float = 1.0
        self.temp_fg: float = 1.0

    @staticmethod
    def _predict_stage_proba(
        models: tuple[Any, Any, Any],
        X: pd.DataFrame,
        temperature: float,
    ) -> np.ndarray:
        import scipy.special as sp

        lgb, cb, xgb = models
        p_raw = (
            0.40 * lgb.predict_proba(_prepare_model_X(lgb, X))[:, 1]
            + 0.30 * cb.predict_proba(_prepare_model_X(cb, X))[:, 1]
            + 0.30 * xgb.predict_proba(_prepare_model_X(xgb, X))[:, 1]
        )
        return sp.expit(sp.logit(np.clip(p_raw, 1e-5, 1.0 - 1e-5)) / max(temperature, 0.1))

    def fit(self, X: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "HierarchicalOpportunityClassifier":
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
        y_arr = np.asarray(y, dtype=int)
        y_inact = (y_arr == self.inact_idx).astype(int)
        sw = np.asarray(sample_weight, dtype=np.float32) if sample_weight is not None else None

        for clf in (self.clf_inact_lgb, self.clf_inact_cb, self.clf_inact_xgb):
            _fit_with_sample_weight(clf, X, y_inact, sw)

        m_act = y_arr != self.inact_idx
        y_ret = np.isin(y_arr[m_act], self.retail_indices).astype(int)
        sw_act = sw[m_act] if sw is not None else None
        for clf in (self.clf_ret_lgb, self.clf_ret_cb, self.clf_ret_xgb):
            _fit_with_sample_weight(clf, X[m_act], y_ret, sw_act)

        ever_fuel_mask = (X["ever_fuel"].to_numpy() > 0) if "ever_fuel" in X.columns else np.ones(len(y_arr), dtype=bool)
        m_fs = np.isin(y_arr, [self.stable_idx, self.fg_idx]) & ever_fuel_mask
        y_fg = (y_arr[m_fs] == self.fg_idx).astype(int)
        sw_fs = sw[m_fs] if sw is not None else None
        for clf in (self.clf_fg_lgb, self.clf_fg_cb, self.clf_fg_xgb):
            _fit_with_sample_weight(clf, X[m_fs], y_fg, sw_fs)

        m_ret = np.isin(y_arr, self.retail_indices)
        y_ret_multi = np.array([self.retail_idx_to_local[yi] for yi in y_arr[m_ret]])
        sw_ret = sw[m_ret] if sw is not None else None
        _fit_with_sample_weight(self.clf_ret_multi, X[m_ret], y_ret_multi, sw_ret)

        self.temp_inact = 1.0
        self.temp_ret = 1.0
        self.temp_fg = 1.0
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if isinstance(X, pd.DataFrame) and hasattr(self, "feature_names_in_"):
            if set(self.feature_names_in_).issubset(X.columns) and len(X.columns) != len(self.feature_names_in_):
                X = X[self.feature_names_in_]

        p_inact = self._predict_stage_proba((self.clf_inact_lgb, self.clf_inact_cb, self.clf_inact_xgb), X, self.temp_inact)
        p_ret = self._predict_stage_proba((self.clf_ret_lgb, self.clf_ret_cb, self.clf_ret_xgb), X, self.temp_ret)
        p_fg = self._predict_stage_proba((self.clf_fg_lgb, self.clf_fg_cb, self.clf_fg_xgb), X, self.temp_fg)
        p_ret_multi = self.clf_ret_multi.predict_proba(_prepare_model_X(self.clf_ret_multi, X))

        p_i = np.clip(p_inact, 1e-6, 1.0 - 1e-6)
        p_a = 1.0 - p_i
        p_r = np.clip(p_ret, 1e-6, 1.0 - 1e-6)
        p_f = np.clip(p_fg, 1e-6, 1.0 - 1e-6)

        if "ever_fuel" in X.columns:
            never_fuel = X["ever_fuel"].to_numpy() == 0
            p_f = p_f.copy()
            p_f[never_fuel] = 0.0

        joint = np.zeros((len(X), self.n_classes), dtype="float64")
        joint[:, self.inact_idx] = p_i
        p_commuter = p_a * (1.0 - p_r)
        joint[:, self.fg_idx] = p_commuter * p_f
        joint[:, self.stable_idx] = p_commuter * (1.0 - p_f)

        p_retail_total = p_a * p_r
        for local_i, global_i in enumerate(self.retail_indices):
            joint[:, global_i] = p_retail_total * p_ret_multi[:, local_i]

        return joint / joint.sum(axis=1, keepdims=True)

