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
class TabMPLRRegressor:
    """TabM (BatchEnsemble MLP) with Periodic / PLR Numerical Embeddings.

    Gorishniy et al. (NeurIPS 2022, ICLR 2025).
    """

    def __init__(
        self,
        seed: int = 42,
        n_epochs: int = 15,
        lr: float = 0.005,
        batch_size: int = 1024,
        k_heads: int = 8,
    ) -> None:
        self.seed = seed
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.k_heads = k_heads
        self.model: Any = None
        self.scaler: Any = None
        self.imputer: Any = None
        self.freqs = np.array([0.5, 1.0, 2.0, 4.0], dtype=np.float32)

    def _transform(self, X: np.ndarray) -> np.ndarray:
        X_sc = self.scaler.transform(X)
        n_feats = min(40, X_sc.shape[1])
        X_sub = X_sc[:, :n_feats]
        parts = [X_sc]
        for w in self.freqs:
            parts.append(np.sin(2.0 * np.pi * w * X_sub))
            parts.append(np.cos(2.0 * np.pi * w * X_sub))
        return np.column_stack(parts).astype(np.float32)

    def fit(self, X: pd.DataFrame | np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "TabMPLRRegressor":
        import torch
        import torch.nn as nn
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import QuantileTransformer

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        X_arr = np.nan_to_num(np.asarray(X, dtype=np.float32), nan=0.0)
        y_arr = np.asarray(y, dtype=np.float32).reshape(-1, 1)

        self.imputer = SimpleImputer(strategy="median")
        X_imp = self.imputer.fit_transform(X_arr)
        self.scaler = QuantileTransformer(
            output_distribution="uniform",
            n_quantiles=min(1000, len(X_imp)),
            random_state=self.seed,
        )
        self.scaler.fit(X_imp)

        X_emb = self._transform(X_imp)
        in_dim = X_emb.shape[1]

        class BatchEnsembleLinear(nn.Module):
            def __init__(self, in_features: int, out_features: int, k: int):
                super().__init__()
                self.k = k
                self.w = nn.Parameter(torch.randn(in_features, out_features) / (in_features ** 0.5))
                self.r = nn.Parameter(torch.ones(k, in_features) + 0.02 * torch.randn(k, in_features))
                self.s = nn.Parameter(torch.ones(k, out_features) + 0.02 * torch.randn(k, out_features))
                self.b = nn.Parameter(torch.zeros(k, out_features))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # x: (k, B, in_features)
                xr = x * self.r.unsqueeze(1)
                out = torch.matmul(xr, self.w) * self.s.unsqueeze(1) + self.b.unsqueeze(1)
                return out

        class TabMNet(nn.Module):
            def __init__(self, in_features: int, k: int = 8):
                super().__init__()
                self.k = k
                self.norm = nn.LayerNorm(in_features)
                self.layer1 = BatchEnsembleLinear(in_features, 128, k)
                self.act1 = nn.SiLU()
                self.drop1 = nn.Dropout(0.15)
                self.layer2 = BatchEnsembleLinear(128, 64, k)
                self.act2 = nn.SiLU()
                self.drop2 = nn.Dropout(0.15)
                self.head_w = nn.Parameter(torch.randn(k, 64, 1) / 8.0)
                self.head_b = nn.Parameter(torch.zeros(k, 1))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # x: (B, in_features)
                x_norm = self.norm(x)
                x_k = x_norm.unsqueeze(0).expand(self.k, -1, -1)
                h1 = self.drop1(self.act1(self.layer1(x_k)))
                h2 = self.drop2(self.act2(self.layer2(h1)))
                out = (torch.matmul(h2, self.head_w).squeeze(-1) + self.head_b).permute(1, 0)
                return out

        self.model = TabMNet(in_dim, self.k_heads)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.n_epochs, eta_min=1e-4)
        criterion = nn.HuberLoss(delta=1.0, reduction="none")

        x_t = torch.tensor(X_emb, dtype=torch.float32)
        y_t = torch.tensor(y_arr, dtype=torch.float32)
        w_t = torch.tensor(sample_weight if sample_weight is not None else np.ones(len(y_arr)), dtype=torch.float32).reshape(-1, 1)

        n = len(X_emb)
        self.model.train()
        for epoch in range(self.n_epochs):
            perm = torch.randperm(n)
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                xb, yb, wb = x_t[idx], y_t[idx], w_t[idx]
                heads_out = self.model(xb)
                loss = (criterion(heads_out, yb.expand(-1, self.k_heads)).mean(dim=1, keepdim=True) * wb).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
        self.model.eval()
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        import torch
        X_arr = np.nan_to_num(np.asarray(X, dtype=np.float32), nan=0.0)
        X_imp = self.imputer.transform(X_arr)
        X_emb = self._transform(X_imp)
        with torch.no_grad():
            x_t = torch.tensor(X_emb, dtype=torch.float32)
            heads_out = self.model(x_t)
            pred = heads_out.mean(dim=1).cpu().numpy()
        return np.clip(pred, 0.0, None)


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
            n_estimators=650, learning_rate=0.018, num_leaves=28, min_child_samples=85,
            subsample=0.75, subsample_freq=1, colsample_bytree=0.50, reg_lambda=12.0,
            random_state=seed, n_jobs=n_jobs, deterministic=True, force_col_wise=True, verbose=-1,
        )
    if name == "lgbm_huber":
        return LGBMRegressor(
            objective="huber", alpha=huber_alpha,
            n_estimators=650, learning_rate=0.018, num_leaves=28, min_child_samples=85,
            subsample=0.75, subsample_freq=1, colsample_bytree=0.50, reg_lambda=12.0,
            random_state=seed, n_jobs=n_jobs, deterministic=True, force_col_wise=True, verbose=-1,
        )
    if name == "lgbm_mse":
        return LGBMRegressor(
            objective="regression",
            n_estimators=650, learning_rate=0.018, num_leaves=28, min_child_samples=85,
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
    if name in ("mlp", "tabm"):
        return TabMPLRRegressor(seed=seed)
    raise ValueError(f"Unknown regressor {name!r}")


def make_classifier(name: str, seed: int, n_jobs: int, use_gpu: bool = False) -> Any:
    """Multiclass probabilistic classifier."""
    if name == "lgbm":
        return LGBMClassifier(
            objective="multiclass", n_estimators=450, learning_rate=0.018, num_leaves=30,
            min_child_samples=85, subsample=0.77, subsample_freq=1, colsample_bytree=0.58,
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


CAT_STRUCTURAL_FEATURES: tuple[str, ...] = (
    "cat_primary_location_id",
    "cat_top_nonfuel_level2",
    "cat_top_nonfuel_level3",
    "cat_payday_commuter_archetype",
)


def _fit_with_sample_weight(model: Any, X: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray | None = None) -> None:
    """Safely fits model with sample_weight, properly forwarding to sklearn Pipeline steps and CatBoost cat_features."""
    fit_kwargs: dict[str, Any] = {}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight

    # Forward cat_features to CatBoost if present in X
    if isinstance(model, (CatBoostRegressor, CatBoostClassifier)) and isinstance(X, pd.DataFrame):
        cats_present = [c for c in CAT_STRUCTURAL_FEATURES if c in X.columns]
        if cats_present:
            fit_kwargs["cat_features"] = cats_present

    # Handle sklearn Pipeline: sample_weight must be prefixed with the final step name
    if hasattr(model, "steps") and len(model.steps) > 0:
        step_name = model.steps[-1][0]
        if sample_weight is not None:
            try:
                model.fit(X, y, **{f"{step_name}__sample_weight": sample_weight})
                return
            except (TypeError, ValueError):
                pass
        model.fit(X, y)
        return

    try:
        model.fit(X, y, **fit_kwargs)
        return
    except (TypeError, ValueError):
        pass

    if sample_weight is not None:
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
        self.temp_inact: float = 1.0
        self.temp_ret: float = 1.0
        self.temp_fg: float = 1.0

    def _fit_gate_temperature(self, X: pd.DataFrame, y_bin: np.ndarray, sw: np.ndarray | None = None) -> float:
        """Calibrates gate temperature using 3-fold OOF cross-entropy minimization."""
        if len(np.unique(y_bin)) < 2:
            return 1.0
        from sklearn.model_selection import StratifiedKFold
        import scipy.special as sp
        from scipy.optimize import minimize_scalar

        n_pos = int(np.sum(y_bin == 1))
        n_neg = int(np.sum(y_bin == 0))
        n_splits = min(3, max(2, min(n_pos, n_neg)))
        if n_splits < 2:
            return 1.0

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        oof_z = np.zeros(len(y_bin), dtype=np.float64)
        clf = LGBMClassifier(
            n_estimators=100, learning_rate=0.04, num_leaves=16,
            random_state=self.seed, n_jobs=self.n_jobs, verbose=-1,
        )
        for tr_idx, val_idx in skf.split(X, y_bin):
            X_tr, y_tr = X.iloc[tr_idx], y_bin[tr_idx]
            X_val = X.iloc[val_idx]
            sw_tr = sw[tr_idx] if sw is not None else None
            if sw_tr is not None:
                clf.fit(X_tr, y_tr, sample_weight=sw_tr)
            else:
                clf.fit(X_tr, y_tr)
            p_val = np.clip(clf.predict_proba(X_val)[:, 1], 1e-5, 1.0 - 1e-5)
            oof_z[val_idx] = sp.logit(p_val)

        y_float = np.asarray(y_bin, dtype=np.float64)
        w = np.ones(len(y_bin), dtype=np.float64) if sw is None else np.asarray(sw, dtype=np.float64)
        w = w / max(float(w.mean()), 1e-6)

        def nll(T: float) -> float:
            z_t = oof_z / max(T, 0.1)
            loss = np.where(y_float == 1, np.log1p(np.exp(-z_t)), np.log1p(np.exp(z_t)))
            return float(np.mean(loss * w))

        res = minimize_scalar(nll, bounds=(0.5, 3.5), method="bounded")
        return float(res.x)

    def fit(self, X: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "HierarchicalOpportunityClassifier":
        y_arr = np.asarray(y, dtype=int)
        y_inact = (y_arr == self.inact_idx).astype(int)
        sw = np.asarray(sample_weight, dtype=np.float32) if sample_weight is not None else None

        _fit_with_sample_weight(self.clf_inact_lgb, X, y_inact, sw)
        _fit_with_sample_weight(self.clf_inact_cb, X, y_inact, sw)
        _fit_with_sample_weight(self.clf_inact_xgb, X, y_inact, sw)

        m_act = y_arr != self.inact_idx
        y_ret = np.isin(y_arr[m_act], self.retail_indices).astype(int)
        sw_act = sw[m_act] if sw is not None else None
        _fit_with_sample_weight(self.clf_ret_lgb, X[m_act], y_ret, sw_act)
        _fit_with_sample_weight(self.clf_ret_cb, X[m_act], y_ret, sw_act)
        _fit_with_sample_weight(self.clf_ret_xgb, X[m_act], y_ret, sw_act)

        ever_fuel_mask = (X["ever_fuel"].to_numpy() > 0) if "ever_fuel" in X.columns else np.ones(len(y_arr), dtype=bool)
        m_fs = np.isin(y_arr, [self.stable_idx, self.fg_idx]) & ever_fuel_mask
        y_fg = (y_arr[m_fs] == self.fg_idx).astype(int)
        sw_fs = sw[m_fs] if sw is not None else None
        _fit_with_sample_weight(self.clf_fg_lgb, X[m_fs], y_fg, sw_fs)
        _fit_with_sample_weight(self.clf_fg_cb, X[m_fs], y_fg, sw_fs)
        _fit_with_sample_weight(self.clf_fg_xgb, X[m_fs], y_fg, sw_fs)

        m_ret = np.isin(y_arr, self.retail_indices)
        y_ret_multi = np.array([self.retail_idx_to_local[yi] for yi in y_arr[m_ret]])
        sw_ret = sw[m_ret] if sw is not None else None
        _fit_with_sample_weight(self.clf_ret_multi, X[m_ret], y_ret_multi, sw_ret)

        self.temp_inact = self._fit_gate_temperature(X, y_inact, sw=sw)
        self.temp_ret = self._fit_gate_temperature(X[m_act], y_ret, sw=sw_act)
        self.temp_fg = self._fit_gate_temperature(X[m_fs], y_fg, sw=sw_fs)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p_inact_raw = (
            0.40 * self.clf_inact_lgb.predict_proba(X)[:, 1]
            + 0.30 * self.clf_inact_cb.predict_proba(X)[:, 1]
            + 0.30 * self.clf_inact_xgb.predict_proba(X)[:, 1]
        )
        p_ret_raw = (
            0.40 * self.clf_ret_lgb.predict_proba(X)[:, 1]
            + 0.30 * self.clf_ret_cb.predict_proba(X)[:, 1]
            + 0.30 * self.clf_ret_xgb.predict_proba(X)[:, 1]
        )
        p_fg_raw = (
            0.40 * self.clf_fg_lgb.predict_proba(X)[:, 1]
            + 0.30 * self.clf_fg_cb.predict_proba(X)[:, 1]
            + 0.30 * self.clf_fg_xgb.predict_proba(X)[:, 1]
        )
        p_ret_multi = self.clf_ret_multi.predict_proba(X)

        import scipy.special as sp
        p_inact = sp.expit(sp.logit(np.clip(p_inact_raw, 1e-5, 1.0 - 1e-5)) / max(self.temp_inact, 0.1))
        p_ret = sp.expit(sp.logit(np.clip(p_ret_raw, 1e-5, 1.0 - 1e-5)) / max(self.temp_ret, 0.1))
        p_fg = sp.expit(sp.logit(np.clip(p_fg_raw, 1e-5, 1.0 - 1e-5)) / max(self.temp_fg, 0.1))

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
                if cat in self.regs:
                    mu_inc[:, j] = np.clip(self.regs[cat].predict(X), 0.1, 8.0)
                else:
                    mu_inc[:, j] = 1.0
            else:
                p_qual[:, j] = self.base_rates.get(cat, 0.001)
                mu_inc[:, j] = 1.0

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


class StructuralMetaClassifier:
    """Level-2 Structural Meta-Learner (Wolpert 1992 Stacking).

    Blends out-of-fold multi-class probabilities with physical regression hurdles,
    Gaussian quantile exceedance, and survival hazard.
    Trains:
    1. Regularized LGBMClassifier(max_depth=3, num_leaves=7, min_child_samples=100, reg_lambda=25.0)
    2. Calibrated Multinomial LogisticRegression(C=0.5)
    Blends predictions: 0.65 * LGBM + 0.35 * LogReg.
    """

    def __init__(self, n_classes: int = 17, seed: int = 42, n_jobs: int = -1) -> None:
        self.n_classes = n_classes
        self.seed = seed
        self.n_jobs = n_jobs
        self.lgb = LGBMClassifier(
            objective="multiclass", max_depth=3, num_leaves=7,
            min_child_samples=100, learning_rate=0.03, n_estimators=200,
            reg_lambda=25.0, colsample_bytree=0.70, subsample=0.80, subsample_freq=1,
            random_state=seed, n_jobs=n_jobs, deterministic=True, force_col_wise=True, verbose=-1,
        )
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer
        self.logreg = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(C=0.5, max_iter=500, random_state=seed)),
        ])

    def _to_logreg_features(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X).copy()
        for col in X_df.columns:
            col_str = str(col)
            if col_str.startswith("p_") or "prob" in col_str or col_str == "joint_hurdle_inactivity":
                vals = np.clip(np.asarray(X_df[col], dtype=np.float64), 1e-5, 1.0 - 1e-5)
                X_df[col] = np.log(vals / (1.0 - vals))
        return X_df

    def fit(self, X: pd.DataFrame | np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "StructuralMetaClassifier":
        y_arr = np.asarray(y, dtype=int)
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        X_lr = self._to_logreg_features(X_df)
        if sample_weight is not None:
            sw = np.asarray(sample_weight, dtype="float64")
            self.lgb.fit(X_df, y_arr, sample_weight=sw)
            self.logreg.fit(X_lr, y_arr, lr__sample_weight=sw)
        else:
            self.lgb.fit(X_df, y_arr)
            self.logreg.fit(X_lr, y_arr)
        return self

    def fit_predict_oof(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
        groups: Any | None = None,
    ) -> np.ndarray:
        """Fits meta-learner and produces 100% out-of-fold probability predictions."""
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        X_lr = self._to_logreg_features(X_df)
        y_arr = np.asarray(y, dtype=int)
        sw = np.asarray(sample_weight, dtype="float64") if sample_weight is not None else None
        n = len(X_df)
        oof_probs = np.zeros((n, self.n_classes), dtype=np.float64)

        from sklearn.model_selection import GroupKFold, StratifiedKFold
        if groups is not None and len(np.unique(groups)) >= 3:
            n_splits = min(5, len(np.unique(groups)))
            cv = GroupKFold(n_splits=n_splits)
            splits = list(cv.split(X_df, y_arr, groups=groups))
        else:
            n_splits = min(5, max(2, n // 50))
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
            splits = list(cv.split(X_df, y_arr))

        from sklearn.base import clone
        for tr_idx, val_idx in splits:
            X_tr, y_tr = X_df.iloc[tr_idx], y_arr[tr_idx]
            X_v = X_df.iloc[val_idx]
            X_tr_lr = X_lr.iloc[tr_idx]
            X_v_lr = X_lr.iloc[val_idx]
            sw_tr = sw[tr_idx] if sw is not None else None

            fold_lgb = clone(self.lgb)
            fold_lr = clone(self.logreg)

            if sw_tr is not None:
                fold_lgb.fit(X_tr, y_tr, sample_weight=sw_tr)
                fold_lr.fit(X_tr_lr, y_tr, lr__sample_weight=sw_tr)
            else:
                fold_lgb.fit(X_tr, y_tr)
                fold_lr.fit(X_tr_lr, y_tr)

            p_lgb = fold_lgb.predict_proba(X_v)
            p_lr = fold_lr.predict_proba(X_v_lr)

            full_p = np.zeros((len(X_v), self.n_classes), dtype=np.float64)
            for i, cls_idx in enumerate(fold_lgb.classes_):
                if cls_idx < self.n_classes:
                    full_p[:, cls_idx] += 0.65 * p_lgb[:, i]

            lr_model = fold_lr.named_steps["lr"]
            for i, cls_idx in enumerate(lr_model.classes_):
                if cls_idx < self.n_classes:
                    full_p[:, cls_idx] += 0.35 * p_lr[:, i]

            row_s = full_p.sum(axis=1, keepdims=True)
            oof_probs[val_idx] = full_p / np.maximum(row_s, 1e-12)

        # Fit final full model on all training data for future test inference
        self.fit(X_df, y_arr, sample_weight=sw)
        return oof_probs

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        X_lr = self._to_logreg_features(X_df)
        p_lgb = self.lgb.predict_proba(X_df)
        p_lr = self.logreg.predict_proba(X_lr)

        full_probs = np.zeros((len(X_df), self.n_classes), dtype=np.float64)
        for i, cls_idx in enumerate(self.lgb.classes_):
            if cls_idx < self.n_classes:
                full_probs[:, cls_idx] += 0.65 * p_lgb[:, i]

        lr_model = self.logreg.named_steps["lr"]
        for i, cls_idx in enumerate(lr_model.classes_):
            if cls_idx < self.n_classes:
                full_probs[:, cls_idx] += 0.35 * p_lr[:, i]

        row_sums = full_probs.sum(axis=1, keepdims=True)
        return full_probs / np.maximum(row_sums, 1e-12)

