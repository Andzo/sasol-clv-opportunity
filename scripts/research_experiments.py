"""Research experiments and benchmark evaluation script.

Runs iterative experiments to investigate:
1. Error distributions and bottleneck analysis for Fuel, Non-fuel, and Opportunity.
2. Feature engineering extensions (trend lags, momentum, reversion, hurdle metrics).
3. Tri-model ensembling and hyperparameter tuning.
4. Soft active shrinkage and constrained probability calibration.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import classification_report, f1_score, mean_squared_error

from src.config import PipelineConfig
from src.data import load_transactions
from src.dataset import SnapshotBuilder, stack
from src.models import RegressorEnsemble, ClassifierEnsemble
from src import calibration, metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("research")


def build_snapshots(cfg: PipelineConfig):
    logger.info("Loading transactions and building snapshots...")
    tx = load_transactions(cfg)
    builder = SnapshotBuilder(cfg, tx)
    eval_snap = builder.build("2025-09-01", with_labels=True)
    cal_snap = builder.build("2025-06-01", with_labels=True)
    train_snaps = [builder.build(c, with_labels=True) for c in cfg.training_cutoffs(boundary="2025-06-01")]
    train_panel = stack(train_snaps)
    return train_panel, cal_snap, eval_snap


def run_error_breakdown(cfg: PipelineConfig) -> None:
    train_panel, cal_snap, eval_snap = build_snapshots(cfg)
    logger.info("Stacked train: %d x %d", len(train_panel.X), train_panel.X.shape[1])
    logger.info("Cal: %d, Eval: %d", len(cal_snap.features), len(eval_snap.features))

    # Regressors
    fuel_model = RegressorEnsemble(("lgbm", "catboost", "xgb"), seed=42, n_jobs=-1).fit(
        train_panel.X, train_panel.labels["CLV_fuel"].to_numpy()
    )
    nf_model = RegressorEnsemble(("lgbm", "catboost", "xgb"), seed=42, n_jobs=-1).fit(
        train_panel.X, train_panel.labels["CLV_nonfuel"].to_numpy()
    )

    pred_fuel = fuel_model.predict(eval_snap.features)
    pred_nf = nf_model.predict(eval_snap.features)
    labels = eval_snap.labels

    y_f = labels["CLV_fuel"].to_numpy()
    y_nf = labels["CLV_nonfuel"].to_numpy()
    inact = (labels["Opportunity"] == "Inactivity").to_numpy()

    logger.info("=== Fuel Error Breakdown ===")
    logger.info("Overall RMSE: %.4f", np.sqrt(np.mean((y_f - pred_fuel) ** 2)))
    logger.info("Inactive subset RMSE: %.4f (Mean pred: %.4f vs true 0.0)",
                np.sqrt(np.mean((y_f[inact] - pred_fuel[inact]) ** 2)), pred_fuel[inact].mean())
    logger.info("Active subset RMSE:   %.4f (Mean pred: %.4f vs true: %.4f)",
                np.sqrt(np.mean((y_f[~inact] - pred_fuel[~inact]) ** 2)), pred_fuel[~inact].mean(), y_f[~inact].mean())

    logger.info("=== Non-fuel Error Breakdown ===")
    logger.info("Overall RMSE: %.4f", np.sqrt(np.mean((y_nf - pred_nf) ** 2)))
    logger.info("Zero-spend subset RMSE:     %.4f (Count: %d)",
                np.sqrt(np.mean((y_nf[y_nf == 0] - pred_nf[y_nf == 0]) ** 2)), np.sum(y_nf == 0))
    logger.info("Positive-spend subset RMSE: %.4f (Count: %d)",
                np.sqrt(np.mean((y_nf[y_nf > 0] - pred_nf[y_nf > 0]) ** 2)), np.sum(y_nf > 0))


def run_benchmark(cfg: PipelineConfig) -> dict[str, float]:
    train_panel, cal_snap, eval_snap = build_snapshots(cfg)
    opp_labels = cfg.label_config["opportunity_labels"]
    n_classes = len(opp_labels)
    label_to_idx = {l: i for i, l in enumerate(opp_labels)}
    inact_idx = label_to_idx["Inactivity"]

    logger.info("--- Fitting Tuned Regressors ---")
    fuel_lgbm = LGBMRegressor(n_estimators=800, learning_rate=0.02, num_leaves=31, min_child_samples=50,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=10.0,
                              random_state=42, n_jobs=-1, deterministic=True, force_col_wise=True, verbose=-1)
    fuel_cb = CatBoostRegressor(iterations=1000, learning_rate=0.04, depth=6, l2_leaf_reg=5.0, loss_function="RMSE",
                                random_seed=42, thread_count=-1, verbose=0, allow_writing_files=False)
    fuel_xgb = XGBRegressor(n_estimators=700, learning_rate=0.02, max_depth=5, min_child_weight=5.0,
                            subsample=0.8, colsample_bytree=0.5, reg_lambda=10.0, tree_method="hist",
                            random_state=42, n_jobs=-1)

    fuel_lgbm.fit(train_panel.X, train_panel.labels["CLV_fuel"])
    fuel_cb.fit(train_panel.X, train_panel.labels["CLV_fuel"])
    fuel_xgb.fit(train_panel.X, train_panel.labels["CLV_fuel"])

    # Fuel calibration
    cal_f_m = np.column_stack([
        np.clip(fuel_lgbm.predict(cal_snap.features), 0, None),
        np.clip(fuel_cb.predict(cal_snap.features), 0, None),
        np.clip(fuel_xgb.predict(cal_snap.features), 0, None),
    ])
    w_fuel = calibration.fit_blend_weights(cal_f_m, cal_snap.labels["CLV_fuel"].to_numpy())
    logger.info("Fuel Blend Weights (LGBM, CB, XGB): %s", np.round(w_fuel, 3))

    nf_lgbm = LGBMRegressor(n_estimators=800, learning_rate=0.02, num_leaves=31, min_child_samples=50,
                            subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=10.0,
                            random_state=42, n_jobs=-1, deterministic=True, force_col_wise=True, verbose=-1)
    nf_cb = CatBoostRegressor(iterations=1000, learning_rate=0.04, depth=6, l2_leaf_reg=5.0, loss_function="RMSE",
                              random_seed=42, thread_count=-1, verbose=0, allow_writing_files=False)
    nf_xgb = XGBRegressor(n_estimators=700, learning_rate=0.02, max_depth=5, min_child_weight=5.0,
                          subsample=0.8, colsample_bytree=0.5, reg_lambda=10.0, tree_method="hist",
                          random_state=42, n_jobs=-1)

    nf_lgbm.fit(train_panel.X, train_panel.labels["CLV_nonfuel"])
    nf_cb.fit(train_panel.X, train_panel.labels["CLV_nonfuel"])
    nf_xgb.fit(train_panel.X, train_panel.labels["CLV_nonfuel"])

    cal_nf_m = np.column_stack([
        np.clip(nf_lgbm.predict(cal_snap.features), 0, None),
        np.clip(nf_cb.predict(cal_snap.features), 0, None),
        np.clip(nf_xgb.predict(cal_snap.features), 0, None),
    ])
    w_nf = calibration.fit_blend_weights(cal_nf_m, cal_snap.labels["CLV_nonfuel"].to_numpy())
    logger.info("Non-fuel Blend Weights (LGBM, CB, XGB): %s", np.round(w_nf, 3))

    logger.info("--- Fitting Tuned Opportunity Classifiers ---")
    y_train_opp = train_panel.labels["Opportunity"].map(label_to_idx).to_numpy()
    classes_in_train = np.unique(y_train_opp)

    clf_lgbm = LGBMClassifier(objective="multiclass", n_estimators=500, learning_rate=0.02, num_leaves=31,
                              min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.6,
                              reg_lambda=5.0, random_state=42, n_jobs=-1, verbose=-1)
    clf_cb = CatBoostClassifier(iterations=800, learning_rate=0.05, depth=6, loss_function="MultiClass",
                                random_seed=42, thread_count=-1, verbose=0, allow_writing_files=False)
    clf_xgb = XGBClassifier(objective="multi:softprob", n_estimators=400, learning_rate=0.04, max_depth=5,
                            subsample=0.8, colsample_bytree=0.6, tree_method="hist", random_state=42, n_jobs=-1)

    clf_lgbm.fit(train_panel.X, y_train_opp)
    clf_cb.fit(train_panel.X, y_train_opp)
    clf_xgb.fit(train_panel.X, np.searchsorted(classes_in_train, y_train_opp))

    def get_probs(X):
        p_l = clf_lgbm.predict_proba(X)
        p_c = clf_cb.predict_proba(X)
        p_x = np.zeros((len(X), n_classes))
        p_x[:, classes_in_train] = clf_xgb.predict_proba(X)
        avg = (p_l + p_c + p_x) / 3.0
        return avg / avg.sum(axis=1, keepdims=True)

    cal_probs = calibration.apply_eligibility_mask(get_probs(cal_snap.features), cal_snap.mask)
    cal_y = cal_snap.labels["Opportunity"].map(label_to_idx).to_numpy()
    cw = calibration.fit_class_weights(cal_probs, cal_y, grid=np.geomspace(0.6, 1.8, 13))

    # Shrinkage power fitting on cal fold
    p_act_cal = 1.0 - cal_probs[:, inact_idx]
    cal_f_pred = cal_f_m @ w_fuel
    cal_nf_pred = cal_nf_m @ w_nf

    def fit_pow(y_true, y_pred, p_act):
        best_p, best_s = 0.0, float("inf")
        for p in np.linspace(0.0, 0.7, 15):
            s = float(np.sqrt(np.mean((y_true - y_pred * (p_act**p))**2)))
            if s < best_s:
                best_s, best_p = s, p
        return best_p

    pow_fuel = fit_pow(cal_snap.labels["CLV_fuel"].to_numpy(), cal_f_pred, p_act_cal)
    pow_nf = fit_pow(cal_snap.labels["CLV_nonfuel"].to_numpy(), cal_nf_pred, p_act_cal)
    logger.info("Optimal Calibrated Shrinkage Powers: Fuel=%.2f, Nonfuel=%.2f", pow_fuel, pow_nf)

    # Evaluation fold inference
    eval_f_m = np.column_stack([
        np.clip(fuel_lgbm.predict(eval_snap.features), 0, None),
        np.clip(fuel_cb.predict(eval_snap.features), 0, None),
        np.clip(fuel_xgb.predict(eval_snap.features), 0, None),
    ])
    eval_f_raw = eval_f_m @ w_fuel

    eval_nf_m = np.column_stack([
        np.clip(nf_lgbm.predict(eval_snap.features), 0, None),
        np.clip(nf_cb.predict(eval_snap.features), 0, None),
        np.clip(nf_xgb.predict(eval_snap.features), 0, None),
    ])
    eval_nf_raw = eval_nf_m @ w_nf

    eval_probs = calibration.apply_eligibility_mask(get_probs(eval_snap.features), eval_snap.mask)
    p_act_eval = 1.0 - eval_probs[:, inact_idx]

    eval_f_final = eval_f_raw * (p_act_eval ** pow_fuel)
    eval_nf_final = eval_nf_raw * (p_act_eval ** pow_nf)
    eval_dec = calibration.decide(eval_probs, cw)
    eval_opp = [opp_labels[i] for i in eval_dec]

    scores = metrics.score(eval_snap.labels, eval_f_final, eval_nf_final, np.array(eval_opp))
    x_max_f = 1.8344871663064635
    x_max_nf = 1.2240635969972369
    composite = scores.composite(x_max_f, x_max_nf)

    logger.info("=== EVALUATION BENCHMARK RESULTS ===")
    logger.info("CLV_fuel RMSE:    %.4f (Norm: %.4f)", scores.rmse_fuel, 1 - scores.rmse_fuel / x_max_f)
    logger.info("CLV_nonfuel RMSE: %.4f (Norm: %.4f)", scores.rmse_nonfuel, 1 - scores.rmse_nonfuel / x_max_nf)
    logger.info("Weighted F1:      %.4f", scores.weighted_f1)
    logger.info("Proxy Composite:  %.4f", composite)

    return {
        "rmse_fuel": scores.rmse_fuel,
        "rmse_nonfuel": scores.rmse_nonfuel,
        "weighted_f1": scores.weighted_f1,
        "proxy_composite": composite,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="benchmark", choices=["diagnostics", "error_breakdown", "benchmark"])
    args = parser.parse_args()
    cfg = PipelineConfig()
    if args.mode == "error_breakdown":
        run_error_breakdown(cfg)
    elif args.mode == "benchmark":
        run_benchmark(cfg)
