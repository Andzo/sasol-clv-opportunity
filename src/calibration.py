"""Post-processing: feasibility masking, decision-weight calibration, blend weights.

The legacy Nelder-Mead optimiser was effectively a no-op: weighted F1 of an argmax
decision is piecewise-constant in the weights, so the simplex never sees a gradient
(weights moved by < 1.3% in the audit). Coordinate ascent over a log-spaced grid is
used instead and is guaranteed never to decrease the in-sample objective.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
import scipy.special as sp
from scipy.optimize import minimize, nnls

DEFAULT_GRID: np.ndarray = np.geomspace(0.5, 3.5, 25)
DOMINANT_TIGHT_GRID: np.ndarray = np.geomspace(0.85, 1.15, 17)
MAJOR_GROWTH_GRID: np.ndarray = np.geomspace(0.80, 2.50, 19)
MINOR_ADOPTION_GRID: np.ndarray = np.geomspace(0.25, 1.25, 17)


def apply_eligibility_mask(probs: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Zero infeasible classes and renormalise each row to sum to 1."""
    masked = probs * mask.astype("float64")
    sums = masked.sum(axis=1, keepdims=True)
    empty = sums[:, 0] <= 0
    if empty.any():
        has_feasible = mask.sum(axis=1, keepdims=True) > 0
        fallback = np.where(has_feasible, mask.astype("float64"), 1.0)
        masked[empty] = fallback[empty]
        sums = masked.sum(axis=1, keepdims=True)
    return masked / sums


def weighted_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """Support-weighted F1 (identical to sklearn ``average='weighted'``, ``zero_division=0``)."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    support = np.bincount(y_true, minlength=n_classes).astype("float64")
    predicted = np.bincount(y_pred, minlength=n_classes).astype("float64")
    tp = np.bincount(y_true[y_true == y_pred], minlength=n_classes).astype("float64")
    denom = support + predicted
    f1 = np.divide(2.0 * tp, denom, out=np.zeros(n_classes), where=denom > 0)
    return float((f1 * support).sum() / support.sum())


def decide(probs: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Arg-max decision with optional per-class multiplicative weights."""
    return np.argmax(probs if weights is None else probs * weights, axis=1)


def apply_temperature_scaling(probs: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Scale softmax probabilities by temperature T in log space.

    T < 1.0 sharpens high-confidence probabilities;
    T > 1.0 softens/flattens probability distribution.
    Guaranteed to sum to 1.0 along axis=1.
    """
    if temperature <= 0:
        raise ValueError("Temperature must be strictly positive")
    if np.isclose(temperature, 1.0):
        return np.asarray(probs, dtype="float64")
    log_p = np.log(np.clip(probs, 1e-12, 1.0)) / float(temperature)
    log_p -= log_p.max(axis=1, keepdims=True)
    exp_p = np.exp(log_p)
    return exp_p / exp_p.sum(axis=1, keepdims=True)


def fit_class_weights(
    probs: np.ndarray,
    y_true: np.ndarray,
    grid: np.ndarray = DEFAULT_GRID,
    n_passes: int = 4,
    dominant_classes: Sequence[int] | None = None,
    major_growth_classes: Sequence[int] | None = None,
) -> np.ndarray:
    """Coordinate ascent on per-class decision weights maximising weighted F1.

    If dominant_classes is provided (e.g. Inactivity, Stable, Fuel Growth), searches
    a tighter bounded grid for dominant classes to prevent rare-category false alarms
    from cannibalizing high-support classes, while searching the broader grid for
    secondary and minority classes.
    """
    n_classes = probs.shape[1]
    weights = np.ones(n_classes)
    best = weighted_f1(y_true, decide(probs, weights), n_classes)
    dom_set = set(dominant_classes) if dominant_classes is not None else set()
    maj_set = set(major_growth_classes) if major_growth_classes is not None else set()

    for _ in range(n_passes):
        improved = False
        for c in range(n_classes):
            if c in dom_set:
                c_grid = DOMINANT_TIGHT_GRID
            elif c in maj_set:
                c_grid = MAJOR_GROWTH_GRID
            elif major_growth_classes is not None:
                c_grid = MINOR_ADOPTION_GRID
            else:
                c_grid = grid
            for value in c_grid:
                trial = weights.copy()
                trial[c] = value
                score = weighted_f1(y_true, decide(probs, trial), n_classes)
                if score > best + 1e-12:
                    best, weights, improved = score, trial, True
        if not improved:
            break
    return weights


def fit_temperature_and_class_weights(
    probs: np.ndarray,
    y_true: np.ndarray,
    mask: np.ndarray | None = None,
    temperatures: Sequence[float] = (0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10),
    dominant_classes: Sequence[int] | None = None,
    n_trials: int = 250,
    seed: int = 42,
) -> tuple[float, np.ndarray]:
    """Jointly search for optimal temperature T and class decision weights W maximizing Weighted F1.

    First evaluates coordinate ascent across the candidate temperatures to establish a strong
    warm-start baseline. If n_trials > 0, refines the joint threshold vector via Optuna's
    continuous TPESampler with dominant class protection, guaranteed never to perform worse
    than coordinate ascent.
    """
    best_f1 = -1.0
    best_temp = 1.0
    best_weights = np.ones(probs.shape[1])
    n_classes = probs.shape[1]
    dom_set = set(dominant_classes) if dominant_classes is not None else set()
    major_retail_growth_indices = [c for c in [0, 1, 3] if c < n_classes and c not in dom_set]
    minor_adoption_indices = [c for c in range(n_classes) if c not in dom_set and c not in major_retail_growth_indices]

    for temp in temperatures:
        p_scaled = apply_temperature_scaling(probs, temp)
        if mask is not None:
            p_scaled = apply_eligibility_mask(p_scaled, mask)
        cw = fit_class_weights(
            p_scaled, y_true,
            dominant_classes=dominant_classes,
            major_growth_classes=major_retail_growth_indices,
        )
        score = weighted_f1(y_true, decide(p_scaled, cw), n_classes)
        if score > best_f1 + 1e-12:
            best_f1 = score
            best_temp = float(temp)
            best_weights = cw

    if n_trials <= 0:
        return best_temp, best_weights

    # Refine via continuous TPE in collapsed, low-dimensional bounded parameter space
    # (Dominant classes individually + 1 shared multiplier for retail/adoption classes)
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    # Dominant trio (Inactivity, Stable, Fuel Growth)
    dom_set = set(dominant_classes) if dominant_classes is not None else set()
    temp_list = [float(t) for t in temperatures]

    # Major retail growth classes (Beverages, Food & beverage, Other) vs minor adoptions
    major_retail_growth_indices = [c for c in [0, 1, 3] if c < n_classes and c not in dom_set]
    minor_adoption_indices = [c for c in range(n_classes) if c not in dom_set and c not in major_retail_growth_indices]

    # Enqueue coordinate ascent solution as trial 0
    init_params = {"temperature": best_temp}
    for c in dom_set:
        init_params[f"w_{c}"] = float(best_weights[c])
    if major_retail_growth_indices:
        init_params["w_major_growth"] = float(np.median([best_weights[c] for c in major_retail_growth_indices]))
    if minor_adoption_indices:
        init_params["w_minor_adoption"] = float(np.median([best_weights[c] for c in minor_adoption_indices]))
    study.enqueue_trial(init_params)

    # Enqueue uniform weights as trial 1
    uniform_params = {"temperature": 1.0 if 1.0 in temperatures else best_temp}
    for c in dom_set:
        uniform_params[f"w_{c}"] = 1.0
    if major_retail_growth_indices:
        uniform_params["w_major_growth"] = 1.0
    if minor_adoption_indices:
        uniform_params["w_minor_adoption"] = 1.0
    study.enqueue_trial(uniform_params)

    def objective(trial: optuna.Trial) -> float:
        t = trial.suggest_categorical("temperature", temp_list)
        w = np.ones(n_classes)
        for c in dom_set:
            w[c] = trial.suggest_float(f"w_{c}", 0.80, 1.25)
        if major_retail_growth_indices:
            w_maj = trial.suggest_float("w_major_growth", 0.80, 3.50)
            for c in major_retail_growth_indices:
                w[c] = w_maj
        if minor_adoption_indices:
            w_min = trial.suggest_float("w_minor_adoption", 0.25, 1.25)
            for c in minor_adoption_indices:
                w[c] = w_min
        p_s = apply_temperature_scaling(probs, t)
        if mask is not None:
            p_s = apply_eligibility_mask(p_s, mask)
        return weighted_f1(y_true, decide(p_s, w), n_classes)

    study.optimize(objective, n_trials=n_trials)

    opt_temp = float(study.best_params["temperature"])
    opt_weights = np.ones(n_classes, dtype="float64")
    for c in dom_set:
        opt_weights[c] = float(study.best_params[f"w_{c}"])
    if major_retail_growth_indices and "w_major_growth" in study.best_params:
        opt_weights[major_retail_growth_indices] = float(study.best_params["w_major_growth"])
    if minor_adoption_indices and "w_minor_adoption" in study.best_params:
        opt_weights[minor_adoption_indices] = float(study.best_params["w_minor_adoption"])

    p_opt = apply_temperature_scaling(probs, opt_temp)
    if mask is not None:
        p_opt = apply_eligibility_mask(p_opt, mask)
    opt_score = weighted_f1(y_true, decide(p_opt, opt_weights), n_classes)

    if opt_score >= best_f1 - 1e-12:
        return opt_temp, opt_weights
    return best_temp, best_weights


def fit_blend_weights(
    member_preds: np.ndarray,
    y_true: np.ndarray,
    hurdle_indices: Sequence[int] | None = None,
    min_hurdle_weight: float = 0.0,
) -> np.ndarray:
    """Non-negative least-squares blend weights with optional hurdle weight floor constraint."""
    y_arr = np.asarray(y_true, dtype="float64")
    n_members = member_preds.shape[1]
    weights, _ = nnls(member_preds, y_arr)
    w_sum = weights.sum()
    if w_sum > 0:
        w_norm = weights / w_sum
    else:
        w_norm = np.full(n_members, 1.0 / n_members)

    if hurdle_indices is not None and min_hurdle_weight > 0.0:
        h_sum = sum(w_norm[i] for i in hurdle_indices)
        if h_sum < min_hurdle_weight:
            def loss(w: np.ndarray) -> float:
                diff = member_preds @ w - y_arr
                return float(np.mean(diff ** 2)) + 0.005 * float(np.sum(w ** 2))

            bounds = [(0.0, 1.0) for _ in range(n_members)]
            constraints = [
                {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
                {"type": "ineq", "fun": lambda w: sum(w[i] for i in hurdle_indices) - min_hurdle_weight},
            ]
            w0 = w_norm.copy()
            w_h_boost = (min_hurdle_weight - h_sum) / max(len(hurdle_indices), 1)
            for i in hurdle_indices:
                w0[i] += w_h_boost
            w0 = np.clip(w0, 0.0, 1.0)
            w0 = w0 / w0.sum()

            opt = minimize(loss, w0, method="SLSQP", bounds=bounds, constraints=constraints)
            if opt.success and np.sum(opt.x) > 0:
                w_norm = np.clip(opt.x, 0.0, 1.0)
                w_norm = w_norm / w_norm.sum()
    return w_norm


def fit_shrinkage_power(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    p_active: np.ndarray,
    powers: Sequence[float] = tuple(np.linspace(0.0, 0.7, 15)),
) -> float:
    """Fit optimal soft shrinkage power alpha minimizing RMSE on calibration fold."""
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    p_active = np.clip(np.asarray(p_active, dtype="float64"), 0.0, 1.0)
    best_power = 0.0
    best_rmse = float("inf")
    for p in powers:
        adjusted = y_pred * (p_active ** p)
        score = float(np.sqrt(np.mean((y_true - adjusted) ** 2)))
        if score < best_rmse:
            best_rmse = score
            best_power = float(p)
    return best_power


def apply_shrinkage(y_pred: np.ndarray, p_active: np.ndarray, power: float) -> np.ndarray:
    """Scale continuous regression predictions by p_active^power."""
    if power <= 0:
        return np.asarray(y_pred, dtype="float64")
    p_active = np.clip(np.asarray(p_active, dtype="float64"), 0.0, 1.0)
    return np.asarray(y_pred, dtype="float64") * (p_active ** power)


def fit_linear_calibration(
    y_true: np.ndarray, y_pred: np.ndarray, force_zero_intercept: bool = True
) -> tuple[float, float]:
    """Fit optimal slope a and intercept b minimizing MSE on calibration fold: y ~ a * y_pred + b.

    Forces intercept b = 0.0 by default to prevent zero-floor destruction on zero-inflated targets.
    """
    y_t = np.asarray(y_true, dtype="float64")
    y_p = np.asarray(y_pred, dtype="float64")
    if force_zero_intercept:
        denom = float(np.sum(y_p ** 2))
        if denom <= 1e-12:
            return 1.0, 0.0
        slope = float(np.clip(np.sum(y_t * y_p) / denom, 0.80, 1.25))
        return slope, 0.0

    res = minimize(
        lambda p: np.mean((y_t - (p[0] * y_p + p[1])) ** 2),
        x0=[1.0, 0.0],
        bounds=[(0.80, 1.25), (-0.25, 0.25)],
    )
    return float(res.x[0]), float(res.x[1])


def apply_linear_calibration(y_pred: np.ndarray, slope: float, intercept: float = 0.0) -> np.ndarray:
    """Apply slope and intercept shift with non-negativity clipping."""
    return np.clip(slope * np.asarray(y_pred, dtype="float64") + intercept, 0.0, None)



def fit_dual_shrinkage_powers(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    p_active: np.ndarray,
    p_store: np.ndarray,
    base_active_power: float | None = None,
    alpha_grid: Sequence[float] = (0.05, 0.10, 0.15, 0.20),
    beta_grid: Sequence[float] = (0.05, 0.08, 0.10, 0.12, 0.15, 0.20),
) -> tuple[float, float]:
    """Fit optimal dual soft shrinkage powers (alpha, beta) minimizing RMSE on calibration fold.

    If base_active_power is provided (e.g. churn shrinkage fitted on fuel), alpha is anchored
    to this physically derived churn discount and beta is fitted over beta_grid. This prevents
    co-linearity between p_active and p_store from choking out store attachment hurdle shrinkage.
    """
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    p_active = np.clip(np.asarray(p_active, dtype="float64"), 0.0, 1.0)
    p_store = np.clip(np.asarray(p_store, dtype="float64"), 0.0, 1.0)

    best_alpha, best_beta = 0.0, 0.0
    best_rmse = float("inf")

    alphas = (base_active_power,) if base_active_power is not None else alpha_grid

    for a in alphas:
        for b in beta_grid:
            adjusted = y_pred * (p_active ** a) * (p_store ** b)
            score = float(np.sqrt(np.mean((y_true - adjusted) ** 2)))
            if score < best_rmse:
                best_rmse = score
                best_alpha = float(a)
                best_beta = float(b)

    return best_alpha, best_beta


def apply_dual_shrinkage(
    y_pred: np.ndarray,
    p_active: np.ndarray,
    p_store: np.ndarray,
    alpha: float,
    beta: float,
) -> np.ndarray:
    """Scale continuous regression predictions by (p_active^alpha) * (p_store^beta)."""
    p_active = np.clip(np.asarray(p_active, dtype="float64"), 0.0, 1.0)
    p_store = np.clip(np.asarray(p_store, dtype="float64"), 0.0, 1.0)
    res = np.asarray(y_pred, dtype="float64")
    if alpha > 0:
        res = res * (p_active ** alpha)
    if beta > 0:
        res = res * (p_store ** beta)
    return res


def apply_inactivity_gate(
    y_pred: np.ndarray,
    p_active: np.ndarray,
    threshold: float = 0.05,
) -> np.ndarray:
    """Zero continuous predictions for customers whose active probability is strictly below threshold."""
    if threshold <= 0:
        return np.asarray(y_pred, dtype="float64")
    out = np.asarray(y_pred, dtype="float64").copy()
    out[np.asarray(p_active) < threshold] = 0.0
    return out


def fit_inactivity_threshold(
    y_fuel_true: np.ndarray,
    y_fuel_pred: np.ndarray,
    y_nf_true: np.ndarray,
    y_nf_pred: np.ndarray,
    p_active: np.ndarray,
    thresholds: Sequence[float] = (0.01, 0.02, 0.03, 0.04, 0.05, 0.08),
) -> float:
    """Find the optimal zero-gating threshold tau minimizing combined RMSE on calibration instances."""
    best_tau = 0.03
    best_loss = float("inf")
    y_fuel_true = np.asarray(y_fuel_true, dtype="float64")
    y_fuel_pred = np.asarray(y_fuel_pred, dtype="float64")
    y_nf_true = np.asarray(y_nf_true, dtype="float64")
    y_nf_pred = np.asarray(y_nf_pred, dtype="float64")
    p_active = np.asarray(p_active, dtype="float64")

    for tau in thresholds:
        f_gated = apply_inactivity_gate(y_fuel_pred, p_active, threshold=tau)
        nf_gated = apply_inactivity_gate(y_nf_pred, p_active, threshold=tau)
        loss = float(np.sqrt(np.mean((y_fuel_true - f_gated)**2)) + np.sqrt(np.mean((y_nf_true - nf_gated)**2)))
        if loss < best_loss:
            best_loss = loss
            best_tau = float(tau)

    return best_tau


def fit_isotonic_hurdle_calibration(
    y_pos_true: np.ndarray,
    y_pos_pred: np.ndarray,
) -> object:
    """Fit monotonic Isotonic Regression mapping Huber/MAE positive predictions to conditional expectation E[Z|X, Z>0]."""
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
    y_t = np.asarray(y_pos_true, dtype="float64")
    y_p = np.asarray(y_pos_pred, dtype="float64")
    iso.fit(y_p, y_t)
    return iso


def apply_isotonic_hurdle_calibration(
    y_pos_pred: np.ndarray,
    iso_model: object,
) -> np.ndarray:
    """Apply fitted isotonic recalibration model to positive spend predictions."""
    y_p = np.asarray(y_pos_pred, dtype="float64")
    return np.asarray(iso_model.predict(y_p), dtype="float64")


def fit_decile_mean_multipliers(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    p_hurdle: np.ndarray,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit monotonic decile multipliers to eliminate decile subgroup bias (y_true_bar - y_pred_bar)^2 = 0."""
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    bin_edges = np.quantile(p_hurdle, quantiles)
    bin_edges[0] = -1e-6
    bin_edges[-1] = 1.0 + 1e-6

    multipliers = np.ones(n_bins, dtype=np.float64)
    for b in range(n_bins):
        mask = (p_hurdle >= bin_edges[b]) & (p_hurdle < bin_edges[b + 1])
        if mask.sum() > 0:
            pred_mean = np.mean(y_pred[mask])
            true_mean = np.mean(y_true[mask])
            if pred_mean > 1e-6:
                multipliers[b] = np.clip(true_mean / pred_mean, 0.70, 1.40)
    return bin_edges, multipliers


def apply_decile_mean_multipliers(
    y_pred: np.ndarray,
    p_hurdle: np.ndarray,
    bin_edges: np.ndarray,
    multipliers: np.ndarray,
    target_global_mean: float | None = None,
) -> np.ndarray:
    """Apply decile multipliers and optionally re-normalize to lock the global target mean."""
    out = np.asarray(y_pred, dtype="float64").copy()
    n_bins = len(multipliers)
    for b in range(n_bins):
        mask = (p_hurdle >= bin_edges[b]) & (p_hurdle < bin_edges[b + 1])
        out[mask] *= multipliers[b]

    if target_global_mean is not None and target_global_mean > 0:
        current_mean = np.mean(out)
        if current_mean > 0:
            out *= (target_global_mean / current_mean)

    return out


def apply_soft_zero_floor(p: np.ndarray, tau_0: float = 0.025) -> np.ndarray:
    """Continuously zero out low probability tail without step-function discontinuity."""
    p_arr = np.clip(np.asarray(p, dtype="float64"), 0.0, 1.0)
    return np.maximum(0.0, (p_arr - tau_0) / (1.0 - tau_0))


def fit_member_debiasing(member_preds: np.ndarray, y_true: np.ndarray) -> list[tuple[float, float]]:
    """Fits linear recalibration (a_m, b_m) per member on positive predictions with zero intercept."""
    params: list[tuple[float, float]] = []
    n_members = member_preds.shape[1]
    y_arr = np.asarray(y_true, dtype="float64")
    for m in range(n_members):
        pred_m = member_preds[:, m]
        pos_mask = (pred_m > 0) & (y_arr > 0)
        if pos_mask.sum() >= 20:
            a, b = fit_linear_calibration(y_arr[pos_mask], pred_m[pos_mask], force_zero_intercept=True)
        else:
            a, b = 1.0, 0.0
        params.append((float(a), 0.0))
    return params



def apply_member_debiasing(member_preds: np.ndarray, debias_params: list[tuple[float, float]]) -> np.ndarray:
    """Applies per-member linear recalibration on positive predictions."""
    out = member_preds.copy()
    for m, (a, b) in enumerate(debias_params):
        pos_mask = out[:, m] > 0
        out[pos_mask, m] = np.clip(apply_linear_calibration(out[pos_mask, m], a, b), 0.0, None)
    return out


def compute_physical_hurdle_probabilities(
    X: pd.DataFrame,
    hurdle_model: Any,
    final_fuel_pred: np.ndarray,
    store_model: Any,
    norm_scale: float = 2.352825386185113,
) -> tuple[np.ndarray, np.ndarray]:
    """Computes exact probabilistic fuel hurdle inversion and joint inactivity probabilities.

    Under Rule 5 & Rule 6:
        H_fuel = max(m3_fuel_rand + 20, 1.25 * m3_fuel_rand)
        z_hurdle = (1 / norm_scale) * ln(1 + max(1.25 * m3_litres, m3_litres + 20 / P_pump))
        P_phys(Fuel Growth) = I(ever_fuel) * P_fuel * [1 - Phi((z_hurdle - (0.65*mu_pos + 0.35*Z_fuel)) / sigma_pos)]
        P_phys(Inactivity) = (1 - P_fuel) * (1 - P_store) * I(m1_baskets == 0)
    """
    n = len(X)
    m3_fuel_litres = X["m3_fuel_litres"].to_numpy() if "m3_fuel_litres" in X.columns else np.zeros(n)
    m3_fuel_rand = X["m3_fuel_rand"].to_numpy() if "m3_fuel_rand" in X.columns else np.zeros(n)
    ever_fuel = (X["ever_fuel"].to_numpy() > 0) if "ever_fuel" in X.columns else np.ones(n, dtype=bool)

    # Local pump price estimate
    pump_price = np.clip(
        np.where(m3_fuel_litres > 0.1, m3_fuel_rand / np.maximum(m3_fuel_litres, 0.1), 23.50),
        20.5, 27.5
    )

    required_litres = np.maximum(1.25 * m3_fuel_litres, m3_fuel_litres + 20.0 / pump_price)
    z_hurdle = (1.0 / norm_scale) * np.log1p(required_litres)

    if hasattr(hurdle_model, "predict_conditional_pos") and hasattr(hurdle_model, "predict_proba_active"):
        p_fuel = hurdle_model.predict_proba_active(X)
        mu_pos, sigma_pos = hurdle_model.predict_conditional_pos(X)
    else:
        p_fuel = np.clip(hurdle_model.predict(X) / np.maximum(final_fuel_pred, 1e-4), 0.0, 1.0)
        mu_pos = np.clip(final_fuel_pred / np.maximum(p_fuel, 0.05), 0.0, None)
        sigma_pos = np.maximum(0.14, 0.20 * mu_pos)

    # Probabilistic CDF inversion via standard normal erf
    z_score = (z_hurdle - (0.65 * mu_pos + 0.35 * final_fuel_pred)) / np.maximum(sigma_pos, 0.05)
    phi = 0.5 * (1.0 + sp.erf(z_score / np.sqrt(2.0)))
    p_phys_fg = np.where(ever_fuel, p_fuel * (1.0 - phi), 0.0)

    # Joint Inactivity probability
    p_store = store_model.predict_proba(X) if hasattr(store_model, "predict_proba") else np.zeros(n)
    m1_baskets = X["m1_baskets"].to_numpy() if "m1_baskets" in X.columns else np.zeros(n)
    is_m1_zero = m1_baskets == 0
    p_phys_inact = (1.0 - p_fuel) * (1.0 - p_store) * is_m1_zero.astype(float)

    return np.clip(p_phys_fg, 0.0, 1.0), np.clip(p_phys_inact, 0.0, 1.0)


def compute_customer_blup(
    customer_residuals: dict[str, list[tuple[pd.Timestamp, float]]],
    target_cutoff: pd.Timestamp,
    kappa: float = 3.5,
) -> dict[str, float]:
    """Computes recency- and season-weighted Customer Empirical Bayes (BLUP) fixed effect.

    u_i(T) = sum(w_k * r_{i,k}) / (sum(w_k) + kappa)
    for cutoffs t_k <= T - 3 months (strictly non-overlapping with outcome window).
    """
    target_month = target_cutoff.month
    cutoff_boundary = target_cutoff - pd.DateOffset(months=3)
    blup_effects: dict[str, float] = {}

    for cust_id, history in customer_residuals.items():
        w_sum = 0.0
        weighted_r_sum = 0.0
        for t_k, r_k in history:
            if t_k <= cutoff_boundary:
                w = 2.0 if t_k.month == target_month else 1.0
                w_sum += w
                weighted_r_sum += w * r_k
        if w_sum > 0:
            blup_effects[cust_id] = float(weighted_r_sum / (w_sum + kappa))
        else:
            blup_effects[cust_id] = 0.0
    return blup_effects


def fit_blup_shrinkage(
    y_true: np.ndarray,
    base_preds: np.ndarray,
    blup_vec: np.ndarray,
    active_mask: np.ndarray,
) -> float:
    """Finds optimal BLUP shrinkage eta in [0.0, 0.25] minimizing RMSE on active instances."""
    if active_mask.sum() < 20 or np.all(blup_vec == 0):
        return 0.0
    y_act = y_true[active_mask]
    base_act = base_preds[active_mask]
    blup_act = blup_vec[active_mask]
    # Zero-mean center the BLUP within active spenders
    blup_centered = blup_act - np.mean(blup_act)
    denom = np.sum(blup_centered ** 2)
    if denom <= 1e-12:
        return 0.0
    res_act = y_act - base_act
    eta = float(np.sum(res_act * blup_centered) / denom)
    return float(np.clip(eta, 0.0, 0.25))


def apply_customer_blup(
    base_preds: np.ndarray,
    blup_vec: np.ndarray,
    eta: float,
    active_mask: np.ndarray,
) -> np.ndarray:
    """Applies zero-mean centered Customer BLUP fixed effect to active spenders."""
    if eta <= 0.0 or active_mask.sum() == 0 or np.all(blup_vec == 0):
        return base_preds.copy()
    out = base_preds.copy()
    blup_act = blup_vec[active_mask]
    blup_centered = blup_act - np.mean(blup_act)
    out[active_mask] = np.clip(base_preds[active_mask] + eta * blup_centered, 0.0, None)
    return out


def decide_with_prior_alignment(
    probs: np.ndarray,
    class_weights: np.ndarray,
    target_priors: np.ndarray,
    macro_classes: tuple[int, ...] | None = None,
    tolerance: float = 0.001,
    max_steps: int = 300,
    step_size: float = 0.20,
) -> np.ndarray:
    """Prior-Constrained Expected Support-Weighted F1 Decision Rule via Dual Ascent.

    Finds dual Lagrange multipliers lambda in R^K such that macro class proportions
    align with historical target priors without threshold drift.
    """
    n, k = probs.shape
    u = np.log(probs * class_weights + 1e-12)
    lambdas = np.zeros(k, dtype=np.float64)
    target = np.asarray(target_priors, dtype=np.float64)
    target = target / target.sum()

    if macro_classes is None:
        macro_list = list(range(k))
    else:
        macro_list = [c for c in macro_classes if c < k]
    if not macro_list:
        macro_list = list(range(k))

    for _ in range(max_steps):
        pred = np.argmax(u - lambdas, axis=1)
        counts = np.bincount(pred, minlength=k) / n
        diff = counts - target
        if np.max(np.abs(diff[macro_list])) <= tolerance:
            break
        for c in macro_list:
            lambdas[c] += step_size * diff[c]

    return np.argmax(u - lambdas, axis=1)


def compute_precision_weighted_target_priors(
    raw_target_priors: np.ndarray,
    oof_f1_per_class: np.ndarray | None,
    macro_classes: tuple[int, ...] = (0, 1, 2),
) -> np.ndarray:
    """Computes precision-shrunk target prior capacities under Support-Weighted F1 Bayes decision rule.

    For low-precision minority classes (Retail and Adoption):
        r_k = clip(F1_k^OOF / F1_mean^OOF, 0.35, 1.05)
        pi_tilde_k = pi_target_k * r_k
    Redistributes remaining probability mass across macro classes (Stable, Fuel Growth, Inactivity)
    according to their historical summer prior ratio.
    """
    target = np.asarray(raw_target_priors, dtype=np.float64).copy()
    target = target / target.sum()
    k = len(target)

    if oof_f1_per_class is None:
        return target

    f1s = np.asarray(oof_f1_per_class, dtype=np.float64)
    valid_f1s = f1s[f1s > 0]
    mean_f1 = float(np.mean(valid_f1s)) if len(valid_f1s) > 0 else 0.50

    minority_classes = [c for c in range(k) if c not in macro_classes]
    shrunk_minority_mass = 0.0

    for c in minority_classes:
        f1_c = f1s[c] if c < len(f1s) else 0.30
        r_c = float(np.clip(f1_c / max(mean_f1, 1e-4), 0.35, 1.05))
        target[c] = target[c] * r_c
        shrunk_minority_mass += target[c]

    remaining_macro_mass = max(1.0 - shrunk_minority_mass, 0.50)
    raw_macro_sum = sum(raw_target_priors[c] for c in macro_classes)
    if raw_macro_sum > 0:
        for c in macro_classes:
            target[c] = remaining_macro_mass * (raw_target_priors[c] / raw_macro_sum)

    return target / target.sum()


def arbitrate_commuter_pool(
    opp_labels: np.ndarray,
    features: pd.DataFrame,
    pred_fuel: np.ndarray,
    p_phys_fg: np.ndarray,
    p_clf_fg: np.ndarray,
    p_clf_stable: np.ndarray,
    target_fg_capacity: int,
    weights: tuple[float, float, float] = (0.40, 0.35, 0.25),
    ever_fuel_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Direct Physical Fuel Hurdle Arbitration between Existing-category growth: Fuel and Stable.

    Computes unified Calibrated Physical Fuel Growth Score:
        Score_i = w1 * rank_norm(R_phys_i) + w2 * P_exceed_i^fuel + w3 * P_clf(FG | Commuter)
    and rank-assigns exactly target_fg_capacity to Fuel Growth.
    """
    labels = np.asarray(opp_labels, dtype=object).copy()
    n = len(labels)
    fg_label = "Existing-category growth: Fuel"
    st_label = "Stable"

    is_commuter = (labels == fg_label) | (labels == st_label)
    commuter_indices = np.where(is_commuter)[0]
    if len(commuter_indices) == 0:
        return labels

    m3_fuel_litres = features["m3_fuel_litres"].to_numpy() if "m3_fuel_litres" in features.columns else np.zeros(n)
    m3_fuel_rand = features["m3_fuel_rand"].to_numpy() if "m3_fuel_rand" in features.columns else np.zeros(n)
    m1_fuel_baskets = features["m1_fuel_baskets"].to_numpy() if "m1_fuel_baskets" in features.columns else (
        features["m1_baskets"].to_numpy() if "m1_baskets" in features.columns else np.zeros(n)
    )
    days_since_fuel = features["days_since_last_fuel"].to_numpy() if "days_since_last_fuel" in features.columns else np.full(n, 999.0)

    if ever_fuel_mask is not None:
        ever_fuel = np.asarray(ever_fuel_mask, dtype=bool)
    elif "active_m6_fuel" in features.columns:
        ever_fuel = features["active_m6_fuel"].to_numpy() > 0
    elif "ever_fuel" in features.columns:
        ever_fuel = features["ever_fuel"].to_numpy() > 0
    else:
        ever_fuel = np.ones(n, dtype=bool)

    litres_pred = np.expm1(np.minimum(pred_fuel * 2.352825386185113, 15.0))
    pump_price = np.clip(
        np.where(m3_fuel_litres > 0.1, m3_fuel_rand / np.maximum(m3_fuel_litres, 0.1), 23.50),
        20.5, 27.5,
    )
    h_litres = np.maximum(1.25 * m3_fuel_litres, m3_fuel_litres + 20.0 / pump_price)
    r_phys = (litres_pred + 0.5) / (h_litres + 0.5)

    from scipy.stats import rankdata
    r_phys_commuter = r_phys[commuter_indices]
    rank_r_phys = rankdata(r_phys_commuter) / len(r_phys_commuter)

    p_exceed_commuter = p_phys_fg[commuter_indices]

    denom_clf = p_clf_fg[commuter_indices] + p_clf_stable[commuter_indices] + 1e-12
    p_cond_commuter = p_clf_fg[commuter_indices] / denom_clf

    w1, w2, w3 = weights
    blended_score = w1 * rank_r_phys + w2 * p_exceed_commuter + w3 * p_cond_commuter

    dormant_penalty = (
        (~ever_fuel[commuter_indices])
        | ((m3_fuel_litres[commuter_indices] <= 5.0) & (m1_fuel_baskets[commuter_indices] == 0) & (days_since_fuel[commuter_indices] > 90))
    )
    blended_score[dormant_penalty] = -1.0

    k_target = int(np.clip(target_fg_capacity, 0, len(commuter_indices)))
    sorted_order = np.argsort(-blended_score)
    top_fg_indices = commuter_indices[sorted_order[:k_target]]
    stable_indices = commuter_indices[sorted_order[k_target:]]

    labels[top_fg_indices] = fg_label
    labels[stable_indices] = st_label
    return labels



