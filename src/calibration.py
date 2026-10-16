"""Post-processing: feasibility masking, decision-weight calibration, blend weights.

The legacy Nelder-Mead optimiser was effectively a no-op: weighted F1 of an argmax
decision is piecewise-constant in the weights, so the simplex never sees a gradient
(weights moved by < 1.3% in the audit). Coordinate ascent over a log-spaced grid is
used instead and is guaranteed never to decrease the in-sample objective.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.optimize import nnls

DEFAULT_GRID: np.ndarray = np.geomspace(0.5, 12.0, 45)
DOMINANT_TIGHT_GRID: np.ndarray = np.geomspace(0.85, 1.15, 17)


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

    for _ in range(n_passes):
        improved = False
        for c in range(n_classes):
            c_grid = DOMINANT_TIGHT_GRID if c in dom_set else grid
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

    for temp in temperatures:
        p_scaled = apply_temperature_scaling(probs, temp)
        if mask is not None:
            p_scaled = apply_eligibility_mask(p_scaled, mask)
        cw = fit_class_weights(p_scaled, y_true, dominant_classes=dominant_classes)
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

    dom_set = set(dominant_classes) if dominant_classes is not None else set()
    temp_list = [float(t) for t in temperatures]

    # Enqueue coordinate ascent solution as trial 0
    init_params = {"temperature": best_temp}
    for c in dom_set:
        init_params[f"w_{c}"] = float(best_weights[c])
    retail_indices = [c for c in range(n_classes) if c not in dom_set]
    if retail_indices:
        init_params["w_retail_shared"] = float(np.median([best_weights[c] for c in retail_indices]))
    study.enqueue_trial(init_params)

    # Enqueue uniform weights as trial 1
    uniform_params = {"temperature": 1.0 if 1.0 in temperatures else best_temp}
    for c in dom_set:
        uniform_params[f"w_{c}"] = 1.0
    if retail_indices:
        uniform_params["w_retail_shared"] = 1.0
    study.enqueue_trial(uniform_params)

    def objective(trial: optuna.Trial) -> float:
        t = trial.suggest_categorical("temperature", temp_list)
        w = np.ones(n_classes)
        for c in dom_set:
            w[c] = trial.suggest_float(f"w_{c}", 0.80, 1.25)
        if retail_indices:
            w_ret = trial.suggest_float("w_retail_shared", 0.50, 12.0)
            for c in retail_indices:
                w[c] = w_ret
        p_s = apply_temperature_scaling(probs, t)
        if mask is not None:
            p_s = apply_eligibility_mask(p_s, mask)
        return weighted_f1(y_true, decide(p_s, w), n_classes)

    study.optimize(objective, n_trials=n_trials)

    opt_temp = float(study.best_params["temperature"])
    opt_weights = np.ones(n_classes, dtype="float64")
    for c in dom_set:
        opt_weights[c] = float(study.best_params[f"w_{c}"])
    if retail_indices and "w_retail_shared" in study.best_params:
        opt_weights[retail_indices] = float(study.best_params["w_retail_shared"])

    p_opt = apply_temperature_scaling(probs, opt_temp)
    if mask is not None:
        p_opt = apply_eligibility_mask(p_opt, mask)
    opt_score = weighted_f1(y_true, decide(p_opt, opt_weights), n_classes)

    if opt_score >= best_f1 - 1e-12:
        return opt_temp, opt_weights
    return best_temp, best_weights


def fit_blend_weights(member_preds: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Non-negative least-squares blend weights (normalised); uniform on degenerate fits."""
    weights, _ = nnls(member_preds, np.asarray(y_true, dtype="float64"))
    if weights.sum() <= 0:
        return np.full(member_preds.shape[1], 1.0 / member_preds.shape[1])
    return weights / weights.sum()


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


def fit_linear_calibration(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """Fit optimal slope a and intercept b minimizing MSE on calibration fold: y ~ a * y_pred + b."""
    from scipy.optimize import minimize
    y_t = np.asarray(y_true, dtype="float64")
    y_p = np.asarray(y_pred, dtype="float64")
    res = minimize(
        lambda p: np.mean((y_t - (p[0] * y_p + p[1])) ** 2),
        x0=[1.0, 0.0],
        bounds=[(0.80, 1.25), (-0.25, 0.25)],
    )
    return float(res.x[0]), float(res.x[1])


def apply_linear_calibration(y_pred: np.ndarray, slope: float, intercept: float) -> np.ndarray:
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


