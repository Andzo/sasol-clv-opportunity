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

DEFAULT_GRID: np.ndarray = np.geomspace(0.5, 2.5, 33)
DOMINANT_TIGHT_GRID: np.ndarray = np.geomspace(0.8, 1.25, 17)


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


