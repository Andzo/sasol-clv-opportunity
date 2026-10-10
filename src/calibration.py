"""Post-processing: feasibility masking, decision-weight calibration, blend weights.

The legacy Nelder-Mead optimiser was effectively a no-op: weighted F1 of an argmax
decision is piecewise-constant in the weights, so the simplex never sees a gradient
(weights moved by < 1.3% in the audit). Coordinate ascent over a log-spaced grid is
used instead and is guaranteed never to decrease the in-sample objective.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import nnls

DEFAULT_GRID: np.ndarray = np.geomspace(0.25, 6.0, 23)


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
    n_passes: int = 3,
) -> np.ndarray:
    """Coordinate ascent on per-class decision weights maximising weighted F1.

    Deterministic; returns all-ones if no improvement is found.
    """
    n_classes = probs.shape[1]
    weights = np.ones(n_classes)
    best = weighted_f1(y_true, decide(probs, weights), n_classes)
    for _ in range(n_passes):
        improved = False
        for c in range(n_classes):
            for value in grid:
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
