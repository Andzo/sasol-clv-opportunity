"""Post-processing and metric calibration for Sasol CLV & Opportunity challenge."""
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import f1_score, root_mean_squared_error


def apply_eligibility_mask(probs, mask):
    """
    Zeroes out impossible classes using the historical eligibility mask
    and renormalizes probabilities.
    """
    masked_probs = probs * mask.astype(float)
    # Avoid division by zero: if all masked (should never happen), fall back to uniform over mask
    row_sums = masked_probs.sum(axis=1, keepdims=True)
    zero_mask = (row_sums == 0).flatten()
    if zero_mask.any():
        masked_probs[zero_mask] = mask[zero_mask].astype(float)
        row_sums = masked_probs.sum(axis=1, keepdims=True)
    return masked_probs / row_sums


def optimize_class_weights_for_weighted_f1(probs, y_true_int, num_classes=17, max_iters=200):
    """
    Optimizes class multipliers w to maximize Weighted F1:
    pred = argmax_c (w_c * prob_c).
    """
    init_weights = np.ones(num_classes)

    def objective(w):
        scaled_probs = probs * w
        preds = np.argmax(scaled_probs, axis=1)
        score = f1_score(y_true_int, preds, average='weighted', zero_division=0)
        return -score  # Minimize negative F1

    res = minimize(
        objective,
        init_weights,
        method='Nelder-Mead',
        options={'maxiter': max_iters, 'disp': False}
    )
    best_weights = res.x
    best_weights = np.clip(best_weights, 0.01, 100.0)
    return best_weights


def tune_zero_threshold(preds, y_true, lower=0.0, upper=0.5, steps=50):
    """
    Finds the threshold tau such that setting preds < tau to 0 minimizes RMSE.
    """
    best_tau = 0.0
    best_rmse = root_mean_squared_error(y_true, preds)

    for tau in np.linspace(lower, upper, steps):
        cur_preds = np.where(preds < tau, 0.0, preds)
        rmse = root_mean_squared_error(y_true, cur_preds)
        if rmse < best_rmse:
            best_rmse = rmse
            best_tau = tau

    return best_tau
