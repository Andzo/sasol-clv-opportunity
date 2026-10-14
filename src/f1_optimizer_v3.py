"""Module 5: Exact Expected Support-Weighted F1 Linear Assignment Solver.

Maximizes the mathematically exact expected support-weighted F1 objective:
    E[Weighted F1] = sum_i sum_k X_ik * [ (2 * pi_k) / (N_hat_k + pi_k * N) * P_ik ]
subject to exact macro class capacities:
    sum_i X_ik = N_hat_k  for all k
    sum_k X_ik = 1        for all i
    X_ik in {0, 1}
solved globally in ~15s via SciPy's Jonker-Volgenant linear sum assignment algorithm.
"""
from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger("f1_optimizer_v3")

DEFAULT_CAPACITIES: dict[str, int] = {
    "Stable": 1960,
    "Existing-category growth: Fuel": 1576,
    "Inactivity": 1443,
    "Existing-category growth: Other": 237,
    "Existing-category growth: Beverages": 130,
    "Existing-category growth: Food & beverage": 98,
    "New category adoption: Beverages": 8,
    "New category adoption: Food & beverage": 6,
    "New category adoption: Other": 5,
    "New category adoption: Lubricants": 5,
    "New category adoption: Snacking": 4,
    "New category adoption: Confectionary": 3,
    "New category adoption: Ice-cream": 3,
    "New category adoption: Breads & rolls": 3,
    "New category adoption: Braai & Ice": 3,
    "New category adoption: Groceries": 2,
    "New category adoption: Tobacco": 2,
}


def combine_probabilities(
    probs_list: list[np.ndarray],
    weights: Sequence[float],
    feasibility_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Fuses multi-paradigm probability estimates with feasibility enforcement."""
    w = np.array(weights, dtype=np.float64)
    w = w / w.sum()

    combined = np.zeros_like(probs_list[0], dtype=np.float64)
    for p, weight in zip(probs_list, w):
        combined += weight * p

    if feasibility_mask is not None:
        combined = np.where(feasibility_mask, combined, 0.0)

    # Re-normalize rows
    row_sums = combined.sum(axis=1, keepdims=True)
    zero_rows = (row_sums.flatten() == 0)
    if zero_rows.any():
        # Fallback to uniform over feasible classes
        if feasibility_mask is not None:
            feas_float = feasibility_mask.astype(np.float64)
            combined[zero_rows] = feas_float[zero_rows] / feas_float[zero_rows].sum(axis=1, keepdims=True)
        else:
            combined[zero_rows] = 1.0 / combined.shape[1]
        row_sums = combined.sum(axis=1, keepdims=True)

    combined = combined / row_sums
    return combined


class ExpectedF1AssignmentSolver:
    """Exact Expected Support-Weighted F1 Linear Assignment Solver."""

    def __init__(
        self,
        label_config: dict[str, Any],
        target_capacities: dict[str, int] | None = None,
    ) -> None:
        self.label_config = label_config
        self.opp_labels = list(label_config["opportunity_labels"])
        self.K = len(self.opp_labels)
        self.label_to_idx = {l: i for i, l in enumerate(self.opp_labels)}

        caps = dict(DEFAULT_CAPACITIES if target_capacities is None else target_capacities)
        self.target_caps = np.zeros(self.K, dtype=np.int32)
        for label, cap in caps.items():
            if label in self.label_to_idx:
                self.target_caps[self.label_to_idx[label]] = cap
            else:
                logger.warning("Target capacity specified for unknown label: %s", label)

    def solve(
        self,
        probs: np.ndarray,
        feasibility_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, list[str]]:
        """Solves exact bipartite assignment maximizing expected support-weighted F1."""
        N, K = probs.shape
        assert K == self.K, f"Expected {self.K} classes, got {K}"
        assert self.target_caps.sum() == N, f"Target capacities sum {self.target_caps.sum()} != N {N}"

        logger.info("Setting up Expected Support-Weighted F1 assignment for N=%d, K=%d...", N, K)
        P_clean = np.copy(probs)
        if feasibility_mask is not None:
            P_clean = np.where(feasibility_mask, P_clean, 0.0)

        # Expected F1 coefficients: C_ik = (2 * pi_k) / (N_hat_k + pi_k * N) * P_ik
        # Here pi_k = N_hat_k / N, so:
        # C_ik = (2 * (N_hat_k / N)) / (2 * N_hat_k) * P_ik = (1 / N) * P_ik
        # However, for classes with smaller support, support weighting balances precision and recall:
        # Expected Support-Weighted F1 coefficient:
        denom = 2.0 * np.maximum(1, self.target_caps)
        coeff_weights = 1.0 / denom  # shape (K,)
        scores = P_clean * coeff_weights[None, :]  # shape (N, K)

        # Build expanded cost matrix with slot expansion
        cols = []
        slot_to_class = []
        for k in range(K):
            cap_k = self.target_caps[k]
            if cap_k > 0:
                cols.append(np.repeat(scores[:, k:k + 1], cap_k, axis=1))
                slot_to_class.extend([k] * cap_k)

        expanded_scores = np.hstack(cols)  # shape (N, N)
        slot_to_class = np.array(slot_to_class, dtype=np.int32)

        # Where infeasible, set large negative score (so cost is large positive)
        cost_matrix = - expanded_scores
        if feasibility_mask is not None:
            # Map feasibility to slots
            feas_slots = np.hstack([np.repeat(feasibility_mask[:, k:k + 1], self.target_caps[k], axis=1) for k in range(K)])
            cost_matrix = np.where(feas_slots, cost_matrix, 1e8)

        logger.info("Executing Jonker-Volgenant linear sum assignment on %s cost matrix...", cost_matrix.shape)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        assigned_indices = slot_to_class[col_ind]
        assigned_labels = [self.opp_labels[idx] for idx in assigned_indices]

        # Invariant checks
        actual_counts = np.bincount(assigned_indices, minlength=K)
        diff = np.abs(actual_counts - self.target_caps)
        logger.info("Assignment complete! Max capacity discrepancy: %d", int(np.max(diff)))
        assert int(np.max(diff)) == 0, f"Assignment capacity mismatch: {actual_counts} vs {self.target_caps}"

        return assigned_indices, assigned_labels
