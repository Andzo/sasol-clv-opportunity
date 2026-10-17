"""Unit tests for calibration and post-processing algorithms."""
import numpy as np
import pandas as pd
from src.calibration import (
    apply_eligibility_mask,
    apply_inactivity_gate,
    apply_temperature_scaling,
    decide,
    fit_blend_weights,
    fit_class_weights,
    fit_inactivity_threshold,
    fit_temperature_and_class_weights,
    weighted_f1,
)


def test_apply_eligibility_mask():
    probs = np.array([
        [0.4, 0.4, 0.2],
        [0.1, 0.8, 0.1],
    ])
    mask = np.array([
        [True, False, True],  # class 1 masked
        [False, False, False], # all masked -> fallback uniform
    ])
    masked = apply_eligibility_mask(probs, mask)
    
    # Row 0: class 1 zeroed, remaining renormalized
    assert masked[0, 1] == 0.0
    assert np.isclose(masked[0, 0], 0.4 / 0.6)
    assert np.isclose(masked[0, 2], 0.2 / 0.6)
    assert np.isclose(masked[0].sum(), 1.0)

    # Row 1: fallback uniform over mask (all false -> uniform)
    assert np.isclose(masked[1].sum(), 1.0)


def test_fit_class_weights_improves_or_maintains_f1():
    rng = np.random.default_rng(42)
    n_samples, n_classes = 500, 5
    probs = rng.dirichlet(np.ones(n_classes), size=n_samples)
    y_true = rng.integers(0, n_classes, size=n_samples)

    init_preds = decide(probs)
    init_f1 = weighted_f1(y_true, init_preds, n_classes)

    weights = fit_class_weights(probs, y_true, n_passes=2)
    cal_preds = decide(probs, weights)
    cal_f1 = weighted_f1(y_true, cal_preds, n_classes)

    assert cal_f1 >= init_f1 - 1e-9


def test_apply_temperature_scaling():
    probs = np.array([
        [0.7, 0.2, 0.1],
        [0.34, 0.33, 0.33],
    ])
    # T = 1.0 is identity
    assert np.allclose(apply_temperature_scaling(probs, 1.0), probs)

    # T < 1.0 sharpens mode
    sharp = apply_temperature_scaling(probs, 0.5)
    assert sharp[0, 0] > probs[0, 0]
    assert np.allclose(sharp.sum(axis=1), 1.0)

    # T > 1.0 softens mode
    soft = apply_temperature_scaling(probs, 2.0)
    assert soft[0, 0] < probs[0, 0]
    assert np.allclose(soft.sum(axis=1), 1.0)


def test_fit_temperature_and_class_weights():
    rng = np.random.default_rng(42)
    n_samples, n_classes = 300, 4
    probs = rng.dirichlet(np.ones(n_classes), size=n_samples)
    y_true = rng.integers(0, n_classes, size=n_samples)

    # 1. Warm-start baseline (n_trials=0)
    base_T, base_cw = fit_temperature_and_class_weights(
        probs, y_true, temperatures=[0.85, 0.90, 1.00], dominant_classes=[0], n_trials=0
    )
    p_base = apply_temperature_scaling(probs, base_T)
    base_f1 = weighted_f1(y_true, decide(p_base, base_cw), n_classes)

    # 2. Optuna TPE refinement (n_trials=50 for fast unit test)
    best_T, cw = fit_temperature_and_class_weights(
        probs, y_true, temperatures=[0.85, 0.90, 1.00], dominant_classes=[0], n_trials=50, seed=42
    )
    assert best_T in [0.85, 0.90, 1.00]
    assert len(cw) == n_classes
    assert (cw >= 0).all()
    # Dominant class bounded
    assert 0.80 <= cw[0] <= 1.25

    p_opt = apply_temperature_scaling(probs, best_T)
    opt_f1 = weighted_f1(y_true, decide(p_opt, cw), n_classes)
    # Guaranteed never to be worse than warm-start baseline
    assert opt_f1 >= base_f1 - 1e-9

    # 3. Determinism check
    best_T2, cw2 = fit_temperature_and_class_weights(
        probs, y_true, temperatures=[0.85, 0.90, 1.00], dominant_classes=[0], n_trials=50, seed=42
    )
    assert best_T == best_T2
    assert np.allclose(cw, cw2)


def test_fit_inactivity_threshold():
    y_true = np.array([0.0, 0.0, 1.5, 2.0])
    y_pred = np.array([0.1, 0.05, 1.4, 2.1])
    p_act = np.array([0.02, 0.04, 0.8, 0.9])

    tau = fit_inactivity_threshold(
        y_true, y_pred, y_true, y_pred, p_act, thresholds=[0.01, 0.03, 0.05]
    )
    assert tau in [0.01, 0.03, 0.05]


def test_fit_blend_weights():
    # Synthetic member predictions
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    member_preds = np.column_stack([
        y_true + 0.1,
        y_true - 0.2,
        np.zeros(4)
    ])
    w = fit_blend_weights(member_preds, y_true)
    assert len(w) == 3
    assert np.isclose(w.sum(), 1.0)
    assert (w >= 0).all()
    # Member 2 (zeros) should receive near-zero weight
    assert w[2] < 0.1


def test_shrinkage_calibration():
    from src.calibration import fit_shrinkage_power, apply_shrinkage

    y_true = np.array([2.0, 1.5, 0.0, 0.0, 0.0])
    y_pred = np.array([2.1, 1.4, 0.8, 0.7, 0.9])
    p_active = np.array([0.95, 0.90, 0.10, 0.05, 0.15])

    power = fit_shrinkage_power(y_true, y_pred, p_active)
    assert power > 0.0  # Should learn positive shrinkage to penalize inactive false positives

    shrunk = apply_shrinkage(y_pred, p_active, power)
    init_rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    shrunk_rmse = np.sqrt(np.mean((y_true - shrunk) ** 2))
    assert shrunk_rmse < init_rmse


def test_dual_shrinkage_calibration():
    from src.calibration import fit_dual_shrinkage_powers, apply_dual_shrinkage

    y_true = np.array([2.0, 1.5, 0.0, 0.0, 0.0])
    y_pred = np.array([2.1, 1.4, 0.8, 0.7, 0.9])
    p_active = np.array([0.95, 0.90, 0.10, 0.05, 0.15])
    p_store = np.array([0.90, 0.85, 0.15, 0.10, 0.20])

    alpha, beta = fit_dual_shrinkage_powers(y_true, y_pred, p_active, p_store)
    assert alpha >= 0.0
    assert beta >= 0.0
    assert (alpha + beta) > 0.0

    shrunk = apply_dual_shrinkage(y_pred, p_active, p_store, alpha, beta)
    init_rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    shrunk_rmse = np.sqrt(np.mean((y_true - shrunk) ** 2))
    assert shrunk_rmse < init_rmse

    # Anchored alpha
    a_anchored, b_anchored = fit_dual_shrinkage_powers(
        y_true, y_pred, p_active, p_store, base_active_power=0.15
    )
    assert np.isclose(a_anchored, 0.15)
    assert b_anchored >= 0.0



def test_store_attachment_classifier():
    import pandas as pd
    from src.models import StoreAttachmentClassifier

    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(100, 5)), columns=[f"f{i}" for i in range(5)])
    # Continuous nonfuel labels with zeros
    y = np.clip(rng.normal(loc=0.0, scale=1.0, size=100), 0.0, None)

    clf = StoreAttachmentClassifier(seed=42, n_jobs=1)
    clf.fit(X, y)
    probs = clf.predict_proba(X)

    assert probs.shape == (100,)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_inactivity_gate():
    from src.calibration import apply_inactivity_gate

    y_pred = np.array([2.0, 1.5, 0.8, 0.4])
    p_active = np.array([0.9, 0.5, 0.04, 0.01])

    gated = apply_inactivity_gate(y_pred, p_active, threshold=0.05)
    assert np.isclose(gated[0], 2.0)
    assert np.isclose(gated[1], 1.5)
    assert gated[2] == 0.0
    assert gated[3] == 0.0


def test_fit_class_weights_with_dominant_classes():
    from src.calibration import fit_class_weights, decide, weighted_f1

    rng = np.random.default_rng(42)
    n_samples, n_classes = 400, 5
    probs = rng.dirichlet(np.ones(n_classes), size=n_samples)
    y_true = rng.integers(0, n_classes, size=n_samples)

    weights = fit_class_weights(probs, y_true, dominant_classes=[0, 1])
    assert len(weights) == n_classes
    assert (weights > 0).all()
    # Should improve or maintain F1
    base_f1 = weighted_f1(y_true, decide(probs), n_classes)
    cal_f1 = weighted_f1(y_true, decide(probs, weights), n_classes)
    assert cal_f1 >= base_f1 - 1e-9


def test_pooled_multi_cutoff_calibration():
    """Verify that calibration routines correctly accept pooled multi-snapshot arrays."""
    from src.calibration import (
        fit_blend_weights,
        fit_class_weights,
        fit_shrinkage_power,
        fit_dual_shrinkage_powers,
        apply_shrinkage,
        apply_dual_shrinkage,
        decide,
    )

    rng = np.random.default_rng(42)
    # Simulate 2 snapshots (e.g. 500 rows and 600 rows)
    n1, n2, n_members, n_classes = 500, 600, 3, 5

    # Pooled targets
    y_reg_1 = rng.uniform(0.0, 3.0, size=n1)
    y_reg_2 = rng.uniform(0.0, 3.0, size=n2)
    y_reg_pool = np.concatenate([y_reg_1, y_reg_2])

    y_cls_1 = rng.integers(0, n_classes, size=n1)
    y_cls_2 = rng.integers(0, n_classes, size=n2)
    y_cls_pool = np.concatenate([y_cls_1, y_cls_2])

    # Pooled regressor predictions
    preds_1 = np.column_stack([y_reg_1 + rng.normal(0, 0.1, n1) for _ in range(n_members)])
    preds_2 = np.column_stack([y_reg_2 + rng.normal(0, 0.1, n2) for _ in range(n_members)])
    preds_pool = np.vstack([preds_1, preds_2])

    # Blend weights
    blend = fit_blend_weights(preds_pool, y_reg_pool)
    assert len(blend) == n_members
    assert np.isclose(blend.sum(), 1.0)

    # Class weights on pooled probabilities
    probs_1 = rng.dirichlet(np.ones(n_classes), size=n1)
    probs_2 = rng.dirichlet(np.ones(n_classes), size=n2)
    probs_pool = np.vstack([probs_1, probs_2])
    cw = fit_class_weights(probs_pool, y_cls_pool, dominant_classes=[0, 1])
    assert len(cw) == n_classes
    decisions = decide(probs_pool, cw)
    assert decisions.shape == (n1 + n2,)

    # Shrinkage powers on pooled arrays
    p_act_pool = rng.uniform(0.2, 1.0, size=n1 + n2)
    p_store_pool = rng.uniform(0.1, 0.9, size=n1 + n2)
    y_pred_single = preds_pool @ blend
    alpha = fit_shrinkage_power(y_reg_pool, y_pred_single, p_act_pool)
    assert 0.0 <= alpha <= 1.0
    shrunk_single = apply_shrinkage(y_pred_single, p_act_pool, alpha)
    assert shrunk_single.shape == (n1 + n2,)

    d_alpha, d_beta = fit_dual_shrinkage_powers(y_reg_pool, y_pred_single, p_act_pool, p_store_pool)
    assert 0.0 <= d_alpha <= 1.0
    assert 0.0 <= d_beta <= 1.0
    shrunk_dual = apply_dual_shrinkage(y_pred_single, p_act_pool, p_store_pool, d_alpha, d_beta)
    assert shrunk_dual.shape == (n1 + n2,)


def test_linear_calibration():
    from src.calibration import fit_linear_calibration, apply_linear_calibration

    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([0.9, 1.8, 2.7, 3.6, 4.5])  # slope is ~1.11

    slope, intercept = fit_linear_calibration(y_true, y_pred)
    assert np.isclose(slope, 1.11, atol=0.05)
    assert np.isclose(intercept, 0.0, atol=0.1)

    calibrated = apply_linear_calibration(y_pred, slope, intercept)
    assert np.allclose(calibrated, y_true, atol=0.1)


def test_member_debiasing():
    from src.calibration import fit_member_debiasing, apply_member_debiasing

    rng = np.random.default_rng(42)
    y_true = rng.uniform(0.5, 5.0, size=100)
    # Shrunk predictions (e.g. median shrinkage from Huber loss: slope 0.8)
    preds = (0.8 * y_true + 0.1).reshape(-1, 1)

    params = fit_member_debiasing(preds, y_true)
    assert len(params) == 1
    assert params[0][0] > 1.0  # Should learn to expand slope back towards 1.0/0.8 = 1.25

    debiased = apply_member_debiasing(preds, params)
    init_err = np.mean(np.abs(preds[:, 0] - y_true))
    deb_err = np.mean(np.abs(debiased[:, 0] - y_true))
    assert deb_err < init_err


def test_customer_blup():
    import pandas as pd
    from src.calibration import (
        compute_customer_blup,
        fit_blup_shrinkage,
        apply_customer_blup,
    )

    customer_residuals = {
        "c1": [(pd.Timestamp("2024-03-01"), 0.5), (pd.Timestamp("2024-06-01"), 0.4)],
        "c2": [(pd.Timestamp("2024-06-01"), -0.3)],
    }
    target_cutoff = pd.Timestamp("2024-12-01")

    blup_effects = compute_customer_blup(customer_residuals, target_cutoff, kappa=1.0)
    assert "c1" in blup_effects
    assert blup_effects["c1"] > 0.0
    assert "c2" in blup_effects
    assert blup_effects["c2"] < 0.0

    cids = np.array(["c1", "c2", "c3"])
    blup_vec = np.array([blup_effects.get(c, 0.0) for c in cids])
    preds = np.array([2.0, 2.0, 2.0])
    y_true = np.array([2.3, 1.8, 2.0])
    active_mask = np.ones(3, dtype=bool)

    eta = fit_blup_shrinkage(y_true, preds, blup_vec, active_mask)
    assert 0.0 <= eta <= 0.25

    adjusted = apply_customer_blup(preds, blup_vec, eta, active_mask)
    assert adjusted[0] >= preds[0]
    assert adjusted[1] <= preds[1]


def test_decide_with_prior_alignment():
    from src.calibration import decide_with_prior_alignment

    rng = np.random.default_rng(42)
    n_samples, n_classes = 1000, 4
    logits = rng.normal(size=(n_samples, n_classes))
    logits[:, 0] += 2.0
    exp_logits = np.exp(logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    class_weights = np.ones(n_classes)
    target_prior = np.array([0.25, 0.25, 0.25, 0.25])
    decisions = decide_with_prior_alignment(probs, class_weights, target_prior, max_steps=50)

    assert decisions.shape == (n_samples,)
    uncal_counts = np.bincount(np.argmax(probs, axis=1), minlength=4) / n_samples
    cal_counts = np.bincount(decisions, minlength=4) / n_samples

    uncal_dist = np.sum(np.abs(uncal_counts - target_prior))
    cal_dist = np.sum(np.abs(cal_counts - target_prior))
    assert cal_dist < uncal_dist


def test_conformal_selective_arbitration():
    from src.calibration import conformal_selective_arbitration

    rng = np.random.default_rng(42)
    n, k = 100, 4
    probs = rng.uniform(0.1, 1.0, size=(n, k))
    probs = probs / probs.sum(axis=1, keepdims=True)

    features = pd.DataFrame({
        "m3_fuel_litres": np.full(n, 10.0),
        "m3_fuel_rand": np.full(n, 235.0),
    })
    pred_fuel = np.full(n, 1.0)
    class_weights = np.ones(k)
    target_priors = np.full(k, 0.25)

    decisions = conformal_selective_arbitration(
        probs=probs,
        features=features,
        pred_fuel=pred_fuel,
        class_weights=class_weights,
        target_priors=target_priors,
        tau_conf=0.15,
        fg_idx=1,
        stable_idx=0,
    )
    assert len(decisions) == n
    assert (decisions >= 0).all() and (decisions < k).all()






