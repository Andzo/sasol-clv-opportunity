# Methodology & Technical Architecture Walkthrough

This document explains the end-to-end statistical design, causal validation architecture, feature engineering rationale, and optimization choices behind `sasol-clv-opportunity`. It is structured as a layer-by-layer technical reference linking each modeling decision directly to its implementation in `src/`.

---

## 1. Problem Formulation & Commercial Objectives

The objective is to predict next-quarter customer behavior across Sasol forecourt fuel stations and convenience stores over a strictly forward 3-month horizon $[c, c + 3\text{m})$ relative to a prediction cutoff $c$. Each customer record requires three coupled outputs:

1. **Fuel Volume Forecast (`CLV_fuel`):** Total fuel litres purchased in $[c, c + 3\text{m})$, transformed into normalized log space by [`src/label_rules.py`](../src/label_rules.py) and [`src/label_config.json`](../src/label_config.json):

$$
y_{\text{fuel}} = \frac{\ln(1 + \text{Litres})}{2.352825386185113}
$$

2. **Non-Fuel Spend Forecast (`CLV_nonfuel`):** Total convenience store spend (ZAR) in $[c, c + 3\text{m})$, similarly normalized in log space:

$$
y_{\text{nonfuel}} = \frac{\ln(1 + \text{NonfuelSpend})}{2.4200904988927476}
$$

3. **Opportunity Class (`Opportunity`):** One of 17 mutually exclusive behavioral trajectories describing retention, inactivity, fuel growth, convenience category growth, or new-category cross-sell adoption.

### Evaluation Metric Alignment

All modeling and post-processing stages optimize the joint composite objective computed in [`src/metrics.py`](../src/metrics.py):

$$
\text{Composite} = 0.3 \cdot \left(1 - \frac{\mathrm{RMSE_{fuel}}}{0.740}\right) + 0.3 \cdot \left(1 - \frac{\mathrm{RMSE_{nonfuel}}}{0.816}\right) + 0.4 \cdot \mathrm{F1_{opp}}
$$

where $\mathrm{F1_{opp}}$ denotes Support-Weighted F1 across the 17 Opportunity classes.

Two structural properties of this metric govern the entire pipeline design:
- **Zero-inflation + log-scale tail leverage:** In observed historical cohorts, **65.06%** of `CLV_nonfuel` targets and **29.25%** of `CLV_fuel` targets are exact zeros. Minimizing log-space RMSE requires accurate separation of returning vs. churning accounts *and* unbiased tail conditional expectations without overfitting extreme outliers.
- **Disproportionate weight on multiclass Opportunity ranking (`40%`) and precision-recall calibration:** Under Support-Weighted F1, naive argmax over skewed multiclass posteriors under-predicts medium-frequency classes when seasonal base rates shift between winter (`2025-06-01`), spring (`2025-09-01`), and summer (`2025-12-01`).

---

## 2. Deterministic Target Mechanics & Eligibility Constraints

A critical source of domain signal—and subtle trap if modeled naively—lies in how ground-truth Opportunity labels are generated in [`src/label_rules.py`](../src/label_rules.py):

1. **Inactivity Gate:** A customer becomes `Inactivity` if they record zero qualifying transactions across the entire 3-month forward window.
2. **Qualifying Category Hurdle:** For every non-inactive customer, each of the 24 included product categories is compared between the prior 3-month baseline $[c - 3\text{m}, c)$ and the forward 3-month window $[c, c + 3\text{m})$. A category qualifies as growth/adoption if spend increases by both **≥ +25%** and **≥ R20**.
3. **Competition Winner Selection:** Among all qualifying categories, the label assigns the single category with the **largest positive incremental Rand gain** (`New category adoption: X` if historical pre-cutoff spend in that category was R0; `Existing-category growth: X` if previously purchased). If no category clears both hurdles, the customer is labeled `Stable`.

### Hard Structural Feasibility (`src/eligibility.py`)

Because `Existing-category growth: X` requires prior history in category group $X$ (and `New category adoption: X` requires *no* prior history in that subcategory), multiple classes are mathematically impossible for any given customer at cutoff $c$.

[`src/eligibility.py`](../src/eligibility.py) builds an exact boolean feasibility mask `[n_customers, 17]` strictly from transactions in $(-\infty, c)$:
- A customer who has never purchased `Fuel` (`ever_fuel == 0`) can never be labeled `Existing-category growth: Fuel`.
- Conversely, a customer who has already purchased within a non-fuel category cannot be assigned `New category adoption` for a mapped label whose historical adoption slots are exhausted.

Zeroing impossible class probabilities before temperature scaling and decoding guarantees 100% logically consistent predictions and prevents wasted probability mass.

---

## 3. Causal Temporal Validation & Leakage Prevention

With loyalty panel data spanning multiple monthly cutoffs, standard random K-fold cross-validation leaks both customer autocorrelation and overlapping forward outcome windows (`[c, c + 3m)` overlaps `[c + 1m, c + 4m)`).

### 3-Month Purged Forward Embargo

[`src/config.py`](../src/config.py) and [`src/tournament.py`](../src/tournament.py) enforce a causal embargo between every training cutoff $c_{\text{train}}$ and evaluation cutoff $c_{\text{eval}}$:

$$
c_{\text{train}} \le c_{\text{eval}} - 3\text{ months}
$$

- For any fold evaluated at $c_{\text{eval}}$, all transactions occurring on or after $c_{\text{eval}}$ are excluded before building snapshots (`tx[tx["_time"] < cutoff]`).
- For training snapshots, both feature construction (`< c_train`) and label construction (`[c_train, c_train + 3m)`) complete strictly before $c_{\text{eval}}$, eliminating any temporal target overlap.
- Cached snapshots in [`src/dataset.py`](../src/dataset.py) are keyed by content hashes of `src/features.py`, `src/labels.py`, `src/eligibility.py`, and `src/label_config.json`, ensuring stale feature caches cannot silently mix code revisions.

### Sample Weighting Across Non-Stationary Seasons

Retail fuel and forecourt convenience spending exhibits strong South African seasonality (holiday travel spikes in November–January vs. mid-year winter troughs). Rather than weighting all historical cutoffs uniformly, training rows receive:

$$
w_i = \exp(-0.10 \cdot \Delta\text{months}_i) \cdot s_i
$$

where the seasonal multiplier $s_i = 2.2$ when `month(cutoff)` is in November–January and $s_i = 1.0$ otherwise, normalized to unit mean. Intuitively:
- Exponential decay ($\lambda = 0.10$ per month) down-weights older macro regimes and price environments while retaining enough historical volume for covariance stability.
- The `2.2x` summer multiplier increases effective representation from summer cohorts when calibrating toward the December test cutoff (`2025-12-01`).

### Honest Forward Holdout Protocol (`--stage calibrate`)

In default mode (`ModelConfig.fast()`), Phase 1 runs two embargoed temporal folds (`2025-06-01` weight `1.0x` and `2025-09-01` weight `1.5x`). To prevent calibration overfitting on the holdout itself:
- During holdout evaluation, calibration artifacts (`cal_eval`) are fitted *only* on prior folds (`< 2025-09-01`) and scored out-of-time on `2025-09-01`.
- After verifying holdout metrics, Phase 2 refits calibration across the full out-of-fold pool and retrains all base, hurdle, and classification models on every labeled historical cutoff up to `2025-09-01` (3-month embargo before `2025-12-01` test inference).

---

## 4. Feature Engineering Taxonomy (`src/features.py`)

Feature construction operates exclusively on transactions with `_time < cutoff` and deliberately separates real behavioral signals from nominal price inflation:

### A. Multi-Horizon Rolling Windows & Momentum Ratios
Aggregates counts, baskets, Rand spend, and fuel litres across nested windows (`d14`, `d45`, `m1`, `m2`, `m3`, `q2`, `m6`, and full customer lifetime). Ratio features (`d14/m1`, `m1/m3`, `m3/m6`) capture acceleration, mean-reversion, and sudden churn onset while smoothing uneven calendar month lengths (`30.4375 / 14.0`).

### B. Pump-Price Deflators (Volumetric vs. Nominal Separation)
Because `CLV_fuel` is measured in **litres** while `CLV_nonfuel` and hurdle qualification (`≥ R20`) are measured in **Rand**, pump-price fluctuations confound naive spend histories. [`src/features.py`](../src/features.py) computes station- and customer-level implied fuel price (`Rand / Litres`, clipped to realistic pump bounds `R20.5–R27.5/L`) and relative pump-price indices (`rel_fuel_price_m3`). This separates genuine changes in driving volume from retail fuel price movements.

### C. Hurdle Physics & 24-Category Winner Competition
Because Opportunity depends on whether category spend beats `max(1.25 * m3_spend, m3_spend + 20)`:
- We compute explicit historical hurdle thresholds, remaining Rand headroom (`hurdle - m3_spend`), historical attainability (`q2_spend / hurdle`, `life_rate / hurdle`), and volatility-normalized exceedance Z-scores:

$$
z_{\text{hurdle}} = \frac{H - 3 \cdot \mu_{\text{monthly}}}{\sigma_{\text{monthly}} \sqrt{3} + 1}
$$

- Across all 24 spend categories, we extract expected incremental Rand gaps, number of categories clearing their prior hurdle, and top non-fuel contender spreads (`rule6_fuel_minus_best_retail_inc`).

### D. Probabilistic Customer Lifetime Value (BTYD Priors)
Implements parametric **BG/NBD** (`P(Alive)`, expected 90-day future transactions) and **Gamma-Gamma** (expected monetary spend per visit) estimators fitted via L-BFGS-B strictly on pre-cutoff RFM summaries across overall, fuel-only, and non-fuel-only purchase streams. BTYD features provide smooth Bayesian shrinkage for infrequent shoppers where raw 90-day empirical counts are noisy or zero.

### E. Behavioral Cadence, Burn Rate & Payday Cycles
- **Inter-purchase cadence & Hawkes decay intensity:** Computes time since last fuel/non-fuel basket, mean/std inter-visit intervals (`CV_IPI`), overdue ratios (`days_since_last / mean_ipi`), and exponential Hawkes point-process intensities over short/medium half-lives.
- **Tank burn rate & South African payday indicators:** Approximates days since last meaningful refuel relative to implied fill size and share of shopping during month-end/month-start payday windows (`days 25–31, 1–5`).

### F. Mundlak Within–Between Decomposition & Honest Panel Lags
- **Mundlak features:** Splits customer activity into long-run customer averages ("between" component: habitual basket size and attachment propensity) vs. recent deviations from those averages ("within" component: temporary surge or slump).
- **Station seasonal & Level-3 non-fuel HHI:** Aggregates site-level weekend-vs-weekday ratios and Herfindahl–Hirschman category concentration (`HHI`) across non-fuel baskets.
- **Prior-quarter exact observable outcomes (`prior_q_*`):** Computes the *already realized* labels from `[c - 3m, c)` relative to `[c - 6m, c - 3m)` (strictly `< cutoff`!). This encodes legitimate state transitions—such as two-quarter persistence vs. post-spike mean reversion—with zero future leakage.

### G. Target-Specific Split-Gain Pruning (`src/tournament.py`)
Each snapshot constructs 537 engineered features. Rather than feeding high-dimensional noisy features indiscriminately into all learners, [`select_top_features_by_gain`](../src/tournament.py) runs a regularized LightGBM selector per fold:
- Keeps the top **240** numeric features independently for `CLV_fuel` and `CLV_nonfuel`.
- [`select_top_classification_features`](../src/tournament.py) pools normalized split gain across the three macro binary classification boundaries (**Inactivity**, **Fuel Growth**, and **Active Retail**), retaining essential structural domain features plus the top **220** numeric features alongside all categorical `cat_*` columns.
- This prevents high-dimensional noise from destabilizing linear (`Ridge`) and shallow tree (`ExtraTrees`, `CatBoost`) learners while reducing training memory and variance.

---

## 5. Regression Modeling & Zero-Inflation Strategy (`src/models.py`, `src/hurdle.py`)

No single loss function handles both 29–65% point mass at zero and heavy right-skewed spend tails optimally under RMSE:

### Diverse Direct Regressors (`RegressorEnsemble`)
Trained directly on the log-normalized target over top-240 numeric + structural categorical features:
- **`lgbm_tweedie`:** Compound Poisson-Gamma loss (`power=1.25` for fuel, `1.40` for nonfuel), naturally modeling exact zeros plus right-skewed positive volumes without needing two stages.
- **`xgb_mse` & `xgb_huber`:** Paired MSE (exact L2 conditional mean estimator) and Pseudo-Huber (down-weights anomalous bulk-spend spikes to stabilize splits).
- **`ridge`:** Median-imputed, standardized L2 linear model (`alpha=100.0`) acting as a low-variance linear anchor on continuous RFM/BTYD magnitudes.
- **`extratrees`:** Randomized orthogonal splits (`max_depth=15`, `sqrt` features) providing high variance reduction orthogonal to boosting residual paths.

### Why Two-Part Hurdle & Basket-Decomposed Models (`src/hurdle.py`)
Direct regressors blend activity probability and spend scale inside every leaf. [`src/hurdle.py`](../src/hurdle.py) adds three structural hurdle formulations:

#### 5.1 Bagged Two-Part Hurdle with Anchor Residualization & Heckman IMR (`BaggedTwoPartFuelHurdle` / `BaggedTwoPartNonfuelHurdle`)
Stage 1 (`StoreAttachmentClassifier`) averages LGBM + CatBoost + XGB binary classifiers for $P(Y > 0 \mid X)$. Stage 2 fits a penalized linear Ridge anchor on logged core RFM quantities among active spenders ($Y > 0$), then trains a 5-seed bag (`lgbm_mse`, `lgbm_huber`, `catboost`) on the *anchor residual* $Y - z_{\text{anchor}}$, conditioned on an estimated propensity score $\hat{p}$ and its **Heckman Inverse Mills Ratio (IMR)**:

$$
\lambda(p) = \frac{\phi(\Phi^{-1}(p))}{p}, \quad \mathbb{E}[Y \mid X] = P(Y > 0 \mid X) \cdot \max\bigl(0,\ z_{\text{anchor}} + \hat{\epsilon}(X,\ \lambda(p))\bigr)
$$

Active spenders ($Y > 0$) are a non-random selected subset of the population. Conditioning positive spend regressions on the IMR mitigates selection bias at the activity boundary, while anchor residualization gives tree learners smooth extrapolation outside historical leaf bounds.

#### 5.2 Multi-Quantile Simpson Quadrature Hurdle (`MultiQuantileTwoPartHurdle`)
Models conditional positive quantiles (`q10, q25, q50, q75, q90`) alongside Huber expectation and integrates expected positive spend across the conditional distribution using composite quadrature, while returning spread estimates $(\mu_{\text{pos}}, \sigma_{\text{pos}})$ needed downstream for Gaussian CDF hurdle exceedance.

#### 5.3 Physical Basket × Ticket Regressor (`PhysicalBasketTicketRegressor`)
Decomposes transaction spend into $P(\text{Active}) \times \mathbb{E}[\text{Baskets} \mid \text{Active}] \times \mathbb{E}[\text{Amount per Basket} \mid \text{Active}]$ in physical units before mapping back via $\ln(1 + \hat{Q}) / \text{scale}$, using a bounded second-moment bias adjustment calibrated OOF to avoid systematic log-retransformation under-prediction.

---

## 6. Post-Processing, Stacking & Customer Empirical Bayes (`src/calibration.py`)

All regression calibration steps are fit strictly out-of-fold on TournamentResult pools and executed identically across calibration, forward holdout, and test inference via `_apply_calibrated_inference`:

### Step 1: Per-Member Zero-Intercept Debiasing
Huber, Tweedie, and MSE learners naturally exhibit small systematic scale biases near tails or after log transformation. Each model column $m$ is linearly rescaled without intercept ($b_m = \frac{p_m^\top y}{p_m^\top p_m}$ on positive predictions) so all meta-matrix columns sit on a consistent conditional-mean scale.

### Step 2: Cross-Target Non-Negative Ridge Stacking
Rather than stacking fuel and non-fuel independently, [`build_regression_meta_matrix`](../src/calibration.py) constructs a unified **20-column** meta-matrix containing all 8 fuel models, all 8 non-fuel models, hurdle activity probabilities $(p_{\text{fuel}}, p_{\text{store}})$, and hurdle expected spend $(p_{\text{fuel}} \mu_{\text{fuel}}, p_{\text{store}} \mu_{\text{nf}})$. Non-negative convex ridge optimization solves:

$$
w^{\star} = \arg\min_{w \ge 0,\ \mathbf{1}^{\top} w = 1} \lVert y - M_{\text{reg}} w \rVert_2^2 + \alpha \lVert w \rVert_2^2 \quad \text{s.t.} \quad \sum_{j \in \mathcal{H}} w_j \ge \tau_{\text{hurdle}}
$$

Cross-target columns allow non-fuel spend optimization to borrow regularization strength from overall forecourt trip propensity (`p_fuel_active`), while $\tau_{\text{hurdle}}$ prevents L2 minimization from collapsing specialized hurdle weights during sparse seasons.

### Step 3: Customer Empirical Bayes (BLUP) Residual Shrinkage
Standard cross-sectional models treat recurring customers as i.i.d. snapshots, leaving persistent unobserved heterogeneity (e.g., fleet cards, commercial drivers, multi-car households) in residuals $r_{i, t} = y_{i, t} - \hat y_{i, t}$.

For every evaluation cutoff $T$, [`compute_customer_blup`](../src/calibration.py) computes each customer's Empirical Bayes random intercept strictly from causal past cutoffs $t < T$:

$$
\hat u_{i, T} = \frac{n_{i, <T}}{n_{i, <T} + \lambda_{\text{blup}}} \cdot \bar r_{i, <T}
$$

centered within the active prediction cohort (`mean = 0`) and scaled by an out-of-time optimal shrinkage multiplier $\eta_{\text{blup}} \ge 0$ (`fit_blup_shrinkage`). Active-cohort zero-centering ensures BLUP captures customer *ranking* heterogeneity without shifting the overall seasonal population mean.

### Step 4: Activity Floor Gate
Finally, [`fit_inactivity_threshold`](../src/calibration.py) verifies whether hard-zeroing ultra-low activity candidates improves OOF RMSE; any customer falling below the calibrated dead threshold and having zero 90-day baskets is clamped cleanly to `0.0`.

---

## 7. Multiclass Opportunity Architecture & Prior-Aligned Dual Ascent

### Fusing Flat, Hierarchical & Physical Classifiers
Multiclass probabilities combine two complementary learners plus calibrated physical expectations:

#### 7.1 `ClassifierEnsemble` (`lgbm`, `catboost`, `xgb`)
Direct 17-class `multi:softprob` models weighted `[0.40, 0.25, 0.35]`, with columns safely scattered via `classes_` so missing ultra-rare adoption labels in early folds never misalign class indices.

#### 7.2 `HierarchicalOpportunityClassifier` (`src/models.py`)
Explicitly mirrors the domain decision tree in 4 factorized stages:
- Stage 1: $P(\text{Inactivity} \mid X)$ (3-model binary blend)
- Stage 2: $P(\text{Retail} \mid X,\ \text{Active})$ (3-model binary blend)
- Stage 3: $P(\text{Fuel Growth} \mid X,\ \text{Fuel Growth or Stable})$ restricted to prior fuel buyers (`ever_fuel > 0`)
- Stage 4: $P(c \mid X,\ \text{Retail})$ (multiclass LGBM over non-trio retail categories)

Factorization concentrates training signal cleanly onto hard boundaries—especially separating `Stable` from `Fuel Growth` without interference from unrelated `Inactivity` or minor retail rows.

#### 7.3 Physical Exceedance Coupling (`compute_physical_hurdle_probabilities`)
Computes Gaussian tail probability that calibrated volumetric fuel predictions clear the pump-price-adjusted litre hurdle:

$$
P_{\text{exceed}}(\text{fuel}) = I_{\text{fuel}} \cdot p_{\text{fuel}} \cdot \left[1 - \Phi\left(\frac{z_{\text{hurdle}} - (0.65 \mu_{\text{pos}} + 0.35 \hat y_{\text{fuel}})}{\sigma_{\text{pos}}}\right)\right]
$$

paired with joint hurdle inactivity $P_{\text{phys}}(\text{Inactivity}) = (1 - p_{\text{fuel}})(1 - p_{\text{store}}) \cdot \mathbb{I}(\text{m1} = 0)$.

#### 7.4 Hard Feasibility & Hyper-Active Veto
Impossible transitions are zeroed by `apply_eligibility_mask`, and customers with ≥ 4 baskets in the final month (`d <= 7` days since last visit) have `Inactivity` probability zeroed prior to decoding.

### Why Prior-Aligned Dual Ascent Beats Naive Argmax Under Support-Weighted F1
Under Support-Weighted F1 ($\sum_k \pi_k \cdot \mathrm{F1_k}$), predicting classes purely via unweighted $\arg\max_k P(Y = k \mid X_i)$ is generally Bayes-suboptimal because:
1. F1 is a non-linear ratio of expected true positives to expected predicted counts plus true prevalence: $\frac{2\mathrm{TP_k}}{\pi_k N + \hat{N}_k}$. Adjusting per-class multipliers boosts F1 by trading moderate precision loss for higher recall on high-support and borderline macro classes (Lipton, Elkan & Narayanaswamy, 2014).
2. Covariate and seasonal shift alters December prior proportions compared to mid-year calibration folds.

[`decide_with_prior_alignment`](../src/calibration.py) solves an entropy-free linear assignment dual problem via coordinate ascent on class biases $\lambda \in \mathbb{R}^K$:

$$
\hat{y}_i = \arg\max_k \bigl(\ln P(Y_i = k \mid X_i) + \ln w_k + \lambda_k\bigr)
$$

updating $\lambda_k$ proportionally to $\pi^{\text{target}}_k - \hat{\pi}_k(\lambda)$ until predicted empirical shares align with target seasonal priors across a **4-partition decomposition**:
- Macro partitions (`Stable`, `Existing-category growth: Fuel`, `Inactivity`) are aligned directly to their seasonal priors.
- All 14 minor convenience retail classes (`Other`, `Beverages`, `Food & beverage`, and adoption sub-classes) share one pooled residual retail budget so dual ascent allocates the retail quota to whichever customers have the highest posterior confidence for specific subcategories without forcing arbitrary artificial quotas onto rare classes.

For test inference on `2025-12-01`, seasonal target priors (`PipelineConfig.seasonal_test_priors`) reflect Q4 late-year proportions anchored against historical `train.csv` summer/winter/spring class trajectories. During forward holdout verification (`2025-09-01`), both OOF-estimated priors and holdout-aligned proportions are reported transparently in `reports/evaluation_report.json` to separate calibration shift from model ranking quality.

---

## 8. Key Engineering & Ablation Decisions (What Was Excluded and Why)

Keeping the repository modular and interview-defensible meant removing components whose added complexity did not improve out-of-time generalization:

- **Tabular Neural Networks (`MLP` / `TabM`):** Removed in favor of regularized GBDTs (`LightGBM`, `CatBoost`, `XGBoost`), `Ridge`, and `ExtraTrees`. On 15k–40k tabular RFM panels with heavy zero-inflation and structured `NaN` censoring, GBDTs + linear/hurdle hybrids achieved stronger validation RMSE with much faster deterministic CPU reproducibility.
- **Balanced Class-Weight LightGBM (`lgbm_balanced`):** Excluded because aggressive inverse-frequency reweighting distorted probability calibration for extreme-rare adoption classes (`< 0.5%` prevalence), inflating downstream temperature scaling variance.
- **Level-2 Stacking Meta-Classifier & Post-Hoc Boundary Arbitration:** Flags (`use_meta_classifier`, `use_soft_shrinkage`, `use_commuter_arbitration`) were empirically disabled during final holdout tuning because global OOF calibration + Nelder-Mead physical fusion (`P_exceed`, `joint_inactivity`) + 4-partition dual ascent outperformed extra second-stage tree stacking on out-of-time `2025-09-01` stability. Dead code paths were cleaned from `main` so every active line executes in the default pipeline.

---

## 9. Codebase Walkthrough Guide (End-to-End Execution Flow)

When stepping through the codebase with reviewers, follow the execution path from entry point to output validation:

| Step | Module | Responsibility |
| :--- | :--- | :--- |
| 1. CLI & Config | [`src/main.py`](../src/main.py), [`src/config.py`](../src/config.py) | Parses `--stage`, `--reuse-l1`, `--fast`/`--full-grid`; defines causal cutoffs, seeds, and paths |
| 2. Ingestion | [`src/data.py`](../src/data.py) | Validates schema, scales Rand values by `1e6` (`SCALE`), annotates included categories |
| 3. Target Rules & Masks | [`src/label_rules.py`](../src/label_rules.py), [`src/labels.py`](../src/labels.py), [`src/eligibility.py`](../src/eligibility.py) | Deterministic Zindi target builder & category-history feasibility masks |
| 4. Feature Matrix | [`src/features.py`](../src/features.py), [`src/dataset.py`](../src/dataset.py) | Builds point-in-time causal feature blocks and caches fingerprinted snapshots |
| 5. Model Families | [`src/models.py`](../src/models.py), [`src/hurdle.py`](../src/hurdle.py) | Multi-loss regressors, bagged IMR & multi-quantile hurdles, flat & hierarchical classifiers |
| 6. Tournament & Calibration | [`src/tournament.py`](../src/tournament.py), [`src/calibration.py`](../src/calibration.py) | Runs embargoed OOF folds, debiasing, non-neg ridge stacking, BLUP, and dual ascent |
| 7. Pipeline & Submission | [`src/pipeline.py`](../src/pipeline.py), [`src/submission.py`](../src/submission.py), [`src/metrics.py`](../src/metrics.py) | Evaluates forward holdout, retrains full panel, runs `_apply_calibrated_inference`, validates output CSV |
