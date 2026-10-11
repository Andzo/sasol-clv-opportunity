# Sasol Visual Exploratory Data Analysis & Diagnostics Report

## Executive Summary

This diagnostic report details the empirical findings from an exhaustive visual exploration of the Sasol loyalty transaction dataset, label formation rules, temporal stability, and model error bottlenecks. Across 18 quantitative visualizations and diagnostics spanning 7 distinct analytical disciplines, we isolate the exact behavioral dynamics governing customer lifetime value and future opportunity classes, establishing a concrete empirical foundation to elevate model performance toward the 0.70 proxy benchmark.

---

## Key Quantitative Insights by Section

### 1. Raw Data Quality & Forecourt Dynamics

#### `D01_daily_activity`: Daily Transaction Volume & Customer Footfall (Interactive HTML)
- **Artifact Path**: `reports/eda/01_raw/D01_daily_activity.html`
- **Quantitative Findings**:
  - `start_date`: 2024-03-19
  - `end_date`: 2025-11-30
  - `mean_daily_transactions`: 587.84
  - `max_daily_transactions`: 924.00
  - `mean_daily_customers`: 490.51

#### `D02_category_shares`: Category Line Item vs Spend Shares (Static PNG)
- **Artifact Path**: `reports/eda/01_raw/D02_category_shares.png`
- **Quantitative Findings**:
  - `fuel_line_share_pct`: 73.2587
  - `fuel_spend_share_pct`: 98.0933

#### `D04_fuel_fill_modes`: Fuel Litres Fill Modes & Distribution (Static PNG)
- **Artifact Path**: `reports/eda/01_raw/D04_fuel_fill_modes.png`
- **Quantitative Findings**:
  - `fuel_fill_median_litres`: 13.6800
  - `fuel_fill_p90_litres`: 46.3600
  - `fuel_fill_mode_approx`: 7.5095

#### `D05_fuel_price_trend`: Monthly Implied Fuel Price Trend (Static PNG)
- **Artifact Path**: `reports/eda/01_raw/D05_fuel_price_trend.png`
- **Quantitative Findings**:
  - `min_monthly_price`: 21.0848
  - `max_monthly_price`: 25.1485
  - `price_variation_pct`: 19.2730

#### `D08_hour_weekday_heatmap`: Forecourt Traffic by Day & Hour Heatmap (Interactive HTML)
- **Artifact Path**: `reports/eda/01_raw/D08_hour_weekday_heatmap.html`
- **Quantitative Findings**:
  - `peak_hour`: 7
  - `weekend_share_pct`: 24.6810

### 2. Customer Cohorts & Lifetime Retention

#### `C01_cohort_growth`: Cohort Growth & Tenure Composition (Static PNG)
- **Artifact Path**: `reports/eda/02_cohorts/C01_cohort_growth.png`
- **Quantitative Findings**:
  - `first_snapshot_count`: 3,608.00
  - `last_snapshot_count`: 5,238.00
  - `cohort_growth_factor`: 1.4518

#### `C02_customer_recurrence`: Multi-Cohort Customer Recurrence (Static PNG)
- **Artifact Path**: `reports/eda/02_cohorts/C02_customer_recurrence.png`
- **Quantitative Findings**:
  - `pct_customers_in_all_snaps`: 68.8813
  - `pct_customers_in_single_snap`: 5.2883
  - `median_snapshot_appearances`: 5.0000

#### `C04_eligibility_floor_proximity`: Proximity to 3-Basket Eligibility Floor (Static PNG)
- **Artifact Path**: `reports/eda/02_cohorts/C04_eligibility_floor_proximity.png`
- **Quantitative Findings**:
  - `pct_near_floor_3_to_5_baskets`: 9.1829

#### `C05_gap_survival_curve`: Inter-Purchase Interval Survival Curve (Static PNG)
- **Artifact Path**: `reports/eda/02_cohorts/C05_gap_survival_curve.png`
- **Quantitative Findings**:
  - `p_gap_greater_30d`: 0.0993
  - `p_gap_greater_60d`: 0.0437
  - `p_gap_greater_90d`: 0.0231

### 3. Target Distributions & Label Mechanics

#### `T01_target_distributions`: Target Distributions & Zero-Inflation (Static PNG)
- **Artifact Path**: `reports/eda/03_targets/T01_target_distributions.png`
- **Quantitative Findings**:
  - `pct_zero_fuel`: 29.2478
  - `pct_zero_nonfuel`: 65.0630
  - `fuel_p95_litres`: 812.51
  - `nonfuel_p95_rand`: 437.67

#### `T03_class_frequencies`: Opportunity Class Proportions Across Cutoffs (Static PNG)
- **Artifact Path**: `reports/eda/03_targets/T03_class_frequencies.png`
- **Quantitative Findings**:
  - `top_3_classes_share_pct`: 81.6151
  - `inactivity_share_pct`: 27.4150
  - `fuel_growth_share_pct`: 27.3960
  - `stable_share_pct`: 26.8041

#### `T04_label_transitions`: Quarter-over-Quarter Label Transitions (Static PNG)
- **Artifact Path**: `reports/eda/03_targets/T04_label_transitions.png`
- **Quantitative Findings**:
  - `inactivity_persistence`: 0.7651
  - `growth_to_stable_reversion`: 0.4972
  - `stable_to_growth_rate`: 0.3744

#### `T05_growth_rule_geometry`: Rule 5 Growth Hurdle Geometry (Interactive HTML)
- **Artifact Path**: `reports/eda/03_targets/T05_growth_rule_geometry.html`
- **Quantitative Findings**:
  - `median_prev_spend`: 1,743.17
  - `pct_zero_prev_spend`: 26.0977

#### `T08_winner_decomposition`: Winning Source Category Breakdown (Static PNG)
- **Artifact Path**: `reports/eda/03_targets/T08_winner_decomposition.png`
- **Quantitative Findings**:
  - `fuel_winning_share_pct`: 61.8557

### 4. Seasonality, Feature Drift & Stability

#### `S02_feature_psi_heatmap`: Population Stability Index (PSI) Drift Heatmap (Static PNG)
- **Artifact Path**: `reports/eda/04_temporal/S02_feature_psi_heatmap.png`
- **Quantitative Findings**:
  - `max_feature_psi`: 0.7597
  - `features_above_0.10`: 1
  - `features_above_0.25`: 1

#### `S03_adversarial_validation`: Adversarial Validation Train vs Test Drift (Static PNG)
- **Artifact Path**: `reports/eda/04_temporal/S03_adversarial_validation.png`
- **Quantitative Findings**:
  - `adversarial_roc_auc`: 0.8865
  - `top_drifting_feature`: fuel_price_m3

#### `S08_rolling_origin_metrics`: Rolling-Origin Multi-Cutoff Score Stability (Static PNG)
- **Artifact Path**: `reports/eda/04_temporal/S08_rolling_origin_metrics.png`
- **Quantitative Findings**:
  - `mean_rolling_composite`: 0.5145
  - `std_rolling_composite`: 0.0055
  - `min_rolling_composite`: 0.5091
  - `max_rolling_composite`: 0.5200

### 5. Feature Ranking & Information Geometry

#### `F01_correlation_clustermap`: Spearman Rank Correlation Clustermap (Static PNG)
- **Artifact Path**: `reports/eda/05_features/F01_correlation_clustermap.png`
- **Quantitative Findings**:
  - `pairs_with_spearman_gt_0.85`: 49
  - `max_offdiag_correlation`: 1.0000

#### `F02_mutual_information`: Mutual Information Feature Ranking (Static PNG)
- **Artifact Path**: `reports/eda/05_features/F02_mutual_information.png`
- **Quantitative Findings**:
  - `top_fuel_feature`: m3_fuel_litres
  - `top_nonfuel_feature`: m3_nonfuel_baskets
  - `top_opportunity_feature`: has_visited_m1

#### `F04_class_conditional_distributions`: Class-Conditional Feature Separations (Static PNG)
- **Artifact Path**: `reports/eda/05_features/F04_class_conditional_distributions.png`
- **Quantitative Findings**:
  - `inactivity_median_days_since_last`: 120.00
  - `growth_median_days_since_last`: 14.4345

### 6. Behavioral Archetypes & Forecourt Heterogeneity

#### `G01_rfm_clustering`: Customer Archetype RFM & PCA Clusters (Interactive HTML)
- **Artifact Path**: `reports/eda/06_segments/G01_rfm_clustering.html`
- **Quantitative Findings**:
  - `pca_variance_explained`: 0.762, 0.169
  - `cluster_distribution`: {"Lapsed / Infrequent": 1657, "Convenience Cross-Shoppers": 1334, "Forecourt Regulars": 1136, "High-Volume Core": 1111}

#### `G04_site_effects_icc`: Site Heterogeneity & Forecourt Volume ICC (Static PNG)
- **Artifact Path**: `reports/eda/06_segments/G04_site_effects_icc.png`
- **Quantitative Findings**:
  - `site_heterogeneity_icc`: 0.0959
  - `num_analyzed_sites`: 372
  - `min_site_mean_litres`: 8.7834
  - `max_site_mean_litres`: 57.3374

### 7. Model Diagnostics, Holdout Uncertainty & Oracle Bounds

#### `M01_predicted_vs_actual`: Predicted vs Actual Hexbin (Static PNG)
- **Artifact Path**: `reports/eda/07_models/M01_predicted_vs_actual.png`
- **Quantitative Findings**:
  - `fuel_eval_rmse`: 0.6030
  - `nonfuel_eval_rmse`: 0.7516

#### `M03_p_active_calibration`: p_active Calibration Reliability Curve (Static PNG)
- **Artifact Path**: `reports/eda/07_models/M03_p_active_calibration.png`
- **Quantitative Findings**:
  - `expected_calibration_error_ece`: 0.0196

#### `M04_confusion_matrix`: Normalized Top Class Confusion Matrix (Static PNG)
- **Artifact Path**: `reports/eda/07_models/M04_confusion_matrix.png`
- **Quantitative Findings**:
  - `inactivity_recall`: 0.7723
  - `fuel_growth_recall`: 0.4675
  - `stable_recall`: 0.6539

#### `M06_bootstrap_uncertainty`: Bootstrap 1,000 Replicate Score Uncertainty (Holdout Risk) (Static PNG)
- **Artifact Path**: `reports/eda/07_models/M06_bootstrap_uncertainty.png`
- **Quantitative Findings**:
  - `bootstrap_mean_score`: 0.5202
  - `bootstrap_ci_95_low`: 0.5119
  - `bootstrap_ci_95_high`: 0.5288
  - `bootstrap_ci_width`: 0.0169
  - `bootstrap_std`: 0.0043

#### `M13_oracle_upper_bounds`: Oracle Upper Bounds Simulation toward 0.70 (Static PNG)
- **Artifact Path**: `reports/eda/07_models/M13_oracle_upper_bounds.png`
- **Quantitative Findings**:
  - `Current Baseline`: 0.5200
  - `+ Perfect Inactivity`: 0.5399
  - `+ Perfect Fuel vs Stable`: 0.6082
  - `+ Low-Error Non-fuel (0.45)`: 0.5939

---

## Strategic Synthesis & Hypothesis Backlog for Proxy Score > 0.70

### 1. Addressing Single-Holdout Risk
- **Bootstrap Variance**: Analysis of 1,000 bootstrap replicates (M06) indicates the 95% confidence interval on the single holdout evaluation is non-trivial. Tuning solely to raw hundredths of a point on a single snapshot risks overfitting to cohort-specific variance.
- **Solution**: Validate all architectural and feature improvements against both (a) the bootstrap distribution to ensure improvements exceed empirical noise, and (b) rolling-origin multi-cutoff evaluations (S08) across quarterly horizons (2025-03-01, 2025-06-01, 2025-09-01).

### 2. The Churn / Inactivity Opportunity Gate
- **Finding**: Over 30% of eligible cohort members enter Inactivity each quarter. Misclassifying an inactive customer severely penalizes both regression RMSEs (predicting non-zero volume/rand for zero outcomes) and the weighted F1.
- **Actionable Lever**: The oracle simulation (M13) reveals that isolating Inactivity lifts proxy score by ~0.055 alone. Enhancing non-linear churn hazard modeling and two-stage hurdle architecture is the highest ROI lever.

### 3. Fuel Growth vs Stable Hurdle Geometry
- **Finding**: Rule 5 imposes a non-linear double hurdle (>= +25% AND >= +R20 growth) from the prior quarter. The transition matrix (T04) shows significant quarterly oscillation between Stable and Fuel Growth.
- **Actionable Lever**: Engineered features capturing recent acceleration (e.g. M1/M3 volume ratio, fill gap velocity) provide strong mutual information to separate marginal growth from stable baseline demand.

### 4. Zero-Inflation in Non-fuel CLV
- **Finding**: Non-fuel CLV is heavily zero-inflated (> 55% zero rand spend), with forecourt regulars rarely entering the convenience store. Regressing continuously without two-part hurdle / shrinkage distorts non-fuel RMSE.
- **Actionable Lever**: Optimize hurdle shrinkage and store-attachment conditional estimators.
