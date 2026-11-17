# Sasol Customer Lifetime Value & Opportunity Machine Learning System

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](#license)
[![Holdout Composite](https://img.shields.io/badge/Holdout%20Score-0.2893-brightgreen.svg)](#key-results--evaluation-metrics)

Production-grade machine learning system developed for the **Sasol Customer Lifetime Value (CLV) & Opportunity Prediction Challenge** on [Zindi](https://zindi.africa/). The system forecasts quarterly customer volumetric fuel consumption, convenience store spend, and customer growth/adoption trajectories across Sasol forecourt service stations in South Africa.

---

## Executive Summary & Business Impact

Retail fuel stations operate across two tightly coupled revenue streams: high-frequency **forecourt fuel demand** and higher-margin **convenience store spend** (beverages, food, groceries, and impulse retail). Predicting individual customer spend alongside next-quarter behavioral transitions enables data-driven commercial decision-making across Sasol's loyalty network:

- **Targeted Cross-Sell & CRM Personalization:** Distinguishes fuel-only commuters from customers likely to adopt or expand into convenience store categories (`Beverages`, `Food & beverage`, `Groceries`, etc.), focusing promotional budgets on high-propensity growth cohorts.
- **Proactive Churn Retention:** Identifies customers at risk of 90-day dormancy (`Inactivity`) early, enabling timely loyalty interventions before habitual visits lapse.
- **Forecourt & Store Inventory Allocation:** Provides calibrated quarterly forecasts of volumetric fuel demand (`CLV_fuel`) and non-fuel retail spend (`CLV_nonfuel`) across zero-inflated customer histories (**65.06%** non-fuel zeros; **29.25%** fuel zeros).

**Engineering Highlights:** Zero-redundancy `src/` package layout, leak-free purged multi-cutoff tournament validation, two-part zero-inflated hurdle modeling, Customer Empirical Bayes (BLUP) fixed effects, 4-partition prior-aligned dual coordinate ascent, and an automated `pytest` unit test suite.

---

## Repository Structure

```text
├── pyproject.toml                  # Package metadata, dependencies, and CLI entry points
├── requirements.txt                # Direct pip dependency list
├── pytest.ini                      # Pytest configuration
├── README.md                       # Project overview, architecture, and quickstart
├── docs/
│   └── METHODOLOGY.md              # Full methodology walkthrough & architectural rationale
├── MODEL_BENCHMARK_TRACKER.json    # Automated benchmark history & champion record
├── reports/
│   └── evaluation_report.json      # Latest reproduced holdout evaluation report
├── src/                            # Core modular ML pipeline (zero unused modules)
│   ├── __init__.py
│   ├── main.py                     # Primary CLI entry point (`python -m src.main`)
│   ├── config.py                   # Immutable Paths, ModelConfig, and PipelineConfig
│   ├── label_rules.py              # Standalone deterministic Zindi target generator
│   ├── label_config.json           # Category mappings, thresholds, and target normalizers
│   ├── data.py                     # Schema validation, integer scaling, and local file resolution
│   ├── labels.py                   # Validated wrapper around label_rules
│   ├── eligibility.py              # Category-history feasibility masks for Opportunity classes
│   ├── features.py                 # Longitudinal, RFM, BTYD, hurdle-physics & seasonal features
│   ├── dataset.py                  # Fingerprinted SnapshotBuilder & temporal panel stacking
│   ├── models.py                   # Multi-loss regressors, hierarchical & attachment classifiers
│   ├── hurdle.py                   # Bagged & multi-quantile two-part hurdle models
│   ├── calibration.py              # Debiasing, BLUP, shrinkage & prior-aligned dual ascent
│   ├── metrics.py                  # RMSE, weighted/macro F1 & composite scoring
│   ├── tournament.py               # Multi-cutoff out-of-fold tournament & stacking engine
│   ├── pipeline.py                 # End-to-end training, evaluation & submission orchestration
│   ├── submission.py               # Submission builder & strict schema validation
│   └── tracker.py                  # Automated internal benchmark tracking
└── tests/                          # Automated unit test suite for all pipeline modules
    ├── test_calibration.py
    ├── test_data.py
    ├── test_eligibility.py
    ├── test_features.py
    ├── test_hurdle.py
    ├── test_metrics.py
    ├── test_models.py
    ├── test_speed_and_fusion.py
    ├── test_submission.py
    └── test_tournament.py
```

*(Historical exploratory scripts and intermediate competition experiments are archived separately on the `archive/experiments-and-scripts` branch.)*

---

## Setup & Quickstart

### Prerequisites
- Linux / macOS (tested on Ubuntu 24.04 LTS)
- Python 3.10+ (tested with Python 3.14)
- Optional: NVIDIA CUDA-compatible GPU (`--gpu`)

### Installation
```bash
git clone https://github.com/Andzo/sasol-clv-opportunity.git
cd sasol-clv-opportunity

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"
```

### Dataset Setup
Competition CSV files are excluded from version control in accordance with Zindi data rules. Download the dataset from Zindi and place `train.csv`, `test.csv`, and `SampleSubmission.csv` in either the repository root or a local `data/` directory:

```text
data/
├── train.csv
├── test.csv
└── SampleSubmission.csv
```

If local CSVs are missing, the CLI fails fast with a `FileNotFoundError` explaining where to place the files.

### Running Tests
```bash
pytest -q
```

### Reproducing the Final Pipeline & Submission
```bash
# Inspect all CLI flags
python -m src.main --help

# Reproduce holdout calibration & evaluation on GPU (~13 min from scratch, or ~5s with --reuse-l1)
python -m src.main --gpu --stage calibrate

# Run full end-to-end training, holdout evaluation, full-panel retraining, and test submission generation
python -m src.main --gpu --save-predictions
# or via installed console script (add --gpu for CUDA acceleration):
sasol-clv --gpu --save-predictions
```

---

## Core Architecture & Mathematical Foundations

*(For a complete technical walkthrough covering modeling intuition, design trade-offs, mathematical formulations, and code-level flow, see [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).)*

### 1. Evaluation Metric

The challenge optimizes a weighted composite across fuel volume regression, convenience store spend regression, and 17-class Opportunity classification:

$$
\text{Composite} = 0.3 \cdot \left(1 - \frac{\mathrm{RMSE_{fuel}}}{0.740}\right) + 0.3 \cdot \left(1 - \frac{\mathrm{RMSE_{nonfuel}}}{0.816}\right) + 0.4 \cdot \mathrm{F1_{opp}}
$$

### 2. Multi-Cutoff Out-of-Fold Tournament & Cross-Target Stacking

The pipeline ([`src/pipeline.py`](src/pipeline.py), [`src/tournament.py`](src/tournament.py), [`src/calibration.py`](src/calibration.py)) enforces a 3-month causal embargo so training outcome windows $[c, c + 3\text{m})$ never overlap evaluation cutoffs:

**Target-Specific Gain Pruning:** Selects the top 240 numeric features for `CLV_fuel` and `CLV_nonfuel` via LightGBM split gain and the top 220 features across the 3 macro Opportunity decision boundaries (`Inactivity`, `Fuel Growth`, and `Retail`), preserving all structural categorical features.

**Cross-Target Non-Negative Ridge Stacking:** Combines per-member linear-debiased predictions from `lgbm_tweedie`, `xgb_mse`, `xgb_huber`, `ridge`, `extratrees`, bagged two-part hurdles (`hurdle_bagged`), multi-quantile hurdles (`hurdle_quantile`), and physical basket-ticket regressors (`phys_basket`) over a 20-column cross-target meta-matrix:

$$
\hat y_{\text{raw}} = M_{\text{reg}} w^{\star}, \quad w^{\star} = \arg\min_{w \ge 0,\ \mathbf{1}^{\top} w = 1} \lVert y - M_{\text{reg}} w \rVert_2^2 + \alpha \lVert w \rVert_2^2 \quad \text{s.t.} \quad \sum_{j \in \mathcal{H}} w_j \ge \tau_{\text{hurdle}}
$$

**Customer Empirical Bayes (BLUP):** Shrinks customer-level historical anchor residuals toward zero with active-cohort centering to capture persistent customer heterogeneity.

### 3. Two-Part Hurdle Models & Zero Inflation

Non-fuel spend is **65.06% zero-inflated** and fuel consumption is **29.25% zero-inflated**. Two-part hurdle models ([`src/hurdle.py`](src/hurdle.py)) decouple binary activity classification $P(\text{Active} > 0 \mid X)$ from conditional spend estimation $\mathbb{E}[\text{Spend} \mid \text{Active} > 0, X]$, incorporating Heckman Inverse Mills Ratio selection-bias correction:

$$
\lambda(p) = \frac{\phi(\Phi^{-1}(p))}{p}, \quad \mathbb{E}[Y \mid X] = P(Y > 0 \mid X) \cdot \mathbb{E}[Y \mid X,\ Y > 0,\ \lambda(p)]
$$

### 4. Hierarchical Opportunity Classification & 4-Partition Dual Ascent

Opportunity probabilities fuse a 3-family multiclass [`ClassifierEnsemble`](src/models.py) (`lgbm`, `catboost`, `xgb`) with a 4-stage [`HierarchicalOpportunityClassifier`](src/models.py) (Inactivity gate, Active Retail gate, Fuel Growth vs. Stable boundary around the +25% and ≥ R20 growth rule, and Retail multiclass) plus physical hurdle exceedance $P_{\text{exceed}}(\text{fuel})$ and joint inactivity $(1 - p_{\text{fuel}})(1 - p_{\text{store}})$. Final class assignments are decoded via eligibility masking ([`src/eligibility.py`](src/eligibility.py)) and 4-partition dual coordinate ascent ([`decide_with_prior_alignment`](src/calibration.py)).

---

## Key Results & Evaluation Metrics

### End-to-End Pipeline Performance (`submission_git_aade876_20261018_235042_comp0.2893.csv`)

| Evaluation Stage | Cutoff | Fuel RMSE | Nonfuel RMSE | Opportunity F1 | Composite Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Tournament Fold 1 (Winter)** | `2025-06-01` | `0.5944` | `0.7344` | `0.5031` | `0.29030` |
| **Tournament Fold 2 (Spring)** | `2025-09-01` | `0.5968` | `0.7447` | `0.5018` | `0.28500` |
| **Calibrated Forward Holdout (Count-Aligned)** | `2025-09-01` | **`0.5976`** | **`0.7435`** | **`0.5122`** | **`0.28927`** (`0.2893`) |
| **Calibrated Forward Holdout (OOF-Prior)** | `2025-09-01` | `0.5976` | `0.7435` | `0.5179` | `0.29155` |
| *Reference: Persistence Baseline* | `2025-09-01` | `0.6905` | `0.9296` | `0.1180` | `0.06728` |
| *Reference: Zero Submission Baseline* | `2025-09-01` | `1.8345` | `1.2241` | `0.1133` | `0.04533` |

**Test Submission Class Allocation (`N = 5,488`, `2025-12-01`):** `Stable`: `1,954`, `Existing-category growth: Fuel`: `1,575`, `Inactivity`: `1,449`, `Existing-category growth: Other`: `322`, `Existing-category growth: Beverages`: `150`, `Existing-category growth: Food & beverage`: `37`, `New category adoption: Other`: `1`.

*Fresh GPU calibration check (`python -m src.main --gpu --stage calibrate` on cleaned `main`): Fuel RMSE `0.5976`, Nonfuel RMSE `0.7435`, Opportunity F1 `0.5116`, Composite Score `0.28903` (OOF-prior `0.28954`); folds `0.2910` (Winter) and `0.2836` (Spring). Regression RMSEs match `reports/evaluation_report.json` exactly, with `0.0006` multiclass F1 difference from non-deterministic CUDA tree histogram reductions.*

---

## License

This project is open-sourced under the MIT License.
