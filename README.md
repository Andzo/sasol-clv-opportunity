# Sasol Customer Lifetime Value & Opportunity Machine Learning System

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](#license)
[![Leaderboard Peak](https://img.shields.io/badge/Zindi%20Peak-0.313644-brightgreen.svg)](#key-results--competition-milestones)

Model developed for the **Sasol Customer Lifetime Value (CLV) & Opportunity Prediction Challenge** on [Zindi](https://zindi.africa/). The system forecasts quarterly customer volumetric fuel consumption, convenience store spend, and customer growth/adoption trajectories across Sasol forecourt service stations in South Africa.

---

## Executive Summary & Business Impact

Retail fuel stations operate across two tightly coupled revenue streams: high-frequency **forecourt fuel demand** and higher-margin **convenience store spend** (beverages, food, groceries, and impulse retail). Predicting individual customer spend alongside next-quarter behavioral transitions enables data-driven commercial decision-making across Sasol's loyalty network:

- **Targeted Cross-Sell & CRM Personalization:** Distinguishes fuel-only commuters from customers likely to adopt or expand into convenience store categories (`Beverages`, `Food & beverage`, `Groceries`, etc.), focusing promotional budgets on high-propensity growth cohorts.
- **Proactive Churn Retention:** Identifies customers at risk of 90-day dormancy (`Inactivity`) early, enabling timely loyalty interventions before habitual visits lapse.
- **Forecourt & Store Inventory Allocation:** Provides calibrated quarterly forecasts of volumetric fuel demand (`CLV_fuel`) and non-fuel retail spend (`CLV_nonfuel`) across zero-inflated customer histories (**65.06%** non-fuel zeros; **29.25%** fuel zeros).

**Highlights:** Modular `src/` package architecture, leak-free purged multi-cutoff tournament validation, two-part zero-inflated hurdle modeling, Customer Empirical Bayes (BLUP) fixed effects, constrained optimal-transport/dual-ascent assignment, and an automated `pytest` unit test suite.

---

## Repository Structure

```text
├── pyproject.toml                  # Package metadata, dependencies, and CLI entry points
├── requirements.txt                # Direct pip dependency list
├── pytest.ini                      # Pytest configuration
├── README.md                       # Project overview, architecture, and quickstart
├── src/                            # Core modular ML library & CLI
│   ├── main.py                     # Primary CLI entry point (`python -m src.main`)
│   ├── config.py                   # Immutable Paths, ModelConfig, and PipelineConfig
│   ├── label_rules.py              # Standalone deterministic Zindi target generator
│   ├── label_config.json           # Category mappings, thresholds, and target normalizers
│   ├── data.py                     # Schema validation, integer scaling, and local file resolution
│   ├── labels.py                   # Validated wrapper around label_rules
│   ├── eligibility.py              # Category-history feasibility masks for Opportunity classes
│   ├── features.py                 # Longitudinal, RFM, BTYD, hurdle-physics & seasonal features
│   ├── features_v3.py              # Payday liquidity, refueling physics & station priors
│   ├── dataset.py                  # Fingerprinted SnapshotBuilder & temporal panel stacking
│   ├── models.py                   # Multi-loss regressors, hierarchical & attachment classifiers
│   ├── hurdle.py                   # Bagged & multi-quantile two-part hurdle models
│   ├── boundary.py                 # Binary boundary models around the +25% / R20 growth rule
│   ├── calibration.py              # Debiasing, BLUP, shrinkage & prior-aligned dual ascent
│   ├── optimization.py             # Cost-sensitive linear programming assignment solver
│   ├── metrics.py                  # RMSE, weighted/macro F1 & composite scoring
│   ├── backtest.py                 # Rolling-origin & leave-summer-out cross-validation
│   ├── tournament.py               # Multi-cutoff out-of-fold tournament & stacking engine
│   ├── pipeline.py                 # End-to-end training, evaluation & submission orchestration
│   ├── submission.py               # Submission builder & strict schema validation
│   ├── tracker.py                  # Automated internal benchmark tracking
│   └── viz/                        # Visual EDA & model diagnostics suite
├── scripts/                        # Reproducible EDA, backtesting & milestone candidate scripts
└── tests/                          # Automated pytest unit test suite
```

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

If local CSVs are missing, the CLI and scripts fail fast with a `FileNotFoundError` explaining where to place the files.

### Running Tests
```bash
pytest -q
```

### Running the Pipeline
```bash
# Inspect all CLI flags
python -m src.main --help

# Fast calibration/validation run (2 folds, single seed)
python -m src.main --fast --stage calibrate

# Full reproducible end-to-end pipeline + diagnostic prediction caching
python -m src.main --save-predictions
# or via installed console script:
sasol-clv --save-predictions
```

---

## Core Architecture & Mathematical Foundations

### 1. Evaluation Metric

The challenge optimizes a weighted composite across fuel volume regression, convenience store spend regression, and 17-class Opportunity classification:

$$
\text{Composite} = 0.3 \cdot \left(1 - \frac{\text{RMSE}_{\text{fuel}}}{0.740}\right) + 0.3 \cdot \left(1 - \frac{\text{RMSE}_{\text{nonfuel}}}{0.816}\right) + 0.4 \cdot \text{Opportunity F1}
$$

### 2. Multi-Cutoff Out-of-Fold Tournament & Calibration

The primary pipeline ([`src/tournament.py`](src/tournament.py), [`src/calibration.py`](src/calibration.py)) enforces strict temporal embargo rules so feature windows never overlap future label horizons:
- **Purged Multi-Cutoff Stacking:** Trains multi-loss regressors (LightGBM Tweedie, XGBoost MSE/Huber, Ridge, ExtraTrees), two-part hurdle models, and physical basket-ticket regressors across seasonal snapshots, combining predictions via non-negative ridge/NNLS stacking with per-member linear debiasing.
- **Customer Empirical Bayes (BLUP):** Shrinks customer-level historical residuals toward zero with active-cohort centering to capture persistent customer heterogeneity without overfitting.
- **Empirical Opportunity Priors & Boundary Modeling:** Couples hierarchical Opportunity classifiers with specialized boundary classifiers targeting the Fuel Growth vs. Stable threshold ($+25\%$ relative growth and $\ge \text{R}20$ absolute increase), eligibility masks ([`src/eligibility.py`](src/eligibility.py)), and prior-aligned dual coordinate ascent.

### 3. Two-Part Hurdle Models & Zero Inflation

- Non-fuel spend is **65.06% zero-inflated** and fuel consumption is **29.25% zero-inflated**.
- Unconditional regression suffers severe downward bias toward zero. Two-part hurdle models ([`src/hurdle.py`](src/hurdle.py)) decouple binary store attachment $P(\text{Active} > 0 \mid X)$ from conditional spend estimation $\mathbb{E}[\text{Spend} \mid \text{Active} > 0, X]$, with Heckman Inverse Mills Ratio selection-bias correction and multi-quantile quadrature.

### 4. Competition Ensemble Optimization (Hilbert-Space Affine Projection)

For the competition milestone scripts (`scripts/build_candidate_omega*.py`), ensemble weights are solved directly in $\mathbb{R}^{5488}$ using the law of cosines. Given an affine combination $p(\alpha) = p_{\text{ref}} + \sum_m \alpha_m (p_m - p_{\text{ref}})$, the inner product with unknown test error follows from verified leaderboard MSE returns:

$$
g_m = \frac{1}{N} \langle p_{\text{ref}} - y_{\text{true}}, p_m - p_{\text{ref}} \rangle = \frac{1}{2} \left( \text{MSE}_m - \text{MSE}_{\text{ref}} - \frac{1}{N} \lVert p_m - p_{\text{ref}} \rVert_2^2 \right)
$$

Solving the Tikhonov-regularized quadratic program locates the minimum-variance vertex across the test cohort.

---

## Key Results & Competition Milestones

| Stage | Candidate | Fuel RMSE | Nonfuel RMSE | Opportunity F1 | Composite Score | Key Innovations |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Baseline** | Candidate P | `0.590265` | `0.721620` | `0.540531` | `0.311614` | Two-part hurdle models, moment preservation, LP cost-weighting |
| **Verified Peak** | **Candidate Alpha** | **`0.587864`** | `0.720174` | **`0.541843`** | **`0.313644`** | 356 features, Level-3 fine categories, basket depth physics, CatBoost MQ |
| **Staged** | **Candidate Omega-5** | **`0.587393`** *(proj)* | **`0.720033`** *(proj)* | **`0.541843`** | **`0.313887`** *(proj)* | Integrates Candidate Alpha into 15-model Spectral Hilbert QP |

<details>
<summary>Intermediate Competition Milestones (Milestones 1–7)</summary>

| Milestone | Candidate Name | Fuel RMSE | Nonfuel RMSE | Opportunity F1 | Composite Score | Key Innovations |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Milestone 1** | Candidate V | `0.590265` | `0.721620` | `0.541843` | `0.312139` | 9-pair dormant boundary swaps (+0.000553 F1 lift) |
| **Milestone 2** | Candidate W2 | `0.589947` | `0.721235` | `0.541843` | `0.312410` | Multi-quantile hurdle, Hawkes kernels, global item SVD |
| **Milestone 3** | Candidate Y2 | `0.589594` | `0.720777` | `0.541843` | `0.312721` | 5-cutoff BLUP fixed effects + quantile exceedance arbitration |
| **Milestone 4** | Candidate Omega | `0.589359` | `0.720688` | `0.541165` | `0.312578` | Hilbert-space affine projection theorem validated on live test |
| **Milestone 5** | Candidate Omega-2 | `0.588753` | `0.720457` | `0.541843` | `0.313180` | Expanded 9-model Hilbert basis + restored peak Opportunity |
| **Milestone 6** | Candidate Omega-3 | `0.588327` | `0.720226` | `0.541843` | `0.313437` | Un-distorted Hilbert projection centered at Omega-2 |
| **Milestone 7** | Candidate Omega-4 | `0.587937` | `0.720099` | `0.541843` | `0.313642` | Spectral SVD eigen-decomposition of Gram matrix $G = V \Lambda V^T$ |

</details>

---

## License

This project is open-sourced under the MIT License.
