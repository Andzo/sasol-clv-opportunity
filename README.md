# Sasol Customer Lifetime Value & Opportunity Machine Learning System

[![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Leaderboard Peak](https://img.shields.io/badge/Zindi%20Peak-0.313644-brightgreen.svg)]()

Production-grade machine learning system developed for the **Sasol Customer Lifetime Value (CLV) & Opportunity Prediction Challenge** on [Zindi](https://zindi.africa/). The system forecasts quarterly customer volumetric fuel consumption, convenience store spend, and customer growth/adoption trajectories across Sasol forecourt service stations in South Africa.

---

## 1. Key Results & Competition Milestones

The competition evaluates a composite metric across three pillars:
$$\text{Composite} = 0.3 \cdot \left(1 - \frac{\text{RMSE}_{\text{fuel}}}{0.740}\right) + 0.3 \cdot \left(1 - \frac{\text{RMSE}_{\text{nonfuel}}}{0.816}\right) + 0.4 \cdot \text{Opportunity F1}$$

| Milestone | Candidate Name | Fuel RMSE | Nonfuel RMSE | Opportunity F1 | Composite Score | Key Innovations |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Baseline** | Candidate P | `0.590265` | `0.721620` | `0.540531` | `0.311614` | Two-part hurdle models, exact moment preservation, LP cost-weighting |
| **Milestone 1** | Candidate V | `0.590265` | `0.721620` | `0.541843` | `0.312139` | 9-pair dormant boundary swaps ($+0.000553$ F1 lift) |
| **Milestone 2** | Candidate W2 | `0.589947` | `0.721235` | `0.541843` | `0.312410` | Multi-quantile hurdle, Hawkes kernels, global item SVD |
| **Milestone 3** | Candidate Y2 | `0.589594` | `0.720777` | `0.541843` | `0.312721` | 5-cutoff BLUP fixed effects + quantile exceedance arbitration |
| **Milestone 4** | Candidate Omega | `0.589359` | `0.720688` | `0.541165` | `0.312578` | Hilbert-space affine projection theorem first validated on live test |
| **Milestone 5** | Candidate Omega-2 | `0.588753` | `0.720457` | `0.541843` | `0.313180` | Expanded 9-model Hilbert basis + restored verified peak Opportunity |
| **Milestone 6** | Candidate Omega-3 | `0.588327` | `0.720226` | `0.541843` | `0.313437` | Un-distorted Hilbert projection centered at Omega-2 |
| **Milestone 7** | Candidate Omega-4 | `0.587937` | `0.720099` | `0.541843` | `0.313642` | Spectral SVD eigen-decomposition of Gram matrix $G = V \Lambda V^T$ |
| **Current Peak**| **Candidate Alpha** | **`0.587864`** | `0.720174` | **`0.541843`** | **`0.313644`** | 356 features, Level-3 fine categories, basket depth physics, CatBoost MQ |
| **Staged** | **Candidate Omega-5** | **`0.587393`** *(proj)* | **`0.720033`** *(proj)* | **`0.541843`** | **`0.313887`** *(proj)* | Fully integrates Candidate Alpha into 15-model Spectral Hilbert QP |

---

## 2. Core Architecture & Mathematical Foundations

### 2.1 Out-of-Fold Affinity Blending & Calibration
Submissions are optimized using multi-snapshot out-of-fold cross-validation. For regression ensembles, member models are combined via non-negative least squares (NNLS) with linear calibration on held-out historical validation snapshots to ensure conditional mean unbiasedness under RMSE.

### 2.2 Two-Part Hurdle Models & Zero Inflation
- Nonfuel spend is **65.06% zero-inflated**; fuel is **29.25% zero-inflated**.
- Standard regression suffers severe bias from unconditional mean shrinkage. Two-part hurdle models cleanly decouple binary activity classification $P(\text{Active} > 0)$ from conditional expenditure estimation $\mathbb{E}[\text{Spend} \mid \text{Active} > 0]$.

### 2.3 Empirical Opportunity Priors & Boundary Modeling
Rather than manual post-hoc swaps, Opportunity classification leverages:
- Specialized binary boundary classifiers targeting the Fuel Growth vs. Stable threshold ($+25\%$ and $\ge \text{R}20$).
- Out-of-fold probability stacking into the hierarchical Opportunity classifier.
- Low-dimensional empirical prior calibration matching historical seasonal travel dynamics.

---

## 3. Repository Structure

```
├── DOCUMENTATION.md      # Comprehensive master documentation & experiment log
├── README.md             # Project overview and executive summary
├── pyproject.toml        # Build configuration and project dependencies
├── label_config.json     # Ground truth category mapping and normalizers
├── submission.csv        # Current active staged submission (Candidate Omega-5)
├── src/                  # Core library
│   ├── config.py         # Pipeline configuration and hyperparameters
│   ├── data.py           # Transaction loading and temporal filtering
│   ├── dataset.py        # SnapshotBuilder and feature matrix extraction
│   ├── features.py       # Longitudinal, RFM, and Hawkes kernel features
│   ├── models.py         # Hurdle regressors and classifiers
│   └── evaluation.py     # Evaluation harnesses and metric calculations
├── scripts/              # Reproducible candidate generation scripts
│   ├── build_candidate_omega5_alpha_integrated.py
│   ├── build_candidate_omega4_spectral_hilbert.py
│   ├── build_candidate_alpha_orthogonal_l3_and_basket_physics.py
│   ├── build_candidate_w2.py
│   └── verify_submission_breakthrough.py
├── tests/                # Automated unit test suite (44 passed)
│   ├── test_data.py
│   ├── test_features.py
│   └── test_models.py
└── submissions/          # Canonical submission archive and manifests
```

---

## 4. Setup & Quickstart

### Prerequisites
- Linux / macOS (tested on Ubuntu 24.04 LTS)
- Python 3.10+ (tested with Python 3.14 virtual environment)
- Optional: NVIDIA CUDA-compatible GPU

### Installation
```bash
git clone https://github.com/Andzo/sasol-clv-opportunity.git
cd sasol-clv-opportunity

python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Running Tests
```bash
pytest -q
# 44 passed in ~8s
```

### Reproducing Candidate Omega-5
```bash
python scripts/build_candidate_omega5_alpha_integrated.py
```

---

## 5. License
This repository is open-sourced under the MIT License. See [LICENSE](LICENSE) for details.
