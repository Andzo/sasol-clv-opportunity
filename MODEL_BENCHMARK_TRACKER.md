# Model Benchmark Tracker: Highest-Scoring Champion Record

This file automatically tracks internal evaluation runs across causal validation folds.
The **Champion Record** represents the permanent model-to-beat benchmark.
Whenever a run achieves a strictly higher holdout composite score, the champion is automatically updated.

## Current Champion (Model to Beat)

- **Composite Score:** `0.29760`
- **Fuel RMSE:** `0.5967`
- **Nonfuel RMSE:** `0.7451`
- **Opportunity F1:** `0.5335`
- **Git Commit:** `b75a41f9237c6dade5837f6dbe7f559e46003fbd`
- **Timestamp:** `2026-10-17 03:20:35`
- **Run ID:** `run_20261017_032035`

### Metric Breakdown

- Fuel Score Weight (30%): `0.05811`
- Nonfuel Score Weight (30%): `0.02608`
- Opportunity Score Weight (40%): `0.21341`

### Submission Class Seat Allocations (Sanity Gates)

| Opportunity Class | Count | Gate Range | Gate Status |
|---|---|---|---|
| Stable | 1963 | 1,930 – 1,980 | Valid |
| Fuel Growth | 1583 | 1,555 – 1,600 | Valid |
| Inactivity | 1456 | 1,420 – 1,460 | Valid |
| Retail Growth + Adoption | 486 | 485 – 525 | Valid |
| Adoption (Pure) | 4 | ≤ 50 | Valid |

### Validation Fold Breakdown

| Fold | Cutoff Date | Fuel RMSE | Nonfuel RMSE | Opp F1 | Composite |
|---|---|---|---|---|---|
| Winter | 2025-06-01 | 0.5934 | 0.7340 | 0.5037 | 0.2911 |
| MidWinter | 2025-07-01 | 0.5941 | 0.7398 | 0.4799 | 0.2791 |
| LateWinter | 2025-08-01 | 0.6030 | 0.7378 | 0.4944 | 0.2821 |
| Spring | 2025-09-01 | 0.5981 | 0.7461 | 0.4984 | 0.2826 |

## Historical Internal Runs

| Timestamp | Git Commit | Composite Score | Fuel RMSE | Nonfuel RMSE | Opp F1 | Champion? |
|---|---|---|---|---|---|---|
| 2026-10-17 20:00:46 | `c03b35e` | **0.28880** | 0.5988 | 0.7447 | 0.5134 | No |
| 2026-10-17 11:25:16 | `1729452` | **0.29719** | 0.5970 | 0.7452 | 0.5330 | No |
| 2026-10-17 03:20:35 | `b75a41f` | **0.29760** | 0.5967 | 0.7451 | 0.5335 | 🏆 YES |
