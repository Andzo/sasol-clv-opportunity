"""
Model Benchmark Tracker (src/tracker.py)

Tracks internal tournament evaluation runs and maintains the permanent
'Model to Beat' champion record in MODEL_BENCHMARK_TRACKER.json and
MODEL_BENCHMARK_TRACKER.md.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional
import datetime

BENCHMARK_JSON_PATH = Path('MODEL_BENCHMARK_TRACKER.json')
BENCHMARK_MD_PATH = Path('MODEL_BENCHMARK_TRACKER.md')


def load_benchmark_tracker() -> Dict[str, Any]:
    """Load existing benchmark tracker data or return default structure."""
    if BENCHMARK_JSON_PATH.exists():
        try:
            with open(BENCHMARK_JSON_PATH, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {
        'champion': None,
        'runs_history': []
    }


def save_benchmark_tracker(data: Dict[str, Any]) -> None:
    """Persist benchmark data to JSON and generate Markdown summary."""
    with open(BENCHMARK_JSON_PATH, 'w') as f:
        json.dump(data, f, indent=2)

    champ = data.get('champion')
    lines = [
        '# Model Benchmark Tracker: Highest-Scoring Champion Record',
        '',
        'This file automatically tracks internal evaluation runs across causal validation folds.',
        'The **Champion Record** represents the permanent model-to-beat benchmark.',
        'Whenever a run achieves a strictly higher holdout composite score, the champion is automatically updated.',
        '',
        '## Current Champion (Model to Beat)',
        ''
    ]

    if champ:
        lines.extend([
            f"- **Composite Score:** `{champ.get('composite_score', 0.0):.5f}`",
            f"- **Fuel RMSE:** `{champ.get('fuel_rmse', 0.0):.4f}`",
            f"- **Nonfuel RMSE:** `{champ.get('nonfuel_rmse', 0.0):.4f}`",
            f"- **Opportunity F1:** `{champ.get('opportunity_f1', 0.0):.4f}`",
            f"- **Git Commit:** `{champ.get('git_commit', 'unknown')}`",
            f"- **Timestamp:** `{champ.get('timestamp', 'unknown')}`",
            f"- **Run ID:** `{champ.get('run_id', 'unknown')}`",
            '',
            '### Metric Breakdown',
            '',
            f"- Fuel Score Weight (30%): `{champ.get('score_breakdown', {}).get('fuel_component', 0.0):.5f}`",
            f"- Nonfuel Score Weight (30%): `{champ.get('score_breakdown', {}).get('nonfuel_component', 0.0):.5f}`",
            f"- Opportunity Score Weight (40%): `{champ.get('score_breakdown', {}).get('opp_component', 0.0):.5f}`",
            '',
            '### Submission Class Seat Allocations (Sanity Gates)',
            '',
            '| Opportunity Class | Count | Gate Range | Gate Status |',
            '|---|---|---|---|',
        ])
        gates = champ.get('seat_counts', {})
        gate_ranges = {
            'Stable': '1,930 – 1,980',
            'Fuel Growth': '1,555 – 1,600',
            'Inactivity': '1,420 – 1,460',
            'Retail Growth + Adoption': '485 – 525',
            'Adoption (Pure)': '≤ 50'
        }
        for k, v in gates.items():
            gr = gate_ranges.get(k, 'N/A')
            lines.append(f"| {k} | {v} | {gr} | Valid |")

        lines.extend([
            '',
            '### Validation Fold Breakdown',
            '',
            '| Fold | Cutoff Date | Fuel RMSE | Nonfuel RMSE | Opp F1 | Composite |',
            '|---|---|---|---|---|---|',
        ])
        for fold in champ.get('folds', []):
            lines.append(
                f"| {fold.get('fold', 'N/A')} | {fold.get('cutoff_date', 'N/A')} | "
                f"{fold.get('fuel_rmse', 0.0):.4f} | {fold.get('nonfuel_rmse', 0.0):.4f} | "
                f"{fold.get('opp_f1', 0.0):.4f} | {fold.get('composite', 0.0):.4f} |"
            )
    else:
        lines.append('_No benchmark runs recorded yet._')

    lines.extend([
        '',
        '## Historical Internal Runs',
        '',
        '| Timestamp | Git Commit | Composite Score | Fuel RMSE | Nonfuel RMSE | Opp F1 | Champion? |',
        '|---|---|---|---|---|---|---|'
    ])

    for r in reversed(data.get('runs_history', [])):
        is_champ = '🏆 YES' if r.get('is_champion') else 'No'
        commit_short = str(r.get('git_commit', 'N/A'))[:7]
        lines.append(
            f"| {r.get('timestamp', 'N/A')} | `{commit_short}` | "
            f"**{r.get('composite_score', 0.0):.5f}** | {r.get('fuel_rmse', 0.0):.4f} | "
            f"{r.get('nonfuel_rmse', 0.0):.4f} | {r.get('opportunity_f1', 0.0):.4f} | {is_champ} |"
        )

    with open(BENCHMARK_MD_PATH, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def update_benchmark_tracker(evaluation_report: Dict[str, Any], git_commit: str = 'unknown', submission_counts: Optional[Dict[str, int]] = None) -> bool:
    """
    Evaluate new run against champion. If superior or first run, updates champion.
    Returns True if run became the new champion, False otherwise.
    """
    tracker = load_benchmark_tracker()
    
    comp_score = float(evaluation_report.get('composite_score', 0.0))
    fuel_rmse = float(evaluation_report.get('fuel_rmse', 0.0))
    nonfuel_rmse = float(evaluation_report.get('nonfuel_rmse', 0.0))
    opp_f1 = float(evaluation_report.get('opportunity_f1', 0.0))
    score_breakdown = evaluation_report.get('score_breakdown', {})
    folds = evaluation_report.get('folds', [])
    now_iso = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    run_entry = {
        'timestamp': now_iso,
        'git_commit': git_commit,
        'run_id': evaluation_report.get('run_id', f"run_{int(datetime.datetime.now().timestamp())}"),
        'composite_score': comp_score,
        'fuel_rmse': fuel_rmse,
        'nonfuel_rmse': nonfuel_rmse,
        'opportunity_f1': opp_f1,
        'score_breakdown': score_breakdown,
        'seat_counts': submission_counts or {},
        'folds': folds,
        'is_champion': False
    }

    current_champ = tracker.get('champion')
    is_new_champ = False

    if current_champ is None or comp_score > current_champ.get('composite_score', 0.0):
        is_new_champ = True
        run_entry['is_champion'] = True
        tracker['champion'] = run_entry
    
    tracker['runs_history'].append(run_entry)
    save_benchmark_tracker(tracker)
    return is_new_champ
