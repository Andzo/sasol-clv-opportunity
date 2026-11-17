"""Main entry point for the Sasol CLV & Opportunity prediction challenge."""
import argparse
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    _project_root = str(Path(__file__).resolve().parent.parent)
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

from src.config import PipelineConfig
from src.pipeline import run_pipeline


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description="Sasol CLV & Opportunity ML Pipeline Runner")
    parser.add_argument(
        "--no-cache", action="store_true", help="Bypass feature and snapshot disk caches"
    )
    parser.add_argument(
        "--first-train-cutoff", default="2024-09-01", help="Earliest snapshot cutoff (default: 2024-09-01)"
    )
    parser.add_argument(
        "--calibration-cutoff", default="2025-06-01", help="Snapshot cutoff for blend/weight calibration"
    )
    parser.add_argument(
        "--evaluation-cutoff", default="2025-09-01", help="Out-of-time evaluation snapshot cutoff"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for model training"
    )
    parser.add_argument(
        "--save-predictions", action="store_true", help="Cache evaluation and test predictions for EDA & diagnostics"
    )
    parser.add_argument(
        "--fast", action="store_true", default=True,
        help="2-fold (2025-06-01, 2025-09-01) single-seed configuration used for the final submission (default).",
    )
    parser.add_argument(
        "--full-grid", action="store_true",
        help="Run the 4-fold multi-seed grid instead of the default 2-fold single-seed configuration.",
    )
    parser.add_argument(
        "--stage", choices=["full", "calibrate"], default="full",
        help="'calibrate' stops after holdout evaluation (no retraining/submission) - for calibration/level-2 experiments.",
    )
    parser.add_argument(
        "--reuse-l1", action="store_true",
        help="Reuse cached level-1 OOF predictions (auto-invalidated when models/hurdle/features source or config changes).",
    )
    parser.add_argument(
        "--prob-fusion", choices=["blend", "stack"], default=None,
        help="Opportunity probability fusion: legacy fixed blend or multinomial log-loss stacker.",
    )
    parser.add_argument(
        "--gpu", action="store_true", help="Enable NVIDIA GPU acceleration for CatBoost and XGBoost"
    )
    args = parser.parse_args()

    use_fast = args.fast and not args.full_grid
    model_cls = PipelineConfig().models.__class__
    models_cfg = (
        model_cls.fast(seed=args.seed, use_gpu=args.gpu)
        if use_fast else model_cls(seed=args.seed, use_gpu=args.gpu)
    )
    extra: dict = {"tournament_cutoffs": ("2025-06-01", "2025-09-01")} if use_fast else {}
    if args.prob_fusion:
        extra["prob_fusion"] = args.prob_fusion
    cfg = PipelineConfig(
        models=models_cfg,
        **extra,
        first_train_cutoff=args.first_train_cutoff,
        calibration_cutoff=args.calibration_cutoff,
        evaluation_cutoff=args.evaluation_cutoff,
        use_cache=not args.no_cache,
    )

    try:
        summary = run_pipeline(cfg, save_predictions=args.save_predictions, stage=args.stage, reuse_l1=args.reuse_l1)
        print("\n" + "=" * 60)
        print("PIPELINE EXECUTION SUMMARY")
        print("=" * 60)
        print(f"Status:                 {summary['status']}")
        print(f"Evaluation Cutoff:      {summary['evaluation_cutoff']}")
        print(f"CLV_fuel RMSE:          {summary['metrics']['rmse_fuel']:.4f}")
        print(f"CLV_nonfuel RMSE:       {summary['metrics']['rmse_nonfuel']:.4f}")
        print(f"Opportunity Weighted F1:{summary['metrics']['weighted_f1']:.4f}")
        lb_score = summary.get('leaderboard_composite_score', summary.get('proxy_composite_score', 0.0))
        print(f"Leaderboard Score:      {lb_score:.5f}")
        if 'leaderboard_breakdown' in summary:
            bd = summary['leaderboard_breakdown']
            print(f"  > Fuel Contribution:    {bd.get('score_fuel', 0.0):.5f} (30% weight, max=0.740)")
            print(f"  > Nonfuel Contribution: {bd.get('score_nonfuel', 0.0):.5f} (30% weight, max=0.816)")
            print(f"  > Opportunity Contrib:  {bd.get('score_opportunity', 0.0):.5f} (40% weight)")
        print(f"Legacy Proxy Score:     {summary.get('proxy_composite_score', lb_score):.4f}")
        print("=" * 60)
        return 0
    except FileNotFoundError as e:
        print(f"Missing required local dataset file: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Pipeline failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
