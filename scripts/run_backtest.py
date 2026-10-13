import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import backtest
from src.config import PipelineConfig
from src.data import load_transactions
from src.dataset import SnapshotBuilder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_backtest")

def main():
    parser = argparse.ArgumentParser(description="Run rolling-origin backtest")
    parser.add_argument("--cutoffs", nargs="+", default=["2025-06-01", "2025-07-01", "2025-08-01", "2025-09-01"],
                        help="Evaluation cutoff dates")
    parser.add_argument("--skip-seasonal", action="store_true", help="Skip seasonal Dec-Feb fold")
    parser.add_argument("--cache-dir", default="reports/backtest", help="Directory to cache fold predictions")
    parser.add_argument("--gpu", action="store_true", help="Enable NVIDIA GPU acceleration for CatBoost and XGBoost")
    args = parser.parse_args()

    cfg = PipelineConfig(models=PipelineConfig().models.__class__(use_gpu=args.gpu))
    tx = load_transactions(cfg)
    builder = SnapshotBuilder(cfg, tx)

    cache_path = Path(args.cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    logger.info("Starting Rolling-Origin Backtest across cutoffs: %s", args.cutoffs)
    summary = backtest.run_backtest(
        cfg=cfg,
        builder=builder,
        eval_cutoffs=args.cutoffs,
        include_seasonal_fold=not args.skip_seasonal,
        cache_dir=cache_path,
    )

    print("\n" + "="*70)
    print("BACKTEST SUMMARY REPORT")
    print("="*70)
    for f in summary.folds:
        print(f"Cutoff {f.cutoff:10s} | Fuel: {f.scores.rmse_fuel:.4f} | NF: {f.scores.rmse_nonfuel:.4f} | F1: {f.scores.weighted_f1:.4f} | Proxy: {f.proxy_score:.4f} | LB Score: {f.lb_score:.4f}")
    if summary.seasonal_fold_result:
        sf = summary.seasonal_fold_result
        print(f"Seasonal {sf.cutoff:10s} | Fuel: {sf.scores.rmse_fuel:.4f} | NF: {sf.scores.rmse_nonfuel:.4f} | F1: {sf.scores.weighted_f1:.4f} | Proxy: {sf.proxy_score:.4f} | LB Score: {sf.lb_score:.4f}")
    print("-" * 70)
    print(f"Mean Across Folds   | Fuel: {summary.mean_fuel_rmse:.4f} | NF: {summary.mean_nonfuel_rmse:.4f} | F1: {summary.mean_weighted_f1:.4f} | Proxy: {summary.mean_proxy_score:.4f} | LB Score: {summary.mean_lb_score:.4f}")
    print("="*70)

if __name__ == "__main__":
    main()
