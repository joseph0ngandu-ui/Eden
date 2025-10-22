"""
Eden MVP Orchestrator
Fetch -> Backtests -> Metrics -> Grid Optimization -> Summary
"""

from __future__ import annotations
import json
from pathlib import Path
import sys

# Ensure 'eden' package is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "worker" / "python"))

from run_backtests import run_all_backtests, BacktestConfig
from postprocess_metrics import main as postprocess_main
from grid_optimize import run_grid

import argparse


def main():
    parser = argparse.ArgumentParser(description="Eden MVP Runner")
    parser.add_argument("--instrument", type=str, default="Volatility 100 Index")
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--execution-tfs", type=str, default="M1,M5")
    parser.add_argument("--htf-tfs", type=str, default="15M,1H,4H,1D")
    parser.add_argument(
        "--strategies",
        type=str,
        default="ict,mean_reversion,momentum,price_action,ml_generated,ensemble",
    )
    parser.add_argument(
        "--starting-cash", type=float, default=15.0
    )  # Optimal micro account balance
    parser.add_argument("--risk-perc", type=float, default=2.0)
    parser.add_argument("--min-trade-value", type=float, default=0.50)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument(
        "--skip-grid", action="store_true", help="Skip grid optimization"
    )
    args = parser.parse_args()

    results_dir = Path(args.output_dir)
    results_dir.mkdir(exist_ok=True, parents=True)

    exec_tfs = [t.strip() for t in args.execution_tfs.split(",") if t.strip()]
    htf_tfs = [t.strip() for t in args.htf_tfs.split(",") if t.strip()]
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]

    cfg = BacktestConfig(
        days_back=7 if not (args.start_date and args.end_date) else 0,
        starting_cash=args.starting_cash,
        per_order_risk_fraction=(args.risk_perc or 2.0) / 100.0,
        min_trade_value=args.min_trade_value,
        growth_factor=0.5,
    )

    # Run backtests
    metrics = run_all_backtests(
        cfg=cfg,
        execution_tfs=exec_tfs,
        htf_tfs=htf_tfs,
        strategies=strategies,
        instrument=args.instrument,
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=args.output_dir,
    )

    # Post-process metrics
    consolidated = postprocess_main(results_dir)

    # Grid search optimization (use same timeframes/range)
    best = []
    if not args.skip_grid:
        best = run_grid(
            days_back=None if (args.start_date and args.end_date) else cfg.days_back,
            start=args.start_date,
            end=args.end_date,
            execution_tfs=exec_tfs,
            htf_tfs=htf_tfs,
            results_dir=results_dir,
        )

    # Console summary
    print("\n=== Backtest Summary ===")
    for k, v in consolidated.items():
        print(
            f"{k:20s} | NetPnL: {v.get('net_pnl',0):>9.2f} | Sharpe: {v.get('sharpe',0):>5.2f} | MaxDD%: {v.get('max_drawdown_pct',0):>6.2f} | Trades: {v.get('trades',0):>4d}"
        )

    if best:
        print("\nTop grid configs (best_config.json):")
        print(json.dumps(best[:3], indent=2))
    else:
        print("\nGrid optimization skipped.")

    # Save run config
    (results_dir / "run_config.json").write_text(json.dumps(cfg.__dict__, indent=2))


if __name__ == "__main__":
    main()
