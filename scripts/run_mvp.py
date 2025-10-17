"""
Eden MVP Orchestrator
Fetch -> Backtests -> Metrics -> Grid Optimization -> Summary
"""
from __future__ import annotations
import json
from pathlib import Path
import sys

# Ensure 'eden' package is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'worker' / 'python'))

from run_backtests import run_all_backtests, BacktestConfig
from postprocess_metrics import main as postprocess_main
from grid_optimize import run_grid

import argparse


def main():
    parser = argparse.ArgumentParser(description='Eden MVP Runner')
    parser.add_argument('--instrument', type=str, default='Volatility 100 Index')
    parser.add_argument('--start-date', type=str, default=None)
    parser.add_argument('--end-date', type=str, default=None)
    parser.add_argument('--execution-tfs', type=str, default='M1,M5')
    parser.add_argument('--htf-tfs', type=str, default='15M,1H,4H,1D')
    parser.add_argument('--strategies', type=str, default='ict,mean_reversion,momentum,price_action,ml_generated,ensemble')
    parser.add_argument('--starting-cash', type=float, default=15.0)  # Optimal micro account balance
    parser.add_argument('--risk-perc', type=float, default=2.0)
    parser.add_argument('--min-trade-value', type=float, default=0.50)
    parser.add_argument('--output-dir', type=str, default='results')
    parser.add_argument('--skip-grid', action='store_true', help='Skip grid optimization')
    parser.add_argument('--phase2', action='store_true', help='Run Phase-2 baseline vs improved comparison')
    parser.add_argument('--phase3', action='store_true', help='Run Phase-3 ML pipeline, training, and comparison')
    parser.add_argument('--debug', action='store_true', help='Enable extra logging and checks for phase3')
    args = parser.parse_args()

    results_dir = Path(args.output_dir)
    results_dir.mkdir(exist_ok=True, parents=True)

    exec_tfs = [t.strip() for t in args.execution_tfs.split(',') if t.strip()]
    htf_tfs = [t.strip() for t in args.htf_tfs.split(',') if t.strip()]
    strategies = [s.strip() for s in args.strategies.split(',') if s.strip()]

    cfg = BacktestConfig(
        days_back=7 if not (args.start_date and args.end_date) else 0,
        starting_cash=args.starting_cash,
        per_order_risk_fraction=(args.risk_perc or 2.0) / 100.0,
        min_trade_value=args.min_trade_value,
        growth_factor=0.5,
    )

    if not args.phase2 and not args.phase3:
        # Run backtests (single run)
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
    elif args.phase2:
        # Phase-2: baseline vs improved
        import traceback
        seed = 42
        run_meta = {
            'seed': seed,
            'instrument': args.instrument,
            'start_date': args.start_date or '2025-10-07',
            'end_date': args.end_date or '2025-10-14',
            'execution_tfs': exec_tfs,
            'htf_tfs': htf_tfs,
        }
        (results_dir / 'run_config.json').write_text(json.dumps(run_meta, indent=2))
        baseline_dir = results_dir / 'baseline'
        phase2_dir = results_dir / 'phase2_run'
        baseline_dir.mkdir(parents=True, exist_ok=True)
        phase2_dir.mkdir(parents=True, exist_ok=True)
        # Baseline (disable new features)
        cfg_base = BacktestConfig(
            days_back=cfg.days_back,
            starting_cash=10.0,
            per_order_risk_fraction=0.02,
            min_trade_value=0.50,
            growth_factor=0.5,
            enable_vol_norm=False,
            htf_strict_mode=False,
            controller_enable=False,
        )
        try:
            run_all_backtests(
                cfg=cfg_base,
                execution_tfs=exec_tfs,
                htf_tfs=htf_tfs,
                strategies=strategies,
                instrument=args.instrument,
                start_date=args.start_date or '2025-10-07',
                end_date=args.end_date or '2025-10-14',
                output_dir=str(baseline_dir),
            )
        except Exception as e:
            (results_dir / 'error_summary.txt').write_text(f"Baseline run failed: {e}\n{traceback.format_exc()}")
        # Phase-2 run (enable features)
        cfg_p2 = BacktestConfig(
            days_back=cfg.days_back,
            starting_cash=10.0,
            per_order_risk_fraction=0.02,
            min_trade_value=0.50,
            growth_factor=0.5,
            enable_vol_norm=True,
            htf_strict_mode=True,
            controller_enable=True,
            conf_cut_low=0.55,
            conf_cut_mid=0.70,
            conf_cut_high=0.85,
            risk_mult_low=0.35,
            risk_mult_mid=0.60,
            risk_mult_high=1.00,
            volatility_cap=2.6,
        )
        try:
            run_all_backtests(
                cfg=cfg_p2,
                execution_tfs=exec_tfs,
                htf_tfs=htf_tfs,
                strategies=strategies,
                instrument=args.instrument,
                start_date=args.start_date or '2025-10-07',
                end_date=args.end_date or '2025-10-14',
                output_dir=str(phase2_dir),
            )
        except Exception as e:
            (results_dir / 'error_summary.txt').write_text(f"Phase-2 run failed: {e}\n{traceback.format_exc()}")
        # Post-process each then compare
        from postprocess_metrics import main as _pp
        _pp(baseline_dir)
        _pp(phase2_dir)
        # Comparison
        try:
            from scripts_compare_phase2 import compare_runs
        except Exception:
            compare_runs = None
        if compare_runs:
            compare_runs(baseline_dir, phase2_dir, results_dir)
        # Phase-2 grid search (small)
        try:
            from grid_optimize import run_grid_phase2
            top = run_grid_phase2(start=run_meta['start_date'], end=run_meta['end_date'], execution_tfs=exec_tfs, htf_tfs=htf_tfs, results_dir=results_dir)
            if top:
                print("Top Phase-2 configs saved to best_config_phase2.json")
        except Exception as e:
            (results_dir / 'error_summary.txt').write_text(((results_dir / 'error_summary.txt').read_text() if (results_dir / 'error_summary.txt').exists() else '') + f"\nPhase-2 grid failed: {e}")

    # Optional grid search (original)
    best = []
    if not args.skip_grid and not args.phase2:
        best = run_grid(
            days_back=None if (args.start_date and args.end_date) else cfg.days_back,
            start=args.start_date,
            end=args.end_date,
            execution_tfs=exec_tfs,
            htf_tfs=htf_tfs,
            results_dir=results_dir,
        )

    # Save basic summary if single run
    if not args.phase2:
        print("\n=== Backtest Summary ===")
        for k, v in consolidated.items():
            print(f"{k:20s} | NetPnL: {v.get('net_pnl',0):>9.2f} | Sharpe: {v.get('sharpe',0):>5.2f} | MaxDD%: {v.get('max_drawdown_pct',0):>6.2f} | Trades: {v.get('trades',0):>4d}")
        if best:
            print("\nTop grid configs (best_config.json):")
            print(json.dumps(best[:3], indent=2))
        else:
            print("\nGrid optimization skipped.")
    elif args.phase3:
        # Phase-3 pipeline orchestration
        import traceback
        results_dir.mkdir(parents=True, exist_ok=True)
        run_meta = {
            'seed': 20251017,
            'instrument': args.instrument,
            'start_date': args.start_date or '2025-10-07',
            'end_date': args.end_date or '2025-10-14',
            'execution_tfs': exec_tfs,
            'htf_tfs': htf_tfs,
            'base_risk_per_trade': 0.0125,
            'min_trade_value': 0.50,
            'growth_factor': 0.5,
            'volatility_adapter_enabled': True,
            'htf_strict_mode': True,
            'ml_rolling_window_days': 21,
            'ml_eval_window_days': 7,
        }
        (results_dir / 'run_config.json').write_text(json.dumps(run_meta, indent=2))
        # Debug: import check + unit tests
        try:
            subprocess.run([sys.executable, 'worker/python/eden/debug/import_checker.py', '--output', str(results_dir / 'import_status.json')], check=False)
            subprocess.run([sys.executable, 'worker/python/eden/debug/unit_tests.py', '--output', str(results_dir / 'unit_tests_report.json')], check=False)
        except Exception as e:
            (results_dir / 'error_summary.txt').write_text(f"Debug checks failed: {e}\n{traceback.format_exc()}")
        # Baseline run (reuse Phase-2 baseline settings)
        baseline_dir = results_dir / 'baseline'
        phase3_dir = results_dir / 'phase3_run'
        baseline_dir.mkdir(parents=True, exist_ok=True)
        phase3_dir.mkdir(parents=True, exist_ok=True)
        cfg_base = BacktestConfig(
            days_back=cfg.days_back,
            starting_cash=10.0,
            per_order_risk_fraction=0.02,
            min_trade_value=0.50,
            growth_factor=0.5,
            enable_vol_norm=False,
            htf_strict_mode=False,
            controller_enable=False,
        )
        try:
            run_all_backtests(cfg=cfg_base, execution_tfs=exec_tfs, htf_tfs=htf_tfs, strategies=strategies, instrument=args.instrument, start_date=run_meta['start_date'], end_date=run_meta['end_date'], output_dir=str(baseline_dir))
        except Exception as e:
            (results_dir / 'error_summary.txt').write_text(((results_dir / 'error_summary.txt').read_text() if (results_dir / 'error_summary.txt').exists() else '') + f"\nBaseline failed: {e}\n{traceback.format_exc()}")
        # Train stages (placeholder)
        try:
            subprocess.run([sys.executable, 'worker/python/eden/ml/trainer.py', '--stage', 'all', '--train-window-days', '21', '--eval-window-days', '7', '--save-dir', str(results_dir / 'models'), '--seed', '20251017'], check=False)
        except Exception as e:
            (results_dir / 'error_summary.txt').write_text(((results_dir / 'error_summary.txt').read_text() if (results_dir / 'error_summary.txt').exists() else '') + f"\nTrainer failed: {e}\n{traceback.format_exc()}")
        # Phase-3 run (enable staged pipeline)
        cfg_p3 = BacktestConfig(
            days_back=cfg.days_back,
            starting_cash=10.0,
            per_order_risk_fraction=0.0125,
            min_trade_value=0.50,
            growth_factor=0.5,
            enable_vol_norm=True,
            htf_strict_mode=True,
            controller_enable=True,
            enable_stage_pipeline=True,
        )
        try:
            # Enable staged pipeline by passing via BacktestEngine inside run_backtests
            # We'll piggyback decision_log_path used in run_backtests
            run_all_backtests(cfg=cfg_p3, execution_tfs=exec_tfs, htf_tfs=htf_tfs, strategies=strategies, instrument=args.instrument, start_date=run_meta['start_date'], end_date=run_meta['end_date'], output_dir=str(phase3_dir))
        except Exception as e:
            (results_dir / 'error_summary.txt').write_text(((results_dir / 'error_summary.txt').read_text() if (results_dir / 'error_summary.txt').exists() else '') + f"\nPhase3 failed: {e}\n{traceback.format_exc()}")
        # Postprocess both and compare
        try:
            from scripts_compare_phase2 import compare_runs
            compare_runs(baseline_dir, phase3_dir, results_dir)
        except Exception as e:
            (results_dir / 'error_summary.txt').write_text(((results_dir / 'error_summary.txt').read_text() if (results_dir / 'error_summary.txt').exists() else '') + f"\nComparison failed: {e}\n{traceback.format_exc()}")


if __name__ == '__main__':
    main()
