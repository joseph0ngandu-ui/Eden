"""
Grid search optimization for Eden MVP (ensemble-focused)
Searches across risk sizing and core ICT parameters and ranks by objective.
"""
from __future__ import annotations
import itertools
import json
from pathlib import Path
from typing import Dict, List
import sys

import pandas as pd

# Ensure 'eden' package is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'worker' / 'python'))

from run_backtests import BacktestConfig, _prepare_execution_frames
from eden.data.mtf_fetcher import fetch_vix100_data
from eden.strategies.ict import ICTStrategy
from eden.strategies.mean_reversion import MeanReversionStrategy
from eden.strategies.momentum import MomentumStrategy
from eden.strategies.price_action import PriceActionStrategy
from eden.strategies.ml_generated import MLGeneratedStrategy
from eden.backtest.engine import BacktestEngine
from eden.backtest.analyzer import Analyzer


def objective_score(metrics: Dict, penalty: float = 1.0) -> float:
    if not metrics:
        return -1e9
    sharpe = float(metrics.get('sharpe', 0.0))
    max_dd_pct = float(metrics.get('max_drawdown_pct', 100.0))
    return sharpe - penalty * (max_dd_pct / 100.0)


def run_grid(days_back: int | None = 7, start: str | None = None, end: str | None = None, execution_tfs: List[str] | None = None, htf_tfs: List[str] | None = None, results_dir: Path = Path('results')) -> List[Dict]:
    # Fixed strategies for ensemble
    strategies = ['ict', 'mean_reversion', 'momentum', 'price_action', 'ml_generated']

    execution_tfs = execution_tfs or ['M1','M5']
    htf_tfs = htf_tfs or ['15M','1H','4H','1D']

    # Fetch and prepare data once
    from eden.data.mtf_fetcher import MTFDataFetcher
    from datetime import datetime as _dt
    fetcher = MTFDataFetcher()
    if start and end:
        raw = fetcher.fetch_all_timeframes_range(list(set(execution_tfs + htf_tfs)), _dt.fromisoformat(start), _dt.fromisoformat(end), use_cache=True)
    else:
        raw = fetcher.fetch_all_timeframes(list(set(execution_tfs + htf_tfs)), days_back=days_back or 7, use_cache=True)
    fetcher.shutdown()
    exec_frames = _prepare_execution_frames(raw, execution_tfs, htf_tfs)

    # Grid
    entry_thresholds = [0.5, 0.6, 0.7, 0.8]            # maps to ICT min_confidence
    stop_mults = [0.8, 1.2, 1.6, 2.0]
    tp_mults = [1.0, 1.5, 2.0]
    min_trade_values = [0.25, 0.50, 1.00]
    growth_factors = [0.3, 0.5, 0.7]

    trials = []

    for min_conf, stop_m, tp_m, min_tv, gf in itertools.product(
        entry_thresholds, stop_mults, tp_mults, min_trade_values, growth_factors
    ):
        cfg = BacktestConfig(
            days_back=days_back,
            starting_cash=10.0,
            min_confidence=min_conf,
            stop_atr_multiplier=stop_m,
            tp_atr_multiplier=tp_m,
            per_order_risk_fraction=0.02,
            min_trade_value=min_tv,
            growth_factor=gf,
        )
        symbol = 'VIX100'
        all_metrics: Dict[str, Dict] = {}

        # Build ensemble signals per timeframe with current ICT parameters
        for tf, df_exec in exec_frames.items():
            sigs: List[pd.DataFrame] = []
            # ICT with current params
            s = ICTStrategy(
                stop_atr_multiplier=cfg.stop_atr_multiplier,
                tp_atr_multiplier=cfg.tp_atr_multiplier,
                min_confidence=cfg.min_confidence,
            ).on_data(df_exec)
            if s is not None and not s.empty:
                sigs.append(s)
            # Other strategies
            for strat in ['mean_reversion','momentum','price_action','ml_generated']:
                if strat == 'mean_reversion':
                    ss = MeanReversionStrategy().on_data(df_exec)
                elif strat == 'momentum':
                    ss = MomentumStrategy().on_data(df_exec)
                elif strat == 'price_action':
                    ss = PriceActionStrategy().on_data(df_exec)
                else:
                    ss = MLGeneratedStrategy().on_data(df_exec)
                if ss is not None and not ss.empty:
                    sigs.append(ss)
            if not sigs:
                continue
            all_sigs = pd.concat(sigs, ignore_index=True)
            grouped = all_sigs.groupby(['timestamp','side'])['confidence'].mean().reset_index()

            # Run engine with dynamic risk params
            eng = BacktestEngine(
                starting_cash=cfg.starting_cash,
                commission_bps=cfg.commission_bps,
                slippage_bps=cfg.slippage_bps,
                per_order_risk_fraction=cfg.per_order_risk_fraction,
                min_trade_value=cfg.min_trade_value,
                growth_factor=cfg.growth_factor,
            )
            trades = eng.run(df_exec, grouped, symbol=symbol, risk_manager=None)
            an = Analyzer(trades, starting_cash=cfg.starting_cash)
            metrics = an.metrics()
            all_metrics[f'ensemble_{tf}'] = metrics

        # Aggregate metrics: prefer M5 if present, else M1
        primary_key = 'ensemble_M5' if 'ensemble_M5' in all_metrics else 'ensemble_M1'
        primary_metrics = all_metrics.get(primary_key, {})
        score = objective_score(primary_metrics)
        trials.append({
            'params': {
                'min_confidence': min_conf,
                'stop_atr_multiplier': stop_m,
                'tp_atr_multiplier': tp_m,
                'min_trade_value': min_tv,
                'growth_factor': gf,
            },
            'metrics': primary_metrics,
            'score': score
        })

    # Rank and save top 5
    trials_sorted = sorted(trials, key=lambda x: x['score'], reverse=True)[:5]
    (results_dir / 'best_config.json').write_text(json.dumps(trials_sorted, indent=2))
    return trials_sorted


if __name__ == '__main__':
    run_grid(days_back=7)
