#!/usr/bin/env python3
"""
Optimize Volatility Burst v1.3 parameters across symbols.

Searches over small grid for confidence_threshold and TP/SL multipliers.
Saves top-performing combos.
"""

import sys
from pathlib import Path
from itertools import product
from datetime import datetime
import json

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from backtest_volatility_burst import VBBacktester


def run_optimization():
    symbols = [
        "Volatility 75 Index",
        "Volatility 100 Index",
        "Boom 500 Index",
        "Crash 500 Index",
        "Boom 1000 Index",
        "Step Index",
        # "XAUUSD",  # Uncomment if you want to include gold
    ]

    start = datetime(2025, 1, 1)
    end = datetime(2025, 10, 31)

    grid = {
        'confidence_threshold': [0.6, 0.7, 0.8],
        'tp_atr_multiplier': [1.2, 1.5, 2.0],
        'sl_atr_multiplier': [0.8, 1.0, 1.2],
    }

    keys = list(grid.keys())
    combos = list(product(*[grid[k] for k in keys]))

    results_summary = []

    for combo in combos:
        overrides = {k: v for k, v in zip(keys, combo)}
        bt = VBBacktester("config/volatility_burst.yml")
        total_pnl = 0.0
        total_trades = 0
        total_wins = 0
        total_losses = 0
        per_symbol = {}

        for sym in symbols:
            stats = bt.backtest_symbol(sym, start, end, param_overrides=overrides)
            if stats is None:
                continue
            total_pnl += stats['total_pnl']
            total_trades += stats['total_trades']
            total_wins += stats['winning_trades']
            total_losses += stats['losing_trades']
            per_symbol[sym] = stats

        wr = (total_wins / total_trades * 100) if total_trades else 0
        results_summary.append({
            'overrides': overrides,
            'total_pnl': total_pnl,
            'total_trades': total_trades,
            'win_rate': wr,
            'per_symbol': per_symbol,
        })
        print(f"Tried {overrides} -> PnL {total_pnl:+.2f} | Trades {total_trades} | WR {wr:.1f}%")

    # Sort by total_pnl desc
    results_summary.sort(key=lambda x: x['total_pnl'], reverse=True)

    out_dir = Path('reports')
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / 'vb_v1.3_optimization_results.json'
    with open(out_file, 'w') as f:
        json.dump(results_summary, f, indent=2)

    print(f"\nTop 5 combos:")
    for row in results_summary[:5]:
        print(row['overrides'], '| PnL:', f"{row['total_pnl']:+.2f}", '| WR:', f"{row['win_rate']:.1f}%")
    print(f"\nSaved full results to {out_file}")


if __name__ == '__main__':
    run_optimization()
