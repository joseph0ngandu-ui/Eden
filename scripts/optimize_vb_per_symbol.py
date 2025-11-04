#!/usr/bin/env python3
"""
Optimize Volatility Burst v1.3 per symbol (small grid) and save best params.
"""

import sys
from pathlib import Path
from itertools import product
from datetime import datetime
import json

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from backtest_volatility_burst import VBBacktester

SYMBOLS = [
    "Volatility 75 Index",
    "Volatility 100 Index",
    "Boom 500 Index",
    "Crash 500 Index",
    "Boom 1000 Index",
    "Step Index",
    "XAUUSD",
]

GRID = {
    'confidence_threshold': [0.6, 0.7, 0.8],
    'tp_atr_multiplier': [1.2, 1.5],
    'sl_atr_multiplier': [1.0, 1.2],
}

START = datetime(2025, 1, 1)
END = datetime(2025, 10, 31)


def optimize_symbol(symbol: str):
    keys = list(GRID.keys())
    combos = list(product(*[GRID[k] for k in keys]))
    best = None
    bt = VBBacktester("config/volatility_burst.yml")

    for combo in combos:
        overrides = {k: v for k, v in zip(keys, combo)}
        stats = bt.backtest_symbol(symbol, START, END, param_overrides=overrides)
        if stats is None:
            continue
        score = stats['total_pnl']
        if (best is None) or (score > best['stats']['total_pnl']):
            best = {
                'symbol': symbol,
                'overrides': overrides,
                'stats': stats,
            }
    return best


def main():
    results = {}
    best_list = []
    for sym in SYMBOLS:
        print(f"Optimizing {sym}...")
        best = optimize_symbol(sym)
        if best:
            results[sym] = best
            best_list.append(best)
            print(f"  Best {sym}: {best['overrides']} | PnL ${best['stats']['total_pnl']:+.2f} | WR {best['stats']['win_rate']:.1f}% | PF {best['stats']['profit_factor']:.2f}")
        else:
            print(f"  No valid results for {sym}")

    out_dir = Path('reports')
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / 'vb_v1.3_per_symbol_best.json'
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved per-symbol best configs to {out_file}")


if __name__ == '__main__':
    main()
