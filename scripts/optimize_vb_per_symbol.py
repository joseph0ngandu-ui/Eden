#!/usr/bin/env python3
"""
Optimize Volatility Burst v1.3 per symbol (small grid) and save best params.
Adds CLI args for date range and non-destructive outputs, with optional merge to preserve long-term bests.
"""

import sys
from pathlib import Path
from itertools import product
from datetime import datetime
import argparse
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


def optimize_symbol(symbol: str, start: datetime, end: datetime):
    keys = list(GRID.keys())
    combos = list(product(*[GRID[k] for k in keys]))
    best = None
    bt = VBBacktester("config/volatility_burst.yml")

    for combo in combos:
        overrides = {k: v for k, v in zip(keys, combo)}
        stats = bt.backtest_symbol(symbol, start, end, param_overrides=overrides)
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


def merge_results(old_path: Path, new_results: dict, merged_out: Path):
    """Merge old long-term bests with new results without discarding old.
    Selection rule per symbol:
      - If new exists and is profitable (total_pnl>0), prefer new; else keep old.
      - If both profitable, choose higher profit_factor; tie-breaker: higher total_pnl.
    Keep compatibility by setting overrides/stats to selected, and store both snapshots under keys 'long_term' and 'recent'.
    """
    merged = {}
    old = {}
    if old_path and old_path.exists():
        with open(old_path, 'r') as f:
            old = json.load(f)

    symbols = set(old.keys()) | set(new_results.keys())
    for sym in symbols:
        old_rec = old.get(sym)
        new_rec = new_results.get(sym)
        if old_rec and not new_rec:
            # Only old exists
            sel = old_rec
            merged[sym] = {**sel, 'long_term': old_rec, 'recent': None, 'selected': 'old'}
            continue
        if new_rec and not old_rec:
            # Only new exists
            sel = new_rec
            merged[sym] = {**sel, 'long_term': None, 'recent': new_rec, 'selected': 'new'}
            continue
        # Both exist
        old_stats = old_rec['stats']
        new_stats = new_rec['stats']
        old_pf = float(old_stats.get('profit_factor', 0) or 0)
        new_pf = float(new_stats.get('profit_factor', 0) or 0)
        old_pnl = float(old_stats.get('total_pnl', 0) or 0)
        new_pnl = float(new_stats.get('total_pnl', 0) or 0)
        # Selection
        if new_pnl > 0 and (old_pnl <= 0 or (new_pf > old_pf) or (new_pf == old_pf and new_pnl > old_pnl)):
            sel = new_rec
            selected = 'new'
        else:
            sel = old_rec
            selected = 'old'
        merged[sym] = {**sel, 'long_term': old_rec, 'recent': new_rec, 'selected': selected}

    merged_out.parent.mkdir(exist_ok=True)
    with open(merged_out, 'w') as f:
        json.dump(merged, f, indent=2)
    print(f"Saved merged best configs to {merged_out}")


def main():
    ap = argparse.ArgumentParser(description="Optimize VB v1.3 per-symbol with non-destructive outputs")
    ap.add_argument('--start', type=str, default='2025-01-01', help='Start date YYYY-MM-DD')
    ap.add_argument('--end', type=str, default='2025-10-31', help='End date YYYY-MM-DD')
    ap.add_argument('--out', type=str, default='reports/vb_v1.3_per_symbol_best.json', help='Output JSON path')
    ap.add_argument('--merge-with', type=str, default='', help='Existing best JSON to preserve/merge')
    ap.add_argument('--merged-out', type=str, default='reports/vb_v1.3_per_symbol_best_merged.json', help='Merged output JSON path')
    args = ap.parse_args()

    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end)

    results = {}
    for sym in SYMBOLS:
        print(f"Optimizing {sym} ({args.start} → {args.end})...")
        best = optimize_symbol(sym, start, end)
        if best:
            results[sym] = best
            print(f"  Best {sym}: {best['overrides']} | PnL ${best['stats']['total_pnl']:+.2f} | WR {best['stats']['win_rate']:.1f}% | PF {best['stats']['profit_factor']:.2f}")
        else:
            print(f"  No valid results for {sym}")

    out_path = Path(args.out)
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved per-symbol best configs to {out_path}")

    # Optional merge to preserve long-term bests
    if args.merge_with:
        merge_results(Path(args.merge_with), results, Path(args.merged_out))


if __name__ == '__main__':
    main()
