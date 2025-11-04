#!/usr/bin/env python3
"""
Live-Sim (Paper) using VB v1.3 on current profitable symbols.
- Prefers merged best params in reports/vb_v1.3_per_symbol_best_merged.json if present
- Falls back to reports/vb_v1.3_per_symbol_best.json
- Simulates recent period (last 14 days) on M5 MT5 data
- No real orders; produces JSON/CSV in reports/
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import json
import argparse
import pandas as pd
import MetaTrader5 as mt5

sys.path.insert(0, str(Path(__file__).parent))
from backtest_volatility_burst import VBBacktester

REPORTS = Path('reports')
REPORTS.mkdir(exist_ok=True)

RISK_LADDER = [
    (30, 0.20),
    (100, 0.10),
    (500, 0.05),
    (1000, 0.03),
    (float('inf'), 0.01),
]

def risk_pct(equity: float) -> float:
    for thresh, pct in RISK_LADDER:
        if equity < thresh:
            return pct
    return 0.01


def load_profitable_symbols(best_path: Path) -> dict:
    with open(best_path, 'r') as f:
        data = json.load(f)
    # Filter to profitable symbols; if merged file, require long-term profitability
    selected = {}
    for sym, info in data.items():
        # Prefer long-term profitability gate when available (merged file)
        if 'long_term' in info and isinstance(info['long_term'], dict):
            lt_stats = (info['long_term'] or {}).get('stats', {})
            if lt_stats and float(lt_stats.get('total_pnl', 0) or 0) > 0:
                selected[sym] = info['overrides']
        else:
            stats = info.get('stats', {})
            if stats and float(stats.get('total_pnl', 0) or 0) > 0:
                selected[sym] = info['overrides']
    return selected


def main():
    ap = argparse.ArgumentParser(description='Live-sim VB v1.3 portfolio')
    ap.add_argument('--start-equity', type=float, default=100.0)
    ap.add_argument('--days', type=int, default=14)
    ap.add_argument('--best', type=str, default='', help='Path to per-symbol best JSON to use')
    args = ap.parse_args()

    # Resolve best config path with preference for merged best
    if args.best:
        best_cfg_path = Path(args.best)
    else:
        merged = REPORTS / 'vb_v1.3_per_symbol_best_merged.json'
        best_cfg_path = merged if merged.exists() else (REPORTS / 'vb_v1.3_per_symbol_best.json')

    if not best_cfg_path.exists():
        print(f'Best-config file not found: {best_cfg_path}')
        sys.exit(1)

    selected = load_profitable_symbols(best_cfg_path)
    if not selected:
        print('No profitable symbols found in best-config file.')
        sys.exit(1)

    start = datetime.now() - timedelta(days=args.days)
    end = datetime.now()

    bt = VBBacktester('config/volatility_burst.yml')

    all_trades = []
    per_symbol_stats = {}

    for sym, overrides in selected.items():
        stats = bt.backtest_symbol(sym, start, end, param_overrides=overrides)
        if not stats:
            continue
        per_symbol_stats[sym] = stats
        trades = bt.results[sym]['trades']
        for t in trades:
            t['symbol'] = sym
            t['strategy'] = 'VB_v1.3'
        all_trades.extend(trades)

    if not all_trades:
        print('No trades generated for live-sim period.')
        sys.exit(0)

    # Chronological
    df = pd.DataFrame(all_trades)
    df['exit_time'] = pd.to_datetime(df['exit_time'])
    df = df.sort_values('exit_time').reset_index(drop=True)

    equity = args.start_equity
    equity_curve = [equity]
    results = []
    daily_counts = {}

    for _, row in df.iterrows():
        day = pd.to_datetime(row['exit_time']).date()
        daily_counts.setdefault(day, 0)
        conf = float(row.get('confidence') or 0.0)
        # Ultra-small gating when equity < $30
        if equity < 30.0:
            if row['symbol'] != 'Volatility 75 Index':
                continue
            if conf < 0.90 or daily_counts[day] >= 1:
                continue
        r = float(row.get('r_value', 0.0))
        rp = risk_pct(equity)
        pnl = equity * rp * r
        equity += pnl
        equity_curve.append(equity)
        results.append({
            'time': row['exit_time'],
            'symbol': row['symbol'],
            'strategy': row['strategy'],
            'r': r,
            'confidence': conf,
            'risk_pct': rp,
            'pnl': pnl,
            'equity': equity,
        })
        daily_counts[day] += 1

    # Save outputs
    trades_csv = REPORTS / 'live_sim_vb_trades.csv'
    pd.DataFrame(results).to_csv(trades_csv, index=False)

    summary = {
        'timestamp': datetime.now().isoformat(),
        'start_equity': args.start_equity,
        'end_equity': equity,
        'return_percent': (equity / args.start_equity - 1) * 100,
        'period': {'start': start.isoformat(), 'end': end.isoformat()},
        'selected_symbols': list(selected.keys()),
        'per_symbol_stats': per_symbol_stats,
        'risk_ladder': RISK_LADDER,
        'best_config_used': str(best_cfg_path),
    }
    out_json = REPORTS / 'live_sim_vb_summary.json'
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Live-sim VB portfolio: ${args.start_equity:.2f} -> ${equity:.2f} | Return {summary['return_percent']:.1f}% | Symbols {', '.join(selected.keys())}")
    print(f"Best file: {best_cfg_path}")
    print(f"Reports written: {trades_csv.name}, {out_json.name}")


if __name__ == '__main__':
    main()
