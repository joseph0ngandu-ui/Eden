#!/usr/bin/env python3
"""
Live-Sim (Paper) using VB v1.3 on current profitable symbols.
- Uses per-symbol best params from reports/vb_v1.3_per_symbol_best.json
- Simulates recent period (last 14 days) on M5 MT5 data
- No real orders; produces JSON/CSV in reports/
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import json
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
    # Filter to profitable per-symbol stats
    selected = {}
    for sym, info in data.items():
        stats = info.get('stats', {})
        if stats and float(stats.get('total_pnl', 0)) > 0:
            selected[sym] = info['overrides']
    return selected


def main(start_equity: float = 100.0, days: int = 14):
    best_cfg_path = REPORTS / 'vb_v1.3_per_symbol_best.json'
    if not best_cfg_path.exists():
        print('Best-config file not found: reports/vb_v1.3_per_symbol_best.json')
        sys.exit(1)

    selected = load_profitable_symbols(best_cfg_path)
    if not selected:
        print('No profitable symbols found in best-config file.')
        sys.exit(1)

    start = datetime.now() - timedelta(days=days)
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

    equity = start_equity
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
        'start_equity': start_equity,
        'end_equity': equity,
        'return_percent': (equity / start_equity - 1) * 100,
        'period': {'start': start.isoformat(), 'end': end.isoformat()},
        'selected_symbols': list(selected.keys()),
        'per_symbol_stats': per_symbol_stats,
        'risk_ladder': RISK_LADDER,
    }
    out_json = REPORTS / 'live_sim_vb_summary.json'
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Live-sim VB portfolio: ${start_equity:.2f} -> ${equity:.2f} | Return {summary['return_percent']:.1f}% | Symbols {', '.join(selected.keys())}")
    print(f"Reports written: {trades_csv.name}, {out_json.name}")


if __name__ == '__main__':
    main(100.0, 14)
