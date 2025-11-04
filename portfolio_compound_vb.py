#!/usr/bin/env python3
"""
Portfolio-level compounding for small accounts using VB v1.3 per-symbol best params.
Includes gold (XAUUSD). Produces JSON/CSV reports, equity curve, DD.
"""

import sys
from pathlib import Path
from datetime import datetime
import json
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from backtest_volatility_burst import VBBacktester

START = datetime(2025, 1, 1)
END = datetime(2025, 10, 31)
REPORTS_DIR = Path('reports')

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


def load_best_params(path: Path) -> dict:
    with open(path, 'r') as f:
        data = json.load(f)
    return {sym: d['overrides'] for sym, d in data.items()}


def compute_max_drawdown(equity_curve):
    peak = -float('inf')
    max_dd = 0.0
    for e in equity_curve:
        peak = max(peak, e)
        dd = (peak - e) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)
    return max_dd


def run_portfolio_compounding(start_equity=100.0):
    REPORTS_DIR.mkdir(exist_ok=True)
    best_cfg_path = REPORTS_DIR / 'vb_v1.3_per_symbol_best.json'
    best = load_best_params(best_cfg_path)

    symbols = list(best.keys())

    bt = VBBacktester('config/volatility_burst.yml')

    all_trades = []
    per_symbol_stats = {}

    for sym in symbols:
        stats = bt.backtest_symbol(sym, START, END, param_overrides=best[sym])
        if stats is None:
            continue
        per_symbol_stats[sym] = stats
        trades = bt.results[sym]['trades']
        for t in trades:
            t['symbol'] = sym
            t['strategy'] = 'VB_v1.3'
        all_trades.extend(trades)

    # Filter to profitable symbols for compounding
    selected_symbols = [s for s, st in per_symbol_stats.items() if st['total_pnl'] > 0]

    df = pd.DataFrame(all_trades)
    if df.empty:
        print('No trades generated.')
        return

    df = df[df['symbol'].isin(selected_symbols)].copy()
    df['exit_time'] = pd.to_datetime(df['exit_time'])
    df = df.sort_values('exit_time').reset_index(drop=True)

    equity = start_equity
    equity_curve = [equity]
    results = []

    # Adaptive gating for ultra-small accounts
    daily_counts = {}
    for _, row in df.iterrows():
        day = pd.to_datetime(row['exit_time']).date()
        daily_counts.setdefault(day, 0)

        conf = float(row.get('confidence') or 0.0)
        sym = row['symbol']

        # Symbol + confidence gating by equity tier
        allowed = set(selected_symbols)
        if equity < 30.0:
            allowed = {'Volatility 75 Index'}
            if conf < 0.90 or daily_counts[day] >= 1:
                continue
        elif equity < 50.0:
            allowed = {'Volatility 75 Index', 'XAUUSD'}
            if conf < 0.85 or daily_counts[day] >= 2:
                continue
        elif equity < 100.0:
            allowed = {'Volatility 75 Index', 'XAUUSD', 'Crash 500 Index'}
            if conf < 0.80 or daily_counts[day] >= 3:
                continue

        if sym not in allowed:
            continue

        r = float(row.get('r_value', 0.0))
        rp = risk_pct(equity)
        pnl = equity * rp * r
        equity += pnl
        equity_curve.append(equity)
        results.append({
            'time': row['exit_time'],
            'symbol': sym,
            'strategy': row['strategy'],
            'r': r,
            'confidence': conf,
            'risk_pct': rp,
            'pnl': pnl,
            'equity': equity,
        })
        daily_counts[day] += 1

    max_dd = compute_max_drawdown(equity_curve)

    # Outputs
    df_res = pd.DataFrame(results)
    df_res.to_csv(REPORTS_DIR / 'portfolio_vb_v1.3_trades.csv', index=False)

    if not df_res.empty and 'time' in df_res.columns:
        monthly = df_res.copy()
        monthly['month'] = pd.to_datetime(monthly['time']).dt.to_period('M')
        monthly_summary = monthly.groupby('month')['pnl'].sum().to_frame('PnL')
        monthly_summary.to_csv(REPORTS_DIR / 'portfolio_vb_v1.3_monthly.csv')
    else:
        monthly_summary = pd.DataFrame(columns=['PnL'])

    summary = {
        'start_equity': start_equity,
        'end_equity': equity,
        'return_percent': (equity/start_equity - 1) * 100,
        'max_drawdown_percent': max_dd * 100,
        'selected_symbols': selected_symbols,
        'per_symbol_stats': per_symbol_stats,
        'risk_ladder': RISK_LADDER,
        'period': {'start': START.isoformat(), 'end': END.isoformat()},
    }
    with open(REPORTS_DIR / 'portfolio_vb_v1.3_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Portfolio compounding: ${start_equity:.2f} -> ${equity:.2f} | Return {summary['return_percent']:.1f}% | MaxDD {summary['max_drawdown_percent']:.1f}%")
    print(f"Selected symbols: {', '.join(selected_symbols)}")
    print('Reports written to reports/*.csv, *.json')


if __name__ == '__main__':
    run_portfolio_compounding(100.0)
