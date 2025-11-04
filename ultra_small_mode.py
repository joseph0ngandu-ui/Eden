#!/usr/bin/env python3
"""
UltraSmall Mode - V75-only, 1 trade/day, confidence >= 0.97, strict $ risk cap.
Grid-search VB overrides to find best compounding for $10 and $50 starts.
Outputs JSON/CSV under reports/.
"""

import sys
from pathlib import Path
from datetime import datetime
import json
import pandas as pd
from itertools import product

sys.path.insert(0, str(Path(__file__).parent))
from backtest_volatility_burst import VBBacktester

START = datetime(2025,1,1)
END = datetime(2025,10,31)
REPORTS = Path('reports'); REPORTS.mkdir(exist_ok=True)
SYMBOL = 'Volatility 75 Index'

# UltraSmall gating constants
CONF_GATE = 0.97
MAX_TRADES_PER_DAY = 1

# Risk ladder (percent) for tiny balances
RISK_LADDER = [
    (30, 0.20), (50, 0.10), (100, 0.10), (float('inf'), 0.05)
]

# Dollar risk caps by equity tier
RISK_CAPS = [
    (20, 0.50), (30, 1.00), (50, 2.00), (float('inf'), 10.00)
]

def risk_pct(eq: float) -> float:
    for t,p in RISK_LADDER:
        if eq < t: return p
    return 0.05

def risk_cap(eq: float) -> float:
    for t,c in RISK_CAPS:
        if eq < t: return c
    return 10.0


def backtest_with_overrides(overrides: dict):
    bt = VBBacktester('config/volatility_burst.yml')
    stats = bt.backtest_symbol(SYMBOL, START, END, param_overrides=overrides)
    if stats is None: return None, []
    trades = bt.results[SYMBOL]['trades']
    # ensure fields
    for t in trades:
        t['symbol'] = SYMBOL
        t['strategy'] = 'VB_ultra'
        t['confidence'] = float(t.get('confidence') or 0.0)
        t['entry_atr'] = float(t.get('entry_atr') or 0.0)
        t['entry_time'] = pd.to_datetime(t.get('entry_time')) if t.get('entry_time') else None
        t['exit_time'] = pd.to_datetime(t.get('exit_time')) if t.get('exit_time') else None
    df = pd.DataFrame(trades)
    if df.empty:
        return stats, []
    df.sort_values('exit_time', inplace=True)
    return stats, df


# Session windows (UTC hours) and volatility quantile
SESSION_WINDOWS = [(6, 12), (14, 18), (20, 23)]
VOL_Q = 0.80

def in_session(ts):
    if ts is None: return False
    h = ts.hour
    for a,b in SESSION_WINDOWS:
        if a <= h <= b:
            return True
    return False

def compound_ultra(df: pd.DataFrame, start_equity: float) -> dict:
    if df is None or len(df)==0:
        return {'start': start_equity, 'end': start_equity, 'return_pct': 0.0, 'equity_curve': [start_equity]}
    # Apply session gating and volatility quantile on entry
    df_f = df.copy()
    df_f = df_f[df_f['entry_time'].apply(in_session)]
    if not df_f.empty and 'entry_atr' in df_f.columns:
        thr = df_f['entry_atr'].quantile(VOL_Q)
        df_f = df_f[df_f['entry_atr'] >= thr]
    if df_f.empty:
        return {'start': start_equity, 'end': start_equity, 'return_pct': 0.0, 'equity_curve': [start_equity]}

    eq = start_equity
    curve = [eq]
    daily_counts = {}
    kept = []
    for _, row in df_f.iterrows():
        day = row['exit_time'].date()
        daily_counts.setdefault(day, 0)
        conf = float(row.get('confidence') or 0.0)
        if conf < CONF_GATE:  # confidence gate
            continue
        if daily_counts[day] >= MAX_TRADES_PER_DAY:
            continue
        r = float(row.get('r_value', 0.0))
        rp = risk_pct(eq)
        cap = risk_cap(eq)
        risk_amt = min(eq * rp, cap)
        pnl = risk_amt * r
        eq += pnl
        curve.append(eq)
        kept.append({'time': row['exit_time'], 'entry_time': row.get('entry_time'), 'r': r, 'conf': conf, 'entry_atr': row.get('entry_atr'), 'risk_pct': rp, 'risk_amt': risk_amt, 'pnl': pnl, 'equity': eq})
        daily_counts[day] += 1
    return {'start': start_equity, 'end': eq, 'return_pct': (eq/start_equity - 1)*100, 'equity_curve': curve, 'trades': kept}


def run_grid():
    grid = {
        'confidence_threshold': [0.6, 0.7],   # VB internal; gating enforces 0.97
        'tp_atr_multiplier': [1.2, 1.5, 2.0],
        'sl_atr_multiplier': [0.8, 1.0, 1.2],
    }
    keys = list(grid.keys())
    combos = [dict(zip(keys, vals)) for vals in product(*[grid[k] for k in keys])]

    results = []
    best = None
    for ov in combos:
        stats, df_trades = backtest_with_overrides(ov)
        if isinstance(df_trades, list) and len(df_trades)==0:
            score = {'ov': ov, 'end10': 0, 'end50': 0, 'score': -1}
        elif isinstance(df_trades, pd.DataFrame) and df_trades.empty:
            score = {'ov': ov, 'end10': 0, 'end50': 0, 'score': -1}
        else:
            c10 = compound_ultra(df_trades, 10.0)
            c50 = compound_ultra(df_trades, 50.0)
            score_val = (c10['end'] - 10.0) + (c50['end'] - 50.0)
            score = {'ov': ov, 'end10': c10['end'], 'end50': c50['end'], 'score': score_val}
        results.append(score)
        if best is None or score['score'] > best['score']:
            best = score
        print(f"Test {ov} -> $10->{score.get('end10',0):.2f}, $50->{score.get('end50',0):.2f}")

    results.sort(key=lambda x: x['score'], reverse=True)
    with open(REPORTS / 'ultrasmall_v75_grid.json','w') as f:
        json.dump(results, f, indent=2)

    top = results[0]
    # Rebuild best trades and emit detailed CSV for $10/$50 runs
    _, df_trades = backtest_with_overrides(top['ov'])
    c10 = compound_ultra(df_trades, 10.0)
    c50 = compound_ultra(df_trades, 50.0)
    pd.DataFrame(c10['trades']).to_csv(REPORTS / 'ultrasmall_v75_trades_10.csv', index=False)
    pd.DataFrame(c50['trades']).to_csv(REPORTS / 'ultrasmall_v75_trades_50.csv', index=False)
    summary = {'best_overrides': top['ov'], 'c10': {'start': 10.0, 'end': c10['end'], 'return_pct': c10['return_pct']}, 'c50': {'start': 50.0, 'end': c50['end'], 'return_pct': c50['return_pct']}}
    with open(REPORTS / 'ultrasmall_v75_summary.json','w') as f:
        json.dump(summary, f, indent=2)
    print(f"UltraSmall best: {summary}")

if __name__=='__main__':
    run_grid()
