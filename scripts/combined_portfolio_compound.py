#!/usr/bin/env python3
"""
Combined portfolio compounding: choose best per-symbol between VB v1.3 and MA v1.2.
Runs $10, $50, $100 simulations and writes reports.
"""

import sys
from pathlib import Path
from datetime import datetime
import json
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from backtest_volatility_burst import VBBacktester

START = datetime(2025,1,1)
END = datetime(2025,10,31)
REPORTS = Path('reports')

RISK_LADDER = [
    (30, 0.20), (100, 0.10), (500, 0.05), (1000, 0.03), (float('inf'), 0.01)
]

def risk_pct(eq):
    for t,p in RISK_LADDER:
        if eq < t: return p
    return 0.01


def load_vb_best():
    with open(REPORTS / 'vb_v1.3_per_symbol_best.json','r') as f:
        return json.load(f)

def load_ma_results():
    with open(REPORTS / 'ma_v1_2_results.json','r') as f:
        return json.load(f)


def build_trades(vb_best):
    vb_bt = VBBacktester('config/volatility_burst.yml')
    per_symbol = {}
    all_trades = []
    for sym, info in vb_best.items():
        stats = vb_bt.backtest_symbol(sym, START, END, param_overrides=info['overrides'])
        if stats is None: continue
        per_symbol[(sym,'VB')] = stats
        trades = vb_bt.results[sym]['trades']
        for t in trades:
            t['symbol']=sym; t['strategy']='VB'
        all_trades.extend(trades)
    return per_symbol, all_trades


def select_best_symbols(vb_stats, ma_stats):
    selected = {}
    for sym in set([s for s,_ in vb_stats.keys()]).union(set(ma_stats.keys())):
        vb = vb_stats.get((sym,'VB'))
        ma = ma_stats.get(sym, {}).get('stats')
        cand = []
        if vb: cand.append(('VB', vb['total_pnl']))
        if ma: cand.append(('MA', ma['total_pnl']))
        if not cand: continue
        best = max(cand, key=lambda x: x[1])
        if best[1] > 0:
            selected[sym]=best[0]
    return selected


def run_compound(start_equity):
    vb_best = load_vb_best()
    vb_stats, vb_trades = build_trades(vb_best)
    ma_results = load_ma_results()

    chosen = select_best_symbols(vb_stats, ma_results)

    # Collect chosen trades
    chosen_trades = []
    if chosen:
        # add VB trades that are selected
        for t in vb_trades:
            if chosen.get(t['symbol'])=='VB':
                chosen_trades.append(t)
        # add MA trades if selected
        for sym, res in ma_results.items():
            if chosen.get(sym)=='MA':
                for t in res['trades']:
                    t['symbol']=sym; t['strategy']='MA'
                    chosen_trades.append(t)
    df = pd.DataFrame(chosen_trades)
    if df.empty:
        print(f"Start ${start_equity:.2f}: No trades selected.")
        return
    df['exit_time']=pd.to_datetime(df['exit_time'])
    df.sort_values('exit_time', inplace=True)

    eq = start_equity
    eq_curve=[eq]
    results=[]
    daily_counts={}
    for _,row in df.iterrows():
        day = row['exit_time'].date(); daily_counts.setdefault(day,0)
        # ultra-small gating
        if eq<30 and daily_counts[day]>=1: continue
        r = float(row.get('r_value',0.0))
        pnl = eq * risk_pct(eq) * r
        eq += pnl
        eq_curve.append(eq)
        results.append({'time':row['exit_time'],'symbol':row['symbol'],'strategy':row['strategy'],'r':r,'pnl':pnl,'equity':eq})
        daily_counts[day]+=1

    out = {
        'start_equity': start_equity,
        'end_equity': eq,
        'return_percent': (eq/start_equity-1)*100,
        'selected': chosen
    }
    print(f"Combined portfolio: ${start_equity:.2f} -> ${eq:.2f} | Return {out['return_percent']:.1f}% | Symbols {chosen}")
    REPORTS.mkdir(exist_ok=True)
    with open(REPORTS / f'combined_portfolio_{int(start_equity)}.json','w') as f:
        json.dump(out,f,indent=2)

if __name__=='__main__':
    for s in [10.0, 50.0, 100.0]:
        run_compound(s)
