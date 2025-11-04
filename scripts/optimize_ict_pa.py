#!/usr/bin/env python3
"""
Optimize ICT+PA hybrid parameters over a small grid for a given date range.
Outputs top results to reports/ict_pa_optimization.json
"""
import sys
from pathlib import Path
from datetime import datetime
from itertools import product
import argparse
import json
import pandas as pd
import MetaTrader5 as mt5

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from strategies.signals import ICTPA
from risk_ladder import RiskLadder, PositionSizer

SYMBOLS_DEFAULT = [
    'XAUUSD',
    'Volatility 75 Index',
    'Crash 500 Index',
    'Boom 500 Index',
]


def fetch(symbol: str, start: datetime, end: datetime):
    # Try MT5 first
    if mt5.initialize():
        try:
            rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M5, start, end)
            if rates is not None:
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                return df.sort_values('time').reset_index(drop=True)
        finally:
            mt5.shutdown()
    # CSV fallback
    data_dir = Path('data/mt5_feeds')
    for name in [f"{symbol}_M5.csv", f"{symbol}_M1.csv"]:
        csv = data_dir / name
        if csv.exists():
            d = pd.read_csv(csv)
            ts_col = 'time' if 'time' in d.columns else ('timestamp' if 'timestamp' in d.columns else None)
            if ts_col is None:
                continue
            d['time'] = pd.to_datetime(d[ts_col])
            d = d[(d['time']>=start) & (d['time']<=end)].copy()
            if name.endswith('_M1.csv'):
                d = d.set_index('time').resample('5T').agg({'open':'first','high':'max','low':'min','close':'last','tick_volume':'sum' if 'tick_volume' in d.columns else 'sum'})
                d = d.dropna().reset_index()
            else:
                d = d.sort_values('time').reset_index(drop=True)
            return d
    return None


def backtest_one(sym, df, params: dict):
    strat = ICTPA({'params': params})
    df.attrs['symbol'] = sym
    trades = []
    positions = []
    sigs = strat.generate_signals(df)
    by_idx = {t.bar_index: t for t in sigs}
    for idx in range(len(df)):
        d = df.iloc[:idx+1]
        acts = strat.manage_position(d, sym)
        for a in acts:
            if a['action']=='close':
                pos = positions.pop()
                exit_price = a.get('price', d.iloc[-1]['close'])
                profit = (exit_price - pos['entry_price']) if pos['direction']=='LONG' else (pos['entry_price'] - exit_price)
                trades.append({
                    'profit': profit,
                    'r': float(a.get('r_value', 0.0) or 0.0),
                    'exit_time': d.iloc[-1]['time']
                })
        if sym not in strat.open_positions and idx in by_idx:
            t = by_idx[idx]
            strat.on_trade_open(t)
            positions.append({'entry_price': t.entry_price, 'direction': t.direction})
    if positions:
        last = df.iloc[-1]
        for pos in positions:
            profit = (last['close'] - pos['entry_price']) if pos['direction']=='LONG' else (pos['entry_price'] - last['close'])
            trades.append({'profit': profit, 'r': 0.0, 'exit_time': last['time']})
    prof = [t['profit'] for t in trades]
    wins = [p for p in prof if p>0]
    losses = [p for p in prof if p<0]
    return {
        'trades': len(trades),
        'pnl': sum(prof),
        'wr': (len(wins)/len(trades)*100.0) if trades else 0.0,
        'pf': (sum(wins)/abs(sum(losses))) if losses else 0.0,
        'trades_list': trades,
    }


def main():
    ap = argparse.ArgumentParser(description='Optimize ICT+PA')
    ap.add_argument('--start', type=str, required=True)
    ap.add_argument('--end', type=str, required=True)
    ap.add_argument('--symbols', type=str, default=','.join(SYMBOLS_DEFAULT))
    ap.add_argument('--start-equity', type=float, default=100.0)
    ap.add_argument('--tp', type=str, default='2.0,2.5,3.0')
    ap.add_argument('--sl', type=str, default='0.8,1.0')
    ap.add_argument('--minc', type=str, default='0.6,0.7')
    args = ap.parse_args()

    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end)
    symbols = [s.strip() for s in args.symbols.split(',') if s.strip()]

    # Load data once per symbol
    data = {}
    for s in symbols:
        df = fetch(s, start, end)
        if df is not None and len(df)>=100:
            data[s] = df

    if not data:
        print('No data for selected symbols/period.')
        return

    grid = {
        'min_confidence': [float(x) for x in args.minc.split(',') if x],
        'tp_atr_mult': [float(x) for x in args.tp.split(',') if x],
        'sl_atr_mult': [float(x) for x in args.sl.split(',') if x],
        'w_bias': [0.25, 0.35],
        'w_pa': [0.45, 0.55],
        'w_kz': [0.1, 0.2],
    }
    keys = list(grid.keys())
    combos = [dict(zip(keys, vals)) for vals in product(*[grid[k] for k in keys])]

    results = []
    for ov in combos:
        # Collect all trades with r-values across symbols
        all_trades = []
        per = {}
        gross_pnl = 0.0
        total_trades = 0
        wr_num = wr_den = 0.0
        pf_num = pf_den = 0.0
        for s, df in data.items():
            stats = backtest_one(s, df, ov)
            per[s] = {k:v for k,v in stats.items() if k!='trades_list'}
            gross_pnl += stats['pnl']
            total_trades += stats['trades']
            if stats['trades']>0:
                wr_num += stats['wr']*stats['trades']
                wr_den += stats['trades']
                pf_num += stats['pf']*stats['trades']
                pf_den += stats['trades']
            for t in stats['trades_list']:
                t['symbol'] = s
                all_trades.append(t)
        # Risk ladder compounding on combined trades
        all_trades.sort(key=lambda x: x.get('exit_time'))
        rl = RiskLadder(initial_balance=args.start_equity, growth_mode_enabled=True, high_aggression_below=30.0, equity_step_size=25.0, equity_step_drawdown_limit=0.10)
        eq = args.start_equity
        curve = [eq]
        for t in all_trades:
            r = float(t.get('r', 0.0) or 0.0)
            risk_pct = rl.get_adjusted_risk_pct()
            risk_amt = eq * (risk_pct/100.0)
            pnl = risk_amt * r
            eq += pnl
            curve.append(eq)
            rl.update_balance(eq)
        end_equity = eq
        # Compute drawdown
        peak = -1e18
        max_dd = 0.0
        for e in curve:
            peak = max(peak, e)
            if peak>0:
                max_dd = max(max_dd, (peak - e)/peak)
        wr = (wr_num/wr_den) if wr_den>0 else 0.0
        pf = (pf_num/pf_den) if pf_den>0 else 0.0
        score = end_equity - (max_dd*args.start_equity*0.5)
        results.append({'overrides': ov, 'portfolio': {'end_equity': end_equity, 'max_dd_pct': max_dd*100, 'trades': total_trades, 'wr': wr, 'pf': pf, 'score': score}, 'per_symbol': per})
        print(f"Test {ov} -> EndEq ${end_equity:.2f} | Trades {total_trades} | WR {wr:.1f}% | PF {pf:.2f} | MaxDD {max_dd*100:.1f}%")

    results.sort(key=lambda x: (x['portfolio']['score']), reverse=True)
    out = {
        'timestamp': datetime.now().isoformat(),
        'period': {'start': start.isoformat(), 'end': end.isoformat()},
        'symbols': symbols,
        'start_equity': args.start_equity,
        'top': results[:10],
        'best': results[0] if results else None,
    }
    Path('reports').mkdir(exist_ok=True)
    with open('reports/ict_pa_optimization.json','w') as f:
        json.dump(out, f, indent=2)
    print('Saved reports/ict_pa_optimization.json')


if __name__=='__main__':
    main()
