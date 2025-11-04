#!/usr/bin/env python3
"""
Optimize combined MA + ICT/PA strategy with RiskLadder compounding.
- Builds trades from both strategies per symbol, merges by exit_time, compounds from $start_equity.
- Searches small grids for ICT/PA and MA params.
- Uses MT5 first, then CSV fallback (data/mt5_feeds/*_M5.csv or *_M1.csv).
Outputs reports/combined_ictpa_ma_optimization.json
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
from ma_v1_2 import MA_v1_2
from risk_ladder import RiskLadder

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


def trades_ictpa(sym: str, df: pd.DataFrame, params: dict):
    ict = ICTPA({'params': params})
    df.attrs['symbol'] = sym
    trades = []
    positions = []
    sigs = ict.generate_signals(df)
    by_idx = {t.bar_index: t for t in sigs}
    for idx in range(len(df)):
        d = df.iloc[:idx+1]
        acts = ict.manage_position(d, sym)
        for a in acts:
            if a['action']=='close':
                pos = positions.pop()
                exit_price = a.get('price', d.iloc[-1]['close'])
                profit = (exit_price - pos['entry_price']) if pos['direction']=='LONG' else (pos['entry_price'] - exit_price)
                trades.append({'exit_time': d.iloc[-1]['time'], 'r': float(a.get('r_value', 0.0) or 0.0), 'strategy': 'ICTPA'})
        if sym not in ict.open_positions and idx in by_idx:
            t = by_idx[idx]
            ict.on_trade_open(t)
            positions.append({'entry_time': t.entry_time, 'entry_price': t.entry_price, 'direction': t.direction})
    if positions:
        last = df.iloc[-1]
        for pos in positions:
            trades.append({'exit_time': last['time'], 'r': 0.0, 'strategy': 'ICTPA'})
    return trades


def trades_ma(sym: str, df: pd.DataFrame, params: dict):
    ma = MA_v1_2('config/ma_v1_2.yml')
    # apply overrides on MA
    mp = {}
    if 'fast_period' in params: mp['fast_period'] = params['fast_period']
    if 'slow_period' in params: mp['slow_period'] = params['slow_period']
    if 'tp_atr_multiplier' in params: mp['tp_atr_multiplier'] = params['tp_atr_multiplier']
    if 'sl_atr_multiplier' in params: mp['sl_atr_multiplier'] = params['sl_atr_multiplier']
    if mp:
        ma.update_params(**mp)
    df.attrs['symbol'] = sym
    trades = []
    positions = []
    sigs = ma.generate_signals(df)
    by_idx = {t.bar_index: t for t in sigs}
    for idx in range(len(df)):
        d = df.iloc[:idx+1]
        acts = ma.manage_position(d, sym)
        for a in acts:
            if a['action']=='close':
                pos = positions.pop()
                trades.append({'exit_time': d.iloc[-1]['time'], 'r': float(a.get('r_value', 0.0) or 0.0), 'strategy': 'MA'})
        if sym not in ma.open_positions and idx in by_idx:
            t = by_idx[idx]
            ma.on_trade_open(t)
            positions.append({'entry_time': t.entry_time, 'entry_price': t.entry_price, 'direction': t.direction})
    if positions:
        last = df.iloc[-1]
        for _ in positions:
            trades.append({'exit_time': last['time'], 'r': 0.0, 'strategy': 'MA'})
    return trades


def compound_trades(all_trades, start_equity: float):
    all_trades.sort(key=lambda x: x['exit_time'])
    rl = RiskLadder(initial_balance=start_equity, growth_mode_enabled=True, high_aggression_below=30.0, equity_step_size=25.0, equity_step_drawdown_limit=0.10)
    eq = start_equity
    curve = [eq]
    for t in all_trades:
        r = float(t.get('r', 0.0) or 0.0)
        risk_pct = rl.get_adjusted_risk_pct()
        pnl = eq * (risk_pct/100.0) * r
        eq += pnl
        curve.append(eq)
        rl.update_balance(eq)
    # drawdown
    peak = -1e18
    max_dd = 0.0
    for e in curve:
        peak = max(peak, e)
        if peak>0:
            max_dd = max(max_dd, (peak - e)/peak)
    return eq, max_dd*100, curve


def run_once(data: dict, ict_params: dict, ma_params: dict, start_equity: float):
    portfolio_trades = []
    per_symbol = {}
    total = {'trades': 0}
    for sym, df in data.items():
        t_ict = trades_ictpa(sym, df, ict_params)
        t_ma = trades_ma(sym, df, ma_params)
        merged = t_ict + t_ma
        end_eq, max_dd, curve = compound_trades(merged, start_equity)
        per_symbol[sym] = {'trades': len(merged), 'end_equity': end_eq, 'max_dd_pct': max_dd}
        total['trades'] += len(merged)
        portfolio_trades.extend(merged)
    end_eq, max_dd, curve = compound_trades(portfolio_trades, start_equity)
    return {'end_equity': end_eq, 'max_dd_pct': max_dd, 'trades': total['trades'], 'per_symbol': per_symbol}


def main():
    ap = argparse.ArgumentParser(description='Optimize Combined MA + ICT/PA with RiskLadder')
    ap.add_argument('--start', type=str, default='2023-01-01')
    ap.add_argument('--end', type=str, default='2025-11-03')
    ap.add_argument('--symbols', type=str, default=','.join(SYMBOLS_DEFAULT))
    ap.add_argument('--start-equity', type=float, default=1000.0)
    args = ap.parse_args()

    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end)
    symbols = [s.strip() for s in args.symbols.split(',') if s.strip()]

    data = {}
    for s in symbols:
        df = fetch(s, start, end)
        if df is not None and len(df)>=100:
            data[s] = df
    if not data:
        print('No data found.')
        return

    # Grids (kept small but high-RR for ICT/PA; small tweaks for MA)
    ict_grid = {
        'min_confidence': [0.6],
        'tp_atr_mult': [2.0, 2.5, 3.0],
        'sl_atr_mult': [0.8, 1.0],
        'w_bias': [0.25],
        'w_pa': [0.55],
        'w_kz': [0.1],
    }
    ma_grid = {
        'fast_period': [3],
        'slow_period': [10],
        'tp_atr_multiplier': [1.5, 2.0],
        'sl_atr_multiplier': [1.0, 1.2],
    }

    ict_keys = list(ict_grid.keys())
    ma_keys = list(ma_grid.keys())
    ict_combos = [dict(zip(ict_keys, vals)) for vals in product(*[ict_grid[k] for k in ict_keys])]
    ma_combos = [dict(zip(ma_keys, vals)) for vals in product(*[ma_grid[k] for k in ma_keys])]

    results = []
    for ictp in ict_combos:
        for mapar in ma_combos:
            res = run_once(data, ictp, mapar, args.start_equity)
            score = res['end_equity'] - (res['max_dd_pct']*2)  # penalize drawdown
            results.append({'ict': ictp, 'ma': mapar, 'portfolio': res, 'score': score})
            print(f"ICT {ictp} + MA {mapar} -> EndEq ${res['end_equity']:.2f} | Trades {res['trades']} | MaxDD {res['max_dd_pct']:.1f}%")

    results.sort(key=lambda x: x['score'], reverse=True)
    out = {
        'timestamp': datetime.now().isoformat(),
        'period': {'start': start.isoformat(), 'end': end.isoformat()},
        'symbols': symbols,
        'start_equity': args.start_equity,
        'top': results[:10],
        'best': results[0] if results else None,
    }
    Path('reports').mkdir(exist_ok=True)
    with open('reports/combined_ictpa_ma_optimization.json','w') as f:
        json.dump(out, f, indent=2)
    print('Saved reports/combined_ictpa_ma_optimization.json')


if __name__=='__main__':
    main()
