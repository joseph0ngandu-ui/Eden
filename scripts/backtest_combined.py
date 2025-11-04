#!/usr/bin/env python3
"""
Combined backtest (MA + ICT/PA) using MT5-only data with advanced metrics.
Prompts for date range if not provided; prints MT5 connection status.
"""
import sys
from pathlib import Path
from datetime import datetime
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


def prompt_date(prompt_str: str) -> datetime:
    while True:
        s = input(prompt_str).strip()
        try:
            return datetime.fromisoformat(s)
        except Exception:
            print('Please use YYYY-MM-DD')


def connect_mt5() -> bool:
    ok = mt5.initialize()
    print(f"Connected to MT5: {ok}")
    return ok


def fetch_mt5(symbol: str, start: datetime, end: datetime):
    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M5, start, end)
    if rates is None:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df.sort_values('time').reset_index(drop=True)


def trades_ictpa(sym, df, params):
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
                trades.append({'entry_time': pos['entry_time'], 'exit_time': d.iloc[-1]['time'], 'r': float(a.get('r_value',0.0) or 0.0), 'strategy':'ICTPA'})
        if sym not in ict.open_positions and idx in by_idx:
            t = by_idx[idx]
            ict.on_trade_open(t)
            positions.append({'entry_time': t.entry_time, 'entry_price': t.entry_price, 'direction': t.direction})
    if positions:
        last = df.iloc[-1]
        for pos in positions:
            trades.append({'entry_time': pos['entry_time'], 'exit_time': last['time'], 'r': 0.0, 'strategy':'ICTPA'})
    return trades


def trades_ma(sym, df, params):
    ma = MA_v1_2('config/ma_v1_2.yml')
    mp = {}
    for k in ('fast_period','slow_period','tp_atr_multiplier','sl_atr_multiplier'):
        if k in params: mp[k] = params[k]
    if mp: ma.update_params(**mp)
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
                trades.append({'entry_time': pos['entry_time'], 'exit_time': d.iloc[-1]['time'], 'r': float(a.get('r_value',0.0) or 0.0), 'strategy':'MA'})
        if sym not in ma.open_positions and idx in by_idx:
            t = by_idx[idx]
            ma.on_trade_open(t)
            positions.append({'entry_time': t.entry_time, 'entry_price': t.entry_price, 'direction': t.direction})
    if positions:
        last = df.iloc[-1]
        for pos in positions:
            trades.append({'entry_time': pos['entry_time'], 'exit_time': last['time'], 'r': 0.0, 'strategy':'MA'})
    return trades


def metrics_from_trades(trades, start_equity: float):
    # consecutive
    pnl_signs = [1 if t['r']>0 else (-1 if t['r']<0 else 0) for t in trades]
    max_w = max_l = cur_w = cur_l = 0
    for s in pnl_signs:
        if s>0:
            cur_w += 1; cur_l = 0
        elif s<0:
            cur_l += 1; cur_w = 0
        max_w = max(max_w, cur_w)
        max_l = max(max_l, cur_l)
    # holding
    holds = []
    for t in trades:
        try:
            holds.append((pd.to_datetime(t['exit_time']) - pd.to_datetime(t['entry_time'])).total_seconds())
        except Exception:
            pass
    avg_hold_sec = float(pd.Series(holds).mean()) if holds else 0.0
    # ladder compounding
    rl = RiskLadder(initial_balance=start_equity, growth_mode_enabled=True, high_aggression_below=30.0, equity_step_size=25.0, equity_step_drawdown_limit=0.10)
    eq = start_equity
    curve = [eq]
    for t in trades:
        risk_pct = rl.get_adjusted_risk_pct()
        eq += eq * (risk_pct/100.0) * float(t['r'])
        rl.update_balance(eq)
        curve.append(eq)
    peak = -1e18; max_dd = 0.0
    for e in curve:
        peak = max(peak, e)
        if peak>0:
            max_dd = max(max_dd, (peak - e)/peak)
    return {'max_consecutive_wins': max_w, 'max_consecutive_losses': max_l, 'avg_hold_seconds': avg_hold_sec, 'end_equity_ladder': eq, 'max_drawdown_percent_ladder': max_dd*100}


def main():
    ap = argparse.ArgumentParser(description='Combined MT5 backtest (MA + ICT/PA) with prompts')
    ap.add_argument('--start', type=str, default='')
    ap.add_argument('--end', type=str, default='')
    ap.add_argument('--symbols', type=str, default=','.join(SYMBOLS_DEFAULT))
    ap.add_argument('--start-equity', type=float, default=1000.0)
    args = ap.parse_args()

    if not connect_mt5():
        print('MT5 connection failed. Exiting.')
        return

    start = datetime.fromisoformat(args.start) if args.start else prompt_date('Start date (YYYY-MM-DD): ')
    end = datetime.fromisoformat(args.end) if args.end else prompt_date('End date (YYYY-MM-DD): ')
    symbols = [s.strip() for s in args.symbols.split(',') if s.strip()]

    ict_params = {'min_confidence': 0.6, 'tp_atr_mult': 2.0, 'sl_atr_mult': 1.0, 'w_bias': 0.25, 'w_pa': 0.55, 'w_kz': 0.1}
    ma_params = {'fast_period': 3, 'slow_period': 10, 'tp_atr_multiplier': 2.0, 'sl_atr_multiplier': 1.0}

    per_symbol = {}
    for sym in symbols:
        df = fetch_mt5(sym, start, end)
        if df is None or len(df)<100:
            print(f"{sym}: no MT5 data")
            continue
        print(f"{sym}: data loaded {len(df)} bars")
        t1 = trades_ictpa(sym, df, ict_params)
        t2 = trades_ma(sym, df, ma_params)
        merged = sorted(t1 + t2, key=lambda x: x['exit_time'])
        m = metrics_from_trades(merged, args.start_equity)
        per_symbol[sym] = {'trades': len(merged), **m}
        print(f"{sym}: Trades {len(merged)} | EndEq ${m['end_equity_ladder']:.2f} | MaxDD {m['max_drawdown_percent_ladder']:.1f}% | MaxWinRow {m['max_consecutive_wins']} | MaxLossRow {m['max_consecutive_losses']} | AvgHold {m['avg_hold_seconds']/60:.1f}m")

    out = {'timestamp': datetime.now().isoformat(), 'period': {'start': start.isoformat(), 'end': end.isoformat()}, 'symbols': symbols, 'start_equity': args.start_equity, 'per_symbol': per_symbol, 'params': {'ict': ict_params, 'ma': ma_params}}
    Path('reports').mkdir(exist_ok=True)
    with open('reports/combined_backtest_mt5.json','w') as f:
        json.dump(out, f, indent=2, default=str)
    print('Saved reports/combined_backtest_mt5.json')


if __name__ == '__main__':
    main()
