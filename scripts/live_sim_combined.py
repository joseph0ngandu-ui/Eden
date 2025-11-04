#!/usr/bin/env python3
"""
Live-sim (paper) run for Combined MA + ICT/PA using best params from combined optimization report.
- Compounds from $start_equity with RiskLadder
- Period/symbols configurable via CLI
Outputs: reports/combined_live_sim_trades.csv, reports/combined_live_sim_summary.json
"""
import sys
from pathlib import Path
from datetime import datetime
import argparse
import json
import pandas as pd
import MetaTrader5 as mt5

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from risk_ladder import RiskLadder
from strategies.signals import ICTPA
from ma_v1_2 import MA_v1_2

SYMBOLS_DEFAULT = [
    'XAUUSD',
    'Volatility 75 Index',
    'Crash 500 Index',
    'Boom 500 Index',
]


def fetch(symbol: str, start: datetime, end: datetime):
    if mt5.initialize():
        try:
            rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M5, start, end)
            if rates is not None:
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                return df.sort_values('time').reset_index(drop=True)
        finally:
            mt5.shutdown()
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
                trades.append({'exit_time': d.iloc[-1]['time'], 'r': float(a.get('r_value', 0.0) or 0.0), 'strategy': 'ICTPA'})
        if sym not in ict.open_positions and idx in by_idx:
            t = by_idx[idx]
            ict.on_trade_open(t)
            positions.append({'entry_time': t.entry_time, 'entry_price': t.entry_price, 'direction': t.direction})
    if positions:
        last = df.iloc[-1]
        for _ in positions:
            trades.append({'exit_time': last['time'], 'r': 0.0, 'strategy': 'ICTPA'})
    return trades


def trades_ma(sym: str, df: pd.DataFrame, params: dict):
    ma = MA_v1_2('config/ma_v1_2.yml')
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
    rows = []
    for t in all_trades:
        r = float(t.get('r', 0.0) or 0.0)
        risk_pct = rl.get_adjusted_risk_pct()
        pnl = eq * (risk_pct/100.0) * r
        eq += pnl
        curve.append(eq)
        rl.update_balance(eq)
        rows.append({'time': t['exit_time'], 'strategy': t['strategy'], 'r': r, 'risk_pct': risk_pct, 'pnl': pnl, 'equity': eq})
    # drawdown
    peak = -1e18
    max_dd = 0.0
    for e in curve:
        peak = max(peak, e)
        if peak>0:
            max_dd = max(max_dd, (peak - e)/peak)
    return eq, max_dd*100, curve, rows


def main():
    ap = argparse.ArgumentParser(description='Live-sim Combined MA + ICT/PA with RiskLadder')
    ap.add_argument('--start', type=str, default='2025-01-01')
    ap.add_argument('--end', type=str, default='2025-11-03')
    ap.add_argument('--symbols', type=str, default=','.join(SYMBOLS_DEFAULT))
    ap.add_argument('--start-equity', type=float, default=1000.0)
    ap.add_argument('--params-json', type=str, default='reports/combined_ictpa_ma_optimization.json')
    args = ap.parse_args()

    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end)
    symbols = [s.strip() for s in args.symbols.split(',') if s.strip()]

    with open(args.params_json, 'r') as f:
        rep = json.load(f)
    best = rep.get('best') or {}
    ictp = best.get('ict') or {}
    mapar = best.get('ma') or {}

    data = {}
    for s in symbols:
        df = fetch(s, start, end)
        if df is not None and len(df)>=100:
            data[s] = df
    if not data:
        print('No data.')
        return

    portfolio_trades = []
    per_symbol = {}
    for sym, df in data.items():
        t1 = trades_ictpa(sym, df, ictp)
        t2 = trades_ma(sym, df, mapar)
        merged = t1 + t2
        end_eq_s, max_dd_s, _, _ = compound_trades(merged, args.start_equity)
        per_symbol[sym] = {'trades': len(merged), 'end_equity': end_eq_s, 'max_dd_pct': max_dd_s}
        portfolio_trades.extend(merged)

    end_eq, max_dd, curve, rows = compound_trades(portfolio_trades, args.start_equity)
    out_dir = Path('reports'); out_dir.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / 'combined_live_sim_trades.csv', index=False)
    summary = {
        'timestamp': datetime.now().isoformat(),
        'period': {'start': start.isoformat(), 'end': end.isoformat()},
        'symbols': symbols,
        'start_equity': args.start_equity,
        'end_equity': end_eq,
        'return_percent': (end_eq/args.start_equity - 1) * 100.0,
        'max_drawdown_percent': max_dd,
        'per_symbol': per_symbol,
        'params_used': {'ict': ictp, 'ma': mapar}
    }
    with open(out_dir / 'combined_live_sim_summary.json','w') as f:
        json.dump(summary, f, indent=2)
    print(f"Combined live-sim: ${args.start_equity:.2f} -> ${end_eq:.2f} | Return {summary['return_percent']:.1f}% | MaxDD {max_dd:.1f}%")
    print("Reports written: combined_live_sim_trades.csv, combined_live_sim_summary.json")


if __name__=='__main__':
    main()
