#!/usr/bin/env python3
"""
Backtest ICT+Price Action hybrid strategy across selected symbols and 2023-01-01 to 2025-11-03.
Outputs per-symbol stats and portfolio summary JSON.
"""
import sys
from pathlib import Path
from datetime import datetime
import json
import argparse
import pandas as pd
import MetaTrader5 as mt5

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from strategies.signals import ICTPA

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
    # Fallback to CSV (M1) -> resample to M5 if available
    data_dir = Path('data/mt5_feeds')
    for name in [f"{symbol}_M5.csv", f"{symbol}_M1.csv"]:
        csv = data_dir / name
        if csv.exists():
            d = pd.read_csv(csv)
            # normalize timestamp column name
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


def backtest_symbol(strategy: ICTPA, symbol: str, start: datetime, end: datetime):
    print(f"Backtesting {symbol}...", end=' ')
    df = fetch(symbol, start, end)
    if df is None or len(df) < 100:
        print('FAILED')
        return None, []
    df.attrs['symbol'] = symbol
    trades = []
    positions = []

    sigs = strategy.generate_signals(df)
    by_idx = {t.bar_index: t for t in sigs}

    for idx in range(len(df)):
        d = df.iloc[:idx+1]
        acts = strategy.manage_position(d, symbol)
        for a in acts:
            if a['action']=='close':
                pos = positions.pop()
                exit_price = a.get('price', d.iloc[-1]['close'])
                profit = (exit_price - pos['entry_price']) if pos['direction']=='LONG' else (pos['entry_price'] - exit_price)
                trades.append({
                    'entry_time': pos['entry_time'],
                    'exit_time': d.iloc[-1]['time'],
                    'entry_price': pos['entry_price'],
                    'exit_price': exit_price,
                    'direction': pos['direction'],
                    'profit': profit,
                    'r_value': a.get('r_value', 0.0),
                    'confidence': pos.get('confidence', None),
                    'entry_atr': pos.get('atr', None)
                })
        if symbol not in strategy.open_positions and idx in by_idx:
            t = by_idx[idx]
            strategy.on_trade_open(t)
            positions.append({'entry_time': t.entry_time, 'entry_price': t.entry_price,
                              'direction': t.direction, 'confidence': t.confidence, 'atr': t.atr})

    if positions:
        last = df.iloc[-1]
        for pos in positions:
            profit = (last['close'] - pos['entry_price']) if pos['direction']=='LONG' else (pos['entry_price'] - last['close'])
            trades.append({
                'entry_time': pos['entry_time'],
                'exit_time': last['time'],
                'entry_price': pos['entry_price'],
                'exit_price': last['close'],
                'direction': pos['direction'],
                'profit': profit,
                'r_value': 0.0,
                'confidence': pos.get('confidence', None),
                'entry_atr': pos.get('atr', None)
            })

    if not trades:
        print('NO TRADES')
        return None, []

    prof = [t['profit'] for t in trades]
    wins = [p for p in prof if p>0]
    losses = [p for p in prof if p<0]
    stats = {
        'symbol': symbol,
        'total_trades': len(trades),
        'winning_trades': len(wins),
        'losing_trades': len(losses),
        'total_pnl': sum(prof),
        'win_rate': (len(wins)/len(trades))*100.0 if trades else 0.0,
        'profit_factor': (sum(wins)/abs(sum(losses))) if losses else 0.0
    }
    print(f"{len(trades)} trades | PnL ${stats['total_pnl']:+.2f} | WR {stats['win_rate']:.1f}% | PF {stats['profit_factor']:.2f}")
    return stats, trades


def main():
    ap = argparse.ArgumentParser(description='Backtest ICT+PA hybrid')
    ap.add_argument('--start', type=str, default='2025-01-01', help='YYYY-MM-DD')
    ap.add_argument('--end', type=str, default='2025-11-03', help='YYYY-MM-DD')
    ap.add_argument('--symbols', type=str, default=','.join(SYMBOLS_DEFAULT), help='Comma list of symbols')
    args = ap.parse_args()

    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end)
    symbols = [s.strip() for s in args.symbols.split(',') if s.strip()]

    strat = ICTPA()
    per_symbol = {}
    all_trades = []
    wins = losses = 0
    tot_pnl = 0.0
    tot_trades = 0

    for s in symbols:
        stats, trades = backtest_symbol(strat, s, start, end)
        if not stats:
            continue
        per_symbol[s] = stats
        tot_pnl += stats['total_pnl']
        tot_trades += stats['total_trades']
        wins += stats['winning_trades']
        losses += stats['losing_trades']
        for t in trades:
            t['symbol']=s
        all_trades.extend(trades)

    out = {
        'timestamp': datetime.now().isoformat(),
        'period': {'start': start.isoformat(), 'end': end.isoformat()},
        'symbols': symbols,
        'portfolio': {
            'total_trades': tot_trades,
            'wins': wins,
            'losses': losses,
            'win_rate': (wins/tot_trades*100.0) if tot_trades else 0.0,
            'total_pnl': tot_pnl,
        },
        'per_symbol': per_symbol,
    }
    Path('reports').mkdir(exist_ok=True)
    with open('reports/ict_pa_backtest.json','w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"Saved reports/ict_pa_backtest.json")


if __name__=='__main__':
    main()
