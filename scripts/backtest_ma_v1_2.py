#!/usr/bin/env python3
"""
Backtest MA v1.2 strategy per symbol (MT5 M5 data), similar to VB harness.
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import json

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from ma_v1_2 import MA_v1_2

class MABacktester:
    def __init__(self, config_path: str = 'config/ma_v1_2.yml'):
        self.ma = MA_v1_2(config_path)
        self.results = {}

    def fetch(self, symbol, start, end):
        if not mt5.initialize():
            return None
        try:
            rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M5, start, end)
            if rates is None:
                return None
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            return df.sort_values('time').reset_index(drop=True)
        finally:
            mt5.shutdown()

    def backtest_symbol(self, symbol, start, end, overrides=None):
        print(f"Backtesting {symbol}...", end=' ')
        df = self.fetch(symbol, start, end)
        if df is None or len(df)<50:
            print('FAILED')
            return None
        self.ma = MA_v1_2('config/ma_v1_2.yml')
        if overrides:
            self.ma.update_params(**overrides)
        df.attrs['symbol'] = symbol
        trades_data = []
        positions = []

        signals = self.ma.generate_signals(df)
        by_idx = {t.bar_index: t for t in signals}

        for idx in range(len(df)):
            d = df.iloc[:idx+1]
            actions = self.ma.manage_position(d, symbol)
            for a in actions:
                if a['action']=='close':
                    pos = positions.pop()
                    exit_price = a.get('price', d.iloc[-1]['close'])
                    profit = (exit_price - pos['entry_price']) if pos['direction']=='LONG' else (pos['entry_price'] - exit_price)
                    trades_data.append({
                        'entry_time': pos['entry_time'],
                        'exit_time': d.iloc[-1]['time'],
                        'entry_price': pos['entry_price'],
                        'exit_price': exit_price,
                        'direction': pos['direction'],
                        'profit': profit,
                        'r_value': a.get('r_value', 0.0)
                    })
            if symbol not in self.ma.open_positions and idx in by_idx:
                t = by_idx[idx]
                self.ma.on_trade_open(t)
                positions.append({'entry_time': t.entry_time, 'entry_price': t.entry_price, 'direction': t.direction})

        if positions:
            last = df.iloc[-1]
            for pos in positions:
                profit = (last['close'] - pos['entry_price']) if pos['direction']=='LONG' else (pos['entry_price'] - last['close'])
                trades_data.append({
                    'entry_time': pos['entry_time'],
                    'exit_time': last['time'],
                    'entry_price': pos['entry_price'],
                    'exit_price': last['close'],
                    'direction': pos['direction'],
                    'profit': profit,
                    'r_value': 0.0
                })

        if not trades_data:
            print('NO TRADES')
            return None
        profits = [t['profit'] for t in trades_data]
        winning = [p for p in profits if p>0]
        losing = [p for p in profits if p<0]
        stats = {
            'symbol': symbol,
            'total_trades': len(trades_data),
            'winning_trades': len(winning),
            'losing_trades': len(losing),
            'total_pnl': sum(profits),
            'win_rate': (len(winning)/len(trades_data))*100.0,
            'profit_factor': (sum(winning)/abs(sum(losing))) if losing else 0.0
        }
        print(f"{len(trades_data)} trades | PnL ${stats['total_pnl']:+.2f} | WR {stats['win_rate']:.1f}%")
        self.results[symbol] = {'stats': stats, 'trades': trades_data}
        return stats

if __name__=='__main__':
    symbols = [
        'Volatility 75 Index', 'Volatility 100 Index', 'Boom 500 Index', 'Crash 500 Index', 'Boom 1000 Index', 'Step Index', 'XAUUSD'
    ]
    bt = MABacktester('config/ma_v1_2.yml')
    start = datetime(2025,1,1); end = datetime(2025,10,31)
    for s in symbols:
        bt.backtest_symbol(s, start, end)
    Path('reports').mkdir(exist_ok=True)
    with open('reports/ma_v1_2_results.json','w') as f:
        json.dump(bt.results, f, indent=2, default=str)
