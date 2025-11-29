#!/usr/bin/env python3
"""
VIX Strategy Optimizer - Test Multiple Configurations
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading.volatility_burst import VolatilityBurst, VBConfig

def fetch_vix_data(symbol: str = "Volatility 100 Index", months: int = 3):
    """Fetch VIX data."""
    if not mt5.initialize():
        return None
        
    end_date = datetime.now()
    start_date = end_date - timedelta(days=months * 30)
    
    if not mt5.symbol_select(symbol, True):
        mt5.shutdown()
        return None
        
    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M5, start_date, end_date)
    mt5.shutdown()
    
    if rates is None or len(rates) == 0:
        return None
        
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df

def run_backtest_with_config(df: pd.DataFrame, config: VBConfig, label: str):
    """Run backtest with specific config."""
    strategy = VolatilityBurst(config)
    
    equity = 100000.0
    initial_equity = 100000.0
    closed_trades = []
    
    for idx in range(100, len(df)):
        current_time = df.index[idx]
        window = df.iloc[max(0, idx-100):idx+1].copy()
        
        entry_signal = strategy.evaluate_entry("Volatility 100 Index", window)
        
        if entry_signal:
            direction = entry_signal['direction']
            entry_price = entry_signal['entry_price']
            tp = entry_signal['tp']
            sl = entry_signal['sl']
            confidence = entry_signal.get('confidence', 0.5)
            
            # Simulate trade exit
            for future_idx in range(idx+1, min(idx+50, len(df))):
                future_bar = df.iloc[future_idx]
                
                hit_tp = (direction == 'LONG' and future_bar['high'] >= tp) or \
                         (direction == 'SHORT' and future_bar['low'] <= tp)
                hit_sl = (direction == 'LONG' and future_bar['low'] <= sl) or \
                         (direction == 'SHORT' and future_bar['high'] >= sl)
                
                if hit_tp or hit_sl:
                    exit_price = tp if hit_tp else sl
                    pnl_price = (exit_price - entry_price) if direction == 'LONG' else (entry_price - exit_price)
                    
                    risk_amount = equity * 0.0015
                    sl_distance = abs(entry_price - sl)
                    
                    if sl_distance > 0:
                        position_size = risk_amount / sl_distance
                        pnl_dollars = position_size * pnl_price
                    else:
                        pnl_dollars = 0
                    
                    equity += pnl_dollars
                    
                    closed_trades.append({
                        'exit_time': df.index[future_idx],
                        'pnl_dollars': pnl_dollars,
                        'equity': equity,
                        'result': 'TP' if hit_tp else 'SL',
                        'confidence': confidence
                    })
                    break
    
    # Analyze
    if not closed_trades:
        print(f"\n{label}: NO TRADES")
        return None
        
    df_trades = pd.DataFrame(closed_trades)
    total_return = ((equity - initial_equity) / initial_equity) * 100
    
    peak = initial_equity
    max_dd = 0
    for t in closed_trades:
        e = t['equity']
        if e > peak: peak = e
        dd = (peak - e) / peak * 100
        max_dd = max(max_dd, dd)
    
    df_trades['month'] = pd.to_datetime(df_trades['exit_time']).dt.to_period('M')
    monthly = df_trades.groupby('month')['pnl_dollars'].sum()
    monthly_pct = (monthly / initial_equity) * 100
    
    wins = df_trades[df_trades['result'] == 'TP']
    win_rate = (len(wins) / len(df_trades)) * 100
    
    print(f"\n{'='*60}")
    print(f"{label}")
    print(f"{'='*60}")
    print(f"Trades: {len(df_trades)}")
    print(f"Return: {total_return:.2f}%")
    print(f"MaxDD: {max_dd:.2f}%")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Avg Monthly: {monthly_pct.mean():.2f}%")
    
    return {
        'label': label,
        'trades': len(df_trades),
        'return': total_return,
        'max_dd': max_dd,
        'win_rate': win_rate,
        'monthly_avg': monthly_pct.mean()
    }

if __name__ == "__main__":
    print("Fetching VIX data...")
    df = fetch_vix_data("Volatility 100 Index", months=3)
    
    if df is None:
        print("Data fetch failed")
        exit(1)
    
    print(f"Data: {len(df)} bars")
    
    # Test 3 configurations
    configs = [
        # Conservative: Small tweaks
        (VBConfig(
            squeeze_atr_threshold=0.9,
            breakout_atr_multiplier=1.3,
            min_confidence=0.55,
            squeeze_bars=10,
            tp_atr_multiplier=2.0
        ), "CONSERVATIVE"),
        
        # Balanced: Medium tweaks
        (VBConfig(
            squeeze_atr_threshold=1.0,
            breakout_atr_multiplier=1.2,
            min_confidence=0.5,
            squeeze_bars=8,
            tp_atr_multiplier=2.5
        ), "BALANCED"),
        
        # Aggressive: Major tweaks
        (VBConfig(
            squeeze_atr_threshold=1.1,
            breakout_atr_multiplier=1.1,
            min_confidence=0.45,
            squeeze_bars=6,
            tp_atr_multiplier=3.0
        ), "AGGRESSIVE"),
    ]
    
    results = []
    for config, label in configs:
        result = run_backtest_with_config(df, config, label)
        if result:
            results.append(result)
    
    if results:
        print(f"\n\n{'='*60}")
        print("COMPARISON")
        print(f"{'='*60}")
        print(f"{'Config':<15} {'Trades':<8} {'Return':<10} {'MaxDD':<10} {'Win%':<8} {'Monthly':<8}")
        print("-" * 60)
        for r in results:
            print(f"{r['label']:<15} {r['trades']:<8} {r['return']:>6.2f}%   {r['max_dd']:>6.2f}%   {r['win_rate']:>5.1f}%  {r['monthly_avg']:>6.2f}%")
        
        # Find best
        best = max(results, key=lambda x: x['return'])
        print(f"\n🏆 WINNER: {best['label']} ({best['return']:.2f}% return, {best['trades']} trades)")
