#!/usr/bin/env python3
"""
VIX Strategy v1.3 Backtest
Tests the profitable Volatility Burst v1.3 strategy
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading.volatility_burst import VolatilityBurst, VBConfig

def fetch_vix_data(symbol: str = "Volatility 100 Index", months: int = 3):
    """Fetch VIX data."""
    if not mt5.initialize():
        print("MT5 init failed")
        return None
        
    end_date = datetime.now()
    start_date = end_date - timedelta(days=months * 30)
    
    if not mt5.symbol_select(symbol, True):
        print(f"Symbol {symbol} not found")
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

def run_vix_backtest(df: pd.DataFrame):
    """Run backtest."""
    config = VBConfig()
    strategy = VolatilityBurst(config)
    
    equity = 100000.0
    initial_equity = 100000.0
    closed_trades = []
    
    print(f"\nBacktesting VIX v1.3 Strategy")
    print(f"Period: {df.index[0]} to {df.index[-1]}")
    print(f"Bars: {len(df)}\n")
    
    for idx in range(100, len(df)):
        current_time = df.index[idx]
        window = df.iloc[max(0, idx-100):idx+1].copy()
        
        # Check entry
        entry_signal = strategy.evaluate_entry("Volatility 100 Index", window)
        
        if entry_signal:
            # Simulate trade
            direction = entry_signal['direction']
            entry_price = entry_signal['entry_price']
            tp = entry_signal['tp']
            sl = entry_signal['sl']
            confidence = entry_signal.get('confidence', 0.5)
            
            # Find exit (simple simulation - check next 50 bars)
            for future_idx in range(idx+1, min(idx+50, len(df))):
                future_bar = df.iloc[future_idx]
                
                hit_tp = (direction == 'LONG' and future_bar['high'] >= tp) or \
                         (direction == 'SHORT' and future_bar['low'] <= tp)
                hit_sl = (direction == 'LONG' and future_bar['low'] <= sl) or \
                         (direction == 'SHORT' and future_bar['high'] >= sl)
                
                if hit_tp or hit_sl:
                    exit_price = tp if hit_tp else sl
                    pnl_price = (exit_price - entry_price) if direction == 'LONG' else (entry_price - exit_price)
                    
                    # Calculate PnL (0.15% risk)
                    risk_amount = equity * 0.0015
                    sl_distance = abs(entry_price - sl)
                    
                    if sl_distance > 0:
                        position_size = risk_amount / sl_distance
                        pnl_dollars = position_size * pnl_price
                    else:
                        pnl_dollars = 0
                    
                    equity += pnl_dollars
                    
                    closed_trades.append({
                        'entry_time': current_time,
                        'exit_time': df.index[future_idx],
                        'direction': direction,
                        'pnl_dollars': pnl_dollars,
                        'equity': equity,
                        'result': 'TP' if hit_tp else 'SL',
                        'confidence': confidence
                    })
                    break
    
    # Analyze
    if not closed_trades:
        print("No trades")
        return
        
    df_trades = pd.DataFrame(closed_trades)
    total_return = ((equity - initial_equity) / initial_equity) * 100
    
    # Drawdown
    peak = initial_equity
    max_dd = 0
    for t in closed_trades:
        e = t['equity']
        if e > peak: peak = e
        dd = (peak - e) / peak * 100
        max_dd = max(max_dd, dd)
    
    # Monthly
    df_trades['month'] = pd.to_datetime(df_trades['exit_time']).dt.to_period('M')
    monthly = df_trades.groupby('month')['pnl_dollars'].sum()
    monthly_pct = (monthly / initial_equity) * 100
    
    # Win stats
    wins = df_trades[df_trades['result'] == 'TP']
    win_rate = (len(wins) / len(df_trades)) * 100 if len(df_trades) > 0 else 0
    
    print(f"{'='*60}")
    print(f"VIX V1.3 STRATEGY RESULTS")
    print(f"{'='*60}")
    print(f"Total Trades: {len(df_trades)}")
    print(f"Final Equity: ${equity:,.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Max Drawdown: {max_dd:.2f}%")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Avg Confidence: {df_trades['confidence'].mean():.2f}")
    
    print(f"\nMONTHLY BREAKDOWN:")
    for month, val in monthly_pct.items():
        print(f"  {month}: {val:>6.2f}%")
    
    print(f"\nAVG MONTHLY: {monthly_pct.mean():.2f}%")

if __name__ == "__main__":
    print("Fetching VIX data...")
    df = fetch_vix_data("Volatility 100 Index", months=3)
    
    if df is not None:
        run_vix_backtest(df)
    else:
        print("Data fetch failed")
