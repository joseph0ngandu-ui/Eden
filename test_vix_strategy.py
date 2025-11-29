#!/usr/bin/env python3
"""
VIX (Volatility Burst) Strategy Backtest
Tests the Volatility Burst v1.3 strategy on Volatility 100 Index
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading.volatility_burst_enhanced import VolatilityBurst
from trading.models import Trade

def fetch_vix_data(symbol: str = "Volatility 100 Index", months: int = 3):
    """Fetch historical data for VIX."""
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
        print("No data")
        return None
        
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df

def run_vix_backtest(df: pd.DataFrame):
    """Run backtest on VIX strategy."""
    strategy = VolatilityBurst()
    
    equity = 100000.0
    initial_equity = 100000.0
    closed_trades = []
    open_trades = []
    
    print(f"Backtesting VIX Strategy...")
    print(f"Period: {df.index[0]} to {df.index[-1]}")
    print(f"Bars: {len(df)}")
    
    for idx in range(100, len(df)):
        current_time = df.index[idx]
        window = df.iloc[max(0, idx-100):idx+1].copy()
        
        # Manage exits first
        active_trades = []
        for trade in open_trades:
            row = df.iloc[idx]
            
            # Check TP/SL
            hit_tp = (trade['direction'] == 'LONG' and row['high'] >= trade['tp']) or \
                     (trade['direction'] == 'SHORT' and row['low'] <= trade['tp'])
            hit_sl = (trade['direction'] == 'LONG' and row['low'] <= trade['sl']) or \
                     (trade['direction'] == 'SHORT' and row['high'] >= trade['sl'])
            
            if hit_tp or hit_sl:
                exit_price = trade['tp'] if hit_tp else trade['sl']
                pnl_price = (exit_price - trade['entry']) if trade['direction'] == 'LONG' else (trade['entry'] - exit_price)
                
                # Calculate PnL (0.15% risk per trade)
                risk_amount = equity * 0.0015
                sl_distance = abs(trade['entry'] - trade['sl'])
                
                if sl_distance > 0:
                    position_size = risk_amount / sl_distance
                    pnl_dollars = position_size * pnl_price
                else:
                    pnl_dollars = 0
                
                equity += pnl_dollars
                
                closed_trades.append({
                    'entry_time': trade['entry_time'],
                    'exit_time': current_time,
                    'pnl_dollars': pnl_dollars,
                    'equity': equity,
                    'result': 'TP' if hit_tp else 'SL',
                    'confidence': trade.get('confidence', 0)
                })
            else:
                active_trades.append(trade)
        
        open_trades = active_trades
        
        # Check new entries (max 1 open)
        if len(open_trades) == 0:
            signals = strategy.generate_signals(window)
            
            if signals:
                signal = signals[-1]  # Take most recent
                if signal.confidence >= strategy.confidence_threshold:
                    open_trades.append({
                        'direction': signal.direction,
                        'entry': signal.entry_price,
                        'sl': signal.sl,
                        'tp': signal.tp,
                        'entry_time': current_time,
                        'confidence': signal.confidence
                    })
    
    # Analyze results
    if not closed_trades:
        print("\nNo trades executed")
        return
        
    df_trades = pd.DataFrame(closed_trades)
    
    # Calculate metrics
    total_return = ((equity - initial_equity) / initial_equity) * 100
    
    # Drawdown
    peak = initial_equity
    max_dd = 0
    for t in closed_trades:
        e = t['equity']
        if e > peak: peak = e
        dd = (peak - e) / peak * 100
        max_dd = max(max_dd, dd)
    
    # Monthly breakdown
    df_trades['month'] = pd.to_datetime(df_trades['exit_time']).dt.to_period('M')
    monthly = df_trades.groupby('month')['pnl_dollars'].sum()
    monthly_pct = (monthly / initial_equity) * 100
    
    # Win stats
    wins = df_trades[df_trades['result'] == 'TP']
    win_rate = (len(wins) / len(df_trades)) * 100
    
    print(f"\n{'='*60}")
    print(f"VIX STRATEGY PERFORMANCE")
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
    
    return {
        'return': total_return,
        'max_dd': max_dd,
        'win_rate': win_rate,
        'trades': len(df_trades)
    }

if __name__ == "__main__":
    print("Fetching VIX data...")
    df = fetch_vix_data("Volatility 100 Index", months=3)
    
    if df is not None:
        results = run_vix_backtest(df)
    else:
        print("Failed to fetch data")
