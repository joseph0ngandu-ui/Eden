#!/usr/bin/env python3
"""
ICT Strategy Backtest Runner

Tests the ICT strategies (Silver Bullet, Unicorn, Venom) on historical data
for appropriate forex pairs and synthetic indices.
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading.ict_strategies import ICTStrategyBot, Bar
from trading.models import Trade

def fetch_mt5_data(symbol: str, start_date: datetime, end_date: datetime, timeframe=mt5.TIMEFRAME_M1) -> Optional[pd.DataFrame]:
    """Fetch historical data from MT5."""
    if not mt5.initialize():
        print(f"MT5 initialize() failed, error code = {mt5.last_error()}")
        return None
    
    # Select symbol
    if not mt5.symbol_select(symbol, True):
        print(f"Failed to select {symbol}")
        return None
    
    # Fetch data
    rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
    mt5.shutdown()
    
    if rates is None or len(rates) == 0:
        print(f"No data retrieved for {symbol}")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
   
    return df

def run_ict_backtest(symbol: str, strategy_name: str, start_date: datetime, end_date: datetime):
    """Run backtest for a specific ICT strategy."""
    print(f"\n{'='*80}")
    print(f"Backtesting {strategy_name} on {symbol}")
    print(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"{'='*80}\n")
    
    # Fetch data
    df = fetch_mt5_data(symbol, start_date, end_date)
    if df is None:
        print(f"Failed to fetch data for {symbol}")
        return None
    
    print(f"Data points fetched: {len(df)}")
    
    # Initialize strategy
    bot = ICTStrategyBot()
    
    # Track trades
    trades = []
    open_trade: Optional[Trade] = None
    
    # Run through each bar
    for idx, row in df.iterrows():
        bar = Bar(
            time=row['time'],
            open=row['open'],
            high=row['high'],
            low=row['low'],
            close=row['close'],
            volume=row['tick_volume']
        )
        
        # Add to bot's bar history
        bot.bars.append(bar)
        bot.engine.update_structure(bot.bars)
        
        # Check for entry signal (if no open trade)
        if open_trade is None:
            if strategy_name == "Silver Bullet":
                signal = bot.run_2023_silver_bullet(bar, symbol)
            elif strategy_name == "Unicorn":
                signal = bot.run_2024_unicorn(bar, symbol)
            elif strategy_name == "Venom":
                signal = bot.run_2025_venom(bar, symbol)
            else:
                signal = None
            
            if signal:
                open_trade = signal
                open_trade.entry_time = bar.time
                print(f"[{bar.time}] ENTRY {signal.direction} @ {signal.entry_price:.5f} | TP: {signal.tp:.5f} | SL: {signal.sl:.5f}")
        else:
            # Check for exit
            hit_tp = False
            hit_sl = False
            
            if open_trade.direction == "LONG":
                hit_tp = bar.high >= open_trade.tp
                hit_sl = bar.low <= open_trade.sl
            else:  # SHORT
                hit_tp = bar.low <= open_trade.tp
                hit_sl = bar.high >= open_trade.sl
            
            if hit_tp:
                exit_price = open_trade.tp
                pnl = (exit_price - open_trade.entry_price) if open_trade.direction == "LONG" else (open_trade.entry_price - exit_price)
                trade_result = {
                    'entry_time': open_trade.entry_time,
                    'exit_time': bar.time,
                    'direction': open_trade.direction,
                    'entry_price': open_trade.entry_price,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'result': 'TP'
                }
                trades.append(trade_result)
                print(f"[{bar.time}] EXIT TP @ {exit_price:.5f} | PnL: {pnl:.5f}")
                open_trade = None
            elif hit_sl:
                exit_price = open_trade.sl
                pnl = (exit_price - open_trade.entry_price) if open_trade.direction == "LONG" else (open_trade.entry_price - exit_price)
                trade_result = {
                    'entry_time': open_trade.entry_time,
                    'exit_time': bar.time,
                    'direction': open_trade.direction,
                    'entry_price': open_trade.entry_price,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'result': 'SL'
                }
                trades.append(trade_result)
                print(f"[{bar.time}] EXIT SL @ {exit_price:.5f} | PnL: {pnl:.5f}")
                open_trade = None
    
    # Calculate statistics
    if len(trades) == 0:
        print("\nNo trades executed!")
        return None
    
    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] < 0]
    
    total_pnl = sum(t['pnl'] for t in trades)
    win_rate = len(wins) / len(trades) * 100 if trades else 0
    avg_win = sum(t['pnl'] for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t['pnl'] for t in losses) / len(losses) if losses else 0
    profit_factor = abs(sum(t['pnl'] for t in wins) / sum(t['pnl'] for t in losses)) if losses and sum(t['pnl'] for t in losses) != 0 else 0
    
    print(f"\n{'='*80}")
    print(f"BACKTEST RESULTS - {strategy_name} on {symbol}")
    print(f"{'='*80}")
    print(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Total Trades: {len(trades)}")
    print(f"Winning Trades: {len(wins)} ({win_rate:.2f}%)")
    print(f"Losing Trades: {len(losses)}")
    print(f"Total P&L: {total_pnl:.5f}")
    print(f"Average Win: {avg_win:.5f}")
    print(f"Average Loss: {avg_loss:.5f}")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"{'='*80}\n")
    
    return {
        'symbol': symbol,
        'strategy': strategy_name,
        'total_trades': len(trades),
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'trades': trades
    }

if __name__ == "__main__":
    # Define test parameters
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)  # Last 30 days
    
    # ICT strategies are designed for Forex pairs
    symbols = [
        "EURUSD",
        "GBPUSD",
        "USDJPY",
    ]
    
    strategies = [
        "Silver Bullet",
        "Unicorn",
        "Venom"
    ]
    
    all_results = []
    
    for symbol in symbols:
        for strategy in strategies:
            result = run_ict_backtest(symbol, strategy, start_date, end_date)
            if result:
                all_results.append(result)
    
    # Summary
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}")
    for result in all_results:
        print(f"{result['strategy']:15} | {result['symbol']:10} | Trades: {result['total_trades']:4} | Win Rate: {result['win_rate']:5.1f}% | P&L: {result['total_pnl']:8.5f} | PF: {result['profit_factor']:4.2f}")
    print(f"{'='*80}\n")
