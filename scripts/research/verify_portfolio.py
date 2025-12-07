#!/usr/bin/env python3
"""
FULL PORTFOLIO VALIDATION (FundedNext Compliance)
Calculates: Equity Curve, Max Drawdown, Consecutive Losses, Net P&L
For all active strategies BEFORE deployment.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta, time as dt_time
from dataclasses import dataclass
from typing import Optional, List, Dict
import pandas as pd
import numpy as np
import logging

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TradeResult:
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    r_multiple: float
    time: pd.Timestamp
    strategy: str

def calculate_atr(df: pd.DataFrame, period=14):
    h, l, c = df['high'], df['low'], df['close'].shift(1)
    tr = pd.concat([h-l, (h-c).abs(), (l-c).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def simulate_london_breakout(df: pd.DataFrame, symbol: str) -> List[TradeResult]:
    """Simulate London Breakout trades and return results."""
    results = []
    
    # TEST ALL PAIRS - no filter
    if len(df) < 300: return results
    
    atr_series = calculate_atr(df)
    ema_50 = df['close'].ewm(span=50).mean()
    
    for i in range(200, len(df) - 100):
        current = df.iloc[i]
        timestamp = current.name
        hour = pd.to_datetime(timestamp).hour
        
        # Time Filter: 08:00 - 11:00 Server
        if not (8 <= hour <= 11): continue
        
        # Asian Range
        today_date = timestamp.date()
        range_start = pd.Timestamp.combine(today_date, dt_time(0, 0))
        range_end = pd.Timestamp.combine(today_date, dt_time(8, 0))
        
        mask = (df.index >= range_start) & (df.index < range_end)
        range_data = df.loc[mask]
        if len(range_data) < 10: continue
        
        range_high = range_data['high'].max()
        range_low = range_data['low'].min()
        range_width = range_high - range_low
        
        atr = atr_series.iloc[i]
        if pd.isna(atr) or atr == 0: continue
        if range_width > atr * 3.0: continue
        
        trend_up = current['close'] > ema_50.iloc[i]
        trend_down = current['close'] < ema_50.iloc[i]
        price = current['close']
        
        signal = None
        if price > range_high and trend_up:
            sl = range_low - atr * 0.3
            risk = price - sl
            tp = price + risk * 1.5
            signal = ("LONG", price, sl, tp)
        elif price < range_low and trend_down:
            sl = range_high + atr * 0.3
            risk = sl - price
            tp = price - risk * 1.5
            signal = ("SHORT", price, sl, tp)
        
        if signal:
            direction, entry, sl, tp = signal
            future = df.iloc[i+1:i+100]
            
            # Simulate outcome
            exit_price = None
            for idx, row in future.iterrows():
                if direction == "LONG":
                    if row['low'] <= sl:
                        exit_price = sl
                        break
                    if row['high'] >= tp:
                        exit_price = tp
                        break
                else:
                    if row['high'] >= sl:
                        exit_price = sl
                        break
                    if row['low'] <= tp:
                        exit_price = tp
                        break
            
            if exit_price is None:
                exit_price = future.iloc[-1]['close']
            
            if direction == "LONG":
                pnl = exit_price - entry
            else:
                pnl = entry - exit_price
            
            risk_amt = abs(entry - sl)
            r_mult = pnl / risk_amt if risk_amt > 0 else 0
            
            results.append(TradeResult(
                symbol=symbol,
                direction=direction,
                entry_price=entry,
                exit_price=exit_price,
                r_multiple=r_mult,
                time=timestamp,
                strategy="London_Breakout"
            ))
    
    return results

def calculate_metrics(trades: List[TradeResult]) -> Dict:
    """Calculate comprehensive trading metrics."""
    if not trades:
        return {"error": "No trades"}
    
    r_values = [t.r_multiple for t in trades]
    
    # Equity Curve (Cumulative R)
    equity = np.cumsum(r_values)
    
    # Max Drawdown
    peak = np.maximum.accumulate(equity)
    drawdown = peak - equity
    max_dd = np.max(drawdown)
    
    # Consecutive Losses
    max_consec_loss = 0
    current_streak = 0
    for r in r_values:
        if r < 0:
            current_streak += 1
            max_consec_loss = max(max_consec_loss, current_streak)
        else:
            current_streak = 0
    
    # Win Rate
    wins = len([r for r in r_values if r > 0])
    wr = wins / len(r_values) * 100
    
    return {
        "total_trades": len(trades),
        "total_r": sum(r_values),
        "win_rate": wr,
        "max_drawdown_r": max_dd,
        "max_consecutive_losses": max_consec_loss,
        "avg_win": np.mean([r for r in r_values if r > 0]) if wins > 0 else 0,
        "avg_loss": np.mean([r for r in r_values if r < 0]) if len(r_values) - wins > 0 else 0,
        "profit_factor": abs(sum([r for r in r_values if r > 0]) / sum([r for r in r_values if r < 0])) if sum([r for r in r_values if r < 0]) != 0 else float('inf')
    }

def run_validation():
    try:
        import MetaTrader5 as mt5
    except ImportError:
        print("MetaTrader5 not available")
        return
    
    if not mt5.initialize(path=r"C:\Program Files\MetaTrader 5 Terminal\terminal64.exe"):
        print("MT5 init failed")
        return
    
    print("="*70)
    print("  LONDON BREAKOUT - COMPREHENSIVE RESEARCH")
    print("  (ALL London Session Pairs)")
    print("="*70)
    
    # ALL potential London session pairs
    symbols = [
        'EURUSDm', 'GBPUSDm', 'GBPJPYm', 'EURGBPm', 'EURJPYm',
        'GBPCHFm', 'EURCHFm', 'GBPAUDm', 'GBPNZDm', 'GBPCADm',
        'EURAUDm', 'EURNZDm', 'EURCADm'
    ]
    tf = mt5.TIMEFRAME_M15
    end = datetime.now()
    start = end - timedelta(days=90)
    
    all_trades = []
    
    for symbol in symbols:
        rates = mt5.copy_rates_range(symbol, tf, start, end)
        if rates is None or len(rates) < 500:
            print(f"{symbol}: Insufficient data")
            continue
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        trades = simulate_london_breakout(df, symbol)
        all_trades.extend(trades)
        
        if trades:
            metrics = calculate_metrics(trades)
            print(f"\n--- {symbol} ---")
            print(f"  Trades: {metrics['total_trades']}")
            print(f"  Total R: {metrics['total_r']:.1f}R")
            print(f"  Win Rate: {metrics['win_rate']:.1f}%")
            print(f"  Max DD: {metrics['max_drawdown_r']:.1f}R")
            print(f"  Max Consec Losses: {metrics['max_consecutive_losses']}")
    
    # Combined Portfolio Metrics
    if all_trades:
        print("\n" + "="*70)
        print("  COMBINED PORTFOLIO")
        print("="*70)
        combined = calculate_metrics(all_trades)
        print(f"  Total Trades: {combined['total_trades']}")
        print(f"  Total R: {combined['total_r']:.1f}R")
        print(f"  Win Rate: {combined['win_rate']:.1f}%")
        print(f"  Max Drawdown: {combined['max_drawdown_r']:.1f}R")
        print(f"  Max Consec Losses: {combined['max_consecutive_losses']}")
        print(f"  Profit Factor: {combined['profit_factor']:.2f}")
        
        # DEPLOYMENT VERDICT
        print("\n" + "="*70)
        print("  DEPLOYMENT VERDICT")
        print("="*70)
        
        passed = True
        if combined['total_r'] < 15:
            print("  ❌ FAIL: Total R < 15")
            passed = False
        else:
            print(f"  ✅ PASS: Total R = {combined['total_r']:.1f}R (> 15)")
        
        if combined['win_rate'] < 40:
            print("  ❌ FAIL: Win Rate < 40%")
            passed = False
        else:
            print(f"  ✅ PASS: Win Rate = {combined['win_rate']:.1f}% (> 40%)")
        
        if combined['max_drawdown_r'] > 8:
            print("  ❌ FAIL: Max DD > 8R (Risk of daily limit breach)")
            passed = False
        else:
            print(f"  ✅ PASS: Max DD = {combined['max_drawdown_r']:.1f}R (< 8R)")
        
        if combined['max_consecutive_losses'] > 8:
            print("  ⚠️ WARNING: Max Consec Losses > 8 (Psychological risk)")
        else:
            print(f"  ✅ PASS: Max Consec Losses = {combined['max_consecutive_losses']} (< 8)")
        
        print("\n  FINAL: " + ("✅ APPROVED FOR DEPLOYMENT" if passed else "❌ REJECTED"))
    
    mt5.shutdown()

if __name__ == "__main__":
    run_validation()
