#!/usr/bin/env python3
"""
MULTI-FRONTIER STRATEGY RESEARCH
Tests 4 strategy concepts with comprehensive validation.
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

logging.basicConfig(level=logging.WARNING)

@dataclass
class TradeResult:
    r_multiple: float
    time: pd.Timestamp
    strategy: str

def calculate_atr(df: pd.DataFrame, period=14):
    h, l, c = df['high'], df['low'], df['close'].shift(1)
    tr = pd.concat([h-l, (h-c).abs(), (l-c).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def calculate_metrics(trades: List[TradeResult]) -> Dict:
    if not trades:
        return {"total_trades": 0, "total_r": 0, "win_rate": 0, "max_dd": 0, "max_consec": 0}
    
    r_values = [t.r_multiple for t in trades]
    equity = np.cumsum(r_values)
    peak = np.maximum.accumulate(equity)
    drawdown = peak - equity
    max_dd = np.max(drawdown)
    
    max_consec = 0
    streak = 0
    for r in r_values:
        if r < 0:
            streak += 1
            max_consec = max(max_consec, streak)
        else:
            streak = 0
    
    wins = len([r for r in r_values if r > 0])
    
    return {
        "total_trades": len(trades),
        "total_r": sum(r_values),
        "win_rate": wins / len(r_values) * 100 if r_values else 0,
        "max_dd": max_dd,
        "max_consec": max_consec
    }

# ============================================================
# FRONTIER 1: NY CLOSE REVERSION
# ============================================================
def test_ny_close_reversion(df: pd.DataFrame, symbol: str) -> List[TradeResult]:
    """Fade extended prices at NY close (20:00-22:00)."""
    results = []
    if len(df) < 200: return results
    
    atr_s = calculate_atr(df)
    
    for i in range(100, len(df) - 50):
        bar = df.iloc[i]
        hour = pd.to_datetime(bar.name).hour
        
        # Only 20:00-21:00 Server Time
        if hour != 20: continue
        
        atr = atr_s.iloc[i]
        if pd.isna(atr) or atr == 0: continue
        
        # Calculate session VWAP (approx using SMA20)
        vwap = df['close'].iloc[i-20:i].mean()
        price = bar['close']
        
        # Extension check: >1.5 ATR from VWAP
        ext_long = price > vwap + 1.5 * atr
        ext_short = price < vwap - 1.5 * atr
        
        signal = None
        if ext_long:  # Fade SHORT
            sl = price + 0.5 * atr
            tp = vwap
            signal = ("SHORT", price, sl, tp)
        elif ext_short:  # Fade LONG
            sl = price - 0.5 * atr
            tp = vwap
            signal = ("LONG", price, sl, tp)
        
        if signal:
            direction, entry, sl, tp = signal
            future = df.iloc[i+1:i+50]
            exit_price = None
            
            for _, row in future.iterrows():
                if direction == "LONG":
                    if row['low'] <= sl: exit_price = sl; break
                    if row['high'] >= tp: exit_price = tp; break
                else:
                    if row['high'] >= sl: exit_price = sl; break
                    if row['low'] <= tp: exit_price = tp; break
            
            if exit_price is None:
                exit_price = future.iloc[-1]['close'] if len(future) > 0 else entry
            
            pnl = (exit_price - entry) if direction == "LONG" else (entry - exit_price)
            risk = abs(entry - sl)
            r_mult = pnl / risk if risk > 0 else 0
            
            results.append(TradeResult(r_mult, bar.name, "NY_Close_Reversion"))
    
    return results

# ============================================================
# FRONTIER 2: SESSION OVERLAP SCALPING
# ============================================================
def test_session_overlap(df: pd.DataFrame, symbol: str) -> List[TradeResult]:
    """Momentum scalps during London/NY overlap (13:00-17:00)."""
    results = []
    if len(df) < 200: return results
    
    atr_s = calculate_atr(df)
    ema_10 = df['close'].ewm(span=10).mean()
    ema_30 = df['close'].ewm(span=30).mean()
    
    for i in range(100, len(df) - 30):
        bar = df.iloc[i]
        hour = pd.to_datetime(bar.name).hour
        
        # Only 13:00-17:00 Server
        if not (13 <= hour <= 17): continue
        
        atr = atr_s.iloc[i]
        if pd.isna(atr) or atr == 0: continue
        
        # EMA Cross + Momentum
        ema_bull = ema_10.iloc[i] > ema_30.iloc[i] and ema_10.iloc[i-1] <= ema_30.iloc[i-1]
        ema_bear = ema_10.iloc[i] < ema_30.iloc[i] and ema_10.iloc[i-1] >= ema_30.iloc[i-1]
        
        signal = None
        price = bar['close']
        
        if ema_bull:
            sl = price - 0.5 * atr
            tp = price + 0.5 * atr  # 1:1 RR
            signal = ("LONG", price, sl, tp)
        elif ema_bear:
            sl = price + 0.5 * atr
            tp = price - 0.5 * atr
            signal = ("SHORT", price, sl, tp)
        
        if signal:
            direction, entry, sl, tp = signal
            future = df.iloc[i+1:i+30]
            exit_price = None
            
            for _, row in future.iterrows():
                if direction == "LONG":
                    if row['low'] <= sl: exit_price = sl; break
                    if row['high'] >= tp: exit_price = tp; break
                else:
                    if row['high'] >= sl: exit_price = sl; break
                    if row['low'] <= tp: exit_price = tp; break
            
            if exit_price is None:
                exit_price = future.iloc[-1]['close'] if len(future) > 0 else entry
            
            pnl = (exit_price - entry) if direction == "LONG" else (entry - exit_price)
            risk = abs(entry - sl)
            r_mult = pnl / risk if risk > 0 else 0
            
            results.append(TradeResult(r_mult, bar.name, "Session_Overlap"))
    
    return results

# ============================================================
# FRONTIER 3: MOMENTUM CONTINUATION (D1 Based)
# ============================================================
def test_momentum_continuation(df_m15: pd.DataFrame, df_d1: pd.DataFrame, symbol: str) -> List[TradeResult]:
    """Enter pullbacks after strong D1 candles."""
    results = []
    if len(df_d1) < 30 or len(df_m15) < 200: return results
    
    # Calculate ADR
    daily_ranges = df_d1['high'] - df_d1['low']
    adr = daily_ranges.rolling(14).mean()
    
    for i in range(20, len(df_d1) - 2):
        day = df_d1.iloc[i]
        day_range = day['high'] - day['low']
        
        if pd.isna(adr.iloc[i]) or adr.iloc[i] == 0: continue
        
        # Strong day: Range > 1.3x ADR
        if day_range < 1.3 * adr.iloc[i]: continue
        
        # Direction
        bullish = day['close'] > day['open']
        
        # Next day entry (simplified: at open of next day)
        next_day = df_d1.iloc[i+1]
        entry = next_day['open']
        
        if bullish:
            sl = day['low']
            risk = entry - sl
            tp = entry + risk * 1.5
            direction = "LONG"
        else:
            sl = day['high']
            risk = sl - entry
            tp = entry - risk * 1.5
            direction = "SHORT"
        
        # Check outcome on next day
        if direction == "LONG":
            if next_day['low'] <= sl:
                r_mult = -1.0
            elif next_day['high'] >= tp:
                r_mult = 1.5
            else:
                pnl = next_day['close'] - entry
                r_mult = pnl / risk if risk > 0 else 0
        else:
            if next_day['high'] >= sl:
                r_mult = -1.0
            elif next_day['low'] <= tp:
                r_mult = 1.5
            else:
                pnl = entry - next_day['close']
                r_mult = pnl / risk if risk > 0 else 0
        
        results.append(TradeResult(r_mult, day.name, "Momentum_Continuation"))
    
    return results

def run_research():
    try:
        import MetaTrader5 as mt5
    except ImportError:
        print("MetaTrader5 not available")
        return
    
    if not mt5.initialize(path=r"C:\Program Files\MetaTrader 5 Terminal\terminal64.exe"):
        print("MT5 init failed")
        return
    
    print("="*70)
    print("  MULTI-FRONTIER STRATEGY RESEARCH")
    print("  (MaxDD < 5R Requirement)")
    print("="*70)
    
    symbols = ['EURUSDm', 'GBPUSDm', 'USDJPYm']
    end = datetime.now()
    start = end - timedelta(days=90)
    
    frontiers = {
        "NY Close Reversion": {},
        "Session Overlap": {},
        "Momentum Continuation": {}
    }
    
    for symbol in symbols:
        # M15 Data
        rates_m15 = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M15, start, end)
        if rates_m15 is None or len(rates_m15) < 500: continue
        df_m15 = pd.DataFrame(rates_m15)
        df_m15['time'] = pd.to_datetime(df_m15['time'], unit='s')
        df_m15.set_index('time', inplace=True)
        
        # D1 Data
        rates_d1 = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_D1, start - timedelta(days=30), end)
        df_d1 = None
        if rates_d1 is not None and len(rates_d1) > 30:
            df_d1 = pd.DataFrame(rates_d1)
            df_d1['time'] = pd.to_datetime(df_d1['time'], unit='s')
            df_d1.set_index('time', inplace=True)
        
        # Test Frontier 1
        trades = test_ny_close_reversion(df_m15, symbol)
        if trades:
            frontiers["NY Close Reversion"][symbol] = calculate_metrics(trades)
        
        # Test Frontier 2
        trades = test_session_overlap(df_m15, symbol)
        if trades:
            frontiers["Session Overlap"][symbol] = calculate_metrics(trades)
        
        # Test Frontier 3
        if df_d1 is not None:
            trades = test_momentum_continuation(df_m15, df_d1, symbol)
            if trades:
                frontiers["Momentum Continuation"][symbol] = calculate_metrics(trades)
    
    # Print Results
    for frontier, results in frontiers.items():
        print(f"\n{'='*70}")
        print(f"  FRONTIER: {frontier}")
        print(f"{'='*70}")
        
        total_r = 0
        total_trades = 0
        max_dd_overall = 0
        
        for symbol, m in results.items():
            print(f"  {symbol}: {m['total_trades']} trades | WR: {m['win_rate']:.1f}% | R: {m['total_r']:.1f}R | MaxDD: {m['max_dd']:.1f}R")
            total_r += m['total_r']
            total_trades += m['total_trades']
            max_dd_overall = max(max_dd_overall, m['max_dd'])
        
        # Verdict
        if total_trades > 0:
            passed = total_r > 15 and max_dd_overall < 5
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"\n  COMBINED: {total_trades} trades | {total_r:.1f}R | MaxDD: {max_dd_overall:.1f}R | {status}")
    
    mt5.shutdown()

if __name__ == "__main__":
    run_research()
