#!/usr/bin/env python3
"""
ASIAN RANGE FADE STRATEGY
Research Implementation

Thesis: During Asian Session (00:00 - 08:00), pairs range-bound.
Logic: Fade the edges of a defined range (High/Low of pre-Asian).
"""

import sys
from pathlib import Path
from datetime import datetime, time, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import logging

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TradeSignal:
    symbol: str
    direction: str
    entry_price: float
    sl: float
    tp: float
    time: pd.Timestamp
    reason: str

class AsianFadeEngine:
    def __init__(self):
        self.session_start = time(0, 0)
        self.session_end = time(8, 0) # Trade until 8 AM GMT
        
    def calculate_atr(self, df: pd.DataFrame, period=14):
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean().iloc[-1]

    def calculate_rsi(self, series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs.iloc[-1]))

    def evaluate(self, df: pd.DataFrame, symbol: str) -> Optional[TradeSignal]:
        # Pre-checks
        if len(df) < 200: return None
        
        current = df.iloc[-1]
        timestamp = current.name
        t = timestamp.time()
        
        # 1. Time Filter: Asian Session (01:00 - 09:00 Server Time to avoid rollover)
        start_trade = time(1, 0)
        end_trade = time(9, 0)
        
        if not (start_trade <= t <= end_trade):
            return None
            
        # 2. Define Fixed Range (20:00 Prev Day - 00:00 Today)
        # We need to find the data rows corresponding to this window
        # Current date
        today_date = timestamp.date()
        prev_date = today_date - timedelta(days=1)
        
        # Construct timestamps
        range_start = pd.Timestamp.combine(prev_date, time(20, 0))
        range_end = pd.Timestamp.combine(today_date, time(0, 0))
        
        # Slice df
        # Ensure df index is localized or matching. Assuming naive match for now.
        # df might be small slice from runner, check start time.
        # Runner passes full history? No, runner passed window_slice of 100 bars.
        # 100 bars M5 = 500 mins = 8 hours.
        # If current is 04:00, window goes back to 20:00 previous day (8 hours ago).
        # So we barely have the data.
        # Need larger window in runner.
        
        # ALERT: We need to update runner to pass larger window.
        # For now, let's assume we can get the range from the slice provided IF it covers it.
        # If not, return None.
        
        range_data = df[(df.index >= range_start) & (df.index < range_end)]
        
        if len(range_data) < 10: # Insufficient data
            return None
            
        range_high = range_data['high'].max()
        range_low = range_data['low'].min()
        range_width = range_high - range_low
        
        atr = self.calculate_atr(df)
        
        # 3. Volatility Filter (Tight Range)
        if range_width > 2.0 * atr: # Stricter
            return None
        if range_width < 0.5 * atr: # Too flat
            return None
            
        # 4. Entry Logic
        price = current['close']
        rsi = self.calculate_rsi(df['close'])
        
        # Fade High
        # Trigger: Price touches High or goes slightly above (Fakeout) then comes back?
        # Simple Fade: Price > High * 0.999 (Touched)
        # And RSI > 55 (Momentum slowing? No, RSI usually high at top)
        # Let's say: Price within 3 pips of High.
        
        # Use relative distance
        dist_to_high = abs(range_high - price)
        dist_to_low = abs(price - range_low)
        threshold = range_width * 0.1 # top 10%
        
        if price >= (range_high - threshold):
             if rsi > 60:
                 sl = range_high + (range_width * 0.5) # Wide SL? Or ATR based?
                 sl = range_high + atr
                 tp = range_low + (range_width * 0.5) # Mid-range
                 
                 risk = abs(price - sl)
                 reward = abs(price - tp)
                 if risk > 0 and reward/risk > 1.0:
                     return TradeSignal(symbol, "SHORT", price, sl, tp, timestamp, "Fixed Range Fade")

        if price <= (range_low + threshold):
             if rsi < 40:
                 sl = range_low - atr
                 tp = range_low + (range_width * 0.5)
                 
                 risk = abs(price - sl)
                 reward = abs(price - tp)
                 if risk > 0 and reward/risk > 1.0:
                     return TradeSignal(symbol, "LONG", price, sl, tp, timestamp, "Fixed Range Fade")
                     
        return None

def run_backtest():
    try:
        import MetaTrader5 as mt5
    except ImportError:
        print("MetaTrader5 not available")
        return
        
    if not mt5.initialize(path=r"C:\Program Files\MetaTrader 5 Terminal\terminal64.exe"):
        print("MT5 init failed")
        return
        
    print("="*60)
    print("  ASIAN FADE RESEARCH (M5)")
    print("="*60)
    
    symbols = ['USDJPYm', 'AUDUSDm', 'EURUSDm', 'GBPUSDm']
    tf = mt5.TIMEFRAME_M5
    
    end = datetime.now()
    start = end - timedelta(days=60) # Last 60 days
    
    engine = AsianFadeEngine()
    
    for symbol in symbols:
        rates = mt5.copy_rates_range(symbol, tf, start, end)
        if rates is None or len(rates) < 100:
            print(f"{symbol}: No data")
            continue
            
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        trades = []
        
        # Iterate (skip warmup)
        for i in range(300, len(df) - 100): 
            window_slice = df.iloc[i-300:i+1] 
            signal = engine.evaluate(window_slice, symbol)
            
            if signal:
                future = df.iloc[i+1:i+200]  # Look ahead 200 bars (~16 hours max hold)
                res = check_outcome(signal, future)
                trades.append(res)
        
        if not trades:
            print(f"{symbol}: No trades")
            continue
            
        total_r = sum(trades)
        wins = [t for t in trades if t > 0]
        wr = len(wins)/len(trades)*100
        print(f"{symbol}: {len(trades)} trades | WR: {wr:.1f}% | Total: {total_r:.1f}R")

    mt5.shutdown()

def check_outcome(signal, future):
    entry = signal.entry_price
    risk = abs(entry - signal.sl)
    reward = abs(signal.tp - entry)
    r_mult = reward / risk if risk > 0 else 0
    
    for idx, row in future.iterrows():
        if signal.direction == "LONG":
            if row['low'] <= signal.sl: return -1.0
            if row['high'] >= signal.tp: return r_mult
        else:
            if row['high'] >= signal.sl: return -1.0
            if row['low'] <= signal.tp: return r_mult
            
    # Close at end (Timelimit)
    close = future.iloc[-1]['close']
    if signal.direction == "LONG": pnl = close - entry
    else: pnl = entry - close
    return pnl / risk if risk > 0 else 0

if __name__ == "__main__":
    run_backtest()
