#!/usr/bin/env python3
"""
LONDON SESSION BREAKOUT RESEARCH (Phase 4)
Captures directional momentum at London Open (07:00-10:00 GMT).
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta, time
from dataclasses import dataclass
from typing import Optional, List
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

class LondonBreakoutEngine:
    def calculate_atr(self, df: pd.DataFrame, period=14):
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def calculate_adr(self, df: pd.DataFrame, period=14):
        """Average Daily Range (High - Low per day)."""
        df_copy = df.copy()
        df_copy['date'] = df_copy.index.date
        daily = df_copy.groupby('date').agg({'high': 'max', 'low': 'min'})
        daily['range'] = daily['high'] - daily['low']
        return daily['range'].rolling(period).mean().iloc[-1] if len(daily) > period else daily['range'].mean()

    def evaluate(self, df: pd.DataFrame, symbol: str) -> Optional[TradeSignal]:
        """Evaluate London Breakout Signal."""
        if len(df) < 100: return None
        
        current = df.iloc[-1]
        timestamp = current.name
        t = timestamp.time()
        
        # 1. Time Filter: London Entry Window (07:00 - 10:00 Server Time)
        # Assuming Server Time = GMT+2/+3, so 07:00 GMT = 09:00/10:00 Server
        # I'll use 08:00 - 11:00 Server Time to be safe
        start_trade = time(8, 0)
        end_trade = time(11, 0)
        
        if not (start_trade <= t <= end_trade):
            return None
        
        # 2. Define Asian Range (00:00 - 07:00 Server Time => 22:00 - 05:00 GMT approx)
        # Use 00:00 - 08:00 Server for wider range
        today_date = timestamp.date()
        range_start = pd.Timestamp.combine(today_date, time(0, 0))
        range_end = pd.Timestamp.combine(today_date, time(8, 0))
        
        range_data = df[(df.index >= range_start) & (df.index < range_end)]
        
        if len(range_data) < 10: # Not enough data for range
            return None
            
        range_high = range_data['high'].max()
        range_low = range_data['low'].min()
        range_width = range_high - range_low
        
        # 3. ADR Filter: Ensure volatility is available
        adr = self.calculate_adr(df)
        if range_width > adr * 0.8: # Range too wide (already extended)
            return None
            
        # 4. Trend Filter: EMA 50
        ema_50 = df['close'].ewm(span=50).mean()
        trend_up = current['close'] > ema_50.iloc[-1]
        trend_down = current['close'] < ema_50.iloc[-1]
        
        # 5. ATR for SL/TP
        atr_series = self.calculate_atr(df)
        atr = atr_series.iloc[-1]
        price = current['close']
        
        # 6. Breakout Detection
        breakout_long = price > range_high and trend_up
        breakout_short = price < range_low and trend_down
        
        if breakout_long:
            sl = range_low - atr * 0.3
            risk = price - sl
            tp = price + risk * 1.5 # 1.5R
            return TradeSignal(symbol, "LONG", price, sl, tp, timestamp, "London Breakout")
            
        if breakout_short:
            sl = range_high + atr * 0.3
            risk = sl - price
            tp = price - risk * 1.5
            return TradeSignal(symbol, "SHORT", price, sl, tp, timestamp, "London Breakout")
            
        return None

def check_outcome(signal, future, max_bars=200):
    """Check trade outcome. Returns R-multiple."""
    entry = signal.entry_price
    risk = abs(entry - signal.sl)
    reward = abs(signal.tp - entry)
    r_mult = reward / risk if risk > 0 else 0
    
    for idx, row in future.iterrows():
        if signal.direction == "LONG":
            if row['low'] <= signal.sl: return -1.0
            if row['high'] >= signal.tp: return r_mult
        else: # SHORT
            if row['high'] >= signal.sl: return -1.0
            if row['low'] <= signal.tp: return r_mult
    
    # Timeout: Use last close
    close = future.iloc[-1]['close']
    if signal.direction == "LONG":
        pnl = close - entry
    else:
        pnl = entry - close
    return pnl / risk if risk > 0 else 0

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
    print("  LONDON BREAKOUT RESEARCH (M15)")
    print("="*60)
    
    symbols = ['GBPUSDm', 'EURUSDm', 'GBPJPYm']
    tf = mt5.TIMEFRAME_M15
    end = datetime.now()
    start = end - timedelta(days=90) # 3 Months
    
    engine = LondonBreakoutEngine()
    
    for symbol in symbols:
        rates = mt5.copy_rates_range(symbol, tf, start, end)
        if rates is None or len(rates) < 500:
            print(f"{symbol}: Insufficient data")
            continue
            
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        trades = []
        
        # Iterate through bars
        for i in range(300, len(df) - 200):
            window = df.iloc[i-200:i+1] # 200 bars history
            signal = engine.evaluate(window, symbol)
            
            if signal:
                future = df.iloc[i+1:i+200]
                result = check_outcome(signal, future)
                trades.append(result)
        
        if trades:
            wins = len([t for t in trades if t > 0])
            wr = wins / len(trades) * 100
            total_r = sum(trades)
            print(f"{symbol}: {len(trades)} trades | WR: {wr:.1f}% | Total: {total_r:.1f}R")
        else:
            print(f"{symbol}: No trades")

    mt5.shutdown()

if __name__ == "__main__":
    run_backtest()
