#!/usr/bin/env python3
"""
VOLATILITY EXPANSION STRATEGY (INDICES)
Research Implementation

Thesis: Indices (US30, NAS100) trend strongly after volatility compression (Squeeze).
Logic: Bollinger Band Squeeze -> Breakout -> Trend
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
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

class VolatilityExpansionEngine:
    def __init__(self, initial_balance=100000.0):
        self.balance = initial_balance
        self.bandwidth_history: Dict[str, List[float]] = {}
        
    def calculate_bb(self, df: pd.DataFrame, period=20, std_dev=2.0) -> pd.DataFrame:
        df = df.copy()
        df['sma'] = df['close'].rolling(period).mean()
        df['std'] = df['close'].rolling(period).std()
        df['upper'] = df['sma'] + (df['std'] * std_dev)
        df['lower'] = df['sma'] - (df['std'] * std_dev)
        df['bandwidth'] = (df['upper'] - df['lower']) / df['sma']
        return df

    def get_avg_bandwidth(self, symbol: str) -> float:
        if symbol not in self.bandwidth_history: return 0.005 # Default fallback
        return np.mean(self.bandwidth_history[symbol])

    def evaluate(self, df: pd.DataFrame, symbol: str) -> Optional[TradeSignal]:
        if len(df) < 50: return None
        
        # Calculate BB
        bb = self.calculate_bb(df)
        current = bb.iloc[-1]
        prev = bb.iloc[-2]
        
        # Update Bandwidth History for Relative Squeeze
        bw = current['bandwidth']
        if symbol not in self.bandwidth_history:
            self.bandwidth_history[symbol] = []
        self.bandwidth_history[symbol].append(bw)
        if len(self.bandwidth_history[symbol]) > 200:
            self.bandwidth_history[symbol].pop(0)
            
        avg_bw = self.get_avg_bandwidth(symbol)
        
        # 1. Squeeze Condition (Relative)
        # We look for a squeeze that happened RECENTLY (e.g. within last 3 bars)
        # to allow for the breakout candle to be expanding.
        recent_squeeze = False
        for i in range(1, 6):
            if bb.iloc[-i]['bandwidth'] < avg_bw * 0.8:
                recent_squeeze = True
                break
                
        if not recent_squeeze:
            return None
            
        # 2. Breakout Condition
        # Price closes outside bands
        breakout_long = current['close'] > current['upper']
        breakout_short = current['close'] < current['lower']
        
        # 3. Expansion Confirmation (Bandwidth increasing)
        expanding = current['bandwidth'] > prev['bandwidth']
        
        if not expanding:
            return None

        # Filter: NY Session Only (13:30 - 20:00 GMT approx)
        # Assuming df index is datetime
        hour = current.name.hour
        is_ny = 13 <= hour <= 20
        if not is_ny:
            return None

        atr = (df['high'] - df['low']).rolling(14).mean().iloc[-1]

        if breakout_long:
            sl = current['sma'] # SL at Middle Band (Reversion limit)
            tp = current['close'] + (current['close'] - sl) * 1.5 # 1.5 RR
            return TradeSignal(symbol, "LONG", current['close'], sl, tp, current.name, "BB Breakout Long")
            
        if breakout_short:
            sl = current['sma']
            tp = current['close'] - (sl - current['close']) * 1.5
            return TradeSignal(symbol, "SHORT", current['close'], sl, tp, current.name, "BB Breakout Short")
            
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
    print("  VOLATILITY EXPANSION RESEARCH (Indices)")
    print("="*60)
    
    symbols = ['US30m', 'USTECm', 'US500m']
    timeframes = [(mt5.TIMEFRAME_M15, 'M15', 50)]
    
    end = datetime.now()
    start = end - timedelta(days=60)
    
    for tf, tf_name, future_bars in timeframes:
        print(f"\n--- TIMEFRAME: {tf_name} ---")
        engine = VolatilityExpansionEngine()
        
        for symbol in symbols:
            rates = mt5.copy_rates_range(symbol, tf, start, end)
            if rates is None or len(rates) < 100:
                print(f"{symbol}: No data (Check symbol name?)")
                continue
                
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            # Pre-calculate Trend
            df['ema_50'] = df['close'].rolling(50).mean()
            
            trades = []
            
            # Warmup bandwidth history
            for i in range(200, len(df) - future_bars):
                window_slice = df.iloc[i-100:i+1]
                trend_val = df['ema_50'].iloc[i]
                
                # Pass trend to evaluate? Or calc inside? 
                # Calc inside window_slice is fine but rolling(50) needs 50 bars. window has 100. OK.
                
                signal = engine.evaluate(window_slice, symbol)
                
                # Apply Trend Filter externally or inside?
                # Inside is cleaner but requires edit to evaluate.
                # I'll apply it here to save edits to class logic if possible?
                # No, better to edit evaluate.
                
                if signal:
                    # Filter
                    if signal.direction == "LONG" and signal.entry_price < trend_val:
                        continue
                    if signal.direction == "SHORT" and signal.entry_price > trend_val:
                        continue
                        
                    # Check outcome
                    future = df.iloc[i+1:i+future_bars]
                    res = check_outcome(signal, future)
                    trades.append(res)
            
            if not trades:
                print(f"{symbol}: No trades")
                continue
                
            wins = [t for t in trades if t > 0]
            wr = len(wins)/len(trades)*100
            total = sum(trades)
            print(f"{symbol}: {len(trades)} trades | WR: {wr:.1f}% | Total: {total:.1f}R")

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
    
    # Close at end
    close = future.iloc[-1]['close']
    
    if signal.direction == "LONG":
        pnl = close - entry
    else:
        pnl = entry - close
    return pnl / risk if risk > 0 else 0
    
    # Close at end
    entry = signal.entry_price
    close = future.iloc[-1]['close']
    risk = abs(entry - signal.sl)
    if risk == 0: return 0
    
    if signal.direction == "LONG":
        pnl = close - entry
    else:
        pnl = entry - close
    return pnl / risk

if __name__ == "__main__":
    run_backtest()
