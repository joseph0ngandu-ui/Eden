#!/usr/bin/env python3
"""
PHASE 10: HIGH VELOCITY SCALPING PROTOTYPE
Testing "Momentum Burst" logic to capture explosive moves that ignore spread.
Focus: Volume Ignition + Price Displacement.
"""

import sys
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from datetime import datetime, time as dt_time, timedelta
from pathlib import Path
import logging
import traceback

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Phase10")

class Phase10Research:
    def __init__(self):
        if not mt5.initialize():
            print("MT5 Init Failed")
            sys.exit(1)
            
    def get_data(self, symbol, timeframe, days=15):
        try:
            utc_from = datetime.now() - timedelta(days=days)
            rates = mt5.copy_rates_from(symbol, timeframe, datetime.now(), 20000) 
            if rates is None: 
                print(f"No data for {symbol}")
                return None
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            df = df[df.index > utc_from]
            # print(f"Loaded {len(df)} bars for {symbol}")
            return df
        except Exception as e:
            print(f"Error getting data for {symbol}: {e}")
            return None

    def calculate_atr(self, df, period=14):
        tr = df['high'] - df['low']
        return tr.rolling(period).mean()

    def outcome(self, future, direction, entry, sl, tp):
        for idx, row in future.iterrows():
            if direction == "LONG":
                if row['low'] <= sl: return -1.0
                if row['high'] >= tp: return 1.5 # Target 1.5R (High velocity targets)
            else:
                if row['high'] >= sl: return -1.0
                if row['low'] <= tp: return 1.5
        return 0.0

    def simulate_pnl(self, trades, name):
        clean_trades = [t for t in trades if isinstance(t, (int, float))]
        count = len(clean_trades)
        total_r = sum(clean_trades)
        
        equity = [0]
        for r in clean_trades:
            equity.append(equity[-1] + r)
        
        peak = -99999
        max_dd = 0
        for e in equity:
            if e > peak: peak = e
            dd = peak - e
            if dd > max_dd: max_dd = dd
            
        result_str = f"Strategy: {name:<25} | Trades: {count:<4} | Total R: {total_r:>6.2f} | Max DD: {max_dd:>5.2f}R"
        print(result_str)
        return total_r, max_dd

    # ==========================
    # MOMENTUM BURST STRATEGY
    # ==========================
    def test_momentum_burst(self, symbol, timeframe=mt5.TIMEFRAME_M5):
        try:
            df = self.get_data(symbol, timeframe, days=14) # Last 2 weeks
            if df is None: return 0, 0
            
            trades = []
            atr = self.calculate_atr(df)
            
            # Indicators
            vol_ma = df['tick_volume'].rolling(20).mean()
            ema_50 = df['close'].ewm(span=50).mean()
            
            for i in range(50, len(df)):
                curr = df.iloc[i]
                
                # 1. Volume Ignition (3x average)
                if curr['tick_volume'] < vol_ma.iloc[i] * 3.0: continue
                
                # 2. Displacement (Body > 1.5x ATR)
                body = abs(curr['close'] - curr['open'])
                curr_atr = atr.iloc[i]
                if body < curr_atr * 1.5: continue
                
                # 3. Context (Trend + Location)
                if curr['close'] > curr['open']: # Bullish Burst
                    if curr['close'] > ema_50.iloc[i]: # With Trend
                        entry = curr['close']
                        sl = curr['low']
                        # Safety: If stop is too tight/wide
                        if (entry - sl) < curr_atr * 0.5: sl = entry - curr_atr * 0.5
                        tp = entry + (entry - sl) * 1.5
                        
                        trades.append(self.outcome(df.iloc[i:], "LONG", entry, sl, tp))
                        
                else: # Bearish Burst
                    if curr['close'] < ema_50.iloc[i]: # With Trend
                        entry = curr['close']
                        sl = curr['high']
                        if (sl - entry) < curr_atr * 0.5: sl = entry + curr_atr * 0.5
                        tp = entry - (sl - entry) * 1.5
                        
                        trades.append(self.outcome(df.iloc[i:], "SHORT", entry, sl, tp))
                        
            return self.simulate_pnl(trades, f"MomBurst_{symbol}")
            
        except Exception as e:
            print(f"CRASH in MomBurst {symbol}: {e}")
            traceback.print_exc()
            return 0, 0

if __name__ == "__main__":
    r = Phase10Research()
    print("=== PHASE 10: MOMENTUM BURST (IGNITION) ===")
    print("Hypothesis: Huge volume + Huge candle = Follow Through")
    
    # 1. Gold (High Volatility)
    r.test_momentum_burst("XAUUSDm", mt5.TIMEFRAME_M5)
    r.test_momentum_burst("XAUUSDm", mt5.TIMEFRAME_M1)

    # 2. Indices (Explosive)
    r.test_momentum_burst("USTECm", mt5.TIMEFRAME_M5)
    
    # 3. Volatile Forex
    r.test_momentum_burst("GBPJPYm", mt5.TIMEFRAME_M5)
    r.test_momentum_burst("EURUSDm", mt5.TIMEFRAME_M5)
