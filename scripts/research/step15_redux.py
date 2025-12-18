#!/usr/bin/env python3
"""
PHASE 15: STRATEGY RESURRECTION (London & Forex Judas)

1. London Breakout (GBPCAD)
   - Logic: Measure Asia Range (00:00 - 08:00 Server).
   - Entry: Break of High/Low after 08:00.
   - Filter: Range size < X ATR.
   
2. Judas Swing (EURUSD/USDJPY)
   - Logic: Sweep of Asia High/Low.
   - Entry: Reversal Pattern (FVG) after Sweep.
   - Stop: Tight (Swing Extreme).
"""

import sys
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from datetime import datetime, timedelta
from pathlib import Path
import logging

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.ERROR)

class Phase15Research:
    def __init__(self):
        if not mt5.initialize():
            print("MT5 Init Failed")
            sys.exit(1)
            
    def get_data(self, symbol, timeframe, days=15):
        utc_from = datetime.now() - timedelta(days=days)
        rates = mt5.copy_rates_from(symbol, timeframe, datetime.now(), 50000) 
        if rates is None: return None
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        df = df[df.index > utc_from]
        return df

    def calculate_atr(self, df):
        high = df['high']
        low = df['low']
        close = df['close']
        tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
        return tr.rolling(14).mean()

    def outcome(self, future, direction, entry, sl, tp, spread):
        real_entry = entry + spread if direction == "LONG" else entry
        
        for idx, row in future.iterrows():
            if direction == "LONG":
                # Check SL (Bid <= SL)
                if row['low'] <= sl: return -1.0
                # Check TP (Bid >= TP)
                if row['high'] >= tp: return 2.0 # Fixed 2R for these tests
            else: # SHORT
                # Check SL (Ask >= SL -> Bid >= SL - Spread)
                if row['high'] >= (sl - spread): return -1.0
                # Check TP
                if row['low'] <= (tp - spread): return 2.0
                
        return 0.0

    def test_london_breakout(self):
        symbol = "GBPCADm" # The "Reserved" pair
        print(f"\nTesting London Breakout ({symbol})...")
        
        df = self.get_data(symbol, mt5.TIMEFRAME_M15, days=8) # Bad Week
        if df is None: return
        
        df['atr'] = self.calculate_atr(df)
        spread = 0.00025 # Approx 2.5 pips GBPCAD
        
        trades = []
        
        # Iterate Days
        days = df.groupby(df.index.date)
        
        for date, day_data in days:
            # 1. Define Asia Range (00:00 - 08:00)
            # Server time usually GMT+2/3. London Open is 10:00 Server?
            # Let's assume standard Broker (GMT+2 winter, GMT+3 summer).
            # NY is 16:00. London is 10:00.
            # Asia is 00:00 - 09:00?
            # Let's simple check 01:00 to 09:00.
            
            asia = day_data.between_time('01:00', '09:00')
            if len(asia) < 5: continue
            
            asia_high = asia['high'].max()
            asia_low = asia['low'].min()
            asia_range = asia_high - asia_low
            
            # Filter: Range too big?
            atr = day_data['atr'].iloc[0] if not pd.isna(day_data['atr'].iloc[0]) else 0
            if atr > 0 and asia_range > atr * 3: continue # Expanded range = chop
            
            # 2. Look for Breakout (09:00 to 12:00)
            london = day_data.between_time('09:00', '13:00')
            
            triggered = False
            
            for t, row in london.iterrows():
                if triggered: break
                
                # Break High
                if row['close'] > asia_high:
                    sl = asia_low # Stop at other side (Classic)
                    # Optimization: Stop at mid-range?
                    sl = asia_high - asia_range * 0.5
                    tp = row['close'] + (row['close'] - sl) * 2.0
                    
                    outcome = self.outcome(day_data.loc[t:].iloc[1:], "LONG", row['close'], sl, tp, spread)
                    trades.append(outcome)
                    triggered = True
                    
                # Break Low
                elif row['close'] < asia_low:
                    sl = asia_high - asia_range * 0.5
                    tp = row['close'] - (sl - row['close']) * 2.0
                    
                    outcome = self.outcome(day_data.loc[t:].iloc[1:], "SHORT", row['close'], sl, tp, spread)
                    trades.append(outcome)
                    triggered = True

        clean = [t for t in trades if t != 0]
        print(f"London BO: Trades {len(clean)} | Total R: {sum(clean):.2f}")

    def test_judas_forex(self, symbol="EURUSDm"):
        print(f"\nTesting Judas Forex ({symbol})...")
        df = self.get_data(symbol, mt5.TIMEFRAME_M5, days=8) 
        if df is None: return
        
        spread = 0.00012
        trades = []
        
        days = df.groupby(df.index.date)
        
        for date, day_data in days:
            # Asia Range
            asia = day_data.between_time('00:00', '09:00')
            if len(asia) < 10: continue
            
            asia_high = asia['high'].max()
            asia_low = asia['low'].min()
            
            # London Session (09:00 - 12:00)
            london = day_data.between_time('09:00', '12:00')
            
            swept_high = False
            swept_low = False
            
            for i in range(2, len(london)):
                row = london.iloc[i]
                prev = london.iloc[i-1]
                
                # Check for Sweep of High
                if row['high'] > asia_high and not swept_high:
                    # Look for Bearish Reversal (Close back inside?)
                    if row['close'] < asia_high:
                         # Entry Short
                         sl = row['high'] # Tight stop at sweep high
                         entry = row['close']
                         tp = entry - (sl - entry) * 2.5 # High RR
                         
                         outcome = self.outcome(day_data.loc[row.name:].iloc[1:], "SHORT", entry, sl, tp, spread)
                         trades.append(outcome)
                         swept_high = True # One per session
                
                # Check for Sweep of Low
                if row['low'] < asia_low and not swept_low:
                    # Look for Bullish Reversal
                    if row['close'] > asia_low:
                        # Entry Long
                        sl = row['low']
                        entry = row['close']
                        tp = entry + (entry - sl) * 2.5
                        
                        outcome = self.outcome(day_data.loc[row.name:].iloc[1:], "LONG", entry, sl, tp, spread)
                        trades.append(outcome)
                        swept_low = True

        clean = [t for t in trades if t != 0]
        print(f"Judas {symbol}: Trades {len(clean)} | Total R: {sum(clean):.2f}")

if __name__ == "__main__":
    r = Phase15Research()
    print("=== PHASE 15: STRATEGY RESURRECTION ===")
    r.test_london_breakout()
    r.test_judas_forex("EURUSDm")
    r.test_judas_forex("USDJPYm")
