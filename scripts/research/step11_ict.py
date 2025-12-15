#!/usr/bin/env python3
"""
PHASE 11: ICT/SMC HYBRID RESEARCH PROTOTYPE
Testing advanced ICT concepts to find "Spread Eating" setups.
1. "Judas Swing" (Asian Sweep + FVG Reversal)
2. "Silver Bullet" (Time-Based Continuation)
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
logger = logging.getLogger("Phase11")

class Phase11Research:
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
            return df
        except Exception as e:
            print(f"Error getting data for {symbol}: {e}")
            return None

    def calculate_atr(self, df, period=14):
        tr = df['high'] - df['low']
        return tr.rolling(period).mean()

    def outcome(self, future, direction, entry, sl, tp):
        filled = False
        for idx, row in future.iterrows():
            # Check Fill First
            if not filled:
                if direction == "LONG":
                     if row['low'] <= entry: filled = True
                else:
                     if row['high'] >= entry: filled = True
                if not filled: continue # Wait for fill
            
            # Once Filled, Check Outcome
            if direction == "LONG":
                if row['low'] <= sl: return -1.0
                if row['high'] >= tp: return 2.0 
            else:
                if row['high'] >= sl: return -1.0
                if row['low'] <= tp: return 2.0
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
    # STRATEGY 1: JUDAS SWING (Sweep + FVG)
    # ==========================
    def test_judas_swing(self, symbol, timeframe=mt5.TIMEFRAME_M5):
        try:
            df = self.get_data(symbol, timeframe, days=14)
            if df is None: return 0, 0
            
            trades = []
            atr = self.calculate_atr(df)
            
            # Identify Asian Ranges
            # For each day, get High/Low from 00:00 to 08:00
            
            # Simple day iteration
            days = df.groupby(df.index.date)
            
            for date, day_data in days:
                # 1. Asian Range
                asian_data = day_data.between_time("00:00", "08:00")
                if len(asian_data) == 0: continue
                
                asian_high = asian_data['high'].max()
                asian_low = asian_data['low'].min()
                
                # 2. Look for Sweep (08:00 - 16:00)
                session_data = day_data.between_time("08:00", "16:00")
                if len(session_data) < 5: continue
                
                swept_high = False
                swept_low = False
                
                fvgs = []
                
                for i in range(2, len(session_data)):
                    c1 = session_data.iloc[i-2]
                    c2 = session_data.iloc[i-1]
                    c3 = session_data.iloc[i] # Current Candle
                    
                    # Check Sweep
                    if c3['high'] > asian_high: swept_high = True
                    if c3['low'] < asian_low: swept_low = True
                    
                    # Check FVG Formation
                    # Bullish FVG
                    if c1['high'] < c3['low'] and c2['close'] > c2['open']:
                        fvgs.append({'type': 'BULL', 'top': c3['low'], 'bot': c1['high'], 'time': c3.name})
                        
                    # Bearish FVG
                    if c1['low'] > c3['high'] and c2['close'] < c2['open']:
                        fvgs.append({'type': 'BEAR', 'top': c1['low'], 'bot': c3['high'], 'time': c3.name})
                        
                    # === LOGIC ===
                    # If we swept HIGH -> Look for BEARISH FVG entry
                    if swept_high and not swept_low:
                         # Check recent Bearish FVGs
                         for fvg in fvgs:
                             # FVG must be recent (within 30 mins) and after the sweep?
                             # Simplified: If price is IN the FVG now, sell.
                             if fvg['type'] == 'BEAR':
                                 if c3['high'] >= fvg['bot'] and c3['low'] <= fvg['top']:
                                     # ENTRY SIGNAl
                                     entry = fvg['bot']
                                     sl = session_data['high'].iloc[:i+1].max() # High of session so far
                                     risk = sl - entry
                                     
                                     curr_atr = atr.loc[c3.name]
                                     if risk < curr_atr * 0.5 or risk > curr_atr * 4.0: continue
                                     
                                     tp = entry - risk * 2.0
                                     
                                     # Execute
                                     outcome = self.outcome(session_data.iloc[i+1:], "SHORT", entry, sl, tp)
                                     trades.append(outcome)
                                     
                                     # Reset for day to avoid spamming
                                     swept_high = False 
                                     swept_low = False # Stop trading for day
                                     break
                                     
                    # If we swept LOW -> Look for BULLISH FVG entry
                    if swept_low and not swept_high:
                        for fvg in fvgs:
                            if fvg['type'] == 'BULL':
                                if c3['low'] <= fvg['top'] and c3['high'] >= fvg['bot']:
                                    entry = fvg['top']
                                    sl = session_data['low'].iloc[:i+1].min()
                                    risk = entry - sl
                                    
                                    curr_atr = atr.loc[c3.name]
                                    if risk < curr_atr * 0.5 or risk > curr_atr * 4.0: continue
                                    
                                    tp = entry + risk * 2.0
                                    
                                    outcome = self.outcome(session_data.iloc[i+1:], "LONG", entry, sl, tp)
                                    trades.append(outcome)
                                    
                                    swept_low = False
                                    swept_high = False
                                    break
                                    
            return self.simulate_pnl(trades, f"Judas_{symbol}")
            
        except Exception as e:
            print(f"CRASH in Judas {symbol}: {e}")
            traceback.print_exc()
            return 0, 0

    # ==========================
    # STRATEGY 2: SILVER BULLET (Time + FVG)
    # ==========================
    def test_silver_bullet(self, symbol, timeframe=mt5.TIMEFRAME_M5):
        # 10:00 - 11:00 NY Time
        # Assuming Server Time is UTC+2 (EET)
        # NY 10:00 = 15:00 UTC = 17:00 Server
        # Adjust as needed. Let's assume generic 'active' hours for prototype
        try:
            df = self.get_data(symbol, timeframe, days=14)
            if df is None: return 0, 0
            
            trades = []
            atr = self.calculate_atr(df)
            
            start_hour = 16 # 10 AM NY approx
            end_hour = 17   # 11 AM NY approx
            
            for i in range(5, len(df)):
                c1 = df.iloc[i-2]
                c2 = df.iloc[i-1]
                c3 = df.iloc[i]
                timestamp = c3.name
                
                if not (start_hour <= timestamp.hour <= end_hour): continue
                
                # Identify FVG
                # Bullish FVG
                bull_fvg = False
                if c1['high'] < c3['low'] and c2['close'] > c2['open']:
                    bull_fvg = True
                    entry = c3['low'] # Limit entry at top of FVG
                    sl = c1['low']
                    
                # Bearish FVG
                bear_fvg = False
                if c1['low'] > c3['high'] and c2['close'] < c2['open']:
                    bear_fvg = True
                    entry = c3['high']
                    sl = c1['high']
                    
                curr_atr = atr.iloc[i]
                
                # Market Structure Filter (Simple MA)
                ema_50 = df['close'].ewm(span=50).mean().iloc[i]
                
                if bull_fvg and c3['close'] > ema_50:
                    trades.append(self.outcome(df.iloc[i+1:], "LONG", entry, sl, entry + (entry-sl)*2))
                    
                if bear_fvg and c3['close'] < ema_50:
                    trades.append(self.outcome(df.iloc[i+1:], "SHORT", entry, sl, entry - (sl-entry)*2))

            return self.simulate_pnl(trades, f"SilverBullet_{symbol}")

        except Exception as e:
            print(f"CRASH in SilverBullet {symbol}: {e}")
            return 0, 0

if __name__ == "__main__":
    r = Phase11Research()
    print("=== PHASE 11: ICT HYBRID (SPREAD EATERS) ===")
    
    # 1. Gold
    r.test_judas_swing("XAUUSDm", mt5.TIMEFRAME_M5)
    r.test_silver_bullet("XAUUSDm", mt5.TIMEFRAME_M5)

    # 2. Indices
    r.test_judas_swing("USTECm", mt5.TIMEFRAME_M5)
    r.test_silver_bullet("USTECm", mt5.TIMEFRAME_M5)
    
    # 3. Forex
    r.test_judas_swing("EURUSDm", mt5.TIMEFRAME_M5)
    r.test_silver_bullet("EURUSDm", mt5.TIMEFRAME_M5)
