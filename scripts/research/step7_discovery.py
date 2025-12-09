#!/usr/bin/env python3
"""
PHASE 7 DISCOVERY HARNESS (DEBUG MODE)
Testing 3 New Alpha Sources:
1. London Breakout (Trend) - GBPJPY, GBPUSD
2. Asian Fade (Range) - EURUSD, USDJPY
3. ICT FVG (Smc) - XAUUSD, USTEC, EURUSD
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
logger = logging.getLogger("Phase7")

class Phase7Discovery:
    def __init__(self):
        if not mt5.initialize():
            print("MT5 Init Failed")
            sys.exit(1)
            
    def get_data(self, symbol, timeframe, days=90):
        try:
            utc_from = datetime.now() - timedelta(days=days)
            rates = mt5.copy_rates_from(symbol, timeframe, datetime.now(), 15000) 
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
                if row['high'] >= tp: return 2.0 # Fixed 1:2 for simulation
            else:
                if row['high'] >= sl: return -1.0
                if row['low'] <= tp: return 2.0
        return 0.0

    def simulate_pnl(self, trades, name):
        # Flatten list if needed
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
            
        result_str = f"Strategy: {name:<20} | Trades: {count:<4} | Total R: {total_r:>6.2f} | Max DD: {max_dd:>5.2f}R"
        print(result_str)
        with open("discovery_results.txt", "a") as f:
            f.write(result_str + "\n")
        return total_r, max_dd

    # ==========================
    # STRATEGY 1: London Breakout
    # ==========================
    def test_london_breakout(self, symbol):
        try:
            df = self.get_data(symbol, mt5.TIMEFRAME_M15, days=90)
            if df is None: return 0, 0
            
            trades = []
            atr = self.calculate_atr(df)
            
            # Debug counters
            ignored_time = 0
            ignored_range = 0
            ignored_vol = 0
            entries = 0
            
            for i in range(50, len(df)):
                curr = df.iloc[i]
                timestamp = curr.name
                
                # 1. Time Filter (08:00 - 09:00)
                if not (8 <= timestamp.hour <= 10): 
                    ignored_time += 1
                    continue
                
                today_start = timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
                asian_session = df[(df.index >= today_start) & (df.index < timestamp)]
                if len(asian_session) < 10: 
                    ignored_range += 1
                    continue
                
                range_high = asian_session['high'].max()
                range_low = asian_session['low'].min()
                range_height = range_high - range_low
                
                curr_atr = atr.iloc[i]
                
                # Volatility Filter
                if range_height > curr_atr * 2.0: 
                    ignored_vol += 1
                    continue 
                
                entry_price = curr['close']
                
                if entry_price > range_high:
                    sl = range_low
                    if (entry_price - sl) > curr_atr * 3: continue 
                    tp = entry_price + (entry_price - sl) * 2.0
                    trades.append(self.outcome(df.iloc[i:], "LONG", entry_price, sl, tp))
                    entries += 1
                    
                elif entry_price < range_low:
                    sl = range_high
                    if (sl - entry_price) > curr_atr * 3: continue
                    tp = entry_price - (sl - entry_price) * 2.0
                    trades.append(self.outcome(df.iloc[i:], "SHORT", entry_price, sl, tp))
                    entries += 1
            
            # print(f"DEBUG {symbol}: TimeSkip={ignored_time} RangeSkip={ignored_range} VolSkip={ignored_vol} Entries={entries}")
            return self.simulate_pnl(trades, f"LondonBO_{symbol}")
        except Exception as e:
            print(f"CRASH in LondonBO {symbol}: {e}")
            traceback.print_exc()
            return 0, 0

    # ==========================
    # STRATEGY 2: Asian Fade
    # ==========================
    def test_asian_fade(self, symbol):
        try:
            df = self.get_data(symbol, mt5.TIMEFRAME_M5, days=90)
            if df is None: return 0, 0
            
            trades = []
            atr = self.calculate_atr(df)
            
            for i in range(50, len(df)):
                curr = df.iloc[i]
                timestamp = curr.name
                
                # 1. Time: 01:00 - 07:00
                if not (1 <= timestamp.hour <= 7): continue
                
                # 2. Bollinger Logic
                sma = df['close'].iloc[i-20:i].mean()
                std = df['close'].iloc[i-20:i].std()
                upper = sma + 2*std
                lower = sma - 2*std
                
                curr_price = curr['close']
                curr_atr = atr.iloc[i]
                if pd.isna(curr_atr): continue
                
                if curr_price > upper:
                    sl = curr_price + curr_atr * 1.5
                    tp = sma
                    trades.append(self.outcome(df.iloc[i:], "SHORT", curr_price, sl, tp))
                    
                elif curr_price < lower:
                    sl = curr_price - curr_atr * 1.5
                    tp = sma
                    trades.append(self.outcome(df.iloc[i:], "LONG", curr_price, sl, tp))
                    
            return self.simulate_pnl(trades, f"AsianFade_{symbol}")
        except Exception as e:
            print(f"CRASH in AsianFade {symbol}: {e}")
            traceback.print_exc()
            return 0, 0

    # ==========================
    # STRATEGY 3: ICT FVG
    # ==========================
    def test_ict_fvg(self, symbol):
        try:
            df = self.get_data(symbol, mt5.TIMEFRAME_M15, days=90)
            if df is None: return 0, 0
            
            trades = []
            fvgs = [] 
            
            atr = self.calculate_atr(df)
            
            for i in range(5, len(df)):
                c1 = df.iloc[i-2]
                c2 = df.iloc[i-1] 
                c3 = df.iloc[i]   
                timestamp = c3.name
                
                # Session Filter ONLY (Relaxed)
                if not (6 <= timestamp.hour <= 22): continue
                
                # Bullish FVG
                if c1['high'] < c3['low'] and c2['close'] > c2['open']:
                     fvgs.append({'type': 'BULL', 'top': c3['low'], 'bot': c1['high'], 'time': i})
                
                # Bearish FVG
                if c1['low'] > c3['high'] and c2['close'] < c2['open']:
                     fvgs.append({'type': 'BEAR', 'top': c1['low'], 'bot': c3['high'], 'time': i})
                
                curr_price = c3['close']
                curr_atr = atr.iloc[i]
                if pd.isna(curr_atr): continue
                
                for fvg in fvgs[:]:
                    if i - fvg['time'] > 12: 
                        fvgs.remove(fvg)
                        continue
                    
                    if fvg['type'] == 'BULL':
                        if c3['low'] <= fvg['top'] and c3['high'] >= fvg['bot']:
                            sl = fvg['bot']
                            risk = curr_price - sl
                            if risk < curr_atr * 0.1 or risk > curr_atr * 3.0: continue
                            
                            tp = curr_price + risk * 2.5
                            trades.append(self.outcome(df.iloc[i:], "LONG", curr_price, sl, tp))
                            fvgs.remove(fvg)
                            
                    elif fvg['type'] == 'BEAR':
                        if c3['high'] >= fvg['bot'] and c3['low'] <= fvg['top']:
                            sl = fvg['top']
                            risk = sl - curr_price
                            if risk < curr_atr * 0.1 or risk > curr_atr * 3.0: continue
                            
                            tp = curr_price - risk * 2.5
                            trades.append(self.outcome(df.iloc[i:], "SHORT", curr_price, sl, tp))
                            fvgs.remove(fvg)
    
            return self.simulate_pnl(trades, f"ICT_FVG_{symbol}")
        except Exception as e:
            print(f"CRASH in ICT_FVG {symbol}: {e}")
            traceback.print_exc()
            return 0, 0

if __name__ == "__main__":
    d = Phase7Discovery()
    print("=== PHASE 7 DISCOVERY ===")
    
    # 1. London Breakout
    d.test_london_breakout("GBPJPYm")
    d.test_london_breakout("GBPUSDm")
    
    # 2. Asian Fade
    d.test_asian_fade("EURUSDm")
    d.test_asian_fade("USDJPYm")
    
    # 3. ICT FVG
    d.test_ict_fvg("XAUUSDm")
    d.test_ict_fvg("USTECm")
    d.test_ict_fvg("EURUSDm")
