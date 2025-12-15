#!/usr/bin/env python3
"""
PHASE 12: ICT OPTIMIZATION (SMART RISK)
Testing "Aggressive" ICT setups with Break-Even logic.
1. "Judas Swing Aggressive" (Tight Stop at FVG Candle)
2. "Silver Bullet Trend" (H1 Filter)
3. "Auto-BE" (Move to Entry at +1R)
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
logger = logging.getLogger("Phase12")

class Phase12Research:
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

    def outcome_smart(self, future, direction, entry, sl, tp):
        """
        Simulates trade with Break-Even logic.
        If price moves +1R in favor -> SL moves to Entry.
        """
        filled = False
        initial_risk = abs(entry - sl)
        be_triggered = False
        current_sl = sl # Dynamic SL
        
        for idx, row in future.iterrows():
            # 1. Check Fill
            if not filled:
                if direction == "LONG":
                     if row['low'] <= entry: filled = True
                else:
                     if row['high'] >= entry: filled = True
                if not filled: continue 
            
            # 2. Check Outcomes (Once Filled)
            if direction == "LONG":
                # Check Death
                if row['low'] <= current_sl: 
                    if be_triggered: return 0.0 # Stopped at BE
                    return -1.0 # Stopped at Loss
                
                # Check Win
                if row['high'] >= tp: return 3.0 # Target 3R (Aggressive)
                
                # Check BE Trigger (+1R)
                if not be_triggered:
                    if row['high'] >= entry + initial_risk:
                        be_triggered = True
                        current_sl = entry # Move SL to Entry
                        # print("BE Triggered")
                        
            else: # SHORT
                if row['high'] >= current_sl:
                    if be_triggered: return 0.0
                    return -1.0
                
                if row['low'] <= tp: return 3.0
                
                if not be_triggered:
                    if row['low'] <= entry - initial_risk:
                        be_triggered = True
                        current_sl = entry

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
    # STRATEGY 1: AGGRESSIVE JUDAS (Tight Stop + BE)
    # ==========================
    def test_aggressive_judas(self, symbol, timeframe=mt5.TIMEFRAME_M5):
        try:
            df = self.get_data(symbol, timeframe, days=15)
            if df is None: return 0, 0
            
            trades = []
            atr = self.calculate_atr(df)
            days = df.groupby(df.index.date)
            
            for date, day_data in days:
                asian_data = day_data.between_time("00:00", "08:00")
                if len(asian_data) == 0: continue
                asian_high = asian_data['high'].max()
                asian_low = asian_data['low'].min()
                
                session_data = day_data.between_time("08:00", "16:00")
                if len(session_data) < 5: continue
                
                swept_high = False
                swept_low = False
                fvgs = []
                
                for i in range(2, len(session_data)):
                    c1 = session_data.iloc[i-2]
                    c2 = session_data.iloc[i-1]
                    c3 = session_data.iloc[i] 
                    
                    if c3['high'] > asian_high: swept_high = True
                    if c3['low'] < asian_low: swept_low = True
                    
                    # FVG Detection
                    if c1['high'] < c3['low'] and c2['close'] > c2['open']:
                        fvgs.append({'type': 'BULL', 'top': c3['low'], 'bot': c1['high'], 'candle_low': c1['low']})
                    if c1['low'] > c3['high'] and c2['close'] < c2['open']:
                        fvgs.append({'type': 'BEAR', 'top': c1['low'], 'bot': c3['high'], 'candle_high': c1['high']})
                        
                    # Logic
                    if swept_high and not swept_low:
                        for fvg in fvgs:
                             if fvg['type'] == 'BEAR':
                                 if c3['high'] >= fvg['bot'] and c3['low'] <= fvg['top']:
                                     entry = fvg['bot']
                                     # === AGGRESSIVE CHANGE: Stop at FVG Candle High ===
                                     sl = fvg['candle_high'] 
                                     risk = sl - entry
                                     
                                     curr_atr = atr.loc[c3.name]
                                     if risk < curr_atr * 0.2: sl = entry + curr_atr * 0.2 # Min Stop
                                     if risk > curr_atr * 2.0: continue # Skip wide stops
                                     
                                     risk = sl - entry
                                     tp = entry - risk * 3.0 # Target 3R
                                     
                                     trades.append(self.outcome_smart(session_data.iloc[i+1:], "SHORT", entry, sl, tp))
                                     
                                     swept_high = False; swept_low = False; break

                    if swept_low and not swept_high:
                        for fvg in fvgs:
                            if fvg['type'] == 'BULL':
                                if c3['low'] <= fvg['top'] and c3['high'] >= fvg['bot']:
                                    entry = fvg['top']
                                    # === AGGRESSIVE CHANGE: Stop at FVG Candle Low ===
                                    sl = fvg['candle_low']
                                    risk = entry - sl
                                    
                                    curr_atr = atr.loc[c3.name]
                                    if risk < curr_atr * 0.2: sl = entry - curr_atr * 0.2
                                    if risk > curr_atr * 2.0: continue
                                    
                                    risk = entry - sl
                                    tp = entry + risk * 3.0
                                    
                                    trades.append(self.outcome_smart(session_data.iloc[i+1:], "LONG", entry, sl, tp))
                                    swept_low = False; swept_high = False; break
                                    
            return self.simulate_pnl(trades, f"AggJudas_{symbol}")
        except Exception as e:
            logger.error(f"Crash: {e}")
            return 0, 0

    # ==========================
    # STRATEGY 2: TREND SILVER BULLET (H1 Filter)
    # ==========================
    def test_trend_silver_bullet(self, symbol, timeframe=mt5.TIMEFRAME_M5):
        try:
            # Get H1 Data for Trend
            h1_df = self.get_data(symbol, mt5.TIMEFRAME_H1, days=15)
            h1_df['ema50'] = h1_df['close'].ewm(span=50).mean()
            
            df = self.get_data(symbol, timeframe, days=15)
            if df is None: return 0, 0
            
            trades = []
            atr = self.calculate_atr(df)
            
            start_hour = 16 
            end_hour = 17   
            
            for i in range(5, len(df)):
                c1 = df.iloc[i-2]
                c2 = df.iloc[i-1]
                c3 = df.iloc[i]
                timestamp = c3.name
                
                if not (start_hour <= timestamp.hour <= end_hour): continue
                
                # H1 Trend Check
                # Find latest H1 close before current time
                try:
                    h1_trend_val = h1_df.loc[h1_df.index <= timestamp].iloc[-1]['ema50']
                    h1_close = h1_df.loc[h1_df.index <= timestamp].iloc[-1]['close']
                except: continue
                
                h1_bullish = h1_close > h1_trend_val
                
                # FVG Logic
                curr_atr = atr.iloc[i]
                
                if c1['high'] < c3['low'] and c2['close'] > c2['open']: # Bull FVG
                    if h1_bullish: # TREND ALIGNED ONLY
                        entry = c3['low']
                        sl = c1['low'] # Aggressive Stop
                        risk = entry - sl
                        if risk < curr_atr * 0.2 or risk > curr_atr * 2.5: continue
                        
                        trades.append(self.outcome_smart(df.iloc[i+1:], "LONG", entry, sl, entry + risk*3))

                if c1['low'] > c3['high'] and c2['close'] < c2['open']: # Bear FVG
                    if not h1_bullish: # TREND ALIGNED ONLY
                        entry = c3['high']
                        sl = c1['high']
                        risk = sl - entry
                        if risk < curr_atr * 0.2 or risk > curr_atr * 2.5: continue
                        
                        trades.append(self.outcome_smart(df.iloc[i+1:], "SHORT", entry, sl, entry - risk*3))

            return self.simulate_pnl(trades, f"TrendBullet_{symbol}")

        except Exception as e:
            logger.error(f"Crash SB: {e}")
            return 0, 0

if __name__ == "__main__":
    r = Phase12Research()
    print("=== PHASE 12: SMART RISK ICT (BE + TIGHT STOP) ===")
    
    # 1. Gold
    r.test_aggressive_judas("XAUUSDm", mt5.TIMEFRAME_M5)
    
    # 2. Indices
    r.test_aggressive_judas("USTECm", mt5.TIMEFRAME_M5)
    r.test_trend_silver_bullet("USTECm", mt5.TIMEFRAME_M5)
    
    # 3. Forex
    r.test_trend_silver_bullet("EURUSDm", mt5.TIMEFRAME_M5)
