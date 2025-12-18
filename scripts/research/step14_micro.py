#!/usr/bin/env python3
"""
PHASE 14: MICRO-SCALPING RESEARCH (M1 FLOW)
Strategy: "Flow Scalper"
Logic: 
1. M5 Trend Confirmation (ADX > 25, Price > EMA20).
2. M1 Pullback to significant EMA (EMA50).
3. Limit Check: Entry at EMA Price (simulate liquidity making).
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Phase14")

class Phase14Research:
    def __init__(self):
        if not mt5.initialize():
            print("MT5 Init Failed")
            sys.exit(1)
            
    def get_data(self, symbol, timeframe, days=15):
        try:
            utc_from = datetime.now() - timedelta(days=days)
            rates = mt5.copy_rates_from(symbol, timeframe, datetime.now(), 50000) 
            if rates is None: 
                print(f"No data for {symbol} {timeframe}")
                return None
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            df = df[df.index > utc_from]
            return df
        except Exception as e:
            print(f"Error getting data for {symbol}: {e}")
            return None

    def calculate_indicators_m5(self, df):
        df['ema20'] = df['close'].ewm(span=20).mean()
        
        # ADX (Simplified)
        high = df['high']
        low = df['low']
        close = df['close']
        tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
        up = high - high.shift()
        down = low.shift() - low
        pos_dm = np.where((up > down) & (up > 0), up, 0)
        neg_dm = np.where((down > up) & (down > 0), down, 0)
        tr14 = tr.rolling(14).sum()
        pdm14 = pd.Series(pos_dm, index=df.index).rolling(14).sum()
        ndm14 = pd.Series(neg_dm, index=df.index).rolling(14).sum()
        pdi = 100 * (pdm14 / tr14)
        ndi = 100 * (ndm14 / tr14)
        dx = 100 * (abs(pdi - ndi) / (pdi + ndi))
        df['adx'] = dx.rolling(14).mean()
        return df

    def calculate_indicators_m1(self, df):
        df['ema50'] = df['close'].ewm(span=50).mean()
        df['atr'] = (df['high'] - df['low']).rolling(14).mean()
        return df

    def outcome(self, future, direction, entry, sl, tp, spread):
        # M1 Scalping is SENSITIVE to spread.
        # Long Entry: Bid + Spread (Ask).
        # Short Entry: Bid.
        # But we are using LIMIT orders at EMA. 
        # So Entry Price is fixed. Fill condition is the variable.
        
        filled = False
        
        for idx, row in future.iterrows():
            # Check Fill (Limit Order)
            if not filled:
                if direction == "LONG":
                    # Buy Limit at Entry. Filled if Low (Bid) <= Entry.
                    # Actually, if creating liquidity, filled if Ask hits Entry? No.
                    # Buying Limit: Filled if Ask price drops to Limit? No, typically Limit is below market.
                    # Market Price (Ask) must touch Limit.
                    # Chart Low is Bid. Ask Low is Bid Low + Spread.
                    ask_low = row['low'] + spread
                    if ask_low <= entry: filled = True
                else: # SHORT
                    # Sell Limit at Entry. Filled if Bid >= Entry.
                    # Chart High is Bid.
                    if row['high'] >= entry: filled = True
                
                if not filled: continue

            # Check Outcomes (Once Filled)
            if direction == "LONG":
                # Exit SL: Bid <= SL
                if row['low'] <= sl: return -1.0
                # Exit TP: Bid >= TP
                if row['high'] >= tp: return 1.5
            else: # SHORT
                # Exit SL: Ask >= SL -> Bid + Spread >= SL -> Bid >= SL - Spread
                ask_high = row['high'] + spread
                if ask_high >= sl: return -1.0
                # Exit TP: Ask <= TP -> Bid + Spread <= TP -> Bid <= TP - Spread
                ask_low = row['low'] + spread
                if ask_low <= tp: return 1.5
                
        return 0.0

    def test_flow_scalper(self, symbol):
        # Get Data
        m5 = self.get_data(symbol, mt5.TIMEFRAME_M5, days=8) # 8 Days (cover bad week)
        m1 = self.get_data(symbol, mt5.TIMEFRAME_M1, days=8)
        
        if m5 is None or m1 is None: return 0, 0
        
        # Indicators
        m5 = self.calculate_indicators_m5(m5)
        m1 = self.calculate_indicators_m1(m1)
        
        # Spread Estimator
        spread = 0.00012
        if "JPY" in symbol: spread = 0.012
        if "XAU" in symbol: spread = 0.25
        
        trades = []
        
        # Iterate M1 bars
        # For each M1 bar, check active M5 bar state
        
        for i in range(100, len(m1)-1): # Look at completed m1 bar
            current_m1 = m1.iloc[i]
            timestamp = current_m1.name
            
            # Find corresponding M5 bar (latest closed before or valid context)
            # To avoid lookahead, we must use M5 bar that started BEFORE current_m1
            # Or just resample logic.
            # Simplify: Get M5 state as of timestamp
            try:
                m5_idx = m5.index.get_indexer([timestamp], method='ffill')[0]
                state_m5 = m5.iloc[m5_idx]
            except: continue
            
            # M5 FILTER: Strong Trend
            # Long: ADX > 25, Close > EMA20, EMA20 Slope Up?
            if state_m5['adx'] < 25: continue
            
            m5_trend_up = state_m5['close'] > state_m5['ema20']
            m5_trend_down = state_m5['close'] < state_m5['ema20']
            
            # M1 TRIGGER: Pullback to EMA50
            # We place limit at EMA50 if price is close?
            # Or detect touch?
            # Strategy: "Touch Limit".
            # Check if current M1 Low touched EMA50 (Long)
            
            ema_val = current_m1['ema50']
            atr = current_m1['atr']
            
            if m5_trend_up:
                # Pullback Long
                # If Low <= EMA <= High (Touched)
                if current_m1['low'] <= ema_val and current_m1['close'] > ema_val: # Rejection wick preferrable
                    entry = ema_val
                    sl = entry - atr * 2.0 # Tight Stop (2 ATR M1)
                    tp = entry + (entry - sl) * 1.5
                    
                    trades.append(self.outcome(m1.iloc[i+1:], "LONG", entry, sl, tp, spread))
                    
            elif m5_trend_down:
                # Pullback Short
                if current_m1['high'] >= ema_val and current_m1['close'] < ema_val:
                    entry = ema_val
                    sl = entry + atr * 2.0
                    tp = entry - (sl - entry) * 1.5
                    
                    trades.append(self.outcome(m1.iloc[i+1:], "SHORT", entry, sl, tp, spread))

        # Result PnL
        clean = [t for t in trades if t != 0]
        total = sum(clean)
        count = len(clean)
        print(f"Strategy: Flow_{symbol:<10} | Trades: {count:<4} | Total R: {total:>6.2f}")
        return total

if __name__ == "__main__":
    r = Phase14Research()
    print("=== PHASE 14: MICRO-SCALPING (M1 FLOW) ===")
    r.test_flow_scalper("EURUSDm")
    r.test_flow_scalper("USDJPYm")
    r.test_flow_scalper("XAUUSDm")
