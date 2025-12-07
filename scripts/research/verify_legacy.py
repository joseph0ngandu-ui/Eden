#!/usr/bin/env python3
"""
LEGACY STRATEGY AUDIT
Verifying the performance of pre-existing M5 Forex strategies.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
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

class LegacyStrategyEngine:
    def calculate_atr(self, df: pd.DataFrame, period=14):
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean().iloc[-1]

    # --- STRATEGIES ---
    
    def volatility_squeeze(self, df: pd.DataFrame, symbol: str) -> Optional[TradeSignal]:
        if len(df) < 60: return None
        signal = df.iloc[-2]
        current = df.iloc[-1]
        atr = self.calculate_atr(df)
        closes = df['close']
        vol_20 = closes.iloc[-21:-1].pct_change().std()
        vol_50 = closes.iloc[-51:-1].pct_change().std()
        if np.isnan(vol_20) or np.isnan(vol_50): return None
        if vol_20 > vol_50 * 0.6: return None
        high_20 = df['high'].iloc[-22:-2].max()
        low_20 = df['low'].iloc[-22:-2].min()
        if signal['close'] > high_20:
            sl = low_20
            if (current['close'] - sl) > 2 * atr: sl = current['close'] - 2 * atr
            tp = current['close'] + (current['close'] - sl) * 1.5
            return TradeSignal(symbol, "LONG", current['close'], sl, tp, current.name, "VolSqueeze")
        if signal['close'] < low_20:
            sl = high_20
            if (sl - current['close']) > 2 * atr: sl = current['close'] + 2 * atr
            tp = current['close'] - (sl - current['close']) * 1.5
            return TradeSignal(symbol, "SHORT", current['close'], sl, tp, current.name, "VolSqueeze")
        return None

    def quiet_before_storm(self, df: pd.DataFrame, symbol: str) -> Optional[TradeSignal]:
        # Logic from ProStrategyEngine
        # if "GBP" not in symbol and "XAU" not in symbol: return None (We will test ALL to see)
        if len(df) < 50: return None
        signal = df.iloc[-2]
        current = df.iloc[-1]
        atr = self.calculate_atr(df)
        if atr == 0: return None
        tr = df['high'] - df['low']
        avg_vol = tr.iloc[-22:-2].mean()
        recent_vol = tr.iloc[-7:-2].mean()
        if recent_vol > avg_vol * 0.6: return None
        high_10 = df['high'].iloc[-12:-2].max()
        low_10 = df['low'].iloc[-12:-2].min()
        body = abs(signal['close'] - signal['open'])
        avg_body = abs(df['close'] - df['open']).iloc[-22:-2].mean()
        if body < avg_body * 1.2: return None
        if signal['close'] > high_10 and signal['close'] > signal['open']:
            sl = df['low'].iloc[-7:-2].min() - atr * 0.2
            tp = current['close'] + (current['close'] - sl) * 2.5
            return TradeSignal(symbol, "LONG", current['close'], sl, tp, current.name, "QuietStorm")
        if signal['close'] < low_10 and signal['close'] < signal['open']:
            sl = df['high'].iloc[-7:-2].max() + atr * 0.2
            tp = current['close'] - (sl - current['close']) * 2.5
            return TradeSignal(symbol, "SHORT", current['close'], sl, tp, current.name, "QuietStorm")
        return None

    def triple_candle_breakout(self, df: pd.DataFrame, symbol: str) -> Optional[TradeSignal]:
        if len(df) < 50: return None
        breakout = df.iloc[-2]
        inside2 = df.iloc[-3]
        inside1 = df.iloc[-4]
        mother = df.iloc[-5]
        current = df.iloc[-1]
        atr = self.calculate_atr(df)
        mother_range = mother['high'] - mother['low']
        if mother_range < atr * 0.8: return None
        if not (inside1['high'] < mother['high'] and inside1['low'] > mother['low']): return None
        if not (inside2['high'] < inside1['high'] and inside2['low'] > inside1['low']): return None
        if breakout['close'] > mother['high']:
            sl = inside2['low'] - atr * 0.1
            tp = current['close'] + (current['close'] - sl) * 2.5
            return TradeSignal(symbol, "LONG", current['close'], sl, tp, current.name, "TripleCandle")
        if breakout['close'] < mother['low']:
            sl = inside2['high'] + atr * 0.1
            tp = current['close'] - (sl - current['close']) * 2.5
            return TradeSignal(symbol, "SHORT", current['close'], sl, tp, current.name, "TripleCandle")
        return None

    def vwap_reversion_m5(self, df: pd.DataFrame, symbol: str) -> Optional[TradeSignal]:
        # if "EUR" not in symbol and "GBP" not in symbol: return None
        if len(df) < 100: return None
        last_time = df.index[-1]
        # VWAP calc needs today's data
        today = last_time.date()
        today_data = df[df.index.date == today]
        if len(today_data) < 10: return None
        
        tp = (today_data['high'] + today_data['low'] + today_data['close']) / 3
        pv = tp * today_data['tick_volume']
        vwap = (pv.cumsum() / today_data['tick_volume'].cumsum()).iloc[-1]
        
        atr = self.calculate_atr(df)
        band_dist = atr * 3.0
        row = df.iloc[-1]
        
        if row['close'] > vwap + band_dist:
            sl = row['high'] + atr * 0.5
            return TradeSignal(symbol, "SHORT", row['close'], sl, vwap, row.name, "VWAP")
        if row['close'] < vwap - band_dist:
            sl = row['low'] - atr * 0.5
            return TradeSignal(symbol, "LONG", row['close'], sl, vwap, row.name, "VWAP")
        return None

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
            
    close = future.iloc[-1]['close']
    if signal.direction == "LONG": pnl = close - entry
    else: pnl = entry - close
    return pnl / risk if risk > 0 else 0

def run_audit():
    try:
        import MetaTrader5 as mt5
    except ImportError:
        print("MetaTrader5 not available")
        return
        
    if not mt5.initialize(path=r"C:\Program Files\MetaTrader 5 Terminal\terminal64.exe"):
        print("MT5 init failed")
        return
        
    print("="*60)
    print("  LEGACY STRATEGY AUDIT (M5)")
    print("="*60)
    
    symbols = ['EURUSDm', 'GBPUSDm', 'USDJPYm']
    tf = mt5.TIMEFRAME_M5
    end = datetime.now()
    start = end - timedelta(days=60)
    
    engine = LegacyStrategyEngine()
    strategies = [
        engine.volatility_squeeze,
        engine.quiet_before_storm,
        engine.triple_candle_breakout,
        engine.vwap_reversion_m5
    ]
    
    for strategy in strategies:
        strat_name = strategy.__name__
        print(f"\n>>> Testing: {strat_name} <<<")
        
        total_pnl = 0
        total_trades = 0
        
        for symbol in symbols:
            # Skip if strategy has specific filters (checking in loop)
            # engine methods have internal checks but returns None.
            
            rates = mt5.copy_rates_range(symbol, tf, start, end)
            if rates is None or len(rates) < 100:
                continue # No data
                
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            trades = []
            
            # Sub-sample loop for speed (step 1 is fine for audit)
            for i in range(200, len(df) - 100):
                window = df.iloc[i-100:i+1] # 100 bars history
                signal = strategy(window, symbol)
                if signal:
                    future = df.iloc[i+1:i+200]
                    res = check_outcome(signal, future)
                    trades.append(res)
            
            if trades:
                tr_pnl = sum(trades)
                wr = len([t for t in trades if t>0])/len(trades)*100
                print(f"  {symbol}: {len(trades)} trades | WR: {wr:.1f}% | {tr_pnl:.1f}R")
                total_pnl += tr_pnl
                total_trades += len(trades)
            else:
                 print(f"  {symbol}: No trades")
        
        print(f"  [SUMMARY] {strat_name}: {total_trades} trades | TOTAL: {total_pnl:.1f}R")

    mt5.shutdown()

if __name__ == "__main__":
    run_audit()
