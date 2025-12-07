#!/usr/bin/env python3
"""
REGIME DETECTION BACKTEST
Compares strategy performance WITH vs WITHOUT regime filtering.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from trading.regime_detector import RegimeDetector, MarketRegime

@dataclass
class TradeResult:
    r_multiple: float
    regime: Optional[MarketRegime]
    symbol: str
    time: pd.Timestamp

def calculate_atr(df, period=14):
    h, l, c = df['high'], df['low'], df['close'].shift(1)
    tr = pd.concat([h-l, (h-c).abs(), (l-c).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def simulate_index_breakout(df: pd.DataFrame, symbol: str, detector: RegimeDetector) -> List[TradeResult]:
    """Simulate Index Volatility Expansion trades."""
    results = []
    if len(df) < 200: return results
    
    atr_s = calculate_atr(df)
    sma_20 = df['close'].rolling(20).mean()
    std_20 = df['close'].rolling(20).std()
    upper = sma_20 + 2 * std_20
    lower = sma_20 - 2 * std_20
    bandwidth = (upper - lower) / sma_20
    avg_bw = bandwidth.rolling(50).mean()
    ema_50 = df['close'].ewm(span=50).mean()
    
    for i in range(100, len(df) - 50):
        bar = df.iloc[i]
        hour = pd.to_datetime(bar.name).hour
        
        # NY Session only
        if not (13 <= hour <= 20): continue
        
        # Squeeze check
        if bandwidth.iloc[i] >= avg_bw.iloc[i] * 0.8: continue
        
        recent_squeeze = False
        for j in range(1, 4):
            if bandwidth.iloc[i-j-1] < avg_bw.iloc[i-j-1] * 0.8:
                recent_squeeze = True
                break
        if not recent_squeeze: continue
        
        price = bar['close']
        atr = atr_s.iloc[i]
        if pd.isna(atr) or atr == 0: continue
        
        # Breakout detection
        long_breakout = price > upper.iloc[i] and price > ema_50.iloc[i]
        short_breakout = price < lower.iloc[i] and price < ema_50.iloc[i]
        
        if not long_breakout and not short_breakout: continue
        
        # Get regime (using H1 approximation - take every 4th bar of M15)
        regime_window = df.iloc[max(0,i-200):i+1:4]  # Approx H1
        regime = detector.detect(regime_window, symbol) if len(regime_window) > 50 else None
        
        # Simulate trade
        entry = price
        direction = "LONG" if long_breakout else "SHORT"
        sl = sma_20.iloc[i]  # Middle band
        risk = abs(entry - sl)
        tp = entry + risk * 1.5 if direction == "LONG" else entry - risk * 1.5
        
        # Check outcome
        future = df.iloc[i+1:i+50]
        exit_price = None
        for _, row in future.iterrows():
            if direction == "LONG":
                if row['low'] <= sl: exit_price = sl; break
                if row['high'] >= tp: exit_price = tp; break
            else:
                if row['high'] >= sl: exit_price = sl; break
                if row['low'] <= tp: exit_price = tp; break
        
        if exit_price is None:
            exit_price = future.iloc[-1]['close'] if len(future) > 0 else entry
        
        pnl = (exit_price - entry) if direction == "LONG" else (entry - exit_price)
        r_mult = pnl / risk if risk > 0 else 0
        
        results.append(TradeResult(r_mult, regime, symbol, bar.name))
    
    return results

def calculate_metrics(trades: List[TradeResult]) -> Dict:
    if not trades:
        return {"trades": 0, "r": 0, "wr": 0, "dd": 0}
    
    r_vals = [t.r_multiple for t in trades]
    equity = np.cumsum(r_vals)
    peak = np.maximum.accumulate(equity)
    dd = np.max(peak - equity)
    wins = len([r for r in r_vals if r > 0])
    
    return {
        "trades": len(trades),
        "r": sum(r_vals),
        "wr": wins / len(r_vals) * 100,
        "dd": dd
    }

def run_backtest():
    try:
        import MetaTrader5 as mt5
    except ImportError:
        print("MetaTrader5 not available")
        return
    
    if not mt5.initialize(path=r"C:\Program Files\MetaTrader 5 Terminal\terminal64.exe"):
        print("MT5 init failed")
        return
    
    print("="*70)
    print("  REGIME DETECTION BACKTEST")
    print("  Comparing WITH vs WITHOUT regime filtering")
    print("="*70)
    
    detector = RegimeDetector()
    symbols = ['US30m', 'USTECm', 'US500m']
    tf = mt5.TIMEFRAME_M15
    end = datetime.now()
    start = end - timedelta(days=90)
    
    all_trades = []
    
    for symbol in symbols:
        rates = mt5.copy_rates_range(symbol, tf, start, end)
        if rates is None or len(rates) < 500: continue
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        trades = simulate_index_breakout(df, symbol, detector)
        if trades:
            m = calculate_metrics(trades)
            print(f"{symbol}: {m['trades']} trades | {m['r']:.1f}R | WR: {m['wr']:.1f}%")
        all_trades.extend(trades)
    
    if not all_trades:
        print("No trades generated")
        mt5.shutdown()
        return
    
    # Split by regime
    print("\n" + "="*70)
    print("  REGIME COMPARISON")
    print("="*70)
    
    # WITHOUT regime filter (all trades)
    baseline = calculate_metrics(all_trades)
    print(f"\nBASELINE (No Filter): {baseline['trades']} trades | {baseline['r']:.1f}R | WR: {baseline['wr']:.1f}% | MaxDD: {baseline['dd']:.1f}R")
    
    # WITH regime filter (skip unfavorable)
    favorable_trades = []
    for t in all_trades:
        if t.regime is None:
            favorable_trades.append(t)  # No regime data = include
        elif t.regime.is_favorable_for_breakout():
            favorable_trades.append(t)
    
    filtered = calculate_metrics(favorable_trades)
    print(f"FILTERED (Favorable): {filtered['trades']} trades | {filtered['r']:.1f}R | WR: {filtered['wr']:.1f}% | MaxDD: {filtered['dd']:.1f}R")
    
    # WITH regime risk adjustment
    adjusted_r = sum([t.r_multiple * (t.regime.risk_multiplier if t.regime else 1.0) for t in all_trades])
    print(f"RISK ADJUSTED: Same trades | {adjusted_r:.1f}R (effective) | Risk scaled by regime")
    
    # Verdict
    print("\n" + "="*70)
    print("  VERDICT")
    print("="*70)
    
    improvement = filtered['r'] - baseline['r']
    dd_improvement = baseline['dd'] - filtered['dd']
    
    if filtered['r'] > baseline['r'] or filtered['dd'] < baseline['dd']:
        print(f"  ✅ Regime filtering IMPROVES results")
        print(f"     R Change: {improvement:+.1f}R | DD Change: {dd_improvement:+.1f}R")
        print(f"  → RECOMMENDED: Enable regime filtering")
    else:
        print(f"  ❌ Regime filtering does NOT improve results")
        print(f"     R Change: {improvement:+.1f}R | DD Change: {dd_improvement:+.1f}R")
        print(f"  → RECOMMENDED: Keep regime detection for LOGGING ONLY")
    
    mt5.shutdown()

if __name__ == "__main__":
    run_backtest()
