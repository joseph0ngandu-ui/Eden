#!/usr/bin/env python3
"""
MOMENTUM CONTINUATION - EXPANDED RESEARCH
Low DD strategy (2.6R) - Testing on ALL pairs to maximize R.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

@dataclass
class TradeResult:
    r_multiple: float
    symbol: str

def calculate_metrics(trades: List[TradeResult]) -> Dict:
    if not trades:
        return {"total_trades": 0, "total_r": 0, "win_rate": 0, "max_dd": 0, "max_consec": 0}
    
    r_values = [t.r_multiple for t in trades]
    equity = np.cumsum(r_values)
    peak = np.maximum.accumulate(equity)
    drawdown = peak - equity
    max_dd = np.max(drawdown)
    
    max_consec = 0
    streak = 0
    for r in r_values:
        if r < 0:
            streak += 1
            max_consec = max(max_consec, streak)
        else:
            streak = 0
    
    wins = len([r for r in r_values if r > 0])
    
    return {
        "total_trades": len(trades),
        "total_r": sum(r_values),
        "win_rate": wins / len(r_values) * 100 if r_values else 0,
        "max_dd": max_dd,
        "max_consec": max_consec
    }

def test_momentum_continuation(df_d1: pd.DataFrame, symbol: str) -> List[TradeResult]:
    """Enter pullbacks after strong D1 candles."""
    results = []
    if len(df_d1) < 30: return results
    
    # Calculate ADR
    daily_ranges = df_d1['high'] - df_d1['low']
    adr = daily_ranges.rolling(14).mean()
    
    for i in range(20, len(df_d1) - 2):
        day = df_d1.iloc[i]
        day_range = day['high'] - day['low']
        
        if pd.isna(adr.iloc[i]) or adr.iloc[i] == 0: continue
        
        # Strong day: Range > 1.3x ADR
        if day_range < 1.3 * adr.iloc[i]: continue
        
        # Direction
        bullish = day['close'] > day['open']
        
        # Next day entry
        next_day = df_d1.iloc[i+1]
        entry = next_day['open']
        
        if bullish:
            sl = day['low']
            risk = entry - sl
            if risk <= 0: continue
            tp = entry + risk * 1.5
            direction = "LONG"
        else:
            sl = day['high']
            risk = sl - entry
            if risk <= 0: continue
            tp = entry - risk * 1.5
            direction = "SHORT"
        
        # Check outcome
        if direction == "LONG":
            if next_day['low'] <= sl:
                r_mult = -1.0
            elif next_day['high'] >= tp:
                r_mult = 1.5
            else:
                pnl = next_day['close'] - entry
                r_mult = pnl / risk
        else:
            if next_day['high'] >= sl:
                r_mult = -1.0
            elif next_day['low'] <= tp:
                r_mult = 1.5
            else:
                pnl = entry - next_day['close']
                r_mult = pnl / risk
        
        results.append(TradeResult(r_mult, symbol))
    
    return results

def run_research():
    try:
        import MetaTrader5 as mt5
    except ImportError:
        print("MetaTrader5 not available")
        return
    
    if not mt5.initialize(path=r"C:\Program Files\MetaTrader 5 Terminal\terminal64.exe"):
        print("MT5 init failed")
        return
    
    print("="*70)
    print("  MOMENTUM CONTINUATION - EXPANDED RESEARCH")
    print("  (ALL Pairs, MaxDD < 5R Requirement)")
    print("="*70)
    
    # ALL potential pairs
    symbols = [
        'EURUSDm', 'GBPUSDm', 'USDJPYm', 'AUDUSDm', 'NZDUSDm', 'USDCADm', 'USDCHFm',
        'EURJPYm', 'GBPJPYm', 'AUDJPYm', 'CADJPYm', 'CHFJPYm',
        'EURGBPm', 'EURAUDm', 'GBPAUDm', 'GBPNZDm', 'GBPCADm',
        'AUDNZDm', 'AUDCADm'
    ]
    
    end = datetime.now()
    start = end - timedelta(days=120)  # 4 months for more data
    
    all_trades = []
    pair_results = {}
    
    for symbol in symbols:
        rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_D1, start, end)
        if rates is None or len(rates) < 30:
            continue
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        trades = test_momentum_continuation(df, symbol)
        if trades:
            metrics = calculate_metrics(trades)
            pair_results[symbol] = metrics
            all_trades.extend(trades)
            print(f"{symbol}: {metrics['total_trades']} trades | WR: {metrics['win_rate']:.1f}% | R: {metrics['total_r']:.1f}R | MaxDD: {metrics['max_dd']:.1f}R")
    
    # Combined Results
    if all_trades:
        combined = calculate_metrics(all_trades)
        print(f"\n{'='*70}")
        print(f"  COMBINED PORTFOLIO")
        print(f"{'='*70}")
        print(f"  Total Trades: {combined['total_trades']}")
        print(f"  Total R: {combined['total_r']:.1f}R")
        print(f"  Win Rate: {combined['win_rate']:.1f}%")
        print(f"  Max Drawdown: {combined['max_dd']:.1f}R")
        print(f"  Max Consec Losses: {combined['max_consec']}")
        
        # Verdict
        print(f"\n{'='*70}")
        print(f"  VERDICT")
        print(f"{'='*70}")
        
        passed = True
        if combined['total_r'] < 15:
            print(f"  ❌ Total R: {combined['total_r']:.1f}R < 15R")
            passed = False
        else:
            print(f"  ✅ Total R: {combined['total_r']:.1f}R >= 15R")
        
        if combined['max_dd'] > 5:
            print(f"  ❌ Max DD: {combined['max_dd']:.1f}R > 5R")
            passed = False
        else:
            print(f"  ✅ Max DD: {combined['max_dd']:.1f}R <= 5R")
        
        print(f"\n  FINAL: {'✅ APPROVED' if passed else '❌ REJECTED'}")
        
        # Show profitable pairs
        if passed:
            print(f"\n  PROFITABLE PAIRS:")
            for sym, m in sorted(pair_results.items(), key=lambda x: x[1]['total_r'], reverse=True):
                if m['total_r'] > 0:
                    print(f"    {sym}: +{m['total_r']:.1f}R")
    
    mt5.shutdown()

if __name__ == "__main__":
    run_research()
