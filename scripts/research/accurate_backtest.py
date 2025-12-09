#!/usr/bin/env python3
"""
ACCURATE PORTFOLIO BACKTEST
Directly imports and uses ProStrategyEngine to ensure 100% logic match with live bot.
"""

import sys
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import ACTUAL Bot Logic
from trading.pro_strategies import ProStrategyEngine
from trading.models import Trade
from trading.regime_detector import RegimeDetector

@dataclass
class BacktestResult:
    trades: List[Dict]
    equity_curve: pd.Series
    metrics: Dict

class AccurateBacktester:
    def __init__(self):
        self.engine = ProStrategyEngine()
        self.regime_detector = RegimeDetector()
        self.strategies_config = [
            # INDEX M15 (High Alloc 1.4x)
            {"symbol": "USTECm", "tf": mt5.TIMEFRAME_M15, "type": "index"},
            {"symbol": "US500m", "tf": mt5.TIMEFRAME_M15, "type": "index"},
            
            # GOLD SMART SWEEP (High Alloc)
            {"symbol": "XAUUSDm", "tf": mt5.TIMEFRAME_M15, "type": "gold_sweep"},
            
            # FOREX M5 (Low Alloc 0.8x)
            {"symbol": "EURUSDm", "tf": mt5.TIMEFRAME_M5, "type": "forex"},
            {"symbol": "USDJPYm", "tf": mt5.TIMEFRAME_M5, "type": "forex"},
            
            # ASIAN FADE M5 (New - High Alpha)
            {"symbol": "EURUSDm", "tf": mt5.TIMEFRAME_M5, "type": "asian_fade"},
            # REMOVED USDJPYm (Performance Drag)
            
            # MOMENTUM D1 (High Alloc 1.4x)
            {"symbol": "USDCADm", "tf": mt5.TIMEFRAME_D1, "type": "momentum"},
            {"symbol": "EURUSDm", "tf": mt5.TIMEFRAME_D1, "type": "momentum"},
            {"symbol": "EURJPYm", "tf": mt5.TIMEFRAME_D1, "type": "momentum"},
            {"symbol": "CADJPYm", "tf": mt5.TIMEFRAME_D1, "type": "momentum"},
        ]

    def get_risk_multiplier(self, strategy_type: str) -> float:
        # UNIFORM ALLOCATION (Find optimal base risk)
        return 1.0

    def simulate_trade(self, trade: Trade, future_data: pd.DataFrame) -> Dict:
        entry_price = trade.entry_price
        sl = trade.sl
        tp = trade.tp
        direction = trade.direction
        
        exit_price = entry_price
        exit_time = future_data.index[-1]
        exit_reason = "end_of_data"
        
        for t, row in future_data.iterrows():
            if direction == "LONG":
                if row['low'] <= sl: 
                    exit_price = sl
                    exit_time = t
                    exit_reason = "sl"
                    break
                if row['high'] >= tp:
                    exit_price = tp
                    exit_time = t
                    exit_reason = "tp"
                    break
            else: # SHORT
                if row['high'] >= sl:
                    exit_price = sl
                    exit_time = t
                    exit_reason = "sl"
                    break
                if row['low'] <= tp:
                    exit_price = tp
                    exit_time = t
                    exit_reason = "tp"
                    break
        
        risk_per_share = abs(entry_price - sl)
        if direction == "LONG":
            pnl_amt = exit_price - entry_price
        else:
            pnl_amt = entry_price - exit_price
            
        r_multiple = pnl_amt / risk_per_share if risk_per_share > 0 else 0
        r_multiple -= 0.05  # Commission approximation
        
        return {
            "time": trade.entry_time,
            "symbol": trade.symbol,
            "direction": trade.direction,
            "entry": entry_price,
            "exit": exit_price,
            "exit_time": exit_time,
            "r": r_multiple,
            "reason": exit_reason,
            "strategy": trade.strategy
        }

    def run(self, days=90):
        if not mt5.initialize():
            print("MT5 Init Failed")
            return []

        all_trades = []
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        print(f"Running Accurate Backtest ({days} days)...")
        
        for config in self.strategies_config:
            symbol = config['symbol']
            tf = config['tf']
            stype = config['type']
            
            print(f"Processing {symbol} ({stype})...")
            
            rates = mt5.copy_rates_range(symbol, tf, start_date, end_date)
            
            min_bars = 200 if tf != mt5.TIMEFRAME_D1 else 20
            
            if rates is None or len(rates) < min_bars:
                print(f"  No data for {symbol}")
                continue
                
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            LOOKBACK = 20 if stype == "momentum" else 100
            
            for i in range(LOOKBACK, len(df)-1):
                window = df.iloc[i-LOOKBACK:i+1]
                
                try:
                    signal = None
                    if stype == "index":
                        signal = self.engine.index_volatility_expansion(window, symbol)
                    elif stype == "gold_sweep":
                        signal = self.engine.gold_smart_sweep(window, symbol)
                    elif stype == "forex":
                        signal = self.engine.volatility_squeeze(window, symbol)
                    elif stype == "asian_fade":
                        signal = self.engine.asian_fade_range(window, symbol)
                    elif stype == "momentum":
                        signal = self.engine.momentum_continuation(window, symbol)
                    
                    if signal:
                        future = df.iloc[i:i+101] if stype == "momentum" else df.iloc[i+1:i+101]
                        if len(future) > 0:
                            result = self.simulate_trade(signal, future)
                            risk_mult = self.get_risk_multiplier(stype)
                            result['r_weighted'] = result['r'] * risk_mult
                            result['stype'] = stype
                            all_trades.append(result)
                except Exception as e:
                    pass

        mt5.shutdown()
        return all_trades

    def analyze(self, trades):
        if not trades:
            print("No trades generated.")
            return

        df = pd.DataFrame(trades)
        df.sort_values('time', inplace=True)
        
        print("\n" + "="*70)
        print("WEIGHTED PORTFOLIO BACKTEST (90 Days)")
        print("Index/Momentum: 1.4x | Forex: 0.8x")
        print("="*70)
        
        # Use uniform 1.0x multipliers, vary base risk
        risk_levels = [0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
        
        print(f"{'Base Risk':<10} | {'Return':<10} | {'Max DD':<10} | {'Monthly':<10} | {'Verdict'}")
        print("-" * 70)
        
        for base_risk in risk_levels:
            current_pnl = []
            for _, t in df.iterrows():
                # Use the r_weighted which already has the multiplier applied
                pnl = t['r_weighted'] * base_risk
                current_pnl.append(pnl)
            
            equity = 100 + np.cumsum(current_pnl)
            peak = np.maximum.accumulate(equity)
            dd = (peak - equity)
            max_dd = np.max(dd)
            total_ret = equity[-1] - 100
            monthly_ret = total_ret / 3
            
            verdict = "✅ PASS" if max_dd < 9.5 and monthly_ret > 8 else "⚠️ SLOW" if max_dd < 9.5 else "❌ FAIL"
            
            print(f"{base_risk:<9}% | {total_ret:>8.2f}% | {max_dd:>8.2f}% | {monthly_ret:>8.2f}% | {verdict}")
    
        print("\nBreakdown by Symbol:")
        summary = df.groupby('symbol').agg({
            'r': ['count', 'sum'],
            'r_weighted': 'sum'
        })
        print(summary)
            
        return df

if __name__ == "__main__":
    bt = AccurateBacktester()
    trades = bt.run(days=90)
    bt.analyze(trades)
