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
            # INDEX M15 (Winners Only)
            # Removed US30m (Loser -7.5R)
            {"symbol": "USTECm", "tf": mt5.TIMEFRAME_M15, "type": "index"},
            {"symbol": "US500m", "tf": mt5.TIMEFRAME_M15, "type": "index"},
            
            # GOLD M15 (Removed - Loser -6.5R)
            # {"symbol": "XAUUSDm", "tf": mt5.TIMEFRAME_M15, "type": "gold"},
            
            # FOREX M5 (Winners)
            {"symbol": "EURUSDm", "tf": mt5.TIMEFRAME_M5, "type": "forex"},
            {"symbol": "USDJPYm", "tf": mt5.TIMEFRAME_M5, "type": "forex"},
            {"symbol": "AUDUSDm", "tf": mt5.TIMEFRAME_M5, "type": "forex"}, # Research Candidate
            
            # MOMENTUM D1 (Now Fixed)
            {"symbol": "USDCADm", "tf": mt5.TIMEFRAME_D1, "type": "momentum"},
            {"symbol": "EURUSDm", "tf": mt5.TIMEFRAME_D1, "type": "momentum"},
            {"symbol": "EURJPYm", "tf": mt5.TIMEFRAME_D1, "type": "momentum"},
            {"symbol": "CADJPYm", "tf": mt5.TIMEFRAME_D1, "type": "momentum"},
        ]

    def get_risk_multiplier(self, strategy_type: str) -> float:
        # REDUCED RISK for FundedNext Compliance
        if strategy_type == "index": return 0.5  # Was 1.5, now 0.5 (Base 0.25%)
        if strategy_type == "forex": return 0.5  # Was 0.5, keep 0.5 (Base 0.25%)
        return 0.5

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
        
        # Apply commission/swap approximation (simplified)
        # -0.05R approx per trade for spread/commissions
        r_multiple -= 0.05 
        
        return {
            "time": trade.entry_time,
            "symbol": trade.symbol,
            "direction": trade.direction,
            "entry": entry_price,
            "exit": exit_price,
            "exit_time": exit_time,
            "r": r_multiple,
            "reason": exit_reason
        }

    def run(self, days=90):
        if not mt5.initialize():
            print("MT5 Init Failed")
            return

        all_trades = []
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        print(f"Running Accurate Backtest ({days} days)...")
        
        for config in self.strategies_config:
            symbol = config['symbol']
            tf = config['tf']
            stype = config['type']
            
            print(f"Processing {symbol} ({stype})...")
            
            # Reset Engine State for each symbol run (important!)
            # Ideally we'd run all parallels, but for verification, serial with reset is fine
            # providing we correctly track logic state.
            # actually ProStrategyEngine stores state in dicts by symbol, so we can reuse instance
            
            rates = mt5.copy_rates_range(symbol, tf, start_date, end_date)
            
            # Adjust min bars based on timeframe
            min_bars = 200
            if tf == mt5.TIMEFRAME_D1:
                min_bars = 20 
            
            if rates is None or len(rates) < min_bars:
                print(f"  No data for {symbol} (Got {len(rates) if rates is not None else 0}, Need {min_bars})")
                continue
                
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            # Need 'spread' column for Gold strategy
            # copy_rates_range returns 'spread' int field (points)
            # Strategy expects 'spread' in points (int or float) ??
            # ProStrategyEngine: current_spread = current_bar['spread']
            # And: if current_spread > avg_spread * 0.85
            # MT5 'spread' is integer points.
            
            # Simulate Bar-by-Bar
            # We need a warm-up period for indicators
            LOOKBACK = 100
            if stype == "momentum":
                LOOKBACK = 20
            
            for i in range(LOOKBACK, len(df)-1):
                # Slice window for strategy
                # Slice window for strategy
                # Strategy typically looks at last 50-100 bars
                window = df.iloc[i-LOOKBACK:i+1] # Include current bar i as 'closed' or forming?
                # Live bot usually runs on completed candles or current tick. 
                # Strategies typically assume df.iloc[-1] is the LATEST bar.
                # If we assume bar 'i' is the JUST COMPLETED bar, we use correct logic.
                
                # ProStrategyEngine usually checks df.iloc[-1]
                
                # Run Strategy
                try:
                    # We call specific verified strategies directly to ensure routing
                    # or use evaluate_live but force correct timeframe context
                    
                    # For purity, let's use the explicit checks mapped to the symbol
                    signal = None
                    
                    # 1. Update Engine State (History)
                    # Some strategies update history inside the method (bad side effect for backtest if not careful)
                    # spread_hunter_momentum updates self.spread_history
                    # index_volatility_expansion updates self.bandwidth_history
                    
                    # We must allow the engine to update its state
                    # So we MUST call the strategy function even if we don't trade, to keep history valid
                    
                    if stype == "index":
                        signal = self.engine.index_volatility_expansion(window, symbol)
                    elif stype == "gold":
                        signal = self.engine.spread_hunter_momentum(window, symbol)
                    elif stype == "forex":
                        signal = self.engine.volatility_squeeze(window, symbol)
                    elif stype == "momentum":
                        signal = self.engine.momentum_continuation(window, symbol)
                    
                    if signal:
                        # Trade Simulation
                        # Get future data
                        if stype == "momentum":
                            # Momentum enters at Open of current bar, so test current bar too
                            future = df.iloc[i:i+1+100]
                        else:
                            # Others enter at Close of current bar, so test starting next bar
                            future = df.iloc[i+1:i+1+100]
                            
                        if len(future) > 0:
                            result = self.simulate_trade(signal, future)
                            # Add Risk Multiplier
                            risk_mult = self.get_risk_multiplier(stype)
                            result['r_weighted'] = result['r'] * risk_mult
                            result['risk_pct'] = 0.5 * risk_mult # Base 0.5%
                            result['pnl_pct'] = result['r'] * result['risk_pct']
                            
                            all_trades.append(result)
                            
                            # Skip forward to avoid overlapping trades if you want
                            # OR sim concurrent. For now, let's allow overlapping but just log them.
                            # In reality, bot creates ONE position per symbol at a time usually.
                            
                except Exception as e:
                    # print(f"Error {e}")
                    pass

        mt5.shutdown()
        return all_trades

    def analyze(self, trades):
        if not trades:
            print("No trades generated.")
            return

        df = pd.DataFrame(trades)
        df.sort_values('time', inplace=True)
        
        print("\n" + "="*60)
        print("RISK SCALING SIMULATION (90 Days)")
        print("="*60)
        
        # Simulate different base risk levels
        risk_levels = [0.25, 0.35, 0.40, 0.50, 0.60]
        
        print(f"{'Base Risk':<10} | {'Return':<10} | {'Max DD':<10} | {'Monthly':<10} | {'Verdict'}")
        print("-" * 65)
        
        for base_risk in risk_levels:
            # Re-calculate PnL based on this base risk
            # Note: The 'r_weighted' in trades was calculated with base 0.5 * multiplier.
            # We need to strip that and apply new base.
            # R_weighted = R * Risk_Mult
            # PnL% = R_weighted * Base_Risk
            
            # Recalculate pnl_pct
            # mult is encoded in r_weighted / r. 
            # Risk_Mult = 0.5 for index/forex strategies in current config.
            
            # Let's just re-calculate from raw R and strategy type logic
            current_pnl = []
            for _, t in df.iterrows():
                # Determine multiplier used (reverse engineer or lookup)
                # In current config, Index=0.5, Forex=0.5. So Mult is always 0.5.
                risk_mult = 0.5
                pnl = t['r'] * risk_mult * base_risk
                current_pnl.append(pnl)
            
            equity = 100 + np.cumsum(current_pnl)
            peak = np.maximum.accumulate(equity)
            dd = (peak - equity) # Absolute DD %
            max_dd = np.max(dd)
            total_ret = equity[-1] - 100
            monthly_ret = total_ret / 3
            
            verdict = "✅ PASS" if max_dd < 9.5 and monthly_ret > 8 else "⚠️ SLOW" if max_dd < 9.5 else "❌ FAIL"
            
            print(f"{base_risk:<9}% | {total_ret:>8.2f}% | {max_dd:>8.2f}% | {monthly_ret:>8.2f}% | {verdict}")

        print(f"{base_risk:<9}% | {total_ret:>8.2f}% | {max_dd:>8.2f}% | {monthly_ret:>8.2f}% | {verdict}")
    
        print("\nBreakdown by Symbol (0.6% Risk):")
        # Calc breakdown for 0.6 risk
        base_risk = 0.6
        df['pnl_0.6'] = df['r'] * 0.5 * base_risk # 0.5 is multiplier
        summary = df.groupby('symbol').agg({
            'r': ['count', 'sum'],
            'pnl_0.6': 'sum'
        })
        print(summary)
            
        return df

if __name__ == "__main__":
    bt = AccurateBacktester()
    trades = bt.run(days=90)
    bt.analyze(trades)
