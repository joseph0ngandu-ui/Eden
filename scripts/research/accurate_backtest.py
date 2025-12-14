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
            # INDEX M15 (High Alloc)
            {"symbol": "USTECm", "tf": mt5.TIMEFRAME_M15, "type": "index"},
            {"symbol": "US500m", "tf": mt5.TIMEFRAME_M15, "type": "index"},
            
            # GOLD SMART SWEEP (High Alloc)
            {"symbol": "XAUUSDm", "tf": mt5.TIMEFRAME_M15, "type": "gold_sweep"},
            
            # SCALPING REMOVED (Failed Spread Stress Test)
            
            # MOMENTUM D1 (High Alloc)
            {"symbol": "USDCADm", "tf": mt5.TIMEFRAME_D1, "type": "momentum"},
            {"symbol": "EURUSDm", "tf": mt5.TIMEFRAME_D1, "type": "momentum"},
            {"symbol": "EURJPYm", "tf": mt5.TIMEFRAME_D1, "type": "momentum"},
            {"symbol": "CADJPYm", "tf": mt5.TIMEFRAME_D1, "type": "momentum"},
        ]

    def get_spread(self, symbol: str) -> float:
        # CONSERVATIVE ESTIMATES (FundedNext Challenge Conditions)
        if "XAU" in symbol: return 0.25  # Gold 25 cents
        if "USTEC" in symbol: return 1.5 # Nasdaq 1.5 points
        if "US500" in symbol: return 0.4 # S&P 0.4 points
        if "JPY" in symbol: return 0.012 # 1.2 pips
        return 0.00012 # 1.2 pips for EURUSD/others

    def simulate_trade(self, trade: Trade, future_data: pd.DataFrame) -> Dict:
        # Apply Spread to Entry
        spread = self.get_spread(trade.symbol)
        
        # Long enters at Ask (Price + Spread)
        # Short enters at Bid (Price)
        real_entry = trade.entry_price + spread if trade.direction == "LONG" else trade.entry_price
        
        sl = trade.sl
        tp = trade.tp
        direction = trade.direction
        
        exit_price = real_entry
        exit_time = future_data.index[-1]
        exit_reason = "end_of_data"
        
        for t, row in future_data.iterrows():
            if direction == "LONG":
                # Exit Longs at Bid (Low/High from chart are Bid)
                # Check SL (Bid <= SL)
                if row['low'] <= sl: 
                    exit_price = sl # Executed at SL (Slippage handled by Commission pad? Add extra slippage?)
                    exit_price -= spread * 0.1 # Slight slippage
                    exit_time = t
                    exit_reason = "sl"
                    break
                # Check TP (Bid >= TP)
                if row['high'] >= tp:
                    exit_price = tp
                    exit_time = t
                    exit_reason = "tp"
                    break
            else: # SHORT
                # Exit Shorts at Ask (Bid + Spread)
                # Check SL (Ask >= SL -> Bid + Spread >= SL -> Bid >= SL - Spread)
                # So if Chart High >= SL - Spread, we hit SL
                if row['high'] >= (sl - spread):
                    exit_price = sl + spread * 0.1 # Slippage
                    exit_time = t
                    exit_reason = "sl"
                    break
                # Check TP (Ask <= TP -> Bid + Spread <= TP -> Bid <= TP - Spread)
                if row['low'] <= (tp - spread):
                    exit_price = tp
                    exit_time = t
                    exit_reason = "tp"
                    break
        
        # Calculate Risk and PnL
        # Risk is Distance from Ideal Entry to SL
        risk_per_share = abs(trade.entry_price - sl)
        
        if direction == "LONG":
            pnl_amt = exit_price - real_entry
        else:
            pnl_amt = real_entry - exit_price
            
        r_multiple = pnl_amt / risk_per_share if risk_per_share > 0 else 0
        
        # SUBTRACT COMMISSIONS (FundedNext ~$3/lot -> 0.03% of Notional?)
        # 0.05R is a good dynamic proxy for Commission + Swap
        r_multiple -= 0.05 
        
        return {
            "time": trade.entry_time,
            "symbol": trade.symbol,
            "direction": trade.direction,
            "entry": real_entry,
            "exit": exit_price,
            "exit_time": exit_time,
            "r": r_multiple,
            "reason": exit_reason,
            "strategy": trade.strategy
        }

    def get_risk_multiplier(self, strategy_type: str) -> float:
        return 1.0

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
                    import traceback
                    traceback.print_exc()
                    print(f"Error in backtest loop: {e}")

        mt5.shutdown()
        return all_trades

    def analyze(self, trades):
        if not trades:
            print("No trades generated.")
            return

        df = pd.DataFrame(trades)
        df.sort_values('time', inplace=True)
        
        # Load Config Risk
        import yaml
        try:
            with open("config/config.yaml", "r") as f:
                config = yaml.safe_load(f)
                risk_per_trade = config.get("risk_management", {}).get("risk_per_trade", 0.5)
        except:
            risk_per_trade = 0.5 # Default
            
        print("\n" + "="*70)
        print(f"FUNDED NEXT 10K CHALLENGE SIMULATION (90 Days)")
        print(f"Configured Risk Per Trade: {risk_per_trade}%")
        print("="*70)
        
        INITIAL_BALANCE = 10000
        balance = INITIAL_BALANCE
        equity_curve = [INITIAL_BALANCE]
        daily_balances = {} # For Daily DD check
        
        metrics = {
            "wins": 0, "losses": 0, "gross_profit": 0, "gross_loss": 0,
            "max_dd_dollar": 0, "max_dd_percent": 0,
            "max_daily_loss": 0
        }
        
        for idx, t in df.iterrows():
            # Calculate PnL in Dollars
            # r_weighted includes strategy allocations (1.4x / 0.8x)
            # risk_amt = balance * (risk_per_trade / 100)
            # pnl = risk_amt * t['r_weighted'] (since r_weighted is R * allocator)
            # Wait, r_weighted was: result['r'] * risk_mult
            
            # Using FIXED balance risk (FundedNext usually static or equity based? Static is safer)
            risk_amt = INITIAL_BALANCE * (risk_per_trade / 100.0)
            pnl = risk_amt * t['r_weighted']
            
            balance += pnl
            equity_curve.append(balance)
            
            # Metrics
            if pnl > 0:
                metrics["wins"] += 1
                metrics["gross_profit"] += pnl
            else:
                metrics["losses"] += 1
                metrics["gross_loss"] += abs(pnl)
                
            # Daily Loss Logic
            date_str = t['time'].strftime('%Y-%m-%d')
            if date_str not in daily_balances:
                daily_balances[date_str] = {"start": equity_curve[-2], "low": balance}
            else:
                if balance < daily_balances[date_str]["low"]:
                     daily_balances[date_str]["low"] = balance
        
        # Calculate DD
        peak_equity = INITIAL_BALANCE
        max_dd_dollar = 0
        
        for eq in equity_curve:
            if eq > peak_equity: peak_equity = eq
            dd = peak_equity - eq
            if dd > max_dd_dollar: max_dd_dollar = dd
            
        metrics["max_dd_dollar"] = max_dd_dollar
        metrics["max_dd_percent"] = (max_dd_dollar / INITIAL_BALANCE) * 100
        
        # Calculate Max Daily Loss
        max_daily_dd = 0
        for date, data in daily_balances.items():
            start = data["start"]
            low = data["low"]
            loss = start - low
            if loss > max_daily_dd: max_daily_dd = loss
            
        final_balance = balance
        total_pnl = final_balance - INITIAL_BALANCE
        return_pct = (total_pnl / INITIAL_BALANCE) * 100
        
        # Report
        print(f"Initial Balance:   ${INITIAL_BALANCE:,.2f}")
        print(f"Final Balance:     ${final_balance:,.2f}")
        print(f"Net Profit:        ${total_pnl:,.2f} ({return_pct:.2f}%)")
        print(f"Total Trades:      {len(df)}")
        print(f"Win Rate:          {metrics['wins']/len(df)*100:.1f}%")
        print("-" * 30)
        print(f"Max Drawdown:      ${metrics['max_dd_dollar']:,.2f} ({metrics['max_dd_percent']:.2f}%) -> Limit: 9.5% ($950)")
        print(f"Max Daily Loss:    ${max_daily_dd:,.2f} ({(max_daily_dd/INITIAL_BALANCE)*100:.2f}%) -> Limit: 4.5% ($450)")
        print("-" * 30)
        
        passed = True
        fail_reasons = []
        if metrics["max_dd_percent"] >= 9.5: 
            passed = False
            fail_reasons.append("Max Drawdown Exceeded")
        if (max_daily_dd/INITIAL_BALANCE)*100 >= 4.5:
             passed = False
             fail_reasons.append("Daily Loss Exceeded")
             
        # Target usually 8% ($800)
        target_hit = return_pct >= 8.0
        
        if passed and target_hit:
            print("🏆 VERDICT: CHALLENGE PASSED")
        elif passed:
            print("⚠️ VERDICT: PASSED SAFETY, PROFIT PENDING (Extend Time)")
        else:
            print(f"❌ VERDICT: FAILED ({', '.join(fail_reasons)})")
            
        print("\nBreakdown by Strategy Type:")
        summary = df.groupby('stype').agg({
            'r_weighted': 'sum',
            'symbol': 'count'
        })
        print(summary)
            
        return df

if __name__ == "__main__":
    bt = AccurateBacktester()
    trades = bt.run(days=90)
    bt.analyze(trades)
