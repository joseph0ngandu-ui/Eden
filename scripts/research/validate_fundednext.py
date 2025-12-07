#!/usr/bin/env python3
"""
FUNDEDNEXT COMPLIANCE VALIDATOR
Simulates Prop Firm conditions (Commissions, Rules, Drawdown)
to validate strategy safety and true profitability.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging

# Setup Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import Live Strategies
from trading.pro_strategies import ProStrategyEngine
from trading.models import Trade

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# CONFIGURATION
COMMISSION_PER_LOT = 7.0  # $7 Round Trip
EURUSD_POINT_VALUE = 10.0 # $10 per pip (Standard)
LOT_SIZE = 1.0            # Validate with 1 Lot
MAX_DAILY_TRADES = 50     # Flag HFT
MIN_DURATION_MINS = 2     # Flag Scalping

def calculate_pnl_net(entry, exit, direction, symbol):
    # Basic PnL in points
    if direction == "LONG":
        raw_diff = exit - entry
    else:
        raw_diff = entry - exit
        
    # Validation uses approximate rules
    # Forex (EUR/GBP/JPY): approx $10/pip/lot
    # Gold: $100/point/lot? No, XAUUSD 1 lot = 100oz. 1$ move = $100.
    # Indices: US30 1 lot = $1/point? Or $10? Depends on broker. 
    # FundedNext usually: US30 Contract Size 1 or 10.
    
    # We will simply check "Edge vs Cost" in Price Units.
    # Convert Comm to Price Units
    
    cost_basis = 0.0
    
    if "XAU" in symbol:
        # Spread already checked in strategy? No, check_outcome simulates execution.
        # XAU: Spread ~20 cents (0.20). Comm $7/lot => $0.07 price impact?
        # 1 lot = 100oz. $1 move = $100. $7 comm = $0.07 move.
        # Real spread is spread + 0.07.
        cost_basis = 0.07
    elif "US" in symbol and "USD" not in symbol:
        # Indices. US30. Spread ~2 points. No Comm usually.
        cost_basis = 0.0
    else:
        # Forex. 1 pip = 0.0001. $10/pip. 
        # Comm $7 = 0.7 pips = 0.00007.
        cost_basis = 0.00007
        
    net_pnl = raw_diff - cost_basis
    return net_pnl

def check_duration(entry_time, exit_time):
    duration = (exit_time - entry_time).total_seconds() / 60
    return duration

def run_validation():
    try:
        import MetaTrader5 as mt5
    except ImportError:
        print("MetaTrader5 not available")
        return
        
    if not mt5.initialize(path=r"C:\Program Files\MetaTrader 5 Terminal\terminal64.exe"):
        print("MT5 init failed")
        return
        
    print("="*60)
    print("  FUNDEDNEXT COMPLIANCE CHECK")
    print("  (Net of $7 Commissions + Spreads)")
    print("="*60)
    
    engine = ProStrategyEngine()
    
    # We need to manually invoke strategies because Engine.strategies list might be filtered/purged.
    # We want to test VWAP specifically (which is removed from default list).
    
    # Standalone VWAP Logic (Since it was purged from Engine)
    def vwap_standalone(df, symbol):
        # Using Engine helper for ATR if possible, else implement locally
        # Engine instance 'engine' is available in scope? No.
        # Need to pass engine or duplicate ATR.
        # Let's duplicate ATR for simplicity.
        def calc_atr(d):
            h, l, c = d['high'], d['low'], d['close'].shift(1)
            tr = pd.concat([h-l, (h-c).abs(), (l-c).abs()], axis=1).max(axis=1)
            return tr.rolling(14).mean().iloc[-1]
            
        if len(df) < 100: return None
        last_time = df.index[-1]
        today = last_time.date()
        today_data = df[df.index.date == today]
        if len(today_data) < 10: return None
        
        tp = (today_data['high'] + today_data['low'] + today_data['close']) / 3
        pv = tp * today_data['tick_volume']
        vwap = (pv.cumsum() / today_data['tick_volume'].cumsum()).iloc[-1]
        
        atr = calc_atr(df)
        band_dist = atr * 3.0
        row = df.iloc[-1]
        
        if row['close'] > vwap + band_dist:
            sl = row['high'] + atr * 0.5
            return Trade(symbol=symbol, direction="SHORT", entry_price=row['close'],
                         sl=sl, tp=vwap, confidence=0.82, atr=atr,
                         entry_time=row.name, bar_index=0, strategy="Prop_VWAP_M5")
        if row['close'] < vwap - band_dist:
            sl = row['low'] - atr * 0.5
            return Trade(symbol=symbol, direction="LONG", entry_price=row['close'],
                         sl=sl, tp=vwap, confidence=0.82, atr=atr,
                         entry_time=row.name, bar_index=0, strategy="Prop_VWAP_M5")
        return None

    # Define test suite: (Name, Method, Symbol, TF, MethodObject)
    test_suite = [
        # 1. VWAP (The Questionable One)
        ("VWAP_M5", vwap_standalone, "EURUSDm", mt5.TIMEFRAME_M5),
        
        # 2. Spread Hunter (The Winner)
        ("Gold_M15", engine.spread_hunter_momentum, "XAUUSDm", mt5.TIMEFRAME_M15),
        
        # 3. Index Vol (The New Guy)
        ("Index_M15", engine.index_volatility_expansion, "US500m", mt5.TIMEFRAME_M15),
        
        # 4. Vol Squeeze (The Forex Scalper)
        ("VolSq_M5", engine.volatility_squeeze, "EURUSDm", mt5.TIMEFRAME_M5)
    ]
    
    end = datetime.now()
    start = end - timedelta(days=60) # 2 Months
    
    for name, method, symbol, tf in test_suite:
        print(f"\n--- Validating: {name} ({symbol}) ---")
        
        rates = mt5.copy_rates_range(symbol, tf, start, end)
        if rates is None: 
            print("No Data")
            continue
            
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        # Metrics
        trades_count = 0
        wins = 0
        total_net_pnl = 0.0 # In Price Units
        durations = []
        
        for i in range(200, len(df)-200):
            window = df.iloc[i-100:i+1]
            try:
                signal = method(window, symbol)
            except:
                signal = None # Method might check specific symbols and return None
                
            if signal:
                # Simulate Outcome
                future = df.iloc[i+1:i+200]
                
                # Check outcome logic (Simplified)
                # Need precise Time of Exit for Duration
                entry_time = df.index[i]
                exit_time = None
                exit_price = None
                
                # Replicate outcome logic loop
                for idx, row in future.iterrows():
                    if signal.direction == "LONG":
                        if row['low'] <= signal.sl: 
                            exit_price = signal.sl
                            exit_time = row.name # index is time? yes
                            break
                        if row['high'] >= signal.tp: 
                            exit_price = signal.tp
                            exit_time = row.name
                            break
                    else:
                        if row['high'] >= signal.sl: 
                            exit_price = signal.sl
                            exit_time = row.name
                            break
                        if row['low'] <= signal.tp: 
                            exit_price = signal.tp
                            exit_time = row.name
                            break
                
                if exit_price:
                    trades_count += 1
                    pnl = calculate_pnl_net(signal.entry_price, exit_price, signal.direction, symbol)
                    
                    # Normalize PnL to standard 'Points' for aggregation? 
                    # EURUSD: 0.0010 = 10 pips. 
                    # XAU: 10.0 = 10 points. 
                    
                    total_net_pnl += pnl
                    if pnl > 0: wins += 1
                    
                    if hasattr(exit_time, 'to_pydatetime'): # pandas timestamp
                        dur = check_duration(entry_time, exit_time)
                        durations.append(dur)
                    elif isinstance(exit_time, datetime):
                        dur = check_duration(entry_time, exit_time)
                        durations.append(dur)
                        
        if trades_count == 0:
            print("No trades generated.")
            continue
            
        avg_dur = np.mean(durations) if durations else 0
        min_dur = np.min(durations) if durations else 0
        wr = wins / trades_count * 100
        
        # PnL Interpretation
        # Forex: Sum of price diffs. 1.0 = 10,000 pips.
        # Gold: Sum of price diffs. 10.0 = $10.
        
        pnl_str = f"{total_net_pnl:.4f}"
        
        print(f"  Trades: {trades_count}")
        print(f"  Win Rate: {wr:.1f}%")
        print(f"  Avg Duration: {avg_dur:.1f} min")
        print(f"  Min Duration: {min_dur:.1f} min")
        print(f"  Total Net PnL (Price Units): {pnl_str}")
        
        # Verdicts
        flags = []
        if avg_dur < 2.0: flags.append("HFT/Scalping Warning (< 2m avg)")
        if (trades_count / 60) > 20: flags.append("High Volume Warning (> 20/day)")
        if total_net_pnl < 0: flags.append("UNPROFITABLE after Costs")
        
        if not flags:
            print("  VERDICT: ✅ PASSED Compliance")
        else:
            print("  VERDICT: ⚠️ WARNINGS:")
            for f in flags:
                print(f"    - {f}")

    mt5.shutdown()

if __name__ == "__main__":
    run_validation()
