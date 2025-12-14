
import sys
import os
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime
import traceback

# Add project root to path
import yaml
from trading.pro_strategies import ProStrategyEngine
# from infrastructure.config_loader import ConfigLoader

def verify_system():
    print("="*60)
    print("🚀 EDEN TRADING SYSTEM DIAGNOSTIC")
    print("="*60)
    
    # 1. Check MT5 Connection
    if not mt5.initialize():
        print(f"❌ MT5 CONNECTION FAILED: {mt5.last_error()}")
        return
    else:
        print(f"✅ MT5 Connected. Version: {mt5.version()}")
        account = mt5.account_info()
        if account:
            print(f"   Account: {account.login} | Balance: {account.balance} | Equity: {account.equity}")
        else:
            print("   ⚠️ Failed to get account info")

    # 2. Load Config
    try:
        with open("c:\\Users\\opc\\Desktop\\Eden\\config\\config.yaml", "r") as f:
            config = yaml.safe_load(f)
        symbols = config['trading']['symbols']
        risk = config['risk_management']['risk_per_trade']
        print(f"✅ Config Loaded. Risk: {risk}%")
        print(f"   Active Symbols: {symbols}")
    except Exception as e:
        print(f"❌ CONFIG LOAD FAILED: {e}")
        return

    # 3. Strategy Engine Test
    print("\n🔍 TESTING STRATEGY LOGIC & DATA FEED:")
    try:
        engine = ProStrategyEngine()
        print("✅ ProStrategyEngine Initialized")
    except Exception as e:
        print(f"❌ ENGINE INIT FAILED: {e}")
        return

    # 4. Symbol & Data Check
    for symbol in symbols:
        print(f"\n   Checking {symbol}...")
        
        # Test M5 Data (Primary for Scalpers)
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 100)
        if rates is None or len(rates) == 0:
            print(f"   ❌ NO M5 DATA for {symbol} (Error: {mt5.last_error()})")
            continue
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        print(f"      ✅ M5 Data OK ({len(df)} bars). Last: {df.index[-1]}")
        
        # Test Strategy Execution
        try:
            # We call evaluate_live for M5
            # Note: evaluate_live usually takes 'timeframe' arg if calling directly?
            # Or we iterate strategies?
            # ProStrategyEngine.evaluate_live(df, symbol, timeframe)
            
            # Let's test execution
            trade = engine.evaluate_live(df, symbol, 5)
            status = "SIGNAL" if trade else "NO SIGNAL"
            print(f"      ✅ Strategy Exec M5: OK ({status})")
            
            # Test M15 (for Gold/Index)
            if "XAU" in symbol or "US" in symbol:
                rates_m15 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 100)
                if rates_m15 is not None:
                     df_m15 = pd.DataFrame(rates_m15)
                     df_m15['time'] = pd.to_datetime(df_m15['time'], unit='s')
                     df_m15.set_index('time', inplace=True)
                     trade_m15 = engine.evaluate_live(df_m15, symbol, 15)
                     status_m15 = "SIGNAL" if trade_m15 else "NO SIGNAL"
                     print(f"      ✅ Strategy Exec M15: OK ({status_m15})")
            
        except Exception as e:
            print(f"      ❌ STRATEGY CRASH: {e}")
            traceback.print_exc()

    print("\n" + "="*60)
    print("✅ DIAGNOSTIC COMPLETE")
    print("="*60)
    mt5.shutdown()

if __name__ == "__main__":
    verify_system()
