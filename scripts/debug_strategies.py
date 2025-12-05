
import sys
import os
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import MetaTrader5 as mt5
from trading.pro_strategies import ProStrategyEngine

def initialize_mt5():
    mt5_path = r"C:\Program Files\MetaTrader 5 Terminal\terminal64.exe"
    if not mt5.initialize(path=mt5_path):
        print(f"❌ MT5 initialization failed: {mt5.last_error()}")
        return False
    print("✓ MT5 connected")
    return True

def fetch_data(symbol, n_bars=1000):
    if not mt5.symbol_select(symbol, True):
        print(f"  ⚠ {symbol} not available")
        return None
    
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, n_bars)
    if rates is None or len(rates) == 0:
        return None
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df

def main():
    if not initialize_mt5():
        return

    symbol = "EURUSDm"
    print(f"Fetching data for {symbol}...")
    df = fetch_data(symbol, 5000)
    
    if df is None:
        print("No data")
        return

    print(f"Data: {len(df)} bars")
    
    engine = ProStrategyEngine()
    
    print("Running evaluation loop...")
    signals = 0
    
    # Simulate loop
    for i in range(100, len(df)):
        window = df.iloc[:i+1]
        signal = engine.evaluate_live(window, symbol)
        if signal:
            signals += 1
            print(f"Signal at {window.index[-1]}: {signal.direction} ({signal.strategy})")
            
    print(f"Total signals: {signals}")
    mt5.shutdown()

if __name__ == "__main__":
    main()
