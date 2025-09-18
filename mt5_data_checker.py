#!/usr/bin/env python3
"""
MT5 Data Availability Checker
=============================

Check what historical data is available in MT5 for different symbols and timeframes.
This will help us find the right date range to use for backtesting.
"""

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta

def check_mt5_data_availability():
    """Check MT5 data availability"""
    
    print("🔍 MT5 Data Availability Checker")
    print("=" * 50)
    
    # Connect to MT5
    if not mt5.initialize():
        print(f"❌ MT5 initialization failed: {mt5.last_error()}")
        return
    
    print("✅ Connected to MT5")
    
    # Get account info
    account_info = mt5.account_info()
    if account_info:
        print(f"📊 Account: {account_info.login} ({account_info.server})")
    
    symbols = ['XAUUSD', 'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF']
    timeframes = [
        (mt5.TIMEFRAME_M15, "15M"),
        (mt5.TIMEFRAME_H1, "1H"), 
        (mt5.TIMEFRAME_H4, "4H"),
        (mt5.TIMEFRAME_D1, "1D")
    ]
    
    # Test different date ranges
    date_ranges = [
        ("Recent 30 days", datetime.now() - timedelta(days=30), datetime.now()),
        ("Recent 90 days", datetime.now() - timedelta(days=90), datetime.now()),
        ("2025 YTD", datetime(2025, 1, 1), datetime.now()),
        ("2024 Full Year", datetime(2024, 1, 1), datetime(2024, 12, 31)),
        ("2023 Full Year", datetime(2023, 1, 1), datetime(2023, 12, 31)),
        ("Last 6 months", datetime.now() - timedelta(days=180), datetime.now()),
    ]
    
    print("\n📅 Testing Data Availability:")
    print("=" * 80)
    
    for range_name, start_date, end_date in date_ranges:
        print(f"\n🗓️ {range_name} ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}):")
        print("-" * 60)
        
        for symbol in symbols:
            print(f"{symbol:>8}: ", end="")
            
            for tf, tf_name in timeframes:
                try:
                    rates = mt5.copy_rates_range(symbol, tf, start_date, end_date)
                    if rates is not None and len(rates) > 0:
                        print(f"{tf_name}={len(rates):>4} ", end="")
                    else:
                        print(f"{tf_name}=   0 ", end="")
                except:
                    print(f"{tf_name}= ERR ", end="")
            print()
    
    print("\n" + "=" * 80)
    
    # Find the best available period
    print("\n🎯 Finding Best Available Data Period:")
    best_period = None
    max_data_score = 0
    
    for range_name, start_date, end_date in date_ranges:
        data_score = 0
        symbol_data = {}
        
        for symbol in symbols:
            symbol_score = 0
            for tf, tf_name in timeframes:
                try:
                    rates = mt5.copy_rates_range(symbol, tf, start_date, end_date)
                    if rates is not None and len(rates) > 100:  # Need at least 100 bars
                        symbol_score += len(rates)
                except:
                    pass
            
            symbol_data[symbol] = symbol_score
            data_score += symbol_score
        
        if data_score > max_data_score:
            max_data_score = data_score
            best_period = (range_name, start_date, end_date, symbol_data)
    
    if best_period:
        range_name, start_date, end_date, symbol_data = best_period
        print(f"✅ Best Period Found: {range_name}")
        print(f"   📅 Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"   📊 Total Data Score: {max_data_score:,}")
        print(f"   🎯 Symbols with Data:")
        
        for symbol, score in symbol_data.items():
            if score > 0:
                print(f"      • {symbol}: {score:,} data points")
        
        print(f"\n💡 Recommended Settings:")
        print(f"   start_date = datetime({start_date.year}, {start_date.month}, {start_date.day})")
        print(f"   end_date = datetime({end_date.year}, {end_date.month}, {end_date.day})")
    
    else:
        print("❌ No suitable data period found")
    
    # Disconnect
    mt5.shutdown()
    print("\n✅ Disconnected from MT5")

if __name__ == "__main__":
    check_mt5_data_availability()