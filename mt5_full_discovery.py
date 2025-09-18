#!/usr/bin/env python3
"""
MT5 Complete Data Discovery
===========================

Find all available data for EURUSDm, GBPUSDm, XAUUSDm, US30m, USTECm
Focus on finding maximum M5 data for entries with top-down analysis.
"""

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta

def discover_all_symbols_data():
    """Discover data for all requested symbols"""
    
    print("🔍 MT5 Complete Data Discovery")
    print("=" * 80)
    
    # Connect to MT5
    if not mt5.initialize():
        print(f"❌ MT5 initialization failed: {mt5.last_error()}")
        return None
    
    print("✅ Connected to MT5")
    
    account_info = mt5.account_info()
    if account_info:
        print(f"📊 Account: {account_info.login} ({account_info.server})")
    
    # Target symbols
    target_symbols = ['EURUSDm', 'GBPUSDm', 'XAUUSDm', 'US30m', 'USTECm']
    
    # Also check variations
    symbol_variations = {
        'EURUSDm': ['EURUSDm', 'EURUSD', 'EUR/USD'],
        'GBPUSDm': ['GBPUSDm', 'GBPUSD', 'GBP/USD'],
        'XAUUSDm': ['XAUUSDm', 'XAUUSD', 'GOLD', 'Gold'],
        'US30m': ['US30m', 'US30', 'DJ30', 'DOW30', 'DJI'],
        'USTECm': ['USTECm', 'USTEC', 'NAS100', 'NASDAQ', 'NDX']
    }
    
    # Get all available symbols
    all_symbols = mt5.symbols_get()
    if not all_symbols:
        print("❌ No symbols found")
        mt5.shutdown()
        return None
    
    print(f"✅ Total symbols on server: {len(all_symbols)}")
    
    # Find matching symbols
    found_symbols = {}
    for target, variations in symbol_variations.items():
        for symbol in all_symbols:
            if symbol.visible and any(v.upper() in symbol.name.upper() for v in variations):
                found_symbols[target] = symbol.name
                print(f"✅ Found {target}: {symbol.name}")
                break
    
    print(f"\n📈 Checking data availability for found symbols...")
    print("=" * 80)
    
    # Timeframes to check (focus on M5 for entries)
    timeframes = [
        (mt5.TIMEFRAME_M5, "M5"),   # Primary entry timeframe
        (mt5.TIMEFRAME_M15, "M15"),
        (mt5.TIMEFRAME_M30, "M30"),
        (mt5.TIMEFRAME_H1, "H1"),
        (mt5.TIMEFRAME_H4, "H4"),
        (mt5.TIMEFRAME_D1, "D1")
    ]
    
    # Test different periods to find maximum data
    test_periods = [
        (30, "Last 30 days"),
        (90, "Last 90 days"),
        (180, "Last 180 days"),
        (365, "Last 365 days"),
        (730, "Last 2 years"),
        (1095, "Last 3 years")
    ]
    
    best_config = {
        'symbols': [],
        'period_days': 0,
        'start_date': None,
        'end_date': None,
        'total_m5_bars': 0,
        'data_summary': {}
    }
    
    for days, period_name in test_periods:
        start_date = datetime.now() - timedelta(days=days)
        end_date = datetime.now()
        
        print(f"\n📅 {period_name} ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}):")
        print("-" * 70)
        
        period_m5_total = 0
        period_symbols = []
        period_data = {}
        
        for target, symbol_name in found_symbols.items():
            print(f"{symbol_name:>12}: ", end="")
            
            symbol_has_data = False
            symbol_m5_bars = 0
            
            for tf, tf_name in timeframes:
                try:
                    rates = mt5.copy_rates_range(symbol_name, tf, start_date, end_date)
                    if rates is not None and len(rates) > 0:
                        count = len(rates)
                        
                        if tf_name == "M5":
                            symbol_m5_bars = count
                            period_m5_total += count
                        
                        if count >= 100:  # Minimum viable data
                            symbol_has_data = True
                            print(f"{tf_name}={count:>5} ", end="")
                        else:
                            print(f"{tf_name}={count:>5} ", end="")
                    else:
                        print(f"{tf_name}=    0 ", end="")
                except:
                    print(f"{tf_name}=  ERR ", end="")
            
            print()
            
            if symbol_has_data and symbol_m5_bars >= 1000:  # Need good M5 data
                period_symbols.append(symbol_name)
                period_data[symbol_name] = symbol_m5_bars
        
        # Check if this is the best period
        if period_m5_total > best_config['total_m5_bars'] and len(period_symbols) >= 2:
            best_config = {
                'symbols': period_symbols,
                'period_days': days,
                'start_date': start_date,
                'end_date': end_date,
                'total_m5_bars': period_m5_total,
                'data_summary': period_data
            }
    
    print("\n" + "=" * 80)
    print("🎯 OPTIMAL CONFIGURATION FOUND:")
    
    if best_config['symbols']:
        print(f"\n✅ Best Period: Last {best_config['period_days']} days")
        print(f"📅 Date Range: {best_config['start_date'].strftime('%Y-%m-%d')} to {best_config['end_date'].strftime('%Y-%m-%d')}")
        print(f"📊 Total M5 Bars: {best_config['total_m5_bars']:,}")
        print(f"\n🎯 Symbols with Best M5 Data:")
        for symbol, m5_bars in best_config['data_summary'].items():
            print(f"   • {symbol}: {m5_bars:,} M5 bars")
        
        print(f"\n💡 Configuration to Use:")
        print(f"   symbols = {best_config['symbols']}")
        print(f"   start_date = datetime({best_config['start_date'].year}, {best_config['start_date'].month}, {best_config['start_date'].day})")
        print(f"   end_date = datetime({best_config['end_date'].year}, {best_config['end_date'].month}, {best_config['end_date'].day})")
        print(f"   primary_timeframe = 'M5'  # For entries")
        print(f"   analysis_timeframes = ['D1', 'H4', 'H1', 'M30', 'M15', 'M5']  # Top-down")
    else:
        print("❌ No viable configuration found")
    
    mt5.shutdown()
    return best_config

if __name__ == "__main__":
    config = discover_all_symbols_data()
    if config:
        print("\n✅ Discovery complete! Ready for optimization.")
    else:
        print("\n❌ Discovery failed!")