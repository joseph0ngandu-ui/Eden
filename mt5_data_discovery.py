#!/usr/bin/env python3
"""
MT5 Data Discovery Tool
=======================

Find all available symbols and data on the MT5 server.
This will help us identify what real data we can work with.
"""

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta

def discover_mt5_data():
    """Discover all available MT5 data"""
    
    print("🔍 MT5 Data Discovery Tool")
    print("=" * 50)
    
    # Connect to MT5
    if not mt5.initialize():
        print(f"❌ MT5 initialization failed: {mt5.last_error()}")
        return None
    
    print("✅ Connected to MT5")
    
    # Get account info
    account_info = mt5.account_info()
    if account_info:
        print(f"📊 Account: {account_info.login} ({account_info.server})")
    
    print("\n🔍 Discovering Available Symbols...")
    
    # Get all symbols
    all_symbols = mt5.symbols_get()
    
    if not all_symbols:
        print("❌ No symbols found")
        mt5.shutdown()
        return None
    
    print(f"✅ Found {len(all_symbols)} total symbols")
    
    # Filter for major forex and gold symbols
    major_symbols = []
    for symbol in all_symbols:
        symbol_name = symbol.name
        if any(pair in symbol_name for pair in ['EUR', 'GBP', 'USD', 'JPY', 'CHF', 'AUD', 'CAD', 'XAU', 'GOLD']):
            if symbol.visible:  # Only visible symbols
                major_symbols.append(symbol)
    
    print(f"📈 Found {len(major_symbols)} relevant trading symbols:")
    for symbol in major_symbols[:20]:  # Show first 20
        print(f"   • {symbol.name} - {symbol.description}")
    
    if len(major_symbols) > 20:
        print(f"   ... and {len(major_symbols) - 20} more")
    
    # Test data availability for top symbols
    test_symbols = [s.name for s in major_symbols[:10]]  # Test top 10
    timeframes = [
        (mt5.TIMEFRAME_M1, "M1"),
        (mt5.TIMEFRAME_M5, "M5"),
        (mt5.TIMEFRAME_M15, "M15"),
        (mt5.TIMEFRAME_M30, "M30"),
        (mt5.TIMEFRAME_H1, "H1"), 
        (mt5.TIMEFRAME_H4, "H4"),
        (mt5.TIMEFRAME_D1, "D1")
    ]
    
    print(f"\n📊 Testing Data Availability for Top Symbols:")
    print("=" * 80)
    
    # Test different lookback periods
    lookback_days = [7, 30, 90, 180, 365, 730]  # 1 week to 2 years
    
    best_symbols = {}
    
    for days in lookback_days:
        start_date = datetime.now() - timedelta(days=days)
        end_date = datetime.now()
        
        print(f"\n📅 Last {days} days ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}):")
        print("-" * 70)
        
        for symbol in test_symbols:
            print(f"{symbol:>12}: ", end="")
            symbol_data = {}
            
            for tf, tf_name in timeframes:
                try:
                    rates = mt5.copy_rates_range(symbol, tf, start_date, end_date)
                    if rates is not None and len(rates) > 0:
                        count = len(rates)
                        symbol_data[tf_name] = count
                        if count >= 100:  # Good amount of data
                            print(f"{tf_name}={count:>4} ", end="")
                        else:
                            print(f"{tf_name}={count:>4} ", end="")
                    else:
                        print(f"{tf_name}=   0 ", end="")
                        symbol_data[tf_name] = 0
                except Exception as e:
                    print(f"{tf_name}= ERR ", end="")
                    symbol_data[tf_name] = 0
            
            print()
            
            # Store best data for each symbol
            total_bars = sum(symbol_data.values())
            if symbol not in best_symbols or total_bars > sum(best_symbols[symbol]['data'].values()):
                best_symbols[symbol] = {
                    'period': f"last_{days}_days",
                    'start_date': start_date,
                    'end_date': end_date,
                    'data': symbol_data,
                    'total_bars': total_bars
                }
    
    print("\n" + "=" * 80)
    
    # Find the best symbols and periods
    print("\n🎯 Best Available Data:")
    
    viable_symbols = []
    for symbol, info in best_symbols.items():
        if info['total_bars'] > 500:  # Need minimum viable data
            viable_symbols.append((symbol, info))
    
    if viable_symbols:
        # Sort by total data available
        viable_symbols.sort(key=lambda x: x[1]['total_bars'], reverse=True)
        
        print(f"✅ Found {len(viable_symbols)} symbols with substantial data:")
        
        recommended_symbols = []
        recommended_start = None
        recommended_end = None
        
        for symbol, info in viable_symbols[:5]:  # Top 5
            print(f"\n📈 {symbol}:")
            print(f"   Period: {info['period']} ({info['start_date'].strftime('%Y-%m-%d')} to {info['end_date'].strftime('%Y-%m-%d')})")
            print(f"   Total Bars: {info['total_bars']:,}")
            print("   Timeframes: ", end="")
            for tf, count in info['data'].items():
                if count >= 100:
                    print(f"{tf}={count} ", end="")
            print()
            
            # Check if good for multi-timeframe analysis
            has_ltf = any(info['data'][tf] >= 100 for tf in ['M1', 'M5', 'M15'])
            has_htf = any(info['data'][tf] >= 50 for tf in ['H4', 'D1'])
            
            if has_ltf and has_htf:
                recommended_symbols.append(symbol)
                if recommended_start is None:
                    recommended_start = info['start_date']
                    recommended_end = info['end_date']
        
        if recommended_symbols:
            print(f"\n💡 Recommended Configuration:")
            print(f"   Symbols: {recommended_symbols}")
            print(f"   Start Date: {recommended_start.strftime('%Y-%m-%d')}")
            print(f"   End Date: {recommended_end.strftime('%Y-%m-%d')}")
            print(f"   HTF Analysis: H4, D1")
            print(f"   LTF Entries: M5, M15")
            
            result = {
                'symbols': recommended_symbols,
                'start_date': recommended_start,
                'end_date': recommended_end,
                'htf_timeframes': ['H4', 'D1'],
                'ltf_timeframes': ['M5', 'M15'],
                'data_details': {s: info for s, info in viable_symbols if s in recommended_symbols}
            }
            
            mt5.shutdown()
            return result
    
    print("❌ No symbols with sufficient data found")
    mt5.shutdown()
    return None

if __name__ == "__main__":
    result = discover_mt5_data()
    if result:
        print(f"\n✅ Data discovery complete!")
    else:
        print(f"\n❌ Data discovery failed!")