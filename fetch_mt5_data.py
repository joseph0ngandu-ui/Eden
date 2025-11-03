#!/usr/bin/env python3
"""
MT5 Data Fetcher for all 10 instruments
Fetches 1-week of 1-minute OHLCV data for optimization and backtesting
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from pathlib import Path

# Configuration - Exact broker symbols from MT5
INSTRUMENTS = {
    "Volatility 25 Index": "VIX25",
    "Volatility 50 Index": "VIX50", 
    "Volatility 75 Index": "VIX75",
    "Volatility 100 Index": "VIX100",
    "Boom 1000 Index": "Boom1000",
    "Boom 500 Index": "Boom500",
    "Crash 1000 Index": "Crash1000",
    "Crash 500 Index": "Crash500",
    "Step Index": "StepIndex",
    "XAUUSD": "XAUUSD"
}

# Multiple timeframes for analysis
TIMEFRAMES = {
    'M1': mt5.TIMEFRAME_M1,
    'M5': mt5.TIMEFRAME_M5,
    'M15': mt5.TIMEFRAME_M15,
    'H1': mt5.TIMEFRAME_H1,
    'H4': mt5.TIMEFRAME_H4
}

DATA_DIR = Path("data/mt5_feeds")
DAYS_BACK = 7  # 1 week of data

class MT5DataFetcher:
    def __init__(self):
        self.data_dir = DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.connection_status = False
        self.available_symbols = {}
        
    def connect(self):
        """Initialize MT5 connection"""
        if not mt5.initialize():
            print(f"❌ MT5 initialization failed: {mt5.last_error()}")
            return False
        
        self.connection_status = True
        print("✅ MT5 initialized successfully")
        return True
    
    def disconnect(self):
        """Close MT5 connection"""
        mt5.shutdown()
        self.connection_status = False
        print("✅ MT5 disconnected")
    
    def discover_available_symbols(self) -> dict:
        """Discover available symbols matching our target instruments"""
        print("\n🔍 Discovering available symbols on broker...")
        
        symbols = mt5.symbols_total()
        print(f"   Total symbols available: {symbols}")
        
        target_keywords = ['VIX', 'Volatility', 'Boom', 'Crash', 'Step', 'XAUUSD', 'Gold']
        matching_symbols = {}
        
        # Get all symbols
        all_symbols = mt5.symbols_get()
        
        if all_symbols:
            for symbol_info in all_symbols:
                symbol_name = symbol_info.name
                
                # Check if symbol matches any keywords
                for keyword in target_keywords:
                    if keyword.lower() in symbol_name.lower():
                        if keyword not in matching_symbols:
                            matching_symbols[keyword] = []
                        matching_symbols[keyword].append(symbol_name)
        
        if matching_symbols:
            print("\n   ✅ Found matching symbols:")
            for keyword, symbols_list in matching_symbols.items():
                print(f"      {keyword}: {symbols_list}")
        else:
            print("   ⚠️  No matching symbols found - check with broker")
        
        return matching_symbols
    
    def fetch_instrument_data(self, symbol: str, timeframe_name: str = 'M1', days_back: int = DAYS_BACK) -> pd.DataFrame:
        """
        Fetch OHLCV data for a single instrument and timeframe
        
        Args:
            symbol: Trading symbol (e.g., "VIX75")
            timeframe_name: Timeframe key ('M1', 'M5', 'M15', 'H1', 'H4')
            days_back: Number of days to fetch
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            timeframe = TIMEFRAMES.get(timeframe_name, mt5.TIMEFRAME_M1)
            
            # Calculate date range
            utc_to = datetime.now()
            utc_from = utc_to - timedelta(days=days_back)
            
            print(f"  📊 {symbol} {timeframe_name}...", end=' ', flush=True)
            
            # Fetch rates
            rates = mt5.copy_rates_range(symbol, timeframe, utc_from, utc_to)
            
            if rates is None or len(rates) == 0:
                print(f"⚠️  No data")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df = df[['time', 'open', 'high', 'low', 'close', 'tick_volume', 'real_volume']]
            df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'real_volume']
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            print(f"✅ {len(df)} candles")
            return df
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            return None
    
    def save_instrument_data(self, symbol: str, timeframe: str, df: pd.DataFrame):
        """Save instrument data to CSV"""
        if df is None or len(df) == 0:
            return
        
        file_path = self.data_dir / f"{symbol}_{timeframe}.csv"
        df.to_csv(file_path, index=False)
    
    def fetch_all_instruments(self) -> dict:
        """Fetch data for all instruments across multiple timeframes"""
        all_data = {}
        
        if not self.connect():
            return all_data
        
        try:
            # First discover available symbols
            available = self.discover_available_symbols()
            
            print("\n" + "="*70)
            print("FETCHING MULTI-TIMEFRAME DATA FOR ALL INSTRUMENTS")
            print("="*70)
            print(f"Timeframes: {', '.join(TIMEFRAMES.keys())}")
            print(f"Date Range: {DAYS_BACK} days")
            print("="*70 + "\n")
            
            # Fetch each instrument across all timeframes
            for broker_symbol, display_name in INSTRUMENTS.items():
                print(f"\n{display_name}:")
                all_data[display_name] = {}
                
                for timeframe_name in TIMEFRAMES.keys():
                    df = self.fetch_instrument_data(broker_symbol, timeframe_name, DAYS_BACK)
                    if df is not None and len(df) > 0:
                        all_data[display_name][timeframe_name] = df
                        self.save_instrument_data(display_name, timeframe_name, df)
                    else:
                        all_data[display_name][timeframe_name] = None
        
        finally:
            self.disconnect()
        
        return all_data
    
    def generate_data_summary(self, all_data: dict):
        """Generate summary of fetched data"""
        summary = {
            "fetch_timestamp": datetime.now().isoformat(),
            "days_back": DAYS_BACK,
            "timeframes": list(TIMEFRAMES.keys()),
            "instruments": {}
        }
        
        total_candles = 0
        
        for symbol, timeframes_data in all_data.items():
            summary["instruments"][symbol] = {}
            for timeframe_name, df in timeframes_data.items():
                if df is not None and len(df) > 0:
                    candles = len(df)
                    date_range = f"{df['timestamp'].min().date()} to {df['timestamp'].max().date()}"
                    summary["instruments"][symbol][timeframe_name] = {
                        "candles": candles,
                        "date_range": date_range,
                        "status": "OK"
                    }
                    total_candles += candles
                else:
                    summary["instruments"][symbol][timeframe_name] = {"status": "FAILED"}
        
        summary["total_candles"] = total_candles
        
        # Save summary
        summary_file = self.data_dir / "fetch_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print("\n" + "="*70)
        print("📈 MULTI-TIMEFRAME DATA FETCH COMPLETE")
        print("="*70)
        print(f"Instruments: {len(all_data)}")
        print(f"Timeframes: {', '.join(TIMEFRAMES.keys())}")
        print(f"Total Candles: {total_candles:,}")
        print("="*70)
        
        return summary

def main():
    """Main execution"""
    print("\n" + "="*60)
    print("🚀 EDEN MT5 DATA FETCHER")
    print("Fetching 1-week of 1-minute data for all 10 instruments")
    print("="*60)
    
    fetcher = MT5DataFetcher()
    all_data = fetcher.fetch_all_instruments()
    summary = fetcher.generate_data_summary(all_data)
    
    if len(all_data) == len(INSTRUMENTS):
        print("\n✅ All instruments fetched successfully!")
        return True
    else:
        print(f"\n⚠️  Only {len(all_data)}/{len(INSTRUMENTS)} instruments fetched")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
