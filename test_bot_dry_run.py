#!/usr/bin/env python3
"""
Test script for Eden bot dry-run with single account and default symbols.
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from trading_bot import TradingBot
    print("✓ TradingBot imported successfully")
    
    # Initialize bot with default config
    symbols = ["Volatility 75 Index"]
    print(f"✓ Initializing bot with symbols: {symbols}")
    
    bot = TradingBot(
        symbols=symbols,
        config_path="config.yaml"
    )
    print("✓ Bot initialized successfully")
    
    # Connect to MT5
    print("✓ Connecting to MT5...")
    if bot.connect():
        print("✓ MT5 connection successful")
        
        # Get account info
        import MetaTrader5 as mt5
        account = mt5.account_info()
        if account:
            print(f"✓ Account: {account.login}")
            print(f"✓ Balance: ${account.balance:.2f}")
            print(f"✓ Server: {account.server}")
        
        # Test fetching data
        print(f"✓ Testing data fetch for {symbols[0]}...")
        df = bot.fetch_recent_data(symbols[0], bars=50)
        if df is not None:
            print(f"✓ Fetched {len(df)} bars of data")
            print(f"✓ Latest close: {df['close'].iloc[-1]:.2f}")
            
            # Test signal calculation
            signal = bot.calculate_signal(df)
            print(f"✓ Signal calculated: {signal} (1=BUY, 0=NEUTRAL, -1=SELL)")
        else:
            print("✗ Failed to fetch data")
        
        # Disconnect
        bot.disconnect()
        print("✓ Bot dry-run test completed successfully")
    else:
        print("✗ MT5 connection failed")
        sys.exit(1)

except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Missing dependencies. Installing required packages...")
    sys.exit(1)
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
