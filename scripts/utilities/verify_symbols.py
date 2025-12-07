"""
Verify new trading symbols are valid in MT5
"""
import MetaTrader5 as mt5

# Initialize MT5
if not mt5.initialize():
    print(f"❌ MT5 initialization failed: {mt5.last_error()}")
    exit(1)

print("✅ MT5 Connected\n")
print(f"Broker: {mt5.account_info().server}\n")

# Symbols to test
test_symbols = [
    "EURUSDm", "GBPUSDm", "USDJPYm", "AUDJPYm", 
    "XAUUSDm", "AUDUSDm", "USDCADm"
]

print("=== Testing Trading Symbols ===\n")
valid_symbols = []
invalid_symbols = []

for symbol in test_symbols:
    info = mt5.symbol_info(symbol)
    if info:
        print(f"✅ {symbol:<12} - {info.description}")
        
        # Try to get some data
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 10)
        if rates is not None and len(rates) > 0:
            print(f"   📊 Data available: {len(rates)} bars")
            valid_symbols.append(symbol)
        else:
            print(f"   ⚠️  Symbol found but no data available")
            invalid_symbols.append(symbol)
    else:
        print(f"❌ {symbol:<12} - NOT FOUND")
        invalid_symbols.append(symbol)

print(f"\n{'='*50}")
print(f"✅ Valid symbols: {len(valid_symbols)}/{len(test_symbols)}")
print(f"Valid: {', '.join(valid_symbols)}")
if invalid_symbols:
    print(f"❌ Invalid: {', '.join(invalid_symbols)}")

mt5.shutdown()
print("\n✅ Symbol verification complete!")
