"""
Check available symbols in MT5 and verify symbol names
"""
import MetaTrader5 as mt5
import os

# Initialize MT5
mt5_path = os.getenv("MT5_PATH", r"C:\Program Files\MetaTrader 5 Terminal\terminal64.exe")
if not mt5.initialize(path=mt5_path):
    if not mt5.initialize():
        print(f"❌ MT5 initialization failed: {mt5.last_error()}")
        exit(1)

print("✅ MT5 Connected\n")

# Get account info
account_info = mt5.account_info()
if account_info:
    print(f"📊 Broker: {account_info.server}")
    print(f"💰 Balance: ${account_info.balance:.2f}")
    print(f"📈 Equity: ${account_info.equity:.2f}\n")

# Common symbols we're looking for (with possible variations)
symbol_patterns = [
    "VIX", "VOLATILITY", "V75", "V50", "V25", "V100",  # Volatility indices
    "EUR", "GBP", "USD", "JPY", "AUD", "NZD", "CAD", "CHF",  # Forex pairs
    "XAU", "GOLD",  # Gold
]

print("=== Searching for Trading Symbols ===\n")

# Get all symbols
all_symbols = mt5.symbols_get()
print(f"Total symbols available: {len(all_symbols)}\n")

# Find matches
matches = []
for symbol in all_symbols:
    symbol_name = symbol.name
    for pattern in symbol_patterns:
        if pattern in symbol_name.upper():
            matches.append(symbol_name)
            break

print(f"Found {len(matches)} matching symbols:\n")
for sym in sorted(set(matches)):
    # Get symbol info
    info = mt5.symbol_info(sym)
    if info:
        print(f"  ✓ {sym:<20} - {info.description}")

# Check specific symbols from config
print("\n=== Checking Current Config Symbols ===\n")
config_symbols = ["Volatility 75 Index", "EURUSD", "GBPUSD", "XAUUSD"]

for sym in config_symbols:
    info = mt5.symbol_info(sym)
    if info:
        print(f"  ✅ {sym:<25} - VALID - {info.description}")
    else:
        print(f"  ❌ {sym:<25} - NOT FOUND")
        # Try to find similar
        similar = [s for s in matches if any(word in s.upper() for word in sym.upper().split())]
        if similar:
            print(f"     Possible alternatives: {', '.join(similar[:3])}")

mt5.shutdown()
print("\n✅ Symbol check complete!")
