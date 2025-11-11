import MetaTrader5 as mt5

# Initialize MT5 connection
if not mt5.initialize():
    print("MT5 initialization failed")
    print(f"Error code: {mt5.last_error()}")
else:
    print("✓ MT5 initialized successfully")
    
    # Get terminal info
    info = mt5.terminal_info()
    if info is not None:
        print(f"✓ Terminal: {info.name}")
        print(f"✓ Company: {info.company}")
        print(f"✓ Path: {info.path}")
        print(f"✓ Connected to server: {info.connected}")
        print(f"✓ Trade allowed: {info.trade_allowed}")
    
    # Get account info
    account_info = mt5.account_info()
    if account_info is not None:
        print(f"✓ Account: {account_info.login}")
        print(f"✓ Server: {account_info.server}")
        print(f"✓ Balance: ${account_info.balance:.2f}")
        print(f"✓ Equity: ${account_info.equity:.2f}")
        print(f"✓ Leverage: 1:{account_info.leverage}")
    
    # Shutdown
    mt5.shutdown()
    print("✓ MT5 connection test completed")
