import MetaTrader5 as mt5
import os
import sys

def verify_connection():
    print("Initializing MetaTrader 5...")
    
    # Attempt to initialize
    if not mt5.initialize():
        print(f"initialize() failed, error code = {mt5.last_error()}")
        return False
        
    print(f"MetaTrader 5 package version: {mt5.__version__}")
    print(f"MetaTrader 5 terminal version: {mt5.version()}")
    
    # Check connection status
    terminal_info = mt5.terminal_info()
    if terminal_info:
        print(f"\nTerminal Info:")
        print(f"  Path: {terminal_info.path}")
        print(f"  Name: {terminal_info.name}")
        print(f"  Connected: {terminal_info.connected}")
        
    account_info = mt5.account_info()
    if account_info:
        print(f"\nAccount Info:")
        print(f"  Login: {account_info.login}")
        print(f"  Server: {account_info.server}")
        print(f"  Balance: {account_info.balance}")
        print(f"  Equity: {account_info.equity}")
    else:
        print("\nNo account information available. Are you logged in?")
        
    mt5.shutdown()
    return True

if __name__ == "__main__":
    if verify_connection():
        print("\nSUCCESS: MetaTrader 5 connection verified!")
        sys.exit(0)
    else:
        print("\nFAILURE: Could not connect to MetaTrader 5.")
        sys.exit(1)
