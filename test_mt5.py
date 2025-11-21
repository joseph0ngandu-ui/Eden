import MetaTrader5 as mt5
import os

path = r"C:\Program Files\MetaTrader 5 Terminal\terminal64.exe"
print(f"Testing connection to: {path}")
print(f"Path exists: {os.path.exists(path)}")

try:
    if not mt5.initialize(path=path):
        print("initialize() failed, error code =", mt5.last_error())
    else:
        print("initialize() succeeded")
        print(mt5.terminal_info())
        mt5.shutdown()
except Exception as e:
    print(f"Exception during initialize: {e}")
