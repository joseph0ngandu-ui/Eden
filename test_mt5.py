import MetaTrader5 as mt5
import sys
import os

print("Python executable:", sys.executable)
print("MetaTrader5 module file:", mt5.__file__)
print("MetaTrader5 version:", getattr(mt5, '__version__', 'unknown'))

# Try to locate terminal64.exe and add its directory to PATH/DLL search
mt5_terminal_path = r'C:\Program Files\MetaTrader 5 Terminal\terminal64.exe'
if os.path.isfile(mt5_terminal_path):
    print("MT5 terminal found at:", mt5_terminal_path)
    mt5_dir = os.path.dirname(mt5_terminal_path)
    print("Will add to PATH:", mt5_dir)
    os.environ['PATH'] = mt5_dir + ';' + os.environ.get('PATH', '')
else:
    print("MT5 terminal NOT found at:", mt5_terminal_path)

print("Attempting mt5.initialize()...")
init_ok = mt5.initialize()
print("Initialize result:", init_ok)
print("Last error:", mt5.last_error())
if init_ok:
    print("Connection succeeded.")
    mt5.shutdown()