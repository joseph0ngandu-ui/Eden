import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta

if not mt5.initialize(path=r"C:\Program Files\MetaTrader 5 Terminal\terminal64.exe"):
    print("MT5 Init Failed")
    exit()

symbol = "EURUSDm"
tf = mt5.TIMEFRAME_M5
days = 180

rates = mt5.copy_rates_range(symbol, tf, datetime.now() - timedelta(days=days), datetime.now())

if rates is None:
    print("No data found")
else:
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    print(f"Symbol: {symbol}")
    print(f"Timeframe: M5")
    print(f"Records: {len(df)}")
    if len(df) > 0:
        print(f"Start: {df['time'].iloc[0]}")
        print(f"End:   {df['time'].iloc[-1]}")
    else:
        print("Empty DataFrame")

mt5.shutdown()
