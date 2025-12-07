import MetaTrader5 as mt5
from datetime import datetime, timedelta
import pandas as pd

def check_d1_data():
    if not mt5.initialize():
        print("MT5 Init Failed")
        return

    symbols = ['USDCADm', 'EURUSDm', 'EURJPYm', 'CADJPYm']
    end = datetime.now()
    start = end - timedelta(days=120)

    print(f"Checking D1 data from {start.date()} to {end.date()}")

    for sym in symbols:
        # Ensure symbol is selected
        if not mt5.symbol_select(sym, True):
            print(f"{sym}: Failed to select")
            continue

        rates = mt5.copy_rates_range(sym, mt5.TIMEFRAME_D1, start, end)
        
        if rates is None:
            print(f"{sym}: No rates returned (None)")
            err = mt5.last_error()
            print(f"  Error: {err}")
        elif len(rates) == 0:
            print(f"{sym}: Empty rates list")
        else:
            df = pd.DataFrame(rates)
            print(f"{sym}: {len(rates)} bars found. First: {pd.to_datetime(df['time'].iloc[0], unit='s')}")

    mt5.shutdown()

if __name__ == "__main__":
    check_d1_data()
