#!/usr/bin/env python3
"""
Historical data provider using MetaTrader5
"""

from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
import MetaTrader5 as mt5

TIMEFRAME_MAP = {
    'M1': mt5.TIMEFRAME_M1,
    'M5': mt5.TIMEFRAME_M5,
    'M15': mt5.TIMEFRAME_M15,
    'M30': mt5.TIMEFRAME_M30,
    'H1': mt5.TIMEFRAME_H1,
}

def fetch_ohlc(symbol: str, timeframe: str = 'M5', days: int = 90) -> Optional[pd.DataFrame]:
    if not mt5.initialize():
        return None
    try:
        tf = TIMEFRAME_MAP.get(timeframe, mt5.TIMEFRAME_M5)
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days)
        rates = mt5.copy_rates_range(symbol, tf, from_date, to_date)
        if rates is None:
            return None
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df
    finally:
        mt5.shutdown()
