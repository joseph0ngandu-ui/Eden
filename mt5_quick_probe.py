#!/usr/bin/env python3
"""
Quick MT5 data probe
- Connects to MetaTrader 5
- Checks last 90 days of data for specified symbols across M5/M15/H1/H4/D1
- Prints a JSON summary to stdout
"""
import json
from datetime import datetime, timedelta

try:
    import MetaTrader5 as mt5
except ImportError:
    print(json.dumps({"error": "MetaTrader5 module not installed. pip install MetaTrader5"}))
    raise SystemExit(1)

SYMBOLS = ["EURUSDm", "GBPUSDm", "XAUUSDm", "US30m", "USTECm"]
TIMEFRAMES = {
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
}

end = datetime.now()
start = end - timedelta(days=90)

result = {
    "mt5_initialized": False,
    "account": None,
    "server": None,
    "start": start.isoformat(),
    "end": end.isoformat(),
    "symbols": {},
}

try:
    if not mt5.initialize():
        result["error"] = f"MT5 initialization failed: {mt5.last_error()}"
        print(json.dumps(result))
        raise SystemExit(0)
    result["mt5_initialized"] = True

    acc = mt5.account_info()
    if acc:
        result["account"] = getattr(acc, "login", None)
        result["server"] = getattr(acc, "server", None)

    for sym in SYMBOLS:
        try:
            # ensure symbol is visible
            try:
                mt5.symbol_select(sym, True)
            except Exception:
                pass

            tf_counts = {}
            for name, tf in TIMEFRAMES.items():
                try:
                    rates = mt5.copy_rates_range(sym, tf, start, end)
                    tf_counts[name] = 0 if rates is None else len(rates)
                except Exception as e:
                    tf_counts[name] = f"ERR:{type(e).__name__}"
            result["symbols"][sym] = tf_counts
        except Exception as e:
            result["symbols"][sym] = {"error": str(e)}

finally:
    try:
        mt5.shutdown()
    except Exception:
        pass

print(json.dumps(result))
