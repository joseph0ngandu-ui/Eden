from pathlib import Path
from datetime import datetime, timezone, timedelta
from eden.data.loader import DataLoader

symbol = "Volatility 100 Index"
# Use last 3 months for intraday data (MT5 typically limits intraday history)
end = datetime.now(timezone.utc).date().isoformat()
start = (datetime.now(timezone.utc) - timedelta(days=90)).date().isoformat()
intraday_timeframes = ["1M", "5M", "15M", "1H"]

print(f"Fetching recent intraday data from {start} to {end}")

# Prefer MT5, no external fallbacks; DataLoader will merge+dedupe into data/layered
loader = DataLoader(cache_dir=Path("data/cache"))

for tf in intraday_timeframes:
    try:
        print(f"Processing {symbol} {tf}...")
        df = loader.get_ohlcv(
            symbol=symbol,
            timeframe=tf,
            start=start,
            end=end,
            allow_network=False,
            force_refresh=True,  # Force refresh to bypass any existing cache
            prefer_mt5=True,
        )
        if df is None or df.empty:
            print(f"[WARN] No MT5 data returned for {symbol} {tf}")
            continue
        layered_file = Path("data/layered") / f"{symbol.upper()}_{tf.upper()}.csv"
        print(f"[OK] {symbol} {tf}: {len(df)} rows merged -> {layered_file}")

        # Show date range for verification
        if not df.empty:
            first_date = df.index[0].strftime("%Y-%m-%d %H:%M")
            last_date = df.index[-1].strftime("%Y-%m-%d %H:%M")
            print(f"    Data range: {first_date} to {last_date}")

    except Exception as e:
        print(f"[ERROR] {symbol} {tf}: {e}")
        import traceback

        print(f"    Details: {traceback.format_exc()}")

print("\nIntraday backfill complete!")
