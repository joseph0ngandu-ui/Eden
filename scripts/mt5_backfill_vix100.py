from pathlib import Path
from datetime import datetime, timezone
from eden.data.loader import DataLoader

symbol = "Volatility 100 Index"
start = "2010-01-01"
end = datetime.now(timezone.utc).date().isoformat()
timeframes = ["1M", "5M", "15M", "1H", "4H", "1D"]

# Prefer MT5, no external fallbacks; DataLoader will merge+dedupe into data/layered
loader = DataLoader(cache_dir=Path("data/cache"))

for tf in timeframes:
    try:
        df = loader.get_ohlcv(
            symbol=symbol,
            timeframe=tf,
            start=start,
            end=end,
            allow_network=False,
            force_refresh=False,
            prefer_mt5=True,
        )
        if df is None or df.empty:
            print(f"[WARN] No MT5 data returned for {symbol} {tf}")
            continue
        layered_file = Path("data/layered") / f"{symbol.upper()}_{tf.upper()}.csv"
        print(f"[OK] {symbol} {tf}: {len(df)} rows merged -> {layered_file}")
    except Exception as e:
        print(f"[ERROR] {symbol} {tf}: {e}")