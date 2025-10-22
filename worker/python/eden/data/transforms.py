import pandas as pd


def normalize_timeframe(timeframe: str) -> str:
    """Normalize common timeframe aliases to Eden's canonical notation.
    Canonical set: M1, 5M, 15M, 1H, 4H, 1D, 1W, 1MO
    Notes:
      - M1 means 1 minute (use 1M internally for pandas resample rule mapping)
      - 1MO means 1 month (avoid ambiguity with M1)
      - Accept aliases like 1m, m1, 1min, h4, 4h, d1, w1, mn1, 1month, etc.
    """
    tf = (timeframe or "").strip().upper()
    aliases = {
        # Minutes
        "M1": "M1",
        "1M": "M1",
        "1MIN": "M1",
        "MIN1": "M1",
        "1MINUTE": "M1",
        "5M": "5M",
        "05M": "5M",
        "15M": "15M",
        # Hours
        "1H": "1H",
        "H1": "1H",
        "4H": "4H",
        "H4": "4H",
        # Days/Weeks/Months
        "1D": "1D",
        "D1": "1D",
        "1W": "1W",
        "W1": "1W",
        "1MO": "1MO",
        "MO1": "1MO",
        "MN1": "1MO",
        "1MN": "1MO",
        "1MONTH": "1MO",
    }
    return aliases.get(tf, tf)


def timeframe_to_rule(timeframe: str) -> str:
    tf = normalize_timeframe(timeframe)
    # We map canonical codes to pandas offset aliases
    mapping = {
        "M1": "1T",  # 1 minute
        "5M": "5T",
        "15M": "15T",
        "1H": "1H",
        "4H": "4H",
        "1D": "1D",
        "1W": "1W",
        "1MO": "M",
    }
    return mapping.get(tf, tf)


def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    rule = timeframe_to_rule(timeframe)
    df["close"].resample(rule).ohlc()
    vol = df["volume"].resample(rule).sum()
    open_ = df["open"].resample(rule).first()
    high_ = df["high"].resample(rule).max()
    low_ = df["low"].resample(rule).min()
    close_ = df["close"].resample(rule).last()
    out = pd.DataFrame(
        {
            "open": open_,
            "high": high_,
            "low": low_,
            "close": close_,
            "volume": vol,
        }
    ).dropna()
    return out


def to_utc(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def normalize_symbol(symbol: str) -> str:
    return symbol.upper()
