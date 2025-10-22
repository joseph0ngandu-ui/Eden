import numpy as np
import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.clip(lower=0)).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / (loss + 1e-12)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr = pd.concat(
        [(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window).mean()


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def vwap(df: pd.DataFrame, window: int = 20) -> pd.Series:
    price = (df["high"] + df["low"] + df["close"]) / 3
    vol = df["volume"]
    pv = (price * vol).rolling(window).sum()
    vv = vol.rolling(window).sum()
    return pv / (vv + 1e-12)


def compute_fair_value_gaps(df: pd.DataFrame) -> pd.DataFrame:
    # FVG: bullish if current low > previous high; bearish if current high < previous low
    prev_high = df["high"].shift(1)
    prev_low = df["low"].shift(1)
    bull = df["low"] > prev_high
    bear = df["high"] < prev_low
    out = pd.DataFrame(
        {
            "fvg_bull": bull.astype(int),
            "fvg_bear": bear.astype(int),
            "fvg_gap_low": np.where(bull, prev_high, np.nan),
            "fvg_gap_high": np.where(bear, prev_low, np.nan),
        },
        index=df.index,
    )
    return out


def detect_liquidity_sweeps(df: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    # Sweep highs/lows beyond rolling extremes then close back in range
    hh = df["high"].rolling(lookback).max().shift(1)
    ll = df["low"].rolling(lookback).min().shift(1)
    sweep_high = (df["high"] > hh) & (df["close"] < df["open"])
    sweep_low = (df["low"] < ll) & (df["close"] > df["open"])
    return pd.DataFrame(
        {"sweep_high": sweep_high.astype(int), "sweep_low": sweep_low.astype(int)},
        index=df.index,
    )


def identify_order_blocks(
    df: pd.DataFrame, window: int = 5, vol_mult: float = 1.5
) -> pd.DataFrame:
    # Identify candles with volume spikes and directional blocks
    vol_ma = df["volume"].rolling(window).mean()
    vol_spike = df["volume"] > (vol_ma * vol_mult)
    bull_block = (df["close"] > df["open"]) & vol_spike
    bear_block = (df["close"] < df["open"]) & vol_spike
    return pd.DataFrame(
        {"ob_bull": bull_block.astype(int), "ob_bear": bear_block.astype(int)},
        index=df.index,
    )
