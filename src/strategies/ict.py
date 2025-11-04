#!/usr/bin/env python3
"""
ICT module: Higher-timeframe (HTF) bias, Order Blocks (OB), Fair Value Gaps (FVG), Liquidity Sweeps.
Vectorized detectors with graceful fallbacks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np


@dataclass
class HTFBias:
    direction: Optional[str]
    strength: float  # 0..1


class ICT:
    """ICT feature detectors on OHLCV data (M5 base).

    Methods are pure/data-driven and return pandas Series or structured outputs.
    """

    def __init__(self, htf_minutes: int = 60, ma_period: int = 50):
        self.htf_minutes = htf_minutes
        self.ma_period = ma_period

    def compute_htf_bias(self, df: pd.DataFrame) -> HTFBias:
        """Compute higher-timeframe bias using MA slope on resampled data.

        - direction: LONG/SHORT/None
        - strength: normalized slope magnitude in [0,1]
        """
        if df.empty or 'time' not in df.columns:
            return HTFBias(None, 0.0)
        d = df.set_index('time').resample(f'{self.htf_minutes}T').agg({'open':'first','high':'max','low':'min','close':'last'})
        if len(d) < self.ma_period + 2:
            return HTFBias(None, 0.0)
        ma = d['close'].rolling(self.ma_period, min_periods=1).mean()
        slope = (ma - ma.shift(1))
        last = slope.iloc[-1]
        direction = 'LONG' if last > 0 else ('SHORT' if last < 0 else None)
        # normalize by rolling std of slope
        denom = slope.rolling(20, min_periods=1).std().iloc[-1]
        strength = float(np.tanh((abs(last) / (denom if denom and not np.isnan(denom) and denom != 0 else 1e-6))))
        return HTFBias(direction, strength)

    def detect_fvg(self, df: pd.DataFrame) -> pd.Series:
        """Detect Fair Value Gaps (3-candle logic).
        FVG up if low[i] > high[i-2]; FVG down if high[i] < low[i-2]. Returns +1 (up-gap), -1 (down-gap), 0 none.
        """
        if len(df) < 3:
            return pd.Series(np.zeros(len(df), dtype=int), index=df.index)
        up = (df['low'] > df['high'].shift(2)).astype(int)
        down = -((df['high'] < df['low'].shift(2)).astype(int))
        return (up + down).fillna(0).astype(int)

    def detect_order_blocks(self, df: pd.DataFrame, lookback: int = 20) -> pd.Series:
        """Heuristic OB detection: mark the last opposing candle before a displacement (range expansion).
        Returns +1 bullish OB zones, -1 bearish OB zones, 0 otherwise.
        """
        if len(df) < 5:
            return pd.Series(np.zeros(len(df), dtype=int), index=df.index)
        body = (df['close'] - df['open']).abs()
        range_expansion = (df['high'] - df['low']) > (df['high'] - df['low']).rolling(lookback, min_periods=1).mean() * 1.25
        bull_ob = ((df['close'] > df['open']) & range_expansion & (df['open'] < df['low'].shift(1))).astype(int)
        bear_ob = -(((df['close'] < df['open']) & range_expansion & (df['open'] > df['high'].shift(1))).astype(int))
        return (bull_ob + bear_ob).fillna(0).astype(int)

    def detect_liquidity_sweep(self, df: pd.DataFrame, swing_lookback: int = 20) -> pd.Series:
        """Liquidity sweep: wick takes prior swing high/low then closes back inside.
        Returns +1 for buy-side sweep (took highs), -1 for sell-side sweep (took lows).
        """
        if len(df) < swing_lookback + 2:
            return pd.Series(np.zeros(len(df), dtype=int), index=df.index)
        prior_high = df['high'].rolling(swing_lookback, min_periods=1).max().shift(1)
        prior_low = df['low'].rolling(swing_lookback, min_periods=1).min().shift(1)
        took_highs = (df['high'] > prior_high) & (df['close'] < prior_high)
        took_lows = (df['low'] < prior_low) & (df['close'] > prior_low)
        return (took_highs.astype(int) - took_lows.astype(int)).fillna(0).astype(int)
