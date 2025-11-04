#!/usr/bin/env python3
"""
Price Action module: BOS/CHoCH, rejection/engulfing candles, momentum shifts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np


@dataclass
class PAConfirm:
    direction: Optional[str]
    strength: float  # 0..1


class PriceAction:
    def __init__(self, swing_lookback: int = 20):
        self.swing_lookback = swing_lookback

    def _swings(self, df: pd.DataFrame, lb: int) -> pd.DataFrame:
        high_roll_max = df['high'].rolling(lb, min_periods=1).max()
        low_roll_min = df['low'].rolling(lb, min_periods=1).min()
        return pd.DataFrame({'swing_high': high_roll_max, 'swing_low': low_roll_min})

    def detect_bos_choch(self, df: pd.DataFrame) -> pd.Series:
        """Detect basic BOS/CHoCH via break of prior swing high/low."""
        swings = self._swings(df, self.swing_lookback)
        bos_up = (df['close'] > swings['swing_high'].shift(1)).astype(int)
        bos_dn = -((df['close'] < swings['swing_low'].shift(1)).astype(int))
        return (bos_up + bos_dn).fillna(0).astype(int)

    def detect_rejection(self, df: pd.DataFrame) -> pd.Series:
        """Pin-bar style rejection: long wick vs body.
        +1 bullish rejection, -1 bearish rejection.
        """
        body = (df['close'] - df['open']).abs()
        upper_wick = df['high'] - df[['close','open']].max(axis=1)
        lower_wick = df[['close','open']].min(axis=1) - df['low']
        avg_range = (df['high'] - df['low']).rolling(20, min_periods=1).mean()
        bull = ((lower_wick > body * 1.5) & (lower_wick > avg_range * 0.4)).astype(int)
        bear = -(((upper_wick > body * 1.5) & (upper_wick > avg_range * 0.4)).astype(int))
        return (bull + bear).fillna(0).astype(int)

    def detect_momentum_shift(self, df: pd.DataFrame, fast: int = 5, slow: int = 13) -> pd.Series:
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        up = ((ema_fast > ema_slow) & (ema_fast.shift(1) <= ema_slow.shift(1))).astype(int)
        dn = -(((ema_fast < ema_slow) & (ema_fast.shift(1) >= ema_slow.shift(1))).astype(int))
        return (up + dn).fillna(0).astype(int)
