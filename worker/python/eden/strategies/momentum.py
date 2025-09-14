from __future__ import annotations
import pandas as pd

from .base import StrategyBase


class MomentumStrategy(StrategyBase):
    name = "momentum"

    def on_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # Momentum: EMA50 crosses EMA200
        ema50 = df['ema_50'] if 'ema_50' in df.columns else df['close'].ewm(span=50, adjust=False).mean()
        ema200 = df['ema_200'] if 'ema_200' in df.columns else df['close'].ewm(span=200, adjust=False).mean()
        cross_up = (ema50 > ema200) & (ema50.shift(1) <= ema200.shift(1))
        cross_dn = (ema50 < ema200) & (ema50.shift(1) >= ema200.shift(1))
        signals = []
        for ts, b, s in zip(df.index, cross_up, cross_dn):
            if b:
                signals.append({"timestamp": ts, "side": "buy", "confidence": 0.6})
            elif s:
                signals.append({"timestamp": ts, "side": "sell", "confidence": 0.6})
        return pd.DataFrame(signals, columns=["timestamp", "side", "confidence"])
