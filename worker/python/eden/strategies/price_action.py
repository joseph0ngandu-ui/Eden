from __future__ import annotations
import pandas as pd

from .base import StrategyBase


class PriceActionStrategy(StrategyBase):
    name = "price_action"

    def on_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # Simple price action: engulfing patterns
        o, c = df['open'], df['close']
        prev_o, prev_c = o.shift(1), c.shift(1)
        bull_engulf = (c > o) & (prev_c < prev_o) & (c > prev_o) & (o < prev_c)
        bear_engulf = (c < o) & (prev_c > prev_o) & (c < prev_o) & (o > prev_c)
        signals = []
        for ts, b, s in zip(df.index, bull_engulf, bear_engulf):
            if b:
                signals.append({"timestamp": ts, "side": "buy", "confidence": 0.6})
            elif s:
                signals.append({"timestamp": ts, "side": "sell", "confidence": 0.6})
        return pd.DataFrame(signals, columns=["timestamp", "side", "confidence"])
