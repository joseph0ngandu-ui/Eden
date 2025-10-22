from __future__ import annotations
import pandas as pd

from .base import StrategyBase


class MeanReversionStrategy(StrategyBase):
    name = "mean_reversion"

    def on_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # Mean reversion using z-score to SMA20
        price = df["close"]
        sma20 = df["sma_20"] if "sma_20" in df.columns else price.rolling(20).mean()
        std20 = price.rolling(min(20, max(5, len(price) // 3 or 5))).std()
        z = (price - sma20) / (std20.replace(0, 1e-6) + 1e-6)
        buy = z < -1.0
        sell = z > 1.0
        signals = []
        for ts, b, s in zip(df.index, buy, sell):
            if b:
                signals.append({"timestamp": ts, "side": "buy", "confidence": 0.5})
            elif s:
                signals.append({"timestamp": ts, "side": "sell", "confidence": 0.5})
        return pd.DataFrame(signals, columns=["timestamp", "side", "confidence"])
