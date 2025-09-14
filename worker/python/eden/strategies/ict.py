from __future__ import annotations
import pandas as pd

from .base import StrategyBase


class ICTStrategy(StrategyBase):
    name = "ict"

    def on_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # Simplified ICT: buy on bullish FVG + sweep_low; sell on bearish FVG + sweep_high
        cond_buy = (df.get('fvg_bull', 0) == 1) & (df.get('sweep_low', 0) == 1)
        cond_sell = (df.get('fvg_bear', 0) == 1) & (df.get('sweep_high', 0) == 1)
        signals = []
        for ts, buy, sell in zip(df.index, cond_buy, cond_sell):
            if buy:
                signals.append({"timestamp": ts, "side": "buy", "confidence": 0.7})
            elif sell:
                signals.append({"timestamp": ts, "side": "sell", "confidence": 0.7})
        return pd.DataFrame(signals, columns=["timestamp", "side", "confidence"])
