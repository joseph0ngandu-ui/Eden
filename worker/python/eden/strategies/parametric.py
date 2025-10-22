from __future__ import annotations
import pandas as pd
from dataclasses import dataclass

from .base import StrategyBase


@dataclass
class RuleBasedParamStrategy(StrategyBase):
    name: str = "parametric"
    params: dict = None

    def __post_init__(self):
        if self.params is None:
            self.params = {}

    def on_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # Generic rule templates with parameters
        rsi_col = df.get("rsi_14")
        ema_fast = df.get("ema_50")
        ema_slow = df.get("ema_200")
        macd_hist = df.get("macd_hist")

        import pandas as pd

        signals = []
        th_buy_rsi = float(self.params.get("buy_rsi_max", 35))
        th_sell_rsi = float(self.params.get("sell_rsi_min", 65))
        use_cross = bool(self.params.get("use_ema_cross", True))
        macd_bias = float(self.params.get("macd_hist_bias", 0.0))

        # Conditions
        if rsi_col is None:
            rsi_col = df["close"].rolling(14).apply(lambda x: 50, raw=False)
        cond_buy = rsi_col < th_buy_rsi
        cond_sell = rsi_col > th_sell_rsi
        if use_cross and ema_fast is not None and ema_slow is not None:
            cond_buy = cond_buy & (ema_fast > ema_slow)
            cond_sell = cond_sell & (ema_fast < ema_slow)
        if macd_hist is not None:
            cond_buy = cond_buy & (macd_hist > macd_bias)
            cond_sell = cond_sell & (macd_hist < -macd_bias)

        for ts, b, s in zip(df.index, cond_buy, cond_sell):
            if b:
                signals.append({"timestamp": ts, "side": "buy", "confidence": 0.5})
            elif s:
                signals.append({"timestamp": ts, "side": "sell", "confidence": 0.5})
        return pd.DataFrame(signals, columns=["timestamp", "side", "confidence"])
