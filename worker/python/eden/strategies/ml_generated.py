from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

from .base import StrategyBase

class MLGeneratedStrategy(StrategyBase):
    name = "ml_generated"

    def __init__(self, model_path: Path | None = None):
        self.model_path = model_path or Path("models/sample_model.joblib")
        self.model = None
        try:
            import joblib
            if self.model_path.exists():
                self.model = joblib.load(self.model_path)
        except Exception:
            self.model = None

    def on_data(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            # Fallback heuristic: buy when RSI < 35, sell when RSI > 65
            rsi = df.get('rsi_14', pd.Series(index=df.index, dtype=float))
            buy = rsi < 35
            sell = rsi > 65
            signals = []
            for ts, b, s in zip(df.index, buy, sell):
                if b:
                    signals.append({"timestamp": ts, "side": "buy", "confidence": 0.4})
                elif s:
                    signals.append({"timestamp": ts, "side": "sell", "confidence": 0.4})
            return pd.DataFrame(signals, columns=["timestamp", "side", "confidence"])
        else:
            # Example: model outputs prob_up; convert to signals
            import numpy as np
            feats = df.select_dtypes(include=['float64','float32','int64','int32']).fillna(0.0)
            prob = self.model.predict_proba(feats)[:, 1]
            buy = prob > 0.6
            sell = prob < 0.4
            signals = []
            for ts, b, s, p in zip(df.index, buy, sell, prob):
                if b:
                    signals.append({"timestamp": ts, "side": "buy", "confidence": float(p)})
                elif s:
                    signals.append({"timestamp": ts, "side": "sell", "confidence": float(1-p)})
            return pd.DataFrame(signals, columns=["timestamp", "side", "confidence"])
            return pd.DataFrame(signals)
