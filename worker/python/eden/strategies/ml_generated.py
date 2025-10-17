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
        self.feature_alignment = None
        try:
            import joblib
            if self.model_path.exists():
                self.model = joblib.load(self.model_path)
                
                # Load feature alignment if available
                alignment_path = self.model_path.parent / "feature_alignment.json"
                if alignment_path.exists():
                    with open(alignment_path, 'r') as f:
                        self.feature_alignment = json.load(f)
        except Exception:
            self.model = None
            self.feature_alignment = None

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
            
            # Apply feature alignment if available
            if self.feature_alignment:
                # De-duplicate while preserving order
                ordered = []
                seen = set()
                for f in self.feature_alignment:
                    if f not in seen:
                        seen.add(f)
                        ordered.append(f)
                # Align features to exact training order and fill missing with 0
                feats = df.drop(columns=['open','high','low','close','volume'], errors='ignore')
                feats = feats.select_dtypes(include=[np.number])
                feats = feats.reindex(columns=ordered, fill_value=0.0).fillna(0.0)
            else:
                feats = df.select_dtypes(include=['float64','float32','int64','int32']).fillna(0.0)
            
            prob = self.model.predict_proba(feats)[:, 1]
            # Relaxed thresholds to increase signal volume for HF scenarios
            buy = prob > 0.55
            sell = prob < 0.45
            signals = []
            for ts, b, s, p in zip(df.index, buy, sell, prob):
                if b:
                    signals.append({"timestamp": ts, "side": "buy", "confidence": float(p)})
                elif s:
                    signals.append({"timestamp": ts, "side": "sell", "confidence": float(1-p)})
            return pd.DataFrame(signals, columns=["timestamp", "side", "confidence"])
