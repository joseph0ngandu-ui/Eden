from __future__ import annotations
import numpy as np

# Minimal LSTM probability inference placeholder to avoid hard dependency on torch
# If torch is available, you can replace this with a real model.


def infer_lstm_probability(df) -> np.ndarray:
    """Return directional probability per row using a lightweight heuristic.
    This is a placeholder for a real LSTM; it falls back to a momentum-based proxy.
    """
    try:
        import pandas as pd

        # Simple momentum heuristic: normalized RSI as probability
        rsi = df.get("rsi_14")
        if rsi is None:
            return np.full(len(df), 0.5)
        rsi = rsi.fillna(50.0)
        prob = (rsi.clip(0, 100) / 100.0).values
        # Smooth
        if len(prob) > 5:
            prob = (
                pd.Series(prob, index=df.index).rolling(5, min_periods=1).mean().values
            )
        return prob
    except Exception:
        return np.full(len(df), 0.5)
