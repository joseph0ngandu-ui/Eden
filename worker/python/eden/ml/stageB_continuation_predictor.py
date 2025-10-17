from __future__ import annotations
from typing import Dict, Tuple
import math


def run_stageB_row(row: Dict) -> Tuple[float, float]:
    """Lightweight continuation/reversal probability and expected return estimate.
    Uses EMA slope and recent momentum; returns (P_continue, expected_return_estimate).
    """
    close = float(row.get('close', 0.0))
    ema50 = float(row.get('ema_50', close))
    ema200 = float(row.get('ema_200', close))
    rsi14 = float(row.get('rsi_14', 50.0))
    macd_hist = float(row.get('macd_hist', 0.0))

    slope = (ema50 - ema200) / (abs(ema200) + 1e-6)
    rsi_bias = (rsi14 - 50.0) / 50.0
    m = 0.6 * slope + 0.2 * rsi_bias + 0.2 * math.tanh(macd_hist)
    p = 1.0 / (1.0 + math.exp(-3.0 * m))  # squash

    exp_ret = 0.001 * m  # small expected return proxy
    return float(p), float(exp_ret)