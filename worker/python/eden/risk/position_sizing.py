from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Literal


def fixed_fraction(equity: float, risk_frac: float, atr: float | None = None, atr_mult: float = 1.0, price: float | None = None) -> float:
    # position size by risking fraction of equity. If ATR provided, normalize by ATR * mult, else by price
    risk_amount = equity * max(0.0, min(risk_frac, 1.0))
    denom = (atr * atr_mult) if (atr and atr > 0) else (price if price and price > 0 else 1.0)
    return float(max(0.0, risk_amount / denom))


def kelly_fraction(win_rate: float, win_loss_ratio: float, cap: float = 0.1) -> float:
    # Kelly f* = p - (1-p)/b
    p = max(0.0, min(1.0, win_rate))
    b = max(1e-6, win_loss_ratio)
    f = p - (1 - p) / b
    return float(max(0.0, min(cap, f)))


def volatility_based(equity: float, atr: float, atr_mult: float = 1.0) -> float:
    if atr <= 0:
        return 0.0
    risk_per_unit = atr * atr_mult
    risk_budget = equity * 0.01
    return float(risk_budget / risk_per_unit)
