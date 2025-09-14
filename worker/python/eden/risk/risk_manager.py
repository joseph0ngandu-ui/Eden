from __future__ import annotations
from dataclasses import dataclass


@dataclass
class RiskManager:
    max_leverage: float = 5.0
    per_order_risk_fraction: float = 0.02

    def allow_order(self, symbol: str, side: str, qty: float, price: float, equity: float) -> bool:
        # simple risk check: notional <= per_order_risk_fraction * equity * max_leverage
        notional = qty * price
        limit = self.per_order_risk_fraction * equity * self.max_leverage
        return notional <= limit
