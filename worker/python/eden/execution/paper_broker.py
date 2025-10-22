from __future__ import annotations
import logging
from dataclasses import dataclass

from .broker_base import BrokerInterface


@dataclass
class PaperBroker(BrokerInterface):
    starting_cash: float = 100000.0
    slippage_bps: float = 1.0

    def __post_init__(self):
        self.cash = self.starting_cash
        self.positions = {}
        self.log = logging.getLogger("eden.execution.paper_broker")

    def place_order(self, symbol: str, side: str, qty: float, price: float):
        slip = price * (self.slippage_bps / 10000.0)
        exec_price = price + slip if side == "buy" else price - slip
        notional = exec_price * qty
        if side == "buy":
            self.cash -= notional
            self.positions[symbol] = self.positions.get(symbol, 0.0) + qty
        else:
            self.cash += notional
            self.positions[symbol] = self.positions.get(symbol, 0.0) - qty
        return {"status": "filled", "price": exec_price}

    def cancel_order(self, order_id: str):
        return {"status": "canceled"}

    def get_balance(self):
        return {"cash": self.cash}

    def get_positions(self):
        return self.positions.copy()
