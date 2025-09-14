from __future__ import annotations
import logging
from dataclasses import dataclass

from .broker_base import BrokerInterface

try:
    import ccxt  # type: ignore
except Exception:
    ccxt = None


@dataclass
class CCXTBroker(BrokerInterface):
    exchange_name: str = "binance"

    def __post_init__(self):
        self.log = logging.getLogger("eden.execution.ccxt_broker")
        if ccxt is None:
            self.exchange = None
            self.log.warning("ccxt not installed; CCXTBroker disabled. Fallback to paper recommended.")
        else:
            try:
                cls = getattr(ccxt, self.exchange_name)
                self.exchange = cls()
            except Exception as e:
                self.exchange = None
                self.log.warning("Exchange %s init failed: %s", self.exchange_name, e)

    def place_order(self, *args, **kwargs):
        if self.exchange is None:
            raise RuntimeError("CCXTBroker unavailable (ccxt missing or exchange init failed)")
        raise NotImplementedError

    def cancel_order(self, order_id: str):
        if self.exchange is None:
            raise RuntimeError("CCXTBroker unavailable")
        raise NotImplementedError

    def get_balance(self):
        if self.exchange is None:
            return {"cash": 0.0}
        return self.exchange.fetch_balance()

    def get_positions(self):
        return {}
