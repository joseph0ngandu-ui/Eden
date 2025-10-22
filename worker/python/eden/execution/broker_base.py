from __future__ import annotations


class BrokerInterface:
    def place_order(self, *args, **kwargs):
        raise NotImplementedError

    def cancel_order(self, order_id: str):
        raise NotImplementedError

    def get_balance(self):
        raise NotImplementedError

    def get_positions(self):
        raise NotImplementedError
