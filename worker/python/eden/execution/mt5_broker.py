from __future__ import annotations
import logging
import os
from dataclasses import dataclass


def is_mt5_available() -> bool:
    try:
        import MetaTrader5 as mt5  # type: ignore

        return True
    except Exception:
        return False


@dataclass
class MT5Broker:
    login: int | None = None
    password: str | None = None
    server: str | None = None

    @classmethod
    def from_env(cls):
        return cls(
            login=int(os.getenv("MT5_LOGIN")) if os.getenv("MT5_LOGIN") else None,
            password=os.getenv("MT5_PASSWORD"),
            server=os.getenv("MT5_SERVER"),
        )

    def connect(self) -> bool:
        try:
            import MetaTrader5 as mt5  # type: ignore

            if not mt5.initialize():
                return False
            if self.login and self.password and self.server:
                mt5.login(self.login, password=self.password, server=self.server)
            return True
        except Exception:
            logging.getLogger("eden.execution.mt5_broker").warning(
                "MT5 not available; fallback to paper"
            )
            return False
