from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

from pydantic import BaseModel


class EdenConfig(BaseModel):
    symbols: List[str] = ["XAUUSD", "EURUSD", "US30", "NAS100", "GBPUSD"]
    timeframe: str = "1D"
    start: str = "2018-01-01"
    end: str = "2023-12-31"
    starting_cash: float = 100000.0
    commission_bps: float = 1.0
    slippage_bps: float = 1.0
    broker: str = "paper"
    strategy: str = "ensemble"
    log_level: str = "INFO"


def load_config(path: str | None) -> EdenConfig:
    """Load YAML config if provided; otherwise use defaults.

    Supports .env env vars override for simple fields.
    """
    import yaml

    cfg = EdenConfig()
    if path and Path(path).exists():
        data = yaml.safe_load(Path(path).read_text()) or {}
        cfg = EdenConfig(**{**cfg.model_dump(), **data})

    # env overrides
    cfg.log_level = os.getenv("EDEN_LOG_LEVEL", cfg.log_level)
    live = os.getenv("EDEN_LIVE")
    if live == "1":
        cfg.broker = os.getenv("EDEN_BROKER", cfg.broker)

    return cfg
