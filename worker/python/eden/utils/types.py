from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal
from datetime import datetime

@dataclass
class Order:
    timestamp: datetime
    symbol: str
    side: Literal['buy','sell']
    qty: float
    price: float
    order_type: Literal['market','limit','stop'] = 'market'
    id: Optional[str] = None

@dataclass
class Trade:
    open_time: datetime
    close_time: datetime
    symbol: str
    side: Literal['long','short']
    qty: float
    entry_price: float
    exit_price: float
    pnl: float
    strategy: str
    tag: Optional[str] = None
    rrr: float = 0.0

@dataclass
class Signal:
    timestamp: datetime
    side: Literal['buy','sell']
    confidence: float = 1.0
    meta: Optional[dict] = None
