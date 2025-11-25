from dataclasses import dataclass
from enum import Enum
from typing import Optional
from datetime import datetime
import pandas as pd

class OrderStatus(Enum):
    """Order status types."""
    PENDING = "PENDING"
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"

@dataclass
class Trade:
    """Trade signal object."""
    symbol: str
    direction: str  # "LONG" or "SHORT"
    entry_price: float
    tp: float
    sl: float
    confidence: float
    atr: float
    entry_time: pd.Timestamp
    bar_index: int
    strategy: str = "Unknown"

@dataclass
class Position:
    """Open position tracking."""
    symbol: str
    direction: str
    entry_price: float
    tp: float
    sl: float
    entry_bar_index: int
    entry_time: pd.Timestamp
    atr: float
    confidence: float
    strategy: str = "Unknown"
    stop_moved: bool = False

@dataclass
class LiveOrder:
    """Represents a live trading order."""
    ticket: int
    symbol: str
    type: str  # BUY or SELL
    volume: float
    entry_price: float
    entry_time: datetime
    status: OrderStatus
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    profit_loss: float = 0.0
