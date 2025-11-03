"""Eden Trading System - Core Modules"""

from .backtest_engine import BacktestEngine, BacktestStats, Position, print_backtest_report
from .trading_bot import TradingBot, OrderStatus

__version__ = "1.0.0"
__all__ = [
    "BacktestEngine",
    "BacktestStats", 
    "Position",
    "TradingBot",
    "OrderStatus",
    "print_backtest_report",
]
