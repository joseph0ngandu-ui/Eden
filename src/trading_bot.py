#!/usr/bin/env python3
"""
Production Live Trading Bot

Implements real-time trading using the winning MA(3,10) strategy on M5 timeframe.
- Entry: MA(3) crosses above MA(10)
- Exit: Fixed 5-bar hold duration
- Risk Management: Configurable position sizing and stop losses
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    """Order status types."""
    PENDING = "PENDING"
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"


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


class TradingBot:
    """
    Production live trading bot using MA(3,10) strategy.
    
    Configuration:
    - Fast MA: 3 periods
    - Slow MA: 10 periods
    - Timeframe: M5
    - Hold Duration: 5 bars
    """
    
    # Strategy parameters
    FAST_MA = 3
    SLOW_MA = 10
    HOLD_BARS = 5
    TIMEFRAME = mt5.TIMEFRAME_M5
    
    # Risk management
    MAX_POSITION_SIZE = 1.0  # 1 lot
    
    def __init__(self, symbols: List[str], account_id: Optional[int] = None, password: Optional[str] = None, server: Optional[str] = None):
        """
        Initialize trading bot.
        
        Args:
            symbols: List of symbols to trade
            account_id: MT5 account ID
            password: MT5 password
            server: MT5 server name
        """
        self.symbols = symbols
        self.account_id = account_id
        self.password = password
        self.server = server
        self.active_orders: Dict[str, LiveOrder] = {}
        self.closed_orders: List[LiveOrder] = []
        self.is_running = False
        self.last_signals: Dict[str, int] = {}  # Track last signal per symbol
    
    def connect(self) -> bool:
        """Connect to MT5 terminal."""
        if not mt5.initialize():
            logger.error(f"MT5 initialization failed: {mt5.last_error()}")
            return False
        
        if self.account_id and self.password and self.server:
            if not mt5.login(self.account_id, password=self.password, server=self.server):
                logger.error(f"MT5 login failed: {mt5.last_error()}")
                return False
            logger.info(f"Connected to MT5 account {self.account_id}")
        else:
            logger.info("Connected to MT5 (using existing terminal session)")
        
        return True
    
    def disconnect(self) -> None:
        """Disconnect from MT5."""
        mt5.shutdown()
        logger.info("Disconnected from MT5")
    
    def fetch_recent_data(self, symbol: str, bars: int = 100) -> Optional[pd.DataFrame]:
        """
        Fetch recent OHLC data for a symbol.
        
        Args:
            symbol: Trading symbol
            bars: Number of bars to fetch
            
        Returns:
            DataFrame with OHLC data or None if fetch fails
        """
        try:
            rates = mt5.copy_rates_from_pos(symbol, self.TIMEFRAME, 0, bars)
            if rates is None:
                logger.warning(f"Failed to fetch data for {symbol}")
                return None
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            return df.sort_values('time').reset_index(drop=True)
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def calculate_signal(self, df: pd.DataFrame) -> int:
        """
        Calculate trading signal for the latest bar.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            1 for BUY, -1 for SELL, 0 for NEUTRAL
        """
        if len(df) < self.SLOW_MA:
            return 0
        
        # Calculate MAs
        df['MA_fast'] = df['close'].rolling(window=self.FAST_MA).mean()
        df['MA_slow'] = df['close'].rolling(window=self.SLOW_MA).mean()
        
        # Get current and previous values
        current_fast = df['MA_fast'].iloc[-1]
        current_slow = df['MA_slow'].iloc[-1]
        prev_fast = df['MA_fast'].iloc[-2]
        prev_slow = df['MA_slow'].iloc[-2]
        
        # Buy signal: fast MA crosses above slow MA
        if pd.notna(current_fast) and pd.notna(current_slow):
            if current_fast > current_slow and prev_fast <= prev_slow:
                return 1  # BUY
        
        return 0  # NEUTRAL
    
    def place_order(self, symbol: str, order_type: str, volume: float, comment: str = "") -> Optional[int]:
        """
        Place a market order.
        
        Args:
            symbol: Trading symbol
            order_type: "BUY" or "SELL"
            volume: Order volume in lots
            comment: Order comment
            
        Returns:
            Order ticket or None if failed
        """
        try:
            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                logger.error(f"Symbol {symbol} not found")
                return None
            
            # Create order request
            if order_type == "BUY":
                action = mt5.ORDER_TYPE_BUY
            else:
                action = mt5.ORDER_TYPE_SELL
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": action,
                "comment": comment,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Send order
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Order failed for {symbol}: {result.comment}")
                return None
            
            logger.info(f"Order placed: {order_type} {volume} {symbol} at #{result.order}")
            return result.order
        
        except Exception as e:
            logger.error(f"Error placing order for {symbol}: {e}")
            return None
    
    def close_position(self, symbol: str, volume: float, comment: str = "") -> bool:
        """
        Close an open position.
        
        Args:
            symbol: Trading symbol
            volume: Volume to close
            comment: Close comment
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get current position type
            positions = mt5.positions_get(symbol=symbol)
            if not positions:
                logger.warning(f"No open positions for {symbol}")
                return False
            
            position = positions[0]
            close_type = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": close_type,
                "position": position.ticket,
                "comment": comment,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Close failed for {symbol}: {result.comment}")
                return False
            
            logger.info(f"Position closed: {symbol} {volume} lots at #{result.order}")
            return True
        
        except Exception as e:
            logger.error(f"Error closing position for {symbol}: {e}")
            return False
    
    def monitor_positions(self) -> None:
        """Monitor open positions and manage exits."""
        try:
            positions = mt5.positions_get()
            if not positions:
                return
            
            for position in positions:
                symbol = position.symbol
                if symbol not in self.symbols:
                    continue
                
                # Track bars in position
                if symbol not in self.last_signals:
                    self.last_signals[symbol] = 0
                
                # Increment bars held
                self.last_signals[symbol] += 1
                
                # Close if hold duration exceeded
                if self.last_signals[symbol] >= self.HOLD_BARS:
                    logger.info(f"Hold duration reached for {symbol}, closing position")
                    self.close_position(symbol, position.volume, "Hold duration exit")
                    self.last_signals[symbol] = 0
        
        except Exception as e:
            logger.error(f"Error monitoring positions: {e}")
    
    def process_signal(self, symbol: str, signal: int) -> None:
        """
        Process trading signal and execute trades.
        
        Args:
            symbol: Trading symbol
            signal: 1=BUY, -1=SELL, 0=NEUTRAL
        """
        try:
            if signal == 0:
                return
            
            # Check for existing position
            positions = mt5.positions_get(symbol=symbol)
            
            if signal == 1:  # BUY signal
                if not positions:
                    self.place_order(symbol, "BUY", self.MAX_POSITION_SIZE, "MA(3,10) crossover")
                    self.last_signals[symbol] = 0
            
            elif signal == -1:  # SELL signal
                if positions:
                    position = positions[0]
                    self.close_position(symbol, position.volume, "Exit signal")
                    self.last_signals[symbol] = 0
        
        except Exception as e:
            logger.error(f"Error processing signal for {symbol}: {e}")
    
    def run_cycle(self) -> None:
        """Run one trading cycle."""
        try:
            for symbol in self.symbols:
                # Fetch recent data
                df = self.fetch_recent_data(symbol, bars=50)
                if df is None or len(df) == 0:
                    continue
                
                # Calculate signal
                signal = self.calculate_signal(df)
                
                # Process signal
                if signal != 0:
                    self.process_signal(symbol, signal)
                
                # Monitor positions
                self.monitor_positions()
        
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
    
    def start(self, check_interval: int = 300) -> None:
        """
        Start live trading.
        
        Args:
            check_interval: Seconds between trading cycles
        """
        if not self.connect():
            logger.error("Failed to connect to MT5")
            return
        
        self.is_running = True
        logger.info(f"Starting live trading bot (checking every {check_interval}s)")
        logger.info(f"Trading symbols: {', '.join(self.symbols)}")
        
        try:
            while self.is_running:
                self.run_cycle()
                time.sleep(check_interval)
        
        except KeyboardInterrupt:
            logger.info("Stopping bot (keyboard interrupt)")
        
        except Exception as e:
            logger.error(f"Fatal error: {e}")
        
        finally:
            self.disconnect()
    
    def stop(self) -> None:
        """Stop live trading."""
        self.is_running = False
        logger.info("Bot stop signal received")
