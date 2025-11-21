#!/usr/bin/env python3
"""
Live Health Monitor & Risk Manager

Lightweight watchdog that:
1. Checks MT5 API connectivity
2. Monitors internet connection
3. Tracks drawdown and auto-disables trading when max drawdown is breached
4. Pauses trading on disconnection and resumes on reconnection
"""

import socket
import logging
import time
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, Callable
from enum import Enum
import MetaTrader5 as mt5

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status types."""
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNHEALTHY = "UNHEALTHY"


class RiskLevel(Enum):
    """Risk level types."""
    SAFE = "SAFE"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class HealthMonitor:
    """Monitor system health and manage risk."""
    
    def __init__(
        self,
        max_drawdown_percent: float = None,
        check_interval: int = 60,
        health_check_callback: Callable = None
    ):
        """
        Initialize health monitor.
        
        Args:
            max_drawdown_percent: Maximum allowed drawdown percentage (e.g., 10.0 for 10%)
            check_interval: Seconds between health checks
            health_check_callback: Function to call on health changes (health_status, risk_level)
        """
        self.max_drawdown_percent = max_drawdown_percent
        self.check_interval = check_interval
        self.health_check_callback = health_check_callback
        
        # Health tracking
        self.mt5_connected = False
        self.internet_connected = False
        self.health_status = HealthStatus.HEALTHY
        self.risk_level = RiskLevel.SAFE
        self.trading_enabled = True
        
        # Risk tracking
        self.peak_balance = 0.0
        self.current_balance = 0.0
        self.drawdown_percent = 0.0
        self.last_check_time = datetime.now()
        self.disconnection_time = None
        
        # Statistics
        self.mt5_check_count = 0
        self.internet_check_count = 0
        self.failures = {
            'mt5_failures': 0,
            'internet_failures': 0,
        }
    
    def check_internet(self, host: str = "8.8.8.8", port: int = 53, timeout: int = 3) -> bool:
        """
        Check internet connectivity.
        
        Args:
            host: DNS server to check (default: Google DNS)
            port: Port to check
            timeout: Timeout in seconds
            
        Returns:
            True if internet is available, False otherwise
        """
        try:
            socket.setdefaulttimeout(timeout)
            socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
            self.internet_connected = True
            return True
        except (socket.timeout, socket.error):
            self.internet_connected = False
            return False
        finally:
            socket.setdefaulttimeout(None)
    
    def check_mt5_connection(self) -> bool:
        """
        Check MT5 API connectivity.
        
        Returns:
            True if MT5 is connected, False otherwise
        """
        try:
            # Check if terminal info can be retrieved
            info = mt5.terminal_info()
            if info is not None:
                self.mt5_connected = True
                return True
        except Exception:
            pass
        
        self.mt5_connected = False
        return False
    
    def update_balance(self, current_balance: float) -> None:
        """
        Update current balance and track drawdown.
        
        Args:
            current_balance: Current account balance
        """
        self.current_balance = current_balance
        
        # Update peak balance
        if current_balance > self.peak_balance:
            self.peak_balance = current_balance
        
        # Calculate drawdown
        if self.peak_balance > 0:
            self.drawdown_percent = ((self.peak_balance - current_balance) / self.peak_balance) * 100
        
        # Check if max drawdown breached
        self._check_risk_level()
    
    def _check_risk_level(self) -> None:
        """Check current risk level based on drawdown."""
        prev_risk = self.risk_level
        
        if self.max_drawdown_percent is None:
            self.risk_level = RiskLevel.SAFE
            self.trading_enabled = True
        elif self.drawdown_percent >= self.max_drawdown_percent:
            self.risk_level = RiskLevel.CRITICAL
            self.trading_enabled = False
            logger.warning(f"⚠️ CRITICAL: Max drawdown reached: {self.drawdown_percent:.2f}% >= {self.max_drawdown_percent:.2f}%")
        elif self.drawdown_percent >= (self.max_drawdown_percent * 0.8):
            self.risk_level = RiskLevel.WARNING
            self.trading_enabled = True
            logger.warning(f"⚠️ WARNING: Drawdown approaching limit: {self.drawdown_percent:.2f}% / {self.max_drawdown_percent:.2f}%")
        else:
            self.risk_level = RiskLevel.SAFE
            self.trading_enabled = True
        
        # Callback on risk level change
        if prev_risk != self.risk_level and self.health_check_callback:
            self.health_check_callback(self.health_status, self.risk_level)
    
    def _update_health_status(self) -> None:
        """Update overall health status."""
        prev_status = self.health_status
        
        if self.mt5_connected and self.internet_connected:
            self.health_status = HealthStatus.HEALTHY
        elif self.mt5_connected or self.internet_connected:
            self.health_status = HealthStatus.DEGRADED
        else:
            self.health_status = HealthStatus.UNHEALTHY
        
        # Callback on health status change
        if prev_status != self.health_status and self.health_check_callback:
            self.health_check_callback(self.health_status, self.risk_level)
    
    def check_health(self) -> bool:
        """
        Run full health check.
        
        Returns:
            True if healthy, False otherwise
        """
        self.mt5_check_count += 1
        self.internet_check_count += 1
        
        # Check MT5
        if not self.check_mt5_connection():
            self.failures['mt5_failures'] += 1
            logger.warning("MT5 connection check failed")
        
        # Check internet
        if not self.check_internet():
            self.failures['internet_failures'] += 1
            logger.warning("Internet connectivity check failed")
            if self.disconnection_time is None:
                self.disconnection_time = datetime.now()
        else:
            if self.disconnection_time is not None:
                reconnect_duration = datetime.now() - self.disconnection_time
                logger.info(f"✓ Reconnected after {reconnect_duration.total_seconds():.1f}s")
                self.disconnection_time = None
        
        # Update health status
        self._update_health_status()
        self.last_check_time = datetime.now()
        
        return self.health_status == HealthStatus.HEALTHY
    
    def get_status(self) -> Dict:
        """
        Get current health and risk status.
        
        Returns:
            Dictionary with status information
        """
        return {
            'health_status': self.health_status.value,
            'risk_level': self.risk_level.value,
            'trading_enabled': self.trading_enabled,
            'mt5_connected': self.mt5_connected,
            'internet_connected': self.internet_connected,
            'current_balance': self.current_balance,
            'peak_balance': self.peak_balance,
            'drawdown_percent': round(self.drawdown_percent, 2),
            'max_drawdown_percent': self.max_drawdown_percent,
            'last_check': self.last_check_time.isoformat(),
            'disconnection_duration': (datetime.now() - self.disconnection_time).total_seconds() if self.disconnection_time else 0,
        }
    
    def print_status(self) -> None:
        """Print current status to logger."""
        status = self.get_status()
        
        logger.info(f"\n{'='*60}")
        logger.info(f"HEALTH MONITOR STATUS")
        logger.info(f"{'='*60}")
        logger.info(f"Overall Health: {status['health_status']}")
        logger.info(f"Risk Level: {status['risk_level']}")
        logger.info(f"Trading Enabled: {status['trading_enabled']}")
        logger.info(f"MT5 Connected: {status['mt5_connected']}")
        logger.info(f"Internet Connected: {status['internet_connected']}")
        logger.info(f"Current Balance: {status['current_balance']:.2f}")
        logger.info(f"Peak Balance: {status['peak_balance']:.2f}")
        logger.info(f"Drawdown: {status['drawdown_percent']:.2f}% / {status['max_drawdown_percent']}%")
        if status['disconnection_duration'] > 0:
            logger.info(f"Disconnection Duration: {status['disconnection_duration']:.1f}s")
        logger.info(f"{'='*60}\n")


class RiskManager:
    """Manages trading risk and position limits."""
    
    def __init__(
        self,
        max_position_size: float = 1.0,
        max_concurrent_positions: int = 10,
        max_daily_loss_percent: float = None,
    ):
        """
        Initialize risk manager.
        
        Args:
            max_position_size: Maximum position size in lots
            max_concurrent_positions: Maximum concurrent open positions
            max_daily_loss_percent: Maximum daily loss percentage
        """
        self.max_position_size = max_position_size
        self.max_concurrent_positions = max_concurrent_positions
        self.max_daily_loss_percent = max_daily_loss_percent
        
        self.daily_pnl = 0.0
        self.daily_reset_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        self.current_positions = {}
    

    def can_open_position(self, symbol: str, current_positions_count: int = 0) -> bool:
        """
        Check if a new position can be opened.
        
        Args:
            symbol: Trading symbol
            current_positions_count: Current number of open positions
            
        Returns:
            True if position can be opened, False otherwise
        """
        if current_positions_count >= self.max_concurrent_positions:
            logger.warning(f"Cannot open position for {symbol}: max concurrent positions ({self.max_concurrent_positions}) reached")
            return False
        
        return True
    
    def get_position_size(self, available_margin: float, symbol: str = None) -> float:
        """
        Calculate safe position size.
        
        Args:
            available_margin: Available margin
            symbol: Trading symbol (optional)
            
        Returns:
            Safe position size in lots
        """
        return min(self.max_position_size, available_margin / 100)
    
    def update_daily_pnl(self, pnl: float) -> None:
        """
        Update daily PnL.
        
        Args:
            pnl: Trade PnL
        """
        # Reset if new day
        now = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        if now > self.daily_reset_time:
            self.daily_pnl = 0.0
            self.daily_reset_time = now
        
        self.daily_pnl += pnl
    
    def is_daily_loss_limit_reached(self, initial_balance: float) -> bool:
        """
        Check if daily loss limit is reached.
        
        Args:
            initial_balance: Initial balance
            
        Returns:
            True if limit is reached, False otherwise
        """
        if self.max_daily_loss_percent is None:
            return False
        
        max_loss = (initial_balance * self.max_daily_loss_percent) / 100
        return self.daily_pnl <= -max_loss