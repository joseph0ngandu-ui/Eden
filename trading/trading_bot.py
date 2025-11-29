#!/usr/bin/env python3
"""
Production Live Trading Bot

Implements real-time trading using the winning MA(3,10) strategy on M5 timeframe.
- Entry: MA(3) crosses above MA(10)
- Exit: Fixed 5-bar hold duration
- Risk Management: Configurable position sizing and stop losses
- Health Monitoring: MT5 API and internet connectivity checks
- Trade Journaling: Automatic CSV export to logs/trade_history.csv
- Volatility Adaptation: Adaptive parameters based on market conditions
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
import os
from pathlib import Path
import json

from .config_loader import ConfigLoader
from .trade_journal import TradeJournal
from .health_monitor import HealthMonitor, RiskManager, HealthStatus, RiskLevel
from .volatility_adapter import VolatilityAdapter
from .risk_ladder import RiskLadder, PositionSizer, RiskTier
from .exit_logic import ExitManager, ExitConfig  # v1.2: Advanced exit logic

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


from trading.models import LiveOrder, OrderStatus, Trade, Position


class TradingBot:
    """
    Production live trading bot using Volatility Burst v1.3 strategy.
    
    Configuration:
    - Strategy: Volatility Burst (Squeeze + Breakout)
    - Timeframe: M5
    - Risk Management: ATR-based dynamic sizing
    """
    
    TIMEFRAME = mt5.TIMEFRAME_M5
    
    def __init__(self, symbols: List[str], account_id: Optional[int] = None, password: Optional[str] = None, server: Optional[str] = None, config_path: Optional[str] = None):
        """Initialize trading bot."""
        # Load configuration
        self.config = ConfigLoader(config_path)
        risk_config = self.config.get_risk_management()
        
        # Initialize Strategies
        from trading.volatility_burst_enhanced import VolatilityBurst
        from trading.pro_strategies import ProStrategyEngine
        
        self.strategies = []
        
        # Strategy 1: Volatility Burst (VIX-specific)
        self.volatility_burst = VolatilityBurst(config_path="config/volatility_burst.yml")
        self.strategies.append(self.volatility_burst)
        
        # Strategy 2: Pro Multi-Strategy Engine (Prop Firm Certified)
        self.pro_strategies = ProStrategyEngine()
        self.strategies.append(self.pro_strategies)
        
        # Initialize components
        self.symbols = symbols or self.config.get_trading_symbols()
        self.account_id = account_id
        self.password = password
        self.server = server

        # Modes
        self.shadow_mode = bool(os.getenv('EDEN_SHADOW', '0') == '1')
        
        # Trade journal
        self.trade_journal = TradeJournal(log_dir="logs")
        
        # Health monitoring
        self.health_monitor = HealthMonitor(
            max_drawdown_percent=risk_config.get('max_drawdown_percent'),
            check_interval=60,
            health_check_callback=self._on_health_change
        )
        
        # Risk Manager (Daily Loss Limit)
        self.risk_manager = RiskManager(
            max_position_size=risk_config.get('max_position_size', 1.0),
            max_concurrent_positions=risk_config.get('max_positions', 7),
            max_daily_loss_percent=risk_config.get('max_daily_loss_percent', 2.0)
        )
        
        # External order bridge
        self.order_queue_path = Path(__file__).resolve().parent.parent / 'logs' / 'order_queue.jsonl'
        
        self.is_running = False
        self.initial_balance = 0.0
        
        self._log_startup_banner()
    
    def _log_startup_banner(self) -> None:
        """Log startup banner."""
        banner = f"\n{'='*80}\n"
        banner += f"Eden Live Bot - Hybrid Aggressive (Pro + VIX)\n"
        banner += f"Symbols={len(self.symbols)} | Shadow Mode={self.shadow_mode}\n"
        banner += f"Max Positions=7 | Risk=0.15% | Target=16% monthly\n"
        banner += f"{'='*80}\n"
        logger.info(banner)
    
    def _on_health_change(self, health_status: HealthStatus, risk_level: RiskLevel) -> None:
        """Callback for health status changes."""
        if risk_level == RiskLevel.CRITICAL:
            logger.error("[STOP] CRITICAL RISK - Auto-disabling live trading")
            self.is_running = False
    
    def connect(self) -> bool:
        """Connect to MT5 terminal."""
        mt5_path = os.getenv("MT5_PATH", r"C:\Program Files\MetaTrader 5 Terminal\terminal64.exe")
        if not mt5.initialize(path=mt5_path):
            if not mt5.initialize():
                logger.error(f"MT5 initialization failed: {mt5.last_error()}")
                return False
        
        if self.account_id and self.password and self.server:
            if not mt5.login(self.account_id, password=self.password, server=self.server):
                logger.error(f"MT5 login failed: {mt5.last_error()}")
                return False
        
        account_info = mt5.account_info()
        if account_info:
            self.initial_balance = account_info.balance
            self.health_monitor.peak_balance = self.initial_balance
            logger.info(f"Connected. Balance: {self.initial_balance:.2f}")
        
        return True
    
    def disconnect(self) -> None:
        mt5.shutdown()
        logger.info("Disconnected from MT5")
    
    def fetch_recent_data(self, symbol: str, bars: int = 100) -> Optional[pd.DataFrame]:
        """Fetch recent OHLC data."""
        try:
            rates = mt5.copy_rates_from_pos(symbol, self.TIMEFRAME, 0, bars)
            if rates is None:
                return None
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            return df.sort_values('time').reset_index(drop=True)
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None

    def place_order(self, trade_signal) -> bool:
        """Place order based on Trade object."""
        try:
            # Check Daily Loss Limit
            if self.risk_manager.is_daily_loss_limit_reached(self.initial_balance):
                logger.warning(f"SKIPPING TRADE: Daily loss limit reached ({self.risk_manager.daily_pnl:.2f})")
                return False

            symbol = trade_signal.symbol
            volume = 0.01 # Fixed for now, or use dynamic sizing
            
            # Shadow mode check
            if self.shadow_mode:
                logger.info(f"PAPER TRADE: {trade_signal.direction} {symbol} @ {trade_signal.entry_price} ({trade_signal.strategy})")
                if trade_signal.strategy == "VolatilityBurst":
                    self.volatility_burst.on_trade_open(trade_signal)
                elif trade_signal.strategy.startswith("ICT"):
                    self.ict_bot.on_trade_open(trade_signal)
                return True

            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                logger.error(f"Symbol {symbol} not found")
                return False
            
            action = mt5.ORDER_TYPE_BUY if trade_signal.direction == "LONG" else mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(symbol).ask if action == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).bid
            
            base_request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": action,
                "price": price,
                "sl": trade_signal.sl,
                "tp": trade_signal.tp,
                "deviation": 20,
                "deviation": 20,
                "comment": f"{trade_signal.strategy} {trade_signal.confidence:.2f}",
            }
            
            # Try different filling modes
            # Note: Some brokers require specific filling modes
            fill_modes = [
                mt5.ORDER_FILLING_FOK,
                mt5.ORDER_FILLING_IOC,
                mt5.ORDER_FILLING_RETURN,
                0 # Default/Unspecified
            ]
            
            result = None
            for mode in fill_modes:
                req = dict(base_request)
                req["type_filling"] = mode
                result = mt5.order_send(req)
                
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    break
                elif result.retcode == mt5.TRADE_RETCODE_INVALID_FILL or result.retcode == mt5.TRADE_RETCODE_UNSUPPORTED_FILLING:
                    continue # Try next mode
            
            if not result or result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Order failed: {result.comment if result else 'Unknown'} (Retcode: {result.retcode if result else 'None'})")
                return False
                
            logger.info(f"LIVE TRADE: {trade_signal.direction} {symbol} #{result.order}")
            logger.info(f"LIVE TRADE: {trade_signal.direction} {symbol} #{result.order}")
            
            if trade_signal.strategy == "VolatilityBurst":
                self.volatility_burst.on_trade_open(trade_signal)
            elif trade_signal.strategy.startswith("Pro_"):
                self.pro_strategies.on_trade_open(trade_signal)
            
            # Log to journal
            self.trade_journal.add_trade(
                symbol=symbol,
                trade_type=trade_signal.direction,
                entry_price=result.price,
                entry_time=datetime.utcnow(),
                volume=volume,
                notes=f"Ticket: {result.order} | Conf: {trade_signal.confidence:.2f}",
                metadata={'strategy': trade_signal.strategy, 'ticket': result.order}
            )
            self.trade_journal.export_csv()
            return True
            
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return False

    def manage_positions(self):
        """Manage open positions using strategy logic."""
        for strategy in self.strategies:
            for symbol in list(strategy.open_positions.keys()):
                df = self.fetch_recent_data(symbol, bars=50)
                if df is None: continue
                
                actions = strategy.manage_position(df, symbol)
                for action in actions:
                    if action['action'] == 'close':
                        self.close_position(symbol, action.get('price'), action.get('reason'))
                    elif action['action'] == 'trail_stop':
                        self.modify_position(symbol, action.get('new_sl'))

    def close_position(self, symbol: str, price: float = None, reason: str = "") -> bool:
        """Close position in MT5."""
        if self.shadow_mode:
            logger.info(f"PAPER CLOSE: {symbol} ({reason})")
            return True
            
        positions = mt5.positions_get(symbol=symbol)
        if not positions: return False
        
        pos = positions[0]
        type_close = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        price = price or (mt5.symbol_info_tick(symbol).bid if type_close == mt5.ORDER_TYPE_SELL else mt5.symbol_info_tick(symbol).ask)
        
        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": pos.volume,
            "type": type_close,
            "position": pos.ticket,
            "price": price,
            "comment": reason
        }
        result = mt5.order_send(req)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"Closed {symbol}: {reason}")
            
            # Calculate realized PnL (approximate)
            # In production, fetch actual deal profit from history
            try:
                deal_profit = result.profit if hasattr(result, 'profit') else 0.0
                # Fallback: Estimate based on price diff if profit not returned immediately
                # For now, we rely on history or balance update in next cycle
                pass
            except:
                pass
                
            return True
        return False

    def modify_position(self, symbol: str, new_sl: float) -> bool:
        """Modify SL in MT5."""
        if self.shadow_mode:
            logger.info(f"PAPER TRAIL: {symbol} SL->{new_sl}")
            return True
            
        positions = mt5.positions_get(symbol=symbol)
        if not positions: return False
        pos = positions[0]
        
        req = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": pos.ticket,
            "sl": new_sl,
            "tp": pos.tp
        }
        result = mt5.order_send(req)
        return result.retcode == mt5.TRADE_RETCODE_DONE

    def run_cycle(self) -> None:
        """Run one trading cycle."""
        self.health_monitor.check_health()
        if not self.health_monitor.trading_enabled: return
        
        # Reset daily counters if new day (simplified check)
        # self.strategy.reset_daily_trades() # Implement proper day check
        
        for symbol in self.symbols:
            df = self.fetch_recent_data(symbol, bars=100)
            if df is None: continue
            
            # 1. Manage existing positions
            self.manage_positions()
            
            # 2. Check for new entries
            for strategy in self.strategies:
                trade_signal = strategy.evaluate_live(df, symbol)
                if trade_signal:
                    self.place_order(trade_signal)

    def start(self, check_interval: int = 60) -> None:
        """Start the bot."""
        if not self.connect(): return
        self.is_running = True
        logger.info(f"Started. Polling every {check_interval}s")
        
        try:
            while self.is_running:
                self.run_cycle()
                time.sleep(check_interval)
        except KeyboardInterrupt:
            logger.info("Stopping...")
        finally:
            self.disconnect()
