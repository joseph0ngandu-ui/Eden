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

from config_loader import ConfigLoader
from trade_journal import TradeJournal
from health_monitor import HealthMonitor, RiskManager, HealthStatus, RiskLevel
from volatility_adapter import VolatilityAdapter
from risk_ladder import RiskLadder, PositionSizer, RiskTier
from exit_logic import ExitManager, ExitConfig  # v1.2: Advanced exit logic

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
    
    def __init__(self, symbols: List[str], account_id: Optional[int] = None, password: Optional[str] = None, server: Optional[str] = None, config_path: Optional[str] = None):
        """
        Initialize trading bot.
        
        Args:
            symbols: List of symbols to trade
            account_id: MT5 account ID
            password: MT5 password
            server: MT5 server name
            config_path: Path to strategy.yml configuration
        """
        # Load configuration
        self.config = ConfigLoader(config_path)
        strategy_params = self.config.get_strategy_params()
        risk_config = self.config.get_risk_management()
        
        # Strategy parameters (loaded from config)
        self.FAST_MA = strategy_params['fast_ma_period']
        self.SLOW_MA = strategy_params['slow_ma_period']
        self.HOLD_BARS = strategy_params['hold_bars']
        self.TIMEFRAME = mt5.TIMEFRAME_M5
        self.MAX_POSITION_SIZE = risk_config['position_size']
        
        # Initialize components
        self.symbols = symbols or self.config.get_trading_symbols()
        self.account_id = account_id
        self.password = password
        self.server = server

        # Modes
        self.shadow_mode = bool(os.getenv('EDEN_SHADOW', '0') == '1') or bool(self.config.get_parameter('development.simulation_mode', False))
        
        # Trade journal
        self.trade_journal = TradeJournal(log_dir="logs")
        
        # Health monitoring and risk management
        self.health_monitor = HealthMonitor(
            max_drawdown_percent=risk_config.get('max_drawdown_percent'),
            check_interval=60,
            health_check_callback=self._on_health_change
        )
        self.risk_manager = RiskManager(
            max_position_size=self.MAX_POSITION_SIZE,
            max_concurrent_positions=risk_config['max_concurrent_positions'],
            max_daily_loss_percent=risk_config.get('max_daily_loss_percent')
        )
        
        # Volatility adaptation
        self.volatility_adapter = VolatilityAdapter(base_hold_bars=self.HOLD_BARS)

        # External order bridge (API → bot via file queue)
        self.order_queue_path = Path(__file__).resolve().parent.parent / 'logs' / 'order_queue.jsonl'
        
        # v1.2: Advanced exit logic
        exit_config = ExitConfig(
            min_hold_bars=3,
            max_hold_bars=4,
            breakeven_move_ratio=0.8,
            min_reward_ratio=1.5,
            max_reward_ratio=2.0,
            atr_period=14,
            trailing_stop_enable=True,
            use_momentum_exit=True
        )
        self.exit_manager = ExitManager(config=exit_config)
        logger.info("Exit Logic v2 enabled: Adaptive holds, trailing stops, dynamic TP")
        
        # Growth mode & Risk Ladder
        growth_config = self.config.get_growth_mode_config()
        self.growth_mode_enabled = growth_config['enabled']
        self.risk_ladder: Optional[RiskLadder] = None
        self.position_sizer: Optional[PositionSizer] = None
        if self.growth_mode_enabled:
            self.risk_ladder = RiskLadder(
                initial_balance=self.initial_balance or 100.0,
                growth_mode_enabled=True,
                high_aggression_below=growth_config['high_aggression_below'],
                equity_step_size=growth_config['equity_step_size'],
                equity_step_drawdown_limit=growth_config['equity_step_drawdown_limit'],
            )
            self.position_sizer = PositionSizer(self.risk_ladder, pip_value=growth_config['pip_value'])
            logger.info(f"Growth Mode enabled: {self.risk_ladder.current_tier.tier.value}")
        
        # Trading state
        self.active_orders: Dict[str, LiveOrder] = {}
        self.closed_orders: List[LiveOrder] = []
        self.is_running = False
        self.last_signals: Dict[str, int] = {}  # Track last signal per symbol
        self.initial_balance = 0.0
        
        # Log header with version and parameters
        self._log_startup_banner()
    
    def _log_startup_banner(self) -> None:
        """Log startup banner with version and configuration."""
        version = self.config.get_version()
        banner = f"\n{'='*80}\n"
        banner += f"Eden v{version} - MA({self.FAST_MA},{self.SLOW_MA}) Strategy | M5 Timeframe\n"
        banner += f"HOLD={self.HOLD_BARS} bars | Symbols={len(self.symbols)}\n"
        banner += f"Risk Cap: {self.health_monitor.max_drawdown_percent}% | Max Positions: {self.risk_manager.max_concurrent_positions}\n"
        banner += f"{'='*80}\n"
        logger.info(banner)
    
    def _on_health_change(self, health_status: HealthStatus, risk_level: RiskLevel) -> None:
        """Callback for health status changes."""
        logger.warning(f"[WARN]️ Health Status Changed: {health_status.value} | Risk: {risk_level.value}")
        
        if health_status == HealthStatus.UNHEALTHY:
            logger.error("[ERROR] System UNHEALTHY - Pausing trades")
        elif health_status == HealthStatus.DEGRADED:
            logger.warning("[WARN]️ System DEGRADED - Monitor closely")
        else:
            logger.info("[OK] System HEALTHY")
        
        if risk_level == RiskLevel.CRITICAL:
            logger.error("[STOP] CRITICAL RISK - Auto-disabling live trading")
            self.is_running = False
    
    def connect(self) -> bool:
        """Connect to MT5 terminal.

        Tries an explicit terminal path first (env/known path), then falls back to the
        default MetaTrader5 discovery so we don't get spurious 'x64 not found' errors.
        """
        # Prefer explicit path if provided
        mt5_path = os.getenv("MT5_PATH", r"C:\Program Files\MetaTrader 5 Terminal\terminal64.exe")
        initialized = False

        # Try with explicit path first
        try:
            logger.info(f"Attempting to connect to MT5 at: {mt5_path}")
            initialized = mt5.initialize(path=mt5_path)
        except Exception as e:
            logger.error(f"Explicit path initialization failed: {e}")
            initialized = False

        # Fallback to default discovery if needed
        if not initialized:
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
        
        # Get initial balance
        account_info = mt5.account_info()
        if account_info:
            self.initial_balance = account_info.balance
            self.health_monitor.peak_balance = self.initial_balance
            logger.info(f"Account Balance: {self.initial_balance:.2f}")
            
            # Initialize Risk Ladder with actual balance if growth mode enabled
            if self.growth_mode_enabled and self.risk_ladder is None:
                growth_config = self.config.get_growth_mode_config()
                self.risk_ladder = RiskLadder(
                    initial_balance=self.initial_balance,
                    growth_mode_enabled=True,
                    high_aggression_below=growth_config['high_aggression_below'],
                    equity_step_size=growth_config['equity_step_size'],
                    equity_step_drawdown_limit=growth_config['equity_step_drawdown_limit'],
                )
                self.position_sizer = PositionSizer(self.risk_ladder, pip_value=growth_config['pip_value'])
                logger.info(f"Risk Ladder initialized: {self.risk_ladder.current_tier.tier.value} | ${self.initial_balance:.2f}")
        
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
    
    def place_order(self, symbol: str, order_type: str, volume: float = None, comment: str = "", atr: float = None, strategy_name: str = None, strategy_params: Dict = None) -> Optional[int]:
        """
        Place a market order with dynamic position sizing.
        
        Args:
            symbol: Trading symbol
            order_type: "BUY" or "SELL"
            volume: Order volume in lots (if None, calculated dynamically)
            comment: Order comment
            atr: ATR value for position sizing (optional)
            
        Returns:
            Order ticket or None if failed
        """
        try:
            # Calculate dynamic position size if not provided
            if volume is None or self.growth_mode_enabled:
                if self.position_sizer and self.health_monitor:
                    sizing = self.position_sizer.calculate(
                        equity=self.health_monitor.current_balance,
                        atr=atr
                    )
                    volume = sizing['lot_size']
                    logger.debug(f"Dynamic sizing: {volume}L (tier: {sizing['tier']}, risk: {sizing['risk_pct']:.1f}%)")
                else:
                    volume = self.MAX_POSITION_SIZE
            
            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                logger.error(f"Symbol {symbol} not found")
                return None
            
            # Shadow/paper mode: log and skip live execution
            price_tick = mt5.symbol_info_tick(symbol)
            price = None
            if price_tick:
                price = price_tick.ask if order_type == "BUY" else price_tick.bid

            if self.shadow_mode or (self.health_monitor and not self.health_monitor.trading_enabled):
                self.trade_journal.add_trade(
                    symbol=symbol,
                    trade_type=order_type,
                    entry_price=price or 0.0,
                    entry_time=datetime.utcnow(),
                    volume=volume,
                    notes=f"PAPER | {comment}",
                    metadata={
                        'strategy_name': strategy_name or 'unknown',
                        'strategy_params': strategy_params or {},
                        'mode': 'PAPER'
                    }
                )
                self.trade_journal.export_csv()
                logger.info(f"Paper trade logged: {order_type} {volume}L {symbol}")
                return 0

            # Create order request with adaptive filling
            if order_type == "BUY":
                action = mt5.ORDER_TYPE_BUY
            else:
                action = mt5.ORDER_TYPE_SELL
            
            base_request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": action,
                "price": price,
                "deviation": 20,
                "comment": comment,
            }

            tried = []
            fill_modes = [
                getattr(mt5, 'ORDER_FILLING_FOK', None),
                getattr(mt5, 'ORDER_FILLING_IOC', None),
                getattr(mt5, 'ORDER_FILLING_RETURN', None),
            ]

            result = None
            for fm in [m for m in fill_modes if m is not None]:
                req = dict(base_request)
                req["type_filling"] = fm
                result = mt5.order_send(req)
                tried.append((fm, result.retcode, result.comment))
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    break
            
            if not result or result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Order failed for {symbol}: {result.comment if result else 'NO_RESULT'} | tried={tried}")
                return None
            
            # Log live order
            self.trade_journal.add_trade(
                symbol=symbol,
                trade_type=order_type,
                entry_price=price or 0.0,
                entry_time=datetime.utcnow(),
                volume=volume,
                notes=f"LIVE | ticket={result.order} | {comment}",
                metadata={
                    'strategy_name': strategy_name or 'unknown',
                    'strategy_params': strategy_params or {},
                    'mode': 'LIVE',
                    'ticket': result.order,
                }
            )
            self.trade_journal.export_csv()

            logger.info(f"Order placed: {order_type} {volume}L {symbol} ticket #{result.order}")
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
        """Process trading signal and execute trades.

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
    
    def _process_external_orders(self) -> None:
        """Process any orders enqueued by the API via the file-based bridge."""
        try:
            if not self.order_queue_path.exists():
                return

            # Read and then truncate the queue file (simple at-most-once semantics)
            with open(self.order_queue_path, 'r', encoding='utf-8') as f:
                lines = [ln.strip() for ln in f.readlines() if ln.strip()]

            if not lines:
                return

            # Clear the file immediately so we don't re-run the same commands
            with open(self.order_queue_path, 'w', encoding='utf-8'):
                pass

            for line in lines:
                try:
                    cmd = json.loads(line)
                except Exception:
                    continue

                if cmd.get('type') == 'test_order':
                    sym = cmd.get('symbol', 'Volatility 75 Index')
                    side = cmd.get('side', 'BUY')
                    vol = float(cmd.get('volume', 0.01))
                    comment = f"API-bridge test order ({side})"
                    logger.info("Processing external test order from queue: %s %s %s", sym, side, vol)
                    self.place_order(sym, side, volume=vol, comment=comment)
        except Exception as e:
            logger.error(f"Error processing external orders: {e}")

    def run_cycle(self) -> None:
        """Run one trading cycle with health monitoring."""
        try:
            # Health check
            self.health_monitor.check_health()
            
            # Check if trading is disabled
            if not self.health_monitor.trading_enabled:
                logger.warning("[WARN]️ Trading disabled due to risk level")
                return
            
            # Update account balance
            account_info = mt5.account_info()
            if account_info:
                self.health_monitor.update_balance(account_info.balance)
                
                # Update risk ladder if growth mode enabled
                if self.risk_ladder:
                    self.risk_ladder.update_balance(account_info.balance)
            
            # Process any external orders queued by the API
            self._process_external_orders()

            for symbol in self.symbols:
                # Fetch recent data
                df = self.fetch_recent_data(symbol, bars=50)
                if df is None or len(df) == 0:
                    continue
                
                # Get volatility metrics
                atr = self.volatility_adapter.calculate_atr(df).iloc[-1]
                std_dev = self.volatility_adapter.calculate_std_dev(df).iloc[-1]
                volatility_metrics = self.volatility_adapter.classify_volatility(atr, std_dev, df)
                
                # Get adaptive parameters
                adaptive_hold = self.volatility_adapter.get_adaptive_hold_duration(volatility_metrics)
                adaptive_fast_ma, adaptive_slow_ma = self.volatility_adapter.get_adaptive_ma_params(volatility_metrics)
                
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
        Start live trading with health monitoring and trade journaling.
        
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
            # Export trade journal on shutdown
            self.trade_journal.export_csv()
            self.trade_journal.print_summary()
            
            # Print health status
            self.health_monitor.print_status()
            
            # Print Risk Ladder status
            if self.risk_ladder:
                self.risk_ladder.print_status()
            
            self.disconnect()
    
    def stop(self) -> None:
        """Stop live trading."""
        self.is_running = False
        logger.info("Bot stop signal received")

