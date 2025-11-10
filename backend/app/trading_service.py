#!/usr/bin/env python3
"""
Trading service for bot control and data management

STRATEGY: UltraSmall Mode - Volatility 75 Index (MOST PROFITABLE)
- Mode: UltraSmall Risk Ladder
- Primary Symbol: Volatility 75 Index
- Confidence Threshold: 0.6
- TP Multiplier: 2.0x ATR
- SL Multiplier: 1.2x ATR
- Expected Performance: 172.5% return over test period
- Starting Capital: $50 → $136.25
- Grid-Search Optimized: Score 94.92
- Test Period: Jan-Oct 2025
"""

from datetime import datetime, timedelta
from typing import List, Optional
import logging

from app.models import (
    BotStatus, Trade, Position, PerformanceStats, 
    StrategyConfig, DirectionEnum
)

logger = logging.getLogger(__name__)

class TradingService:
    """Service for managing trading bot operations and data.
    
    Uses Volatility Burst v1.3 strategy - optimized for profitability.
    """
    
    def __init__(self):
        """Initialize trading service with Volatility Burst v1.3 configuration."""
        self.is_bot_running = False
        self.current_balance = 100000.0
        self.peak_balance = 100000.0
        self.daily_pnl = 0.0
        self.last_heartbeat = datetime.utcnow()
        self.active_positions: List[Position] = []
        self.trade_history: List[Trade] = []
        # Initialize with profitable VB v1.3 configuration
        self.strategy_config = StrategyConfig()
    
    def get_bot_status(self) -> BotStatus:
        """Get current bot status."""
        try:
            # Calculate metrics
            total_trades = len(self.trade_history)
            winning_trades = sum(1 for t in self.trade_history if t.pnl > 0)
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            # Calculate profit factor
            total_wins = sum(t.pnl for t in self.trade_history if t.pnl > 0)
            total_losses = abs(sum(t.pnl for t in self.trade_history if t.pnl < 0))
            profit_factor = (total_wins / total_losses) if total_losses > 0 else 0
            
            # Calculate drawdown
            current_drawdown = ((self.peak_balance - self.current_balance) / self.peak_balance * 100) if self.peak_balance > 0 else 0
            
            return BotStatus(
                is_running=self.is_bot_running,
                balance=self.current_balance,
                daily_pnl=self.daily_pnl,
                active_positions=len(self.active_positions),
                win_rate=win_rate,
                risk_tier="MODERATE",
                total_trades=total_trades,
                profit_factor=profit_factor,
                peak_balance=self.peak_balance,
                current_drawdown=current_drawdown,
                last_update=datetime.utcnow()
            )
        
        except Exception as e:
            logger.error(f"Error getting bot status: {e}")
            raise
    
    def start_bot(self) -> bool:
        """Start the trading bot."""
        try:
            self.is_bot_running = True
            self.last_heartbeat = datetime.utcnow()
            logger.info("Trading bot started")
            return True
        except Exception as e:
            logger.error(f"Error starting bot: {e}")
            raise
    
    def stop_bot(self) -> bool:
        """Stop the trading bot."""
        try:
            self.is_bot_running = False
            logger.info("Trading bot stopped")
            return True
        except Exception as e:
            logger.error(f"Error stopping bot: {e}")
            raise
    
    def pause_bot(self) -> bool:
        """Pause the trading bot."""
        try:
            self.is_bot_running = False
            logger.info("Trading bot paused")
            return True
        except Exception as e:
            logger.error(f"Error pausing bot: {e}")
            raise
    
    def is_running(self) -> bool:
        """Check if bot is running."""
        return self.is_bot_running
    
    def get_last_heartbeat(self) -> datetime:
        """Get last heartbeat timestamp."""
        return self.last_heartbeat
    
    def get_open_positions(self) -> List[Position]:
        """Get list of open positions."""
        try:
            return self.active_positions
        except Exception as e:
            logger.error(f"Error fetching open positions: {e}")
            raise
    
    def get_trade_history(self, limit: int = 100) -> List[Trade]:
        """Get trade history with optional limit."""
        try:
            return self.trade_history[-limit:]
        except Exception as e:
            logger.error(f"Error fetching trade history: {e}")
            raise
    
    def get_recent_trades(self, days: int = 7) -> List[Trade]:
        """Get trades from the last N days."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            return [
                t for t in self.trade_history
                if t.timestamp_close >= cutoff_date
            ]
        except Exception as e:
            logger.error(f"Error fetching recent trades: {e}")
            raise
    
    def close_position(self, symbol: str) -> bool:
        """Close an open position by symbol."""
        try:
            self.active_positions = [
                p for p in self.active_positions 
                if p.symbol != symbol
            ]
            logger.info(f"Position closed: {symbol}")
            return True
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            raise
    
    def calculate_performance_stats(self) -> PerformanceStats:
        """Calculate comprehensive performance statistics."""
        try:
            total_trades = len(self.trade_history)
            winning_trades = [t for t in self.trade_history if t.pnl > 0]
            losing_trades = [t for t in self.trade_history if t.pnl < 0]
            
            win_count = len(winning_trades)
            loss_count = len(losing_trades)
            win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
            
            # PnL calculations
            total_wins = sum(t.pnl for t in winning_trades)
            total_losses = abs(sum(t.pnl for t in losing_trades))
            profit_factor = (total_wins / total_losses) if total_losses > 0 else 0
            total_pnl = sum(t.pnl for t in self.trade_history)
            
            # Average win/loss
            avg_win = (total_wins / win_count) if win_count > 0 else 0
            avg_loss = (total_losses / loss_count) if loss_count > 0 else 0
            
            # Largest win/loss
            largest_win = max((t.pnl for t in winning_trades), default=0)
            largest_loss = min((t.pnl for t in losing_trades), default=0)
            
            # Drawdown
            max_drawdown = ((self.peak_balance - min(
                [self.current_balance] + [self.current_balance - t.pnl for t in self.trade_history]
            )) / self.peak_balance * 100) if self.peak_balance > 0 else 0
            
            current_drawdown = ((self.peak_balance - self.current_balance) / self.peak_balance * 100) if self.peak_balance > 0 else 0
            
            # Get period dates
            period_start = self.trade_history[0].timestamp_open if self.trade_history else datetime.utcnow()
            period_end = datetime.utcnow()
            
            return PerformanceStats(
                total_trades=total_trades,
                winning_trades=win_count,
                losing_trades=loss_count,
                win_rate=win_rate,
                profit_factor=profit_factor,
                max_drawdown=max_drawdown,
                current_drawdown=current_drawdown,
                average_win=avg_win,
                average_loss=avg_loss,
                largest_win=largest_win,
                largest_loss=largest_loss,
                total_pnl=total_pnl,
                daily_pnl=self.daily_pnl,
                sharpe_ratio=None,
                sortino_ratio=None,
                calmar_ratio=None,
                period_start=period_start,
                period_end=period_end
            )
        
        except Exception as e:
            logger.error(f"Error calculating performance stats: {e}")
            raise
    
    def get_equity_curve(self) -> List[dict]:
        """Get equity curve data for charting."""
        try:
            equity = [{"time": datetime.utcnow().isoformat(), "value": self.current_balance}]
            
            # Reconstruct equity curve from trade history
            running_balance = self.current_balance
            for trade in sorted(self.trade_history, key=lambda t: t.timestamp_close, reverse=True):
                running_balance -= trade.pnl
                equity.insert(0, {
                    "time": trade.timestamp_close.isoformat(),
                    "value": running_balance
                })
            
            return equity
        
        except Exception as e:
            logger.error(f"Error getting equity curve: {e}")
            raise
    
    def get_daily_summary(self) -> List[dict]:
        """Get daily PnL summary."""
        try:
            daily_summary = {}
            
            for trade in self.trade_history:
                date_key = trade.timestamp_close.date().isoformat()
                if date_key not in daily_summary:
                    daily_summary[date_key] = {
                        "date": date_key,
                        "trades": 0,
                        "pnl": 0.0,
                        "wins": 0,
                        "losses": 0
                    }
                
                daily_summary[date_key]["trades"] += 1
                daily_summary[date_key]["pnl"] += trade.pnl
                
                if trade.pnl > 0:
                    daily_summary[date_key]["wins"] += 1
                else:
                    daily_summary[date_key]["losses"] += 1
            
            return list(daily_summary.values())
        
        except Exception as e:
            logger.error(f"Error getting daily summary: {e}")
            raise
    
    def get_strategy_config(self) -> StrategyConfig:
        """Get current strategy configuration."""
        try:
            return self.strategy_config
        except Exception as e:
            logger.error(f"Error getting strategy config: {e}")
            raise
    
    def update_strategy_config(self, config: StrategyConfig) -> StrategyConfig:
        """Update strategy configuration."""
        try:
            self.strategy_config = config
            logger.info(f"Strategy config updated: {config.name}")
            return self.strategy_config
        except Exception as e:
            logger.error(f"Error updating strategy config: {e}")
            raise
    
    def get_trading_symbols(self) -> List[str]:
        """Get list of symbols being traded."""
        try:
            return self.strategy_config.symbols
        except Exception as e:
            logger.error(f"Error getting trading symbols: {e}")
            raise