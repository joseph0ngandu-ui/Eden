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
from pathlib import Path

import pandas as pd
import MetaTrader5 as mt5

from app.models import (
    BotStatus, Trade, Position, PerformanceStats, 
    StrategyConfig, DirectionEnum
)

logger = logging.getLogger(__name__)


class TradingService:
    """Service for managing trading bot operations and data.
    
    Backed by real trade history (logs/trade_history.csv) and live MT5 positions
    instead of in-memory placeholders so the iOS app reflects actual trading state.
    """
    
    def __init__(self):
        """Initialize trading service with Volatility Burst v1.3 configuration."""
        # Core runtime state
        self.is_bot_running = False
        self.starting_balance = 100000.0  # base capital used for equity curve
        self.current_balance = self.starting_balance
        self.peak_balance = self.starting_balance
        self.daily_pnl = 0.0
        self.last_heartbeat = datetime.utcnow()

        # Path to shared trade journal CSV written by the live TradingBot
        self.trade_log_path = (
            Path(__file__).resolve().parent.parent / "logs" / "trade_history.csv"
        )

        # Initialize with profitable VB v1.3 configuration
        self.strategy_config = StrategyConfig()
    
    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------

    def _load_trades_df(self) -> pd.DataFrame:
        """Load the shared trade journal CSV into a DataFrame.

        Returns an empty DataFrame if the file is missing or unreadable.
        """
        if not self.trade_log_path.exists():
            return pd.DataFrame()
        try:
            df = pd.read_csv(self.trade_log_path)
            if "entry_time" in df.columns:
                df["entry_time"] = pd.to_datetime(df["entry_time"], errors="coerce")
            if "exit_time" in df.columns:
                df["exit_time"] = pd.to_datetime(df["exit_time"], errors="coerce")
            return df
        except Exception as e:
            logger.error(f"Error loading trade log CSV: {e}")
            return pd.DataFrame()

    def _get_closed_trades_df(self) -> pd.DataFrame:
        """Return only closed trades from the trade journal."""
        df = self._load_trades_df()
        if df.empty or "status" not in df.columns:
            return pd.DataFrame()
        return df[df["status"] == "CLOSED"].copy()

    def _compute_equity_curve_from_closed(self, closed_df: pd.DataFrame) -> list:
        """Build an equity curve from closed trades.

        Uses starting_balance plus cumulative realized PnL.
        """
        if closed_df.empty:
            # Single-point curve at starting balance
            return [{"time": datetime.utcnow(), "value": float(self.starting_balance)}]

        closed_df = closed_df.sort_values("exit_time")
        balance = float(self.starting_balance)
        curve = []

        # Optional initial point just before the first trade closes
        first_time = closed_df["exit_time"].dropna().min()
        if pd.notna(first_time):
            curve.append({"time": first_time - timedelta(seconds=1), "value": balance})

        for _, row in closed_df.iterrows():
            pnl = float(row.get("pnl") or 0.0)
            balance += pnl
            ts = row.get("exit_time")
            if pd.isna(ts):
                ts = datetime.utcnow()
            curve.append({"time": ts, "value": balance})

        return curve

    # ---------------------------------------------------------------------
    # Public API used by FastAPI endpoints
    # ---------------------------------------------------------------------

    def get_bot_status(self) -> BotStatus:
        """Get current bot status based on real trade history and live positions."""
        try:
            closed_df = self._get_closed_trades_df()
            total_trades = len(closed_df)

            if total_trades > 0:
                winning_trades = closed_df[closed_df["pnl"] > 0]
                losing_trades = closed_df[closed_df["pnl"] < 0]
                win_rate = (len(winning_trades) / total_trades * 100.0) if total_trades > 0 else 0.0

                total_wins = float(winning_trades["pnl"].sum()) if not winning_trades.empty else 0.0
                total_losses = float(abs(losing_trades["pnl"].sum())) if not losing_trades.empty else 0.0
                profit_factor = (total_wins / total_losses) if total_losses > 0 else 0.0

                equity_curve = self._compute_equity_curve_from_closed(closed_df)
                current_balance = equity_curve[-1]["value"] if equity_curve else float(self.starting_balance)
                peak_balance = max(p["value"] for p in equity_curve) if equity_curve else current_balance
                current_drawdown = (
                    ((peak_balance - current_balance) / peak_balance * 100.0)
                    if peak_balance > 0
                    else 0.0
                )
            else:
                win_rate = 0.0
                profit_factor = 0.0
                current_balance = float(self.starting_balance)
                peak_balance = float(self.starting_balance)
                current_drawdown = 0.0

            # Update cached state
            self.current_balance = current_balance
            self.peak_balance = max(self.peak_balance, peak_balance)

            # Active positions come from live MT5, not an in-memory list
            open_positions = self.get_open_positions()

            return BotStatus(
                is_running=self.is_bot_running,
                balance=current_balance,
                daily_pnl=self.daily_pnl,
                active_positions=len(open_positions),
                win_rate=win_rate,
                risk_tier="MODERATE",
                total_trades=total_trades,
                profit_factor=profit_factor,
                peak_balance=self.peak_balance,
                current_drawdown=current_drawdown,
                last_update=datetime.utcnow(),
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
        """Get list of open positions from live MT5 account.

        Falls back to an empty list if MT5 is unavailable.
        """
        try:
            positions: List[Position] = []

            # Attempt to read directly from MT5 terminal
            initialized = mt5.initialize()
            if not initialized:
                logger.warning("MT5 initialize() failed when fetching open positions")
                return positions

            try:
                mt5_positions = mt5.positions_get()
            finally:
                mt5.shutdown()

            if not mt5_positions:
                return positions

            for pos in mt5_positions:
                try:
                    direction = DirectionEnum.BUY if pos.type == mt5.ORDER_TYPE_BUY else DirectionEnum.SELL
                    ts = datetime.fromtimestamp(pos.time)
                    positions.append(
                        Position(
                            id=int(pos.ticket),
                            symbol=pos.symbol,
                            direction=direction,
                            entry=float(pos.price_open),
                            current=float(pos.price_current),
                            pnl=float(pos.profit),
                            confidence=0.0,
                            bars=0,
                            timestamp=ts,
                            volume=float(pos.volume),
                        )
                    )
                except Exception as inner:
                    logger.warning(f"Failed to map MT5 position to API model: {inner}")

            return positions
        except Exception as e:
            logger.error(f"Error fetching open positions: {e}")
            raise
    
    def get_trade_history(self, limit: int = 100) -> List[Trade]:
        """Get trade history with optional limit from the trade journal CSV.

        Only closed trades are returned, newest first.
        """
        try:
            closed_df = self._get_closed_trades_df()
            if closed_df.empty:
                return []

            closed_df = closed_df.sort_values("exit_time", ascending=False).head(limit)
            trades: List[Trade] = []

            for idx, row in closed_df.iterrows():
                try:
                    direction = DirectionEnum.BUY if row.get("type") == "BUY" else DirectionEnum.SELL
                    entry_time = row.get("entry_time")
                    exit_time = row.get("exit_time")
                    ticket = row.get("ticket")
                    trade = Trade(
                        id=int(ticket) if pd.notna(ticket) else int(idx),
                        symbol=str(row.get("symbol")),
                        direction=direction,
                        entry=float(row.get("entry_price", 0.0)),
                        exit=float(row.get("exit_price", 0.0)),
                        pnl=float(row.get("pnl", 0.0)),
                        timestamp_open=entry_time if isinstance(entry_time, datetime) else datetime.utcnow(),
                        timestamp_close=exit_time if isinstance(exit_time, datetime) else datetime.utcnow(),
                        bars_held=0,
                        r_value=0.0,
                        commission=float(row.get("commission", 0.0)),
                        swap=0.0,
                    )
                    trades.append(trade)
                except Exception as inner:
                    logger.warning(f"Failed to map trade log row to Trade model: {inner}")

            return trades
        except Exception as e:
            logger.error(f"Error fetching trade history: {e}")
            raise
    
    def get_recent_trades(self, days: int = 7) -> List[Trade]:
        """Get trades from the last N days based on the trade journal CSV."""
        try:
            closed_df = self._get_closed_trades_df()
            if closed_df.empty:
                return []

            cutoff = datetime.utcnow() - timedelta(days=days)
            recent = closed_df[closed_df["exit_time"] >= cutoff]
            if recent.empty:
                return []

            recent = recent.sort_values("exit_time", ascending=False)
            trades: List[Trade] = []
            for idx, row in recent.iterrows():
                try:
                    direction = DirectionEnum.BUY if row.get("type") == "BUY" else DirectionEnum.SELL
                    entry_time = row.get("entry_time")
                    exit_time = row.get("exit_time")
                    ticket = row.get("ticket")
                    trade = Trade(
                        id=int(ticket) if pd.notna(ticket) else int(idx),
                        symbol=str(row.get("symbol")),
                        direction=direction,
                        entry=float(row.get("entry_price", 0.0)),
                        exit=float(row.get("exit_price", 0.0)),
                        pnl=float(row.get("pnl", 0.0)),
                        timestamp_open=entry_time if isinstance(entry_time, datetime) else datetime.utcnow(),
                        timestamp_close=exit_time if isinstance(exit_time, datetime) else datetime.utcnow(),
                        bars_held=0,
                        r_value=0.0,
                        commission=float(row.get("commission", 0.0)),
                        swap=0.0,
                    )
                    trades.append(trade)
                except Exception as inner:
                    logger.warning(f"Failed to map recent trade row to Trade model: {inner}")

            return trades
        except Exception as e:
            logger.error(f"Error fetching recent trades: {e}")
            raise
    
    def close_position(self, symbol: str) -> bool:
        """Close an open position by symbol on the live MT5 account.

        This mirrors the close logic used by the TradingBot, but runs from the API
        process so iOS-initiated closes affect real trades.
        """
        try:
            if not mt5.initialize():
                logger.error("MT5 initialization failed while attempting to close position")
                return False

            try:
                positions = mt5.positions_get(symbol=symbol)
                if not positions:
                    logger.warning(f"No open MT5 positions for {symbol} to close")
                    return False

                position = positions[0]
                close_type = (
                    mt5.ORDER_TYPE_SELL
                    if position.type == mt5.ORDER_TYPE_BUY
                    else mt5.ORDER_TYPE_BUY
                )

                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": float(position.volume),
                    "type": close_type,
                    "position": position.ticket,
                    "comment": "API close_position",
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }

                result = mt5.order_send(request)
                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    logger.error(f"MT5 close failed for {symbol}: {result.comment}")
                    return False

                logger.info(f"MT5 position closed: {symbol} {position.volume} lots, ticket #{result.order}")
                return True
            finally:
                mt5.shutdown()
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            raise
    
    def calculate_performance_stats(self) -> PerformanceStats:
        """Calculate comprehensive performance statistics from real trade history."""
        try:
            closed_df = self._get_closed_trades_df()
            if closed_df.empty:
                now = datetime.utcnow()
                return PerformanceStats(
                    total_trades=0,
                    winning_trades=0,
                    losing_trades=0,
                    win_rate=0.0,
                    profit_factor=0.0,
                    max_drawdown=0.0,
                    current_drawdown=0.0,
                    average_win=0.0,
                    average_loss=0.0,
                    largest_win=0.0,
                    largest_loss=0.0,
                    total_pnl=0.0,
                    daily_pnl=self.daily_pnl,
                    sharpe_ratio=None,
                    sortino_ratio=None,
                    calmar_ratio=None,
                    period_start=now,
                    period_end=now,
                )

            total_trades = len(closed_df)
            winning_trades = closed_df[closed_df["pnl"] > 0]
            losing_trades = closed_df[closed_df["pnl"] < 0]

            win_count = len(winning_trades)
            loss_count = len(losing_trades)
            win_rate = (win_count / total_trades * 100.0) if total_trades > 0 else 0.0

            total_wins = float(winning_trades["pnl"].sum()) if win_count > 0 else 0.0
            total_losses = float(abs(losing_trades["pnl"].sum())) if loss_count > 0 else 0.0
            profit_factor = (total_wins / total_losses) if total_losses > 0 else 0.0
            total_pnl = float(closed_df["pnl"].sum())

            avg_win = (total_wins / win_count) if win_count > 0 else 0.0
            avg_loss = (total_losses / loss_count) if loss_count > 0 else 0.0

            largest_win = float(winning_trades["pnl"].max()) if win_count > 0 else 0.0
            largest_loss = float(losing_trades["pnl"].min()) if loss_count > 0 else 0.0

            equity_curve = self._compute_equity_curve_from_closed(closed_df)
            balances = [p["value"] for p in equity_curve]
            peak_balance = max(balances) if balances else self.starting_balance
            min_balance = min(balances) if balances else self.starting_balance
            max_drawdown = (
                ((peak_balance - min_balance) / peak_balance * 100.0)
                if peak_balance > 0
                else 0.0
            )

            current_balance = balances[-1] if balances else self.starting_balance
            current_drawdown = (
                ((peak_balance - current_balance) / peak_balance * 100.0)
                if peak_balance > 0
                else 0.0
            )

            period_start = closed_df["entry_time"].dropna().min() or datetime.utcnow()
            period_end = closed_df["exit_time"].dropna().max() or datetime.utcnow()

            # Update cached balances
            self.current_balance = current_balance
            self.peak_balance = max(self.peak_balance, peak_balance)

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
                period_end=period_end,
            )
        except Exception as e:
            logger.error(f"Error calculating performance stats: {e}")
            raise
    
    def get_equity_curve(self) -> List[dict]:
        """Get equity curve data for charting based on trade_history.csv."""
        try:
            closed_df = self._get_closed_trades_df()
            if closed_df.empty:
                return [{"time": datetime.utcnow().isoformat(), "value": float(self.starting_balance)}]

            curve = self._compute_equity_curve_from_closed(closed_df)
            # Serialize datetimes to ISO strings for JSON responses
            return [
                {"time": p["time"].isoformat(), "value": p["value"]}
                for p in curve
            ]
        except Exception as e:
            logger.error(f"Error getting equity curve: {e}")
            raise
    
    def get_daily_summary(self) -> List[dict]:
        """Get daily PnL summary derived from closed trades in trade_history.csv."""
        try:
            closed_df = self._get_closed_trades_df()
            if closed_df.empty:
                return []

            daily_summary: dict = {}
            for _, row in closed_df.iterrows():
                exit_time = row.get("exit_time")
                if not isinstance(exit_time, datetime):
                    continue
                date_key = exit_time.date().isoformat()
                if date_key not in daily_summary:
                    daily_summary[date_key] = {
                        "date": date_key,
                        "trades": 0,
                        "pnl": 0.0,
                        "wins": 0,
                        "losses": 0,
                    }

                pnl = float(row.get("pnl", 0.0))
                daily_summary[date_key]["trades"] += 1
                daily_summary[date_key]["pnl"] += pnl
                if pnl > 0:
                    daily_summary[date_key]["wins"] += 1
                elif pnl < 0:
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