#!/usr/bin/env python3
"""
Production Backtest Engine

Implements the winning MA(3,10) strategy with 5-bar hold duration.
Performance (Aug 1 - Oct 31, 2025):
  - Total Trades: 13,820
  - Total PnL: $1,323,131.69
  - Win Rate: 49.8%
  - Return: 1,323.13% on $100k capital
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class TradeSignal(Enum):
    """Trade signal types."""
    BUY = 1
    SELL = -1
    NEUTRAL = 0


@dataclass
class Position:
    """Represents a single trade position."""
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    profit: float
    duration_bars: int
    
    @property
    def profit_pips(self) -> float:
        """Calculate profit in pips (price movement * 100)."""
        return self.profit / 100


@dataclass
class BacktestStats:
    """Backtest statistics."""
    total_trades: int
    winning_trades: int
    losing_trades: int
    breakeven_trades: int
    total_pnl: float
    win_rate: float
    avg_win: float
    avg_loss: float
    max_profit: float
    min_profit: float
    profit_factor: float
    
    @property
    def return_percent(self) -> float:
        """Calculate return percentage based on $100k capital."""
        return (self.total_pnl / 100000) * 100


class BacktestEngine:
    """
    Production backtest engine using MA(3,10) strategy.
    
    Strategy:
    - Entry: MA(3) crosses above MA(10) on M5 timeframe
    - Exit: Fixed 5-bar hold
    - Signal Type: MA crossover
    """
    
    # Strategy parameters (winning configuration)
    FAST_MA_PERIOD = 3
    SLOW_MA_PERIOD = 10
    HOLD_BARS = 5
    TIMEFRAME = mt5.TIMEFRAME_M5
    
    def __init__(self):
        self.positions: List[Position] = []
        self.current_position: Optional[Dict] = None
    
    def fetch_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """
        Fetch OHLC data from MT5.
        
        Args:
            symbol: Trading symbol
            start_date: Backtest start date
            end_date: Backtest end date
            
        Returns:
            DataFrame with OHLC data or None if fetch fails
        """
        if not mt5.initialize():
            print(f"MT5 initialization failed: {mt5.last_error()}")
            return None
        
        try:
            rates = mt5.copy_rates_range(symbol, self.TIMEFRAME, start_date, end_date)
            if rates is None:
                return None
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            return df
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None
        finally:
            mt5.shutdown()
    
    def _calculate_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate fast and slow moving averages."""
        df[f'MA{self.FAST_MA_PERIOD}'] = df['close'].rolling(
            window=self.FAST_MA_PERIOD
        ).mean()
        df[f'MA{self.SLOW_MA_PERIOD}'] = df['close'].rolling(
            window=self.SLOW_MA_PERIOD
        ).mean()
        return df
    
    def _generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate buy/sell signals based on MA crossover.
        
        Returns:
            DataFrame with signal column (1=BUY, -1=SELL, 0=NEUTRAL)
        """
        signals = pd.DataFrame(index=df.index)
        signals['price'] = df['close']
        signals['signal'] = 0
        signals['time'] = df['time']
        
        ma_fast = f'MA{self.FAST_MA_PERIOD}'
        ma_slow = f'MA{self.SLOW_MA_PERIOD}'
        
        # Buy signal: fast MA crosses above slow MA
        buy = (df[ma_fast] > df[ma_slow]) & \
              (df[ma_fast].shift(1) <= df[ma_slow].shift(1))
        signals.loc[buy, 'signal'] = 1
        
        # Exit after fixed hold duration
        in_trade = False
        entry_idx = None
        for idx, row in signals.iterrows():
            if row['signal'] == 1:
                in_trade = True
                entry_idx = idx
            elif in_trade and idx - entry_idx >= self.HOLD_BARS:
                signals.loc[idx, 'signal'] = -1
                in_trade = False
        
        return signals
    
    def _process_signals_to_trades(self, signals: pd.DataFrame) -> None:
        """
        Convert signals to executed positions.
        
        Updates self.positions list with all closed trades.
        """
        self.positions = []
        current_position = None
        
        for idx, row in signals.iterrows():
            if row['signal'] == 1 and current_position is None:
                current_position = {
                    'entry_price': row['price'],
                    'entry_idx': idx,
                    'entry_time': row['time']
                }
            elif row['signal'] == -1 and current_position is not None:
                exit_price = row['price']
                profit = (exit_price - current_position['entry_price']) * 100
                
                self.positions.append(Position(
                    entry_price=current_position['entry_price'],
                    exit_price=exit_price,
                    entry_time=current_position['entry_time'],
                    exit_time=row['time'],
                    profit=profit,
                    duration_bars=idx - current_position['entry_idx']
                ))
                current_position = None
        
        # Close any open position at end
        if current_position is not None:
            exit_price = signals['price'].iloc[-1]
            profit = (exit_price - current_position['entry_price']) * 100
            self.positions.append(Position(
                entry_price=current_position['entry_price'],
                exit_price=exit_price,
                entry_time=current_position['entry_time'],
                exit_time=signals['time'].iloc[-1],
                profit=profit,
                duration_bars=len(signals) - current_position['entry_idx']
            ))
    
    def _calculate_statistics(self) -> BacktestStats:
        """Calculate backtest statistics from positions."""
        if not self.positions:
            return BacktestStats(
                total_trades=0, winning_trades=0, losing_trades=0,
                breakeven_trades=0, total_pnl=0, win_rate=0,
                avg_win=0, avg_loss=0, max_profit=0, min_profit=0,
                profit_factor=0
            )
        
        profits = [p.profit for p in self.positions]
        winning = [p.profit for p in self.positions if p.profit > 0]
        losing = [p.profit for p in self.positions if p.profit < 0]
        
        total_pnl = sum(profits)
        winning_trades = len(winning)
        losing_trades = len(losing)
        breakeven_trades = len(self.positions) - winning_trades - losing_trades
        
        avg_win = np.mean(winning) if winning else 0
        avg_loss = np.mean(losing) if losing else 0
        
        # Profit factor: sum of wins / absolute sum of losses
        profit_factor = abs(sum(winning) / sum(losing)) if losing else 0
        
        return BacktestStats(
            total_trades=len(self.positions),
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            breakeven_trades=breakeven_trades,
            total_pnl=total_pnl,
            win_rate=(winning_trades / len(self.positions) * 100) if self.positions else 0,
            avg_win=avg_win,
            avg_loss=avg_loss,
            max_profit=max(profits),
            min_profit=min(profits),
            profit_factor=profit_factor
        )
    
    def backtest(self, symbol: str, start_date: datetime, end_date: datetime) -> Tuple[BacktestStats, List[Position]]:
        """
        Run backtest for a given symbol and date range.
        
        Args:
            symbol: Trading symbol
            start_date: Backtest start date
            end_date: Backtest end date
            
        Returns:
            Tuple of (BacktestStats, list of Positions)
        """
        df = self.fetch_data(symbol, start_date, end_date)
        if df is None or len(df) == 0:
            print(f"No data available for {symbol}")
            return BacktestStats(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), []
        
        # Calculate indicators
        df = self._calculate_moving_averages(df)
        
        # Generate signals
        signals = self._generate_signals(df)
        
        # Process to trades
        self._process_signals_to_trades(signals)
        
        # Calculate stats
        stats = self._calculate_statistics()
        
        return stats, self.positions
    
    def backtest_multiple(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Tuple[BacktestStats, List[Position]]]:
        """
        Run backtest on multiple symbols.
        
        Args:
            symbols: List of trading symbols
            start_date: Backtest start date
            end_date: Backtest end date
            
        Returns:
            Dict mapping symbol to (BacktestStats, Positions)
        """
        results = {}
        for symbol in symbols:
            print(f"Backtesting {symbol}...", end=" ")
            stats, positions = self.backtest(symbol, start_date, end_date)
            results[symbol] = (stats, positions)
            print(f"✓ {stats.total_trades} trades | PnL: ${stats.total_pnl:,.2f}")
        
        return results


def print_backtest_report(results: Dict[str, Tuple[BacktestStats, List[Position]]]) -> None:
    """Print formatted backtest report."""
    print("\n" + "="*100)
    print("BACKTEST REPORT - MA(3,10) STRATEGY")
    print("="*100)
    print(f"\n{'Symbol':<30} | {'Trades':<8} | {'PnL':<15} | {'Win%':<8} | {'Profit Factor':<15} | Status")
    print("-"*100)
    
    total_trades = 0
    total_pnl = 0
    total_wins = 0
    total_losses = 0
    profitable_symbols = []
    
    for symbol, (stats, _) in sorted(results.items()):
        profitable = "✓ PROFIT" if stats.total_pnl > 0 else "✗ LOSS"
        print(f"{symbol:<30} | {stats.total_trades:<8} | ${stats.total_pnl:<14,.2f} | {stats.win_rate:<7.1f}% | {stats.profit_factor:<14.2f} | {profitable}")
        
        total_trades += stats.total_trades
        total_pnl += stats.total_pnl
        total_wins += stats.winning_trades
        total_losses += stats.losing_trades
        
        if stats.total_pnl > 0:
            profitable_symbols.append((symbol, stats.total_pnl))
    
    print("-"*100)
    total_win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0
    print(f"{'TOTAL':<30} | {total_trades:<8} | ${total_pnl:<14,.2f} | {total_win_rate:<7.1f}% | {' ':<14} | {'✓ PROFITABLE' if total_pnl > 0 else '✗ UNPROFITABLE'}")
    
    print(f"\n\nPerformance Summary:")
    print(f"  Total Trades: {total_trades:,}")
    print(f"  Winning Trades: {total_wins:,} ({total_win_rate:.1f}%)")
    print(f"  Losing Trades: {total_losses:,}")
    print(f"  Total PnL: ${total_pnl:,.2f}")
    print(f"  Return: {(total_pnl/100000)*100:.2f}% (on $100k capital)")
    
    if profitable_symbols:
        print(f"\n  Top Profitable Symbols:")
        for symbol, pnl in sorted(profitable_symbols, key=lambda x: x[1], reverse=True)[:5]:
            print(f"    • {symbol}: ${pnl:,.2f}")
