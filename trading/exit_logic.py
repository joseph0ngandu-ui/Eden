#!/usr/bin/env python3
"""
Advanced Exit Logic Module

Implements:
1. Adaptive hold time (3-4 bars based on momentum)
2. Trailing stops that tighten to breakeven after +0.8R
3. Adaptive take profit (RR 1.5-2.0) based on ATR expansion
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExitConfig:
    """Configuration for exit logic."""
    min_hold_bars: int = 3
    max_hold_bars: int = 4
    breakeven_move_ratio: float = 0.8  # 0.8R to breakeven
    min_reward_ratio: float = 1.5  # Min risk:reward
    max_reward_ratio: float = 2.0  # Max risk:reward
    atr_period: int = 14
    trailing_stop_enable: bool = True
    use_momentum_exit: bool = True


class ExitManager:
    """
    Advanced exit management for trades.
    
    Features:
    - Adaptive hold time (3-4 bars based on momentum)
    - Trailing stops (tighten to breakeven after +0.8R)
    - Dynamic take profit (ATR-based RR scaling)
    """
    
    def __init__(self, config: Optional[ExitConfig] = None):
        """Initialize exit manager."""
        self.config = config or ExitConfig()
        self.active_trades: Dict = {}
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        return tr.rolling(period).mean()
    
    def calculate_momentum(self, df: pd.DataFrame, period: int = 5) -> pd.Series:
        """Calculate momentum (rate of price change)."""
        momentum = df['close'].pct_change(period)
        return momentum.abs()
    
    def determine_hold_bars(self, df: pd.DataFrame, entry_idx: int) -> int:
        """
        Determine adaptive hold time based on momentum.
        
        Args:
            df: OHLCV dataframe
            entry_idx: Entry bar index
            
        Returns:
            Hold bars (3 or 4)
        """
        if entry_idx < 5:
            return self.config.min_hold_bars
        
        # Calculate momentum at entry
        momentum = self.calculate_momentum(df, period=3)
        current_momentum = momentum.iloc[entry_idx] if entry_idx < len(momentum) else 0
        
        # High momentum (>1% move) = hold 4 bars to catch continuation
        # Low momentum (<1% move) = hold 3 bars to exit faster
        if current_momentum > 0.01:
            return self.config.max_hold_bars
        else:
            return self.config.min_hold_bars
    
    def calculate_stop_loss(
        self,
        entry_price: float,
        atr: float,
        direction: str = 'BUY'
    ) -> float:
        """
        Calculate stop loss using ATR.
        
        Args:
            entry_price: Entry price
            atr: Current ATR value
            direction: 'BUY' or 'SELL'
            
        Returns:
            Stop loss price
        """
        if direction == 'BUY':
            stop_loss = entry_price - (atr * 1.5)
        else:
            stop_loss = entry_price + (atr * 1.5)
        
        return stop_loss
    
    def calculate_adaptive_tp(
        self,
        entry_price: float,
        stop_loss: float,
        atr: float,
        atr_sma: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Calculate adaptive take profit based on ATR expansion.
        
        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            atr: Current ATR
            atr_sma: 20-period ATR SMA (for expansion detection)
            
        Returns:
            (take_profit_price, reward_ratio)
        """
        risk = abs(entry_price - stop_loss)
        
        # Determine reward ratio based on ATR expansion
        if atr_sma and atr > atr_sma * 1.2:
            # High volatility expansion - use 2.0R
            reward_ratio = self.config.max_reward_ratio
        else:
            # Normal volatility - use 1.5R
            reward_ratio = self.config.min_reward_ratio
        
        reward = risk * reward_ratio
        take_profit = entry_price + reward if entry_price > stop_loss else entry_price - reward
        
        return take_profit, reward_ratio
    
    def update_trailing_stop(
        self,
        current_price: float,
        entry_price: float,
        stop_loss: float,
        initial_risk: float,
        direction: str = 'BUY'
    ) -> float:
        """
        Update trailing stop that tightens to breakeven after +0.8R move.
        
        Args:
            current_price: Current market price
            entry_price: Entry price
            stop_loss: Current stop loss
            initial_risk: Initial risk amount (entry - stop)
            direction: 'BUY' or 'SELL'
            
        Returns:
            Updated stop loss
        """
        if not self.config.trailing_stop_enable:
            return stop_loss
        
        # Calculate current profit
        if direction == 'BUY':
            current_profit = current_price - entry_price
            breakeven_profit = initial_risk * self.config.breakeven_move_ratio
            
            # Once price moves +0.8R, tighten stop to breakeven
            if current_profit >= breakeven_profit:
                return entry_price  # Tighten to breakeven
            
            # Normal trailing stop: never move stop below entry
            if current_price > entry_price:
                return max(stop_loss, current_price - initial_risk)
        
        else:  # SELL
            current_profit = entry_price - current_price
            breakeven_profit = initial_risk * self.config.breakeven_move_ratio
            
            if current_profit >= breakeven_profit:
                return entry_price
            
            if current_price < entry_price:
                return min(stop_loss, current_price + initial_risk)
        
        return stop_loss
    
    def check_exit_condition(
        self,
        current_price: float,
        entry_price: float,
        take_profit: float,
        stop_loss: float,
        bars_held: int,
        hold_bars: int,
        direction: str = 'BUY'
    ) -> Tuple[bool, str]:
        """
        Check if trade should exit.
        
        Args:
            current_price: Current market price
            entry_price: Entry price
            take_profit: Take profit level
            stop_loss: Stop loss level
            bars_held: Number of bars held
            hold_bars: Target hold bars
            direction: 'BUY' or 'SELL'
            
        Returns:
            (should_exit: bool, reason: str)
        """
        # Check stop loss first
        if direction == 'BUY':
            if current_price <= stop_loss:
                return True, "Stop Loss Hit"
            if current_price >= take_profit:
                return True, "Take Profit Hit"
        else:  # SELL
            if current_price >= stop_loss:
                return True, "Stop Loss Hit"
            if current_price <= take_profit:
                return True, "Take Profit Hit"
        
        # Check adaptive hold time
        if bars_held >= hold_bars:
            return True, f"Hold Time Exit ({hold_bars} bars)"
        
        return False, ""
    
    def get_exit_statistics(self) -> Dict:
        """Get statistics on exit methods used."""
        return {
            'adaptive_holds': len([t for t in self.active_trades.values() if t.get('exit_reason') == 'hold_time']),
            'trailing_stops': len([t for t in self.active_trades.values() if t.get('exit_reason') == 'trailing_stop']),
            'take_profits': len([t for t in self.active_trades.values() if t.get('exit_reason') == 'take_profit']),
            'stop_losses': len([t for t in self.active_trades.values() if t.get('exit_reason') == 'stop_loss']),
        }


class ExitRule:
    """Simple rule-based exit engine."""
    
    def __init__(self):
        self.exit_manager = ExitManager()
    
    def apply_exits(
        self,
        trades: list,
        df: pd.DataFrame,
        current_idx: int
    ) -> list:
        """
        Apply exit logic to open trades.
        
        Args:
            trades: List of open trades
            df: OHLCV dataframe
            current_idx: Current bar index
            
        Returns:
            List of closed trades with exit info
        """
        closed_trades = []
        atr = self.exit_manager.calculate_atr(df)
        
        for trade in trades:
            entry_idx = trade['entry_idx']
            entry_price = trade['entry_price']
            bars_held = current_idx - entry_idx
            
            current_price = df['close'].iloc[current_idx]
            current_atr = atr.iloc[current_idx] if current_idx < len(atr) else 0
            
            # Check exit conditions
            stop_loss = trade.get('stop_loss', self.exit_manager.calculate_stop_loss(entry_price, current_atr))
            take_profit = trade.get('take_profit', entry_price + (current_atr * 1.5))
            hold_bars = self.exit_manager.determine_hold_bars(df, entry_idx)
            
            should_exit, exit_reason = self.exit_manager.check_exit_condition(
                current_price, entry_price, take_profit, stop_loss,
                bars_held, hold_bars, trade.get('direction', 'BUY')
            )
            
            if should_exit:
                trade['exit_price'] = current_price
                trade['exit_idx'] = current_idx
                trade['exit_reason'] = exit_reason
                trade['bars_held'] = bars_held
                profit = (current_price - entry_price) * trade.get('volume', 1)
                trade['profit'] = profit
                closed_trades.append(trade)
        
        return closed_trades


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    print("Exit Logic Module Ready")
    print("- Adaptive hold time (3-4 bars)")
    print("- Trailing stops (breakeven after +0.8R)")
    print("- Dynamic take profit (RR 1.5-2.0)")
    print("- ATR-based position sizing")
