#!/usr/bin/env python3
"""
VIX100 Specialized Strategy Framework
====================================

Complete strategy framework adapted specifically for VIX100 synthetic market behavior.
Removes all forex-specific logic and focuses on 24/7 continuous volatility trading.

Strategy Categories:
1. ICT Strategies (adapted for synthetic markets)
2. Volatility-Based Strategies (VIX100 specific)
3. Price Action Strategies (synthetic patterns)
4. Machine Learning Strategies (pattern discovery)
5. Hybrid Strategies (combining multiple approaches)

Author: Eden AI System
Version: 1.0
Date: October 13, 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging

from vix100_indicators import VIX100IndicatorSuite, calculate_vix100_strength
from eden_vix100_system import VIXSignal

logger = logging.getLogger(__name__)

@dataclass
class StrategySignal:
    """Enhanced strategy signal with VIX100 context"""
    timestamp: datetime
    strategy_name: str
    strategy_family: str
    side: str  # 'buy' or 'sell'
    confidence: float  # 0.0 to 1.0
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_percentage: float
    volatility_context: Dict
    synthetic_patterns: List[str]
    market_regime: str
    timeframe: str
    signal_strength: float
    expected_duration_minutes: int

class VIX100StrategyBase(ABC):
    """Base class for all VIX100 strategies"""
    
    def __init__(self, name: str, family: str, min_confidence: float = 0.6):
        self.name = name
        self.family = family
        self.min_confidence = min_confidence
        self.active = True
        self.performance_stats = {
            'total_signals': 0,
            'successful_signals': 0,
            'win_rate': 0.0,
            'avg_confidence': 0.0
        }
    
    @abstractmethod
    def generate_signal(self, df: pd.DataFrame, indicators: Dict, 
                       current_time: datetime) -> Optional[StrategySignal]:
        """Generate trading signal based on current market conditions"""
        pass
    
    def update_performance(self, signal: StrategySignal, outcome: str):
        """Update strategy performance statistics"""
        self.performance_stats['total_signals'] += 1
        if outcome == 'win':
            self.performance_stats['successful_signals'] += 1
        
        self.performance_stats['win_rate'] = (
            self.performance_stats['successful_signals'] / 
            self.performance_stats['total_signals']
        )
    
    def should_trade(self, market_regime: str, volatility_level: float) -> bool:
        """Determine if strategy should trade in current conditions"""
        return self.active and volatility_level > 0.1

class VIX100ICTStrategy(VIX100StrategyBase):
    """ICT strategies adapted for VIX100 synthetic markets"""
    
    def __init__(self, strategy_type: str):
        super().__init__(f"vix100_ict_{strategy_type}", "ICT_VIX100")
        self.strategy_type = strategy_type
        
        # VIX100-specific ICT parameters (no forex sessions)
        self.volatility_cycles = {
            'low': (0, 6),      # UTC hours of typically lower volatility
            'medium': (6, 12),  # Building volatility
            'high': (12, 18),   # Peak volatility period
            'evening': (18, 24) # Evening wind-down
        }
    
    def generate_signal(self, df: pd.DataFrame, indicators: Dict, 
                       current_time: datetime) -> Optional[StrategySignal]:
        """Generate ICT signal adapted for VIX100"""
        
        if len(df) < 50:
            return None
        
        if self.strategy_type == "liquidity_sweep":
            return self._detect_liquidity_sweep(df, indicators, current_time)
        elif self.strategy_type == "fair_value_gap":
            return self._detect_fair_value_gap(df, indicators, current_time)
        elif self.strategy_type == "order_block":
            return self._detect_order_block(df, indicators, current_time)
        elif self.strategy_type == "volatility_displacement":
            return self._detect_volatility_displacement(df, indicators, current_time)
        
        return None
    
    def _detect_liquidity_sweep(self, df: pd.DataFrame, indicators: Dict, 
                               current_time: datetime) -> Optional[StrategySignal]:
        """Detect liquidity sweeps in VIX100 - adapted for synthetic market"""
        
        # Look for recent highs/lows being taken out
        recent_high = df['high'].rolling(20).max()
        recent_low = df['low'].rolling(20).min()
        
        current_high = df['high'].iloc[-1]
        current_low = df['low'].iloc[-1]
        current_close = df['close'].iloc[-1]
        
        # Detect sweep patterns
        bullish_sweep = (current_high > recent_high.iloc[-2]) and (current_close < current_high)
        bearish_sweep = (current_low < recent_low.iloc[-2]) and (current_close > current_low)
        
        if not (bullish_sweep or bearish_sweep):
            return None
        
        # VIX100-specific validation using volatility
        vol_pressure = indicators.get('vol_pressure_basic', 0)
        vol_burst = indicators.get('vol_burst_signal', 0)
        
        if vol_pressure < 0.005:  # Too quiet for liquidity sweep
            return None
        
        side = 'buy' if bearish_sweep else 'sell'  # Counter-trend after sweep
        entry_price = current_close
        
        # Dynamic stops based on volatility
        atr = (df['high'] - df['low']).rolling(14).mean().iloc[-1]
        volatility_multiplier = max(vol_pressure * 100, 1.5)
        
        stop_distance = atr * volatility_multiplier
        stop_loss = entry_price - stop_distance if side == 'buy' else entry_price + stop_distance
        take_profit = entry_price + stop_distance * 2 if side == 'buy' else entry_price - stop_distance * 2
        
        # Confidence based on sweep clarity and volume
        confidence = min(0.7 + vol_pressure * 2 + (vol_burst * 0.1), 0.95)
        
        if confidence < self.min_confidence:
            return None
        
        return StrategySignal(
            timestamp=current_time,
            strategy_name=self.name,
            strategy_family=self.family,
            side=side,
            confidence=confidence,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_percentage=min(confidence * 0.02, 0.025),
            volatility_context={
                'vol_pressure': vol_pressure,
                'vol_burst': vol_burst,
                'sweep_type': 'bullish' if bullish_sweep else 'bearish'
            },
            synthetic_patterns=['liquidity_sweep'],
            market_regime=indicators.get('regime', 'unknown'),
            timeframe='M5',
            signal_strength=confidence,
            expected_duration_minutes=30
        )
    
    def _detect_fair_value_gap(self, df: pd.DataFrame, indicators: Dict, 
                              current_time: datetime) -> Optional[StrategySignal]:
        """Detect Fair Value Gaps in VIX100 - accounting for synthetic volatility"""
        
        if len(df) < 10:
            return None
        
        # Look for gaps in price action (3-candle pattern)
        for i in range(3, len(df)):
            candle1 = df.iloc[i-2]  # First candle
            candle2 = df.iloc[i-1]  # Middle candle (displacement)
            candle3 = df.iloc[i]    # Current candle
            
            # Bullish FVG: gap between candle1 high and candle3 low
            if (candle2['close'] > candle2['open'] and  # Bullish displacement
                candle1['high'] < candle3['low']):       # Gap exists
                
                gap_size = candle3['low'] - candle1['high']
                gap_percentage = gap_size / candle2['close']
                
                # VIX100 requires larger gaps due to synthetic nature
                if gap_percentage > 0.001:  # Minimum 0.1% gap
                    
                    return self._create_fvg_signal(
                        current_time, 'buy', candle1['high'], candle3['low'],
                        gap_percentage, indicators, df
                    )
            
            # Bearish FVG: gap between candle1 low and candle3 high  
            elif (candle2['close'] < candle2['open'] and  # Bearish displacement
                  candle1['low'] > candle3['high']):      # Gap exists
                
                gap_size = candle1['low'] - candle3['high']
                gap_percentage = gap_size / candle2['close']
                
                if gap_percentage > 0.001:
                    
                    return self._create_fvg_signal(
                        current_time, 'sell', candle3['high'], candle1['low'],
                        gap_percentage, indicators, df
                    )
        
        return None
    
    def _create_fvg_signal(self, current_time: datetime, side: str, 
                          gap_low: float, gap_high: float, gap_percentage: float,
                          indicators: Dict, df: pd.DataFrame) -> StrategySignal:
        """Create FVG signal for VIX100"""
        
        entry_price = df['close'].iloc[-1]
        gap_mid = (gap_low + gap_high) / 2
        
        # Target gap fill
        if side == 'buy':
            take_profit = gap_mid
            stop_loss = entry_price - (gap_high - gap_low)
        else:
            take_profit = gap_mid
            stop_loss = entry_price + (gap_high - gap_low)
        
        confidence = min(0.6 + gap_percentage * 100, 0.9)
        
        return StrategySignal(
            timestamp=current_time,
            strategy_name=self.name,
            strategy_family=self.family,
            side=side,
            confidence=confidence,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_percentage=confidence * 0.015,
            volatility_context={
                'gap_size_pct': gap_percentage,
                'gap_range': (gap_low, gap_high)
            },
            synthetic_patterns=['fair_value_gap'],
            market_regime=indicators.get('regime', 'unknown'),
            timeframe='M5',
            signal_strength=confidence,
            expected_duration_minutes=45
        )
    
    def _detect_order_block(self, df: pd.DataFrame, indicators: Dict, 
                           current_time: datetime) -> Optional[StrategySignal]:
        """Detect Order Blocks in VIX100 - synthetic market adaptation"""
        
        if len(df) < 30:
            return None
        
        # Find significant price moves (displacement)
        price_changes = df['close'].pct_change(5)
        displacement_threshold = price_changes.rolling(50).quantile(0.8)
        
        recent_displacement = price_changes.iloc[-5:].max()
        
        if recent_displacement > displacement_threshold.iloc[-1]:
            # Found displacement, look for the last opposing candle before it
            for i in range(len(df) - 6, max(len(df) - 20, 0), -1):
                candle = df.iloc[i]
                next_candle = df.iloc[i + 1]
                
                if recent_displacement > 0:  # Bullish displacement
                    # Look for last bearish candle before displacement
                    if (candle['close'] < candle['open'] and 
                        next_candle['close'] > next_candle['open']):
                        
                        return self._create_order_block_signal(
                            current_time, 'buy', candle, indicators, df
                        )
                
                else:  # Bearish displacement
                    # Look for last bullish candle before displacement
                    if (candle['close'] > candle['open'] and 
                        next_candle['close'] < next_candle['open']):
                        
                        return self._create_order_block_signal(
                            current_time, 'sell', candle, indicators, df
                        )
        
        return None
    
    def _create_order_block_signal(self, current_time: datetime, side: str,
                                  ob_candle: pd.Series, indicators: Dict, 
                                  df: pd.DataFrame) -> StrategySignal:
        """Create Order Block signal"""
        
        entry_price = df['close'].iloc[-1]
        
        # Order block levels
        if side == 'buy':
            ob_high = ob_candle['high']
            ob_low = ob_candle['low']
            stop_loss = ob_low - (ob_high - ob_low) * 0.1
            take_profit = entry_price + (ob_high - ob_low) * 2
        else:
            ob_high = ob_candle['high'] 
            ob_low = ob_candle['low']
            stop_loss = ob_high + (ob_high - ob_low) * 0.1
            take_profit = entry_price - (ob_high - ob_low) * 2
        
        # Confidence based on order block strength
        ob_size = abs(ob_candle['close'] - ob_candle['open'])
        avg_body_size = abs(df['close'] - df['open']).rolling(20).mean().iloc[-1]
        confidence = min(0.65 + (ob_size / avg_body_size) * 0.1, 0.9)
        
        return StrategySignal(
            timestamp=current_time,
            strategy_name=self.name,
            strategy_family=self.family,
            side=side,
            confidence=confidence,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_percentage=confidence * 0.02,
            volatility_context={
                'ob_range': (ob_candle['low'], ob_candle['high']),
                'ob_strength': ob_size / avg_body_size
            },
            synthetic_patterns=['order_block'],
            market_regime=indicators.get('regime', 'unknown'),
            timeframe='M5',
            signal_strength=confidence,
            expected_duration_minutes=60
        )
    
    def _detect_volatility_displacement(self, df: pd.DataFrame, indicators: Dict, 
                                      current_time: datetime) -> Optional[StrategySignal]:
        """VIX100-specific: Detect volatility-driven displacement"""
        
        vol_pressure = indicators.get('vol_pressure_basic', 0)
        vol_burst = indicators.get('vol_burst_signal', 0)
        compression = indicators.get('compression_state', 0)
        
        # Look for compression followed by expansion
        if compression and vol_burst:
            # Displacement direction
            recent_momentum = df['close'].pct_change(3).iloc[-1]
            side = 'buy' if recent_momentum > 0 else 'sell'
            
            entry_price = df['close'].iloc[-1]
            atr = (df['high'] - df['low']).rolling(14).mean().iloc[-1]
            
            stop_distance = atr * (1 + vol_pressure * 5)
            stop_loss = entry_price - stop_distance if side == 'buy' else entry_price + stop_distance
            take_profit = entry_price + stop_distance * 3 if side == 'buy' else entry_price - stop_distance * 3
            
            confidence = 0.7 + vol_pressure * 2
            confidence = min(confidence, 0.95)
            
            return StrategySignal(
                timestamp=current_time,
                strategy_name=self.name,
                strategy_family=self.family,
                side=side,
                confidence=confidence,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_percentage=0.025,
                volatility_context={
                    'vol_pressure': vol_pressure,
                    'compression_to_expansion': True
                },
                synthetic_patterns=['volatility_displacement'],
                market_regime='burst',
                timeframe='M5',
                signal_strength=confidence,
                expected_duration_minutes=20
            )
        
        return None

class VIX100VolatilityStrategy(VIX100StrategyBase):
    """Pure volatility-based strategies for VIX100"""
    
    def __init__(self, strategy_type: str):
        super().__init__(f"vix100_volatility_{strategy_type}", "Volatility_VIX100")
        self.strategy_type = strategy_type
    
    def generate_signal(self, df: pd.DataFrame, indicators: Dict, 
                       current_time: datetime) -> Optional[StrategySignal]:
        
        if self.strategy_type == "compression_breakout":
            return self._compression_breakout_strategy(df, indicators, current_time)
        elif self.strategy_type == "volatility_burst":
            return self._volatility_burst_strategy(df, indicators, current_time)
        elif self.strategy_type == "mean_reversion":
            return self._volatility_mean_reversion(df, indicators, current_time)
        elif self.strategy_type == "cycle_trading":
            return self._volatility_cycle_trading(df, indicators, current_time)
        
        return None
    
    def _compression_breakout_strategy(self, df: pd.DataFrame, indicators: Dict, 
                                     current_time: datetime) -> Optional[StrategySignal]:
        """Trade breakouts from volatility compression"""
        
        compression = indicators.get('compression_state', 0)
        compression_intensity = indicators.get('compression_intensity', 0)
        
        if not compression or compression_intensity < 0.3:
            return None
        
        # Look for breakout from compression
        bb_upper = indicators.get('bb_upper', df['close'].iloc[-1] * 1.01)
        bb_lower = indicators.get('bb_lower', df['close'].iloc[-1] * 0.99)
        current_price = df['close'].iloc[-1]
        
        if current_price > bb_upper:  # Bullish breakout
            side = 'buy'
            entry_price = current_price
            stop_loss = bb_lower
            take_profit = entry_price + (entry_price - bb_lower) * 2
            
        elif current_price < bb_lower:  # Bearish breakout
            side = 'sell'
            entry_price = current_price
            stop_loss = bb_upper
            take_profit = entry_price - (bb_upper - entry_price) * 2
        else:
            return None
        
        confidence = 0.65 + compression_intensity * 0.3
        
        return StrategySignal(
            timestamp=current_time,
            strategy_name=self.name,
            strategy_family=self.family,
            side=side,
            confidence=confidence,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_percentage=0.015,
            volatility_context={
                'compression_intensity': compression_intensity,
                'breakout_direction': side
            },
            synthetic_patterns=['compression', 'breakout'],
            market_regime='compression',
            timeframe='M5',
            signal_strength=confidence,
            expected_duration_minutes=40
        )
    
    def _volatility_burst_strategy(self, df: pd.DataFrame, indicators: Dict, 
                                 current_time: datetime) -> Optional[StrategySignal]:
        """Trade during volatility bursts"""
        
        vol_burst = indicators.get('vol_burst_signal', 0)
        vol_pressure = indicators.get('vol_pressure_basic', 0)
        
        if not vol_burst or vol_pressure < 0.01:
            return None
        
        # Direction based on momentum
        momentum = df['close'].pct_change(3).iloc[-1]
        
        # Only trade if momentum is strong
        if abs(momentum) < 0.005:
            return None
        
        side = 'buy' if momentum > 0 else 'sell'
        entry_price = df['close'].iloc[-1]
        
        # Tight stops during bursts
        atr = (df['high'] - df['low']).rolling(14).mean().iloc[-1]
        stop_distance = atr * 1.5
        
        stop_loss = entry_price - stop_distance if side == 'buy' else entry_price + stop_distance
        take_profit = entry_price + stop_distance * 2 if side == 'buy' else entry_price - stop_distance * 2
        
        confidence = min(0.6 + vol_pressure * 10 + abs(momentum) * 50, 0.9)
        
        return StrategySignal(
            timestamp=current_time,
            strategy_name=self.name,
            strategy_family=self.family,
            side=side,
            confidence=confidence,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_percentage=0.02,
            volatility_context={
                'vol_burst': vol_burst,
                'vol_pressure': vol_pressure,
                'momentum': momentum
            },
            synthetic_patterns=['volatility_burst'],
            market_regime='burst',
            timeframe='M5',
            signal_strength=confidence,
            expected_duration_minutes=15
        )
    
    def _volatility_mean_reversion(self, df: pd.DataFrame, indicators: Dict, 
                                 current_time: datetime) -> Optional[StrategySignal]:
        """Mean reversion during extreme volatility"""
        
        vol_pressure = indicators.get('vol_pressure_basic', 0)
        vol_percentile = indicators.get('vol_pressure_percentile_50', 50)
        
        # Only trade when volatility is extreme
        if vol_percentile < 90:
            return None
        
        # Look for price extremes relative to moving average
        sma_20 = df['close'].rolling(20).mean().iloc[-1]
        current_price = df['close'].iloc[-1]
        
        distance_from_ma = abs(current_price - sma_20) / sma_20
        
        # Need significant distance for mean reversion
        if distance_from_ma < 0.02:
            return None
        
        # Trade back to mean
        side = 'buy' if current_price < sma_20 else 'sell'
        entry_price = current_price
        
        # Target is the moving average
        take_profit = sma_20
        
        # Stop beyond the extreme
        if side == 'buy':
            stop_loss = current_price - distance_from_ma * current_price * 0.5
        else:
            stop_loss = current_price + distance_from_ma * current_price * 0.5
        
        confidence = min(0.6 + distance_from_ma * 10, 0.85)
        
        return StrategySignal(
            timestamp=current_time,
            strategy_name=self.name,
            strategy_family=self.family,
            side=side,
            confidence=confidence,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_percentage=0.015,
            volatility_context={
                'vol_percentile': vol_percentile,
                'distance_from_ma': distance_from_ma,
                'mean_price': sma_20
            },
            synthetic_patterns=['mean_reversion'],
            market_regime='chaos',
            timeframe='M5',
            signal_strength=confidence,
            expected_duration_minutes=30
        )
    
    def _volatility_cycle_trading(self, df: pd.DataFrame, indicators: Dict, 
                                current_time: datetime) -> Optional[StrategySignal]:
        """Trade based on volatility cycles"""
        
        cycle_position = indicators.get('cycle_position', 0.5)
        vol_proxy = indicators.get('vol_proxy', 0)
        
        if cycle_position is None or np.isnan(cycle_position):
            return None
        
        # Trade at cycle extremes
        if cycle_position < 0.2:  # Near cycle low - expect volatility increase
            side = 'buy'  # Buy volatility
            confidence = 0.6 + (0.2 - cycle_position) * 2
            
        elif cycle_position > 0.8:  # Near cycle high - expect volatility decrease
            side = 'sell'  # Sell volatility
            confidence = 0.6 + (cycle_position - 0.8) * 2
            
        else:
            return None  # Middle of cycle, no clear signal
        
        entry_price = df['close'].iloc[-1]
        atr = (df['high'] - df['low']).rolling(14).mean().iloc[-1]
        
        stop_distance = atr * 1.5
        stop_loss = entry_price - stop_distance if side == 'buy' else entry_price + stop_distance
        take_profit = entry_price + stop_distance * 2.5 if side == 'buy' else entry_price - stop_distance * 2.5
        
        confidence = min(confidence, 0.8)
        
        return StrategySignal(
            timestamp=current_time,
            strategy_name=self.name,
            strategy_family=self.family,
            side=side,
            confidence=confidence,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_percentage=0.018,
            volatility_context={
                'cycle_position': cycle_position,
                'vol_proxy': vol_proxy,
                'cycle_phase': 'low' if cycle_position < 0.5 else 'high'
            },
            synthetic_patterns=['volatility_cycle'],
            market_regime='trend',
            timeframe='M15',
            signal_strength=confidence,
            expected_duration_minutes=90
        )

class VIX100PriceActionStrategy(VIX100StrategyBase):
    """Price action strategies adapted for VIX100"""
    
    def __init__(self, strategy_type: str):
        super().__init__(f"vix100_pa_{strategy_type}", "PriceAction_VIX100")
        self.strategy_type = strategy_type
    
    def generate_signal(self, df: pd.DataFrame, indicators: Dict, 
                       current_time: datetime) -> Optional[StrategySignal]:
        
        if self.strategy_type == "spike_retrace":
            return self._spike_retrace_strategy(df, indicators, current_time)
        elif self.strategy_type == "false_breakout":
            return self._false_breakout_strategy(df, indicators, current_time)
        elif self.strategy_type == "support_resistance":
            return self._support_resistance_strategy(df, indicators, current_time)
        
        return None
    
    def _spike_retrace_strategy(self, df: pd.DataFrame, indicators: Dict, 
                              current_time: datetime) -> Optional[StrategySignal]:
        """Trade spike and retrace patterns"""
        
        spike_up = indicators.get('price_spike_up', 0)
        spike_down = indicators.get('price_spike_down', 0)
        retrace_up = indicators.get('spike_retrace_up', 0)
        retrace_down = indicators.get('spike_retrace_down', 0)
        
        if retrace_up:  # Bullish spike followed by retrace - buy the dip
            side = 'buy'
            entry_price = df['close'].iloc[-1]
            
            # Stop below the retrace low
            recent_low = df['low'].iloc[-3:].min()
            stop_loss = recent_low * 0.999
            
            # Target above the spike high
            recent_high = df['high'].iloc[-5:].max()
            take_profit = recent_high * 1.001
            
            confidence = 0.7
            
        elif retrace_down:  # Bearish spike followed by retrace - sell the bounce
            side = 'sell'
            entry_price = df['close'].iloc[-1]
            
            # Stop above the retrace high
            recent_high = df['high'].iloc[-3:].max()
            stop_loss = recent_high * 1.001
            
            # Target below the spike low
            recent_low = df['low'].iloc[-5:].min()
            take_profit = recent_low * 0.999
            
            confidence = 0.7
            
        else:
            return None
        
        return StrategySignal(
            timestamp=current_time,
            strategy_name=self.name,
            strategy_family=self.family,
            side=side,
            confidence=confidence,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_percentage=0.015,
            volatility_context={
                'spike_type': 'up' if retrace_up else 'down',
                'retrace_confirmed': True
            },
            synthetic_patterns=['spike_retrace'],
            market_regime='recovery',
            timeframe='M5',
            signal_strength=confidence,
            expected_duration_minutes=25
        )
    
    def _false_breakout_strategy(self, df: pd.DataFrame, indicators: Dict, 
                               current_time: datetime) -> Optional[StrategySignal]:
        """Trade false breakouts in VIX100"""
        
        false_breakout_up = indicators.get('false_breakout_up', 0)
        false_breakout_down = indicators.get('false_breakout_down', 0)
        
        if not (false_breakout_up or false_breakout_down):
            return None
        
        current_price = df['close'].iloc[-1]
        
        if false_breakout_up:  # Failed upward breakout - go short
            side = 'sell'
            entry_price = current_price
            
            # Stop above the false breakout high
            breakout_high = df['high'].iloc[-2:].max()
            stop_loss = breakout_high * 1.002
            
            # Target support level
            support = df['low'].rolling(20).min().iloc[-1]
            take_profit = support
            
        else:  # false_breakout_down - Failed downward breakout - go long
            side = 'buy'
            entry_price = current_price
            
            # Stop below the false breakout low
            breakout_low = df['low'].iloc[-2:].min()
            stop_loss = breakout_low * 0.998
            
            # Target resistance level
            resistance = df['high'].rolling(20).max().iloc[-1]
            take_profit = resistance
        
        confidence = 0.75  # False breakouts are high probability
        
        return StrategySignal(
            timestamp=current_time,
            strategy_name=self.name,
            strategy_family=self.family,
            side=side,
            confidence=confidence,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_percentage=0.02,
            volatility_context={
                'false_breakout_type': 'up' if false_breakout_up else 'down'
            },
            synthetic_patterns=['false_breakout'],
            market_regime='recovery',
            timeframe='M5',
            signal_strength=confidence,
            expected_duration_minutes=35
        )
    
    def _support_resistance_strategy(self, df: pd.DataFrame, indicators: Dict, 
                                   current_time: datetime) -> Optional[StrategySignal]:
        """Dynamic support/resistance for VIX100"""
        
        resistance = df['high'].rolling(20).max().iloc[-1]
        support = df['low'].rolling(20).min().iloc[-1]
        current_price = df['close'].iloc[-1]
        
        # Calculate distance to levels
        distance_to_resistance = (resistance - current_price) / current_price
        distance_to_support = (current_price - support) / current_price
        
        # Trade near key levels
        if distance_to_resistance < 0.003:  # Near resistance
            side = 'sell'
            entry_price = current_price
            stop_loss = resistance * 1.002
            take_profit = support
            
            confidence = 0.65
            
        elif distance_to_support < 0.003:  # Near support
            side = 'buy'  
            entry_price = current_price
            stop_loss = support * 0.998
            take_profit = resistance
            
            confidence = 0.65
            
        else:
            return None
        
        # Adjust confidence based on how close we are to the level
        proximity_boost = max(0, (0.003 - min(distance_to_resistance, distance_to_support)) / 0.003 * 0.2)
        confidence += proximity_boost
        
        return StrategySignal(
            timestamp=current_time,
            strategy_name=self.name,
            strategy_family=self.family,
            side=side,
            confidence=min(confidence, 0.85),
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_percentage=0.015,
            volatility_context={
                'support_level': support,
                'resistance_level': resistance,
                'level_type': 'resistance' if side == 'sell' else 'support'
            },
            synthetic_patterns=['support_resistance'],
            market_regime='range',
            timeframe='M15',
            signal_strength=confidence,
            expected_duration_minutes=50
        )

class VIX100StrategyManager:
    """Manages all VIX100 strategies"""
    
    def __init__(self):
        self.strategies = {}
        self.indicator_suite = VIX100IndicatorSuite()
        self.initialize_strategies()
    
    def initialize_strategies(self):
        """Initialize all VIX100 strategies"""
        
        # ICT Strategies
        ict_types = ["liquidity_sweep", "fair_value_gap", "order_block", "volatility_displacement"]
        for ict_type in ict_types:
            strategy = VIX100ICTStrategy(ict_type)
            self.strategies[strategy.name] = strategy
        
        # Volatility Strategies
        vol_types = ["compression_breakout", "volatility_burst", "mean_reversion", "cycle_trading"]
        for vol_type in vol_types:
            strategy = VIX100VolatilityStrategy(vol_type)
            self.strategies[strategy.name] = strategy
        
        # Price Action Strategies
        pa_types = ["spike_retrace", "false_breakout", "support_resistance"]
        for pa_type in pa_types:
            strategy = VIX100PriceActionStrategy(pa_type)
            self.strategies[strategy.name] = strategy
        
        logger.info(f"Initialized {len(self.strategies)} VIX100 strategies")
    
    def generate_all_signals(self, df: pd.DataFrame, 
                           current_time: datetime) -> List[StrategySignal]:
        """Generate signals from all active strategies"""
        
        # Calculate indicators
        df_with_indicators = self.indicator_suite.calculate_all_indicators(df)
        
        # Extract indicator values for the latest bar
        indicators = {}
        if not df_with_indicators.empty:
            latest_row = df_with_indicators.iloc[-1]
            for col in df_with_indicators.columns:
                if col not in ['open', 'high', 'low', 'close', 'volume', 'tick_volume']:
                    indicators[col] = latest_row[col] if not pd.isna(latest_row[col]) else 0
        
        # Get market analysis
        market_analysis = self.indicator_suite.get_market_analysis(df_with_indicators)
        regime = market_analysis.get('regime', {}).get('current', 'unknown') if market_analysis.get('status') == 'success' else 'unknown'
        indicators['regime'] = regime
        
        # Generate signals from each strategy
        signals = []
        for strategy_name, strategy in self.strategies.items():
            if not strategy.active:
                continue
                
            try:
                signal = strategy.generate_signal(df_with_indicators, indicators, current_time)
                if signal and signal.confidence >= strategy.min_confidence:
                    signals.append(signal)
                    
            except Exception as e:
                logger.error(f"Error generating signal from {strategy_name}: {e}")
        
        # Sort by confidence
        signals.sort(key=lambda x: x.confidence, reverse=True)
        
        return signals
    
    def update_strategy_performance(self, strategy_name: str, outcome: str):
        """Update strategy performance"""
        if strategy_name in self.strategies:
            # Note: This would need the original signal to update properly
            pass
    
    def get_strategy_stats(self) -> Dict:
        """Get performance statistics for all strategies"""
        stats = {}
        for name, strategy in self.strategies.items():
            stats[name] = {
                'active': strategy.active,
                'performance': strategy.performance_stats,
                'family': strategy.family
            }
        return stats
    
    def enable_strategy(self, strategy_name: str):
        """Enable a strategy"""
        if strategy_name in self.strategies:
            self.strategies[strategy_name].active = True
    
    def disable_strategy(self, strategy_name: str):
        """Disable a strategy"""
        if strategy_name in self.strategies:
            self.strategies[strategy_name].active = False

if __name__ == "__main__":
    # Test the strategy framework
    print("VIX100 Strategy Framework - Testing")
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2025-01-01', periods=200, freq='5min')
    
    sample_data = pd.DataFrame({
        'open': 100 + np.cumsum(np.random.randn(200) * 0.1),
        'high': 100 + np.cumsum(np.random.randn(200) * 0.1) + np.random.rand(200) * 2,
        'low': 100 + np.cumsum(np.random.randn(200) * 0.1) - np.random.rand(200) * 2,
        'close': 100 + np.cumsum(np.random.randn(200) * 0.1),
        'tick_volume': np.random.randint(100, 1000, 200)
    }, index=dates)
    
    sample_data['high'] = sample_data[['open', 'close', 'high']].max(axis=1)
    sample_data['low'] = sample_data[['open', 'close', 'low']].min(axis=1)
    
    # Initialize strategy manager
    strategy_manager = VIX100StrategyManager()
    
    # Generate signals
    current_time = datetime.now()
    signals = strategy_manager.generate_all_signals(sample_data, current_time)
    
    print(f"Generated {len(signals)} signals")
    for signal in signals[:3]:  # Show top 3 signals
        print(f"  {signal.strategy_name}: {signal.side} @ {signal.confidence:.2f} confidence")
    
    # Get strategy stats
    stats = strategy_manager.get_strategy_stats()
    print(f"Active strategies: {sum(1 for s in stats.values() if s['active'])}")