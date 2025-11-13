#!/usr/bin/env python3
"""
Volatility-Aware MA Crossover Strategy

Adapts hold duration and signal sensitivity based on:
- ATR (Average True Range)
- Standard Deviation
- Volatility Index (VIX-like metrics)

Lightweight overnight optimizer for parameter tuning.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class VolatilityMetrics:
    """Store volatility metrics."""
    atr: float
    std_dev: float
    volatility_percentile: float
    volatility_level: str  # LOW, MEDIUM, HIGH


class VolatilityAdapter:
    """Adapts trading parameters based on market volatility."""
    
    def __init__(
        self,
        base_hold_bars: int = 5,
        atr_period: int = 14,
    ):
        """
        Initialize volatility adapter.
        
        Args:
            base_hold_bars: Base hold duration in bars
            atr_period: ATR calculation period
        """
        self.base_hold_bars = base_hold_bars
        self.atr_period = atr_period
        self.volatility_history: List[float] = []
    
    def calculate_atr(self, df: pd.DataFrame, period: int = None) -> pd.Series:
        """
        Calculate Average True Range.
        
        Args:
            df: DataFrame with High, Low, Close columns
            period: ATR period (default: self.atr_period)
            
        Returns:
            Series with ATR values
        """
        if period is None:
            period = self.atr_period
        
        high = df['high']
        low = df['low']
        close = df['close']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # ATR
        atr = tr.rolling(window=period).mean()
        return atr
    
    def calculate_std_dev(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """
        Calculate standard deviation of close prices.
        
        Args:
            df: DataFrame with close prices
            period: Lookback period
            
        Returns:
            Series with std dev values
        """
        return df['close'].rolling(window=period).std()
    
    def classify_volatility(self, atr: float, std_dev: float, df: pd.DataFrame) -> VolatilityMetrics:
        """
        Classify current volatility level.
        
        Args:
            atr: Current ATR value
            std_dev: Current standard deviation
            df: Full OHLC DataFrame for context
            
        Returns:
            VolatilityMetrics object
        """
        if len(df) < self.atr_period:
            return VolatilityMetrics(atr, std_dev, 50.0, "MEDIUM")
        
        # Calculate historical ATR mean and std
        atr_series = self.calculate_atr(df, self.atr_period)
        atr_mean = atr_series.mean()
        atr_std = atr_series.std()
        
        # Percentile-based classification
        if atr_std > 0:
            z_score = (atr - atr_mean) / atr_std
        else:
            z_score = 0
        
        # Convert z-score to percentile (0-100)
        percentile = min(100, max(0, 50 + (z_score * 10)))
        
        # Classify
        if percentile > 66:
            volatility_level = "HIGH"
        elif percentile > 33:
            volatility_level = "MEDIUM"
        else:
            volatility_level = "LOW"
        
        self.volatility_history.append(atr)
        
        return VolatilityMetrics(atr, std_dev, percentile, volatility_level)
    
    def get_adaptive_hold_duration(self, volatility_metrics: VolatilityMetrics) -> int:
        """
        Calculate adaptive hold duration based on volatility.
        
        High volatility = longer hold (wait for noise to settle)
        Low volatility = shorter hold (quick scalps)
        
        Args:
            volatility_metrics: VolatilityMetrics object
            
        Returns:
            Adaptive hold duration in bars
        """
        if volatility_metrics.volatility_level == "HIGH":
            # Longer holds for high volatility
            hold_bars = int(self.base_hold_bars * 1.5)
        elif volatility_metrics.volatility_level == "LOW":
            # Shorter holds for low volatility
            hold_bars = max(2, int(self.base_hold_bars * 0.7))
        else:  # MEDIUM
            hold_bars = self.base_hold_bars
        
        logger.debug(f"Adaptive hold duration: {hold_bars} bars ({volatility_metrics.volatility_level} volatility)")
        return hold_bars
    
    def get_adaptive_ma_params(self, volatility_metrics: VolatilityMetrics) -> Tuple[int, int]:
        """
        Get adaptive MA parameters based on volatility.
        
        High volatility = slower MAs (less whipsaw)
        Low volatility = faster MAs (catch moves quicker)
        
        Args:
            volatility_metrics: VolatilityMetrics object
            
        Returns:
            Tuple of (fast_ma_period, slow_ma_period)
        """
        if volatility_metrics.volatility_level == "HIGH":
            # Slower MAs for high volatility
            fast_ma = 5
            slow_ma = 13
        elif volatility_metrics.volatility_level == "LOW":
            # Faster MAs for low volatility
            fast_ma = 2
            slow_ma = 8
        else:  # MEDIUM
            fast_ma = 3
            slow_ma = 10
        
        logger.debug(f"Adaptive MA params: MA({fast_ma}, {slow_ma}) ({volatility_metrics.volatility_level} volatility)")
        return fast_ma, slow_ma
    
    def get_adaptive_stop_loss(self, volatility_metrics: VolatilityMetrics, entry_price: float) -> float:
        """
        Calculate adaptive stop loss based on volatility.
        
        Args:
            volatility_metrics: VolatilityMetrics object
            entry_price: Entry price
            
        Returns:
            Stop loss price
        """
        atr_multiplier = 1.5 if volatility_metrics.volatility_level == "HIGH" else 1.0
        stop_loss = entry_price - (volatility_metrics.atr * atr_multiplier)
        
        return stop_loss


class ParameterOptimizer:
    """Lightweight daily parameter optimizer for overnight optimization."""
    
    def __init__(
        self,
        min_trades: int = 10,
        optimization_timeframe: str = "hourly"
    ):
        """
        Initialize parameter optimizer.
        
        Args:
            min_trades: Minimum trades required for optimization
            optimization_timeframe: "hourly", "daily", "weekly"
        """
        self.min_trades = min_trades
        self.optimization_timeframe = optimization_timeframe
        self.optimization_history: List[Dict] = []
    
    def optimize_parameters(
        self,
        trade_history: List[Dict],
        date_range: Tuple[datetime, datetime] = None,
    ) -> Dict:
        """
        Run lightweight parameter optimization on trade history.
        
        Runs overnight using latest data to find best MA periods and hold duration.
        
        Args:
            trade_history: List of trade dictionaries with 'symbol', 'pnl', 'hold_bars', 'ma_fast', 'ma_slow'
            date_range: Optional date range for optimization
            
        Returns:
            Dictionary with optimized parameters and performance metrics
        """
        if len(trade_history) < self.min_trades:
            logger.warning(f"Not enough trades for optimization: {len(trade_history)} < {self.min_trades}")
            return {}
        
        df = pd.DataFrame(trade_history)
        
        # Filter by date range if provided
        if date_range:
            start, end = date_range
            df['timestamp'] = pd.to_datetime(df.get('timestamp', df.get('entry_time')))
            df = df[(df['timestamp'] >= start) & (df['timestamp'] <= end)]
        
        if len(df) < self.min_trades:
            logger.warning(f"Not enough trades in date range for optimization: {len(df)} < {self.min_trades}")
            return {}
        
        # Group by symbol and optimize
        results = {}
        
        for symbol in df['symbol'].unique():
            symbol_trades = df[df['symbol'] == symbol]
            
            if len(symbol_trades) < self.min_trades:
                continue
            
            # Calculate metrics by hold duration
            hold_analysis = self._analyze_hold_duration(symbol_trades)
            
            # Calculate metrics by MA parameters
            ma_analysis = self._analyze_ma_parameters(symbol_trades)
            
            results[symbol] = {
                'total_trades': len(symbol_trades),
                'total_pnl': symbol_trades['pnl'].sum(),
                'win_rate': (symbol_trades['pnl'] > 0).sum() / len(symbol_trades) * 100,
                'hold_analysis': hold_analysis,
                'ma_analysis': ma_analysis,
            }
        
        # Find best parameters
        best_params = self._find_best_parameters(results)
        
        # Log optimization results
        self.optimization_history.append({
            'timestamp': datetime.now(),
            'results': results,
            'best_params': best_params,
        })
        
        return best_params
    
    def _analyze_hold_duration(self, trades: pd.DataFrame) -> Dict:
        """Analyze performance by hold duration."""
        analysis = {}
        
        for hold_bars in sorted(trades['hold_bars'].unique()):
            hold_trades = trades[trades['hold_bars'] == hold_bars]
            
            analysis[hold_bars] = {
                'count': len(hold_trades),
                'pnl': hold_trades['pnl'].sum(),
                'win_rate': (hold_trades['pnl'] > 0).sum() / len(hold_trades) * 100,
            }
        
        return analysis
    
    def _analyze_ma_parameters(self, trades: pd.DataFrame) -> Dict:
        """Analyze performance by MA parameters."""
        analysis = {}
        
        for ma_combo in trades.groupby(['ma_fast', 'ma_slow']):
            ma_fast, ma_slow = ma_combo[0]
            ma_trades = ma_combo[1]
            
            key = f"MA({ma_fast},{ma_slow})"
            analysis[key] = {
                'count': len(ma_trades),
                'pnl': ma_trades['pnl'].sum(),
                'win_rate': (ma_trades['pnl'] > 0).sum() / len(ma_trades) * 100,
            }
        
        return analysis
    
    def _find_best_parameters(self, results: Dict) -> Dict:
        """Find best parameters across all symbols."""
        best_hold_bars = None
        best_hold_pnl = float('-inf')
        best_ma_params = None
        best_ma_pnl = float('-inf')
        
        all_hold_analysis = {}
        all_ma_analysis = {}
        
        # Aggregate analysis
        for symbol, data in results.items():
            for hold_bars, metrics in data['hold_analysis'].items():
                if hold_bars not in all_hold_analysis:
                    all_hold_analysis[hold_bars] = {'pnl': 0, 'count': 0}
                all_hold_analysis[hold_bars]['pnl'] += metrics['pnl']
                all_hold_analysis[hold_bars]['count'] += metrics['count']
            
            for ma_combo, metrics in data['ma_analysis'].items():
                if ma_combo not in all_ma_analysis:
                    all_ma_analysis[ma_combo] = {'pnl': 0, 'count': 0}
                all_ma_analysis[ma_combo]['pnl'] += metrics['pnl']
                all_ma_analysis[ma_combo]['count'] += metrics['count']
        
        # Find best hold duration
        for hold_bars, metrics in all_hold_analysis.items():
            if metrics['pnl'] > best_hold_pnl:
                best_hold_pnl = metrics['pnl']
                best_hold_bars = hold_bars
        
        # Find best MA parameters
        for ma_combo, metrics in all_ma_analysis.items():
            if metrics['pnl'] > best_ma_pnl:
                best_ma_pnl = metrics['pnl']
                best_ma_params = ma_combo
        
        return {
            'optimized_hold_bars': best_hold_bars,
            'hold_pnl': best_hold_pnl,
            'optimized_ma_params': best_ma_params,
            'ma_pnl': best_ma_pnl,
            'optimization_time': datetime.now().isoformat(),
            'timestamp': datetime.now().isoformat(),
        }
    
    def get_optimization_report(self) -> str:
        """Get optimization report."""
        if not self.optimization_history:
            return "No optimizations run yet"
        
        latest = self.optimization_history[-1]
        report = f"\n{'='*60}\n"
        report += f"PARAMETER OPTIMIZATION REPORT\n"
        report += f"{'='*60}\n"
        report += f"Optimization Time: {latest['timestamp']}\n"
        report += f"Best Parameters: {latest['best_params']}\n"
        report += f"{'='*60}\n"
        
        return report