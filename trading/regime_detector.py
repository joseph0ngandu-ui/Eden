#!/usr/bin/env python3
"""
Market Regime Detector

Analyzes market conditions to dynamically adjust strategy behavior.
Regimes: HIGH_VOL, LOW_VOL, TRENDING, RANGING, NORMAL
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class VolatilityRegime(Enum):
    HIGH = "HIGH_VOL"
    LOW = "LOW_VOL"
    NORMAL = "NORMAL"


class TrendRegime(Enum):
    STRONG_UP = "STRONG_UPTREND"
    STRONG_DOWN = "STRONG_DOWNTREND"
    WEAK_TREND = "WEAK_TREND"
    RANGING = "RANGING"


@dataclass
class MarketRegime:
    volatility: VolatilityRegime
    trend: TrendRegime
    atr_ratio: float  # Current ATR / Average ATR
    adx_value: float
    risk_multiplier: float  # Suggested position size adjustment
    
    def is_favorable_for_breakout(self) -> bool:
        """Breakout strategies perform well in normal/high vol + trending."""
        return (self.volatility != VolatilityRegime.LOW and 
                self.trend in [TrendRegime.STRONG_UP, TrendRegime.STRONG_DOWN, TrendRegime.WEAK_TREND])
    
    def is_favorable_for_mean_reversion(self) -> bool:
        """Mean reversion works in low vol + ranging."""
        return (self.volatility == VolatilityRegime.LOW and 
                self.trend == TrendRegime.RANGING)
    
    def __str__(self):
        return f"[{self.volatility.value}|{self.trend.value}] Risk: {self.risk_multiplier:.1f}x"


class RegimeDetector:
    """
    Detects market regime using multiple indicators.
    
    Usage:
        detector = RegimeDetector()
        regime = detector.detect(df)
        if regime.is_favorable_for_breakout():
            # Execute breakout strategy
    """
    
    def __init__(self, 
                 atr_period: int = 14,
                 atr_lookback: int = 50,
                 adx_period: int = 14,
                 ema_period: int = 50):
        self.atr_period = atr_period
        self.atr_lookback = atr_lookback
        self.adx_period = adx_period
        self.ema_period = ema_period
        
        # Cache for regime per symbol (avoid recalculating every bar)
        self._cache: Dict[str, Tuple[pd.Timestamp, MarketRegime]] = {}
        self._cache_duration = pd.Timedelta(hours=1)  # Recalculate hourly
    
    def calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Average True Range."""
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        
        tr1 = high - low
        tr2 = (high - close).abs()
        tr3 = (low - close).abs()
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(self.atr_period).mean()
    
    def calculate_adx(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Average Directional Index."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # +DM and -DM
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        # True Range
        tr = self.calculate_atr(df) * self.atr_period  # Undo the averaging
        
        # Smoothed +DI and -DI
        plus_di = 100 * (plus_dm.rolling(self.adx_period).mean() / tr.rolling(self.adx_period).mean())
        minus_di = 100 * (minus_dm.rolling(self.adx_period).mean() / tr.rolling(self.adx_period).mean())
        
        # DX and ADX
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(self.adx_period).mean()
        
        return adx
    
    def calculate_ema_slope(self, df: pd.DataFrame, periods: int = 10) -> float:
        """Calculate EMA slope (direction and strength)."""
        ema = df['close'].ewm(span=self.ema_period).mean()
        if len(ema) < periods:
            return 0.0
        
        slope = (ema.iloc[-1] - ema.iloc[-periods]) / periods
        # Normalize by ATR for comparability across instruments
        atr = self.calculate_atr(df).iloc[-1]
        if pd.isna(atr) or atr == 0:
            return 0.0
        
        return slope / atr
    
    def detect(self, df: pd.DataFrame, symbol: str = "UNKNOWN") -> MarketRegime:
        """
        Detect current market regime.
        
        Args:
            df: OHLC DataFrame with at least 100 bars
            symbol: Symbol name for caching
            
        Returns:
            MarketRegime with volatility and trend classification
        """
        if len(df) < self.atr_lookback + 20:
            # Insufficient data, return neutral regime
            return MarketRegime(
                volatility=VolatilityRegime.NORMAL,
                trend=TrendRegime.WEAK_TREND,
                atr_ratio=1.0,
                adx_value=20.0,
                risk_multiplier=1.0
            )
        
        # Check cache
        current_time = df.index[-1] if hasattr(df.index, '__getitem__') else pd.Timestamp.now()
        if symbol in self._cache:
            cached_time, cached_regime = self._cache[symbol]
            if current_time - cached_time < self._cache_duration:
                return cached_regime
        
        # Calculate indicators
        atr_series = self.calculate_atr(df)
        current_atr = atr_series.iloc[-1]
        avg_atr = atr_series.iloc[-self.atr_lookback:].mean()
        atr_ratio = current_atr / avg_atr if avg_atr > 0 else 1.0
        
        adx_series = self.calculate_adx(df)
        current_adx = adx_series.iloc[-1] if not pd.isna(adx_series.iloc[-1]) else 20.0
        
        ema_slope = self.calculate_ema_slope(df)
        
        # Classify Volatility
        if atr_ratio > 1.5:
            vol_regime = VolatilityRegime.HIGH
        elif atr_ratio < 0.6:
            vol_regime = VolatilityRegime.LOW
        else:
            vol_regime = VolatilityRegime.NORMAL
        
        # Classify Trend
        if current_adx > 30:
            if ema_slope > 0.1:
                trend_regime = TrendRegime.STRONG_UP
            elif ema_slope < -0.1:
                trend_regime = TrendRegime.STRONG_DOWN
            else:
                trend_regime = TrendRegime.WEAK_TREND
        elif current_adx > 20:
            trend_regime = TrendRegime.WEAK_TREND
        else:
            trend_regime = TrendRegime.RANGING
        
        # Calculate Risk Multiplier
        # Reduce size in high vol, increase in trending markets
        risk_mult = 1.0
        if vol_regime == VolatilityRegime.HIGH:
            risk_mult *= 0.5  # Cut size in half during high vol
        elif vol_regime == VolatilityRegime.LOW:
            risk_mult *= 0.75  # Slightly reduce in low vol (less opportunity)
        
        if trend_regime in [TrendRegime.STRONG_UP, TrendRegime.STRONG_DOWN]:
            risk_mult *= 1.25  # Increase in strong trends
        elif trend_regime == TrendRegime.RANGING:
            risk_mult *= 0.75  # Reduce in ranging (more choppy)
        
        # Cap multiplier
        risk_mult = max(0.25, min(1.5, risk_mult))
        
        regime = MarketRegime(
            volatility=vol_regime,
            trend=trend_regime,
            atr_ratio=atr_ratio,
            adx_value=current_adx,
            risk_multiplier=risk_mult
        )
        
        # Cache result
        self._cache[symbol] = (current_time, regime)
        
        logger.info(f"[REGIME] {symbol}: {regime}")
        
        return regime
    
    def should_trade_strategy(self, regime: MarketRegime, strategy_type: str) -> Tuple[bool, str]:
        """
        Determine if a strategy should be active given the current regime.
        
        Args:
            regime: Current MarketRegime
            strategy_type: One of 'breakout', 'momentum', 'mean_reversion', 'trend'
            
        Returns:
            (should_trade, reason)
        """
        strategy_type = strategy_type.lower()
        
        if strategy_type in ['breakout', 'momentum', 'index', 'volatility_expansion']:
            if regime.volatility == VolatilityRegime.LOW:
                return False, "Low volatility - breakouts likely to fail"
            if regime.trend == TrendRegime.RANGING:
                return False, "Ranging market - momentum strategies underperform"
            return True, "Favorable for momentum/breakout"
        
        elif strategy_type in ['mean_reversion', 'fade', 'squeeze']:
            if regime.volatility == VolatilityRegime.HIGH:
                return False, "High volatility - mean reversion risky"
            if regime.trend in [TrendRegime.STRONG_UP, TrendRegime.STRONG_DOWN]:
                return False, "Strong trend - don't fight the trend"
            return True, "Favorable for mean reversion"
        
        elif strategy_type in ['trend', 'continuation', 'momentum_continuation']:
            if regime.trend == TrendRegime.RANGING:
                return False, "No trend detected"
            return True, "Trend present - continuation viable"
        
        else:
            # Unknown strategy type, allow by default
            return True, "Default allow"


# Singleton instance for easy import
_detector_instance: Optional[RegimeDetector] = None

def get_regime_detector() -> RegimeDetector:
    """Get singleton RegimeDetector instance."""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = RegimeDetector()
    return _detector_instance
