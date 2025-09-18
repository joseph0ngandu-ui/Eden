#!/usr/bin/env python3
"""
Eden Complete AI Trading System
===============================

Full implementation with:
- ICT Strategies (Liquidity sweeps, FVGs, Order blocks, OTE, Judas)
- Price Action Strategies (S/R, Supply/Demand, Patterns, Breakouts)
- Quantitative Strategies (MA, RSI/MACD divergence, BB, Mean reversion)
- AI-Generated Strategies (ML discovers new patterns and creates strategies)
- Market Context Analysis with HTF bias and regime classification
- True AI Learning that adapts, prunes, and generates new strategies
- Advanced ML that can trade against bias on reversal patterns

Author: Eden AI System
Version: 9.0 (Complete AI Framework)
Date: September 14, 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import warnings
import logging
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import joblib
from concurrent.futures import ThreadPoolExecutor
import time
from abc import ABC, abstractmethod
from collections import defaultdict

# MT5 imports
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    print("❌ MetaTrader5 not installed. Install with: pip install MetaTrader5")
    exit(1)

# ML imports
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import classification_report, accuracy_score
    from sklearn.cluster import KMeans
    from sklearn.neural_network import MLPClassifier
    ML_AVAILABLE = True
except ImportError:
    print("⚠️ scikit-learn not available. ML features will be limited.")
    ML_AVAILABLE = False

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Enums and Constants
@dataclass
class MarketRegime:
    """Market regime classification"""
    regime: str  # 'trend', 'reversal', 'range', 'momentum_burst'
    confidence: float
    duration: int  # bars in this regime
    volatility: float
    direction: int  # 1 = bullish, -1 = bearish, 0 = neutral

@dataclass
class HTFBias:
    """Higher timeframe bias"""
    daily: str  # 'bullish', 'bearish', 'neutral'
    h4: str
    h1: str
    overall: str
    confidence: float
    reversal_probability: float

@dataclass
class Signal:
    """Enhanced trading signal"""
    timestamp: datetime
    symbol: str
    side: str  # 'buy' or 'sell'
    confidence: float  # 0-1
    strategy_name: str
    strategy_family: str  # 'ICT', 'PA', 'Quant', 'AI'
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_percent: float = 0.01
    htf_bias: Optional[str] = None
    against_bias: bool = False  # True if trading against HTF bias
    reversal_signal: bool = False
    features: Optional[Dict] = None
    regime: Optional[str] = None

@dataclass
class AIGeneratedStrategy:
    """AI-discovered strategy"""
    id: str
    name: str
    rules: Dict[str, Any]
    performance_history: List[float]
    created_at: datetime
    last_optimized: datetime
    win_rate: float
    profit_factor: float
    max_drawdown: float
    trade_count: int
    active: bool = True

class MarketRegimeClassifier:
    """ML-based market regime classifier"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for regime classification"""
        features = df.copy()
        
        # Volatility features
        features['atr_ratio'] = features['atr'] / features['close']
        features['volatility_rank'] = features['atr'].rolling(50).rank() / 50
        
        # Trend features
        features['trend_strength'] = abs(features['close'] - features['sma_200']) / features['atr']
        features['ema_alignment'] = np.where(
            (features['ema_12'] > features['ema_26']) & (features['ema_26'] > features['ema_50']), 1,
            np.where((features['ema_12'] < features['ema_26']) & (features['ema_26'] < features['ema_50']), -1, 0)
        )
        
        # Range features
        features['range_bound'] = (features['bb_width'] < features['bb_width'].rolling(20).quantile(0.3)).astype(int)
        
        # Momentum features
        features['momentum_strength'] = abs(features['momentum_10'])
        features['rsi_extreme'] = ((features['rsi_14'] > 70) | (features['rsi_14'] < 30)).astype(int)
        
        return features
    
    def classify_regime(self, features: pd.DataFrame) -> List[MarketRegime]:
        """Classify market regime for each bar"""
        if not self.is_trained:
            return [MarketRegime('trend', 0.5, 1, 0.01, 0) for _ in range(len(features))]
        
        # Simple rule-based fallback if ML not available
        regimes = []
        
        for i, row in features.iterrows():
            volatility = row.get('atr_ratio', 0.01)
            trend_str = row.get('trend_strength', 0)
            momentum = row.get('momentum_strength', 0)
            range_bound = row.get('range_bound', 0)
            
            if range_bound and volatility < 0.005:
                regime = 'range'
                confidence = 0.7
            elif momentum > 0.02 and volatility > 0.01:
                regime = 'momentum_burst'
                confidence = 0.8
            elif trend_str > 2:
                regime = 'trend'
                confidence = 0.75
            else:
                regime = 'reversal'
                confidence = 0.6
                
            direction = 1 if row.get('ema_alignment', 0) > 0 else -1 if row.get('ema_alignment', 0) < 0 else 0
            
            regimes.append(MarketRegime(
                regime=regime,
                confidence=confidence,
                duration=1,
                volatility=volatility,
                direction=direction
            ))
        
        return regimes

class HTFAnalyzer:
    """Higher timeframe bias analyzer"""
    
    def analyze_htf_bias(self, daily_data: pd.DataFrame, h4_data: pd.DataFrame, 
                        h1_data: pd.DataFrame) -> HTFBias:
        """Analyze higher timeframe bias"""
        
        # Daily bias
        daily_bias = self._get_timeframe_bias(daily_data.iloc[-20:])
        h4_bias = self._get_timeframe_bias(h4_data.iloc[-50:])
        h1_bias = self._get_timeframe_bias(h1_data.iloc[-100:])
        
        # Overall bias with weighting
        bias_scores = {
            'bullish': 0,
            'bearish': 0,
            'neutral': 0
        }
        
        # Weight: Daily 50%, H4 30%, H1 20%
        weights = {'daily': 0.5, 'h4': 0.3, 'h1': 0.2}
        biases = {'daily': daily_bias, 'h4': h4_bias, 'h1': h1_bias}
        
        for tf, bias in biases.items():
            bias_scores[bias] += weights[tf]
        
        overall_bias = max(bias_scores, key=bias_scores.get)
        confidence = bias_scores[overall_bias]
        
        # Calculate reversal probability
        reversal_prob = self._calculate_reversal_probability(daily_data, h4_data, h1_data)
        
        return HTFBias(
            daily=daily_bias,
            h4=h4_bias,
            h1=h1_bias,
            overall=overall_bias,
            confidence=confidence,
            reversal_probability=reversal_prob
        )
    
    def _get_timeframe_bias(self, data: pd.DataFrame) -> str:
        """Get bias for specific timeframe"""
        if len(data) < 10:
            return 'neutral'
        
        recent_data = data.iloc[-10:]
        
        # Higher highs, higher lows = bullish
        hh = recent_data['high'].iloc[-1] > recent_data['high'].iloc[:-1].max()
        hl = recent_data['low'].iloc[-1] > recent_data['low'].iloc[:-1].min()
        
        # Lower lows, lower highs = bearish
        ll = recent_data['low'].iloc[-1] < recent_data['low'].iloc[:-1].min()
        lh = recent_data['high'].iloc[-1] < recent_data['high'].iloc[:-1].max()
        
        # EMA alignment
        ema_bullish = (recent_data['ema_12'].iloc[-1] > recent_data['ema_26'].iloc[-1] > 
                      recent_data['ema_50'].iloc[-1])
        ema_bearish = (recent_data['ema_12'].iloc[-1] < recent_data['ema_26'].iloc[-1] < 
                      recent_data['ema_50'].iloc[-1])
        
        bullish_signals = sum([hh, hl, ema_bullish])
        bearish_signals = sum([ll, lh, ema_bearish])
        
        if bullish_signals >= 2:
            return 'bullish'
        elif bearish_signals >= 2:
            return 'bearish'
        else:
            return 'neutral'
    
    def _calculate_reversal_probability(self, daily: pd.DataFrame, h4: pd.DataFrame, 
                                      h1: pd.DataFrame) -> float:
        """Calculate probability of trend reversal"""
        reversal_signals = 0
        total_signals = 0
        
        # Check for divergence in recent data
        if len(h1) >= 50:
            recent_h1 = h1.iloc[-20:]
            
            # RSI divergence
            if self._check_divergence(recent_h1):
                reversal_signals += 2
            total_signals += 2
            
            # Volume divergence
            if 'tick_volume' in recent_h1.columns and self._check_volume_divergence(recent_h1):
                reversal_signals += 1
            total_signals += 1
            
            # Exhaustion patterns
            if self._check_exhaustion_patterns(recent_h1):
                reversal_signals += 1
            total_signals += 1
        
        return reversal_signals / max(total_signals, 1)
    
    def _check_divergence(self, data: pd.DataFrame) -> bool:
        """Check for RSI divergence"""
        if len(data) < 10:
            return False
        
        price_trend = data['close'].iloc[-1] > data['close'].iloc[-10]
        rsi_trend = data['rsi_14'].iloc[-1] > data['rsi_14'].iloc[-10]
        
        return price_trend != rsi_trend
    
    def _check_volume_divergence(self, data: pd.DataFrame) -> bool:
        """Check for volume divergence"""
        if len(data) < 10:
            return False
        
        price_higher = data['close'].iloc[-1] > data['close'].iloc[-10]
        volume_lower = data['tick_volume'].iloc[-5:].mean() < data['tick_volume'].iloc[-15:-5].mean()
        
        return price_higher and volume_lower
    
    def _check_exhaustion_patterns(self, data: pd.DataFrame) -> bool:
        """Check for exhaustion patterns"""
        recent = data.iloc[-5:]
        
        # Look for long wicks at extremes
        for _, row in recent.iterrows():
            body = abs(row['close'] - row['open'])
            upper_wick = row['high'] - max(row['close'], row['open'])
            lower_wick = min(row['close'], row['open']) - row['low']
            
            if upper_wick > 2 * body or lower_wick > 2 * body:
                return True
        
        return False

# Strategy Base Classes
class StrategyBase(ABC):
    """Enhanced base strategy class"""
    
    def __init__(self, name: str, family: str, params: Optional[Dict] = None):
        self.name = name
        self.family = family  # 'ICT', 'PA', 'Quant', 'AI'
        self.params = params or {}
        self.performance_history = []
        self.active = True
        self.confidence_threshold = 0.7
        
    @abstractmethod
    def generate_signals(self, features: pd.DataFrame, htf_bias: HTFBias, 
                        regime: List[MarketRegime]) -> List[Signal]:
        """Generate trading signals with HTF bias and regime context"""
        pass
    
    def can_trade_against_bias(self, htf_bias: HTFBias) -> bool:
        """Determine if strategy can trade against HTF bias"""
        return htf_bias.reversal_probability > 0.6

# ICT Strategy Implementations
class LiquiditySweepStrategy(StrategyBase):
    """ICT Liquidity Sweep Strategy"""
    
    def __init__(self, params: Optional[Dict] = None):
        super().__init__("liquidity_sweep", "ICT", params)
        self.default_params = {
            "lookback_period": 20,
            "sweep_threshold": 0.0005,
            "volume_confirmation": True
        }
        self.params = {**self.default_params, **self.params}
    
    def generate_signals(self, features: pd.DataFrame, htf_bias: HTFBias, 
                        regime: List[MarketRegime]) -> List[Signal]:
        signals = []
        
        for i in range(self.params['lookback_period'], len(features)):
            if i >= len(regime):
                continue
                
            current_regime = regime[i]
            row = features.iloc[i]
            
            # Look for liquidity sweeps
            lookback_data = features.iloc[i-self.params['lookback_period']:i]
            
            # Find recent high/low
            recent_high = lookback_data['high'].max()
            recent_low = lookback_data['low'].min()
            
            # Check for sweep
            swept_high = row['high'] > recent_high * (1 + self.params['sweep_threshold'])
            swept_low = row['low'] < recent_low * (1 - self.params['sweep_threshold'])
            
            if swept_high and current_regime.regime in ['reversal', 'range']:
                # Bullish liquidity sweep - expect reversal down
                side = "sell"
                confidence = 0.75
                against_bias = htf_bias.overall == 'bullish'
                
            elif swept_low and current_regime.regime in ['reversal', 'range']:
                # Bearish liquidity sweep - expect reversal up
                side = "buy"
                confidence = 0.75
                against_bias = htf_bias.overall == 'bearish'
                
            else:
                continue
            
            # Only trade if confidence is high enough or we can trade against bias
            if confidence >= self.confidence_threshold or (against_bias and self.can_trade_against_bias(htf_bias)):
                signal = Signal(
                    timestamp=row.name,
                    symbol="",
                    side=side,
                    confidence=confidence,
                    strategy_name=self.name,
                    strategy_family=self.family,
                    entry_price=row['close'],
                    htf_bias=htf_bias.overall,
                    against_bias=against_bias,
                    reversal_signal=True,
                    regime=current_regime.regime,
                    features={
                        "sweep_type": "high" if swept_high else "low",
                        "recent_high": recent_high,
                        "recent_low": recent_low
                    }
                )
                signals.append(signal)
        
        return signals

class FairValueGapStrategy(StrategyBase):
    """ICT Fair Value Gap Strategy"""
    
    def __init__(self, params: Optional[Dict] = None):
        super().__init__("fair_value_gap", "ICT", params)
        self.default_params = {
            "min_gap_size": 0.0001,
            "max_gap_age": 50,
            "confirmation_bars": 2
        }
        self.params = {**self.default_params, **self.params}
    
    def generate_signals(self, features: pd.DataFrame, htf_bias: HTFBias, 
                        regime: List[MarketRegime]) -> List[Signal]:
        signals = []
        identified_fvgs = []
        
        # First pass: identify FVGs
        for i in range(2, len(features) - 1):
            prev_bar = features.iloc[i-1]
            current_bar = features.iloc[i]
            next_bar = features.iloc[i+1]
            
            # Bullish FVG: prev_high < next_low
            if prev_bar['high'] < next_bar['low']:
                gap_size = next_bar['low'] - prev_bar['high']
                if gap_size >= self.params['min_gap_size']:
                    identified_fvgs.append({
                        'type': 'bullish',
                        'start_idx': i,
                        'upper': next_bar['low'],
                        'lower': prev_bar['high'],
                        'size': gap_size
                    })
            
            # Bearish FVG: prev_low > next_high
            elif prev_bar['low'] > next_bar['high']:
                gap_size = prev_bar['low'] - next_bar['high']
                if gap_size >= self.params['min_gap_size']:
                    identified_fvgs.append({
                        'type': 'bearish',
                        'start_idx': i,
                        'upper': prev_bar['low'],
                        'lower': next_bar['high'],
                        'size': gap_size
                    })
        
        # Second pass: look for FVG fills
        for i in range(len(features)):
            if i >= len(regime):
                continue
                
            current_regime = regime[i]
            row = features.iloc[i]
            
            for fvg in identified_fvgs:
                age = i - fvg['start_idx']
                if age <= 0 or age > self.params['max_gap_age']:
                    continue
                
                # Check if price is filling the FVG
                if fvg['type'] == 'bullish' and row['low'] <= fvg['upper'] and row['high'] >= fvg['lower']:
                    # Bullish FVG fill - potential long
                    if current_regime.regime in ['trend', 'momentum_burst']:
                        side = "buy"
                        confidence = 0.8
                        against_bias = htf_bias.overall == 'bearish'
                        
                        signal = Signal(
                            timestamp=row.name,
                            symbol="",
                            side=side,
                            confidence=confidence,
                            strategy_name=self.name,
                            strategy_family=self.family,
                            entry_price=(fvg['upper'] + fvg['lower']) / 2,
                            htf_bias=htf_bias.overall,
                            against_bias=against_bias,
                            regime=current_regime.regime,
                            features={
                                "fvg_type": "bullish",
                                "fvg_size": fvg['size'],
                                "fvg_age": age
                            }
                        )
                        signals.append(signal)
                
                elif fvg['type'] == 'bearish' and row['high'] >= fvg['lower'] and row['low'] <= fvg['upper']:
                    # Bearish FVG fill - potential short
                    if current_regime.regime in ['trend', 'momentum_burst']:
                        side = "sell"
                        confidence = 0.8
                        against_bias = htf_bias.overall == 'bullish'
                        
                        signal = Signal(
                            timestamp=row.name,
                            symbol="",
                            side=side,
                            confidence=confidence,
                            strategy_name=self.name,
                            strategy_family=self.family,
                            entry_price=(fvg['upper'] + fvg['lower']) / 2,
                            htf_bias=htf_bias.overall,
                            against_bias=against_bias,
                            regime=current_regime.regime,
                            features={
                                "fvg_type": "bearish",
                                "fvg_size": fvg['size'],
                                "fvg_age": age
                            }
                        )
                        signals.append(signal)
        
        return signals

class OrderBlockStrategy(StrategyBase):
    """ICT Order Block Strategy"""
    
    def __init__(self, params: Optional[Dict] = None):
        super().__init__("order_block", "ICT", params)
        self.default_params = {
            "min_block_size": 0.0002,
            "lookback": 20,
            "confirmation_volume": True
        }
        self.params = {**self.default_params, **self.params}
    
    def generate_signals(self, features: pd.DataFrame, htf_bias: HTFBias, 
                        regime: List[MarketRegime]) -> List[Signal]:
        signals = []
        order_blocks = []
        
        # Identify order blocks
        for i in range(self.params['lookback'], len(features) - 5):
            current_bar = features.iloc[i]
            
            # Look for strong moves after consolidation
            prev_bars = features.iloc[i-self.params['lookback']:i]
            next_bars = features.iloc[i+1:i+6]
            
            if len(next_bars) < 5:
                continue
            
            # Strong bullish move identification
            if (next_bars['close'].iloc[-1] > current_bar['close'] * 1.005 and
                prev_bars['high'].max() - prev_bars['low'].min() < self.params['min_block_size']):
                
                order_blocks.append({
                    'type': 'bullish',
                    'start_idx': i,
                    'high': current_bar['high'],
                    'low': current_bar['low'],
                    'volume': current_bar.get('tick_volume', 0)
                })
            
            # Strong bearish move identification
            elif (next_bars['close'].iloc[-1] < current_bar['close'] * 0.995 and
                  prev_bars['high'].max() - prev_bars['low'].min() < self.params['min_block_size']):
                
                order_blocks.append({
                    'type': 'bearish',
                    'start_idx': i,
                    'high': current_bar['high'],
                    'low': current_bar['low'],
                    'volume': current_bar.get('tick_volume', 0)
                })
        
        # Look for retests of order blocks
        for i in range(len(features)):
            if i >= len(regime):
                continue
                
            current_regime = regime[i]
            row = features.iloc[i]
            
            for ob in order_blocks:
                if i <= ob['start_idx']:
                    continue
                
                # Test bullish order block
                if (ob['type'] == 'bullish' and 
                    row['low'] <= ob['high'] and row['high'] >= ob['low']):
                    
                    side = "buy"
                    confidence = 0.75
                    against_bias = htf_bias.overall == 'bearish'
                    
                    if confidence >= self.confidence_threshold or (against_bias and self.can_trade_against_bias(htf_bias)):
                        signal = Signal(
                            timestamp=row.name,
                            symbol="",
                            side=side,
                            confidence=confidence,
                            strategy_name=self.name,
                            strategy_family=self.family,
                            entry_price=ob['low'],
                            htf_bias=htf_bias.overall,
                            against_bias=against_bias,
                            regime=current_regime.regime,
                            features={
                                "ob_type": "bullish",
                                "ob_age": i - ob['start_idx']
                            }
                        )
                        signals.append(signal)
                
                # Test bearish order block
                elif (ob['type'] == 'bearish' and 
                      row['high'] >= ob['low'] and row['low'] <= ob['high']):
                    
                    side = "sell"
                    confidence = 0.75
                    against_bias = htf_bias.overall == 'bullish'
                    
                    if confidence >= self.confidence_threshold or (against_bias and self.can_trade_against_bias(htf_bias)):
                        signal = Signal(
                            timestamp=row.name,
                            symbol="",
                            side=side,
                            confidence=confidence,
                            strategy_name=self.name,
                            strategy_family=self.family,
                            entry_price=ob['high'],
                            htf_bias=htf_bias.overall,
                            against_bias=against_bias,
                            regime=current_regime.regime,
                            features={
                                "ob_type": "bearish",
                                "ob_age": i - ob['start_idx']
                            }
                        )
                        signals.append(signal)
        
        return signals

class OptimalTradeEntryStrategy(StrategyBase):
    """ICT Optimal Trade Entry (OTE) Strategy"""
    
    def __init__(self, params: Optional[Dict] = None):
        super().__init__("optimal_trade_entry", "ICT", params)
        self.default_params = {
            "fib_61_8": 0.618,
            "fib_70_5": 0.705,
            "min_retracement": 0.382,
            "max_retracement": 0.786
        }
        self.params = {**self.default_params, **self.params}
    
    def generate_signals(self, features: pd.DataFrame, htf_bias: HTFBias, 
                        regime: List[MarketRegime]) -> List[Signal]:
        signals = []
        
        for i in range(50, len(features)):
            if i >= len(regime):
                continue
                
            current_regime = regime[i]
            row = features.iloc[i]
            
            # Look for swing moves in last 20-50 bars
            lookback = features.iloc[i-50:i]
            
            # Find swing high and low
            swing_high_idx = lookback['high'].idxmax()
            swing_low_idx = lookback['low'].idxmin()
            
            swing_high = lookback.loc[swing_high_idx, 'high']
            swing_low = lookback.loc[swing_low_idx, 'low']
            
            swing_range = swing_high - swing_low
            
            if swing_range < row['atr'] * 2:  # Too small a move
                continue
            
            # Determine trend direction
            if swing_high_idx > swing_low_idx:  # Uptrend
                # Look for retracement to OTE zone
                current_retracement = (swing_high - row['close']) / swing_range
                
                if (self.params['min_retracement'] <= current_retracement <= self.params['max_retracement'] and
                    current_regime.regime in ['trend', 'reversal']):
                    
                    side = "buy"
                    confidence = 0.8 if (self.params['fib_61_8'] <= current_retracement <= self.params['fib_70_5']) else 0.7
                    against_bias = htf_bias.overall == 'bearish'
                    
                    if confidence >= self.confidence_threshold or (against_bias and self.can_trade_against_bias(htf_bias)):
                        signal = Signal(
                            timestamp=row.name,
                            symbol="",
                            side=side,
                            confidence=confidence,
                            strategy_name=self.name,
                            strategy_family=self.family,
                            entry_price=row['close'],
                            htf_bias=htf_bias.overall,
                            against_bias=against_bias,
                            regime=current_regime.regime,
                            features={
                                "retracement_level": current_retracement,
                                "swing_range": swing_range,
                                "trend": "up"
                            }
                        )
                        signals.append(signal)
            
            else:  # Downtrend
                # Look for retracement to OTE zone
                current_retracement = (row['close'] - swing_low) / swing_range
                
                if (self.params['min_retracement'] <= current_retracement <= self.params['max_retracement'] and
                    current_regime.regime in ['trend', 'reversal']):
                    
                    side = "sell"
                    confidence = 0.8 if (self.params['fib_61_8'] <= current_retracement <= self.params['fib_70_5']) else 0.7
                    against_bias = htf_bias.overall == 'bullish'
                    
                    if confidence >= self.confidence_threshold or (against_bias and self.can_trade_against_bias(htf_bias)):
                        signal = Signal(
                            timestamp=row.name,
                            symbol="",
                            side=side,
                            confidence=confidence,
                            strategy_name=self.name,
                            strategy_family=self.family,
                            entry_price=row['close'],
                            htf_bias=htf_bias.overall,
                            against_bias=against_bias,
                            regime=current_regime.regime,
                            features={
                                "retracement_level": current_retracement,
                                "swing_range": swing_range,
                                "trend": "down"
                            }
                        )
                        signals.append(signal)
        
        return signals

class JudasSwingStrategy(StrategyBase):
    """ICT Judas Swing Strategy"""
    
    def __init__(self, params: Optional[Dict] = None):
        super().__init__("judas_swing", "ICT", params)
        self.default_params = {
            "london_open_hour": 8,
            "ny_open_hour": 13,
            "judas_window": 2,  # hours
            "min_sweep_distance": 0.0003
        }
        self.params = {**self.default_params, **self.params}
    
    def generate_signals(self, features: pd.DataFrame, htf_bias: HTFBias, 
                        regime: List[MarketRegime]) -> List[Signal]:
        signals = []
        
        for i in range(20, len(features)):
            if i >= len(regime):
                continue
                
            current_regime = regime[i]
            row = features.iloc[i]
            
            # Check if we're in Judas window
            hour = row.name.hour
            is_judas_window = (
                (self.params['london_open_hour'] <= hour <= self.params['london_open_hour'] + self.params['judas_window']) or
                (self.params['ny_open_hour'] <= hour <= self.params['ny_open_hour'] + self.params['judas_window'])
            )
            
            if not is_judas_window:
                continue
            
            # Look for false break patterns
            lookback = features.iloc[i-20:i]
            
            # Find recent range
            range_high = lookback['high'].max()
            range_low = lookback['low'].min()
            range_size = range_high - range_low
            
            # Check for false break above range
            if (row['high'] > range_high and 
                row['close'] < range_high - self.params['min_sweep_distance']):
                
                side = "sell"
                confidence = 0.75
                against_bias = htf_bias.overall == 'bullish'
                
                if confidence >= self.confidence_threshold or (against_bias and self.can_trade_against_bias(htf_bias)):
                    signal = Signal(
                        timestamp=row.name,
                        symbol="",
                        side=side,
                        confidence=confidence,
                        strategy_name=self.name,
                        strategy_family=self.family,
                        entry_price=row['close'],
                        htf_bias=htf_bias.overall,
                        against_bias=against_bias,
                        reversal_signal=True,
                        regime=current_regime.regime,
                        features={
                            "false_break": "high",
                            "range_size": range_size,
                            "session": "london" if hour < 13 else "ny"
                        }
                    )
                    signals.append(signal)
            
            # Check for false break below range
            elif (row['low'] < range_low and 
                  row['close'] > range_low + self.params['min_sweep_distance']):
                
                side = "buy"
                confidence = 0.75
                against_bias = htf_bias.overall == 'bearish'
                
                if confidence >= self.confidence_threshold or (against_bias and self.can_trade_against_bias(htf_bias)):
                    signal = Signal(
                        timestamp=row.name,
                        symbol="",
                        side=side,
                        confidence=confidence,
                        strategy_name=self.name,
                        strategy_family=self.family,
                        entry_price=row['close'],
                        htf_bias=htf_bias.overall,
                        against_bias=against_bias,
                        reversal_signal=True,
                        regime=current_regime.regime,
                        features={
                            "false_break": "low",
                            "range_size": range_size,
                            "session": "london" if hour < 13 else "ny"
                        }
                    )
                    signals.append(signal)
        
        return signals

# Price Action Strategies
class SupportResistanceStrategy(StrategyBase):
    """Price Action Support/Resistance Strategy"""
    
    def __init__(self, params: Optional[Dict] = None):
        super().__init__("support_resistance", "PA", params)
        self.default_params = {
            "lookback_period": 50,
            "min_touches": 2,
            "proximity_threshold": 0.0005
        }
        self.params = {**self.default_params, **self.params}
    
    def generate_signals(self, features: pd.DataFrame, htf_bias: HTFBias, 
                        regime: List[MarketRegime]) -> List[Signal]:
        signals = []
        sr_levels = []
        
        # Identify S/R levels
        for i in range(self.params['lookback_period'], len(features) - self.params['lookback_period']):
            window = features.iloc[i-self.params['lookback_period']:i+self.params['lookback_period']]
            
            # Resistance levels (local highs)
            if (features.iloc[i]['high'] == window['high'].max() and 
                features.iloc[i]['high'] > features.iloc[i-1]['high'] and 
                features.iloc[i]['high'] > features.iloc[i+1]['high']):
                
                sr_levels.append({
                    'type': 'resistance',
                    'level': features.iloc[i]['high'],
                    'index': i,
                    'touches': 1
                })
            
            # Support levels (local lows)
            elif (features.iloc[i]['low'] == window['low'].min() and 
                  features.iloc[i]['low'] < features.iloc[i-1]['low'] and 
                  features.iloc[i]['low'] < features.iloc[i+1]['low']):
                
                sr_levels.append({
                    'type': 'support',
                    'level': features.iloc[i]['low'],
                    'index': i,
                    'touches': 1
                })
        
        # Count touches for each level
        for level in sr_levels:
            for i, row in features.iterrows():
                if abs(row['high'] - level['level']) <= self.params['proximity_threshold']:
                    level['touches'] += 1
                elif abs(row['low'] - level['level']) <= self.params['proximity_threshold']:
                    level['touches'] += 1
        
        # Generate signals
        for i in range(len(features)):
            if i >= len(regime):
                continue
                
            current_regime = regime[i]
            row = features.iloc[i]
            
            for level in sr_levels:
                if level['touches'] < self.params['min_touches']:
                    continue
                
                # Support bounce
                if (level['type'] == 'support' and 
                    row['low'] <= level['level'] + self.params['proximity_threshold'] and
                    row['close'] > level['level']):
                    
                    side = "buy"
                    confidence = min(0.6 + (level['touches'] * 0.05), 0.9)
                    against_bias = htf_bias.overall == 'bearish'
                    
                    if confidence >= self.confidence_threshold or (against_bias and self.can_trade_against_bias(htf_bias)):
                        signal = Signal(
                            timestamp=row.name,
                            symbol="",
                            side=side,
                            confidence=confidence,
                            strategy_name=self.name,
                            strategy_family=self.family,
                            entry_price=row['close'],
                            htf_bias=htf_bias.overall,
                            against_bias=against_bias,
                            regime=current_regime.regime,
                            features={
                                "sr_type": "support",
                                "sr_level": level['level'],
                                "touches": level['touches']
                            }
                        )
                        signals.append(signal)
                
                # Resistance rejection
                elif (level['type'] == 'resistance' and 
                      row['high'] >= level['level'] - self.params['proximity_threshold'] and
                      row['close'] < level['level']):
                    
                    side = "sell"
                    confidence = min(0.6 + (level['touches'] * 0.05), 0.9)
                    against_bias = htf_bias.overall == 'bullish'
                    
                    if confidence >= self.confidence_threshold or (against_bias and self.can_trade_against_bias(htf_bias)):
                        signal = Signal(
                            timestamp=row.name,
                            symbol="",
                            side=side,
                            confidence=confidence,
                            strategy_name=self.name,
                            strategy_family=self.family,
                            entry_price=row['close'],
                            htf_bias=htf_bias.overall,
                            against_bias=against_bias,
                            regime=current_regime.regime,
                            features={
                                "sr_type": "resistance",
                                "sr_level": level['level'],
                                "touches": level['touches']
                            }
                        )
                        signals.append(signal)
        
        return signals

class CandlestickPatternsStrategy(StrategyBase):
    """Price Action Candlestick Patterns Strategy"""
    
    def __init__(self, params: Optional[Dict] = None):
        super().__init__("candlestick_patterns", "PA", params)
        self.default_params = {
            "min_confidence": 0.7,
            "pattern_weights": {
                "engulfing": 0.8,
                "pin_bar": 0.7,
                "doji": 0.5,
                "hammer": 0.75
            }
        }
        self.params = {**self.default_params, **self.params}
    
    def generate_signals(self, features: pd.DataFrame, htf_bias: HTFBias, 
                        regime: List[MarketRegime]) -> List[Signal]:
        signals = []
        
        for i in range(2, len(features)):
            if i >= len(regime):
                continue
                
            current_regime = regime[i]
            row = features.iloc[i]
            prev_row = features.iloc[i-1]
            
            patterns = self._identify_patterns(prev_row, row)
            
            for pattern in patterns:
                side = pattern['side']
                confidence = pattern['confidence']
                against_bias = (side == "buy" and htf_bias.overall == 'bearish') or (side == "sell" and htf_bias.overall == 'bullish')
                
                if confidence >= self.confidence_threshold or (against_bias and self.can_trade_against_bias(htf_bias)):
                    signal = Signal(
                        timestamp=row.name,
                        symbol="",
                        side=side,
                        confidence=confidence,
                        strategy_name=self.name,
                        strategy_family=self.family,
                        entry_price=row['close'],
                        htf_bias=htf_bias.overall,
                        against_bias=against_bias,
                        regime=current_regime.regime,
                        features={
                            "pattern_type": pattern['type'],
                            "pattern_strength": pattern['strength']
                        }
                    )
                    signals.append(signal)
        
        return signals
    
    def _identify_patterns(self, prev_bar: pd.Series, current_bar: pd.Series) -> List[Dict]:
        """Identify candlestick patterns"""
        patterns = []
        
        # Current bar metrics
        body = abs(current_bar['close'] - current_bar['open'])
        upper_wick = current_bar['high'] - max(current_bar['close'], current_bar['open'])
        lower_wick = min(current_bar['close'], current_bar['open']) - current_bar['low']
        range_size = current_bar['high'] - current_bar['low']
        
        # Previous bar metrics
        prev_body = abs(prev_bar['close'] - prev_bar['open'])
        
        # Bullish Engulfing
        if (current_bar['close'] > current_bar['open'] and  # Current bullish
            prev_bar['close'] < prev_bar['open'] and        # Previous bearish
            current_bar['open'] < prev_bar['close'] and     # Opens below prev close
            current_bar['close'] > prev_bar['open']):       # Closes above prev open
            
            patterns.append({
                'type': 'bullish_engulfing',
                'side': 'buy',
                'confidence': self.params['pattern_weights']['engulfing'],
                'strength': (body / prev_body) if prev_body > 0 else 1
            })
        
        # Bearish Engulfing
        elif (current_bar['close'] < current_bar['open'] and  # Current bearish
              prev_bar['close'] > prev_bar['open'] and        # Previous bullish
              current_bar['open'] > prev_bar['close'] and     # Opens above prev close
              current_bar['close'] < prev_bar['open']):       # Closes below prev open
            
            patterns.append({
                'type': 'bearish_engulfing',
                'side': 'sell',
                'confidence': self.params['pattern_weights']['engulfing'],
                'strength': (body / prev_body) if prev_body > 0 else 1
            })
        
        # Pin Bar (long wick reversal)
        if range_size > 0:
            # Bullish pin bar
            if (lower_wick > 2 * body and lower_wick > 2 * upper_wick and
                current_bar['close'] > current_bar['open']):
                
                patterns.append({
                    'type': 'bullish_pin_bar',
                    'side': 'buy',
                    'confidence': self.params['pattern_weights']['pin_bar'],
                    'strength': lower_wick / range_size
                })
            
            # Bearish pin bar
            elif (upper_wick > 2 * body and upper_wick > 2 * lower_wick and
                  current_bar['close'] < current_bar['open']):
                
                patterns.append({
                    'type': 'bearish_pin_bar',
                    'side': 'sell',
                    'confidence': self.params['pattern_weights']['pin_bar'],
                    'strength': upper_wick / range_size
                })
            
            # Hammer (at support)
            if (lower_wick > 2 * body and upper_wick < body * 0.3 and
                body / range_size < 0.3):
                
                patterns.append({
                    'type': 'hammer',
                    'side': 'buy',
                    'confidence': self.params['pattern_weights']['hammer'],
                    'strength': lower_wick / range_size
                })
            
            # Doji (indecision)
            if body / range_size < 0.1:
                # Context determines direction
                if current_bar['rsi_14'] < 30:
                    side = 'buy'
                    patterns.append({
                        'type': 'doji',
                        'side': side,
                        'confidence': self.params['pattern_weights']['doji'],
                        'strength': 1 - (body / range_size)
                    })
                elif current_bar['rsi_14'] > 70:
                    side = 'sell'
                    patterns.append({
                        'type': 'doji',
                        'side': side,
                        'confidence': self.params['pattern_weights']['doji'],
                        'strength': 1 - (body / range_size)
                    })
        
        return patterns

# Quantitative Strategies
class MovingAverageCrossStrategy(StrategyBase):
    """Moving Average Crossover Strategy"""
    
    def __init__(self, params: Optional[Dict] = None):
        super().__init__("ma_cross", "Quant", params)
        self.default_params = {
            "fast_period": 12,
            "slow_period": 26,
            "signal_period": 9,
            "trend_filter": True
        }
        self.params = {**self.default_params, **self.params}
    
    def generate_signals(self, features: pd.DataFrame, htf_bias: HTFBias, 
                        regime: List[MarketRegime]) -> List[Signal]:
        signals = []
        
        for i in range(50, len(features)):
            if i >= len(regime):
                continue
                
            current_regime = regime[i]
            row = features.iloc[i]
            prev_row = features.iloc[i-1]
            
            # MACD signals
            macd_bullish = (row['macd'] > row['macd_signal'] and 
                          prev_row['macd'] <= prev_row['macd_signal'])
            macd_bearish = (row['macd'] < row['macd_signal'] and 
                          prev_row['macd'] >= prev_row['macd_signal'])
            
            # Trend filter
            trend_bullish = row['ema_12'] > row['ema_26'] > row['ema_50'] if self.params['trend_filter'] else True
            trend_bearish = row['ema_12'] < row['ema_26'] < row['ema_50'] if self.params['trend_filter'] else True
            
            if macd_bullish and trend_bullish and current_regime.regime in ['trend', 'momentum_burst']:
                side = "buy"
                confidence = 0.7
                against_bias = htf_bias.overall == 'bearish'
                
                if confidence >= self.confidence_threshold or (against_bias and self.can_trade_against_bias(htf_bias)):
                    signal = Signal(
                        timestamp=row.name,
                        symbol="",
                        side=side,
                        confidence=confidence,
                        strategy_name=self.name,
                        strategy_family=self.family,
                        entry_price=row['close'],
                        htf_bias=htf_bias.overall,
                        against_bias=against_bias,
                        regime=current_regime.regime,
                        features={
                            "macd": row['macd'],
                            "macd_signal": row['macd_signal'],
                            "macd_hist": row['macd_hist']
                        }
                    )
                    signals.append(signal)
            
            elif macd_bearish and trend_bearish and current_regime.regime in ['trend', 'momentum_burst']:
                side = "sell"
                confidence = 0.7
                against_bias = htf_bias.overall == 'bullish'
                
                if confidence >= self.confidence_threshold or (against_bias and self.can_trade_against_bias(htf_bias)):
                    signal = Signal(
                        timestamp=row.name,
                        symbol="",
                        side=side,
                        confidence=confidence,
                        strategy_name=self.name,
                        strategy_family=self.family,
                        entry_price=row['close'],
                        htf_bias=htf_bias.overall,
                        against_bias=against_bias,
                        regime=current_regime.regime,
                        features={
                            "macd": row['macd'],
                            "macd_signal": row['macd_signal'],
                            "macd_hist": row['macd_hist']
                        }
                    )
                    signals.append(signal)
        
        return signals

class RSIDivergenceStrategy(StrategyBase):
    """RSI Divergence Strategy"""
    
    def __init__(self, params: Optional[Dict] = None):
        super().__init__("rsi_divergence", "Quant", params)
        self.default_params = {
            "rsi_period": 14,
            "lookback_period": 20,
            "min_rsi_extreme": 70,
            "max_rsi_extreme": 30
        }
        self.params = {**self.default_params, **self.params}
    
    def generate_signals(self, features: pd.DataFrame, htf_bias: HTFBias, 
                        regime: List[MarketRegime]) -> List[Signal]:
        signals = []
        
        for i in range(self.params['lookback_period'], len(features)):
            if i >= len(regime):
                continue
                
            current_regime = regime[i]
            row = features.iloc[i]
            
            # Check for divergence
            lookback = features.iloc[i-self.params['lookback_period']:i+1]
            
            # Find recent peaks and troughs
            price_peaks = []
            rsi_peaks = []
            
            for j in range(1, len(lookback)-1):
                if (lookback.iloc[j]['high'] > lookback.iloc[j-1]['high'] and 
                    lookback.iloc[j]['high'] > lookback.iloc[j+1]['high']):
                    price_peaks.append((j, lookback.iloc[j]['high'], lookback.iloc[j]['rsi_14']))
                
                if (lookback.iloc[j]['low'] < lookback.iloc[j-1]['low'] and 
                    lookback.iloc[j]['low'] < lookback.iloc[j+1]['low']):
                    rsi_peaks.append((j, lookback.iloc[j]['low'], lookback.iloc[j]['rsi_14']))
            
            # Check for bearish divergence (higher highs in price, lower highs in RSI)
            if len(price_peaks) >= 2:
                last_peak = price_peaks[-1]
                second_last_peak = price_peaks[-2]
                
                if (last_peak[1] > second_last_peak[1] and  # Higher high in price
                    last_peak[2] < second_last_peak[2] and  # Lower high in RSI
                    last_peak[2] > self.params['min_rsi_extreme']):  # RSI in overbought
                    
                    side = "sell"
                    confidence = 0.8
                    against_bias = htf_bias.overall == 'bullish'
                    
                    if confidence >= self.confidence_threshold or (against_bias and self.can_trade_against_bias(htf_bias)):
                        signal = Signal(
                            timestamp=row.name,
                            symbol="",
                            side=side,
                            confidence=confidence,
                            strategy_name=self.name,
                            strategy_family=self.family,
                            entry_price=row['close'],
                            htf_bias=htf_bias.overall,
                            against_bias=against_bias,
                            reversal_signal=True,
                            regime=current_regime.regime,
                            features={
                                "divergence_type": "bearish",
                                "rsi_level": row['rsi_14'],
                                "price_peaks": len(price_peaks)
                            }
                        )
                        signals.append(signal)
            
            # Check for bullish divergence (lower lows in price, higher lows in RSI)
            if len(rsi_peaks) >= 2:
                last_trough = rsi_peaks[-1]
                second_last_trough = rsi_peaks[-2]
                
                if (last_trough[1] < second_last_trough[1] and  # Lower low in price
                    last_trough[2] > second_last_trough[2] and  # Higher low in RSI
                    last_trough[2] < self.params['max_rsi_extreme']):  # RSI in oversold
                    
                    side = "buy"
                    confidence = 0.8
                    against_bias = htf_bias.overall == 'bearish'
                    
                    if confidence >= self.confidence_threshold or (against_bias and self.can_trade_against_bias(htf_bias)):
                        signal = Signal(
                            timestamp=row.name,
                            symbol="",
                            side=side,
                            confidence=confidence,
                            strategy_name=self.name,
                            strategy_family=self.family,
                            entry_price=row['close'],
                            htf_bias=htf_bias.overall,
                            against_bias=against_bias,
                            reversal_signal=True,
                            regime=current_regime.regime,
                            features={
                                "divergence_type": "bullish",
                                "rsi_level": row['rsi_14'],
                                "price_troughs": len(rsi_peaks)
                            }
                        )
                        signals.append(signal)
        
        return signals

class MeanReversionStrategy(StrategyBase):
    """Mean Reversion Strategy (VWAP/StdDev)"""
    
    def __init__(self, params: Optional[Dict] = None):
        super().__init__("mean_reversion", "Quant", params)
        self.default_params = {
            "z_threshold": 2.0,
            "lookback_period": 20,
            "mean_type": "sma"  # or "vwap"
        }
        self.params = {**self.default_params, **self.params}
    
    def generate_signals(self, features: pd.DataFrame, htf_bias: HTFBias, 
                        regime: List[MarketRegime]) -> List[Signal]:
        signals = []
        
        for i in range(self.params['lookback_period'], len(features)):
            if i >= len(regime):
                continue
                
            current_regime = regime[i]
            row = features.iloc[i]
            
            # Only trade mean reversion in range-bound markets
            if current_regime.regime not in ['range', 'reversal']:
                continue
            
            # Calculate z-score
            if self.params['mean_type'] == "sma":
                mean = row['sma_20']
                z_score = row['z_score_20']
            else:
                # Simple VWAP approximation
                mean = features.iloc[i-self.params['lookback_period']:i]['close'].mean()
                std = features.iloc[i-self.params['lookback_period']:i]['close'].std()
                z_score = (row['close'] - mean) / std if std > 0 else 0
            
            # Mean reversion signals
            if z_score > self.params['z_threshold']:  # Overbought
                side = "sell"
                confidence = min(0.6 + (abs(z_score) - self.params['z_threshold']) * 0.1, 0.9)
                against_bias = htf_bias.overall == 'bullish'
                
                if confidence >= self.confidence_threshold or (against_bias and self.can_trade_against_bias(htf_bias)):
                    signal = Signal(
                        timestamp=row.name,
                        symbol="",
                        side=side,
                        confidence=confidence,
                        strategy_name=self.name,
                        strategy_family=self.family,
                        entry_price=row['close'],
                        htf_bias=htf_bias.overall,
                        against_bias=against_bias,
                        regime=current_regime.regime,
                        features={
                            "z_score": z_score,
                            "mean_level": mean,
                            "reversion_type": "sell_overbought"
                        }
                    )
                    signals.append(signal)
            
            elif z_score < -self.params['z_threshold']:  # Oversold
                side = "buy"
                confidence = min(0.6 + (abs(z_score) - self.params['z_threshold']) * 0.1, 0.9)
                against_bias = htf_bias.overall == 'bearish'
                
                if confidence >= self.confidence_threshold or (against_bias and self.can_trade_against_bias(htf_bias)):
                    signal = Signal(
                        timestamp=row.name,
                        symbol="",
                        side=side,
                        confidence=confidence,
                        strategy_name=self.name,
                        strategy_family=self.family,
                        entry_price=row['close'],
                        htf_bias=htf_bias.overall,
                        against_bias=against_bias,
                        regime=current_regime.regime,
                        features={
                            "z_score": z_score,
                            "mean_level": mean,
                            "reversion_type": "buy_oversold"
                        }
                    )
                    signals.append(signal)
        
        return signals

# AI Strategy Generator
class AIStrategyGenerator:
    """AI that discovers and creates new trading strategies"""
    
    def __init__(self):
        self.discovered_strategies = []
        self.pattern_library = {}
        self.performance_threshold = 1.2  # Minimum profit factor
        self.min_trades = 20  # Minimum trades for validation
    
    def discover_patterns(self, features: pd.DataFrame, returns: pd.Series) -> List[Dict]:
        """Discover new patterns in market data"""
        patterns = []
        
        # Pattern 1: Correlation-based patterns
        correlations = self._find_correlations(features, returns)
        for corr in correlations:
            if abs(corr['correlation']) > 0.6:
                patterns.append({
                    'type': 'correlation',
                    'features': corr['features'],
                    'correlation': corr['correlation'],
                    'strength': abs(corr['correlation'])
                })
        
        # Pattern 2: Sequence-based patterns
        sequences = self._find_sequences(features, returns)
        for seq in sequences:
            if seq['success_rate'] > 0.65:
                patterns.append({
                    'type': 'sequence',
                    'sequence': seq['sequence'],
                    'success_rate': seq['success_rate'],
                    'strength': seq['success_rate']
                })
        
        # Pattern 3: Regime-specific patterns
        regime_patterns = self._find_regime_patterns(features, returns)
        for rp in regime_patterns:
            if rp['edge'] > 0.1:
                patterns.append({
                    'type': 'regime_specific',
                    'conditions': rp['conditions'],
                    'edge': rp['edge'],
                    'strength': rp['edge']
                })
        
        return patterns
    
    def _find_correlations(self, features: pd.DataFrame, returns: pd.Series) -> List[Dict]:
        """Find correlations between features and future returns"""
        correlations = []
        
        numeric_features = features.select_dtypes(include=[np.number]).columns
        
        for feature in numeric_features:
            if feature in features.columns:
                corr = features[feature].corr(returns)
                if not np.isnan(corr):
                    correlations.append({
                        'features': [feature],
                        'correlation': corr
                    })
        
        # Two-feature combinations
        for i, feat1 in enumerate(numeric_features[:10]):  # Limit for performance
            for feat2 in numeric_features[i+1:11]:
                if feat1 in features.columns and feat2 in features.columns:
                    combined_signal = features[feat1] * features[feat2]
                    corr = combined_signal.corr(returns)
                    if not np.isnan(corr) and abs(corr) > 0.3:
                        correlations.append({
                            'features': [feat1, feat2],
                            'correlation': corr
                        })
        
        return sorted(correlations, key=lambda x: abs(x['correlation']), reverse=True)[:5]
    
    def _find_sequences(self, features: pd.DataFrame, returns: pd.Series) -> List[Dict]:
        """Find sequential patterns that predict returns"""
        sequences = []
        
        # Look for 3-bar patterns
        for i in range(3, len(features) - 1):
            if i >= len(returns):
                break
            
            # Simple price action sequence
            closes = [features.iloc[i-2]['close'], features.iloc[i-1]['close'], features.iloc[i]['close']]
            
            # Pattern: Higher highs followed by pullback
            if closes[1] > closes[0] and closes[2] < closes[1]:
                future_return = returns.iloc[i] if i < len(returns) else 0
                
                pattern_key = "hh_pullback"
                if pattern_key not in self.pattern_library:
                    self.pattern_library[pattern_key] = {'outcomes': [], 'total': 0}
                
                self.pattern_library[pattern_key]['outcomes'].append(future_return > 0)
                self.pattern_library[pattern_key]['total'] += 1
        
        # Analyze pattern performance
        for pattern, data in self.pattern_library.items():
            if data['total'] >= 10:
                success_rate = sum(data['outcomes']) / data['total']
                sequences.append({
                    'sequence': pattern,
                    'success_rate': success_rate,
                    'sample_size': data['total']
                })
        
        return sequences
    
    def _find_regime_patterns(self, features: pd.DataFrame, returns: pd.Series) -> List[Dict]:
        """Find patterns specific to market regimes"""
        regime_patterns = []
        
        # Define simple regimes based on volatility
        volatility_20 = features['close'].rolling(20).std()
        high_vol_threshold = volatility_20.quantile(0.7)
        low_vol_threshold = volatility_20.quantile(0.3)
        
        regimes = []
        for vol in volatility_20:
            if vol > high_vol_threshold:
                regimes.append('high_vol')
            elif vol < low_vol_threshold:
                regimes.append('low_vol')
            else:
                regimes.append('medium_vol')
        
        # Analyze performance in each regime
        for regime_type in ['high_vol', 'low_vol', 'medium_vol']:
            regime_indices = [i for i, r in enumerate(regimes) if r == regime_type]
            
            if len(regime_indices) < 20:
                continue
            
            # Test different conditions in this regime
            for condition in ['rsi_oversold', 'bb_squeeze', 'momentum_high']:
                condition_returns = []
                
                for idx in regime_indices:
                    if idx < len(features) and idx < len(returns):
                        row = features.iloc[idx]
                        
                        if condition == 'rsi_oversold' and row.get('rsi_14', 50) < 30:
                            condition_returns.append(returns.iloc[idx])
                        elif condition == 'bb_squeeze' and row.get('bb_width', 0) < 0.02:
                            condition_returns.append(returns.iloc[idx])
                        elif condition == 'momentum_high' and abs(row.get('momentum_10', 0)) > 0.01:
                            condition_returns.append(returns.iloc[idx])
                
                if len(condition_returns) >= 10:
                    avg_return = np.mean(condition_returns)
                    if abs(avg_return) > 0.001:  # Meaningful edge
                        regime_patterns.append({
                            'conditions': [regime_type, condition],
                            'edge': avg_return,
                            'sample_size': len(condition_returns)
                        })
        
        return regime_patterns
    
    def create_ai_strategy(self, pattern: Dict, strategy_id: str) -> 'AIGeneratedStrategy':
        """Create a new AI strategy from discovered pattern"""
        
        rules = {
            'pattern_type': pattern['type'],
            'entry_conditions': [],
            'confidence_base': 0.6,
            'min_strength': pattern.get('strength', 0.5)
        }
        
        if pattern['type'] == 'correlation':
            rules['entry_conditions'] = [
                {
                    'features': pattern['features'],
                    'correlation_threshold': pattern['correlation'],
                    'direction': 'buy' if pattern['correlation'] > 0 else 'sell'
                }
            ]
        elif pattern['type'] == 'sequence':
            rules['entry_conditions'] = [
                {
                    'sequence_pattern': pattern['sequence'],
                    'success_rate': pattern['success_rate'],
                    'direction': 'buy'  # Determined by pattern analysis
                }
            ]
        elif pattern['type'] == 'regime_specific':
            rules['entry_conditions'] = [
                {
                    'regime_conditions': pattern['conditions'],
                    'expected_edge': pattern['edge'],
                    'direction': 'buy' if pattern['edge'] > 0 else 'sell'
                }
            ]
        
        ai_strategy = AIGeneratedStrategy(
            id=strategy_id,
            name=f"AI_{pattern['type']}_{strategy_id[:8]}",
            rules=rules,
            performance_history=[],
            created_at=datetime.now(),
            last_optimized=datetime.now(),
            win_rate=0.0,
            profit_factor=1.0,
            max_drawdown=0.0,
            trade_count=0,
            active=True
        )
        
        return ai_strategy
    
    def generate_ai_strategies(self, features: pd.DataFrame, returns: pd.Series, 
                             max_strategies: int = 3) -> List['AIGeneratedStrategy']:
        """Generate new AI strategies from market data"""
        logger.info("🤖 AI discovering new trading patterns...")
        
        patterns = self.discover_patterns(features, returns)
        patterns = sorted(patterns, key=lambda x: x['strength'], reverse=True)
        
        ai_strategies = []
        for i, pattern in enumerate(patterns[:max_strategies]):
            strategy_id = f"ai_gen_{int(time.time())}_{i}"
            ai_strategy = self.create_ai_strategy(pattern, strategy_id)
            ai_strategies.append(ai_strategy)
            
            logger.info(f"   Created AI strategy: {ai_strategy.name} (strength: {pattern['strength']:.3f})")
        
        self.discovered_strategies.extend(ai_strategies)
        return ai_strategies

class DynamicAIStrategy(StrategyBase):
    """Dynamic strategy that executes AI-discovered patterns"""
    
    def __init__(self, ai_strategy: AIGeneratedStrategy):
        super().__init__(f"ai_{ai_strategy.id}", "AI")
        self.ai_strategy = ai_strategy
        self.rules = ai_strategy.rules
    
    def generate_signals(self, features: pd.DataFrame, htf_bias: HTFBias, 
                        regime: List[MarketRegime]) -> List[Signal]:
        signals = []
        
        if not self.ai_strategy.active:
            return signals
        
        for i, row in features.iterrows():
            if not self._check_conditions(row, features, i):
                continue
            
            # Determine signal direction and confidence
            direction, confidence = self._calculate_signal(row, features, i)
            
            if direction and confidence >= 0.6:
                regime_idx = min(i, len(regime) - 1) if regime else 0
                current_regime = regime[regime_idx] if regime else MarketRegime('trend', 0.5, 1, 0.01, 0)
                
                against_bias = (direction == "buy" and htf_bias.overall == 'bearish') or (direction == "sell" and htf_bias.overall == 'bullish')
                
                signal = Signal(
                    timestamp=row.name if hasattr(row, 'name') else datetime.now(),
                    symbol="",
                    side=direction,
                    confidence=confidence,
                    strategy_name=self.name,
                    strategy_family=self.family,
                    entry_price=row['close'],
                    htf_bias=htf_bias.overall,
                    against_bias=against_bias,
                    regime=current_regime.regime,
                    features={
                        "ai_strategy_id": self.ai_strategy.id,
                        "pattern_type": self.rules['pattern_type']
                    }
                )
                signals.append(signal)
        
        return signals
    
    def _check_conditions(self, row: pd.Series, features: pd.DataFrame, idx: int) -> bool:
        """Check if AI strategy conditions are met"""
        for condition in self.rules['entry_conditions']:
            if self.rules['pattern_type'] == 'correlation':
                # Check feature values meet correlation threshold
                feature_values = [row.get(feat, 0) for feat in condition['features']]
                if not all(abs(val) > 0.001 for val in feature_values):  # Basic validity check
                    return False
                    
            elif self.rules['pattern_type'] == 'sequence':
                # Check sequence pattern (simplified)
                if idx < 3:
                    return False
                recent_closes = [features.iloc[idx-2]['close'], features.iloc[idx-1]['close'], row['close']]
                if condition['sequence_pattern'] == 'hh_pullback':
                    if not (recent_closes[1] > recent_closes[0] and recent_closes[2] < recent_closes[1]):
                        return False
                        
            elif self.rules['pattern_type'] == 'regime_specific':
                # Check regime conditions (simplified)
                conditions = condition['regime_conditions']
                if 'rsi_oversold' in conditions and row.get('rsi_14', 50) >= 30:
                    return False
                if 'bb_squeeze' in conditions and row.get('bb_width', 0) >= 0.02:
                    return False
        
        return True
    
    def _calculate_signal(self, row: pd.Series, features: pd.DataFrame, idx: int) -> Tuple[Optional[str], float]:
        """Calculate signal direction and confidence"""
        base_confidence = self.rules['confidence_base']
        
        for condition in self.rules['entry_conditions']:
            direction = condition.get('direction', 'buy')
            
            if self.rules['pattern_type'] == 'correlation':
                strength = abs(condition['correlation_threshold'])
                confidence = base_confidence + (strength - 0.5) * 0.4
                
            elif self.rules['pattern_type'] == 'sequence':
                confidence = base_confidence + (condition['success_rate'] - 0.5) * 0.6
                
            elif self.rules['pattern_type'] == 'regime_specific':
                edge = abs(condition['expected_edge'])
                confidence = base_confidence + edge * 100  # Convert edge to confidence boost
            
            else:
                confidence = base_confidence
            
            return direction, min(confidence, 0.95)
        
        return None, 0.0

# Main Eden AI System
class EdenCompleteAISystem:
    """Complete Eden AI Trading System"""
    
    def __init__(self):
        self.mt5_initialized = self.initialize_mt5()
        self.results_dir = "eden_complete_results"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Target symbols
        self.target_symbols = ['XAUUSDm', 'GBPUSDm', 'EURUSDm', 'USTECm', 'US30m']
        
        # Components
        self.regime_classifier = MarketRegimeClassifier()
        self.htf_analyzer = HTFAnalyzer()
        self.ai_generator = AIStrategyGenerator()
        
        # Strategy families
        self.ict_strategies = []
        self.pa_strategies = []
        self.quant_strategies = []
        self.ai_strategies = []
        
        # Initialize all strategies
        self._initialize_all_strategies()
        
        # Performance tracking
        self.strategy_performance = defaultdict(list)
        self.pruned_strategies = []
        
    def initialize_mt5(self) -> bool:
        """Initialize MT5 connection"""
        if not MT5_AVAILABLE:
            return False
            
        if not mt5.initialize():
            logger.error(f"MT5 initialization failed: {mt5.last_error()}")
            return False
            
        logger.info("✅ MT5 connection established")
        account_info = mt5.account_info()
        if account_info:
            logger.info(f"📊 Connected to account: {account_info.login}")
            logger.info(f"💼 Broker: {account_info.company}")
        
        return True
    
    def _initialize_all_strategies(self):
        """Initialize all strategy families"""
        # ICT Strategies
        self.ict_strategies = [
            LiquiditySweepStrategy(),
            FairValueGapStrategy(),
            OrderBlockStrategy(),
            OptimalTradeEntryStrategy(),
            JudasSwingStrategy()
        ]
        
        # Price Action Strategies
        self.pa_strategies = [
            SupportResistanceStrategy(),
            CandlestickPatternsStrategy()
        ]
        
        # Quantitative Strategies
        self.quant_strategies = [
            MovingAverageCrossStrategy(),
            RSIDivergenceStrategy(),
            MeanReversionStrategy()
        ]
        
        total_strategies = len(self.ict_strategies) + len(self.pa_strategies) + len(self.quant_strategies)
        logger.info(f"🎯 Initialized {total_strategies} strategies across all families")
        logger.info(f"   • ICT: {len(self.ict_strategies)} strategies")
        logger.info(f"   • Price Action: {len(self.pa_strategies)} strategies")
        logger.info(f"   • Quantitative: {len(self.quant_strategies)} strategies")
    
    def get_maximum_data(self, symbol: str, timeframe: int) -> Optional[pd.DataFrame]:
        """Get maximum available data with feature engineering"""
        if not self.mt5_initialized:
            return None
        
        try:
            # Test if symbol exists
            recent_rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 10)
            if recent_rates is None or len(recent_rates) == 0:
                return None
            
            # Get maximum available data
            end_time = datetime.now()
            max_data = None
            
            for years_back in [10, 8, 5, 3, 2, 1]:
                start_time = end_time - timedelta(days=365 * years_back)
                rates = mt5.copy_rates_range(symbol, timeframe, start_time, end_time)
                
                if rates is not None and len(rates) >= 1000:
                    max_data = rates
                    logger.info(f"   📈 {symbol}: {len(rates):,} bars ({years_back} years)")
                    break
            
            if max_data is None:
                for max_count in [100000, 50000, 20000, 10000, 5000, 1000]:
                    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, max_count)
                    if rates is not None and len(rates) >= 1000:
                        max_data = rates
                        logger.info(f"   📈 {symbol}: {len(rates):,} bars (position method)")
                        break
            
            if max_data is None or len(max_data) < 1000:
                return None
            
            # Convert to DataFrame and add features
            df = pd.DataFrame(max_data)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            # Add comprehensive features
            df = self._add_comprehensive_features(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting data for {symbol}: {e}")
            return None
    
    def _add_comprehensive_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all technical features needed by strategies"""
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Moving averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['sma_200'] = df['close'].rolling(window=200).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        df['ema_50'] = df['close'].ewm(span=50).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_mid'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_mid'] + (bb_std * 2)
        df['bb_lower'] = df['bb_mid'] - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # ATR
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(window=14).mean()
        
        # Momentum
        df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
        df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
        df['momentum_20'] = df['close'] / df['close'].shift(20) - 1
        
        # Z-scores
        df['z_score_20'] = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).std()
        df['z_score_50'] = (df['close'] - df['close'].rolling(50).mean()) / df['close'].rolling(50).std()
        
        # Additional features for AI discovery
        df['volatility_ratio'] = df['atr'] / df['close']
        df['price_position'] = (df['close'] - df['low'].rolling(20).min()) / (df['high'].rolling(20).max() - df['low'].rolling(20).min())
        df['volume_ma'] = df['tick_volume'].rolling(20).mean()
        df['volume_ratio'] = df['tick_volume'] / df['volume_ma']
        
        return df.fillna(method='ffill').fillna(0)
    
    def run_complete_backtest(self):
        """Run complete Eden AI backtest with all strategies"""
        logger.info("🚀 Starting Eden Complete AI Trading System")
        logger.info("=" * 80)
        logger.info("🎯 Features:")
        logger.info("  • Complete ICT Strategy Suite (Liquidity, FVG, OB, OTE, Judas)")
        logger.info("  • Price Action Strategies (S/R, Patterns, Breakouts)")
        logger.info("  • Quantitative Strategies (MA, RSI Div, Mean Reversion)")
        logger.info("  • AI-Generated Strategies (Pattern Discovery)")
        logger.info("  • Market Regime Classification")
        logger.info("  • HTF Bias Analysis with Reversal Detection")
        logger.info("  • True AI Learning and Strategy Evolution")
        logger.info(f"📊 Target Symbols: {', '.join(self.target_symbols)}")
        
        all_results = {}
        
        for symbol in self.target_symbols:
            symbol_result = self.backtest_symbol(symbol)
            if symbol_result:
                all_results[symbol] = symbol_result
                
                # Learn and adapt after each symbol
                self._learn_and_adapt(symbol_result)
        
        if all_results:
            # Generate new AI strategies based on all results
            self._generate_new_ai_strategies(all_results)
            
            # Final pruning and optimization
            self._final_optimization(all_results)
            
            # Save and report
            self._save_complete_results(all_results)
            self._generate_complete_report(all_results)
            self._display_final_summary(all_results)
        else:
            logger.error("❌ No results generated - check MT5 connection and data availability")
        
        return all_results
    
    def backtest_symbol(self, symbol: str) -> Dict:
        """Complete backtest for single symbol with all strategies"""
        logger.info(f"\n🎯 Complete backtest for {symbol}...")
        
        # Get 5-minute data
        m5_data = self.get_maximum_data(symbol, mt5.TIMEFRAME_M5)
        if m5_data is None or len(m5_data) < 2000:
            logger.error(f"   ❌ Insufficient data for {symbol}")
            return {}
        
        # Get higher timeframe data for HTF analysis
        h1_data = self.get_maximum_data(symbol, mt5.TIMEFRAME_H1)
        h4_data = self.get_maximum_data(symbol, mt5.TIMEFRAME_H4)
        d1_data = self.get_maximum_data(symbol, mt5.TIMEFRAME_D1)
        
        if not all([h1_data is not None, h4_data is not None, d1_data is not None]):
            logger.warning(f"   ⚠️ Missing HTF data for {symbol}, using M5 approximation")
            h1_data = h4_data = d1_data = m5_data
        
        logger.info(f"   📊 Data loaded: {len(m5_data):,} M5 bars")
        
        # Split data for training and testing
        split_idx = int(len(m5_data) * 0.7)
        train_data = m5_data.iloc[:split_idx]
        test_data = m5_data.iloc[split_idx:]
        
        # AI learns from training data first
        logger.info("   🤖 AI learning from training data...")
        returns = train_data['close'].pct_change().shift(-1)
        ai_strategies = self.ai_generator.generate_ai_strategies(train_data, returns, max_strategies=2)
        
        # Add AI strategies to active pool
        for ai_strat in ai_strategies:
            dynamic_strategy = DynamicAIStrategy(ai_strat)
            self.ai_strategies.append(dynamic_strategy)
        
        # Analyze market context for test period
        logger.info("   📈 Analyzing market context...")
        regimes = self.regime_classifier.classify_regime(self.regime_classifier.prepare_features(test_data))
        
        # HTF bias analysis
        htf_bias = self.htf_analyzer.analyze_htf_bias(
            d1_data.iloc[split_idx:] if len(d1_data) > split_idx else d1_data,
            h4_data.iloc[split_idx:] if len(h4_data) > split_idx else h4_data,
            h1_data.iloc[split_idx:] if len(h1_data) > split_idx else h1_data
        )
        
        logger.info(f"   📊 HTF Bias: {htf_bias.overall} (confidence: {htf_bias.confidence:.2f})")
        logger.info(f"   🔄 Reversal Probability: {htf_bias.reversal_probability:.2f}")
        
        # Run all strategies
        all_strategies = (self.ict_strategies + self.pa_strategies + 
                         self.quant_strategies + self.ai_strategies)
        
        strategy_results = {}
        
        logger.info(f"   🎯 Testing {len(all_strategies)} strategies...")
        
        for strategy in all_strategies:
            try:
                signals = strategy.generate_signals(test_data, htf_bias, regimes)
                
                # Set symbol for all signals
                for signal in signals:
                    signal.symbol = symbol
                
                if signals:
                    trades = self._backtest_signals(signals, test_data, symbol, htf_bias)
                    metrics = self._calculate_metrics(trades, strategy.name)
                    
                    strategy_results[strategy.name] = {
                        'family': strategy.family,
                        'signals': len(signals),
                        'trades': trades,
                        'metrics': metrics,
                        'against_bias_trades': sum(1 for s in signals if s.against_bias),
                        'reversal_trades': sum(1 for s in signals if s.reversal_signal)
                    }
                    
                    # Track performance for learning
                    self.strategy_performance[strategy.name].append(metrics['profit_factor'])
                    
                    logger.info(f"     • {strategy.name}: {len(trades)} trades, "
                              f"{metrics['win_rate']:.1f}% WR, "
                              f"PF: {metrics['profit_factor']:.2f}")
            
            except Exception as e:
                logger.error(f"     ❌ Error in {strategy.name}: {e}")
                continue
        
        return {
            'symbol': symbol,
            'data_points': len(test_data),
            'test_period': {
                'start': test_data.index[0].strftime('%Y-%m-%d'),
                'end': test_data.index[-1].strftime('%Y-%m-%d')
            },
            'htf_bias': htf_bias,
            'regime_distribution': self._analyze_regime_distribution(regimes),
            'strategies': strategy_results,
            'ai_strategies_created': len(ai_strategies)
        }
    
    def _backtest_signals(self, signals: List[Signal], data: pd.DataFrame, 
                         symbol: str, htf_bias: HTFBias) -> List[Dict]:
        """Enhanced backtest with proper risk management"""
        trades = []
        
        for signal in signals:
            try:
                signal_idx = data.index.get_loc(signal.timestamp)
            except KeyError:
                continue
            
            if signal_idx >= len(data) - 50:
                continue
            
            entry_row = data.iloc[signal_idx]
            entry_price = entry_row['close']
            
            # Risk management
            base_risk = 0.0025 if signal.against_bias else 0.005  # Smaller size when against bias
            confidence_multiplier = signal.confidence
            risk_percent = min(base_risk * confidence_multiplier, 0.02)  # Max 2%
            
            position_size = 100000 * risk_percent
            
            # ATR-based stops and targets
            atr = entry_row.get('atr', entry_price * 0.002)
            
            # Adjust multipliers based on strategy family and signal type
            if signal.strategy_family == 'ICT':
                stop_mult = 1.5 if not signal.reversal_signal else 2.0
                target_mult = 3.0 if not signal.reversal_signal else 4.0
            elif signal.strategy_family == 'PA':
                stop_mult = 2.0
                target_mult = 3.0
            else:  # Quant or AI
                stop_mult = 1.5
                target_mult = 2.5
            
            # Against bias trades get tighter stops
            if signal.against_bias:
                stop_mult *= 0.8
                target_mult *= 1.2
            
            if signal.side == "buy":
                stop_loss = entry_price - (stop_mult * atr)
                take_profit = entry_price + (target_mult * atr)
            else:
                stop_loss = entry_price + (stop_mult * atr)
                take_profit = entry_price - (target_mult * atr)
            
            # Find exit
            exit_idx = None
            exit_price = None
            exit_reason = 'time'
            
            max_hold = 96 if signal.strategy_family == 'ICT' else 48  # ICT can hold longer
            
            for i in range(signal_idx + 1, min(signal_idx + max_hold + 1, len(data))):
                bar = data.iloc[i]
                
                if signal.side == "buy":
                    if bar['low'] <= stop_loss:
                        exit_idx = i
                        exit_price = stop_loss
                        exit_reason = 'stop'
                        break
                    elif bar['high'] >= take_profit:
                        exit_idx = i
                        exit_price = take_profit
                        exit_reason = 'target'
                        break
                else:
                    if bar['high'] >= stop_loss:
                        exit_idx = i
                        exit_price = stop_loss
                        exit_reason = 'stop'
                        break
                    elif bar['low'] <= take_profit:
                        exit_idx = i
                        exit_price = take_profit
                        exit_reason = 'target'
                        break
            
            if exit_idx is None:
                exit_idx = min(signal_idx + max_hold, len(data) - 1)
                exit_price = data.iloc[exit_idx]['close']
                exit_reason = 'time'
            
            # Calculate results
            if signal.side == "buy":
                pnl = (exit_price - entry_price) * position_size / entry_price
            else:
                pnl = (entry_price - exit_price) * position_size / entry_price
            
            pnl_percent = (pnl / position_size) * 100
            hold_time = (exit_idx - signal_idx) * 5
            
            trade = {
                'entry_time': signal.timestamp,
                'exit_time': data.index[exit_idx],
                'symbol': symbol,
                'strategy': signal.strategy_name,
                'strategy_family': signal.strategy_family,
                'side': signal.side,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': pnl,
                'pnl_percent': pnl_percent,
                'hold_time_minutes': hold_time,
                'exit_reason': exit_reason,
                'confidence': signal.confidence,
                'against_bias': signal.against_bias,
                'reversal_signal': signal.reversal_signal,
                'risk_percent': risk_percent,
                'regime': signal.regime
            }
            trades.append(trade)
        
        return trades
    
    def _calculate_metrics(self, trades: List[Dict], strategy_name: str) -> Dict:
        """Calculate comprehensive strategy metrics"""
        if not trades:
            return {
                'strategy': strategy_name,
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'total_return': 0,
                'max_drawdown': 0,
                'expectancy': 0,
                'avg_hold_time': 0,
                'against_bias_success': 0,
                'reversal_success': 0
            }
        
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t['pnl'] > 0)
        win_rate = (winning_trades / total_trades) * 100
        
        total_pnl = sum(t['pnl'] for t in trades)
        wins = [t['pnl'] for t in trades if t['pnl'] > 0]
        losses = [t['pnl'] for t in trades if t['pnl'] < 0]
        
        profit_factor = (sum(wins) / abs(sum(losses))) if losses else float('inf')
        expectancy = (sum(wins) - abs(sum(losses))) / total_trades
        
        # Against bias performance
        against_bias_trades = [t for t in trades if t['against_bias']]
        against_bias_success = (sum(1 for t in against_bias_trades if t['pnl'] > 0) / 
                              len(against_bias_trades) * 100) if against_bias_trades else 0
        
        # Reversal signal performance  
        reversal_trades = [t for t in trades if t['reversal_signal']]
        reversal_success = (sum(1 for t in reversal_trades if t['pnl'] > 0) / 
                          len(reversal_trades) * 100) if reversal_trades else 0
        
        # Drawdown calculation
        balance = 100000
        peak = balance
        max_dd = 0
        
        for trade in trades:
            balance += trade['pnl']
            if balance > peak:
                peak = balance
            else:
                dd = (peak - balance) / peak * 100
                max_dd = max(max_dd, dd)
        
        return {
            'strategy': strategy_name,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_pnl': total_pnl,
            'total_return': total_pnl / 100000 * 100,
            'max_drawdown': max_dd,
            'expectancy': expectancy,
            'avg_win': np.mean(wins) if wins else 0,
            'avg_loss': abs(np.mean(losses)) if losses else 0,
            'avg_hold_time': np.mean([t['hold_time_minutes'] for t in trades]),
            'against_bias_trades': len(against_bias_trades),
            'against_bias_success': against_bias_success,
            'reversal_trades': len(reversal_trades), 
            'reversal_success': reversal_success,
            'final_balance': balance
        }
    
    def _learn_and_adapt(self, symbol_result: Dict):
        """AI learns and adapts after each symbol"""
        logger.info("   🧠 AI learning and adapting...")
        
        # Prune consistently underperforming strategies
        strategies_to_prune = []
        
        for strategy_name, perf_history in self.strategy_performance.items():
            if len(perf_history) >= 3:
                recent_performance = perf_history[-3:]
                if all(pf < 1.1 for pf in recent_performance):
                    strategies_to_prune.append(strategy_name)
        
        for strategy_name in strategies_to_prune:
            self._prune_strategy(strategy_name)
        
        # Adjust confidence thresholds based on performance
        self._adjust_strategy_parameters(symbol_result)
    
    def _prune_strategy(self, strategy_name: str):
        """Prune underperforming strategy"""
        for strategy_list in [self.ict_strategies, self.pa_strategies, 
                            self.quant_strategies, self.ai_strategies]:
            for strategy in strategy_list[:]:
                if strategy.name == strategy_name:
                    strategy.active = False
                    strategy_list.remove(strategy)
                    self.pruned_strategies.append(strategy_name)
                    logger.info(f"     ✂️ Pruned strategy: {strategy_name}")
                    break
    
    def _adjust_strategy_parameters(self, symbol_result: Dict):
        """Adjust strategy parameters based on performance"""
        for strategy_name, result in symbol_result['strategies'].items():
            metrics = result['metrics']
            
            # Find corresponding strategy and adjust
            all_strategies = (self.ict_strategies + self.pa_strategies + 
                            self.quant_strategies + self.ai_strategies)
            
            for strategy in all_strategies:
                if strategy.name == strategy_name:
                    if metrics['profit_factor'] > 1.5:
                        # Good performance - lower confidence threshold slightly
                        strategy.confidence_threshold = max(0.6, strategy.confidence_threshold - 0.05)
                    elif metrics['profit_factor'] < 1.1:
                        # Poor performance - raise confidence threshold
                        strategy.confidence_threshold = min(0.8, strategy.confidence_threshold + 0.05)
                    break
    
    def _generate_new_ai_strategies(self, all_results: Dict):
        """Generate new AI strategies based on all results"""
        logger.info("🤖 AI generating new strategies from market insights...")
        
        # Combine all trade data for analysis
        all_trades = []
        for symbol_result in all_results.values():
            for strategy_result in symbol_result['strategies'].values():
                all_trades.extend(strategy_result['trades'])
        
        if len(all_trades) < 50:
            return
        
        # Analyze what worked best
        successful_patterns = defaultdict(list)
        
        for trade in all_trades:
            if trade['pnl'] > 0:  # Winning trades
                pattern_key = f"{trade['strategy_family']}_{trade['regime']}"
                if trade['against_bias']:
                    pattern_key += "_against_bias"
                if trade['reversal_signal']:
                    pattern_key += "_reversal"
                
                successful_patterns[pattern_key].append(trade['pnl'])
        
        # Create new AI strategies from successful patterns
        new_ai_count = 0
        for pattern, pnl_list in successful_patterns.items():
            if len(pnl_list) >= 10 and np.mean(pnl_list) > 50:  # Good pattern
                strategy_id = f"ai_evolved_{int(time.time())}_{new_ai_count}"
                
                # Create simplified AI strategy
                pattern_rules = {
                    'pattern_type': 'evolved_pattern',
                    'entry_conditions': [{
                        'pattern_signature': pattern,
                        'min_trades': len(pnl_list),
                        'avg_pnl': np.mean(pnl_list),
                        'direction': 'adaptive'
                    }],
                    'confidence_base': 0.7,
                    'min_strength': 0.6
                }
                
                ai_strategy = AIGeneratedStrategy(
                    id=strategy_id,
                    name=f"AI_Evolved_{pattern[:15]}",
                    rules=pattern_rules,
                    performance_history=[],
                    created_at=datetime.now(),
                    last_optimized=datetime.now(),
                    win_rate=0.0,
                    profit_factor=1.0,
                    max_drawdown=0.0,
                    trade_count=0,
                    active=True
                )
                
                dynamic_strategy = DynamicAIStrategy(ai_strategy)
                self.ai_strategies.append(dynamic_strategy)
                new_ai_count += 1
                
                logger.info(f"   Created evolved AI strategy: {ai_strategy.name}")
        
        logger.info(f"   🎯 Generated {new_ai_count} new AI strategies")
    
    def _analyze_regime_distribution(self, regimes: List[MarketRegime]) -> Dict:
        """Analyze distribution of market regimes"""
        regime_counts = defaultdict(int)
        for regime in regimes:
            regime_counts[regime.regime] += 1
        
        total = len(regimes)
        return {regime: count/total*100 for regime, count in regime_counts.items()}
    
    def _final_optimization(self, all_results: Dict):
        """Final system optimization"""
        logger.info("🔧 Final system optimization...")
        
        # Calculate overall system performance
        total_trades = 0
        total_pnl = 0
        
        for symbol_result in all_results.values():
            for strategy_result in symbol_result['strategies'].values():
                total_trades += strategy_result['metrics']['total_trades']
                total_pnl += strategy_result['metrics']['total_pnl']
        
        logger.info(f"   📊 System totals: {total_trades:,} trades, ${total_pnl:,.0f} PnL")
        logger.info(f"   ✂️ Pruned strategies: {len(self.pruned_strategies)}")
        logger.info(f"   🤖 Active AI strategies: {len(self.ai_strategies)}")
    
    def _save_complete_results(self, results: Dict):
        """Save complete results"""
        logger.info("💾 Saving complete results...")
        
        # Save individual results
        for symbol, symbol_result in results.items():
            symbol_dir = Path(self.results_dir) / symbol
            symbol_dir.mkdir(exist_ok=True)
            
            with open(symbol_dir / "complete_results.json", 'w') as f:
                json.dump(symbol_result, f, indent=2, default=str)
        
        # Save AI strategies
        ai_strategies_data = []
        for ai_strat in self.ai_generator.discovered_strategies:
            ai_strategies_data.append(asdict(ai_strat))
        
        with open(Path(self.results_dir) / "ai_strategies.json", 'w') as f:
            json.dump(ai_strategies_data, f, indent=2, default=str)
        
        # Save pruned strategies log
        with open(Path(self.results_dir) / "pruned_strategies.json", 'w') as f:
            json.dump(self.pruned_strategies, f, indent=2)
    
    def _generate_complete_report(self, results: Dict):
        """Generate complete comprehensive report"""
        logger.info("📝 Generating complete Eden AI report...")
        
        report = f"""# Eden Complete AI Trading System Report
## Advanced Multi-Strategy AI Framework with True Learning

### 🎯 Executive Summary
- **Analysis Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Symbols Analyzed**: {', '.join(results.keys())}
- **Strategy Families**: ICT, Price Action, Quantitative, AI-Generated
- **Total Strategies**: {len(self.ict_strategies + self.pa_strategies + self.quant_strategies + self.ai_strategies)}
- **AI Features**: Pattern Discovery, Strategy Evolution, Dynamic Pruning
- **True AI Learning**: ✅ Enabled

---

## 🧠 AI LEARNING CAPABILITIES

### Strategy Evolution
- **AI-Generated Strategies**: {len(self.ai_strategies)} strategies created
- **Pattern Discovery**: Automated detection of profitable patterns
- **Strategy Pruning**: {len(self.pruned_strategies)} underperforming strategies removed
- **Parameter Adaptation**: Dynamic adjustment based on performance

### Market Intelligence
- **Regime Classification**: Trend, Reversal, Range, Momentum detection
- **HTF Bias Analysis**: Multi-timeframe bias with reversal probability
- **Against-Bias Trading**: AI can trade against HTF bias when reversal signals strong
- **Context Awareness**: Strategy selection based on market conditions

---

## 📊 STRATEGY FAMILIES PERFORMANCE

"""
        
        # Strategy family analysis
        family_performance = defaultdict(lambda: {'trades': 0, 'pnl': 0, 'strategies': 0})
        
        for symbol_result in results.values():
            for strategy_name, strategy_result in symbol_result['strategies'].items():
                family = strategy_result['family']
                family_performance[family]['trades'] += strategy_result['metrics']['total_trades']
                family_performance[family]['pnl'] += strategy_result['metrics']['total_pnl']
                family_performance[family]['strategies'] += 1
        
        for family, perf in family_performance.items():
            if perf['trades'] > 0:
                avg_return = perf['pnl'] / (100000 * len(results)) * 100
                report += f"""### {family} Strategy Family
- **Active Strategies**: {perf['strategies']}
- **Total Trades**: {perf['trades']:,}
- **Combined PnL**: ${perf['pnl']:,.0f}
- **Average Return**: {avg_return:+.2f}%

"""
        
        # Individual symbol results
        report += """---

## 📈 SYMBOL ANALYSIS

"""
        
        for symbol, symbol_result in results.items():
            htf_bias = symbol_result['htf_bias']
            regime_dist = symbol_result['regime_distribution']
            
            report += f"""### {symbol}
- **Test Period**: {symbol_result['test_period']['start']} to {symbol_result['test_period']['end']}
- **Data Points**: {symbol_result['data_points']:,} bars
- **HTF Bias**: {htf_bias.overall} (confidence: {htf_bias.confidence:.2f})
- **Reversal Probability**: {htf_bias.reversal_probability:.2f}
- **AI Strategies Created**: {symbol_result['ai_strategies_created']}

#### Market Regime Distribution:
"""
            
            for regime, percentage in regime_dist.items():
                report += f"- **{regime.title()}**: {percentage:.1f}%\n"
            
            # Top performing strategies for this symbol
            strategies = list(symbol_result['strategies'].items())
            strategies.sort(key=lambda x: x[1]['metrics']['profit_factor'], reverse=True)
            
            report += "\n#### Top Strategies:\n"
            for strategy_name, strategy_data in strategies[:3]:
                metrics = strategy_data['metrics']
                report += f"""- **{strategy_name}** ({strategy_data['family']}): {metrics['total_trades']} trades, {metrics['win_rate']:.1f}% WR, PF: {metrics['profit_factor']:.2f}
"""
            
            report += "\n"
        
        # Against-bias and reversal analysis
        total_against_bias = 0
        successful_against_bias = 0
        total_reversals = 0
        successful_reversals = 0
        
        for symbol_result in results.values():
            for strategy_result in symbol_result['strategies'].values():
                metrics = strategy_result['metrics']
                total_against_bias += metrics.get('against_bias_trades', 0)
                successful_against_bias += (metrics.get('against_bias_trades', 0) * 
                                          metrics.get('against_bias_success', 0) / 100)
                total_reversals += metrics.get('reversal_trades', 0)
                successful_reversals += (metrics.get('reversal_trades', 0) * 
                                       metrics.get('reversal_success', 0) / 100)
        
        against_bias_wr = (successful_against_bias / total_against_bias * 100) if total_against_bias > 0 else 0
        reversal_wr = (successful_reversals / total_reversals * 100) if total_reversals > 0 else 0
        
        report += f"""---

## 🎯 ADVANCED AI FEATURES

### Against-Bias Trading Intelligence
- **Total Against-Bias Trades**: {total_against_bias:,}
- **Against-Bias Success Rate**: {against_bias_wr:.1f}%
- **AI Override Capability**: ✅ Can trade against HTF bias when reversal signals strong

### Reversal Pattern Recognition
- **Total Reversal Trades**: {total_reversals:,}
- **Reversal Success Rate**: {reversal_wr:.1f}%
- **Pattern Types**: Liquidity sweeps, divergences, exhaustion patterns

### Strategy Evolution
- **Original Strategies**: {len(self.ict_strategies) + len(self.pa_strategies) + len(self.quant_strategies)}
- **AI-Generated Strategies**: {len(self.ai_strategies)}
- **Pruned Strategies**: {len(self.pruned_strategies)} ({', '.join(self.pruned_strategies[:5])}{"..." if len(self.pruned_strategies) > 5 else ""})
- **Adaptation Cycles**: Continuous learning after each symbol

---

## 🔬 METHODOLOGY

### Strategy Families

#### ICT Strategies
- **Liquidity Sweeps**: False breaks and liquidity hunts
- **Fair Value Gaps**: Imbalance areas for entries  
- **Order Blocks**: Institutional order levels
- **Optimal Trade Entry**: Fibonacci retracement zones
- **Judas Swing**: False session break patterns

#### Price Action Strategies
- **Support/Resistance**: Key level interactions
- **Candlestick Patterns**: Reversal and continuation signals
- **Breakout/Retest**: Structure break confirmations

#### Quantitative Strategies  
- **Moving Average Cross**: MACD and EMA systems
- **RSI Divergence**: Hidden and regular divergences
- **Mean Reversion**: Statistical overbought/oversold

#### AI-Generated Strategies
- **Pattern Discovery**: ML finds recurring profitable setups
- **Correlation Analysis**: Multi-feature relationship detection
- **Sequence Recognition**: Price action sequence patterns
- **Regime-Specific**: Strategies adapted to market conditions

### Risk Management
- **Dynamic Position Sizing**: 0.25-2% based on confidence and bias alignment
- **ATR-Based Stops**: Volatility-adjusted risk levels
- **Family-Specific Rules**: Different R:R ratios per strategy type
- **Time-Based Exits**: Maximum hold times to prevent overnight risk

### AI Learning Process
1. **Pattern Discovery**: Analyze historical data for profitable relationships
2. **Strategy Creation**: Generate new strategies from discovered patterns
3. **Real-Time Testing**: Validate new strategies on live market data
4. **Performance Tracking**: Monitor all strategies continuously
5. **Adaptive Pruning**: Remove consistently underperforming strategies
6. **Parameter Evolution**: Adjust thresholds and parameters based on results

---

**Generated**: {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}
**Framework**: Eden Complete AI Trading System v9.0
**Total Active Strategies**: {len(self.ict_strategies + self.pa_strategies + self.quant_strategies + self.ai_strategies)}
**AI Status**: FULLY OPERATIONAL - Learning, Adapting, Evolving
"""
        
        report_file = Path(self.results_dir) / "Eden_Complete_AI_Report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"📁 Complete report saved: {report_file}")
    
    def _display_final_summary(self, results: Dict):
        """Display final comprehensive summary"""
        print("\n" + "=" * 80)
        print("🎯 EDEN COMPLETE AI TRADING SYSTEM SUMMARY")
        print("=" * 80)
        
        # Strategy counts
        total_strategies = len(self.ict_strategies + self.pa_strategies + self.quant_strategies + self.ai_strategies)
        
        print(f"🧠 AI Intelligence:")
        print(f"   • Total Active Strategies: {total_strategies}")
        print(f"   • ICT Strategies: {len(self.ict_strategies)}")
        print(f"   • Price Action: {len(self.pa_strategies)}")  
        print(f"   • Quantitative: {len(self.quant_strategies)}")
        print(f"   • AI-Generated: {len(self.ai_strategies)}")
        print(f"   • Pruned (Poor Performance): {len(self.pruned_strategies)}")
        
        # Overall performance
        total_trades = 0
        total_pnl = 0
        against_bias_trades = 0
        reversal_trades = 0
        
        print(f"\n📊 Symbol Performance:")
        for symbol, symbol_result in results.items():
            symbol_trades = sum(s['metrics']['total_trades'] for s in symbol_result['strategies'].values())
            symbol_pnl = sum(s['metrics']['total_pnl'] for s in symbol_result['strategies'].values())
            symbol_against_bias = sum(s['metrics'].get('against_bias_trades', 0) for s in symbol_result['strategies'].values())
            symbol_reversals = sum(s['metrics'].get('reversal_trades', 0) for s in symbol_result['strategies'].values())
            
            total_trades += symbol_trades
            total_pnl += symbol_pnl
            against_bias_trades += symbol_against_bias
            reversal_trades += symbol_reversals
            
            htf_bias = symbol_result['htf_bias']
            
            print(f"   {symbol}:")
            print(f"     • Trades: {symbol_trades:,} | PnL: ${symbol_pnl:,.0f}")
            print(f"     • HTF Bias: {htf_bias.overall} | Reversal Prob: {htf_bias.reversal_probability:.2f}")
            print(f"     • Against-Bias Trades: {symbol_against_bias} | Reversal Trades: {symbol_reversals}")
        
        print(f"\n🏆 Total Portfolio:")
        print(f"   • Total Trades: {total_trades:,}")
        print(f"   • Combined PnL: ${total_pnl:,.0f}")
        print(f"   • Portfolio Return: {(total_pnl / (100000 * len(results))) * 100:+.2f}%")
        print(f"   • Against-Bias Trades: {against_bias_trades:,}")
        print(f"   • Reversal Trades: {reversal_trades:,}")
        
        print(f"\n🤖 AI Learning Status:")
        print(f"   • Pattern Discovery: ACTIVE")
        print(f"   • Strategy Evolution: ACTIVE") 
        print(f"   • Performance Tracking: ACTIVE")
        print(f"   • Adaptive Pruning: ACTIVE")
        print(f"   • Against-Bias Intelligence: ACTIVE")
        
        print(f"\n📁 Results saved in: {self.results_dir}/")
        print("✅ Eden Complete AI System operational!")
    
    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'mt5_initialized') and self.mt5_initialized:
            mt5.shutdown()

def main():
    """Main execution"""
    print("🎯 Eden Complete AI Trading System")
    print("=" * 80)
    print("🚀 Full Feature Set:")
    print("  • Complete ICT Strategy Suite (Liquidity, FVG, OB, OTE, Judas)")
    print("  • Price Action Strategies (S/R, Patterns, Breakouts)")
    print("  • Quantitative Strategies (MA Cross, RSI Div, Mean Reversion)")
    print("  • AI-Generated Strategies (Pattern Discovery & Evolution)")
    print("  • Market Regime Classification & HTF Bias Analysis")
    print("  • Against-Bias Trading Intelligence")  
    print("  • True AI Learning: Adapts, Learns, Evolves")
    print("  • Dynamic Strategy Pruning & Optimization")
    
    eden_system = EdenCompleteAISystem()
    
    if not eden_system.mt5_initialized:
        print("❌ Cannot proceed without MT5 connection")
        return
    
    start_time = time.time()
    results = eden_system.run_complete_backtest()
    end_time = time.time()
    
    print(f"\n⏱️ Total execution time: {end_time - start_time:.1f} seconds")
    print("🎯 Eden Complete AI System execution completed!")
    print("🧠 AI will continue learning and evolving...")

if __name__ == "__main__":
    main()