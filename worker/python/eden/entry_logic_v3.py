"""
Eden Entry Logic v3.0 (Mathematical + ML-Heavy)
Advanced quantitative trading system with multi-timeframe analysis and ML filtering
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Try to import ML libraries with fallbacks
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# Try to import technical analysis
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False

logger = logging.getLogger(__name__)

class TrendDirection(Enum):
    BULLISH = 1
    BEARISH = -1
    NEUTRAL = 0

class SignalStrength(Enum):
    VERY_STRONG = 5
    STRONG = 4
    MODERATE = 3
    WEAK = 2
    VERY_WEAK = 1

@dataclass
class HTFTrendScore:
    """Higher timeframe trend analysis"""
    h1_score: float = 0.0
    h4_score: float = 0.0
    d1_score: float = 0.0
    weighted_score: float = 0.0
    direction: TrendDirection = TrendDirection.NEUTRAL
    strength: SignalStrength = SignalStrength.WEAK

@dataclass
class ZoneStrength:
    """Order block / FVG zone strength calculation"""
    fvg_score: float = 0.0
    order_block_score: float = 0.0
    swing_score: float = 0.0
    total_score: float = 0.0
    is_strong: bool = False

@dataclass
class ConfluenceScore:
    """Multi-factor confluence scoring"""
    htf_alignment: float = 0.0
    pullback_to_zone: float = 0.0
    candlestick_confirmation: float = 0.0
    momentum_indicator: float = 0.0
    volume_spike: float = 0.0
    total_score: float = 0.0
    meets_threshold: bool = False

@dataclass
class MLPrediction:
    """Machine learning prediction results"""
    probability: float = 0.5
    confidence: float = 0.0
    features: Dict[str, float] = field(default_factory=dict)
    meets_threshold: bool = False

@dataclass
class EntrySignal:
    """Complete entry signal with all components"""
    timestamp: datetime
    symbol: str
    timeframe: str
    
    # Core components
    htf_trend: HTFTrendScore
    zone_strength: ZoneStrength
    confluence: ConfluenceScore
    ml_prediction: MLPrediction
    
    # Entry details
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profits: List[float] = field(default_factory=list)
    position_size: float = 0.0
    risk_reward_ratio: float = 0.0
    
    # Market context
    volatility_score: float = 0.0
    liquidity_score: float = 0.0
    session_score: float = 0.0
    
    # Final decision
    should_enter: bool = False
    signal_strength: SignalStrength = SignalStrength.WEAK

class EdenEntryLogicV3:
    """
    Advanced Eden Entry Logic v3.0
    Mathematical + ML-Heavy approach with quantitative decision making
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        
        # Weights for HTF trend scoring (must sum to 1)
        self.htf_weights = {
            'D1': 0.5,  # Daily trend most important
            'H4': 0.3,  # 4-hour trend secondary
            'H1': 0.2   # 1-hour trend for fine-tuning
        }
        
        # Zone strength weights
        self.zone_weights = {
            'fvg': 0.4,
            'order_block': 0.3,
            'swing': 0.3
        }
        
        # Confluence weights (must sum to 1)
        self.confluence_weights = {
            'htf_alignment': 0.30,
            'pullback_to_zone': 0.25,
            'candlestick_confirmation': 0.20,
            'momentum_indicator': 0.15,
            'volume_spike': 0.10
        }
        
        # ML model and scaler
        self.ml_model = None
        self.feature_scaler = None
        self.model_trained = False
        
        # Performance tracking
        self.trade_history = []
        self.model_performance = {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0}
        
        logger.info("Eden Entry Logic v3.0 initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration parameters"""
        return {
            # Trend thresholds
            'bullish_threshold': 0.6,
            'bearish_threshold': -0.6,
            
            # Zone strength threshold
            'zone_strength_threshold': 0.7,
            
            # Confluence threshold
            'confluence_threshold': 0.6,
            
            # ML thresholds
            'ml_base_threshold': 0.65,
            'ml_confidence_threshold': 0.7,
            'ml_dynamic_adjustment': True,
            
            # Risk management
            'base_risk_trend': 0.005,    # 0.5% for trend following
            'base_risk_counter': 0.0025, # 0.25% for counter-trend
            'max_risk_multiplier': 2.0,
            
            # ATR multipliers
            'atr_sl_multiplier': 1.5,
            'atr_tp_multiplier': [2.0, 3.5, 5.0],  # Multiple TP levels
            
            # Volatility filters
            'min_volatility_ratio': 1.2,
            'max_spread_pips': 3.0,
            
            # Session filters
            'london_session_boost': 0.1,
            'new_york_session_boost': 0.15,
            'overlap_session_boost': 0.2
        }
    
    def calculate_htf_trend_score(self, data: Dict[str, pd.DataFrame]) -> HTFTrendScore:
        """
        Step 1: Multi-Timeframe Trend Score Calculation
        
        For each HTF (H1/H4/D1):
        Ti = +1 if EMA50 > EMA200 (bullish)
             -1 if EMA50 < EMA200 (bearish)
              0 otherwise (neutral)
        
        Weighted HTF Trend Score = wD1*TD1 + wH4*TH4 + wH1*TH1
        """
        score = HTFTrendScore()
        
        try:
            # Calculate individual timeframe scores
            timeframe_scores = {}
            
            for tf in ['H1', 'H4', 'D1']:
                if tf in data and not data[tf].empty:
                    df = data[tf].copy()
                    
                    # Calculate EMAs
                    if TALIB_AVAILABLE:
                        ema50 = talib.EMA(df['close'].values, timeperiod=50)
                        ema200 = talib.EMA(df['close'].values, timeperiod=200)
                    else:
                        ema50 = df['close'].ewm(span=50).mean().values
                        ema200 = df['close'].ewm(span=200).mean().values
                    
                    # Get latest values
                    if len(ema50) > 200 and len(ema200) > 200:
                        latest_ema50 = ema50[-1]
                        latest_ema200 = ema200[-1]
                        
                        if latest_ema50 > latest_ema200:
                            timeframe_scores[tf] = 1.0  # Bullish
                        elif latest_ema50 < latest_ema200:
                            timeframe_scores[tf] = -1.0  # Bearish
                        else:
                            timeframe_scores[tf] = 0.0  # Neutral
                    else:
                        timeframe_scores[tf] = 0.0
                else:
                    timeframe_scores[tf] = 0.0
            
            # Store individual scores
            score.h1_score = timeframe_scores.get('H1', 0.0)
            score.h4_score = timeframe_scores.get('H4', 0.0)
            score.d1_score = timeframe_scores.get('D1', 0.0)
            
            # Calculate weighted score
            score.weighted_score = (
                self.htf_weights['D1'] * score.d1_score +
                self.htf_weights['H4'] * score.h4_score +
                self.htf_weights['H1'] * score.h1_score
            )
            
            # Determine direction and strength
            if score.weighted_score >= self.config['bullish_threshold']:
                score.direction = TrendDirection.BULLISH
                score.strength = self._calculate_signal_strength(abs(score.weighted_score))
            elif score.weighted_score <= self.config['bearish_threshold']:
                score.direction = TrendDirection.BEARISH
                score.strength = self._calculate_signal_strength(abs(score.weighted_score))
            else:
                score.direction = TrendDirection.NEUTRAL
                score.strength = SignalStrength.WEAK
            
        except Exception as e:
            logger.error(f"Error calculating HTF trend score: {e}")
        
        return score
    
    def calculate_zone_strength(self, data: pd.DataFrame, current_price: float) -> ZoneStrength:
        """
        Step 2: HTF Structure Strength Calculation
        
        Zone_Strength = Σ wi * Si
        Where Si = 1 if feature present, else 0
        wi = weight for feature importance
        """
        strength = ZoneStrength()
        
        try:
            if data.empty:
                return strength
            
            # FVG (Fair Value Gap) detection
            strength.fvg_score = self._detect_fvg_strength(data, current_price)
            
            # Order Block detection
            strength.order_block_score = self._detect_order_block_strength(data, current_price)
            
            # Swing High/Low analysis
            strength.swing_score = self._detect_swing_strength(data, current_price)
            
            # Calculate total weighted score
            strength.total_score = (
                self.zone_weights['fvg'] * strength.fvg_score +
                self.zone_weights['order_block'] * strength.order_block_score +
                self.zone_weights['swing'] * strength.swing_score
            )
            
            # Determine if zone is strong
            strength.is_strong = strength.total_score >= self.config['zone_strength_threshold']
            
        except Exception as e:
            logger.error(f"Error calculating zone strength: {e}")
        
        return strength
    
    def calculate_confluence_score(self, htf_trend: HTFTrendScore, zone_strength: ZoneStrength,
                                 data: pd.DataFrame, current_price: float) -> ConfluenceScore:
        """
        Step 2: Confluence Scoring
        
        Confluence_Score = Σ wi * Si
        Where Si = 1 if condition met, 0 otherwise
        """
        confluence = ConfluenceScore()
        
        try:
            # HTF alignment (30% weight)
            confluence.htf_alignment = 1.0 if abs(htf_trend.weighted_score) >= 0.6 else 0.0
            
            # Pullback to FVG/OB (25% weight)
            confluence.pullback_to_zone = 1.0 if zone_strength.is_strong else 0.0
            
            # Candlestick confirmation (20% weight)
            confluence.candlestick_confirmation = self._analyze_candlestick_patterns(data)
            
            # Momentum indicator (15% weight)
            confluence.momentum_indicator = self._calculate_momentum_score(data)
            
            # Volume spike (10% weight)
            confluence.volume_spike = self._detect_volume_spike(data)
            
            # Calculate total weighted score
            confluence.total_score = (
                self.confluence_weights['htf_alignment'] * confluence.htf_alignment +
                self.confluence_weights['pullback_to_zone'] * confluence.pullback_to_zone +
                self.confluence_weights['candlestick_confirmation'] * confluence.candlestick_confirmation +
                self.confluence_weights['momentum_indicator'] * confluence.momentum_indicator +
                self.confluence_weights['volume_spike'] * confluence.volume_spike
            )
            
            # Check if meets threshold
            confluence.meets_threshold = confluence.total_score >= self.config['confluence_threshold']
            
        except Exception as e:
            logger.error(f"Error calculating confluence score: {e}")
        
        return confluence
    
    def ml_filter_prediction(self, htf_trend: HTFTrendScore, confluence: ConfluenceScore,
                           data: pd.DataFrame, additional_features: Dict[str, float] = None) -> MLPrediction:
        """
        Step 3: ML Filtering
        
        Feature vector: X = [HTF_Trend_Score, Confluence_Score, ATR, Momentum, Volume, ...]
        ML outputs probability: P(success) = fθ(X) ∈ [0,1]
        Dynamic threshold: P_threshold = P_base + k * HTF_Trend_Strength
        """
        prediction = MLPrediction()
        
        if not ML_AVAILABLE or not self.model_trained:
            # Fallback to rule-based scoring
            prediction.probability = self._rule_based_probability(htf_trend, confluence)
            prediction.confidence = 0.5
            prediction.meets_threshold = prediction.probability >= self.config['ml_base_threshold']
            return prediction
        
        try:
            # Build feature vector
            features = self._build_feature_vector(htf_trend, confluence, data, additional_features)
            prediction.features = features
            
            # Scale features
            feature_array = np.array(list(features.values())).reshape(1, -1)
            if self.feature_scaler:
                feature_array = self.feature_scaler.transform(feature_array)
            
            # Get ML prediction
            if self.ml_model:
                prediction.probability = self.ml_model.predict_proba(feature_array)[0][1]  # Probability of success
                
                # Calculate confidence (distance from 0.5)
                prediction.confidence = abs(prediction.probability - 0.5) * 2
                
                # Dynamic threshold adjustment
                if self.config['ml_dynamic_adjustment']:
                    base_threshold = self.config['ml_base_threshold']
                    trend_strength = abs(htf_trend.weighted_score)
                    dynamic_threshold = base_threshold + (0.1 * trend_strength)
                else:
                    dynamic_threshold = self.config['ml_base_threshold']
                
                prediction.meets_threshold = (
                    prediction.probability >= dynamic_threshold and
                    prediction.confidence >= self.config['ml_confidence_threshold']
                )
            
        except Exception as e:
            logger.error(f"Error in ML prediction: {e}")
            # Fallback
            prediction.probability = self._rule_based_probability(htf_trend, confluence)
            prediction.meets_threshold = prediction.probability >= self.config['ml_base_threshold']
        
        return prediction
    
    def calculate_entry_levels(self, signal_type: str, data: pd.DataFrame, 
                             htf_trend: HTFTrendScore, ml_prediction: MLPrediction) -> Dict[str, float]:
        """
        Step 4: Entry Execution (Mathematical)
        
        Risk sizing: Risk_per_trade = Base_Risk * (1 + α * (P(success) - 0.5)) * ATR_M1/ATR_H1
        Entry price: depends on strategy type
        TP/SL: calculated based on ATR and R:R ratios
        """
        levels = {'entry': 0.0, 'stop_loss': 0.0, 'take_profits': [], 'position_size': 0.0}
        
        try:
            current_price = data['close'].iloc[-1]
            
            # Calculate ATR for volatility adjustment
            if TALIB_AVAILABLE and len(data) >= 14:
                atr = talib.ATR(data['high'].values, data['low'].values, data['close'].values, timeperiod=14)
                current_atr = atr[-1] if len(atr) > 0 else current_price * 0.001
            else:
                # Fallback ATR calculation
                high_low = data['high'] - data['low']
                high_close = np.abs(data['high'] - data['close'].shift(1))
                low_close = np.abs(data['low'] - data['close'].shift(1))
                true_range = np.maximum(high_low, np.maximum(high_close, low_close))
                current_atr = true_range.rolling(window=14).mean().iloc[-1]
            
            # Risk sizing calculation
            if signal_type.lower() in ['trend', 'trend_following']:
                base_risk = self.config['base_risk_trend']
            else:
                base_risk = self.config['base_risk_counter']
            
            # Adjust risk based on ML confidence
            confidence_multiplier = 1.0 + (ml_prediction.confidence - 0.5)
            confidence_multiplier = max(0.5, min(confidence_multiplier, self.config['max_risk_multiplier']))
            
            # Volatility adjustment (placeholder - would need multi-timeframe ATR)
            volatility_multiplier = 1.0  # ATR_M1 / ATR_H1 ratio
            
            risk_per_trade = base_risk * confidence_multiplier * volatility_multiplier
            
            # Entry price calculation
            if signal_type.lower() in ['trend', 'trend_following']:
                if htf_trend.direction == TrendDirection.BULLISH:
                    # Long entry - buy above current level
                    pullback_offset = current_atr * 0.25
                    levels['entry'] = current_price + pullback_offset
                else:
                    # Short entry - sell below current level
                    pullback_offset = current_atr * 0.25
                    levels['entry'] = current_price - pullback_offset
            else:
                # Counter-trend/reversal entry
                safety_offset = current_atr * 0.15
                if htf_trend.direction == TrendDirection.BULLISH:
                    levels['entry'] = current_price - safety_offset
                else:
                    levels['entry'] = current_price + safety_offset
            
            # Stop loss calculation
            atr_sl_distance = current_atr * self.config['atr_sl_multiplier']
            
            if htf_trend.direction == TrendDirection.BULLISH:
                levels['stop_loss'] = levels['entry'] - atr_sl_distance
            else:
                levels['stop_loss'] = levels['entry'] + atr_sl_distance
            
            # Take profit levels
            tp_multipliers = self.config['atr_tp_multiplier']
            for i, multiplier in enumerate(tp_multipliers):
                tp_distance = current_atr * multiplier
                if htf_trend.direction == TrendDirection.BULLISH:
                    tp_level = levels['entry'] + tp_distance
                else:
                    tp_level = levels['entry'] - tp_distance
                levels['take_profits'].append(tp_level)
            
            # Position size (simplified - would need account balance)
            risk_amount = risk_per_trade  # This would be risk_per_trade * account_balance
            price_difference = abs(levels['entry'] - levels['stop_loss'])
            if price_difference > 0:
                levels['position_size'] = risk_amount / price_difference
            else:
                levels['position_size'] = 0.0
            
        except Exception as e:
            logger.error(f"Error calculating entry levels: {e}")
        
        return levels
    
    def analyze_market_regime(self, data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        Step 7: Regime Awareness
        
        Regime = ATR_H1/ATR_D1 + Trend_Score
        Only allow appropriate strategies based on regime
        """
        regime_data = {'regime_score': 0.0, 'volatility_ratio': 1.0, 'trend_strength': 0.0}
        
        try:
            # Calculate ATR ratios for regime detection
            if 'H1' in data and 'D1' in data and not data['H1'].empty and not data['D1'].empty:
                h1_data = data['H1']
                d1_data = data['D1']
                
                if TALIB_AVAILABLE:
                    atr_h1 = talib.ATR(h1_data['high'].values, h1_data['low'].values, h1_data['close'].values, 14)
                    atr_d1 = talib.ATR(d1_data['high'].values, d1_data['low'].values, d1_data['close'].values, 14)
                    
                    if len(atr_h1) > 0 and len(atr_d1) > 0:
                        regime_data['volatility_ratio'] = atr_h1[-1] / atr_d1[-1] if atr_d1[-1] != 0 else 1.0
                
                # Calculate trend strength
                htf_trend = self.calculate_htf_trend_score(data)
                regime_data['trend_strength'] = abs(htf_trend.weighted_score)
                
                # Combine for regime score
                regime_data['regime_score'] = regime_data['volatility_ratio'] + regime_data['trend_strength']
            
        except Exception as e:
            logger.error(f"Error analyzing market regime: {e}")
        
        return regime_data
    
    def generate_entry_signal(self, symbol: str, timeframe: str, data: Dict[str, pd.DataFrame]) -> EntrySignal:
        """
        Main entry point for signal generation
        Combines all components into final trading decision
        """
        current_time = datetime.now()
        
        # Initialize signal
        signal = EntrySignal(
            timestamp=current_time,
            symbol=symbol,
            timeframe=timeframe,
            htf_trend=HTFTrendScore(),
            zone_strength=ZoneStrength(),
            confluence=ConfluenceScore(),
            ml_prediction=MLPrediction()
        )
        
        try:
            # Validate data
            if not data or timeframe not in data or data[timeframe].empty:
                logger.warning(f"Insufficient data for {symbol} {timeframe}")
                return signal
            
            current_data = data[timeframe]
            current_price = current_data['close'].iloc[-1]
            
            # Step 1: Calculate HTF trend score
            signal.htf_trend = self.calculate_htf_trend_score(data)
            
            # Step 2: Calculate zone strength
            signal.zone_strength = self.calculate_zone_strength(current_data, current_price)
            
            # Step 3: Calculate confluence score
            signal.confluence = self.calculate_confluence_score(
                signal.htf_trend, signal.zone_strength, current_data, current_price
            )
            
            # Step 4: ML filtering
            additional_features = {
                'volatility_score': self._calculate_volatility_score(current_data),
                'session_score': self._calculate_session_score(current_time),
                'correlation_score': 0.0  # Would need multiple pairs
            }
            
            signal.ml_prediction = self.ml_filter_prediction(
                signal.htf_trend, signal.confluence, current_data, additional_features
            )
            
            # Step 5: Market regime analysis
            regime_data = self.analyze_market_regime(data)
            signal.volatility_score = regime_data['volatility_ratio']
            
            # Step 6: Final decision logic
            signal.should_enter = self._make_final_decision(signal, regime_data)
            
            if signal.should_enter:
                # Step 7: Calculate entry levels
                strategy_type = 'trend' if abs(signal.htf_trend.weighted_score) >= 0.6 else 'counter_trend'
                entry_levels = self.calculate_entry_levels(
                    strategy_type, current_data, signal.htf_trend, signal.ml_prediction
                )
                
                signal.entry_price = entry_levels['entry']
                signal.stop_loss = entry_levels['stop_loss']
                signal.take_profits = entry_levels['take_profits']
                signal.position_size = entry_levels['position_size']
                
                # Calculate R:R ratio
                if len(signal.take_profits) > 0:
                    risk = abs(signal.entry_price - signal.stop_loss)
                    reward = abs(signal.take_profits[0] - signal.entry_price)
                    signal.risk_reward_ratio = reward / risk if risk > 0 else 0
            
            # Determine signal strength
            signal.signal_strength = self._calculate_overall_signal_strength(signal)
            
        except Exception as e:
            logger.error(f"Error generating entry signal for {symbol}: {e}")
        
        return signal
    
    # Helper methods
    def _calculate_signal_strength(self, score: float) -> SignalStrength:
        """Convert numeric score to signal strength enum"""
        if score >= 0.9:
            return SignalStrength.VERY_STRONG
        elif score >= 0.75:
            return SignalStrength.STRONG
        elif score >= 0.6:
            return SignalStrength.MODERATE
        elif score >= 0.4:
            return SignalStrength.WEAK
        else:
            return SignalStrength.VERY_WEAK
    
    def _detect_fvg_strength(self, data: pd.DataFrame, current_price: float) -> float:
        """Detect Fair Value Gap strength"""
        try:
            if len(data) < 3:
                return 0.0
            
            # Simple FVG detection logic
            recent_data = data.tail(10)
            fvg_score = 0.0
            
            for i in range(1, len(recent_data) - 1):
                prev_candle = recent_data.iloc[i-1]
                current_candle = recent_data.iloc[i]
                next_candle = recent_data.iloc[i+1]
                
                # Bullish FVG: gap between prev low and next high
                if prev_candle['low'] > next_candle['high']:
                    gap_size = prev_candle['low'] - next_candle['high']
                    if current_price >= next_candle['high'] and current_price <= prev_candle['low']:
                        fvg_score = min(1.0, gap_size / (current_price * 0.001))  # Normalize by price
                
                # Bearish FVG: gap between prev high and next low
                elif prev_candle['high'] < next_candle['low']:
                    gap_size = next_candle['low'] - prev_candle['high']
                    if current_price <= next_candle['low'] and current_price >= prev_candle['high']:
                        fvg_score = min(1.0, gap_size / (current_price * 0.001))
            
            return fvg_score
        except Exception:
            return 0.0
    
    def _detect_order_block_strength(self, data: pd.DataFrame, current_price: float) -> float:
        """Detect order block strength"""
        try:
            if len(data) < 5:
                return 0.0
            
            # Simplified order block detection
            recent_data = data.tail(20)
            ob_score = 0.0
            
            # Look for significant candles followed by breaks
            for i in range(2, len(recent_data) - 2):
                candle = recent_data.iloc[i]
                candle_range = candle['high'] - candle['low']
                avg_range = recent_data['high'].rolling(10).mean().iloc[i] - recent_data['low'].rolling(10).mean().iloc[i]
                
                if candle_range > avg_range * 1.5:  # Significant candle
                    # Check if price is near this level
                    distance_to_high = abs(current_price - candle['high']) / current_price
                    distance_to_low = abs(current_price - candle['low']) / current_price
                    
                    if min(distance_to_high, distance_to_low) < 0.002:  # Within 0.2%
                        ob_score = max(ob_score, 0.8)
            
            return ob_score
        except Exception:
            return 0.0
    
    def _detect_swing_strength(self, data: pd.DataFrame, current_price: float) -> float:
        """Detect swing high/low strength"""
        try:
            if len(data) < 10:
                return 0.0
            
            swing_score = 0.0
            recent_data = data.tail(50)
            
            # Simple swing detection
            highs = recent_data['high']
            lows = recent_data['low']
            
            # Find recent swing high
            swing_high = highs.rolling(window=5, center=True).max()
            swing_low = lows.rolling(window=5, center=True).min()
            
            # Check proximity to swing levels
            for i in range(len(recent_data) - 5):
                if not pd.isna(swing_high.iloc[i]):
                    distance = abs(current_price - swing_high.iloc[i]) / current_price
                    if distance < 0.001:  # Within 0.1%
                        swing_score = max(swing_score, 0.7)
                
                if not pd.isna(swing_low.iloc[i]):
                    distance = abs(current_price - swing_low.iloc[i]) / current_price
                    if distance < 0.001:
                        swing_score = max(swing_score, 0.7)
            
            return swing_score
        except Exception:
            return 0.0
    
    def _analyze_candlestick_patterns(self, data: pd.DataFrame) -> float:
        """Analyze candlestick patterns for confirmation"""
        try:
            if len(data) < 5:
                return 0.0
            
            recent_candles = data.tail(5)
            pattern_score = 0.0
            
            # Simple pattern recognition
            last_candle = recent_candles.iloc[-1]
            second_last = recent_candles.iloc[-2] if len(recent_candles) > 1 else last_candle
            
            # Bullish patterns
            if (last_candle['close'] > last_candle['open'] and  # Green candle
                last_candle['close'] > second_last['high']):    # Breaking above previous high
                pattern_score = 0.7
            
            # Bearish patterns  
            elif (last_candle['close'] < last_candle['open'] and  # Red candle
                  last_candle['close'] < second_last['low']):     # Breaking below previous low
                pattern_score = 0.7
            
            # Doji patterns (indecision)
            elif abs(last_candle['close'] - last_candle['open']) < (last_candle['high'] - last_candle['low']) * 0.1:
                pattern_score = 0.3
            
            return pattern_score
        except Exception:
            return 0.0
    
    def _calculate_momentum_score(self, data: pd.DataFrame) -> float:
        """Calculate momentum indicator score"""
        try:
            if len(data) < 14:
                return 0.0
            
            if TALIB_AVAILABLE:
                # RSI momentum
                rsi = talib.RSI(data['close'].values, timeperiod=14)
                if len(rsi) > 0:
                    current_rsi = rsi[-1]
                    if 30 < current_rsi < 70:  # Not overbought/oversold
                        return 0.8
                    elif current_rsi > 70 or current_rsi < 30:
                        return 0.3  # Overbought/oversold - caution
            
            # Fallback: simple momentum
            if len(data) >= 10:
                recent_change = (data['close'].iloc[-1] - data['close'].iloc[-10]) / data['close'].iloc[-10]
                return min(1.0, abs(recent_change) * 100)  # Scale momentum
            
            return 0.5
        except Exception:
            return 0.0
    
    def _detect_volume_spike(self, data: pd.DataFrame) -> float:
        """Detect volume spikes"""
        try:
            if 'volume' not in data.columns or len(data) < 10:
                return 0.0
            
            recent_volume = data['volume'].iloc[-1]
            avg_volume = data['volume'].rolling(window=10).mean().iloc[-1]
            
            if avg_volume > 0:
                volume_ratio = recent_volume / avg_volume
                if volume_ratio > 1.5:  # 50% above average
                    return min(1.0, (volume_ratio - 1) / 2)  # Scale volume spike
            
            return 0.0
        except Exception:
            return 0.0
    
    def _calculate_volatility_score(self, data: pd.DataFrame) -> float:
        """Calculate volatility score for risk adjustment"""
        try:
            if len(data) < 14:
                return 1.0
            
            if TALIB_AVAILABLE:
                atr = talib.ATR(data['high'].values, data['low'].values, data['close'].values, timeperiod=14)
                if len(atr) > 0:
                    current_atr = atr[-1]
                    price = data['close'].iloc[-1]
                    volatility_pct = (current_atr / price) * 100
                    return min(3.0, max(0.5, volatility_pct / 2))  # Scale volatility
            
            return 1.0
        except Exception:
            return 1.0
    
    def _calculate_session_score(self, current_time: datetime) -> float:
        """Calculate trading session score"""
        try:
            # Convert to UTC if needed and get hour
            hour = current_time.hour
            
            # London session (7-16 UTC): +0.1 boost
            if 7 <= hour <= 16:
                return self.config['london_session_boost']
            
            # New York session (12-21 UTC): +0.15 boost  
            elif 12 <= hour <= 21:
                return self.config['new_york_session_boost']
            
            # Overlap session (12-16 UTC): +0.2 boost
            elif 12 <= hour <= 16:
                return self.config['overlap_session_boost']
            
            # Asian session or off-hours
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _rule_based_probability(self, htf_trend: HTFTrendScore, confluence: ConfluenceScore) -> float:
        """Fallback rule-based probability when ML is not available"""
        base_prob = 0.5
        
        # Adjust based on trend strength
        trend_adjustment = abs(htf_trend.weighted_score) * 0.2
        
        # Adjust based on confluence
        confluence_adjustment = confluence.total_score * 0.3
        
        final_prob = base_prob + trend_adjustment + confluence_adjustment
        return min(0.95, max(0.05, final_prob))
    
    def _build_feature_vector(self, htf_trend: HTFTrendScore, confluence: ConfluenceScore,
                            data: pd.DataFrame, additional_features: Dict[str, float] = None) -> Dict[str, float]:
        """Build feature vector for ML model"""
        features = {
            'htf_trend_score': htf_trend.weighted_score,
            'confluence_score': confluence.total_score,
            'htf_alignment': confluence.htf_alignment,
            'zone_strength': confluence.pullback_to_zone,
            'candlestick_conf': confluence.candlestick_confirmation,
            'momentum': confluence.momentum_indicator,
            'volume_spike': confluence.volume_spike,
        }
        
        # Add additional features
        if additional_features:
            features.update(additional_features)
        
        # Add time-based features
        current_time = datetime.now()
        features['hour_of_day'] = current_time.hour / 24.0
        features['day_of_week'] = current_time.weekday() / 7.0
        
        return features
    
    def _make_final_decision(self, signal: EntrySignal, regime_data: Dict[str, float]) -> bool:
        """Make final trading decision based on all factors"""
        try:
            # Check minimum requirements
            if not signal.confluence.meets_threshold:
                return False
            
            if not signal.ml_prediction.meets_threshold:
                return False
            
            # Regime-based filtering
            regime_score = regime_data.get('regime_score', 0.0)
            trend_strength = regime_data.get('trend_strength', 0.0)
            
            # For trend-following strategies
            if abs(signal.htf_trend.weighted_score) >= 0.6:
                # Strong trend - allow trend following
                if regime_score > 1.5:  # Strong trending regime
                    return True
            
            # For counter-trend strategies
            else:
                # Ranging market - allow reversal strategies
                if regime_score < 1.2 and signal.ml_prediction.probability > 0.75:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error in final decision: {e}")
            return False
    
    def _calculate_overall_signal_strength(self, signal: EntrySignal) -> SignalStrength:
        """Calculate overall signal strength"""
        try:
            # Combine multiple factors
            strength_score = (
                abs(signal.htf_trend.weighted_score) * 0.3 +
                signal.confluence.total_score * 0.4 +
                signal.ml_prediction.probability * 0.3
            )
            
            return self._calculate_signal_strength(strength_score)
        except Exception:
            return SignalStrength.WEAK
    
    def train_ml_model(self, historical_data: List[Dict[str, Any]]) -> bool:
        """Train the ML model with historical data"""
        if not ML_AVAILABLE:
            logger.warning("ML libraries not available for model training")
            return False
        
        try:
            if len(historical_data) < 100:
                logger.warning("Insufficient data for ML training")
                return False
            
            # Prepare training data
            features_list = []
            labels = []
            
            for record in historical_data:
                if 'features' in record and 'outcome' in record:
                    features_list.append(list(record['features'].values()))
                    labels.append(1 if record['outcome'] else 0)
            
            if len(features_list) == 0:
                return False
            
            X = np.array(features_list)
            y = np.array(labels)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            self.feature_scaler = StandardScaler()
            X_train_scaled = self.feature_scaler.fit_transform(X_train)
            X_test_scaled = self.feature_scaler.transform(X_test)
            
            # Train model
            self.ml_model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
            
            self.ml_model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = self.ml_model.predict(X_test_scaled)
            
            self.model_performance = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0)
            }
            
            self.model_trained = True
            
            logger.info(f"ML model trained successfully. Accuracy: {self.model_performance['accuracy']:.3f}")
            return True
            
        except Exception as e:
            logger.error(f"ML model training failed: {e}")
            return False
    
    def log_trade_outcome(self, signal: EntrySignal, outcome: bool, pnl: float = 0.0):
        """Log trade outcome for model retraining"""
        trade_record = {
            'timestamp': signal.timestamp,
            'symbol': signal.symbol,
            'features': signal.ml_prediction.features,
            'outcome': outcome,
            'pnl': pnl,
            'htf_trend_score': signal.htf_trend.weighted_score,
            'confluence_score': signal.confluence.total_score,
            'ml_probability': signal.ml_prediction.probability
        }
        
        self.trade_history.append(trade_record)
        
        # Retrain model periodically
        if len(self.trade_history) % 50 == 0 and len(self.trade_history) >= 100:
            logger.info("Retraining ML model with recent trade data")
            self.train_ml_model(self.trade_history[-200:])  # Use last 200 trades
    
    def get_model_performance(self) -> Dict[str, float]:
        """Get current ML model performance metrics"""
        return self.model_performance.copy()
    
    def export_model(self, file_path: str) -> bool:
        """Export trained model and scaler"""
        try:
            import pickle
            
            model_data = {
                'model': self.ml_model,
                'scaler': self.feature_scaler,
                'performance': self.model_performance,
                'config': self.config,
                'trained': self.model_trained
            }
            
            with open(file_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Model exported to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Model export failed: {e}")
            return False
    
    def import_model(self, file_path: str) -> bool:
        """Import trained model and scaler"""
        try:
            import pickle
            
            with open(file_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.ml_model = model_data.get('model')
            self.feature_scaler = model_data.get('scaler')
            self.model_performance = model_data.get('performance', {})
            self.model_trained = model_data.get('trained', False)
            
            logger.info(f"Model imported from {file_path}")
            return True
        except Exception as e:
            logger.error(f"Model import failed: {e}")
            return False


# Convenience functions for easy integration
def create_entry_logic(config: Dict[str, Any] = None) -> EdenEntryLogicV3:
    """Create and return Eden Entry Logic v3.0 instance"""
    return EdenEntryLogicV3(config)

def analyze_symbol(symbol: str, timeframe: str, data: Dict[str, pd.DataFrame], 
                  entry_logic: EdenEntryLogicV3 = None) -> EntrySignal:
    """Analyze a symbol and return entry signal"""
    if entry_logic is None:
        entry_logic = create_entry_logic()
    
    return entry_logic.generate_entry_signal(symbol, timeframe, data)