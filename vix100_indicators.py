#!/usr/bin/env python3
"""
VIX100 Advanced Technical Indicators System
===========================================

Specialized technical indicators designed specifically for Volatility Index 100.
These indicators are optimized for synthetic market behavior, continuous trading,
and volatility-based pattern recognition.

Key Features:
- Volatility pressure and compression detection
- Synthetic wave pattern analysis
- Tick burst rate calculations
- Market regime classification for VIX100
- Anomaly detection in synthetic data
- Continuous trading cycle analysis

Author: Eden AI System
Version: 1.0
Date: October 13, 2025
"""

import pandas as pd
import numpy as np
import scipy.signal as signal
from scipy.stats import zscore
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

@dataclass
class VolatilityPressureReading:
    """Volatility pressure analysis result"""
    current_pressure: float
    pressure_percentile: float
    pressure_trend: str  # 'increasing', 'decreasing', 'stable'
    burst_probability: float
    compression_level: float

@dataclass 
class SyntheticWavePattern:
    """Synthetic wave pattern detection"""
    wave_type: str  # 'impulse', 'correction', 'compression', 'expansion'
    amplitude: float
    frequency: float
    phase_shift: float
    confidence: float
    duration_bars: int

@dataclass
class MarketRegimeSignal:
    """VIX100 market regime classification"""
    regime: str  # 'burst', 'compression', 'trend', 'chaos', 'recovery'
    confidence: float
    expected_duration: int
    volatility_forecast: float
    regime_strength: float

class VIX100VolatilityEngine:
    """Advanced volatility analysis for VIX100"""
    
    def __init__(self):
        self.lookback_periods = {
            'short': 20,
            'medium': 50, 
            'long': 100,
            'ultra_long': 200
        }
    
    def calculate_volatility_pressure(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive volatility pressure metrics"""
        df = df.copy()
        
        # Basic volatility pressure (range relative to price)
        df['vol_pressure_basic'] = (df['high'] - df['low']) / df['close']
        
        # Advanced volatility pressure (considering volume and time)
        df['vol_pressure_weighted'] = (df['vol_pressure_basic'] * 
                                     np.log1p(df['tick_volume']) / 
                                     np.log1p(df['tick_volume'].rolling(20).mean()))
        
        # Volatility pressure momentum
        df['vol_pressure_momentum'] = df['vol_pressure_basic'].pct_change(5)
        
        # Volatility pressure acceleration
        df['vol_pressure_acceleration'] = df['vol_pressure_momentum'].diff()
        
        # Pressure percentiles for context
        for period in [20, 50, 100]:
            df[f'vol_pressure_percentile_{period}'] = (
                df['vol_pressure_basic'].rolling(period).rank() / period * 100
            )
        
        # Pressure burst detection
        pressure_std = df['vol_pressure_basic'].rolling(50).std()
        pressure_mean = df['vol_pressure_basic'].rolling(50).mean()
        df['vol_burst_signal'] = (df['vol_pressure_basic'] > 
                                 pressure_mean + 2 * pressure_std).astype(int)
        
        return df
    
    def detect_volatility_compression(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect volatility compression states"""
        df = df.copy()
        
        # Bollinger Band width (normalized)
        df['bb_middle'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_width'] = (2 * df['bb_std']) / df['bb_middle']
        
        # Compression detection
        compression_threshold = df['bb_width'].rolling(100).quantile(0.2)
        df['compression_state'] = (df['bb_width'] < compression_threshold).astype(int)
        
        # Compression intensity
        df['compression_intensity'] = (compression_threshold - df['bb_width']) / compression_threshold
        df['compression_intensity'] = df['compression_intensity'].clip(0, 1)
        
        # Time in compression
        df['compression_duration'] = df.groupby(
            (df['compression_state'] != df['compression_state'].shift()).cumsum()
        ).cumcount()
        
        return df
    
    def analyze_volatility_cycles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze VIX100 volatility cycles"""
        df = df.copy()
        
        # Create volatility proxy
        df['vol_proxy'] = df['vol_pressure_basic'].rolling(10).mean()
        
        # Detect cycles using peak detection
        vol_series = df['vol_proxy'].dropna()
        
        if len(vol_series) > 50:
            # Find peaks and troughs
            peaks, _ = signal.find_peaks(vol_series, height=vol_series.quantile(0.7))
            troughs, _ = signal.find_peaks(-vol_series, height=-vol_series.quantile(0.3))
            
            # Calculate cycle characteristics
            if len(peaks) > 1:
                peak_distances = np.diff(peaks)
                df['avg_cycle_length'] = np.mean(peak_distances) if len(peak_distances) > 0 else np.nan
            
            # Cycle position (where are we in the cycle?)
            df['cycle_position'] = self._calculate_cycle_position(vol_series, peaks, troughs)
        
        return df
    
    def _calculate_cycle_position(self, series: pd.Series, peaks: np.ndarray, troughs: np.ndarray) -> pd.Series:
        """Calculate position within volatility cycle"""
        cycle_position = pd.Series(index=series.index, dtype=float)
        
        for i in range(len(series)):
            # Find nearest peak and trough
            recent_peaks = peaks[peaks < i]
            recent_troughs = troughs[troughs < i]
            
            if len(recent_peaks) > 0 and len(recent_troughs) > 0:
                last_peak = recent_peaks[-1]
                last_trough = recent_troughs[-1]
                
                if last_peak > last_trough:
                    # After peak, heading to trough
                    cycle_position.iloc[i] = 0.5 + (i - last_peak) / (2 * np.mean(np.diff(peaks)) if len(peaks) > 1 else 50)
                else:
                    # After trough, heading to peak
                    cycle_position.iloc[i] = (i - last_trough) / (2 * np.mean(np.diff(peaks)) if len(peaks) > 1 else 50)
        
        return cycle_position.clip(0, 1)

class VIX100SyntheticPatternDetector:
    """Detect patterns specific to synthetic markets"""
    
    def __init__(self):
        self.pattern_memory = {}
    
    def detect_spike_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect spike and retrace patterns common in VIX100"""
        df = df.copy()
        
        # Price spike detection
        price_returns = df['close'].pct_change()
        spike_threshold = price_returns.rolling(50).quantile(0.95)
        
        df['price_spike_up'] = (price_returns > spike_threshold).astype(int)
        df['price_spike_down'] = (price_returns < -spike_threshold).astype(int)
        
        # Spike followed by retrace
        df['spike_retrace_up'] = (
            (df['price_spike_up'] == 1) & 
            (price_returns.shift(-1) < 0) &
            (price_returns.shift(-2) < 0)
        ).astype(int)
        
        df['spike_retrace_down'] = (
            (df['price_spike_down'] == 1) & 
            (price_returns.shift(-1) > 0) &
            (price_returns.shift(-2) > 0)
        ).astype(int)
        
        return df
    
    def detect_false_breakouts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect false breakouts in synthetic markets"""
        df = df.copy()
        
        # Calculate dynamic support/resistance levels
        df['dynamic_resistance'] = df['high'].rolling(20).max()
        df['dynamic_support'] = df['low'].rolling(20).min()
        
        # Breakout detection
        df['breakout_up'] = (df['high'] > df['dynamic_resistance'].shift(1)).astype(int)
        df['breakout_down'] = (df['low'] < df['dynamic_support'].shift(1)).astype(int)
        
        # False breakout detection (breakout followed by reversal)
        df['false_breakout_up'] = (
            (df['breakout_up'] == 1) & 
            (df['close'] < df['dynamic_resistance'].shift(1)) &
            (df['close'].shift(1) < df['dynamic_resistance'].shift(2))
        ).astype(int)
        
        df['false_breakout_down'] = (
            (df['breakout_down'] == 1) & 
            (df['close'] > df['dynamic_support'].shift(1)) &
            (df['close'].shift(1) > df['dynamic_support'].shift(2))
        ).astype(int)
        
        return df
    
    def detect_synthetic_waves(self, df: pd.DataFrame) -> List[SyntheticWavePattern]:
        """Detect wave patterns in synthetic market data"""
        if len(df) < 100:
            return []
        
        patterns = []
        price_series = df['close'].values
        
        # Use wavelets or Fourier analysis to detect patterns
        from scipy.fft import fft, fftfreq
        
        # Apply FFT to detect dominant frequencies
        fft_values = fft(price_series[-100:])  # Last 100 bars
        frequencies = fftfreq(100)
        
        # Find dominant frequency
        magnitude = np.abs(fft_values)
        dominant_freq_idx = np.argmax(magnitude[1:]) + 1  # Skip DC component
        dominant_freq = frequencies[dominant_freq_idx]
        
        if abs(dominant_freq) > 0:
            pattern = SyntheticWavePattern(
                wave_type='impulse' if magnitude[dominant_freq_idx] > np.median(magnitude) * 2 else 'correction',
                amplitude=float(np.std(price_series[-100:])),
                frequency=float(abs(dominant_freq)),
                phase_shift=float(np.angle(fft_values[dominant_freq_idx])),
                confidence=float(magnitude[dominant_freq_idx] / np.sum(magnitude)),
                duration_bars=int(1 / abs(dominant_freq)) if abs(dominant_freq) > 0 else 10
            )
            patterns.append(pattern)
        
        return patterns

class VIX100RegimeClassifier:
    """Classify VIX100 market regimes"""
    
    def __init__(self):
        self.regime_history = []
        self.regime_transitions = {
            'compression': ['burst', 'trend'],
            'burst': ['compression', 'chaos', 'recovery'],
            'trend': ['compression', 'burst'],
            'chaos': ['recovery', 'compression'],
            'recovery': ['trend', 'compression']
        }
    
    def classify_regime(self, df: pd.DataFrame) -> MarketRegimeSignal:
        """Classify current market regime"""
        if len(df) < 50:
            return MarketRegimeSignal('unknown', 0.0, 0, 0.0, 0.0)
        
        # Calculate regime indicators
        volatility = df['vol_pressure_basic'].iloc[-20:].mean()
        volatility_trend = df['vol_pressure_basic'].iloc[-10:].mean() / df['vol_pressure_basic'].iloc[-20:-10].mean()
        
        compression = df['compression_state'].iloc[-10:].mean()
        burst_signals = df['vol_burst_signal'].iloc[-10:].sum()
        
        price_momentum = df['close'].pct_change(10).iloc[-1]
        trend_strength = abs(df['close'].rolling(20).mean().pct_change(10).iloc[-1])
        
        # Regime classification logic
        if compression > 0.7:
            regime = 'compression'
            confidence = compression
            expected_duration = 20
            volatility_forecast = volatility * 0.5
            
        elif burst_signals > 3:
            regime = 'burst'
            confidence = min(burst_signals / 5, 1.0)
            expected_duration = 10
            volatility_forecast = volatility * 2.0
            
        elif trend_strength > 0.02:
            regime = 'trend'
            confidence = min(trend_strength * 50, 1.0)
            expected_duration = 50
            volatility_forecast = volatility * 1.2
            
        elif volatility > df['vol_pressure_basic'].rolling(100).quantile(0.8).iloc[-1]:
            regime = 'chaos'
            confidence = 0.7
            expected_duration = 15
            volatility_forecast = volatility * 1.5
            
        else:
            regime = 'recovery'
            confidence = 0.6
            expected_duration = 30
            volatility_forecast = volatility * 0.8
        
        regime_strength = self._calculate_regime_strength(df, regime)
        
        signal = MarketRegimeSignal(
            regime=regime,
            confidence=confidence,
            expected_duration=expected_duration,
            volatility_forecast=volatility_forecast,
            regime_strength=regime_strength
        )
        
        self.regime_history.append(signal)
        return signal
    
    def _calculate_regime_strength(self, df: pd.DataFrame, regime: str) -> float:
        """Calculate how strong the current regime signal is"""
        if regime == 'compression':
            return df['compression_intensity'].iloc[-10:].mean()
        elif regime == 'burst':
            return min(df['vol_burst_signal'].iloc[-5:].sum() / 5, 1.0)
        elif regime == 'trend':
            return min(abs(df['close'].rolling(10).mean().pct_change(5).iloc[-1]) * 100, 1.0)
        else:
            return 0.5

class VIX100AnomalyDetector:
    """Detect anomalies in VIX100 data streams"""
    
    def __init__(self, contamination: float = 0.1):
        self.contamination = contamination
        self.scaler = StandardScaler()
        self.detector = None
        self.is_fitted = False
    
    def fit_detector(self, df: pd.DataFrame) -> bool:
        """Fit anomaly detector on historical data"""
        if len(df) < 100:
            return False
        
        try:
            # Prepare features for anomaly detection
            features = self._prepare_anomaly_features(df)
            
            if features.shape[0] < 50:
                return False
            
            # Scale features
            features_scaled = self.scaler.fit_transform(features)
            
            # Fit isolation forest
            from sklearn.ensemble import IsolationForest
            self.detector = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=100
            )
            self.detector.fit(features_scaled)
            self.is_fitted = True
            
            return True
        except Exception as e:
            print(f"Error fitting anomaly detector: {e}")
            return False
    
    def detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalies in current data"""
        df = df.copy()
        
        if not self.is_fitted:
            df['anomaly_score'] = 0.0
            df['is_anomaly'] = False
            return df
        
        try:
            features = self._prepare_anomaly_features(df)
            if features.shape[0] == 0:
                df['anomaly_score'] = 0.0
                df['is_anomaly'] = False
                return df
            
            features_scaled = self.scaler.transform(features)
            
            # Get anomaly scores
            scores = self.detector.decision_function(features_scaled)
            predictions = self.detector.predict(features_scaled)
            
            # Add to dataframe
            df['anomaly_score'] = np.nan
            df['is_anomaly'] = False
            
            if len(scores) == len(df):
                df['anomaly_score'] = scores
                df['is_anomaly'] = predictions == -1
            
            return df
            
        except Exception as e:
            print(f"Error detecting anomalies: {e}")
            df['anomaly_score'] = 0.0
            df['is_anomaly'] = False
            return df
    
    def _prepare_anomaly_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare features for anomaly detection"""
        features = []
        
        if len(df) < 10:
            return np.array([])
        
        try:
            # Price-based features
            features.append(df['close'].pct_change().fillna(0))
            features.append(df['vol_pressure_basic'].fillna(df['vol_pressure_basic'].mean()))
            
            if 'tick_volume' in df.columns:
                features.append(np.log1p(df['tick_volume']))
            
            # Volatility features
            if len(df) >= 20:
                features.append(df['close'].rolling(20).std().fillna(df['close'].std()))
            
            # Combine features
            feature_matrix = np.column_stack(features)
            
            # Remove any remaining NaN values
            feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)
            
            return feature_matrix
            
        except Exception as e:
            print(f"Error preparing anomaly features: {e}")
            return np.array([])

class VIX100IndicatorSuite:
    """Complete VIX100 indicator suite"""
    
    def __init__(self):
        self.volatility_engine = VIX100VolatilityEngine()
        self.pattern_detector = VIX100SyntheticPatternDetector()
        self.regime_classifier = VIX100RegimeClassifier()
        self.anomaly_detector = VIX100AnomalyDetector()
        self.is_initialized = False
    
    def initialize(self, historical_data: pd.DataFrame) -> bool:
        """Initialize all indicators with historical data"""
        try:
            # Fit anomaly detector
            success = self.anomaly_detector.fit_detector(historical_data)
            self.is_initialized = success
            return success
        except Exception as e:
            print(f"Error initializing indicators: {e}")
            return False
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all VIX100 indicators"""
        if df.empty:
            return df
        
        result_df = df.copy()
        
        try:
            # Volatility indicators
            result_df = self.volatility_engine.calculate_volatility_pressure(result_df)
            result_df = self.volatility_engine.detect_volatility_compression(result_df)
            result_df = self.volatility_engine.analyze_volatility_cycles(result_df)
            
            # Pattern detection
            result_df = self.pattern_detector.detect_spike_patterns(result_df)
            result_df = self.pattern_detector.detect_false_breakouts(result_df)
            
            # Anomaly detection
            if self.is_initialized:
                result_df = self.anomaly_detector.detect_anomalies(result_df)
            
            # Add tick burst rate if tick_volume is available
            if 'tick_volume' in result_df.columns:
                result_df['tick_burst_rate'] = result_df['tick_volume'].pct_change().rolling(5).std()
            
            # Add synthetic wave momentum
            result_df['wave_momentum'] = result_df['close'].pct_change(10)
            result_df['wave_acceleration'] = result_df['wave_momentum'].diff()
            
            return result_df
            
        except Exception as e:
            print(f"Error calculating indicators: {e}")
            return df
    
    def get_market_analysis(self, df: pd.DataFrame) -> Dict:
        """Get comprehensive market analysis"""
        if len(df) < 50:
            return {'status': 'insufficient_data'}
        
        try:
            # Get regime classification
            regime_signal = self.regime_classifier.classify_regime(df)
            
            # Get synthetic wave patterns
            wave_patterns = self.pattern_detector.detect_synthetic_waves(df)
            
            # Current readings
            current_pressure = df['vol_pressure_basic'].iloc[-1] if 'vol_pressure_basic' in df else 0
            current_compression = df['compression_state'].iloc[-1] if 'compression_state' in df else 0
            
            return {
                'status': 'success',
                'regime': {
                    'current': regime_signal.regime,
                    'confidence': regime_signal.confidence,
                    'expected_duration': regime_signal.expected_duration,
                    'strength': regime_signal.regime_strength
                },
                'volatility': {
                    'current_pressure': current_pressure,
                    'compression_state': bool(current_compression),
                    'forecast': regime_signal.volatility_forecast
                },
                'patterns': {
                    'wave_count': len(wave_patterns),
                    'dominant_pattern': wave_patterns[0].wave_type if wave_patterns else None
                },
                'anomalies': {
                    'recent_count': df['is_anomaly'].iloc[-10:].sum() if 'is_anomaly' in df else 0,
                    'current_score': df['anomaly_score'].iloc[-1] if 'anomaly_score' in df else 0
                }
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

# Utility functions for VIX100 analysis
def calculate_vix100_strength(df: pd.DataFrame) -> float:
    """Calculate overall VIX100 market strength"""
    if len(df) < 20:
        return 0.5
    
    # Combine multiple factors
    volatility_strength = df['vol_pressure_basic'].iloc[-10:].mean()
    trend_strength = abs(df['close'].rolling(10).mean().pct_change(5).iloc[-1])
    volume_strength = df['tick_volume'].iloc[-10:].mean() / df['tick_volume'].rolling(50).mean().iloc[-1]
    
    # Normalize and combine
    combined_strength = (volatility_strength * 0.4 + 
                        trend_strength * 100 * 0.3 + 
                        volume_strength * 0.3)
    
    return min(max(combined_strength, 0), 1)

def get_vix100_trading_zones(df: pd.DataFrame) -> Dict[str, float]:
    """Get dynamic trading zones for VIX100"""
    if len(df) < 50:
        return {}
    
    current_price = df['close'].iloc[-1]
    
    # Calculate dynamic levels based on volatility
    atr = (df['high'] - df['low']).rolling(14).mean().iloc[-1]
    
    zones = {
        'strong_buy': current_price - (atr * 2),
        'buy': current_price - atr,
        'neutral_low': current_price - (atr * 0.5),
        'current': current_price,
        'neutral_high': current_price + (atr * 0.5),
        'sell': current_price + atr,
        'strong_sell': current_price + (atr * 2)
    }
    
    return zones

if __name__ == "__main__":
    # Example usage
    print("VIX100 Indicators System - Ready for Integration")
    
    # Create sample data for testing
    np.random.seed(42)
    dates = pd.date_range('2025-01-01', periods=1000, freq='5min')
    
    sample_data = pd.DataFrame({
        'open': 100 + np.cumsum(np.random.randn(1000) * 0.1),
        'high': 100 + np.cumsum(np.random.randn(1000) * 0.1) + np.random.rand(1000) * 2,
        'low': 100 + np.cumsum(np.random.randn(1000) * 0.1) - np.random.rand(1000) * 2,
        'close': 100 + np.cumsum(np.random.randn(1000) * 0.1),
        'tick_volume': np.random.randint(100, 1000, 1000)
    }, index=dates)
    
    sample_data['high'] = sample_data[['open', 'close', 'high']].max(axis=1)
    sample_data['low'] = sample_data[['open', 'close', 'low']].min(axis=1)
    
    # Test the indicator suite
    indicator_suite = VIX100IndicatorSuite()
    
    # Initialize with historical data
    init_success = indicator_suite.initialize(sample_data[:500])
    print(f"Initialization: {'Success' if init_success else 'Failed'}")
    
    # Calculate indicators
    result = indicator_suite.calculate_all_indicators(sample_data)
    print(f"Calculated {len([col for col in result.columns if col not in sample_data.columns])} new indicators")
    
    # Get market analysis
    analysis = indicator_suite.get_market_analysis(result)
    print("Market Analysis:", analysis)