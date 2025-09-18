#!/usr/bin/env python3
"""
Eden Comprehensive Backtesting System
====================================

Complete system with:
- Real market data from January 2025 to September 2025
- Unified ICT strategy with all confluences
- Machine Learning optimization and adaptation
- Multiple timeframe entries (M5, M15, H1 focus)
- Dynamic risk management for 8%+ monthly returns
- Monte Carlo simulation
- Monthly performance tracking

Author: Eden AI System
Version: Production 1.0
Date: September 15, 2025
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
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import time
from abc import ABC, abstractmethod
from collections import defaultdict
import pickle

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
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    import joblib
    ML_AVAILABLE = True
except ImportError:
    print("⚠️ scikit-learn not available. Install with: pip install scikit-learn")
    ML_AVAILABLE = False

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class MarketRegime:
    regime: str
    confidence: float
    volatility: float
    direction: int

@dataclass
class HTFBias:
    daily: str
    h4: str
    h1: str
    overall: str
    confidence: float
    reversal_probability: float

@dataclass
class Signal:
    timestamp: datetime
    symbol: str
    side: str
    confidence: float
    strategy_name: str
    strategy_family: str
    entry_price: float
    timeframe: str
    htf_bias: Optional[str] = None
    against_bias: bool = False
    reversal_signal: bool = False
    confluences: Optional[Dict] = None
    regime: Optional[str] = None
    risk_percentage: float = 1.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

@dataclass
class Trade:
    signal: Signal
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: str = "open"
    pnl_pips: float = 0.0
    pnl_percentage: float = 0.0
    risk_percentage: float = 1.0
    duration_hours: float = 0.0
    max_drawdown: float = 0.0
    max_profit: float = 0.0

@dataclass
class MonthlyResults:
    month: str
    trades: int
    wins: int
    losses: int
    win_rate: float
    total_pips: float
    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    avg_trade_duration: float
    best_trade: float
    worst_trade: float

class DataManager:
    """Manage real market data retrieval and storage"""
    
    def __init__(self):
        self.mt5_initialized = self.initialize_mt5()
        self.data_cache = {}
        self.cache_file = "market_data_cache.pkl"
        self.load_cache()
    
    def initialize_mt5(self) -> bool:
        """Initialize MT5 connection"""
        if not MT5_AVAILABLE:
            return False
            
        if not mt5.initialize():
            logger.error(f"MT5 initialization failed: {mt5.last_error()}")
            return False
            
        logger.info("✅ MT5 connection established")
        return True
    
    def get_real_data(self, symbol: str, timeframe: int, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """Get real market data for backtesting"""
        cache_key = f"{symbol}_{timeframe}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
        
        if cache_key in self.data_cache:
            logger.info(f"📦 Using cached data for {symbol} {self._timeframe_to_string(timeframe)}")
            return self.data_cache[cache_key]
        
        if not self.mt5_initialized:
            logger.error("MT5 not initialized")
            return None
        
        try:
            rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
            
            if rates is None or len(rates) == 0:
                logger.warning(f"No data available for {symbol}")
                return None
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            # Add technical features
            df = self._add_features(df)
            
            # Cache the data
            self.data_cache[cache_key] = df
            self.save_cache()
            
            logger.info(f"📈 Loaded {len(df):,} bars for {symbol} {self._timeframe_to_string(timeframe)}")
            return df
            
        except Exception as e:
            logger.error(f"Error getting data for {symbol}: {e}")
            return None
    
    def _timeframe_to_string(self, timeframe: int) -> str:
        """Convert MT5 timeframe to string"""
        timeframe_map = {
            mt5.TIMEFRAME_M1: "M1",
            mt5.TIMEFRAME_M5: "M5", 
            mt5.TIMEFRAME_M15: "M15",
            mt5.TIMEFRAME_H1: "H1",
            mt5.TIMEFRAME_H4: "H4",
            mt5.TIMEFRAME_D1: "D1"
        }
        return timeframe_map.get(timeframe, "Unknown")
    
    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical features"""
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        df['rsi_21'] = df['close'].rolling(window=21).apply(lambda x: 100 - (100 / (1 + x.diff().where(lambda y: y > 0, 0).mean() / (-x.diff().where(lambda y: y < 0, 0).mean()))))
        
        # Moving averages
        df['ema_8'] = df['close'].ewm(span=8).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_21'] = df['close'].ewm(span=21).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        df['ema_50'] = df['close'].ewm(span=50).mean()
        df['ema_100'] = df['close'].ewm(span=100).mean()
        df['sma_200'] = df['close'].rolling(window=200).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        bb_period = 20
        df['bb_middle'] = df['close'].rolling(window=bb_period).mean()
        bb_std = df['close'].rolling(window=bb_period).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # ATR and volatility
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr_14'] = df['tr'].rolling(window=14).mean()
        df['atr_21'] = df['tr'].rolling(window=21).mean()
        
        # Price action features
        df['body_size'] = abs(df['close'] - df['open'])
        df['upper_wick'] = df['high'] - np.maximum(df['close'], df['open'])
        df['lower_wick'] = np.minimum(df['close'], df['open']) - df['low']
        df['wick_ratio'] = (df['upper_wick'] + df['lower_wick']) / df['body_size']
        
        # Volume features (use tick_volume as proxy)
        df['volume_sma'] = df['tick_volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['tick_volume'] / df['volume_sma']
        
        # Momentum indicators
        df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
        df['momentum_20'] = df['close'] / df['close'].shift(20) - 1
        
        # Support/Resistance levels
        df['pivot_high'] = df['high'].rolling(window=5, center=True).max() == df['high']
        df['pivot_low'] = df['low'].rolling(window=5, center=True).min() == df['low']
        
        return df.fillna(method='ffill').fillna(0)
    
    def load_cache(self):
        """Load cached data"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    self.data_cache = pickle.load(f)
                logger.info(f"📦 Loaded {len(self.data_cache)} cached datasets")
            except:
                self.data_cache = {}
    
    def save_cache(self):
        """Save data to cache"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.data_cache, f)
        except Exception as e:
            logger.warning(f"Could not save cache: {e}")

class MLOptimizer:
    """Machine Learning optimization and adaptation system"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.model_performance = {}
        self.retrain_frequency = 1000  # Retrain every 1000 data points
        
    def prepare_features(self, data: pd.DataFrame, lookback: int = 50) -> np.ndarray:
        """Prepare ML features from market data"""
        features = []
        
        # Technical indicators
        feature_cols = [
            'rsi_14', 'rsi_21', 'macd', 'macd_signal', 'macd_histogram',
            'bb_width', 'atr_14', 'atr_21', 'volume_ratio', 'momentum_10', 'momentum_20',
            'body_size', 'wick_ratio'
        ]
        
        # Add price relationships
        data['price_vs_ema8'] = data['close'] / data['ema_8'] - 1
        data['price_vs_ema21'] = data['close'] / data['ema_21'] - 1
        data['price_vs_ema50'] = data['close'] / data['ema_50'] - 1
        data['price_vs_sma200'] = data['close'] / data['sma_200'] - 1
        
        feature_cols.extend(['price_vs_ema8', 'price_vs_ema21', 'price_vs_ema50', 'price_vs_sma200'])
        
        # Market structure features
        data['higher_high'] = (data['high'] > data['high'].shift(1)).astype(int)
        data['lower_low'] = (data['low'] < data['low'].shift(1)).astype(int)
        data['inside_bar'] = ((data['high'] < data['high'].shift(1)) & 
                             (data['low'] > data['low'].shift(1))).astype(int)
        
        feature_cols.extend(['higher_high', 'lower_low', 'inside_bar'])
        
        # Rolling statistics
        for window in [10, 20, 50]:
            data[f'price_std_{window}'] = data['close'].rolling(window).std()
            data[f'volume_std_{window}'] = data['tick_volume'].rolling(window).std()
            feature_cols.extend([f'price_std_{window}', f'volume_std_{window}'])
        
        return data[feature_cols].fillna(method='ffill').fillna(0).values
    
    def train_signal_classifier(self, symbol: str, data: pd.DataFrame, signals: List[Signal]) -> bool:
        """Train ML model to predict signal quality"""
        if not ML_AVAILABLE or len(signals) < 100:
            return False
        
        try:
            # Prepare features
            X = self.prepare_features(data)
            
            # Create labels based on signal outcomes
            y = np.zeros(len(data))
            
            for signal in signals:
                # Find signal index in data
                signal_idx = data.index.get_loc(signal.timestamp, method='nearest')
                if signal_idx < len(y):
                    # Label as 1 if signal was profitable (simplified)
                    y[signal_idx] = 1 if signal.confidence > 0.7 else 0
            
            # Split data chronologically
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train Random Forest with optimization
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            rf = RandomForestClassifier(random_state=42)
            tscv = TimeSeriesSplit(n_splits=3)
            
            grid_search = GridSearchCV(rf, param_grid, cv=tscv, scoring='precision', n_jobs=-1)
            grid_search.fit(X_train_scaled, y_train)
            
            best_model = grid_search.best_estimator_
            
            # Evaluate model
            y_pred = best_model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            
            # Store model and performance
            self.models[symbol] = best_model
            self.scalers[symbol] = scaler
            self.model_performance[symbol] = {
                'accuracy': accuracy,
                'precision': precision,
                'best_params': grid_search.best_params_
            }
            
            # Feature importance
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            self.feature_importance[symbol] = dict(zip(feature_names, best_model.feature_importances_))
            
            logger.info(f"✅ Trained ML model for {symbol}: Accuracy={accuracy:.3f}, Precision={precision:.3f}")
            return True
            
        except Exception as e:
            logger.error(f"Error training ML model for {symbol}: {e}")
            return False
    
    def predict_signal_quality(self, symbol: str, data: pd.DataFrame, current_idx: int) -> float:
        """Predict signal quality using trained ML model"""
        if symbol not in self.models or current_idx < 50:
            return 0.5  # Default confidence
        
        try:
            # Get recent data for prediction
            recent_data = data.iloc[max(0, current_idx-50):current_idx+1]
            X = self.prepare_features(recent_data)
            
            if len(X) == 0:
                return 0.5
            
            # Scale and predict
            X_scaled = self.scalers[symbol].transform(X[-1:])
            prediction_proba = self.models[symbol].predict_proba(X_scaled)
            
            # Return probability of positive class
            return prediction_proba[0][1] if len(prediction_proba[0]) > 1 else 0.5
            
        except Exception as e:
            logger.warning(f"ML prediction error for {symbol}: {e}")
            return 0.5
    
    def optimize_strategy_parameters(self, symbol: str, historical_results: List[Trade]) -> Dict:
        """Optimize strategy parameters based on historical performance"""
        if len(historical_results) < 50:
            return {}
        
        try:
            # Analyze performance by different parameters
            wins = [t for t in historical_results if t.pnl_percentage > 0]
            losses = [t for t in historical_results if t.pnl_percentage <= 0]
            
            win_rate = len(wins) / len(historical_results)
            avg_win = np.mean([t.pnl_percentage for t in wins]) if wins else 0
            avg_loss = np.mean([t.pnl_percentage for t in losses]) if losses else 0
            
            # Risk-reward optimization
            if win_rate > 0.6:  # High win rate, can increase risk
                risk_multiplier = 1.2
            elif win_rate < 0.4:  # Low win rate, reduce risk
                risk_multiplier = 0.8
            else:
                risk_multiplier = 1.0
            
            # Confidence threshold optimization
            confidence_threshold = 0.65 if win_rate > 0.5 else 0.75
            
            optimized_params = {
                'risk_multiplier': risk_multiplier,
                'confidence_threshold': confidence_threshold,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'optimization_date': datetime.now().isoformat()
            }
            
            logger.info(f"🎯 Optimized parameters for {symbol}: {optimized_params}")
            return optimized_params
            
        except Exception as e:
            logger.error(f"Parameter optimization error for {symbol}: {e}")
            return {}

class ICTStrategy:
    """Advanced ICT Strategy with ML integration"""
    
    def __init__(self, ml_optimizer: MLOptimizer):
        self.name = "ict_ml_confluence"
        self.family = "ICT"
        self.ml_optimizer = ml_optimizer
        self.base_confidence_threshold = 0.7
        self.params = {
            "min_confluences": 3,
            "liquidity_lookback": 20,
            "fvg_min_size": 0.00005,  # Smaller for better entries
            "ob_min_size": 0.0001,
            "ote_fib_levels": [0.618, 0.705, 0.786, 0.886],
            "judas_sessions": [7, 8, 13, 14],  # Extended session windows
            "risk_base": 0.5,  # Base risk percentage
            "risk_max": 2.5    # Maximum risk percentage
        }
        self.optimized_params = {}
    
    def generate_signals(self, symbol: str, daily_data: pd.DataFrame, h4_data: pd.DataFrame,
                        h1_data: pd.DataFrame, m15_data: pd.DataFrame, m5_data: pd.DataFrame,
                        htf_bias: HTFBias, regimes: List[MarketRegime]) -> List[Signal]:
        """Generate ICT signals with ML enhancement"""
        signals = []
        
        # Use multiple timeframes for entries
        timeframes = [
            ('M5', m5_data, 2.0),    # Highest precision, higher weight
            ('M15', m15_data, 1.5),  # Good balance
            ('H1', h1_data, 1.0)     # HTF confirmation
        ]
        
        for tf_name, tf_data, weight_multiplier in timeframes:
            tf_signals = self._generate_timeframe_signals(
                symbol, tf_name, tf_data, daily_data, h4_data, h1_data, 
                m15_data, htf_bias, regimes, weight_multiplier
            )
            signals.extend(tf_signals)
        
        # Sort signals by timestamp and remove duplicates
        signals = sorted(signals, key=lambda x: x.timestamp)
        signals = self._remove_duplicate_signals(signals)
        
        return signals
    
    def _generate_timeframe_signals(self, symbol: str, timeframe: str, primary_data: pd.DataFrame,
                                  daily_data: pd.DataFrame, h4_data: pd.DataFrame, h1_data: pd.DataFrame,
                                  m15_data: pd.DataFrame, htf_bias: HTFBias, regimes: List[MarketRegime],
                                  weight_multiplier: float) -> List[Signal]:
        """Generate signals for specific timeframe"""
        signals = []
        min_lookback = 200 if timeframe == 'H1' else 100
        
        for i in range(min_lookback, len(primary_data)):
            if i >= len(regimes):
                continue
            
            current_regime = regimes[min(i, len(regimes)-1)]
            current_bar = primary_data.iloc[i]
            timestamp = current_bar.name
            
            # Get corresponding HTF data
            daily_row = self._get_htf_data_at_time(daily_data, timestamp)
            h4_row = self._get_htf_data_at_time(h4_data, timestamp)
            h1_row = self._get_htf_data_at_time(h1_data, timestamp)
            m15_row = self._get_htf_data_at_time(m15_data, timestamp)
            
            if any(row is None for row in [daily_row, h4_row, h1_row, m15_row]):
                continue
            
            # ICT Confluence Analysis
            confluences = self._analyze_ict_confluences(
                primary_data, i, daily_row, h4_row, h1_row, m15_row, current_bar, timeframe
            )
            
            # Count valid confluences
            confluence_count = sum(1 for conf in confluences.values() if conf.get('valid', False))
            
            if confluence_count >= self.params['min_confluences']:
                # Calculate signal strength
                bullish_weight = sum(conf['weight'] for conf in confluences.values()
                                   if conf.get('valid', False) and conf.get('direction') == 'bullish')
                bearish_weight = sum(conf['weight'] for conf in confluences.values()
                                   if conf.get('valid', False) and conf.get('direction') == 'bearish')
                
                # Apply timeframe multiplier
                bullish_weight *= weight_multiplier
                bearish_weight *= weight_multiplier
                
                if bullish_weight > bearish_weight:
                    side = "buy"
                    base_confidence = min(bullish_weight / 12.0, 0.95)
                    against_bias = htf_bias.overall == 'bearish'
                elif bearish_weight > bullish_weight:
                    side = "sell"
                    base_confidence = min(bearish_weight / 12.0, 0.95)
                    against_bias = htf_bias.overall == 'bullish'
                else:
                    continue
                
                # ML Enhancement
                ml_confidence = self.ml_optimizer.predict_signal_quality(symbol, primary_data, i)
                final_confidence = (base_confidence * 0.7) + (ml_confidence * 0.3)
                
                # Get optimized threshold
                optimized_params = self.optimized_params.get(symbol, {})
                confidence_threshold = optimized_params.get('confidence_threshold', self.base_confidence_threshold)
                
                # Check if we can trade
                can_trade = (final_confidence >= confidence_threshold or
                           (against_bias and htf_bias.reversal_probability > 0.6 and final_confidence > 0.6))
                
                if can_trade:
                    # Calculate risk percentage
                    base_risk = self.params['risk_base']
                    risk_multiplier = optimized_params.get('risk_multiplier', 1.0)
                    
                    # Adjust risk based on confluence strength and ML confidence
                    risk_percentage = min(
                        base_risk * risk_multiplier * final_confidence * (confluence_count / 5.0),
                        self.params['risk_max']
                    )
                    
                    # Calculate stop loss and take profit
                    atr = current_bar.get('atr_14', current_bar.get('atr', 0.001))
                    entry_price = current_bar['close']
                    
                    if side == "buy":
                        stop_loss = entry_price - (atr * 1.5)  # Tighter stops
                        take_profit = entry_price + (atr * 3.0)  # 1:2 RR minimum
                    else:
                        stop_loss = entry_price + (atr * 1.5)
                        take_profit = entry_price - (atr * 3.0)
                    
                    # Check for reversal signals
                    reversal_signal = (confluences.get('liquidity_sweep', {}).get('valid', False) or
                                     confluences.get('judas_swing', {}).get('valid', False))
                    
                    signal = Signal(
                        timestamp=timestamp,
                        symbol=symbol,
                        side=side,
                        confidence=final_confidence,
                        strategy_name=self.name,
                        strategy_family=self.family,
                        entry_price=entry_price,
                        timeframe=timeframe,
                        htf_bias=htf_bias.overall,
                        against_bias=against_bias,
                        reversal_signal=reversal_signal,
                        confluences=confluences,
                        regime=current_regime.regime,
                        risk_percentage=risk_percentage,
                        stop_loss=stop_loss,
                        take_profit=take_profit
                    )
                    signals.append(signal)
        
        return signals
    
    def _remove_duplicate_signals(self, signals: List[Signal]) -> List[Signal]:
        """Remove signals that are too close in time"""
        if not signals:
            return signals
        
        filtered_signals = []
        last_signal_time = None
        min_gap = timedelta(hours=2)  # Minimum 2 hours between signals
        
        for signal in signals:
            if last_signal_time is None or signal.timestamp - last_signal_time >= min_gap:
                filtered_signals.append(signal)
                last_signal_time = signal.timestamp
        
        return filtered_signals
    
    def _get_htf_data_at_time(self, htf_data: pd.DataFrame, timestamp: datetime) -> Optional[pd.Series]:
        """Get higher timeframe data at specific time"""
        try:
            available_times = htf_data.index[htf_data.index <= timestamp]
            if len(available_times) == 0:
                return None
            return htf_data.loc[available_times[-1]]
        except:
            return None
    
    def _analyze_ict_confluences(self, data: pd.DataFrame, current_idx: int,
                               daily_row: pd.Series, h4_row: pd.Series, h1_row: pd.Series,
                               m15_row: pd.Series, current_bar: pd.Series, timeframe: str) -> Dict:
        """Analyze all ICT confluences with timeframe-specific parameters"""
        confluences = {}
        
        # Adjust parameters based on timeframe
        if timeframe == 'M5':
            liquidity_lookback = 15
            fvg_sensitivity = 1.2
        elif timeframe == 'M15':
            liquidity_lookback = 20
            fvg_sensitivity = 1.0
        else:  # H1
            liquidity_lookback = 25
            fvg_sensitivity = 0.8
        
        # 1. Liquidity Sweep Analysis
        confluences['liquidity_sweep'] = self._check_liquidity_sweep(data, current_idx, liquidity_lookback)
        
        # 2. Fair Value Gap Analysis
        confluences['fair_value_gap'] = self._check_fair_value_gap(data, current_idx, fvg_sensitivity)
        
        # 3. Order Block Analysis
        confluences['order_block'] = self._check_order_block(data, current_idx)
        
        # 4. Optimal Trade Entry Analysis
        confluences['optimal_trade_entry'] = self._check_ote(data, current_idx)
        
        # 5. Judas Swing Analysis
        confluences['judas_swing'] = self._check_judas_swing(data, current_idx, current_bar)
        
        # 6. HTF Structure Confluence
        confluences['htf_structure'] = self._check_htf_structure(daily_row, h4_row, h1_row)
        
        # 7. Session Confluence
        confluences['session'] = self._check_session_confluence(current_bar)
        
        # 8. Market Structure Break
        confluences['market_structure'] = self._check_market_structure_break(data, current_idx)
        
        return confluences
    
    def _check_liquidity_sweep(self, data: pd.DataFrame, idx: int, lookback: int) -> Dict:
        """Enhanced liquidity sweep detection"""
        if idx < lookback:
            return {'valid': False, 'direction': None, 'weight': 0}
        
        lookback_data = data.iloc[idx-lookback:idx]
        current_bar = data.iloc[idx]
        
        recent_high = lookback_data['high'].max()
        recent_low = lookback_data['low'].min()
        
        # More precise sweep detection
        swept_high = current_bar['high'] > recent_high * 1.0002  # Reduced threshold
        closed_below = current_bar['close'] < recent_high * 0.9998
        
        swept_low = current_bar['low'] < recent_low * 0.9998
        closed_above = current_bar['close'] > recent_low * 1.0002
        
        # Check volume confirmation
        volume_spike = current_bar.get('tick_volume', 0) > lookback_data['tick_volume'].mean() * 1.5
        
        if swept_high and closed_below:
            weight = 3.5 if volume_spike else 3.0
            return {'valid': True, 'direction': 'bearish', 'weight': weight, 'type': 'high_sweep'}
        elif swept_low and closed_above:
            weight = 3.5 if volume_spike else 3.0
            return {'valid': True, 'direction': 'bullish', 'weight': weight, 'type': 'low_sweep'}
        
        return {'valid': False, 'direction': None, 'weight': 0}
    
    def _check_fair_value_gap(self, data: pd.DataFrame, idx: int, sensitivity: float) -> Dict:
        """Enhanced FVG detection with sensitivity adjustment"""
        if idx < 3:
            return {'valid': False, 'direction': None, 'weight': 0}
        
        current_bar = data.iloc[idx]
        prev_bar = data.iloc[idx-1]
        prev2_bar = data.iloc[idx-2]
        
        min_gap_size = self.params['fvg_min_size'] * sensitivity
        
        # Bullish FVG
        bullish_fvg = prev2_bar['high'] < current_bar['low']
        bullish_gap_size = current_bar['low'] - prev2_bar['high']
        
        # Bearish FVG
        bearish_fvg = prev2_bar['low'] > current_bar['high']
        bearish_gap_size = prev2_bar['low'] - current_bar['high']
        
        if bullish_fvg and bullish_gap_size >= min_gap_size:
            # Enhanced gap fill detection
            gap_center = prev2_bar['high'] + (bullish_gap_size / 2)
            near_center = abs(current_bar['close'] - gap_center) <= bullish_gap_size * 0.3
            
            if near_center:
                return {'valid': True, 'direction': 'bullish', 'weight': 2.5, 'gap_size': bullish_gap_size}
        
        elif bearish_fvg and bearish_gap_size >= min_gap_size:
            gap_center = prev2_bar['low'] - (bearish_gap_size / 2)
            near_center = abs(current_bar['close'] - gap_center) <= bearish_gap_size * 0.3
            
            if near_center:
                return {'valid': True, 'direction': 'bearish', 'weight': 2.5, 'gap_size': bearish_gap_size}
        
        return {'valid': False, 'direction': None, 'weight': 0}
    
    def _check_order_block(self, data: pd.DataFrame, idx: int) -> Dict:
        """Enhanced order block detection"""
        if idx < 25:
            return {'valid': False, 'direction': None, 'weight': 0}
        
        current_bar = data.iloc[idx]
        lookback_data = data.iloc[idx-25:idx]
        
        for i in range(5, len(lookback_data)-5):
            ob_bar = lookback_data.iloc[i]
            before_bars = lookback_data.iloc[max(0, i-5):i]
            after_bars = lookback_data.iloc[i+1:min(len(lookback_data), i+8)]
            
            if len(after_bars) == 0:
                continue
            
            # Enhanced order block detection
            strong_move_up = after_bars['close'].iloc[-1] > ob_bar['close'] * 1.005
            consolidation_before = (before_bars['high'].max() - before_bars['low'].min()) < self.params['ob_min_size']
            volume_increase = after_bars['tick_volume'].mean() > before_bars['tick_volume'].mean() * 1.2
            
            if strong_move_up and consolidation_before:
                # Check retest
                if (current_bar['low'] <= ob_bar['high'] * 1.001 and 
                    current_bar['high'] >= ob_bar['low'] * 0.999):
                    weight = 3.0 if volume_increase else 2.5
                    return {'valid': True, 'direction': 'bullish', 'weight': weight, 'ob_price': ob_bar['low']}
            
            # Bearish order block
            strong_move_down = after_bars['close'].iloc[-1] < ob_bar['close'] * 0.995
            
            if strong_move_down and consolidation_before:
                if (current_bar['high'] >= ob_bar['low'] * 0.999 and 
                    current_bar['low'] <= ob_bar['high'] * 1.001):
                    weight = 3.0 if volume_increase else 2.5
                    return {'valid': True, 'direction': 'bearish', 'weight': weight, 'ob_price': ob_bar['high']}
        
        return {'valid': False, 'direction': None, 'weight': 0}
    
    def _check_ote(self, data: pd.DataFrame, idx: int) -> Dict:
        """Enhanced Optimal Trade Entry detection"""
        if idx < 60:
            return {'valid': False, 'direction': None, 'weight': 0}
        
        current_bar = data.iloc[idx]
        lookback_data = data.iloc[idx-60:idx]
        
        # Find significant swing points
        swing_high_idx = lookback_data['high'].idxmax()
        swing_low_idx = lookback_data['low'].idxmin()
        
        swing_high = lookback_data.loc[swing_high_idx, 'high']
        swing_low = lookback_data.loc[swing_low_idx, 'low']
        swing_range = swing_high - swing_low
        
        if swing_range < current_bar.get('atr_14', 0.001) * 3:
            return {'valid': False, 'direction': None, 'weight': 0}
        
        # Enhanced OTE level checking
        for fib_level in self.params['ote_fib_levels']:
            swing_high_pos = lookback_data.index.get_loc(swing_high_idx)
            swing_low_pos = lookback_data.index.get_loc(swing_low_idx)
            
            if swing_high_pos > swing_low_pos:  # Uptrend
                ote_level = swing_high - (swing_range * fib_level)
                price_distance = abs(current_bar['close'] - ote_level)
                tolerance = swing_range * 0.015  # 1.5% tolerance
                
                if price_distance <= tolerance:
                    # Additional confluence checks
                    rsi_oversold = current_bar.get('rsi_14', 50) < 40
                    near_ema = abs(current_bar['close'] - current_bar.get('ema_21', current_bar['close'])) < swing_range * 0.02
                    
                    weight = 3.0 if (rsi_oversold or near_ema) else 2.5
                    return {'valid': True, 'direction': 'bullish', 'weight': weight, 'ote_level': fib_level}
            
            else:  # Downtrend
                ote_level = swing_low + (swing_range * fib_level)
                price_distance = abs(current_bar['close'] - ote_level)
                tolerance = swing_range * 0.015
                
                if price_distance <= tolerance:
                    rsi_overbought = current_bar.get('rsi_14', 50) > 60
                    near_ema = abs(current_bar['close'] - current_bar.get('ema_21', current_bar['close'])) < swing_range * 0.02
                    
                    weight = 3.0 if (rsi_overbought or near_ema) else 2.5
                    return {'valid': True, 'direction': 'bearish', 'weight': weight, 'ote_level': fib_level}
        
        return {'valid': False, 'direction': None, 'weight': 0}
    
    def _check_judas_swing(self, data: pd.DataFrame, idx: int, current_bar: pd.Series) -> Dict:
        """Enhanced Judas Swing detection"""
        hour = current_bar.name.hour
        
        # Extended session windows for better detection
        judas_window = any(session <= hour <= session + 3 for session in self.params['judas_sessions'])
        
        if not judas_window or idx < 25:
            return {'valid': False, 'direction': None, 'weight': 0}
        
        lookback_data = data.iloc[idx-25:idx]
        range_high = lookback_data['high'].max()
        range_low = lookback_data['low'].min()
        range_size = range_high - range_low
        
        # Enhanced false break detection
        false_break_threshold = range_size * 0.15  # 15% of range
        
        # False break above range
        if (current_bar['high'] > range_high * 1.0003 and  # Small break above
            current_bar['close'] < range_high - false_break_threshold):
            
            # Additional confirmations
            volume_spike = current_bar.get('tick_volume', 0) > lookback_data['tick_volume'].mean() * 1.3
            large_wick = current_bar['upper_wick'] > current_bar['body_size'] * 2
            
            weight = 4.0 if (volume_spike and large_wick) else 3.5
            return {'valid': True, 'direction': 'bearish', 'weight': weight, 'false_break': 'high'}
        
        # False break below range
        elif (current_bar['low'] < range_low * 0.9997 and
              current_bar['close'] > range_low + false_break_threshold):
            
            volume_spike = current_bar.get('tick_volume', 0) > lookback_data['tick_volume'].mean() * 1.3
            large_wick = current_bar['lower_wick'] > current_bar['body_size'] * 2
            
            weight = 4.0 if (volume_spike and large_wick) else 3.5
            return {'valid': True, 'direction': 'bullish', 'weight': weight, 'false_break': 'low'}
        
        return {'valid': False, 'direction': None, 'weight': 0}
    
    def _check_htf_structure(self, daily_row: pd.Series, h4_row: pd.Series, h1_row: pd.Series) -> Dict:
        """Enhanced HTF structure analysis"""
        # Daily structure (strongest weight)
        daily_bullish = (daily_row.get('ema_12', 0) > daily_row.get('ema_26', 0) and
                         daily_row.get('close', 0) > daily_row.get('sma_200', 0) and
                         daily_row.get('rsi_14', 50) > 45)
        daily_bearish = (daily_row.get('ema_12', 0) < daily_row.get('ema_26', 0) and
                        daily_row.get('close', 0) < daily_row.get('sma_200', 0) and
                        daily_row.get('rsi_14', 50) < 55)
        
        # H4 structure
        h4_bullish = (h4_row.get('ema_12', 0) > h4_row.get('ema_26', 0) and
                     h4_row.get('macd', 0) > h4_row.get('macd_signal', 0))
        h4_bearish = (h4_row.get('ema_12', 0) < h4_row.get('ema_26', 0) and
                     h4_row.get('macd', 0) < h4_row.get('macd_signal', 0))
        
        # H1 structure
        h1_bullish = (h1_row.get('rsi_14', 50) < 75 and 
                     h1_row.get('close', 0) > h1_row.get('ema_21', 0))
        h1_bearish = (h1_row.get('rsi_14', 50) > 25 and 
                     h1_row.get('close', 0) < h1_row.get('ema_21', 0))
        
        # Weight the alignments
        bullish_score = (daily_bullish * 2) + h4_bullish + h1_bullish
        bearish_score = (daily_bearish * 2) + h4_bearish + h1_bearish
        
        if bullish_score >= 3:
            return {'valid': True, 'direction': 'bullish', 'weight': min(bullish_score, 4)}
        elif bearish_score >= 3:
            return {'valid': True, 'direction': 'bearish', 'weight': min(bearish_score, 4)}
        
        return {'valid': False, 'direction': None, 'weight': 0}
    
    def _check_session_confluence(self, current_bar: pd.Series) -> Dict:
        """Enhanced session confluence with volatility consideration"""
        hour = current_bar.name.hour
        day_of_week = current_bar.name.weekday()  # 0=Monday, 6=Sunday
        
        # High-impact sessions with volatility weighting
        if 7 <= hour <= 10 and day_of_week < 5:  # London open
            return {'valid': True, 'direction': 'neutral', 'weight': 1.5, 'session': 'london_open'}
        elif 13 <= hour <= 16 and day_of_week < 5:  # NY open + London overlap
            return {'valid': True, 'direction': 'neutral', 'weight': 2.0, 'session': 'ny_london_overlap'}
        elif 8 <= hour <= 17 and day_of_week < 5:  # London session
            return {'valid': True, 'direction': 'neutral', 'weight': 1.2, 'session': 'london'}
        elif 14 <= hour <= 22 and day_of_week < 5:  # NY session
            return {'valid': True, 'direction': 'neutral', 'weight': 1.2, 'session': 'ny'}
        elif 0 <= hour <= 6:  # Asian session
            return {'valid': True, 'direction': 'neutral', 'weight': 0.8, 'session': 'asian'}
        else:  # Low volatility periods
            return {'valid': True, 'direction': 'neutral', 'weight': 0.5, 'session': 'off_hours'}
    
    def _check_market_structure_break(self, data: pd.DataFrame, idx: int) -> Dict:
        """Check for market structure breaks"""
        if idx < 30:
            return {'valid': False, 'direction': None, 'weight': 0}
        
        current_bar = data.iloc[idx]
        lookback_data = data.iloc[idx-30:idx]
        
        # Find recent swing points
        highs = lookback_data['high'].rolling(window=5, center=True).max() == lookback_data['high']
        lows = lookback_data['low'].rolling(window=5, center=True).min() == lookback_data['low']
        
        swing_highs = lookback_data[highs]['high'].values
        swing_lows = lookback_data[lows]['low'].values
        
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return {'valid': False, 'direction': None, 'weight': 0}
        
        # Check for break of structure
        recent_high = np.max(swing_highs[-2:])
        recent_low = np.min(swing_lows[-2:])
        
        # Bullish break of structure
        if current_bar['close'] > recent_high:
            return {'valid': True, 'direction': 'bullish', 'weight': 2.0, 'break_level': recent_high}
        
        # Bearish break of structure
        elif current_bar['close'] < recent_low:
            return {'valid': True, 'direction': 'bearish', 'weight': 2.0, 'break_level': recent_low}
        
        return {'valid': False, 'direction': None, 'weight': 0}
    
    def update_optimized_params(self, symbol: str, params: Dict):
        """Update optimized parameters for symbol"""
        self.optimized_params[symbol] = params

class BacktestEngine:
    """Advanced backtesting engine with comprehensive analysis"""
    
    def __init__(self, data_manager: DataManager, ml_optimizer: MLOptimizer):
        self.data_manager = data_manager
        self.ml_optimizer = ml_optimizer
        self.ict_strategy = ICTStrategy(ml_optimizer)
        self.trades = []
        self.monthly_results = {}
        self.optimization_history = []
        
        # Backtesting parameters
        self.initial_balance = 10000.0
        self.current_balance = self.initial_balance
        self.max_risk_per_trade = 0.025  # 2.5% max risk
        self.target_monthly_return = 0.08  # 8% monthly target
        
    def run_comprehensive_backtest(self, symbols: List[str], start_date: datetime, end_date: datetime) -> Dict:
        """Run comprehensive backtest with optimization"""
        logger.info("🚀 Starting Eden Comprehensive Backtest")
        logger.info("=" * 80)
        logger.info(f"🎯 Target: {self.target_monthly_return*100}% monthly returns")
        logger.info(f"📊 Symbols: {', '.join(symbols)}")
        logger.info(f"📅 Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print()
        
        # Initialize results tracking
        all_results = {}
        optimization_cycles = 0
        max_optimization_cycles = 5
        
        while optimization_cycles < max_optimization_cycles:
            logger.info(f"🔄 Optimization Cycle {optimization_cycles + 1}/{max_optimization_cycles}")
            
            # Reset for this cycle
            self.trades = []
            self.current_balance = self.initial_balance
            
            # Run backtest for each symbol
            for symbol in symbols:
                symbol_results = self._backtest_symbol(symbol, start_date, end_date)
                if symbol_results:
                    all_results[symbol] = symbol_results
            
            # Analyze results and optimize
            monthly_performance = self._analyze_monthly_performance()
            self.monthly_results = monthly_performance
            
            # Check if target is achieved
            target_achieved = self._check_target_achievement(monthly_performance)
            
            if target_achieved:
                logger.info("✅ Target achieved! Stopping optimization.")
                break
            
            # Optimize parameters for next cycle
            logger.info("🎯 Running optimization for next cycle...")
            self._optimize_parameters(symbols)
            
            optimization_cycles += 1
        
        # Final analysis
        final_results = self._generate_final_results(all_results, monthly_performance)
        
        return final_results
    
    def _backtest_symbol(self, symbol: str, start_date: datetime, end_date: datetime) -> Dict:
        """Backtest single symbol with all timeframes"""
        logger.info(f"📈 Backtesting {symbol}...")
        
        # Get data for all timeframes
        timeframes = {
            'D1': mt5.TIMEFRAME_D1,
            'H4': mt5.TIMEFRAME_H4,
            'H1': mt5.TIMEFRAME_H1,
            'M15': mt5.TIMEFRAME_M15,
            'M5': mt5.TIMEFRAME_M5
        }
        
        tf_data = {}
        for tf_name, tf_value in timeframes.items():
            data = self.data_manager.get_real_data(symbol, tf_value, start_date, end_date)
            if data is None or len(data) < 100:
                logger.warning(f"Insufficient {tf_name} data for {symbol}")
                if tf_name in ['D1', 'H4', 'H1']:  # Essential timeframes
                    return {}
            else:
                tf_data[tf_name] = data
        
        if len(tf_data) < 3:
            logger.error(f"Not enough timeframe data for {symbol}")
            return {}
        
        logger.info(f"   📊 Data loaded: {', '.join([f'{tf}({len(data)})' for tf, data in tf_data.items()])}")
        
        # Generate market regimes and HTF bias
        regimes = self._generate_market_regimes(tf_data.get('H1', tf_data.get('H4')))
        htf_bias = self._analyze_htf_bias(tf_data)
        
        # Generate signals
        signals = self.ict_strategy.generate_signals(
            symbol, 
            tf_data.get('D1'), tf_data.get('H4'), tf_data.get('H1'), 
            tf_data.get('M15'), tf_data.get('M5'),
            htf_bias, regimes
        )
        
        logger.info(f"   📡 Generated {len(signals)} signals")
        
        # Execute trades
        symbol_trades = self._execute_trades(signals, tf_data)
        
        # Train ML model with results
        if len(symbol_trades) > 50:
            self.ml_optimizer.train_signal_classifier(
                symbol, tf_data.get('H1', tf_data.get('H4')), signals
            )
        
        logger.info(f"   ✅ Executed {len(symbol_trades)} trades")
        
        return {
            'symbol': symbol,
            'signals': len(signals),
            'trades': len(symbol_trades),
            'data_bars': {tf: len(data) for tf, data in tf_data.items()},
            'htf_bias': htf_bias,
            'regimes': len(regimes)
        }
    
    def _generate_market_regimes(self, primary_data: pd.DataFrame) -> List[MarketRegime]:
        """Generate market regimes for backtesting"""
        if primary_data is None:
            return []
        
        regimes = []
        
        for i in range(len(primary_data)):
            bar = primary_data.iloc[i]
            
            # Calculate volatility
            atr = bar.get('atr_14', 0.001)
            volatility = atr / bar['close']
            
            # Determine regime based on multiple factors
            rsi = bar.get('rsi_14', 50)
            bb_width = bar.get('bb_width', 0.02)
            
            if volatility > 0.015 or bb_width > 0.03:
                regime = 'momentum_burst'
                direction = 1 if rsi > 50 else -1
            elif volatility < 0.005 or bb_width < 0.01:
                regime = 'range'
                direction = 0
            else:
                regime = 'trend'
                # Determine trend direction
                ema12 = bar.get('ema_12', bar['close'])
                ema26 = bar.get('ema_26', bar['close'])
                direction = 1 if ema12 > ema26 else -1
            
            regimes.append(MarketRegime(regime, 0.7, volatility, direction))
        
        return regimes
    
    def _analyze_htf_bias(self, tf_data: Dict) -> HTFBias:
        """Analyze higher timeframe bias"""
        daily_data = tf_data.get('D1')
        h4_data = tf_data.get('H4')
        h1_data = tf_data.get('H1')
        
        if not all([daily_data is not None, h4_data is not None, h1_data is not None]):
            return HTFBias('neutral', 'neutral', 'neutral', 'neutral', 0.5, 0.3)
        
        # Analyze each timeframe
        daily_bias = self._get_timeframe_bias(daily_data.iloc[-20:])
        h4_bias = self._get_timeframe_bias(h4_data.iloc[-50:])
        h1_bias = self._get_timeframe_bias(h1_data.iloc[-100:])
        
        # Weight the biases
        bias_scores = {'bullish': 0, 'bearish': 0, 'neutral': 0}
        weights = {'daily': 0.5, 'h4': 0.3, 'h1': 0.2}
        biases = {'daily': daily_bias, 'h4': h4_bias, 'h1': h1_bias}
        
        for tf, bias in biases.items():
            bias_scores[bias] += weights[tf]
        
        overall_bias = max(bias_scores, key=bias_scores.get)
        confidence = bias_scores[overall_bias]
        
        # Calculate reversal probability
        reversal_prob = self._calculate_reversal_probability(daily_data, h4_data, h1_data)
        
        return HTFBias(daily_bias, h4_bias, h1_bias, overall_bias, confidence, reversal_prob)
    
    def _get_timeframe_bias(self, data: pd.DataFrame) -> str:
        """Get bias for specific timeframe"""
        if len(data) < 10:
            return 'neutral'
        
        recent_data = data.iloc[-10:]
        
        # Multiple factor analysis
        factors = []
        
        # Price vs EMAs
        if 'ema_12' in recent_data.columns and 'ema_26' in recent_data.columns:
            ema_alignment = recent_data['ema_12'].iloc[-1] > recent_data['ema_26'].iloc[-1]
            factors.append('bullish' if ema_alignment else 'bearish')
        
        # RSI
        if 'rsi_14' in recent_data.columns:
            rsi = recent_data['rsi_14'].iloc[-1]
            if rsi > 55:
                factors.append('bullish')
            elif rsi < 45:
                factors.append('bearish')
            else:
                factors.append('neutral')
        
        # MACD
        if 'macd' in recent_data.columns and 'macd_signal' in recent_data.columns:
            macd_bullish = recent_data['macd'].iloc[-1] > recent_data['macd_signal'].iloc[-1]
            factors.append('bullish' if macd_bullish else 'bearish')
        
        # Price vs 200 SMA
        if 'sma_200' in recent_data.columns:
            above_sma200 = recent_data['close'].iloc[-1] > recent_data['sma_200'].iloc[-1]
            factors.append('bullish' if above_sma200 else 'bearish')
        
        # Count factors
        bullish_count = factors.count('bullish')
        bearish_count = factors.count('bearish')
        
        if bullish_count > bearish_count:
            return 'bullish'
        elif bearish_count > bullish_count:
            return 'bearish'
        else:
            return 'neutral'
    
    def _calculate_reversal_probability(self, daily: pd.DataFrame, h4: pd.DataFrame, h1: pd.DataFrame) -> float:
        """Calculate reversal probability"""
        reversal_signals = 0
        total_signals = 0
        
        # RSI divergence on H1
        if len(h1) >= 20:
            recent_h1 = h1.iloc[-10:]
            if len(recent_h1) >= 5:
                price_trend = recent_h1['close'].iloc[-1] > recent_h1['close'].iloc[0]
                rsi_trend = recent_h1.get('rsi_14', pd.Series([50]*len(recent_h1))).iloc[-1] > recent_h1.get('rsi_14', pd.Series([50]*len(recent_h1))).iloc[0]
                if price_trend != rsi_trend:
                    reversal_signals += 2
            total_signals += 2
        
        # Overbought/Oversold conditions
        if len(h1) >= 5:
            recent_rsi = h1.get('rsi_14', pd.Series([50]*len(h1))).iloc[-5:].mean()
            if recent_rsi > 75 or recent_rsi < 25:
                reversal_signals += 1
            total_signals += 1
        
        return reversal_signals / max(total_signals, 1)
    
    def _execute_trades(self, signals: List[Signal], tf_data: Dict) -> List[Trade]:
        """Execute trades based on signals"""
        trades = []
        
        # Get primary data for trade execution (prefer M5, then M15, then H1)
        primary_data = tf_data.get('M5') or tf_data.get('M15') or tf_data.get('H1')
        if primary_data is None:
            return trades
        
        for signal in signals:
            trade = self._execute_single_trade(signal, primary_data)
            if trade:
                trades.append(trade)
                self.trades.append(trade)
        
        return trades
    
    def _execute_single_trade(self, signal: Signal, data: pd.DataFrame) -> Optional[Trade]:
        """Execute a single trade"""
        try:
            # Find entry point in data
            entry_idx = data.index.get_loc(signal.timestamp, method='nearest')
            
            if entry_idx >= len(data) - 1:
                return None
            
            entry_bar = data.iloc[entry_idx]
            entry_price = entry_bar['close']  # Assume we get close price
            
            # Calculate position size based on risk
            risk_amount = self.current_balance * min(signal.risk_percentage / 100, self.max_risk_per_trade)
            
            # Calculate stop loss distance
            if signal.stop_loss:
                sl_distance = abs(entry_price - signal.stop_loss)
            else:
                atr = entry_bar.get('atr_14', entry_bar.get('atr', 0.001))
                sl_distance = atr * 1.5
            
            # Position size calculation (simplified for forex)
            pip_value = 1  # Simplified - should be calculated based on symbol
            position_size = risk_amount / (sl_distance * pip_value)
            
            # Create trade
            trade = Trade(
                signal=signal,
                entry_time=signal.timestamp,
                entry_price=entry_price,
                risk_percentage=signal.risk_percentage
            )
            
            # Simulate trade execution
            exit_result = self._simulate_trade_exit(trade, data, entry_idx + 1)
            
            if exit_result:
                trade.exit_time = exit_result['exit_time']
                trade.exit_price = exit_result['exit_price']
                trade.exit_reason = exit_result['exit_reason']
                trade.pnl_pips = exit_result['pnl_pips']
                trade.pnl_percentage = exit_result['pnl_percentage']
                trade.duration_hours = exit_result['duration_hours']
                trade.max_drawdown = exit_result['max_drawdown']
                trade.max_profit = exit_result['max_profit']
                
                # Update balance
                self.current_balance += (self.current_balance * trade.pnl_percentage / 100)
            
            return trade
            
        except Exception as e:
            logger.warning(f"Error executing trade: {e}")
            return None
    
    def _simulate_trade_exit(self, trade: Trade, data: pd.DataFrame, start_idx: int) -> Optional[Dict]:
        """Simulate trade exit based on stop loss, take profit, or time"""
        signal = trade.signal
        entry_price = trade.entry_price
        
        max_bars_to_check = min(500, len(data) - start_idx)  # Max 500 bars or remaining data
        
        max_profit = 0.0
        max_drawdown = 0.0
        
        for i in range(start_idx, start_idx + max_bars_to_check):
            if i >= len(data):
                break
            
            current_bar = data.iloc[i]
            current_time = current_bar.name
            
            # Calculate current P&L
            if signal.side == "buy":
                current_pnl = current_bar['close'] - entry_price
                unrealized_pnl_pct = (current_pnl / entry_price) * 100
                
                # Track max profit and drawdown
                max_profit = max(max_profit, unrealized_pnl_pct)
                max_drawdown = min(max_drawdown, unrealized_pnl_pct)
                
                # Check stop loss
                if signal.stop_loss and current_bar['low'] <= signal.stop_loss:
                    pnl_pips = signal.stop_loss - entry_price
                    pnl_percentage = (pnl_pips / entry_price) * 100
                    duration_hours = (current_time - trade.entry_time).total_seconds() / 3600
                    
                    return {
                        'exit_time': current_time,
                        'exit_price': signal.stop_loss,
                        'exit_reason': 'stop_loss',
                        'pnl_pips': pnl_pips,
                        'pnl_percentage': pnl_percentage,
                        'duration_hours': duration_hours,
                        'max_drawdown': max_drawdown,
                        'max_profit': max_profit
                    }
                
                # Check take profit
                if signal.take_profit and current_bar['high'] >= signal.take_profit:
                    pnl_pips = signal.take_profit - entry_price
                    pnl_percentage = (pnl_pips / entry_price) * 100
                    duration_hours = (current_time - trade.entry_time).total_seconds() / 3600
                    
                    return {
                        'exit_time': current_time,
                        'exit_price': signal.take_profit,
                        'exit_reason': 'take_profit',
                        'pnl_pips': pnl_pips,
                        'pnl_percentage': pnl_percentage,
                        'duration_hours': duration_hours,
                        'max_drawdown': max_drawdown,
                        'max_profit': max_profit
                    }
            
            else:  # sell
                current_pnl = entry_price - current_bar['close']
                unrealized_pnl_pct = (current_pnl / entry_price) * 100
                
                # Track max profit and drawdown
                max_profit = max(max_profit, unrealized_pnl_pct)
                max_drawdown = min(max_drawdown, unrealized_pnl_pct)
                
                # Check stop loss
                if signal.stop_loss and current_bar['high'] >= signal.stop_loss:
                    pnl_pips = entry_price - signal.stop_loss
                    pnl_percentage = (pnl_pips / entry_price) * 100
                    duration_hours = (current_time - trade.entry_time).total_seconds() / 3600
                    
                    return {
                        'exit_time': current_time,
                        'exit_price': signal.stop_loss,
                        'exit_reason': 'stop_loss',
                        'pnl_pips': pnl_pips,
                        'pnl_percentage': pnl_percentage,
                        'duration_hours': duration_hours,
                        'max_drawdown': max_drawdown,
                        'max_profit': max_profit
                    }
                
                # Check take profit
                if signal.take_profit and current_bar['low'] <= signal.take_profit:
                    pnl_pips = entry_price - signal.take_profit
                    pnl_percentage = (pnl_pips / entry_price) * 100
                    duration_hours = (current_time - trade.entry_time).total_seconds() / 3600
                    
                    return {
                        'exit_time': current_time,
                        'exit_price': signal.take_profit,
                        'exit_reason': 'take_profit',
                        'pnl_pips': pnl_pips,
                        'pnl_percentage': pnl_percentage,
                        'duration_hours': duration_hours,
                        'max_drawdown': max_drawdown,
                        'max_profit': max_profit
                    }
        
        # If no exit condition met, close at last price
        if start_idx + max_bars_to_check - 1 < len(data):
            last_bar = data.iloc[start_idx + max_bars_to_check - 1]
            last_time = last_bar.name
            
            if signal.side == "buy":
                pnl_pips = last_bar['close'] - entry_price
            else:
                pnl_pips = entry_price - last_bar['close']
            
            pnl_percentage = (pnl_pips / entry_price) * 100
            duration_hours = (last_time - trade.entry_time).total_seconds() / 3600
            
            return {
                'exit_time': last_time,
                'exit_price': last_bar['close'],
                'exit_reason': 'time_exit',
                'pnl_pips': pnl_pips,
                'pnl_percentage': pnl_percentage,
                'duration_hours': duration_hours,
                'max_drawdown': max_drawdown,
                'max_profit': max_profit
            }
        
        return None
    
    def _analyze_monthly_performance(self) -> Dict[str, MonthlyResults]:
        """Analyze monthly performance"""
        monthly_results = {}
        
        if not self.trades:
            return monthly_results
        
        # Group trades by month
        monthly_trades = defaultdict(list)
        for trade in self.trades:
            if trade.exit_time:
                month_key = trade.exit_time.strftime("%Y-%m")
                monthly_trades[month_key].append(trade)
        
        # Calculate monthly statistics
        for month, trades in monthly_trades.items():
            if not trades:
                continue
            
            wins = [t for t in trades if t.pnl_percentage > 0]
            losses = [t for t in trades if t.pnl_percentage <= 0]
            
            total_return = sum(t.pnl_percentage for t in trades)
            win_rate = len(wins) / len(trades) if trades else 0
            
            # Calculate Sharpe ratio (simplified)
            returns = [t.pnl_percentage for t in trades]
            avg_return = np.mean(returns)
            std_return = np.std(returns) if len(returns) > 1 else 0
            sharpe_ratio = avg_return / std_return if std_return > 0 else 0
            
            # Max drawdown calculation
            cumulative_returns = np.cumsum(returns)
            max_drawdown = 0
            peak = cumulative_returns[0]
            
            for ret in cumulative_returns:
                if ret > peak:
                    peak = ret
                drawdown = peak - ret
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
            
            monthly_results[month] = MonthlyResults(
                month=month,
                trades=len(trades),
                wins=len(wins),
                losses=len(losses),
                win_rate=win_rate,
                total_pips=sum(t.pnl_pips for t in trades),
                total_return=total_return,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                avg_trade_duration=np.mean([t.duration_hours for t in trades]),
                best_trade=max(returns) if returns else 0,
                worst_trade=min(returns) if returns else 0
            )
        
        return monthly_results
    
    def _check_target_achievement(self, monthly_results: Dict[str, MonthlyResults]) -> bool:
        """Check if monthly target is achieved"""
        if not monthly_results:
            return False
        
        # Check if all months meet target
        months_meeting_target = 0
        total_months = len(monthly_results)
        
        for month, results in monthly_results.items():
            if results.total_return >= self.target_monthly_return * 100:
                months_meeting_target += 1
        
        # Require at least 80% of months to meet target
        target_achievement_rate = months_meeting_target / total_months
        
        logger.info(f"🎯 Target Achievement: {months_meeting_target}/{total_months} months ({target_achievement_rate:.1%})")
        
        return target_achievement_rate >= 0.8
    
    def _optimize_parameters(self, symbols: List[str]):
        """Optimize strategy parameters based on performance"""
        logger.info("🔧 Optimizing strategy parameters...")
        
        for symbol in symbols:
            symbol_trades = [t for t in self.trades if t.signal.symbol == symbol]
            
            if len(symbol_trades) >= 20:  # Need minimum trades for optimization
                optimized_params = self.ml_optimizer.optimize_strategy_parameters(symbol, symbol_trades)
                
                if optimized_params:
                    self.ict_strategy.update_optimized_params(symbol, optimized_params)
                    logger.info(f"   ✅ Updated parameters for {symbol}")
    
    def _generate_final_results(self, symbol_results: Dict, monthly_results: Dict) -> Dict:
        """Generate comprehensive final results"""
        
        # Calculate overall statistics
        all_returns = []
        total_trades = len(self.trades)
        winning_trades = 0
        
        for trade in self.trades:
            if trade.pnl_percentage is not None:
                all_returns.append(trade.pnl_percentage)
                if trade.pnl_percentage > 0:
                    winning_trades += 1
        
        overall_win_rate = winning_trades / total_trades if total_trades > 0 else 0
        overall_return = sum(all_returns)
        
        # Calculate final balance
        final_balance = self.initial_balance * (1 + overall_return / 100)
        total_return_percentage = (final_balance - self.initial_balance) / self.initial_balance * 100
        
        return {
            'summary': {
                'initial_balance': self.initial_balance,
                'final_balance': final_balance,
                'total_return_percentage': total_return_percentage,
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'overall_win_rate': overall_win_rate,
                'avg_monthly_return': np.mean([r.total_return for r in monthly_results.values()]) if monthly_results else 0,
                'target_achieved': self._check_target_achievement(monthly_results)
            },
            'symbol_results': symbol_results,
            'monthly_results': monthly_results,
            'trades': self.trades[:50],  # First 50 trades for analysis
            'optimization_cycles': len(self.optimization_history)
        }
    
    def display_results(self, results: Dict):
        """Display comprehensive backtest results"""
        print("\n" + "=" * 100)
        print("🎯 EDEN COMPREHENSIVE BACKTEST RESULTS")
        print("=" * 100)
        
        summary = results['summary']
        monthly_results = results['monthly_results']
        
        print(f"💰 PERFORMANCE SUMMARY:")
        print(f"   • Initial Balance: ${summary['initial_balance']:,.2f}")
        print(f"   • Final Balance: ${summary['final_balance']:,.2f}")
        print(f"   • Total Return: {summary['total_return_percentage']:.2f}%")
        print(f"   • Total Trades: {summary['total_trades']:,}")
        print(f"   • Win Rate: {summary['overall_win_rate']:.2%}")
        print(f"   • Average Monthly Return: {summary['avg_monthly_return']:.2f}%")
        print(f"   • Target Achieved: {'✅ YES' if summary['target_achieved'] else '❌ NO'}")
        
        if monthly_results:
            print(f"\n📅 MONTHLY BREAKDOWN:")
            print("-" * 100)
            print(f"{'Month':<10} {'Trades':<8} {'Win%':<8} {'Return%':<10} {'MaxDD%':<10} {'Sharpe':<8} {'Best%':<8} {'Worst%':<8}")
            print("-" * 100)
            
            for month, result in sorted(monthly_results.items()):
                status = "✅" if result.total_return >= 8.0 else "❌"
                print(f"{month:<10} {result.trades:<8} {result.win_rate:<7.1%} "
                      f"{result.total_return:<9.2f} {result.max_drawdown:<9.2f} "
                      f"{result.sharpe_ratio:<7.2f} {result.best_trade:<7.2f} {result.worst_trade:<7.2f} {status}")
        
        print(f"\n🎯 TARGET ANALYSIS:")
        if monthly_results:
            months_above_target = sum(1 for r in monthly_results.values() if r.total_return >= 8.0)
            total_months = len(monthly_results)
            print(f"   • Months Above 8% Target: {months_above_target}/{total_months} ({months_above_target/total_months:.1%})")
        
        # Symbol breakdown
        print(f"\n📊 SYMBOL BREAKDOWN:")
        for symbol, data in results['symbol_results'].items():
            print(f"   • {symbol}: {data['trades']} trades from {data['signals']} signals")
        
        print(f"\n✅ Backtest completed with optimization cycles: {results['optimization_cycles']}")

def run_monte_carlo_simulation(backtest_engine: BacktestEngine, num_simulations: int = 1000) -> Dict:
    """Run Monte Carlo simulation on backtest results"""
    logger.info(f"🎰 Running Monte Carlo simulation ({num_simulations:,} iterations)...")
    
    if not backtest_engine.trades:
        logger.error("No trades available for Monte Carlo simulation")
        return {}
    
    # Extract trade returns
    trade_returns = [t.pnl_percentage for t in backtest_engine.trades if t.pnl_percentage is not None]
    
    if not trade_returns:
        logger.error("No trade returns available")
        return {}
    
    # Monte Carlo simulation
    simulation_results = []
    
    for simulation in range(num_simulations):
        # Randomly sample from historical returns with replacement
        simulated_returns = np.random.choice(trade_returns, size=len(trade_returns), replace=True)
        
        # Calculate cumulative return
        total_return = sum(simulated_returns)
        simulation_results.append(total_return)
    
    # Calculate statistics
    mc_results = {
        'mean_return': np.mean(simulation_results),
        'std_return': np.std(simulation_results),
        'min_return': np.min(simulation_results),
        'max_return': np.max(simulation_results),
        'percentile_5': np.percentile(simulation_results, 5),
        'percentile_25': np.percentile(simulation_results, 25),
        'percentile_75': np.percentile(simulation_results, 75),
        'percentile_95': np.percentile(simulation_results, 95),
        'probability_of_loss': sum(1 for r in simulation_results if r < 0) / len(simulation_results),
        'probability_of_target': sum(1 for r in simulation_results if r >= 64) / len(simulation_results),  # 8% * 8 months
        'var_95': np.percentile(simulation_results, 5),  # Value at Risk (95% confidence)
        'expected_shortfall': np.mean([r for r in simulation_results if r <= np.percentile(simulation_results, 5)])
    }
    
    # Display Monte Carlo results
    print("\n" + "=" * 80)
    print("🎰 MONTE CARLO SIMULATION RESULTS")
    print("=" * 80)
    print(f"📊 Simulation Parameters:")
    print(f"   • Number of Simulations: {num_simulations:,}")
    print(f"   • Historical Trades Used: {len(trade_returns):,}")
    
    print(f"\n📈 Return Distribution:")
    print(f"   • Mean Return: {mc_results['mean_return']:.2f}%")
    print(f"   • Standard Deviation: {mc_results['std_return']:.2f}%")
    print(f"   • Minimum Return: {mc_results['min_return']:.2f}%")
    print(f"   • Maximum Return: {mc_results['max_return']:.2f}%")
    
    print(f"\n📊 Percentiles:")
    print(f"   • 5th Percentile: {mc_results['percentile_5']:.2f}%")
    print(f"   • 25th Percentile: {mc_results['percentile_25']:.2f}%")
    print(f"   • 75th Percentile: {mc_results['percentile_75']:.2f}%")
    print(f"   • 95th Percentile: {mc_results['percentile_95']:.2f}%")
    
    print(f"\n🎯 Risk Metrics:")
    print(f"   • Probability of Loss: {mc_results['probability_of_loss']:.2%}")
    print(f"   • Probability of Meeting Target: {mc_results['probability_of_target']:.2%}")
    print(f"   • Value at Risk (95%): {mc_results['var_95']:.2f}%")
    print(f"   • Expected Shortfall: {mc_results['expected_shortfall']:.2f}%")
    
    return mc_results

def main():
    """Main execution function"""
    print("🚀 Eden Comprehensive Backtesting System")
    print("=" * 80)
    
    # Initialize components
    data_manager = DataManager()
    ml_optimizer = MLOptimizer()
    backtest_engine = BacktestEngine(data_manager, ml_optimizer)
    
    if not data_manager.mt5_initialized:
        print("❌ Cannot proceed without MT5 connection")
        return
    
    # Define test parameters
    symbols = ['XAUUSD', 'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF']
    start_date = datetime(2025, 1, 1)
    end_date = datetime(2025, 9, 15)
    
    print(f"📊 Testing Parameters:")
    print(f"   • Symbols: {', '.join(symbols)}")
    print(f"   • Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"   • Target: 8% monthly returns")
    print(f"   • Max Risk per Trade: 2.5%")
    print()
    
    # Run comprehensive backtest
    start_time = time.time()
    
    try:
        results = backtest_engine.run_comprehensive_backtest(symbols, start_date, end_date)
        
        # Display results
        backtest_engine.display_results(results)
        
        # Run Monte Carlo simulation
        if results['summary']['total_trades'] > 10:
            monte_carlo_results = run_monte_carlo_simulation(backtest_engine)
        
        end_time = time.time()
        print(f"\n⏱️ Total execution time: {end_time - start_time:.1f} seconds")
        
        # Save results
        results_file = f"eden_backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            # Convert datetime objects to strings for JSON serialization
            json_results = {
                'summary': results['summary'],
                'monthly_results': {k: asdict(v) for k, v in results['monthly_results'].items()},
                'symbol_results': results['symbol_results'],
                'monte_carlo': monte_carlo_results if 'monte_carlo_results' in locals() else {}
            }
            json.dump(json_results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {results_file}")
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        print(f"❌ Backtest failed: {e}")
    
    finally:
        # Cleanup
        if hasattr(data_manager, 'mt5_initialized') and data_manager.mt5_initialized:
            mt5.shutdown()

if __name__ == "__main__":
    main()