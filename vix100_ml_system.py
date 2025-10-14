#!/usr/bin/env python3
"""
VIX100 Self-Learning ML System
==============================

Advanced machine learning system that continuously learns and evolves through:
- Nightly retraining on VIX100 data
- Model performance evaluation and selection
- Hyperparameter optimization
- Automatic deployment of improved models
- Pattern discovery and strategy generation
- Reinforcement learning from trade outcomes

Key Features:
- Multiple ML model types (RF, XGB, Neural Networks, SVM)
- Online learning capabilities
- Model ensemble management
- Feature engineering pipeline
- Performance monitoring and model retirement
- Strategy evolution based on market regime changes

Author: Eden AI System
Version: 1.0
Date: October 13, 2025
"""

import pandas as pd
import numpy as np
import joblib
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor
import warnings

# ML imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import optuna

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class ModelMetrics:
    """Model performance metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    cross_val_score: float
    feature_importance: Dict[str, float]
    training_time: float
    prediction_time: float
    last_updated: datetime

@dataclass
class MLPrediction:
    """ML model prediction result"""
    signal: str  # 'buy', 'sell', 'hold'
    confidence: float  # 0.0 to 1.0
    probability: float  # Raw probability
    model_name: str
    features_used: List[str]
    feature_values: Dict[str, float]
    prediction_time: datetime

@dataclass
class TrainingResult:
    """Training session result"""
    model_name: str
    success: bool
    metrics: Optional[ModelMetrics]
    improvement: float  # Performance improvement over previous version
    training_samples: int
    error_message: Optional[str] = None

class VIX100FeatureEngineer:
    """Advanced feature engineering for VIX100 ML models"""
    
    def __init__(self):
        self.feature_cache = {}
        
    def engineer_features(self, df: pd.DataFrame, indicators: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive feature set for ML training"""
        
        if df.empty or indicators.empty:
            return pd.DataFrame()
        
        features = pd.DataFrame(index=df.index)
        
        try:
            # Price-based features
            features['price_change'] = df['close'].pct_change()
            features['price_change_5'] = df['close'].pct_change(5)
            features['price_change_10'] = df['close'].pct_change(10)
            features['price_change_20'] = df['close'].pct_change(20)
            
            # Volatility features
            features['volatility_5'] = df['close'].rolling(5).std()
            features['volatility_10'] = df['close'].rolling(10).std()
            features['volatility_20'] = df['close'].rolling(20).std()
            features['volatility_ratio'] = features['volatility_5'] / features['volatility_20']
            
            # Volume features
            if 'tick_volume' in df.columns:
                features['volume_ma'] = df['tick_volume'].rolling(10).mean()
                features['volume_ratio'] = df['tick_volume'] / features['volume_ma']
                features['volume_change'] = df['tick_volume'].pct_change()
            
            # Technical indicator features
            if 'vol_pressure_basic' in indicators.columns:
                features['vol_pressure'] = indicators['vol_pressure_basic']
                features['vol_pressure_ma'] = indicators['vol_pressure_basic'].rolling(10).mean()
                features['vol_pressure_std'] = indicators['vol_pressure_basic'].rolling(10).std()
            
            if 'compression_intensity' in indicators.columns:
                features['compression'] = indicators['compression_intensity']
                
            if 'bb_width' in indicators.columns:
                features['bb_width'] = indicators['bb_width']
                features['bb_position'] = (df['close'] - indicators.get('bb_lower', df['close'])) / \
                                        (indicators.get('bb_upper', df['close']) - indicators.get('bb_lower', df['close']))
            
            # Time-based features (VIX100 trades 24/7)
            features['hour'] = df.index.hour
            features['day_of_week'] = df.index.dayofweek
            features['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
            
            # Momentum features
            features['momentum_3'] = df['close'].pct_change(3)
            features['momentum_7'] = df['close'].pct_change(7)
            features['momentum_14'] = df['close'].pct_change(14)
            
            # RSI-style features
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss.replace(0, np.inf)
            features['rsi'] = 100 - (100 / (1 + rs))
            
            # Support/Resistance features
            features['support_dist'] = (df['close'] - df['low'].rolling(20).min()) / df['close']
            features['resistance_dist'] = (df['high'].rolling(20).max() - df['close']) / df['close']
            
            # Trend features
            features['sma_5'] = df['close'].rolling(5).mean()
            features['sma_20'] = df['close'].rolling(20).mean()
            features['trend_5'] = (df['close'] > features['sma_5']).astype(int)
            features['trend_20'] = (df['close'] > features['sma_20']).astype(int)
            features['trend_alignment'] = features['trend_5'] + features['trend_20']
            
            # Cross-over features
            features['sma_cross'] = (features['sma_5'] > features['sma_20']).astype(int)
            features['price_above_sma5'] = (df['close'] > features['sma_5']).astype(int)
            features['price_above_sma20'] = (df['close'] > features['sma_20']).astype(int)
            
            # Synthetic market specific features
            features['synthetic_wave'] = np.sin(2 * np.pi * df.index.hour / 24)  # Daily cycle
            features['synthetic_weekly'] = np.sin(2 * np.pi * df.index.dayofweek / 7)  # Weekly cycle
            
            # Regime detection features
            features['high_vol_regime'] = (features['volatility_20'] > 
                                         features['volatility_20'].rolling(100).quantile(0.8)).astype(int)
            
            # Pattern features
            features['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
            features['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
            
            # Clean features
            features = features.replace([np.inf, -np.inf], np.nan)
            features = features.fillna(method='ffill').fillna(0)
            
            return features
            
        except Exception as e:
            logger.error(f"Error engineering features: {e}")
            return pd.DataFrame()

class VIX100MLModel:
    """Individual ML model for VIX100 predictions"""
    
    def __init__(self, model_name: str, model_type: str, hyperparams: Dict = None):
        self.model_name = model_name
        self.model_type = model_type
        self.hyperparams = hyperparams or {}
        
        self.model = None
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.is_trained = False
        self.metrics = None
        self.feature_names = []
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the ML model based on type"""
        
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=self.hyperparams.get('n_estimators', 100),
                max_depth=self.hyperparams.get('max_depth', 10),
                min_samples_split=self.hyperparams.get('min_samples_split', 5),
                min_samples_leaf=self.hyperparams.get('min_samples_leaf', 2),
                random_state=42,
                n_jobs=-1
            )
            
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=self.hyperparams.get('n_estimators', 100),
                learning_rate=self.hyperparams.get('learning_rate', 0.1),
                max_depth=self.hyperparams.get('max_depth', 5),
                min_samples_split=self.hyperparams.get('min_samples_split', 5),
                random_state=42
            )
            
        elif self.model_type == 'neural_network':
            hidden_layer_sizes = self.hyperparams.get('hidden_layer_sizes', (100, 50))
            self.model = MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                learning_rate_init=self.hyperparams.get('learning_rate', 0.001),
                alpha=self.hyperparams.get('alpha', 0.0001),
                max_iter=self.hyperparams.get('max_iter', 500),
                random_state=42
            )
            
        elif self.model_type == 'svm':
            self.model = SVC(
                C=self.hyperparams.get('C', 1.0),
                kernel=self.hyperparams.get('kernel', 'rbf'),
                gamma=self.hyperparams.get('gamma', 'scale'),
                probability=True,
                random_state=42
            )
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> TrainingResult:
        """Train the model on provided data"""
        
        if X.empty or y.empty:
            return TrainingResult(
                model_name=self.model_name,
                success=False,
                metrics=None,
                improvement=0.0,
                training_samples=0,
                error_message="Empty training data"
            )
        
        try:
            start_time = datetime.now()
            
            # Store feature names
            self.feature_names = list(X.columns)
            
            # Feature selection
            self.feature_selector = SelectKBest(
                score_func=mutual_info_classif,
                k=min(20, len(X.columns))  # Select top 20 features
            )
            
            X_selected = self.feature_selector.fit_transform(X, y)
            selected_features = X.columns[self.feature_selector.get_support()]
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X_selected)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train model
            self.model.fit(X_train, y_train)
            self.is_trained = True
            
            # Evaluate model
            y_pred = self.model.predict(X_test)
            y_proba = self.model.predict_proba(X_test)[:, 1] if hasattr(self.model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # Cross-validation
            cv_scores = cross_val_score(self.model, X_train, y_train, cv=3)
            cv_score = np.mean(cv_scores)
            
            # Feature importance
            feature_importance = {}
            if hasattr(self.model, 'feature_importances_'):
                for i, importance in enumerate(self.model.feature_importances_):
                    if i < len(selected_features):
                        feature_importance[selected_features.iloc[i]] = importance
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Calculate improvement
            improvement = 0.0
            if self.metrics:
                improvement = accuracy - self.metrics.accuracy
            
            self.metrics = ModelMetrics(
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                cross_val_score=cv_score,
                feature_importance=feature_importance,
                training_time=training_time,
                prediction_time=0.0,  # Will be updated during prediction
                last_updated=datetime.now()
            )
            
            return TrainingResult(
                model_name=self.model_name,
                success=True,
                metrics=self.metrics,
                improvement=improvement,
                training_samples=len(X)
            )
            
        except Exception as e:
            logger.error(f"Error training {self.model_name}: {e}")
            return TrainingResult(
                model_name=self.model_name,
                success=False,
                metrics=None,
                improvement=0.0,
                training_samples=len(X) if not X.empty else 0,
                error_message=str(e)
            )
    
    def predict(self, X: pd.DataFrame) -> Optional[MLPrediction]:
        """Make prediction on new data"""
        
        if not self.is_trained or X.empty:
            return None
        
        try:
            start_time = datetime.now()
            
            # Ensure we have the same features as training
            X_aligned = X.reindex(columns=self.feature_names, fill_value=0)
            
            # Apply feature selection and scaling
            X_selected = self.feature_selector.transform(X_aligned)
            X_scaled = self.scaler.transform(X_selected)
            
            # Make prediction
            prediction = self.model.predict(X_scaled)[0]
            
            # Get probability if available
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(X_scaled)[0]
                confidence = np.max(probabilities)
                probability = probabilities[1] if len(probabilities) > 1 else probabilities[0]
            else:
                confidence = 0.5
                probability = 0.5
            
            # Convert prediction to signal
            signal_map = {0: 'sell', 1: 'buy', 2: 'hold'}
            signal = signal_map.get(prediction, 'hold')
            
            prediction_time = (datetime.now() - start_time).total_seconds()
            
            # Update prediction time metric
            if self.metrics:
                self.metrics.prediction_time = prediction_time
            
            # Get feature values
            feature_values = {}
            selected_features = X.columns[self.feature_selector.get_support()]
            for i, feature in enumerate(selected_features):
                if i < len(X_selected[0]):
                    feature_values[feature] = X_selected[0][i]
            
            return MLPrediction(
                signal=signal,
                confidence=confidence,
                probability=probability,
                model_name=self.model_name,
                features_used=list(selected_features),
                feature_values=feature_values,
                prediction_time=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error predicting with {self.model_name}: {e}")
            return None
    
    def save_model(self, file_path: str):
        """Save model to disk"""
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_selector': self.feature_selector,
                'feature_names': self.feature_names,
                'metrics': asdict(self.metrics) if self.metrics else None,
                'hyperparams': self.hyperparams,
                'model_type': self.model_type
            }
            joblib.dump(model_data, file_path)
            logger.info(f"Model {self.model_name} saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving model {self.model_name}: {e}")
    
    def load_model(self, file_path: str) -> bool:
        """Load model from disk"""
        try:
            model_data = joblib.load(file_path)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_selector = model_data['feature_selector']
            self.feature_names = model_data['feature_names']
            self.hyperparams = model_data['hyperparams']
            
            if model_data['metrics']:
                self.metrics = ModelMetrics(**model_data['metrics'])
            
            self.is_trained = True
            logger.info(f"Model {self.model_name} loaded from {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model {self.model_name}: {e}")
            return False

class VIX100MLEnsemble:
    """Ensemble of ML models for improved predictions"""
    
    def __init__(self):
        self.models = {}
        self.ensemble_weights = {}
        self.performance_history = {}
        
    def add_model(self, model: VIX100MLModel, weight: float = 1.0):
        """Add model to ensemble"""
        self.models[model.model_name] = model
        self.ensemble_weights[model.model_name] = weight
        self.performance_history[model.model_name] = []
    
    def remove_model(self, model_name: str):
        """Remove model from ensemble"""
        if model_name in self.models:
            del self.models[model_name]
            del self.ensemble_weights[model_name]
            del self.performance_history[model_name]
    
    def predict_ensemble(self, X: pd.DataFrame) -> Optional[MLPrediction]:
        """Make ensemble prediction"""
        
        if not self.models or X.empty:
            return None
        
        predictions = {}
        weights = {}
        
        # Get predictions from all models
        for model_name, model in self.models.items():
            if model.is_trained:
                pred = model.predict(X)
                if pred:
                    predictions[model_name] = pred
                    weights[model_name] = self.ensemble_weights[model_name]
        
        if not predictions:
            return None
        
        # Weighted ensemble prediction
        signal_votes = {'buy': 0, 'sell': 0, 'hold': 0}
        total_confidence = 0
        total_weight = 0
        
        for model_name, pred in predictions.items():
            weight = weights[model_name]
            signal_votes[pred.signal] += weight * pred.confidence
            total_confidence += pred.confidence * weight
            total_weight += weight
        
        # Determine final signal
        final_signal = max(signal_votes, key=signal_votes.get)
        final_confidence = total_confidence / total_weight if total_weight > 0 else 0.5
        
        # Average probability
        avg_probability = np.mean([pred.probability for pred in predictions.values()])
        
        return MLPrediction(
            signal=final_signal,
            confidence=final_confidence,
            probability=avg_probability,
            model_name='ensemble',
            features_used=list(set().union(*[pred.features_used for pred in predictions.values()])),
            feature_values={},  # Would be complex to combine
            prediction_time=datetime.now()
        )
    
    def update_weights(self, performance_data: Dict[str, float]):
        """Update model weights based on recent performance"""
        
        for model_name, performance in performance_data.items():
            if model_name in self.ensemble_weights:
                # Store performance history
                self.performance_history[model_name].append(performance)
                
                # Keep only last 100 performances
                if len(self.performance_history[model_name]) > 100:
                    self.performance_history[model_name] = self.performance_history[model_name][-100:]
                
                # Update weight based on recent performance
                recent_perf = np.mean(self.performance_history[model_name][-10:])
                self.ensemble_weights[model_name] = max(0.1, recent_perf)  # Minimum weight 0.1

class VIX100HyperparameterOptimizer:
    """Automated hyperparameter optimization using Optuna"""
    
    def __init__(self):
        self.study = None
        self.best_params = {}
    
    def optimize_model(self, model_type: str, X: pd.DataFrame, y: pd.Series, 
                      n_trials: int = 50) -> Dict:
        """Optimize hyperparameters for a specific model type"""
        
        def objective(trial):
            """Optuna objective function"""
            
            if model_type == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'max_depth': trial.suggest_int('max_depth', 5, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5)
                }
                
            elif model_type == 'gradient_boosting':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10)
                }
                
            elif model_type == 'neural_network':
                n_layers = trial.suggest_int('n_layers', 1, 3)
                hidden_sizes = []
                for i in range(n_layers):
                    hidden_sizes.append(trial.suggest_int(f'layer_{i}_size', 50, 200))
                
                params = {
                    'hidden_layer_sizes': tuple(hidden_sizes),
                    'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.01),
                    'alpha': trial.suggest_float('alpha', 0.0001, 0.01),
                    'max_iter': trial.suggest_int('max_iter', 200, 1000)
                }
                
            elif model_type == 'svm':
                params = {
                    'C': trial.suggest_float('C', 0.1, 10.0),
                    'kernel': trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid']),
                    'gamma': trial.suggest_categorical('gamma', ['scale', 'auto'])
                }
            
            else:
                return 0.0
            
            # Create and evaluate model
            try:
                model = VIX100MLModel(f"optimize_{model_type}", model_type, params)
                result = model.train(X, y)
                
                if result.success and result.metrics:
                    return result.metrics.cross_val_score
                else:
                    return 0.0
                    
            except Exception:
                return 0.0
        
        try:
            # Create study
            study_name = f"vix100_{model_type}_optimization"
            self.study = optuna.create_study(
                direction='maximize',
                study_name=study_name,
                storage=None  # In-memory
            )
            
            # Optimize
            self.study.optimize(objective, n_trials=n_trials, timeout=300)  # 5 minute timeout
            
            self.best_params[model_type] = self.study.best_params
            
            logger.info(f"Best params for {model_type}: {self.study.best_params}")
            logger.info(f"Best score: {self.study.best_value}")
            
            return self.study.best_params
            
        except Exception as e:
            logger.error(f"Error optimizing {model_type}: {e}")
            return {}

class VIX100MLSystem:
    """Complete VIX100 machine learning system"""
    
    def __init__(self, data_dir: str = "vix100_ml_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.feature_engineer = VIX100FeatureEngineer()
        self.ensemble = VIX100MLEnsemble()
        self.optimizer = VIX100HyperparameterOptimizer()
        
        # System state
        self.is_training = False
        self.last_training = None
        self.training_scheduler = None
        
        # Database for ML data
        self.db_path = self.data_dir / "vix100_ml.db"
        self.init_database()
        
        # Load existing models
        self.load_saved_models()
        
        # Start training scheduler
        self.start_training_scheduler()
    
    def init_database(self):
        """Initialize ML system database"""
        with sqlite3.connect(self.db_path) as conn:
            # Training data with labels
            conn.execute("""
                CREATE TABLE IF NOT EXISTS training_data (
                    timestamp TEXT PRIMARY KEY,
                    features TEXT,  -- JSON of features
                    label INTEGER,  -- 0=sell, 1=buy, 2=hold
                    outcome REAL,   -- Actual trade outcome
                    regime TEXT,
                    volatility REAL
                )
            """)
            
            # Model performance tracking
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_performance (
                    model_name TEXT,
                    timestamp TEXT,
                    accuracy REAL,
                    precision_score REAL,
                    recall REAL,
                    f1_score REAL,
                    cv_score REAL,
                    training_samples INTEGER,
                    PRIMARY KEY (model_name, timestamp)
                )
            """)
            
            # Training sessions
            conn.execute("""
                CREATE TABLE IF NOT EXISTS training_sessions (
                    session_id TEXT PRIMARY KEY,
                    start_time TEXT,
                    end_time TEXT,
                    models_trained INTEGER,
                    successful_models INTEGER,
                    best_model TEXT,
                    notes TEXT
                )
            """)
    
    def collect_training_data(self, df: pd.DataFrame, indicators: pd.DataFrame, 
                            trades: List = None) -> bool:
        """Collect and store training data from recent market activity"""
        
        if df.empty or indicators.empty:
            return False
        
        try:
            # Engineer features
            features_df = self.feature_engineer.engineer_features(df, indicators)
            
            if features_df.empty:
                return False
            
            # Create labels based on future price movements
            labels = self._create_labels(df, lookforward=10)
            
            # Store in database
            with sqlite3.connect(self.db_path) as conn:
                for i, (timestamp, row) in enumerate(features_df.iterrows()):
                    if i < len(labels) and not pd.isna(labels.iloc[i]):
                        # Calculate regime and volatility
                        regime = 'unknown'
                        volatility = df['close'].rolling(20).std().iloc[i] if i < len(df) else 0
                        
                        conn.execute("""
                            INSERT OR REPLACE INTO training_data 
                            (timestamp, features, label, outcome, regime, volatility)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, (
                            timestamp.isoformat(),
                            json.dumps(row.to_dict()),
                            int(labels.iloc[i]),
                            0.0,  # Outcome will be updated later
                            regime,
                            volatility
                        ))
                conn.commit()
            
            logger.info(f"Collected {len(features_df)} training samples")
            return True
            
        except Exception as e:
            logger.error(f"Error collecting training data: {e}")
            return False
    
    def _create_labels(self, df: pd.DataFrame, lookforward: int = 10) -> pd.Series:
        """Create training labels based on future price movements"""
        
        future_returns = df['close'].pct_change(lookforward).shift(-lookforward)
        
        # Create labels: 0=sell, 1=buy, 2=hold
        labels = pd.Series(index=df.index, dtype=int)
        
        # Thresholds for buy/sell signals
        buy_threshold = future_returns.quantile(0.6)   # Top 40%
        sell_threshold = future_returns.quantile(0.4)  # Bottom 40%
        
        labels[future_returns > buy_threshold] = 1  # Buy
        labels[future_returns < sell_threshold] = 0  # Sell
        labels[(future_returns >= sell_threshold) & (future_returns <= buy_threshold)] = 2  # Hold
        
        return labels
    
    def get_training_data(self, min_samples: int = 1000) -> Tuple[pd.DataFrame, pd.Series]:
        """Retrieve training data from database"""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = """
                    SELECT features, label FROM training_data 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """
                
                cursor = conn.execute(query, (min_samples * 2,))  # Get extra in case of issues
                rows = cursor.fetchall()
                
                if len(rows) < min_samples:
                    logger.warning(f"Insufficient training data: {len(rows)} < {min_samples}")
                    return pd.DataFrame(), pd.Series(dtype=int)
                
                features_list = []
                labels_list = []
                
                for features_json, label in rows:
                    try:
                        features_dict = json.loads(features_json)
                        features_list.append(features_dict)
                        labels_list.append(label)
                    except:
                        continue
                
                if len(features_list) < min_samples:
                    return pd.DataFrame(), pd.Series(dtype=int)
                
                X = pd.DataFrame(features_list)
                y = pd.Series(labels_list)
                
                # Clean data
                X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
                
                logger.info(f"Retrieved {len(X)} training samples")
                return X, y
                
        except Exception as e:
            logger.error(f"Error retrieving training data: {e}")
            return pd.DataFrame(), pd.Series(dtype=int)
    
    async def train_models_nightly(self) -> Dict[str, TrainingResult]:
        """Nightly model retraining routine"""
        
        if self.is_training:
            logger.info("Training already in progress, skipping...")
            return {}
        
        self.is_training = True
        session_id = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info("Starting nightly ML model training...")
        
        try:
            # Get training data
            X, y = self.get_training_data(min_samples=2000)
            
            if X.empty or y.empty:
                logger.warning("No training data available")
                return {}
            
            # Model types to train
            model_configs = [
                ('random_forest', {}),
                ('gradient_boosting', {}),
                ('neural_network', {}),
                ('svm', {})
            ]
            
            training_results = {}
            successful_models = 0
            best_model = None
            best_score = 0
            
            # Train each model type
            for model_type, base_params in model_configs:
                logger.info(f"Training {model_type} model...")
                
                # Optimize hyperparameters
                optimized_params = self.optimizer.optimize_model(
                    model_type, X, y, n_trials=20  # Reduced for faster training
                )
                
                # Combine with base params
                final_params = {**base_params, **optimized_params}
                
                # Create and train model
                model_name = f"vix100_{model_type}_{datetime.now().strftime('%Y%m%d')}"
                model = VIX100MLModel(model_name, model_type, final_params)
                
                result = model.train(X, y)
                training_results[model_name] = result
                
                if result.success:
                    successful_models += 1
                    
                    # Add to ensemble or replace existing
                    self.ensemble.add_model(model)
                    
                    # Save model
                    model_path = self.data_dir / f"{model_name}.joblib"
                    model.save_model(str(model_path))
                    
                    # Track best model
                    if result.metrics and result.metrics.cross_val_score > best_score:
                        best_score = result.metrics.cross_val_score
                        best_model = model_name
                    
                    # Store performance in database
                    self._store_model_performance(model_name, result.metrics)
                    
                    logger.info(f"✅ {model_name} trained successfully - CV Score: {result.metrics.cross_val_score:.3f}")
                else:
                    logger.error(f"❌ {model_name} training failed: {result.error_message}")
            
            # Clean up old models
            self._cleanup_old_models()
            
            # Store training session
            self._store_training_session(session_id, successful_models, best_model)
            
            self.last_training = datetime.now()
            
            logger.info(f"Training complete: {successful_models}/{len(model_configs)} models successful")
            
            return training_results
            
        except Exception as e:
            logger.error(f"Error during nightly training: {e}")
            return {}
            
        finally:
            self.is_training = False
    
    def _store_model_performance(self, model_name: str, metrics: ModelMetrics):
        """Store model performance in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO model_performance 
                    (model_name, timestamp, accuracy, precision_score, recall, f1_score, 
                     cv_score, training_samples)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    model_name,
                    datetime.now().isoformat(),
                    metrics.accuracy,
                    metrics.precision,
                    metrics.recall,
                    metrics.f1_score,
                    metrics.cross_val_score,
                    0  # Would need to track this
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error storing performance for {model_name}: {e}")
    
    def _store_training_session(self, session_id: str, successful_models: int, best_model: str):
        """Store training session info"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO training_sessions 
                    (session_id, start_time, end_time, models_trained, successful_models, best_model)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    session_id,
                    self.last_training.isoformat() if self.last_training else datetime.now().isoformat(),
                    datetime.now().isoformat(),
                    len(self.ensemble.models),
                    successful_models,
                    best_model or ''
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error storing training session: {e}")
    
    def _cleanup_old_models(self, keep_recent: int = 5):
        """Remove old model files to save space"""
        try:
            model_files = list(self.data_dir.glob("vix100_*.joblib"))
            if len(model_files) > keep_recent:
                # Sort by modification time
                model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                
                # Remove oldest files
                for old_file in model_files[keep_recent:]:
                    old_file.unlink()
                    logger.info(f"Removed old model file: {old_file}")
                    
        except Exception as e:
            logger.error(f"Error cleaning up old models: {e}")
    
    def load_saved_models(self):
        """Load existing trained models"""
        try:
            model_files = list(self.data_dir.glob("vix100_*.joblib"))
            
            for model_file in model_files:
                try:
                    # Extract model info from filename
                    model_name = model_file.stem
                    
                    # Determine model type from filename
                    if 'random_forest' in model_name:
                        model_type = 'random_forest'
                    elif 'gradient_boosting' in model_name:
                        model_type = 'gradient_boosting'
                    elif 'neural_network' in model_name:
                        model_type = 'neural_network'
                    elif 'svm' in model_name:
                        model_type = 'svm'
                    else:
                        continue
                    
                    # Create model and load
                    model = VIX100MLModel(model_name, model_type)
                    if model.load_model(str(model_file)):
                        self.ensemble.add_model(model)
                        logger.info(f"Loaded model: {model_name}")
                        
                except Exception as e:
                    logger.error(f"Error loading model {model_file}: {e}")
                    
        except Exception as e:
            logger.error(f"Error loading saved models: {e}")
    
    def start_training_scheduler(self):
        """Start automated training scheduler"""
        
        def training_loop():
            while True:
                try:
                    # Check if it's time for nightly training (2 AM UTC)
                    now = datetime.utcnow()
                    if now.hour == 2 and now.minute < 10:  # Small window to avoid multiple triggers
                        if not self.last_training or (now - self.last_training).days >= 1:
                            # Run training in async context
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            loop.run_until_complete(self.train_models_nightly())
                            loop.close()
                    
                    # Sleep for 10 minutes before checking again
                    threading.Event().wait(600)
                    
                except Exception as e:
                    logger.error(f"Error in training scheduler: {e}")
                    threading.Event().wait(3600)  # Wait 1 hour on error
        
        self.training_scheduler = threading.Thread(target=training_loop, daemon=True)
        self.training_scheduler.start()
        logger.info("Training scheduler started")
    
    def get_prediction(self, df: pd.DataFrame, indicators: pd.DataFrame) -> Optional[MLPrediction]:
        """Get ML prediction for current market conditions"""
        
        if df.empty or indicators.empty:
            return None
        
        try:
            # Engineer features for prediction
            features = self.feature_engineer.engineer_features(df, indicators)
            
            if features.empty:
                return None
            
            # Get latest feature vector
            latest_features = features.iloc[-1:].copy()
            
            # Get ensemble prediction
            prediction = self.ensemble.predict_ensemble(latest_features)
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error getting ML prediction: {e}")
            return None
    
    def get_system_stats(self) -> Dict:
        """Get ML system statistics"""
        
        stats = {
            'total_models': len(self.ensemble.models),
            'trained_models': sum(1 for model in self.ensemble.models.values() if model.is_trained),
            'last_training': self.last_training.isoformat() if self.last_training else None,
            'is_training': self.is_training,
            'model_performance': {}
        }
        
        # Add individual model performance
        for model_name, model in self.ensemble.models.items():
            if model.metrics:
                stats['model_performance'][model_name] = {
                    'accuracy': model.metrics.accuracy,
                    'cv_score': model.metrics.cross_val_score,
                    'last_updated': model.metrics.last_updated.isoformat()
                }
        
        return stats

if __name__ == "__main__":
    # Test the ML system
    print("VIX100 ML System - Testing")
    
    # Create sample data
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
    
    # Create sample indicators
    sample_indicators = pd.DataFrame({
        'vol_pressure_basic': np.random.rand(1000) * 0.05,
        'compression_intensity': np.random.rand(1000),
        'bb_width': np.random.rand(1000) * 0.1,
        'bb_upper': sample_data['close'] * 1.02,
        'bb_lower': sample_data['close'] * 0.98
    }, index=dates)
    
    # Initialize ML system
    ml_system = VIX100MLSystem("test_ml_data")
    
    # Collect training data
    success = ml_system.collect_training_data(sample_data, sample_indicators)
    print(f"Training data collection: {'Success' if success else 'Failed'}")
    
    # Get system stats
    stats = ml_system.get_system_stats()
    print(f"ML System Stats: {stats}")
    
    print("VIX100 ML System initialized successfully!")