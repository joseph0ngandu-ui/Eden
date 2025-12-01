import pandas as pd
import pandas_ta as ta
import numpy as np
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Service for processing raw market data into ML-ready features.
    Handles cleaning, normalization, and technical indicator calculation.
    """

    @staticmethod
    def clean_data(data: List[Dict]) -> pd.DataFrame:
        """
        Convert raw Deriv candle data to a clean DataFrame.
        """
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        
        # Convert epoch to datetime
        if 'epoch' in df.columns:
            df['time'] = pd.to_datetime(df['epoch'], unit='s')
            df.set_index('time', inplace=True)
            df.drop('epoch', axis=1, inplace=True)
        
        # Ensure numeric types
        cols = ['open', 'high', 'low', 'close']
        for col in cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col])
                
        return df

    @staticmethod
    def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators using pandas-ta.
        """
        if df.empty:
            return df

        # RSI
        df.ta.rsi(length=14, append=True)
        
        # MACD
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        
        # Bollinger Bands
        df.ta.bbands(length=20, std=2, append=True)
        
        # ATR (Average True Range) for volatility
        df.ta.atr(length=14, append=True)
        
        # SMA/EMA
        df.ta.sma(length=50, append=True)
        df.ta.ema(length=200, append=True)
        
        # Drop NaN values created by indicators
        df.dropna(inplace=True)
        
        return df

    @staticmethod
    def prepare_sequences(df: pd.DataFrame, target_col: str = 'close', 
                         seq_length: int = 60, prediction_horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for LSTM training.
        
        Args:
            df: DataFrame with features
            target_col: Column to predict
            seq_length: Length of input sequence (lookback)
            prediction_horizon: How many steps ahead to predict
            
        Returns:
            X: Input sequences (samples, seq_length, features)
            y: Target values (samples, )
        """
        if df.empty:
            return np.array([]), np.array([])

        # Feature selection (exclude non-numeric or target if needed)
        # For now, use all numeric columns as features
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        data = df[feature_cols].values
        
        # Normalize data (MinMax scaling per window or global)
        # Simple global normalization for now (production should use scaler pipeline)
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        # Avoid division by zero
        range_val = max_val - min_val
        range_val[range_val == 0] = 1
        normalized_data = (data - min_val) / range_val
        
        X, y = [], []
        
        # Create sequences
        for i in range(len(normalized_data) - seq_length - prediction_horizon + 1):
            X.append(normalized_data[i:(i + seq_length)])
            
            # Target: 1 if price goes up, 0 if down (Binary Classification)
            # Or return the actual future price for Regression
            current_price = data[i + seq_length - 1, df.columns.get_loc(target_col)]
            future_price = data[i + seq_length + prediction_horizon - 1, df.columns.get_loc(target_col)]
            
            # Binary target: 1 if Up, 0 if Down
            target = 1 if future_price > current_price else 0
            y.append(target)
            
        return np.array(X), np.array(y)

    @staticmethod
    def get_feature_dim(df: pd.DataFrame) -> int:
        """Get the number of features."""
        return len(df.select_dtypes(include=[np.number]).columns)
