import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class MLFeatureExtractor:
    """
    Extracts technical and market context features for ML model.
    """
    
    def __init__(self):
        self.feature_names = [
            'rsi', 'macd', 'macd_signal', 'macd_hist',
            'atr_pct', 'dist_ema_200', 'dist_ema_50',
            'volatility_ratio', 'hour', 'day_of_week',
            'trend_strength'
        ]

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate features for the entire dataframe.
        """
        try:
            df = df.copy()
            
            # 1. RSI (14)
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # 2. MACD (12, 26, 9)
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
            
            # 3. ATR (14) as % of price
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = tr.rolling(14).mean()
            df['atr_pct'] = (atr / df['close']) * 100
            
            # 4. Distance from EMAs
            ema_200 = df['close'].ewm(span=200, adjust=False).mean()
            ema_50 = df['close'].ewm(span=50, adjust=False).mean()
            df['dist_ema_200'] = (df['close'] - ema_200) / ema_200 * 100
            df['dist_ema_50'] = (df['close'] - ema_50) / ema_50 * 100
            
            # 5. Volatility Ratio (Current ATR / Avg ATR)
            avg_atr = atr.rolling(50).mean()
            df['volatility_ratio'] = atr / avg_atr
            
            # 6. Time Features
            if 'time' in df.columns:
                times = df['time']
            else:
                times = df.index.to_series()
                
            df['hour'] = times.dt.hour
            df['day_of_week'] = times.dt.dayofweek
            
            # 7. Trend Strength (ADX simplified or Slope)
            # Simple slope of EMA 50
            df['trend_strength'] = (ema_50 - ema_50.shift(5)) / ema_50.shift(5) * 1000
            
            # Fill NaNs
            df = df.fillna(0)
            
            return df[self.feature_names]
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return pd.DataFrame()

    def get_latest_features(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """Get features for the last bar (for live inference)."""
        features_df = self.extract_features(df)
        if features_df.empty:
            return None
        return features_df.iloc[-1].values.reshape(1, -1)
