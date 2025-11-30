import pandas as pd
import numpy as np
import joblib
import logging
from typing import Optional
from pathlib import Path

from trading.ml_features import MLFeatureExtractor

logger = logging.getLogger(__name__)

class MLRiskManager:
    """
    ML-based risk manager that adjusts position sizing based on predicted trade success probability.
    """
    
    def __init__(self, model_path: str = "ml_risk_model.pkl", enable_ml: bool = True):
        """
        Initialize ML Risk Manager.
        
        Args:
            model_path: Path to trained ML model
            enable_ml: Enable/disable ML adjustment (fallback to base risk if disabled)
        """
        self.enable_ml = enable_ml
        self.model = None
        self.feature_extractor = MLFeatureExtractor()
        
        if enable_ml:
            try:
                if Path(model_path).exists():
                    self.model = joblib.load(model_path)
                    logger.info(f"ML model loaded from {model_path}")
                else:
                    logger.warning(f"ML model not found at {model_path}. ML risk adjustment disabled.")
                    self.enable_ml = False
            except Exception as e:
                logger.error(f"Error loading ML model: {e}. ML risk adjustment disabled.")
                self.enable_ml = False
    
    def adjust_risk(
        self, 
        df: pd.DataFrame, 
        base_risk_pct: float, 
        direction: str,
        symbol: str = None
    ) -> float:
        """
        Adjust risk percentage based on ML confidence.
        
        Args:
            df: Price dataframe with OHLCV
            base_risk_pct: Base risk % (e.g., 0.15 or 0.5)
            direction: Trade direction ('LONG' or 'SHORT')
            symbol: Symbol name (optional, for logging)
            
        Returns:
            Adjusted risk percentage (never exceeds base_risk_pct)
        """
        if not self.enable_ml or self.model is None:
            return base_risk_pct
        
        try:
            # Extract features
            features = self.feature_extractor.get_latest_features(df)
            if features is None:
                logger.warning("Failed to extract features. Using base risk.")
                return base_risk_pct
            
            # Add direction feature (1 for LONG, 0 for SHORT)
            direction_encoded = 1 if direction == 'LONG' else 0
            features = np.append(features, direction_encoded).reshape(1, -1)
            
            # Predict win probability
            win_prob = self.model.predict_proba(features)[0, 1]
            
            # Adjust risk based on confidence
            # High confidence (>60%): 100% of base risk
            # Medium confidence (40-60%): 70% of base risk
            # Low confidence (<40%): 30% of base risk
            
            if win_prob >= 0.60:
                risk_multiplier = 1.0
            elif win_prob >= 0.40:
                risk_multiplier = 0.7
            else:
                risk_multiplier = 0.3
            
            adjusted_risk = base_risk_pct * risk_multiplier
            
            log_msg = f"ML Risk Adjustment: Prob={win_prob:.2%}, Multiplier={risk_multiplier:.0%}, Risk={base_risk_pct:.3f}% → {adjusted_risk:.3f}%"
            if symbol:
                log_msg = f"{symbol} | {log_msg}"
            logger.info(log_msg)
            
            return adjusted_risk
            
        except Exception as e:
            logger.error(f"Error in ML risk adjustment: {e}. Using base risk.")
            return base_risk_pct
