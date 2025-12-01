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
        Adjust risk percentage based on ML confidence and Daily DD.
        """
        if not self.enable_ml or self.model is None:
            return base_risk_pct
        
        try:
            # Extract features (simplified for now, should match training)
            # In production, this needs to match the exact features used in training
            # For the portfolio model, we used [volatility, trend, etc.]
            # Here we'll use a simplified heuristic that mimics the ML output for safety
            # until the feature extractor is fully aligned
            
            volatility = df['close'].pct_change().std()
            
            # High volatility -> Reduce risk (Circuit Breaker Logic)
            if volatility > 0.002:
                return base_risk_pct * 0.5
            elif volatility < 0.0005:
                return base_risk_pct * 0.8
            else:
                return base_risk_pct * 1.2
                
        except Exception as e:
            logger.error(f"Error in ML risk adjustment: {e}. Using base risk.")
            return base_risk_pct
