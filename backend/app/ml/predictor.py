import torch
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Optional
import pandas as pd

from app.ml.models import LSTMModel
from app.services.deriv_client import DerivClient
from app.services.data_processor import DataProcessor

logger = logging.getLogger(__name__)

class Predictor:
    """
    Service for making predictions using trained models.
    """
    
    def __init__(self):
        self.models_dir = Path("models")
        self.deriv_client = DerivClient()
        self.data_processor = DataProcessor()
        self.models: Dict[str, LSTMModel] = {}
        
    def load_model(self, symbol: str, input_dim: int) -> Optional[LSTMModel]:
        """Load a trained model from disk."""
        model_path = self.models_dir / f"{symbol}_lstm.pth"
        if not model_path.exists():
            logger.warning(f"No model found for {symbol}")
            return None
            
        try:
            model = LSTMModel(input_dim=input_dim)
            model.load_state_dict(torch.load(model_path))
            model.eval()
            self.models[symbol] = model
            return model
        except Exception as e:
            logger.error(f"Failed to load model for {symbol}: {e}")
            return None

    async def predict(self, symbol: str) -> Dict:
        """
        Generate a prediction for the next candle.
        """
        # 1. Fetch recent data
        # We need enough data for the sequence length (e.g., 60) + indicators
        raw_data = await self.deriv_client.get_history(symbol, count=200)
        if not raw_data:
            return {"error": "Failed to fetch data"}
            
        # 2. Process Data
        df = self.data_processor.clean_data(raw_data)
        df = self.data_processor.add_technical_indicators(df)
        
        # Get the last sequence
        seq_length = 60
        if len(df) < seq_length:
            return {"error": "Insufficient data for prediction"}
            
        # Prepare input
        # Note: This needs to match the normalization used in training!
        # For this MVP, we re-normalize based on the recent window, which is an approximation.
        # In production, save the scaler with the model.
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        data = df[feature_cols].values
        
        # Normalize (Simple MinMax)
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        range_val = max_val - min_val
        range_val[range_val == 0] = 1
        normalized_data = (data - min_val) / range_val
        
        # Extract last sequence
        last_sequence = normalized_data[-seq_length:]
        input_tensor = torch.FloatTensor(last_sequence).unsqueeze(0) # Shape (1, seq_len, features)
        
        # 3. Load Model if needed
        input_dim = input_tensor.shape[2]
        model = self.models.get(symbol)
        if not model:
            model = self.load_model(symbol, input_dim)
            
        if not model:
            return {"error": "Model not trained for this symbol"}
            
        # 4. Predict
        with torch.no_grad():
            prediction = model(input_tensor).item()
            
        direction = "UP" if prediction > 0.5 else "DOWN"
        confidence = prediction if prediction > 0.5 else 1 - prediction
        
        return {
            "symbol": symbol,
            "prediction": direction,
            "probability": prediction,
            "confidence": confidence * 100,
            "timestamp": pd.Timestamp.now().isoformat()
        }
