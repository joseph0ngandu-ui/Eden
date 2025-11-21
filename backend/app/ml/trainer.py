import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import logging
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import asyncio

from app.services.deriv_client import DerivClient
from app.services.data_processor import DataProcessor
from app.ml.models import LSTMModel

logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Handles training of ML models.
    Supports concurrent training using ProcessPoolExecutor.
    """
    
    def __init__(self):
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        self.deriv_client = DerivClient()
        self.data_processor = DataProcessor()

    async def train_model(self, symbol: str, model_type: str = "lstm", data_length: int = 1000, 
                         epochs: int = 50, batch_size: int = 32, learning_rate: float = 0.001):
        """
        Orchestrates the training process:
        1. Fetch data from Deriv (amount based on data_length)
        2. Process data
        3. Train model (CPU intensive, run in executor)
        4. Save model
        """
        logger.info(f"Starting {model_type.upper()} training for {symbol} with {data_length} candles...")
        
        # 1. Fetch Data (Async I/O)
        # Fetch slightly more to account for indicator warmup
        raw_data = await self.deriv_client.get_history(symbol, count=data_length + 200)
        if not raw_data:
            raise Exception(f"No data fetched for {symbol}")
            
        # 2. Process Data (CPU)
        df = self.data_processor.clean_data(raw_data)
        df = self.data_processor.add_technical_indicators(df)
        X, y = self.data_processor.prepare_sequences(df)
        
        if len(X) == 0:
            raise Exception("Insufficient data for training")
            
        input_dim = X.shape[2]
        
        # 3. Train Model (Run in separate process to avoid blocking event loop)
        loop = asyncio.get_running_loop()
        with ProcessPoolExecutor() as pool:
            metrics = await loop.run_in_executor(
                pool, 
                self._run_training_loop, 
                symbol, model_type, X, y, input_dim, epochs, batch_size, learning_rate
            )
            
        logger.info(f"Training completed for {symbol}. Accuracy: {metrics['accuracy']:.2f}%")
        return metrics

    def _run_training_loop(self, symbol: str, model_type: str, X: np.ndarray, y: np.ndarray, input_dim: int, 
                          epochs: int, batch_size: int, learning_rate: float) -> dict:
        """
        The actual training loop running in a separate process.
        """
        # Determine device: MPS (Mac), CUDA (NVIDIA), or CPU
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info(f"Training on Mac GPU (MPS) for {symbol}")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Training on NVIDIA GPU (CUDA) for {symbol}")
        else:
            device = torch.device("cpu")
            logger.info(f"Training on CPU for {symbol}")

        # Convert to PyTorch tensors and move to device
        X_tensor = torch.FloatTensor(X).to(device)
        y_tensor = torch.FloatTensor(y).unsqueeze(1).to(device) # Shape (N, 1)
        
        # Split train/test (80/20)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X_tensor[:split_idx], X_tensor[split_idx:]
        y_train, y_test = y_tensor[:split_idx], y_tensor[split_idx:]
        
        # Create DataLoaders
        train_data = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        
        # Initialize Model using Factory
        from app.ml.models import get_model # Import here to avoid pickling issues if needed
        model = get_model(model_type, input_dim=input_dim).to(device)
        
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training Loop
        model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(train_loader):.4f}")

        # Evaluation
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            predicted = (test_outputs > 0.5).float()
            accuracy = (predicted == y_test).float().mean().item() * 100
            
        # Save Model (move to CPU first for compatibility)
        model.to("cpu")
        model_path = self.models_dir / f"{symbol}_{model_type}.pth"
        torch.save(model.state_dict(), model_path)
        
        return {
            "symbol": symbol,
            "accuracy": accuracy,
            "epochs": epochs,
            "final_loss": epoch_loss / len(train_loader),
            "model_path": str(model_path),
            "device": str(device)
        }
