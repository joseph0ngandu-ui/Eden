from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict
import logging

from app.ml.trainer import ModelTrainer
from app.ml.predictor import Predictor

router = APIRouter(
    prefix="/ml",
    tags=["Machine Learning"]
)

logger = logging.getLogger(__name__)

# Services (Singleton pattern for simplicity in this scope)
trainer = ModelTrainer()
predictor = Predictor()

# In-memory status tracking
training_status: Dict[str, Dict] = {}

class TrainingRequest(BaseModel):
    symbol: str
    model_type: str = "lstm"
    data_length: int = 1000
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001

class PredictionResponse(BaseModel):
    symbol: str
    prediction: str
    probability: float
    confidence: float
    timestamp: str

    background_tasks.add_task(
        run_training_task, 
        request.symbol, 
        request
    )
    return {"message": f"Training started for {request.symbol}", "status": "queued"}

async def run_training_task(symbol: str, params: TrainingRequest):
    """Background task wrapper for training."""
    try:
        training_status[symbol] = {"status": "training", "progress": 0}
        metrics = await trainer.train_model(
            symbol, 
            model_type=params.model_type,
            data_length=params.data_length,
            epochs=params.epochs, 
            batch_size=params.batch_size, 
            learning_rate=params.learning_rate
        )
        training_status[symbol] = {
            "status": "completed", 
            "metrics": metrics
        }
    except Exception as e:
        logger.error(f"Training failed for {symbol}: {e}")
        training_status[symbol] = {"status": "failed", "error": str(e)}

@router.post("/train")
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Start a training session in the background."""
    if request.symbol in training_status and training_status[request.symbol]["status"] == "training":
        raise HTTPException(status_code=400, detail="Training already in progress for this symbol")
        
    background_tasks.add_task(run_training_task, request.symbol, request)
    return {"message": f"Training started for {request.symbol}", "status": "queued"}

@router.get("/status/{symbol}")
async def get_status(symbol: str):
    """Get training status for a symbol."""
    return training_status.get(symbol, {"status": "not_started"})

@router.get("/predict/{symbol}")
async def get_prediction(symbol: str):
    """Get real-time prediction for a symbol."""
    result = await predictor.predict(symbol)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result
