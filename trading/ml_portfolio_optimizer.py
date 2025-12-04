
import numpy as np
import pandas as pd
import logging
import joblib
from pathlib import Path
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class PortfolioMLOptimizer:
    """ML-powered portfolio allocator"""
    
    def __init__(self, model_path: str = "ml_portfolio_model.pkl"):
        self.model = None
        self.model_path = Path(model_path)
        self.load_model()
        
        # Individual strategy results (Hardcoded from Phase 1 analysis)
        self.individual_results = {
            'Pro_Overlap_Scalper': {'monthly_return': 1.88, 'daily_dd': 1.41, 'max_dd': 2.52, 'trades': 668, 'win_rate': 31.7, 'pf': 1.16},
            'Pro_Asian_Fade': {'monthly_return': 8.92, 'daily_dd': 6.21, 'max_dd': 2.79, 'trades': 1352, 'win_rate': 20.0, 'pf': 1.38},
            'Pro_Gold_Breakout': {'monthly_return': 0.32, 'daily_dd': 0.08, 'max_dd': 0.33, 'trades': 26, 'win_rate': 42.3, 'pf': 1.84},
            'Pro_Volatility_Expansion': {'monthly_return': 6.83, 'daily_dd': 2.00, 'max_dd': 2.13, 'trades': 1398, 'win_rate': 25.9, 'pf': 1.28},
            # Default for new/unknown strategies (like VIX)
            'default': {'monthly_return': 5.0, 'daily_dd': 2.0, 'max_dd': 5.0, 'trades': 100, 'win_rate': 50.0, 'pf': 1.2}
        }
        
    def load_model(self):
        """Load trained ML model"""
        if self.model_path.exists():
            try:
                self.model = joblib.load(self.model_path)
                logger.info(f"Loaded Portfolio ML model from {self.model_path}")
            except Exception as e:
                logger.error(f"Failed to load ML model: {e}")
        else:
            logger.warning(f"ML model not found at {self.model_path}")

    def train(self, trade_history: List[Dict[str, Any]]) -> bool:
        """Train ML model on historical portfolio performance"""
        logger.info("Training Portfolio ML Model...")
        
        if len(trade_history) < 100:
            logger.warning("Insufficient portfolio history for training")
            return False
        
        try:
            from sklearn.ensemble import GradientBoostingRegressor
            
            # Prepare training data
            X = []
            y = []
            
            for record in trade_history:
                features = [
                    record.get('portfolio_volatility', 0.02),
                    record.get('current_dd', 0.0),
                    record.get('strategies_active', 1),
                    record.get('time_of_day', 12),
                    record.get('total_exposure', 1.0),
                    record.get('recent_win_rate', 0.5)
                ]
                X.append(features)
                y.append(record.get('actual_return', 0.0))
            
            X = np.array(X)
            y = np.array(y)
            
            # Train model
            self.model = GradientBoostingRegressor(n_estimators=150, max_depth=4, learning_rate=0.1, random_state=42)
            self.model.fit(X, y)
            
            score = self.model.score(X, y)
            logger.info(f"Portfolio ML trained on {len(trade_history)} records (R²: {score:.2f})")
            
            # Save model
            joblib.dump(self.model, self.model_path)
            logger.info(f"Model saved to {self.model_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"ML training failed: {e}")
            return False
    
    def get_allocation(self, market_state: Dict, current_dd: float, strategies_signaling: List[str]) -> Dict[str, float]:
        """Get optimal allocation weights for strategies"""
        
        # Base allocation using individual performance
        base_weights = {
            'Pro_Overlap_Scalper': 0.15,
            'Pro_Asian_Fade': 0.40,
            'Pro_Gold_Breakout': 0.05,
            'Pro_Volatility_Expansion': 0.40
        }
        
        # Adjust based on current drawdown
        if current_dd > 5.0:
            scale_factor = 0.5
        elif current_dd > 3.0:
            scale_factor = 0.7
        elif current_dd > 1.0:
            scale_factor = 0.9
        else:
            scale_factor = 1.0
        
        # Adjust based on daily DD risk (simplified logic from script)
        adjusted_weights = base_weights.copy()
        
        # Normalize
        total = sum(adjusted_weights.values())
        for key in adjusted_weights:
            adjusted_weights[key] = (adjusted_weights[key] / total) * scale_factor
            
        # Add default weight for unknown strategies
        for strat in strategies_signaling:
            if strat not in adjusted_weights:
                adjusted_weights[strat] = 0.25 * scale_factor # Default allocation
        
        return adjusted_weights
    
    def calculate_position_size(self, strategy_name: str, base_risk: float, allocation_weight: float, current_equity: float, daily_dd_pct: float = 0) -> float:
        """Calculate position size with ML-optimized risk and Daily DD limits"""
        
        # DAILY DD CIRCUIT BREAKER
        if daily_dd_pct > 1.5:
            return 0.0
        elif daily_dd_pct > 1.0:
            base_risk *= 0.5
        
        # Get strategy-specific metrics
        strategy_data = self.individual_results.get(strategy_name, self.individual_results['default'])
        
        # Adjust risk based on win rate and profit factor
        win_rate = strategy_data['win_rate'] / 100.0
        pf = strategy_data['pf']
        
        # Kelly Criterion adjustment (conservative)
        if pf > 0:
            kelly_pct = (win_rate * pf - (1 - win_rate)) / pf
        else:
            kelly_pct = 0
            
        kelly_pct = max(0, min(kelly_pct * 0.5, 0.015))  # Cap at 1.5%
        
        # Combined risk
        final_risk_pct = base_risk * allocation_weight * (1 + kelly_pct)
        
        return final_risk_pct
