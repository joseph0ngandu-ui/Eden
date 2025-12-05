
import numpy as np
import pandas as pd
import logging
import joblib
from pathlib import Path
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class PortfolioMLOptimizer:
    """ML-powered portfolio allocator - Tuned for DD Control"""
    
    # RISK LIMITS (Tuned for team synergy)
    MAX_DAILY_DD = 4.5   # Target: < 4.5%
    MAX_OVERALL_DD = 9.5 # Target: < 9.5%
    
    def __init__(self, model_path: str = "ml_portfolio_model.pkl"):
        self.model = None
        self.model_path = Path(model_path)
        self.load_model()
        
        # Individual strategy results (Updated from actual backtest 2025-12-05)
        # Strategies are categorized: KEEP (positive PF, reasonable DD), DISABLE (negative returns or extreme DD)
        self.individual_results = {
            # KEEP - Good performers with manageable risk
            'Pro_Asian_Fade': {'monthly_return': 87.99, 'daily_dd': 6.21, 'max_dd': 18.71, 'trades': 1342, 'win_rate': 20.5, 'pf': 1.31, 'status': 'KEEP'},
            'Pro_Mean_Reversion': {'monthly_return': 40.32, 'daily_dd': 3.0, 'max_dd': 27.35, 'trades': 1579, 'win_rate': 36.0, 'pf': 1.12, 'status': 'KEEP'},
            'Pro_Overlap_Scalper': {'monthly_return': 15.33, 'daily_dd': 2.0, 'max_dd': 23.51, 'trades': 640, 'win_rate': 30.6, 'pf': 1.10, 'status': 'KEEP'},
            'Pro_Gold_Breakout': {'monthly_return': 3.03, 'daily_dd': 0.5, 'max_dd': 4.43, 'trades': 26, 'win_rate': 38.5, 'pf': 1.57, 'status': 'KEEP'},
            'Pro_Trend_Follower': {'monthly_return': 3.0, 'daily_dd': 1.5, 'max_dd': 44.34, 'trades': 2247, 'win_rate': 33.5, 'pf': 1.01, 'status': 'LOW_ALLOC'},
            # DISABLED - Negative returns or poor PF
            'Pro_RSI_Momentum': {'monthly_return': -15.67, 'daily_dd': 5.0, 'max_dd': 87.60, 'trades': 2251, 'win_rate': 24.5, 'pf': 0.97, 'status': 'DISABLED'},
            'Pro_Volatility_Expansion': {'monthly_return': -21.89, 'daily_dd': 5.0, 'max_dd': 70.32, 'trades': 744, 'win_rate': 24.9, 'pf': 0.88, 'status': 'DISABLED'},
            # Default for new/unknown strategies
            'default': {'monthly_return': 5.0, 'daily_dd': 2.0, 'max_dd': 5.0, 'trades': 100, 'win_rate': 50.0, 'pf': 1.2, 'status': 'DEFAULT'}
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
        """Get optimal allocation weights for strategies - Tuned for DD Control"""
        
        # Base allocation weights - ASIAN FADE SOLO
        base_weights = {
            'Pro_Asian_Fade': 1.00,        # 114% monthly 6-mo backtest
            # DISABLED:
            'Pro_Mean_Reversion': 0.0,
            'Pro_Overlap_Scalper': 0.0,
            'Pro_Gold_Breakout': 0.0,
            'Pro_Trend_Follower': 0.0, 
            'Pro_RSI_Momentum': 0.0,
            'Pro_Volatility_Expansion': 0.0
        }
        
        # MAX DD CIRCUIT BREAKER (less aggressive to preserve returns)
        if current_dd > 9.0:
            scale_factor = 0.05  # Near limit
        elif current_dd > 8.0:
            scale_factor = 0.20  # Danger zone
        elif current_dd > 6.0:
            scale_factor = 0.50  # Caution
        elif current_dd > 4.0:
            scale_factor = 0.75  # Light
        else:
            scale_factor = 1.0   # Full allocation
        
        adjusted_weights = base_weights.copy()
        
        # Normalize and scale
        total = sum(adjusted_weights.values())
        if total > 0:
            for key in adjusted_weights:
                adjusted_weights[key] = (adjusted_weights[key] / total) * scale_factor
            
        # Add default weight for unknown strategies
        for strat in strategies_signaling:
            if strat not in adjusted_weights:
                adjusted_weights[strat] = 0.20 * scale_factor
        
        return adjusted_weights
    
    def calculate_position_size(self, strategy_name: str, base_risk: float, allocation_weight: float, current_equity: float, daily_dd_pct: float = 0) -> float:
        """Calculate position size with ML-optimized risk and Daily DD limits"""
        
        # Check if strategy is disabled
        strategy_data = self.individual_results.get(strategy_name, self.individual_results['default'])
        if strategy_data.get('status') == 'DISABLED':
            return 0.0
        
        # DAILY DD CIRCUIT BREAKER (tuned for 4.5% limit)
        if daily_dd_pct >= 4.0:
            return 0.0  # HARD STOP at 4.0%
        elif daily_dd_pct > 3.0:
            base_risk *= 0.25  # Critical zone (75-100% of limit)
        elif daily_dd_pct > 2.0:
            base_risk *= 0.50  # Caution zone (50-75% of limit)
        elif daily_dd_pct > 1.0:
            base_risk *= 0.75  # Light caution (25-50% of limit)
        
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
