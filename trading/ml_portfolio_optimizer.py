
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
        
        # Individual strategy results (Updated from edge diagnostic 2025-12-05)
        # Only strategies with POSITIVE EDGE are enabled
        self.individual_results = {
            # BREAKTHROUGH STRATEGY
            'Pro_SupplyDemand': {'monthly_return': 116.75, 'daily_dd': 2.0, 'max_dd': 8.78, 'trades': 1813, 'win_rate': 55.5, 'pf': 1.87, 'expectancy': 0.387, 'status': 'KEEP'},
            # EDGE POSITIVE - Trade these
            'Pro_Asian_Fade': {'monthly_return': 60.25, 'daily_dd': 6.21, 'max_dd': 49.21, 'trades': 2629, 'win_rate': 20.2, 'pf': 1.17, 'expectancy': 0.137, 'status': 'KEEP'},
            'Pro_Index_Momentum': {'monthly_return': 10.56, 'daily_dd': 2.0, 'max_dd': 21.0, 'trades': 1020, 'win_rate': 31.9, 'pf': 1.09, 'expectancy': 0.062, 'status': 'LOW_ALLOC'},
            # NO EDGE - Disabled (PF < 1.1)
            'Pro_SessionRetest': {'monthly_return': 3.67, 'daily_dd': 2.0, 'max_dd': 22.29, 'trades': 821, 'win_rate': 34.2, 'pf': 1.04, 'expectancy': 0.027, 'status': 'DISABLED'},
            'Pro_Engulfing': {'monthly_return': -6.17, 'daily_dd': 5.0, 'max_dd': 103.45, 'trades': 3137, 'win_rate': 24.7, 'pf': 0.98, 'expectancy': -0.012, 'status': 'DISABLED'},
            'Pro_Power3': {'monthly_return': -6.17, 'daily_dd': 3.0, 'max_dd': 38.24, 'trades': 87, 'win_rate': 11.5, 'pf': 0.52, 'expectancy': -0.425, 'status': 'DISABLED'},
            'Pro_Mean_Reversion': {'monthly_return': 14.53, 'daily_dd': 3.0, 'max_dd': 101.21, 'trades': 3045, 'win_rate': 34.6, 'pf': 1.04, 'expectancy': 0.029, 'status': 'DISABLED'},
            'Pro_Overlap_Scalper': {'monthly_return': 5.0, 'daily_dd': 2.0, 'max_dd': 56.55, 'trades': 1092, 'win_rate': 17.1, 'pf': 1.03, 'expectancy': 0.027, 'status': 'DISABLED'},
            'Pro_Trend_Follower': {'monthly_return': 3.67, 'daily_dd': 1.5, 'max_dd': 59.88, 'trades': 3274, 'win_rate': 25.2, 'pf': 1.01, 'expectancy': 0.007, 'status': 'DISABLED'},
            'Pro_Gold_Breakout': {'monthly_return': -3.17, 'daily_dd': 0.5, 'max_dd': 22.86, 'trades': 79, 'win_rate': 15.2, 'pf': 0.72, 'expectancy': -0.241, 'status': 'DISABLED'},
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
        
        # Base allocation weights - REVERTED (Supply/Demand failed verification)
        # Using verified strategies only
        base_weights = {
            'Pro_Asian_Fade': 0.60,         # 60% - Weak but verified edge
            'Pro_Index_Momentum': 0.40,     # 40% - Diversification
            # FAILED VERIFICATION:
            'Pro_SupplyDemand': 0.0,        # Lookahead bias - not viable
            'Pro_SessionRetest': 0.0,
            'Pro_Engulfing': 0.0,
            'Pro_Power3': 0.0,
            'Pro_Mean_Reversion': 0.0,
            'Pro_Overlap_Scalper': 0.0,
            'Pro_Gold_Breakout': 0.0,
            'Pro_Trend_Follower': 0.0, 
            'Pro_RSI_Momentum': 0.0,
            'Pro_Volatility_Expansion': 0.0
        }
        
        # RELAXED DD CIRCUIT BREAKER (preserve returns)
        if current_dd > 8.5:
            scale_factor = 0.10  # Near 9.5% limit
        elif current_dd > 7.0:
            scale_factor = 0.25  # Danger zone
        elif current_dd > 5.0:
            scale_factor = 0.50  # Caution
        elif current_dd > 3.0:
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
    
    def calculate_kelly_fraction(self, win_rate: float, payoff: float) -> float:
        """Calculate Kelly Fraction: f* = (p(b+1) - 1) / b"""
        if payoff <= 0: return 0.0
        kelly = (win_rate * (payoff + 1) - 1) / payoff
        return max(0.0, kelly) # No negative sizing

    def calculate_position_size(self, strategy_name: str, base_risk: float, allocation_weight: float, current_equity: float, daily_dd_pct: float = 0, recent_trades: List[Dict] = None, current_time: Any = None) -> float:
        """Calculate position size with Kelly Optimization & Streak Cooldown"""
        
        # 1. COOLDOWN LOGIC (Streak Breaker)
        if recent_trades and len(recent_trades) >= 3:
            # Filter for this strategy
            strat_trades = [t for t in recent_trades if t.get('strategy') == strategy_name]
            if len(strat_trades) >= 3:
                # Check outcome of last 3
                last_3 = strat_trades[-3:]
                outcomes = [t.get('pnl', 0) for t in last_3]
                if all(p < 0 for p in outcomes):
                    # 3 Consecutive Losses -> Check time since last
                    last_loss_time = last_3[-1].get('close_time')
                    
                    if last_loss_time and current_time:
                        try:
                            # Parse potential string timestamps
                            t_curr = pd.to_datetime(current_time)
                            t_last = pd.to_datetime(last_loss_time)
                            
                            # COOLDOWN PERIOD: 60 Minutes (or 4 hours?)
                            # Let's say 60 mins for H1/M5? M5 -> 12 bars.
                            if (t_curr - t_last).total_seconds() < 3600:
                                return 0.0 # Cooldown Active
                        except Exception as e:
                            logger.warning(f"Cooldown check error: {e}")
        
        # Check if strategy is disabled
        strategy_data = self.individual_results.get(strategy_name, self.individual_results['default'])
        if strategy_data.get('status') == 'DISABLED':
            return 0.0
        
        # DAILY DD CIRCUIT BREAKER (tuned for 4.5% limit)
        if daily_dd_pct >= 4.0:
            return 0.0  # HARD STOP at 4.0%
        elif daily_dd_pct > 3.0:
            base_risk *= 0.30  # Critical zone
        elif daily_dd_pct > 2.0:
            base_risk *= 0.60  # Caution zone
        
        # 2. KELLY CRITERION SIZING
        # Instead of arbitrary edge boost, use Kelly
        win_rate = strategy_data.get('win_rate', 25.0) / 100.0
        # Estimate payoff ratio from PF and WR? 
        # PF = (AvgWin * nWin) / (AvgLoss * nLoss)
        # PF = b * (p / (1-p)) --> b = PF * (1-p)/p
        pf = strategy_data.get('pf', 1.0)
        p = win_rate
        if p > 0:
            b = pf * (1 - p) / p
        else:
            b = 1.0
            
        kelly_pct = self.calculate_kelly_fraction(p, b)
        
        # Safety: Tenth Kelly (0.10) - Ultra Conservative for < 9.5% DD
        safe_kelly = kelly_pct * 0.10 
        
        # Cap at max risk (e.g. 1%)
        final_risk_pct = min(safe_kelly, 0.01)
        
        # Apply Allocation Weight
        final_risk_pct *= allocation_weight
        
        # Fallback to base_risk if Kelly fails (e.g. erratic stats)
        if final_risk_pct < 0.001: 
             final_risk_pct = base_risk * 0.5 # Conservative fallback
             
        return final_risk_pct
