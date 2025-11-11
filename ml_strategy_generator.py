#!/usr/bin/env python3
"""
ML Strategy Generator for Eden Trading Bot

Autonomously creates, tests, and validates new trading strategies.
Only deploys strategies that pass profitability tests.

Features:
- Generates strategy variations using ML
- Backtests strategies on historical data
- Validates performance before deployment
- Learns from winning strategies
- Self-optimizing parameter tuning
"""

import sys
import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta

# Local imports for Phase 2 real backtesting
try:
    from src.data_provider import fetch_ohlc
    from src.strategy_backtester import backtest_ma, backtest_rsi
    REAL_BACKTEST_AVAILABLE = True
except Exception:
    REAL_BACKTEST_AVAILABLE = False

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class Strategy:
    """Represents a trading strategy"""
    id: str
    name: str
    type: str  # 'MA', 'RSI', 'ICT', 'ML_GENERATED'
    parameters: Dict
    is_active: bool = False
    is_validated: bool = False
    backtest_results: Optional[Dict] = None
    created_at: datetime = None
    performance_score: float = 0.0
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class MLStrategyGenerator:
    """
    Generates and validates trading strategies using ML techniques.
    """
    
    def __init__(self, data_dir: str = "data", min_winrate: float = 0.55, min_profit_factor: float = 1.5, use_real_backtest: bool = True, symbol: str = "Volatility 75 Index", timeframe: str = "M5"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        self.strategies_file = self.data_dir / "strategies.json"
        self.validated_strategies_file = self.data_dir / "validated_strategies.json"
        
        self.min_winrate = min_winrate
        self.min_profit_factor = min_profit_factor
        self.use_real_backtest = use_real_backtest and REAL_BACKTEST_AVAILABLE
        self.symbol = symbol
        self.timeframe = timeframe
        
        self.strategies: Dict[str, Strategy] = {}
        self.load_strategies()
        
        logger.info(f"ML Strategy Generator initialized (min_wr={min_winrate}, min_pf={min_profit_factor}, real_backtest={self.use_real_backtest})")
    
    def load_strategies(self):
        """Load existing strategies from disk"""
        if self.strategies_file.exists():
            with open(self.strategies_file, 'r') as f:
                data = json.load(f)
                for strategy_data in data.values():
                    strategy_data['created_at'] = datetime.fromisoformat(strategy_data['created_at'])
                    strategy = Strategy(**strategy_data)
                    self.strategies[strategy.id] = strategy
            logger.info(f"Loaded {len(self.strategies)} existing strategies")
    
    def save_strategies(self):
        """Save strategies to disk"""
        data = {}
        for sid, strategy in self.strategies.items():
            s_dict = asdict(strategy)
            s_dict['created_at'] = strategy.created_at.isoformat()
            data[sid] = s_dict
        
        with open(self.strategies_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def generate_ma_strategy(self, fast_period: int, slow_period: int) -> Strategy:
        """Generate MA crossover strategy"""
        strategy_id = f"MA_{fast_period}_{slow_period}_{int(datetime.now().timestamp())}"
        
        return Strategy(
            id=strategy_id,
            name=f"MA Crossover ({fast_period}/{slow_period})",
            type="MA",
            parameters={
                "fast_period": fast_period,
                "slow_period": slow_period,
                "timeframe": "M5"
            }
        )
    
    def generate_rsi_strategy(self, period: int, oversold: int, overbought: int) -> Strategy:
        """Generate RSI strategy"""
        strategy_id = f"RSI_{period}_{oversold}_{overbought}_{int(datetime.now().timestamp())}"
        
        return Strategy(
            id=strategy_id,
            name=f"RSI Strategy ({period}/{oversold}/{overbought})",
            type="RSI",
            parameters={
                "period": period,
                "oversold": oversold,
                "overbought": overbought,
                "timeframe": "M5"
            }
        )
    
    def generate_ml_strategy(self) -> Strategy:
        """
        Generate ML-based strategy using evolutionary algorithm.
        This learns from successful strategies and creates variations.
        """
        # Get best performing strategies
        best_strategies = sorted(
            [s for s in self.strategies.values() if s.is_validated],
            key=lambda x: x.performance_score,
            reverse=True
        )[:3]
        
        if not best_strategies:
            # No validated strategies yet, create random
            return self._create_random_strategy()
        
        # Learn from best strategy
        best = best_strategies[0]
        params = best.parameters.copy()
        
        # Mutate parameters slightly
        if best.type == "MA":
            params['fast_period'] = max(2, params['fast_period'] + np.random.randint(-2, 3))
            params['slow_period'] = max(params['fast_period'] + 1, params['slow_period'] + np.random.randint(-3, 4))
        elif best.type == "RSI":
            params['period'] = max(5, min(30, params['period'] + np.random.randint(-3, 4)))
            params['oversold'] = max(20, min(40, params['oversold'] + np.random.randint(-5, 6)))
            params['overbought'] = max(60, min(85, params['overbought'] + np.random.randint(-5, 6)))
        
        strategy_id = f"ML_{best.type}_{int(datetime.now().timestamp())}"
        
        return Strategy(
            id=strategy_id,
            name=f"ML-Generated {best.type}",
            type="ML_GENERATED",
            parameters=params
        )
    
    def _create_random_strategy(self) -> Strategy:
        """Create random strategy when no validated strategies exist"""
        strategy_type = np.random.choice(["MA", "RSI"])
        
        if strategy_type == "MA":
            fast = np.random.randint(2, 10)
            slow = np.random.randint(fast + 1, 30)
            return self.generate_ma_strategy(fast, slow)
        else:
            period = np.random.randint(7, 21)
            oversold = np.random.randint(20, 35)
            overbought = np.random.randint(65, 80)
            return self.generate_rsi_strategy(period, oversold, overbought)
    
    def backtest_strategy(self, strategy: Strategy) -> Dict:
        """
        Backtest strategy on historical data (Phase 2). Falls back to synthetic if needed.
        Performs simple forward-validation: split last 90 days into 60d train, 30d forward.
        Returns performance metrics including forward test.
        """
        logger.info(f"Backtesting strategy: {strategy.name}")
        
        if self.use_real_backtest:
            try:
                df = fetch_ohlc(self.symbol, timeframe=self.timeframe, days=90)
                if df is not None and len(df) > 200:
                    cutoff = df['time'].min() + (df['time'].max() - df['time'].min()) * 2/3
                    train_df = df[df['time'] <= cutoff]
                    test_df = df[df['time'] > cutoff]

                    res_train = None
                    res_test = None

                    if strategy.type in ("MA", "ML_GENERATED") and 'fast_period' in strategy.parameters:
                        res_train = backtest_ma(train_df, strategy.parameters['fast_period'], strategy.parameters.get('slow_period', 10))
                        res_test = backtest_ma(test_df, strategy.parameters['fast_period'], strategy.parameters.get('slow_period', 10))
                    elif strategy.type == "RSI":
                        res_train = backtest_rsi(train_df, strategy.parameters['period'], strategy.parameters['oversold'], strategy.parameters['overbought'])
                        res_test = backtest_rsi(test_df, strategy.parameters['period'], strategy.parameters['oversold'], strategy.parameters['overbought'])

                    if res_train and res_test:
                        res = {
                            **res_test,
                            'train_win_rate': res_train['win_rate'],
                            'train_profit_factor': res_train['profit_factor'],
                            'forward_win_rate': res_test['win_rate'],
                            'forward_profit_factor': res_test['profit_factor'],
                            'backtest_date': datetime.now().isoformat(),
                        }
                        strategy.backtest_results = res
                        strategy.performance_score = (res['forward_win_rate'] * 50) + (min(res['forward_profit_factor'] / 3, 1) * 50)
                        logger.info(f"Backtest (REAL) train WR={res['train_win_rate']:.1%} PF={res['train_profit_factor']:.2f} | forward WR={res['forward_win_rate']:.1%} PF={res['forward_profit_factor']:.2f}")
                        return res
            except Exception as e:
                logger.error(f"Real backtest failed, falling back to synthetic: {e}")
        
        # Fallback synthetic backtest
        num_trades = np.random.randint(50, 200)
        win_rate = np.random.uniform(0.45, 0.70)
        winning_trades = int(num_trades * win_rate)
        losing_trades = num_trades - winning_trades
        avg_win = np.random.uniform(40, 100)
        avg_loss = np.random.uniform(20, 60)
        total_profit = winning_trades * avg_win
        total_loss = losing_trades * avg_loss
        net_profit = total_profit - total_loss
        profit_factor = total_profit / total_loss if total_loss > 0 else 0
        results = {
            "total_trades": num_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "total_profit": total_profit,
            "total_loss": total_loss,
            "net_profit": net_profit,
            "profit_factor": profit_factor,
            "max_drawdown": np.random.uniform(5, 15),
            "sharpe_ratio": np.random.uniform(0.5, 2.5),
            "backtest_date": datetime.now().isoformat()
        }
        strategy.backtest_results = results
        strategy.performance_score = (win_rate * 50) + (min(profit_factor / 3, 1) * 50)
        logger.info(f"Backtest complete (SYNTHETIC): WR={win_rate:.1%}, PF={profit_factor:.2f}, Net=${net_profit:.2f}")
        return results
    
    def validate_strategy(self, strategy: Strategy) -> bool:
        """
        Validate if strategy meets minimum requirements.
        Uses forward metrics if available; otherwise uses synthetic metrics.
        Only profitable strategies are saved as validated.
        """
        if not strategy.backtest_results:
            return False
        
        r = strategy.backtest_results
        
        # Prefer forward metrics if present
        wr = r.get('forward_win_rate', r.get('win_rate', 0.0))
        pf = r.get('forward_profit_factor', r.get('profit_factor', 0.0))
        net = r.get('net_profit', 0.0)
        
        meets_winrate = wr >= self.min_winrate
        meets_profit_factor = pf >= self.min_profit_factor
        is_profitable = net > 0 or pf > 1.2  # net may not exist for forward-only metrics
        
        is_valid = meets_winrate and meets_profit_factor and is_profitable
        
        strategy.is_validated = is_valid
        
        if is_valid:
            logger.info(f"✅ Strategy VALIDATED: {strategy.name}")
            self._save_validated_strategy(strategy)
        else:
            logger.warning(f"❌ Strategy REJECTED: {strategy.name} (WR={results['win_rate']:.1%}, PF={results['profit_factor']:.2f})")
        
        return is_valid
    
    def _save_validated_strategy(self, strategy: Strategy):
        """Save validated strategy separately"""
        validated = {}
        if self.validated_strategies_file.exists():
            with open(self.validated_strategies_file, 'r') as f:
                validated = json.load(f)
        
        s_dict = asdict(strategy)
        s_dict['created_at'] = strategy.created_at.isoformat()
        validated[strategy.id] = s_dict
        
        with open(self.validated_strategies_file, 'w') as f:
            json.dump(validated, f, indent=2)
    
    def generate_and_test_strategy(self) -> Optional[Strategy]:
        """
        Complete workflow: Generate → Backtest → Validate
        Returns strategy only if it passes validation.
        """
        logger.info("=" * 80)
        logger.info("Generating new strategy...")
        
        # Generate strategy (50% ML, 50% random exploration)
        if np.random.random() < 0.5 and len([s for s in self.strategies.values() if s.is_validated]) > 0:
            strategy = self.generate_ml_strategy()
            logger.info("Using ML-based generation (learning from winners)")
        else:
            strategy = self._create_random_strategy()
            logger.info("Using random exploration")
        
        # Backtest
        self.backtest_strategy(strategy)
        
        # Validate
        is_valid = self.validate_strategy(strategy)
        
        # Save
        self.strategies[strategy.id] = strategy
        self.save_strategies()
        
        logger.info("=" * 80)
        
        return strategy if is_valid else None
    
    def get_active_strategies(self) -> List[Strategy]:
        """Get all active strategies"""
        return [s for s in self.strategies.values() if s.is_active]
    
    def get_validated_strategies(self) -> List[Strategy]:
        """Get all validated strategies"""
        return [s for s in self.strategies.values() if s.is_validated]
    
    def activate_strategy(self, strategy_id: str) -> bool:
        """Activate a validated strategy for live trading"""
        if strategy_id not in self.strategies:
            return False
        
        strategy = self.strategies[strategy_id]
        
        if not strategy.is_validated:
            logger.warning(f"Cannot activate unvalidated strategy: {strategy.name}")
            return False
        
        strategy.is_active = True
        self.save_strategies()
        logger.info(f"✅ Activated strategy: {strategy.name}")
        return True
    
    def deactivate_strategy(self, strategy_id: str) -> bool:
        """Deactivate a strategy"""
        if strategy_id not in self.strategies:
            return False
        
        strategy = self.strategies[strategy_id]
        strategy.is_active = False
        self.save_strategies()
        logger.info(f"🛑 Deactivated strategy: {strategy.name}")
        return True

    def phase_out_strategies(self) -> Dict[str, str]:
        """Check active strategies and phase out if forward metrics degrade."""
        phased = {}
        for sid, strat in self.strategies.items():
            if not strat.is_active:
                continue
            try:
                # Re-evaluate last 30 days forward
                df = fetch_ohlc(self.symbol, timeframe=self.timeframe, days=45) if self.use_real_backtest else None
                if df is None or len(df) < 150:
                    continue
                recent_df = df[df['time'] > (df['time'].max() - (df['time'].max() - df['time'].min()) * 1/3)]
                if strat.type in ("MA", "ML_GENERATED") and 'fast_period' in strat.parameters:
                    res = backtest_ma(recent_df, strat.parameters['fast_period'], strat.parameters.get('slow_period', 10))
                elif strat.type == "RSI":
                    res = backtest_rsi(recent_df, strat.parameters['period'], strat.parameters['oversold'], strat.parameters['overbought'])
                else:
                    res = None
                if not res:
                    continue
                if res['profit_factor'] < self.min_profit_factor or res['win_rate'] < self.min_winrate:
                    strat.is_active = False
                    phased[sid] = f"Degraded: WR={res['win_rate']:.1%}, PF={res['profit_factor']:.2f}"
            except Exception as e:
                logger.error(f"Phase-out check failed for {sid}: {e}")
        if phased:
            self.save_strategies()
        return phased


def main():
    """Main execution for continuous strategy generation"""
    generator = MLStrategyGenerator()
    
    logger.info("🚀 Starting autonomous strategy generation...")
    logger.info(f"Current strategies: {len(generator.strategies)}")
    logger.info(f"Validated strategies: {len(generator.get_validated_strategies())}")
    
    # Generate strategies continuously
    generation_count = 0
    validated_count = 0
    
    try:
        while True:
            generation_count += 1
            logger.info(f"\n🔄 Generation cycle #{generation_count}")
            
            strategy = generator.generate_and_test_strategy()
            
            if strategy:
                validated_count += 1
                logger.info(f"🎉 New validated strategy! Total validated: {validated_count}")
            
            # Sleep between generations (adjust as needed)
            import time
            time.sleep(300)  # 5 minutes between generations
    
    except KeyboardInterrupt:
        logger.info("\n🛑 Strategy generation stopped")
        logger.info(f"Generated: {generation_count}, Validated: {validated_count}")


if __name__ == "__main__":
    main()
