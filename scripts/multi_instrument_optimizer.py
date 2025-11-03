#!/usr/bin/env python3
"""
Multi-Instrument Optimization and Backtesting Engine
Runs parallel backtests with Optuna optimization for all instruments and strategies
"""

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np

try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: Optuna not installed. Install with: pip install optuna")


class BacktestEngine:
    """Simple backtesting engine for strategy evaluation"""
    
    def __init__(self, data: pd.DataFrame, config: Dict):
        self.data = data
        self.config = config
        self.trades = []
        
    def run(self, params: Dict) -> Dict[str, float]:
        """Run backtest with given parameters"""
        # Simplified backtest logic - replace with your actual strategy
        equity_curve = [10000]  # Starting capital
        
        # Example: Simple moving average crossover
        fast_ma = self.data['close'].rolling(params.get('fast_period', 20)).mean()
        slow_ma = self.data['close'].rolling(params.get('slow_period', 50)).mean()
        
        position = 0
        for i in range(len(self.data)):
            if i < params.get('slow_period', 50):
                equity_curve.append(equity_curve[-1])
                continue
                
            # Entry logic
            if position == 0 and fast_ma.iloc[i] > slow_ma.iloc[i]:
                position = 1
                entry_price = self.data['close'].iloc[i]
            elif position == 1 and fast_ma.iloc[i] < slow_ma.iloc[i]:
                exit_price = self.data['close'].iloc[i]
                pnl = (exit_price - entry_price) / entry_price * equity_curve[-1]
                equity_curve.append(equity_curve[-1] + pnl)
                position = 0
            else:
                equity_curve.append(equity_curve[-1])
        
        # Calculate metrics
        returns = np.diff(equity_curve) / equity_curve[:-1]
        
        metrics = {
            'total_return': (equity_curve[-1] - equity_curve[0]) / equity_curve[0],
            'sharpe_ratio': np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252),
            'max_drawdown': self._calculate_max_drawdown(equity_curve),
            'win_rate': 0.55,  # Placeholder
            'profit_factor': 1.8,  # Placeholder
            'total_trades': len(self.trades)
        }
        
        return metrics
    
    @staticmethod
    def _calculate_max_drawdown(equity_curve: List[float]) -> float:
        """Calculate maximum drawdown"""
        peak = equity_curve[0]
        max_dd = 0
        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        return max_dd


class MultiInstrumentOptimizer:
    """Optimization engine for multiple instruments and strategies"""
    
    def __init__(self, config_dir: str = "configs/instruments", 
                 data_dir: str = "data",
                 backtest_dir: str = "backtests",
                 parallel_workers: int = 8):
        self.config_dir = Path(config_dir)
        self.data_dir = Path(data_dir)
        self.backtest_dir = Path(backtest_dir)
        self.backtest_dir.mkdir(parents=True, exist_ok=True)
        self.parallel_workers = parallel_workers
        
    def load_config(self, symbol: str) -> Dict:
        """Load configuration for a symbol"""
        config_path = self.config_dir / f"{symbol}.json"
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def load_data(self, symbol: str) -> pd.DataFrame:
        """Load cleaned data for a symbol"""
        data_path = self.data_dir / f"{symbol}_clean.parquet"
        if data_path.exists():
            return pd.read_parquet(data_path)
        else:
            raise FileNotFoundError(f"Data file not found: {data_path}")
    
    def create_objective(self, symbol: str, strategy_name: str, 
                        data: pd.DataFrame, config: Dict):
        """Create Optuna objective function"""
        
        def objective(trial):
            # Define search space based on config
            params = {}
            search_space = config.get('optimization', {}).get('search_space', {})
            
            for param_name, param_range in search_space.items():
                if isinstance(param_range, list) and len(param_range) == 2:
                    if isinstance(param_range[0], int):
                        params[param_name] = trial.suggest_int(param_name, param_range[0], param_range[1])
                    else:
                        params[param_name] = trial.suggest_float(param_name, param_range[0], param_range[1])
            
            # Run backtest
            engine = BacktestEngine(data, config)
            metrics = engine.run(params)
            
            # Objective function: Sharpe with drawdown penalty
            objective_value = metrics['sharpe_ratio'] - (metrics['max_drawdown'] * 2)
            
            return objective_value
        
        return objective
    
    def optimize_strategy(self, symbol: str, strategy_name: str, 
                         n_trials: int = 80) -> Dict:
        """Optimize a single strategy for a symbol"""
        print(f"\n  [{symbol}:{strategy_name}] Starting optimization ({n_trials} trials)...")
        
        # Load data and config
        data = self.load_data(symbol)
        config = self.load_config(symbol)
        
        if not OPTUNA_AVAILABLE:
            print("  Optuna not available, skipping optimization")
            return {'status': 'skipped', 'reason': 'optuna_not_installed'}
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=config.get('meta', {}).get('seed', 42))
        )
        
        # Run optimization
        objective = self.create_objective(symbol, strategy_name, data, config)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        # Get best results
        best_params = study.best_params
        best_value = study.best_value
        
        # Run backtest with best params
        engine = BacktestEngine(data, config)
        best_metrics = engine.run(best_params)
        
        results = {
            'symbol': symbol,
            'strategy': strategy_name,
            'best_params': best_params,
            'best_value': best_value,
            'metrics': best_metrics,
            'n_trials': n_trials,
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"    ✓ Best Sharpe: {best_metrics['sharpe_ratio']:.2f}, "
              f"Return: {best_metrics['total_return']*100:.2f}%, "
              f"MaxDD: {best_metrics['max_drawdown']*100:.2f}%")
        
        return results
    
    def optimize_instrument(self, symbol: str) -> Dict:
        """Optimize all strategies for an instrument"""
        print(f"\n[{symbol}] Optimizing...")
        
        config = self.load_config(symbol)
        results = {
            'symbol': symbol,
            'strategies': {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Check if multiple strategies (XAUUSD) or single (VIX)
        if 'strategies' in config:
            # XAUUSD with multiple strategies
            for strategy_name in config['strategies'].keys():
                result = self.optimize_strategy(symbol, strategy_name, 
                                               n_trials=config.get('optimization', {}).get('trials_initial', 80))
                results['strategies'][strategy_name] = result
        else:
            # VIX with single strategy
            result = self.optimize_strategy(symbol, 'default', 
                                          n_trials=config.get('optimization', {}).get('trials_initial', 80))
            results['strategies']['default'] = result
        
        # Save results
        output_path = self.backtest_dir / f"{symbol}_optimization.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def run_parallel_optimization(self, instruments: List[str] = None):
        """Run optimization for all instruments in parallel"""
        print("\n=== Multi-Instrument Optimization ===\n")
        
        # Load instrument pool if not provided
        if instruments is None:
            pool_path = self.config_dir / "instruments_pool.json"
            with open(pool_path, 'r') as f:
                pool_config = json.load(f)
            instruments = pool_config['instruments']
        
        all_results = {}
        
        # Run optimizations in parallel
        with ProcessPoolExecutor(max_workers=self.parallel_workers) as executor:
            futures = {
                executor.submit(self.optimize_instrument, symbol): symbol 
                for symbol in instruments
            }
            
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    results = future.result()
                    all_results[symbol] = results
                except Exception as e:
                    print(f"  ✗ [{symbol}] Failed: {e}")
                    all_results[symbol] = {'status': 'failed', 'error': str(e)}
        
        # Save consolidated results
        summary_path = self.backtest_dir / "optimization_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n=== Optimization Complete ===")
        print(f"Summary saved to: {summary_path}")
        
        return all_results


if __name__ == "__main__":
    optimizer = MultiInstrumentOptimizer(parallel_workers=8)
    optimizer.run_parallel_optimization()
