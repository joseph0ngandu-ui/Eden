#!/usr/bin/env python3
"""
Optuna-Based Optimization Framework for Eden
Target: 100% weekly returns across all 10 instruments
Supports parallel optimization with 500+ trials per instrument
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
from datetime import datetime
from backtest_engine import BacktestEngine, StrategyGenerator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
INSTRUMENTS = ['VIX25', 'VIX50', 'VIX75', 'VIX100', 'StepIndex', 
               'Boom1000', 'Boom500', 'Crash1000', 'Crash500', 'XAUUSD']

STRATEGIES = {
    'VIX25': ['htf_bias', 'fvg', 'bollinger_rsi'],
    'VIX50': ['htf_bias', 'fvg', 'bollinger_rsi'],
    'VIX75': ['htf_bias', 'fvg', 'bollinger_rsi'],
    'VIX100': ['htf_bias', 'fvg', 'bollinger_rsi'],
    'StepIndex': ['htf_bias', 'ema_crossover', 'bollinger_rsi'],
    'Boom1000': ['htf_bias', 'ema_crossover'],
    'Boom500': ['htf_bias', 'ema_crossover'],
    'Crash1000': ['htf_bias', 'ema_crossover'],
    'Crash500': ['htf_bias', 'ema_crossover'],
    'XAUUSD': ['breakout_volume', 'ema_crossover', 'bollinger_rsi']
}

class ParameterSampler:
    """Sample parameters for strategies"""
    
    @staticmethod
    def get_param_ranges() -> Dict:
        """Get parameter ranges for optimization"""
        return {
            'risk_pct': (0.005, 0.05),           # 0.5% to 5% risk per trade
            'atr_multiplier': (1.0, 5.0),         # 1x to 5x ATR for stops
            'rsi_upper': (60, 90),                # RSI overbought threshold
            'rsi_lower': (10, 40),                # RSI oversold threshold
            'bb_std_dev': (1.0, 3.0),             # Bollinger Bands std dev
            'ema_fast': (10, 30),                 # Fast EMA period
            'ema_slow': (40, 100),                # Slow EMA period
            'volume_multiplier': (1.0, 3.0),      # Volume spike multiplier
            'position_scale': (0.5, 2.0),         # Position size scaling
            'profit_target_ratio': (1.5, 4.0)     # Profit target R:R ratio
        }

class InstrumentOptimizer:
    """Optimize strategies for single instrument"""
    
    def __init__(self, symbol: str, df: pd.DataFrame, trials: int = 500):
        self.symbol = symbol
        self.df = df
        self.trials = trials
        self.engine = BacktestEngine(initial_capital=100000)
        self.results = []
        
    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for Optuna optimization"""
        
        try:
            # Sample parameters
            params = {
                'risk_pct': trial.suggest_float('risk_pct', 0.005, 0.05),
                'atr_multiplier': trial.suggest_float('atr_multiplier', 1.0, 5.0),
                'rsi_upper': trial.suggest_int('rsi_upper', 60, 90),
                'rsi_lower': trial.suggest_int('rsi_lower', 10, 40),
                'bb_std_dev': trial.suggest_float('bb_std_dev', 1.0, 3.0),
                'ema_fast': trial.suggest_int('ema_fast', 10, 30),
                'ema_slow': trial.suggest_int('ema_slow', 40, 100),
                'volume_multiplier': trial.suggest_float('volume_multiplier', 1.0, 3.0),
                'position_scale': trial.suggest_float('position_scale', 0.5, 2.0),
                'profit_target_ratio': trial.suggest_float('profit_target_ratio', 1.5, 4.0)
            }
            
            # Get strategy for this instrument
            strategies = STRATEGIES.get(self.symbol, ['htf_bias'])
            strategy = strategies[trial.number % len(strategies)]
            
            # Run backtest
            metrics = self.engine.backtest_instrument(
                self.symbol, 
                self.df, 
                strategy, 
                params
            )
            
            # Calculate composite score (target: 100% weekly return)
            # Score = Return % - (Drawdown % * penalty)
            return_pct = metrics['return_pct']
            max_dd = abs(metrics['max_drawdown']) * 100
            
            # Penalize high drawdowns, reward high returns
            score = return_pct - (max_dd * 0.5)
            
            # Store result
            self.results.append({
                'trial': trial.number,
                'strategy': strategy,
                'params': params,
                'metrics': metrics,
                'score': score
            })
            
            # Pruning based on score
            trial.report(score, trial.number)
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            return score
            
        except Exception as e:
            logger.error(f"Error in objective function: {str(e)}")
            return -float('inf')
    
    def optimize(self) -> Dict:
        """Run optimization"""
        
        print(f"\n🔍 Optimizing {self.symbol}...")
        print(f"   Trials: {self.trials}")
        print(f"   Strategies: {STRATEGIES.get(self.symbol, ['htf_bias'])}")
        
        # Create study
        sampler = TPESampler(seed=42, n_startup_trials=20)
        pruner = MedianPruner(n_startup_trials=20)
        study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            pruner=pruner
        )
        
        # Optimize
        study.optimize(self.objective, n_trials=self.trials, show_progress_bar=True)
        
        # Get best trial
        best_trial = study.best_trial
        
        result = {
            'symbol': self.symbol,
            'total_trials': len(study.trials),
            'best_trial': best_trial.number,
            'best_score': best_trial.value,
            'best_params': best_trial.params,
            'trials_data': self.results
        }
        
        print(f"\n✅ {self.symbol} Optimization Complete")
        print(f"   Best Score: {best_trial.value:.2f}")
        print(f"   Best Return: {best_trial.value:.2f}%")
        
        return result

class PortfolioOptimizer:
    """Optimize portfolio across all instruments"""
    
    def __init__(self, data_dir: Path = Path("data/mt5_feeds")):
        self.data_dir = data_dir
        self.results = {}
        self.portfolio_summary = {}
        
    def load_instrument_data(self) -> Dict[str, pd.DataFrame]:
        """Load all instrument data"""
        all_data = {}
        
        for symbol in INSTRUMENTS:
            csv_file = self.data_dir / f"{symbol}_1M.csv"
            if csv_file.exists():
                df = pd.read_csv(csv_file)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp').reset_index(drop=True)
                all_data[symbol] = df
                print(f"✅ Loaded {symbol}: {len(df)} candles")
            else:
                print(f"❌ Missing data for {symbol}")
        
        return all_data
    
    def optimize_single_instrument(self, symbol: str, df: pd.DataFrame, trials: int) -> Dict:
        """Optimize single instrument"""
        optimizer = InstrumentOptimizer(symbol, df, trials=trials)
        result = optimizer.optimize()
        self.results[symbol] = result
        return result
    
    def optimize_all_instruments(self, trials: int = 500, max_workers: int = 4) -> Dict:
        """Optimize all instruments in parallel"""
        
        print("\n" + "="*60)
        print("🚀 PORTFOLIO OPTIMIZATION - 100% WEEKLY TARGET")
        print("="*60)
        
        # Load data
        all_data = self.load_instrument_data()
        
        if not all_data:
            print("❌ No data loaded!")
            return {}
        
        print(f"\n📊 Optimizing {len(all_data)} instruments...")
        print(f"   Trials per instrument: {trials}")
        print(f"   Max parallel workers: {max_workers}")
        
        # Parallel optimization
        with ProcessPoolExecutor(max_workers=min(max_workers, len(all_data))) as executor:
            futures = {}
            for symbol, df in all_data.items():
                future = executor.submit(self.optimize_single_instrument, symbol, df, trials)
                futures[future] = symbol
            
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    result = future.result()
                    logger.info(f"✅ {symbol} completed")
                except Exception as e:
                    logger.error(f"❌ {symbol} failed: {str(e)}")
        
        return self.results
    
    def calculate_portfolio_metrics(self) -> Dict:
        """Calculate combined portfolio metrics"""
        
        print("\n" + "="*60)
        print("📈 PORTFOLIO ANALYSIS")
        print("="*60)
        
        total_return = 0
        total_max_dd = 0
        total_trades = 0
        winning_instruments = 0
        
        for symbol, result in self.results.items():
            if 'best_params' in result and 'trials_data' in result:
                best_trial = result['trials_data'][result['best_trial']]
                metrics = best_trial['metrics']
                
                return_pct = metrics['return_pct']
                max_dd = abs(metrics['max_drawdown']) * 100
                
                total_return += return_pct
                total_max_dd += max_dd
                total_trades += metrics['total_trades']
                
                if return_pct > 0:
                    winning_instruments += 1
                
                print(f"\n{symbol}:")
                print(f"  Return: {return_pct:+.2f}%")
                print(f"  Max DD: {max_dd:.2f}%")
                print(f"  Trades: {metrics['total_trades']}")
                print(f"  Win Rate: {metrics['win_rate']:.1%}")
        
        avg_return = total_return / len(self.results) if self.results else 0
        combined_return = total_return  # Sum of all instruments
        
        print("\n" + "="*60)
        print("PORTFOLIO SUMMARY")
        print("="*60)
        print(f"Total Instruments: {len(self.results)}")
        print(f"Winning Instruments: {winning_instruments}/{len(self.results)}")
        print(f"Combined Return: {combined_return:+.2f}%")
        print(f"Average Return per Instrument: {avg_return:+.2f}%")
        print(f"Total Trades: {total_trades}")
        print(f"Combined Max DD: {total_max_dd:.2f}%")
        print("="*60)
        
        self.portfolio_summary = {
            'total_return_pct': float(combined_return),
            'avg_return_pct': float(avg_return),
            'winning_instruments': int(winning_instruments),
            'total_instruments': len(self.results),
            'total_trades': int(total_trades),
            'target_achieved': combined_return >= 100.0,
            'timestamp': datetime.now().isoformat()
        }
        
        return self.portfolio_summary
    
    def save_results(self, output_dir: Path = Path("results/optimization")):
        """Save optimization results"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        with open(output_dir / "optimization_results.json", 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save summary
        with open(output_dir / "portfolio_summary.json", 'w') as f:
            json.dump(self.portfolio_summary, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to {output_dir}/")

def main():
    """Main execution"""
    
    # Optimization configuration
    INITIAL_TRIALS = 100  # Start with 100 trials
    MAX_WORKERS = 4
    
    print("\n" + "="*70)
    print("🎯 EDEN 100% WEEKLY OPTIMIZATION - ROUND 1")
    print("="*70)
    
    optimizer = PortfolioOptimizer()
    
    # Run optimization
    results = optimizer.optimize_all_instruments(trials=INITIAL_TRIALS, max_workers=MAX_WORKERS)
    
    # Analyze results
    portfolio_metrics = optimizer.calculate_portfolio_metrics()
    
    # Save results
    optimizer.save_results()
    
    # Check if target achieved
    if portfolio_metrics['target_achieved']:
        print("\n🎉 TARGET ACHIEVED: 100% Weekly Return!")
    else:
        shortfall = 100.0 - portfolio_metrics['total_return_pct']
        print(f"\n⚠️  Target shortfall: {shortfall:.2f}%")
        print("   Recommendation: Increase trials and re-optimize")

if __name__ == "__main__":
    main()
