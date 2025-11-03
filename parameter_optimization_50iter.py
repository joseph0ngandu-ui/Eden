#!/usr/bin/env python3
"""
50-Iteration Parameter Optimization Engine
- Varies strategy parameters across 50 runs
- Tracks return + drawdown metrics
- Saves best config as default trading engine
- $100 capital per iteration
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class StrategyParameterVariants:
    """Generate parameter variants for each strategy"""
    
    @staticmethod
    def generate_rsi_params():
        """Generate RSI parameter variants"""
        variants = []
        for period in [10, 12, 14, 16, 18, 20]:
            for oversold in [20, 25, 30]:
                for overbought in [70, 75, 80]:
                    variants.append({
                        'period': period,
                        'oversold': oversold,
                        'overbought': overbought,
                        'name': f'RSI-{period}-{oversold}-{overbought}'
                    })
        return variants
    
    @staticmethod
    def generate_ma_params():
        """Generate Moving Average parameter variants"""
        variants = []
        for fast in [8, 10, 12, 14]:
            for slow in [30, 40, 50, 60]:
                for signal in [5, 8, 10]:
                    variants.append({
                        'fast': fast,
                        'slow': slow,
                        'signal': signal,
                        'name': f'MA-{fast}-{slow}-{signal}'
                    })
        return variants
    
    @staticmethod
    def generate_bb_params():
        """Generate Bollinger Bands parameter variants"""
        variants = []
        for period in [15, 18, 20, 22, 25]:
            for std_dev in [1.5, 2.0, 2.5]:
                for rsi_threshold in [20, 25, 30]:
                    variants.append({
                        'period': period,
                        'std_dev': std_dev,
                        'rsi_threshold': rsi_threshold,
                        'name': f'BB-{period}-{std_dev}-{rsi_threshold}'
                    })
        return variants
    
    @staticmethod
    def generate_breakout_params():
        """Generate Breakout parameter variants"""
        variants = []
        for period in [5, 8, 10, 12, 15, 20]:
            for volume_mult in [1.2, 1.5, 2.0]:
                variants.append({
                    'period': period,
                    'volume_mult': volume_mult,
                    'name': f'BO-{period}-{volume_mult}'
                })
        return variants
    
    @staticmethod
    def generate_confluence_params():
        """Generate Confluence parameter variants"""
        variants = []
        for min_signals in [2, 3, 4]:
            for rsi_min in [45, 50, 55]:
                for rsi_max in [45, 50, 55]:
                    if rsi_min < rsi_max:
                        variants.append({
                            'min_signals': min_signals,
                            'rsi_min': rsi_min,
                            'rsi_max': rsi_max,
                            'name': f'CONF-{min_signals}-{rsi_min}-{rsi_max}'
                        })
        return variants


class ParametricBacktest:
    """Backtest with parameterized strategies"""
    
    def __init__(self, df, capital=100):
        self.df = df.copy()
        self.initial_capital = capital
        self.capital = capital
        self.peak_equity = capital
        self.drawdowns = []
        self.trades = []
    
    @staticmethod
    def sma(series, period=20):
        return series.rolling(period).mean()
    
    @staticmethod
    def ema(series, period=20):
        return series.ewm(span=period).mean()
    
    @staticmethod
    def rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def rsi_extreme_signal(self, period=14, oversold=25, overbought=75):
        """Parameterized RSI signal"""
        rsi = self.rsi(self.df['close'], period)
        signals = pd.Series(0, index=self.df.index)
        signals[rsi < oversold] = 1
        signals[rsi > overbought] = -1
        return signals
    
    def ma_crossover_signal(self, fast=12, slow=26, signal_period=9):
        """Parameterized MA crossover"""
        ema_fast = self.ema(self.df['close'], fast)
        ema_slow = self.ema(self.df['close'], slow)
        
        signals = pd.Series(0, index=self.df.index)
        signals[ema_fast > ema_slow] = 1
        signals[ema_fast < ema_slow] = -1
        return signals
    
    def bb_rsi_signal(self, period=20, std_dev=2, rsi_threshold=30):
        """Parameterized Bollinger Bands + RSI"""
        sma = self.sma(self.df['close'], period)
        std = self.df['close'].rolling(period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        
        rsi = self.rsi(self.df['close'], 14)
        
        signals = pd.Series(0, index=self.df.index)
        signals[(self.df['close'] < lower) & (rsi < rsi_threshold)] = 1
        signals[(self.df['close'] > upper) & (rsi > (100 - rsi_threshold))] = -1
        return signals
    
    def breakout_signal(self, period=10, volume_mult=1.2):
        """Parameterized Breakout"""
        high_period = self.df['high'].rolling(period).max()
        low_period = self.df['low'].rolling(period).min()
        vol_ma = self.df['volume'].rolling(20).mean()
        
        signals = pd.Series(0, index=self.df.index)
        signals[(self.df['close'] > high_period) & (self.df['volume'] > vol_ma * volume_mult)] = 1
        signals[(self.df['close'] < low_period) & (self.df['volume'] > vol_ma * volume_mult)] = -1
        return signals
    
    def confluence_signal(self, min_signals=3, rsi_min=50, rsi_max=50):
        """Parameterized Confluence"""
        sma_20 = self.sma(self.df['close'], 20)
        sma_50 = self.sma(self.df['close'], 50)
        rsi = self.rsi(self.df['close'], 14)
        ema_12 = self.ema(self.df['close'], 12)
        ema_26 = self.ema(self.df['close'], 26)
        
        signals = pd.Series(0, index=self.df.index)
        
        # Count aligned signals
        uptrend_signals = (
            (self.df['close'] > sma_20).astype(int) +
            (sma_20 > sma_50).astype(int) +
            (rsi > rsi_min).astype(int) +
            (ema_12 > ema_26).astype(int)
        )
        
        downtrend_signals = (
            (self.df['close'] < sma_20).astype(int) +
            (sma_20 < sma_50).astype(int) +
            (rsi < rsi_max).astype(int) +
            (ema_12 < ema_26).astype(int)
        )
        
        signals[uptrend_signals >= min_signals] = 1
        signals[downtrend_signals >= min_signals] = -1
        return signals
    
    def backtest_with_params(self, signals):
        """Run backtest with given signals"""
        trades = []
        position = None
        entry_price = 0
        
        try:
            for i in range(1, len(self.df)):
                signal = signals.iloc[i]
                price = self.df['close'].iloc[i]
                
                if self.capital > self.peak_equity:
                    self.peak_equity = self.capital
                
                # Exit logic
                if position and ((position == 1 and signal == -1) or (position == -1 and signal == 1)):
                    exit_price = price
                    pnl = (exit_price - entry_price) * position
                    self.capital += pnl
                    
                    trades.append({
                        'pnl': pnl,
                        'pnl_pct': (pnl / (entry_price * abs(position))) * 100
                    })
                    
                    dd = ((self.peak_equity - self.capital) / self.peak_equity) * 100 if self.peak_equity > 0 else 0
                    self.drawdowns.append(dd)
                    
                    position = None
                
                # Entry logic
                if signal != 0 and position is None:
                    entry_price = price
                    position = signal
        except:
            pass
        
        if not trades:
            return {
                'return_pct': 0,
                'trades': 0,
                'wr': 0,
                'max_dd': 0,
                'avg_dd': 0,
                'sharpe_score': 0
            }
        
        trades_arr = np.array([t['pnl'] for t in trades])
        wins = trades_arr[trades_arr > 0]
        losses = trades_arr[trades_arr < 0]
        
        total_return = self.capital - self.initial_capital
        return_pct = (total_return / self.initial_capital) * 100
        
        max_dd = max(self.drawdowns) if self.drawdowns else 0
        avg_dd = np.mean(self.drawdowns) if self.drawdowns else 0
        
        wr = (len(wins) / len(trades)) * 100 if len(trades) > 0 else 0
        
        # Sharpe-like score: return / max_drawdown
        sharpe_score = return_pct / (max_dd + 1e-6) if max_dd > 0 else return_pct
        
        return {
            'return_pct': return_pct,
            'trades': len(trades),
            'wr': wr,
            'max_dd': max_dd,
            'avg_dd': avg_dd,
            'sharpe_score': sharpe_score
        }


def load_instruments():
    """Load all instruments"""
    data_dir = Path("data/mt5_feeds")
    instruments = {}
    
    csv_files = list(data_dir.glob("*_M1.csv"))
    
    for f in csv_files:
        symbol = f.stem.rsplit('_', 1)[0]
        csv_path = data_dir / f"{symbol}_M1.csv"
        
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            instruments[symbol] = df
    
    return instruments


def run_50_parameter_iterations():
    """Run 50 iterations with different parameter combinations"""
    
    print("\n" + "="*100)
    print("50-ITERATION PARAMETER OPTIMIZATION ENGINE")
    print("Testing: RSI, MA Crossover, Bollinger Bands, Breakout, Confluence")
    print("Capital: $100 | Tracking: Return + Drawdown | Saving: Best Config")
    print("="*100)
    
    instruments = load_instruments()
    
    if not instruments:
        print("❌ No instruments found")
        return
    
    print(f"\nLoaded {len(instruments)} instruments\n")
    
    # Generate parameter sets
    rsi_params = StrategyParameterVariants.generate_rsi_params()
    ma_params = StrategyParameterVariants.generate_ma_params()
    bb_params = StrategyParameterVariants.generate_bb_params()
    bo_params = StrategyParameterVariants.generate_breakout_params()
    conf_params = StrategyParameterVariants.generate_confluence_params()
    
    all_params = rsi_params + ma_params + bb_params + bo_params + conf_params
    
    # Select 50 random parameters (one per iteration)
    np.random.seed(42)
    selected_params = np.random.choice(len(all_params), min(50, len(all_params)), replace=False)
    
    iteration_results = []
    
    for iter_idx, param_idx in enumerate(selected_params, 1):
        param_set = all_params[param_idx]
        
        print(f"📊 Iteration {iter_idx}/50 | {param_set['name']}...", end="", flush=True)
        
        iter_data = {
            'iteration': iter_idx,
            'parameters': param_set,
            'instruments': {}
        }
        
        portfolio_metrics = {
            'total_return': 0,
            'total_trades': 0,
            'avg_dd': [],
            'max_dd': [],
            'wr': []
        }
        
        for symbol, df in instruments.items():
            if df is None or len(df) < 100:
                continue
            
            bt = ParametricBacktest(df, capital=100)
            
            # Determine strategy type from parameter name
            strategy_name = param_set['name'].split('-')[0]
            
            if strategy_name == 'RSI':
                signals = bt.rsi_extreme_signal(param_set['period'], param_set['oversold'], param_set['overbought'])
            elif strategy_name == 'MA':
                signals = bt.ma_crossover_signal(param_set['fast'], param_set['slow'], param_set['signal'])
            elif strategy_name == 'BB':
                signals = bt.bb_rsi_signal(param_set['period'], param_set['std_dev'], param_set['rsi_threshold'])
            elif strategy_name == 'BO':
                signals = bt.breakout_signal(param_set['period'], param_set['volume_mult'])
            elif strategy_name == 'CONF':
                signals = bt.confluence_signal(param_set['min_signals'], param_set['rsi_min'], param_set['rsi_max'])
            else:
                continue
            
            result = bt.backtest_with_params(signals)
            iter_data['instruments'][symbol] = result
            
            portfolio_metrics['total_return'] += result['return_pct']
            portfolio_metrics['total_trades'] += result['trades']
            portfolio_metrics['avg_dd'].append(result['avg_dd'])
            portfolio_metrics['max_dd'].append(result['max_dd'])
            portfolio_metrics['wr'].append(result['wr'])
        
        avg_return = portfolio_metrics['total_return'] / len(instruments)
        avg_max_dd = np.mean(portfolio_metrics['max_dd']) if portfolio_metrics['max_dd'] else 0
        avg_wr = np.mean(portfolio_metrics['wr']) if portfolio_metrics['wr'] else 0
        
        # Score: return / (drawdown + 1)
        score = avg_return / (avg_max_dd + 1) if avg_max_dd > 0 else avg_return
        
        iter_data['portfolio_avg_return'] = avg_return
        iter_data['portfolio_avg_dd'] = avg_max_dd
        iter_data['portfolio_avg_wr'] = avg_wr
        iter_data['score'] = score
        
        iteration_results.append(iter_data)
        
        print(f" ✅ Return: {avg_return:+.2f}% | DD: {avg_max_dd:.2f}% | Score: {score:.2f}")
    
    # Find best config
    best_config = max(iteration_results, key=lambda x: x['score'])
    
    print("\n" + "="*100)
    print("OPTIMIZATION COMPLETE")
    print("="*100)
    print(f"\n🏆 BEST CONFIGURATION:")
    print(f"   Iteration: {best_config['iteration']}")
    print(f"   Strategy: {best_config['parameters']['name']}")
    print(f"   Average Return: {best_config['portfolio_avg_return']:+.2f}%")
    print(f"   Average Drawdown: {best_config['portfolio_avg_dd']:.2f}%")
    print(f"   Average Win Rate: {best_config['portfolio_avg_wr']:.2f}%")
    print(f"   Score (Return/DD): {best_config['score']:.4f}")
    
    # Save results
    output_dir = Path("results/backtest")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "parameter_optimization_50iter.json", 'w') as f:
        json.dump({
            'summary': {
                'total_iterations': len(iteration_results),
                'best_iteration': best_config['iteration'],
                'best_parameters': best_config['parameters'],
                'best_score': best_config['score'],
                'timestamp': datetime.now().isoformat()
            },
            'all_iterations': iteration_results
        }, f, indent=2, default=str)
    
    # Save best config as default engine
    best_engine_config = {
        'name': f"Optimized_{best_config['parameters']['name']}",
        'description': f"Auto-generated best config from 50-iteration optimization",
        'version': '2.0',
        'date_optimized': datetime.now().isoformat(),
        'optimization_iteration': best_config['iteration'],
        'optimization_score': best_config['score'],
        
        'account_settings': {
            'starting_cash': 100.0,
            'currency': 'USD',
            'account_type': 'micro'
        },
        
        'strategy_parameters': best_config['parameters'],
        
        'performance_metrics': {
            'avg_return_pct': best_config['portfolio_avg_return'],
            'avg_drawdown_pct': best_config['portfolio_avg_dd'],
            'avg_win_rate_pct': best_config['portfolio_avg_wr']
        },
        
        'instrument_results': best_config['instruments'],
        
        'backtesting': {
            'capital': 100,
            'iterations': 50,
            'method': 'parameter_optimization'
        }
    }
    
    with open(output_dir / "optimal_engine_config.json", 'w') as f:
        json.dump(best_engine_config, f, indent=2, default=str)
    
    print(f"\n✅ Results saved:")
    print(f"   - results/backtest/parameter_optimization_50iter.json")
    print(f"   - results/backtest/optimal_engine_config.json (DEFAULT ENGINE)")
    print("="*100 + "\n")
    
    return best_config


if __name__ == "__main__":
    run_50_parameter_iterations()
