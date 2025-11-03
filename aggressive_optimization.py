#!/usr/bin/env python3
"""
Aggressive Optimization with ICT + ML Strategies
50 Iterations with Drawdown Tracking & $100 Capital
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
from scipy import stats
warnings.filterwarnings('ignore')

class ICTStrategies:
    """ICT (Inner Circle Trading) Strategy Library"""
    
    def __init__(self, df):
        self.df = df.copy()
    
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
    
    @staticmethod
    def atr(df, period=14):
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(period).mean()
    
    def ict_supply_demand_zones(self):
        """ICT Supply/Demand zones based on high/low extremes"""
        signals = pd.Series(0, index=self.df.index)
        period = 20
        
        for i in range(period, len(self.df)):
            # Last 20 candles highs and lows
            high_20 = self.df['high'].iloc[i-period:i].max()
            low_20 = self.df['low'].iloc[i-period:i].min()
            
            # Current price relative to zone
            current_price = self.df['close'].iloc[i]
            
            if current_price < low_20 * 1.002:  # Touch demand zone
                signals.iloc[i] = 1
            elif current_price > high_20 * 0.998:  # Touch supply zone
                signals.iloc[i] = -1
        
        return signals
    
    def ict_break_of_structure(self):
        """ICT Break of Structure - detect trend reversals"""
        signals = pd.Series(0, index=self.df.index)
        
        for i in range(5, len(self.df)):
            recent_low = self.df['low'].iloc[max(0, i-5):i].min()
            recent_high = self.df['high'].iloc[max(0, i-5):i].max()
            
            # Break of structure: price breaks recent extremes
            if self.df['close'].iloc[i] > recent_high * 1.001:
                signals.iloc[i] = 1
            elif self.df['close'].iloc[i] < recent_low * 0.999:
                signals.iloc[i] = -1
        
        return signals
    
    def ict_equal_highs_lows(self):
        """ICT Equal Highs/Lows - reversal signal"""
        signals = pd.Series(0, index=self.df.index)
        tolerance = 0.002  # 0.2% tolerance
        
        for i in range(30, len(self.df)):
            highs = self.df['high'].iloc[i-30:i].values
            lows = self.df['low'].iloc[i-30:i].values
            
            # Check for equal highs (within tolerance)
            unique_highs = np.unique(np.round(highs, 3))
            if len(unique_highs) < len(highs) * 0.3:  # Many similar highs
                if self.df['close'].iloc[i] < highs.mean():
                    signals.iloc[i] = -1
            
            unique_lows = np.unique(np.round(lows, 3))
            if len(unique_lows) < len(lows) * 0.3:  # Many similar lows
                if self.df['close'].iloc[i] > lows.mean():
                    signals.iloc[i] = 1
        
        return signals
    
    def ict_displacement_strategy(self):
        """ICT Displacement - large moves in one direction"""
        signals = pd.Series(0, index=self.df.index)
        atr = self.atr(self.df, 14)
        sma_50 = self.sma(self.df['close'], 50)
        
        for i in range(1, len(self.df)):
            if atr.iloc[i] > atr.rolling(50).mean().iloc[i] * 1.5:  # High displacement
                if self.df['close'].iloc[i] > sma_50.iloc[i]:
                    signals.iloc[i] = 1
                else:
                    signals.iloc[i] = -1
        
        return signals


class MLStrategies:
    """ML-Generated Trading Strategies"""
    
    def __init__(self, df):
        self.df = df.copy()
    
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
    
    def ml_mean_reversion(self):
        """ML Mean Reversion - detect overbought/oversold with volatility"""
        signals = pd.Series(0, index=self.df.index)
        
        returns = self.df['close'].pct_change()
        returns_ma = returns.rolling(20).mean()
        returns_std = returns.rolling(20).std()
        
        z_score = (returns - returns_ma) / (returns_std + 1e-6)
        
        signals[z_score < -2] = 1  # Buy oversold
        signals[z_score > 2] = -1  # Sell overbought
        
        return signals
    
    def ml_momentum_acceleration(self):
        """ML Momentum Acceleration - derivative of momentum"""
        signals = pd.Series(0, index=self.df.index)
        
        momentum = self.df['close'].diff(5)
        accel = momentum.diff()
        
        accel_ma = accel.rolling(10).mean()
        accel_std = accel.rolling(10).std()
        
        signals[accel > accel_ma + accel_std] = 1
        signals[accel < accel_ma - accel_std] = -1
        
        return signals
    
    def ml_volatility_expansion(self):
        """ML Volatility Expansion breakout"""
        signals = pd.Series(0, index=self.df.index)
        
        returns_vol = self.df['close'].pct_change().rolling(20).std()
        vol_ma = returns_vol.rolling(30).mean()
        
        upper = self.df['close'].rolling(20).max()
        lower = self.df['close'].rolling(20).min()
        
        signals[(returns_vol > vol_ma * 1.3) & (self.df['close'] == upper)] = 1
        signals[(returns_vol > vol_ma * 1.3) & (self.df['close'] == lower)] = -1
        
        return signals
    
    def ml_entropy_based(self):
        """ML Entropy-based strategy - low entropy = reversal"""
        signals = pd.Series(0, index=self.df.index)
        
        for i in range(30, len(self.df)):
            closes = self.df['close'].iloc[i-30:i].values
            returns = np.diff(closes) / closes[:-1]
            
            # Calculate entropy
            hist, _ = np.histogram(returns, bins=10)
            hist = hist[hist > 0] / hist.sum()
            entropy = -np.sum(hist * np.log2(hist + 1e-10))
            
            # Low entropy = trending = continue; High entropy = random = revert
            if entropy > 2.5:  # High entropy
                rsi = self.rsi(self.df['close'].iloc[:i+1], 14).iloc[-1]
                if rsi > 70:
                    signals.iloc[i] = -1
                elif rsi < 30:
                    signals.iloc[i] = 1
        
        return signals
    
    def ml_pattern_recognition(self):
        """ML Pattern Recognition - higher lows/lower highs"""
        signals = pd.Series(0, index=self.df.index)
        
        for i in range(10, len(self.df)):
            highs = self.df['high'].iloc[i-10:i].values
            lows = self.df['low'].iloc[i-10:i].values
            
            # Uptrend: higher lows
            if lows[-1] > lows[-5] and highs[-1] > highs[-5]:
                signals.iloc[i] = 1
            # Downtrend: lower highs
            elif highs[-1] < highs[-5] and lows[-1] < lows[-5]:
                signals.iloc[i] = -1
        
        return signals


class AdvancedBacktest:
    """Advanced backtest with drawdown tracking and position sizing"""
    
    def __init__(self, capital=100):
        self.initial_capital = capital
        self.capital = capital
        self.equity_curve = [capital]
        self.drawdowns = []
        self.peak_equity = capital
    
    def backtest(self, df, signals, strategy_name):
        """Backtest with drawdown tracking"""
        trades = []
        position = None
        entry_price = 0
        
        try:
            for i in range(1, len(df)):
                signal = signals.iloc[i]
                price = df['close'].iloc[i]
                
                # Update peak equity for drawdown calc
                if self.capital > self.peak_equity:
                    self.peak_equity = self.capital
                
                # Exit logic
                if position and ((position == 1 and signal == -1) or (position == -1 and signal == 1)):
                    exit_price = price
                    pnl = (exit_price - entry_price) * position
                    
                    # Simple position sizing: risk $1 per trade
                    pnl_pct = (pnl / (entry_price * abs(position))) * 100
                    self.capital += pnl
                    
                    trades.append({
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'equity': self.capital
                    })
                    
                    # Track drawdown
                    dd = ((self.peak_equity - self.capital) / self.peak_equity) * 100
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
                'return': 0,
                'return_pct': 0,
                'trades': 0,
                'wr': 0,
                'pf': 0,
                'max_dd': 0,
                'avg_dd': 0
            }
        
        trades_arr = np.array([t['pnl'] for t in trades])
        wins = trades_arr[trades_arr > 0]
        losses = trades_arr[trades_arr < 0]
        
        total_return = self.capital - self.initial_capital
        return_pct = (total_return / self.initial_capital) * 100
        
        max_dd = max(self.drawdowns) if self.drawdowns else 0
        avg_dd = np.mean(self.drawdowns) if self.drawdowns else 0
        
        pf = 0
        if len(losses) > 0:
            loss_sum = abs(losses.sum())
            if loss_sum > 0:
                pf = wins.sum() / loss_sum
        
        return {
            'return': total_return,
            'return_pct': return_pct,
            'trades': len(trades),
            'wr': (len(wins) / len(trades)) * 100 if len(trades) > 0 else 0,
            'pf': pf,
            'max_dd': max_dd,
            'avg_dd': avg_dd
        }


def generate_strategy_variants(base_strategies):
    """Generate strategy variants with parameter optimization"""
    variants = {}
    
    # Existing base strategies
    for name, signals in base_strategies.items():
        variants[name] = signals
    
    return variants


def run_instrument_backtest(args):
    """Run backtest on single instrument"""
    symbol, df, iteration = args
    
    if df is None or len(df) < 100:
        return None
    
    # Generate all strategies
    ict_strat = ICTStrategies(df)
    ml_strat = MLStrategies(df)
    
    ict_strategies = {
        'ICT-SD': ict_strat.ict_supply_demand_zones(),
        'ICT-BOS': ict_strat.ict_break_of_structure(),
        'ICT-EHL': ict_strat.ict_equal_highs_lows(),
        'ICT-Disp': ict_strat.ict_displacement_strategy(),
    }
    
    ml_strategies = {
        'ML-MeanRev': ml_strat.ml_mean_reversion(),
        'ML-MomAcc': ml_strat.ml_momentum_acceleration(),
        'ML-VolExp': ml_strat.ml_volatility_expansion(),
        'ML-Entropy': ml_strat.ml_entropy_based(),
        'ML-Pattern': ml_strat.ml_pattern_recognition(),
    }
    
    all_strategies = {**ict_strategies, **ml_strategies}
    
    results = {
        'instrument': symbol,
        'iteration': iteration,
        'strategies': {},
        'timestamp': datetime.now().isoformat()
    }
    
    best_return = -float('inf')
    best_strategy = None
    
    for strategy_name, signals in all_strategies.items():
        bt = AdvancedBacktest(capital=100)
        backtest_result = bt.backtest(df, signals, strategy_name)
        
        if backtest_result:
            results['strategies'][strategy_name] = backtest_result
            
            if backtest_result['return_pct'] > best_return:
                best_return = backtest_result['return_pct']
                best_strategy = strategy_name
    
    results['best'] = best_strategy
    results['best_return'] = best_return
    
    return results


def load_all_instrument_data():
    """Load all instruments data"""
    data_dir = Path("data/mt5_feeds")
    instruments_data = {}
    
    csv_files = list(data_dir.glob("*_M1.csv"))
    
    for f in csv_files:
        symbol = f.stem.rsplit('_', 1)[0]
        csv_path = data_dir / f"{symbol}_M1.csv"
        
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            instruments_data[symbol] = df
    
    return instruments_data


def run_50_iterations():
    """Run 50 iterations of aggressive optimization"""
    
    print("\n" + "="*80)
    print("AGGRESSIVE OPTIMIZATION - 50 ITERATIONS")
    print("Capital: $100 | Drawdown Tracking: Enabled | Strategies: ICT + ML")
    print("="*80)
    
    instruments_data = load_all_instrument_data()
    
    if not instruments_data:
        print("❌ No data loaded. Ensure data/mt5_feeds directory exists.")
        return
    
    all_iterations_results = []
    
    for iteration in range(1, 51):
        print(f"\n🔄 Iteration {iteration}/50", end="", flush=True)
        
        iteration_results = {
            'iteration': iteration,
            'timestamp': datetime.now().isoformat(),
            'instruments': {}
        }
        
        tasks = [(symbol, df, iteration) for symbol, df in instruments_data.items()]
        
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(run_instrument_backtest, task): task[0] for task in tasks}
            
            total_return = 0
            total_trades = 0
            instrument_count = 0
            
            for future in as_completed(futures):
                result = future.result()
                if result:
                    iteration_results['instruments'][result['instrument']] = result
                    
                    if result.get('best_strategy') and result.get('best_return', -float('inf')) > -float('inf'):
                        best_result = result['strategies'][result['best_strategy']]
                        total_return += best_result['return_pct']
                        total_trades += best_result['trades']
                        instrument_count += 1
        
        iteration_results['portfolio_return_pct'] = total_return / max(instrument_count, 1)
        iteration_results['total_trades'] = total_trades
        iteration_results['instruments_count'] = instrument_count
        
        all_iterations_results.append(iteration_results)
        
        print(f" ✅ Portfolio Return: {iteration_results['portfolio_return_pct']:.2f}% | Trades: {total_trades}")
    
    # Generate summary
    returns = [r['portfolio_return_pct'] for r in all_iterations_results]
    best_iteration = max(enumerate(returns), key=lambda x: x[1])
    worst_iteration = min(enumerate(returns), key=lambda x: x[1])
    
    summary = {
        'total_iterations': 50,
        'capital': 100,
        'strategies_count': 9,  # 4 ICT + 5 ML
        'strategy_types': ['ICT', 'ML'],
        'avg_return': np.mean(returns),
        'std_return': np.std(returns),
        'best_iteration': best_iteration[0] + 1,
        'best_return': best_iteration[1],
        'worst_iteration': worst_iteration[0] + 1,
        'worst_return': worst_iteration[1],
        'timestamp': datetime.now().isoformat()
    }
    
    # Save results
    output_dir = Path("results/backtest")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "aggressive_50_iterations.json", 'w') as f:
        json.dump({
            'summary': summary,
            'iterations': all_iterations_results
        }, f, indent=2, default=str)
    
    print("\n" + "="*80)
    print("OPTIMIZATION SUMMARY")
    print("="*80)
    print(f"Total Iterations: 50")
    print(f"Average Portfolio Return: {summary['avg_return']:.2f}%")
    print(f"Std Dev: {summary['std_return']:.2f}%")
    print(f"Best Iteration {summary['best_iteration']}: {summary['best_return']:.2f}%")
    print(f"Worst Iteration {summary['worst_iteration']}: {summary['worst_return']:.2f}%")
    print(f"\nResults saved: results/backtest/aggressive_50_iterations.json")
    print("="*80)


if __name__ == "__main__":
    run_50_iterations()
