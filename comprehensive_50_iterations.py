#!/usr/bin/env python3
"""
Comprehensive 50-Iteration Optimization
Combines: Base + ICT + ML Strategies
$100 Capital | Drawdown Tracking | All Instruments
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')


class AllStrategies:
    """All trading strategies: Base + ICT + ML"""
    
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
    
    @staticmethod
    def bollinger_bands(series, period=20, std_dev=2):
        sma = series.rolling(period).mean()
        std = series.rolling(period).std()
        return sma + (std * std_dev), sma, sma - (std * std_dev)
    
    # ==================== BASE STRATEGIES ====================
    
    def breakout_optimized(self):
        """Fixed Breakout - 10 period window"""
        high_10 = self.df['high'].rolling(10).max()
        low_10 = self.df['low'].rolling(10).min()
        
        signals = pd.Series(0, index=self.df.index)
        signals[self.df['close'] > high_10] = 1
        signals[self.df['close'] < low_10] = -1
        return signals
    
    def ema_crossover_filtered(self):
        """EMA Crossover with HTF filter (SMA50)"""
        ema12 = self.ema(self.df['close'], 12)
        ema26 = self.ema(self.df['close'], 26)
        sma50 = self.sma(self.df['close'], 50)
        
        signals = pd.Series(0, index=self.df.index)
        buy = (ema12 > ema26) & (ema12.shift(1) <= ema26.shift(1)) & (self.df['close'] > sma50)
        sell = (ema12 < ema26) & (ema12.shift(1) >= ema26.shift(1))
        
        signals[buy] = 1
        signals[sell] = -1
        return signals
    
    def volume_momentum_optimized(self):
        """Volume Momentum - 1.2x threshold"""
        rsi = self.rsi(self.df['close'], 14)
        vol_ma = self.df['volume'].rolling(20).mean()
        
        signals = pd.Series(0, index=self.df.index)
        buy = (self.df['volume'] > vol_ma * 1.2) & (rsi > 50)
        sell = (self.df['volume'] > vol_ma * 1.2) & (rsi < 50)
        
        signals[buy] = 1
        signals[sell] = -1
        return signals
    
    def rsi_extreme_strategy(self):
        """RSI Extreme - overbought/oversold"""
        rsi = self.rsi(self.df['close'], 14)
        
        signals = pd.Series(0, index=self.df.index)
        signals[rsi < 25] = 1
        signals[rsi > 75] = -1
        return signals
    
    def macd_strategy(self):
        """MACD Crossover"""
        ema12 = self.ema(self.df['close'], 12)
        ema26 = self.ema(self.df['close'], 26)
        macd = ema12 - ema26
        signal_line = macd.ewm(span=9).mean()
        
        signals = pd.Series(0, index=self.df.index)
        signals[macd > signal_line] = 1
        signals[macd < signal_line] = -1
        return signals
    
    def bollinger_rsi_enhanced(self):
        """Bollinger Bands + RSI (tight bands)"""
        upper, mid, lower = self.bollinger_bands(self.df['close'], 15, 1.5)
        rsi = self.rsi(self.df['close'], 14)
        
        signals = pd.Series(0, index=self.df.index)
        signals[(self.df['close'] < lower) & (rsi < 30)] = 1
        signals[(self.df['close'] > upper) & (rsi > 70)] = -1
        return signals
    
    def confluence_strategy(self):
        """Multi-indicator Confluence"""
        sma20 = self.sma(self.df['close'], 20)
        sma50 = self.sma(self.df['close'], 50)
        rsi = self.rsi(self.df['close'], 14)
        ema12 = self.ema(self.df['close'], 12)
        ema26 = self.ema(self.df['close'], 26)
        
        signals = pd.Series(0, index=self.df.index)
        buy = ((self.df['close'] > sma20) & (sma20 > sma50) & (rsi > 50) & (ema12 > ema26))
        sell = ((self.df['close'] < sma20) | (rsi < 50))
        
        signals[buy] = 1
        signals[sell] = -1
        return signals
    
    # ==================== ICT STRATEGIES ====================
    
    def ict_supply_demand_zones(self):
        """ICT Supply/Demand zones"""
        signals = pd.Series(0, index=self.df.index)
        period = 20
        
        for i in range(period, len(self.df)):
            high_20 = self.df['high'].iloc[i-period:i].max()
            low_20 = self.df['low'].iloc[i-period:i].min()
            current_price = self.df['close'].iloc[i]
            
            if current_price < low_20 * 1.002:
                signals.iloc[i] = 1
            elif current_price > high_20 * 0.998:
                signals.iloc[i] = -1
        
        return signals
    
    def ict_break_of_structure(self):
        """ICT Break of Structure"""
        signals = pd.Series(0, index=self.df.index)
        
        for i in range(5, len(self.df)):
            recent_low = self.df['low'].iloc[max(0, i-5):i].min()
            recent_high = self.df['high'].iloc[max(0, i-5):i].max()
            
            if self.df['close'].iloc[i] > recent_high * 1.001:
                signals.iloc[i] = 1
            elif self.df['close'].iloc[i] < recent_low * 0.999:
                signals.iloc[i] = -1
        
        return signals
    
    def ict_equal_highs_lows(self):
        """ICT Equal Highs/Lows"""
        signals = pd.Series(0, index=self.df.index)
        
        for i in range(30, len(self.df)):
            highs = self.df['high'].iloc[i-30:i].values
            lows = self.df['low'].iloc[i-30:i].values
            
            unique_highs = np.unique(np.round(highs, 3))
            if len(unique_highs) < len(highs) * 0.3:
                if self.df['close'].iloc[i] < highs.mean():
                    signals.iloc[i] = -1
            
            unique_lows = np.unique(np.round(lows, 3))
            if len(unique_lows) < len(lows) * 0.3:
                if self.df['close'].iloc[i] > lows.mean():
                    signals.iloc[i] = 1
        
        return signals
    
    def ict_displacement_strategy(self):
        """ICT Displacement"""
        signals = pd.Series(0, index=self.df.index)
        atr = self.atr(self.df, 14)
        sma_50 = self.sma(self.df['close'], 50)
        
        for i in range(1, len(self.df)):
            if atr.iloc[i] > atr.rolling(50).mean().iloc[i] * 1.5:
                if self.df['close'].iloc[i] > sma_50.iloc[i]:
                    signals.iloc[i] = 1
                else:
                    signals.iloc[i] = -1
        
        return signals
    
    # ==================== ML STRATEGIES ====================
    
    def ml_mean_reversion(self):
        """ML Mean Reversion"""
        signals = pd.Series(0, index=self.df.index)
        
        returns = self.df['close'].pct_change()
        returns_ma = returns.rolling(20).mean()
        returns_std = returns.rolling(20).std()
        
        z_score = (returns - returns_ma) / (returns_std + 1e-6)
        
        signals[z_score < -2] = 1
        signals[z_score > 2] = -1
        
        return signals
    
    def ml_momentum_acceleration(self):
        """ML Momentum Acceleration"""
        signals = pd.Series(0, index=self.df.index)
        
        momentum = self.df['close'].diff(5)
        accel = momentum.diff()
        
        accel_ma = accel.rolling(10).mean()
        accel_std = accel.rolling(10).std()
        
        signals[accel > accel_ma + accel_std] = 1
        signals[accel < accel_ma - accel_std] = -1
        
        return signals
    
    def ml_volatility_expansion(self):
        """ML Volatility Expansion"""
        signals = pd.Series(0, index=self.df.index)
        
        returns_vol = self.df['close'].pct_change().rolling(20).std()
        vol_ma = returns_vol.rolling(30).mean()
        
        upper = self.df['close'].rolling(20).max()
        lower = self.df['close'].rolling(20).min()
        
        signals[(returns_vol > vol_ma * 1.3) & (self.df['close'] == upper)] = 1
        signals[(returns_vol > vol_ma * 1.3) & (self.df['close'] == lower)] = -1
        
        return signals
    
    def ml_entropy_based(self):
        """ML Entropy-based"""
        signals = pd.Series(0, index=self.df.index)
        
        for i in range(30, len(self.df)):
            closes = self.df['close'].iloc[i-30:i].values
            returns = np.diff(closes) / closes[:-1]
            
            hist, _ = np.histogram(returns, bins=10)
            hist = hist[hist > 0] / hist.sum()
            entropy = -np.sum(hist * np.log2(hist + 1e-10))
            
            if entropy > 2.5:
                rsi = self.rsi(self.df['close'].iloc[:i+1], 14).iloc[-1]
                if rsi > 70:
                    signals.iloc[i] = -1
                elif rsi < 30:
                    signals.iloc[i] = 1
        
        return signals
    
    def ml_pattern_recognition(self):
        """ML Pattern Recognition"""
        signals = pd.Series(0, index=self.df.index)
        
        for i in range(10, len(self.df)):
            highs = self.df['high'].iloc[i-10:i].values
            lows = self.df['low'].iloc[i-10:i].values
            
            if lows[-1] > lows[-5] and highs[-1] > highs[-5]:
                signals.iloc[i] = 1
            elif highs[-1] < highs[-5] and lows[-1] < lows[-5]:
                signals.iloc[i] = -1
        
        return signals


class AdvancedBacktest:
    """Backtest engine with drawdown tracking"""
    
    def __init__(self, capital=100):
        self.initial_capital = capital
        self.capital = capital
        self.drawdowns = []
        self.peak_equity = capital
    
    def backtest(self, df, signals):
        """Run backtest"""
        trades = []
        position = None
        entry_price = 0
        
        try:
            for i in range(1, len(df)):
                signal = signals.iloc[i]
                price = df['close'].iloc[i]
                
                if self.capital > self.peak_equity:
                    self.peak_equity = self.capital
                
                if position and ((position == 1 and signal == -1) or (position == -1 and signal == 1)):
                    exit_price = price
                    pnl = (exit_price - entry_price) * position
                    self.capital += pnl
                    
                    trades.append({
                        'pnl': pnl,
                        'pnl_pct': (pnl / (entry_price * abs(position))) * 100,
                        'equity': self.capital
                    })
                    
                    dd = ((self.peak_equity - self.capital) / self.peak_equity) * 100 if self.peak_equity > 0 else 0
                    self.drawdowns.append(dd)
                    
                    position = None
                
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


def run_instrument_backtest(args):
    """Backtest single instrument"""
    symbol, df, iteration = args
    
    if df is None or len(df) < 100:
        return None
    
    all_strat = AllStrategies(df)
    
    strategies = {
        # Base Strategies (7)
        'Breakout': all_strat.breakout_optimized(),
        'EMA-HTF': all_strat.ema_crossover_filtered(),
        'Vol-Momentum': all_strat.volume_momentum_optimized(),
        'RSI-Extreme': all_strat.rsi_extreme_strategy(),
        'MACD': all_strat.macd_strategy(),
        'BB-RSI': all_strat.bollinger_rsi_enhanced(),
        'Confluence': all_strat.confluence_strategy(),
        # ICT Strategies (4)
        'ICT-SD': all_strat.ict_supply_demand_zones(),
        'ICT-BOS': all_strat.ict_break_of_structure(),
        'ICT-EHL': all_strat.ict_equal_highs_lows(),
        'ICT-Disp': all_strat.ict_displacement_strategy(),
        # ML Strategies (5)
        'ML-MeanRev': all_strat.ml_mean_reversion(),
        'ML-MomAcc': all_strat.ml_momentum_acceleration(),
        'ML-VolExp': all_strat.ml_volatility_expansion(),
        'ML-Entropy': all_strat.ml_entropy_based(),
        'ML-Pattern': all_strat.ml_pattern_recognition(),
    }
    
    results = {
        'instrument': symbol,
        'iteration': iteration,
        'strategies': {},
        'timestamp': datetime.now().isoformat()
    }
    
    best_return = -float('inf')
    best_strategy = None
    
    for strat_name, signals in strategies.items():
        bt = AdvancedBacktest(capital=100)
        result = bt.backtest(df, signals)
        
        if result:
            results['strategies'][strat_name] = result
            
            if result['return_pct'] > best_return:
                best_return = result['return_pct']
                best_strategy = strat_name
    
    results['best'] = best_strategy
    results['best_return'] = best_return if best_strategy else 0
    
    return results


def load_all_instrument_data():
    """Load all instruments"""
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


def run_comprehensive_optimization():
    """Run 50 iterations with all strategies"""
    
    print("\n" + "="*90)
    print("COMPREHENSIVE 50-ITERATION OPTIMIZATION")
    print("Base (7) + ICT (4) + ML (5) = 16 Total Strategies")
    print("Capital: $100 | Drawdown Tracking | Max Workers: 6")
    print("="*90)
    
    instruments_data = load_all_instrument_data()
    
    if not instruments_data:
        print("❌ No data found in data/mt5_feeds")
        return
    
    print(f"Loaded {len(instruments_data)} instruments\n")
    
    all_results = []
    iteration_summaries = []
    
    for iteration in range(1, 51):
        print(f"📊 Iteration {iteration}/50...", end="", flush=True)
        
        iter_data = {
            'iteration': iteration,
            'timestamp': datetime.now().isoformat(),
            'instruments': {}
        }
        
        tasks = [(symbol, df, iteration) for symbol, df in instruments_data.items()]
        
        with ProcessPoolExecutor(max_workers=6) as executor:
            futures = [executor.submit(run_instrument_backtest, task) for task in tasks]
            
            portfolio_returns = []
            total_trades = 0
            strategy_performance = {}
            
            for future in as_completed(futures):
                result = future.result()
                if result:
                    iter_data['instruments'][result['instrument']] = result
                    
                    if result.get('best'):
                        best_strat = result['best']
                        best_return = result['strategies'][best_strat]['return_pct']
                        portfolio_returns.append(best_return)
                        total_trades += result['strategies'][best_strat]['trades']
                        
                        if best_strat not in strategy_performance:
                            strategy_performance[best_strat] = 0
                        strategy_performance[best_strat] += 1
        
        avg_return = np.mean(portfolio_returns) if portfolio_returns else 0
        iter_data['portfolio_avg_return'] = avg_return
        iter_data['total_trades'] = total_trades
        iter_data['strategy_usage'] = strategy_performance
        
        iteration_summaries.append(iter_data)
        all_results.append(iter_data)
        
        print(f" ✅ Return: {avg_return:+.2f}% | Trades: {total_trades}")
    
    # Summary
    returns = [r['portfolio_avg_return'] for r in iteration_summaries]
    best_idx = np.argmax(returns)
    worst_idx = np.argmin(returns)
    
    summary = {
        'total_iterations': 50,
        'capital': 100,
        'strategies_count': 16,
        'strategy_breakdown': {'base': 7, 'ict': 4, 'ml': 5},
        'avg_return_all_iterations': float(np.mean(returns)),
        'std_return': float(np.std(returns)),
        'best_iteration': best_idx + 1,
        'best_return': float(returns[best_idx]),
        'worst_iteration': worst_idx + 1,
        'worst_return': float(returns[worst_idx]),
        'timestamp': datetime.now().isoformat()
    }
    
    # Save results
    output_dir = Path("results/backtest")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "comprehensive_50_iterations.json", 'w') as f:
        json.dump({
            'summary': summary,
            'iterations': iteration_summaries
        }, f, indent=2, default=str)
    
    print("\n" + "="*90)
    print("OPTIMIZATION COMPLETE")
    print("="*90)
    print(f"Average Return (50 iterations): {summary['avg_return_all_iterations']:+.2f}%")
    print(f"Std Deviation: {summary['std_return']:.2f}%")
    print(f"Best Iteration {summary['best_iteration']}: {summary['best_return']:+.2f}%")
    print(f"Worst Iteration {summary['worst_iteration']}: {summary['worst_return']:+.2f}%")
    print(f"\n✅ Results saved: results/backtest/comprehensive_50_iterations.json")
    print("="*90 + "\n")


if __name__ == "__main__":
    run_comprehensive_optimization()
