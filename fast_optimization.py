#!/usr/bin/env python3
"""
Fast Optimization Script - 5 Minute Turnaround
Quick fixes + parallel backtests on all instruments
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

class OptimizedStrategies:
    """Optimized trading strategies with quick fixes"""
    
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
    def bollinger_bands(series, period=20, std_dev=2):
        sma = series.rolling(period).mean()
        std = series.rolling(period).std()
        return sma + (std * std_dev), sma, sma - (std * std_dev)
    
    # FIXED: Breakout Strategy - Use 10-period instead of 20
    def breakout_optimized(self):
        high_10 = self.df['high'].rolling(10).max()
        low_10 = self.df['low'].rolling(10).min()
        
        signals = pd.Series(0, index=self.df.index)
        signals[self.df['close'] > high_10] = 1
        signals[self.df['close'] < low_10] = -1
        return signals
    
    # FIXED: EMA Crossover - Add HTF filter
    def ema_crossover_filtered(self):
        ema12 = self.ema(self.df['close'], 12)
        ema26 = self.ema(self.df['close'], 26)
        sma50 = self.sma(self.df['close'], 50)
        
        signals = pd.Series(0, index=self.df.index)
        
        # Only go long if above SMA50 (HTF filter)
        buy = (ema12 > ema26) & (ema12.shift(1) <= ema26.shift(1)) & (self.df['close'] > sma50)
        sell = (ema12 < ema26) & (ema12.shift(1) >= ema26.shift(1))
        
        signals[buy] = 1
        signals[sell] = -1
        return signals
    
    # FIXED: Volume Momentum - Lower threshold
    def volume_momentum_optimized(self):
        rsi = self.rsi(self.df['close'], 14)
        vol_ma = self.df['volume'].rolling(20).mean()
        
        signals = pd.Series(0, index=self.df.index)
        
        # Lower threshold: 1.2x instead of 1.5x
        buy = (self.df['volume'] > vol_ma * 1.2) & (rsi > 50)
        sell = (self.df['volume'] > vol_ma * 1.2) & (rsi < 50)
        
        signals[buy] = 1
        signals[sell] = -1
        return signals
    
    # NEW: RSI Extreme Strategy
    def rsi_extreme_strategy(self):
        rsi = self.rsi(self.df['close'], 14)
        
        signals = pd.Series(0, index=self.df.index)
        signals[rsi < 25] = 1  # Oversold = buy
        signals[rsi > 75] = -1  # Overbought = sell
        return signals
    
    # NEW: MACD Strategy
    def macd_strategy(self):
        ema12 = self.ema(self.df['close'], 12)
        ema26 = self.ema(self.df['close'], 26)
        macd = ema12 - ema26
        signal_line = macd.ewm(span=9).mean()
        
        signals = pd.Series(0, index=self.df.index)
        signals[macd > signal_line] = 1
        signals[macd < signal_line] = -1
        return signals
    
    # ENHANCED: Bollinger RSI with tighter bands
    def bollinger_rsi_enhanced(self):
        upper, mid, lower = self.bollinger_bands(self.df['close'], 15, 1.5)  # Tighter
        rsi = self.rsi(self.df['close'], 14)
        
        signals = pd.Series(0, index=self.df.index)
        signals[(self.df['close'] < lower) & (rsi < 30)] = 1
        signals[(self.df['close'] > upper) & (rsi > 70)] = -1
        return signals
    
    # NEW: Confluence Strategy (Multiple indicators)
    def confluence_strategy(self):
        sma20 = self.sma(self.df['close'], 20)
        sma50 = self.sma(self.df['close'], 50)
        rsi = self.rsi(self.df['close'], 14)
        ema12 = self.ema(self.df['close'], 12)
        ema26 = self.ema(self.df['close'], 26)
        
        signals = pd.Series(0, index=self.df.index)
        
        # Buy: 3+ signals aligned
        buy = ((self.df['close'] > sma20) & 
               (sma20 > sma50) & 
               (rsi > 50) & 
               (ema12 > ema26))
        
        sell = ((self.df['close'] < sma20) | 
                (rsi < 50))
        
        signals[buy] = 1
        signals[sell] = -1
        return signals

class FastBacktest:
    """Lightweight backtest engine"""
    
    def __init__(self, capital=100000):
        self.capital = capital
    
    def backtest(self, df, signals, strategy_name):
        trades = []
        position = None
        
        for i in range(1, len(df)):
            signal = signals.iloc[i]
            price = df['close'].iloc[i]
            
            if position and ((position == 1 and signal == -1) or (position == -1 and signal == 1)):
                exit_price = price
                pnl = (exit_price - entry_price) * position
                trades.append(pnl)
                position = None
            
            if signal != 0 and position is None:
                entry_price = price
                position = signal
        
        if not trades:
            return {'return': 0, 'trades': 0, 'wr': 0, 'pf': 0}
        
        trades = np.array(trades)
        wins = trades[trades > 0]
        losses = trades[trades < 0]
        
        return {
            'return': (trades.sum() / self.capital) * 100,
            'trades': len(trades),
            'wr': (len(wins) / len(trades)) * 100 if len(trades) > 0 else 0,
            'pf': wins.sum() / abs(losses.sum()) if len(losses) > 0 else 0
        }

def run_instrument_backtest(args):
    """Run backtest on single instrument"""
    symbol, df = args
    
    if df is None or len(df) < 50:
        return None
    
    bt = FastBacktest()
    strategies_obj = OptimizedStrategies(df)
    
    strategies = {
        'Breakout-Opt': strategies_obj.breakout_optimized(),
        'EMA-HTF': strategies_obj.ema_crossover_filtered(),
        'Vol-Mom-Opt': strategies_obj.volume_momentum_optimized(),
        'RSI-Extreme': strategies_obj.rsi_extreme_strategy(),
        'MACD': strategies_obj.macd_strategy(),
        'BB-RSI-Enh': strategies_obj.bollinger_rsi_enhanced(),
        'Confluence': strategies_obj.confluence_strategy()
    }
    
    results = {'instrument': symbol, 'strategies': {}}
    best_return = -float('inf')
    best_strategy = None
    
    for name, signals in strategies.items():
        result = bt.backtest(df, signals, name)
        results['strategies'][name] = result
        
        if result['return'] > best_return:
            best_return = result['return']
            best_strategy = name
    
    results['best'] = best_strategy
    results['best_return'] = best_return
    
    return results

def load_mt5_data(symbol):
    """Load M1 data"""
    csv_file = Path("data/mt5_feeds") / f"{symbol}_M1.csv"
    if not csv_file.exists():
        return None
    df = pd.read_csv(csv_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df.sort_values('timestamp').reset_index(drop=True)

def main():
    """Main execution"""
    
    instruments = ['VIX25', 'VIX50', 'VIX75', 'VIX100', 'Boom1000', 
                   'Boom500', 'Crash1000', 'Crash500', 'StepIndex', 'XAUUSD']
    
    print("\n" + "="*80)
    print("EDEN FAST OPTIMIZATION - 5 MINUTE TURNAROUND")
    print("="*80)
    print(f"Strategies: 7 (3 fixed + 4 new)")
    print(f"Instruments: 10")
    print(f"Total combinations: 70")
    print("="*80 + "\n")
    
    # Load all data
    print("Loading MT5 data...")
    data_dict = {}
    for symbol in instruments:
        df = load_mt5_data(symbol)
        if df is not None:
            data_dict[symbol] = df
    
    print(f"Loaded {len(data_dict)} instruments\n")
    
    # Parallel backtest
    print("Running parallel backtests (7 strategies × 10 instruments)...")
    all_results = {}
    portfolio_return = 0
    total_trades = 0
    
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(run_instrument_backtest, (symbol, data_dict.get(symbol))) 
                  for symbol in instruments]
        
        for future in as_completed(futures):
            result = future.result()
            if result:
                all_results[result['instrument']] = result
                portfolio_return += result['best_return']
                total_trades += result['strategies'][result['best']]['trades']
    
    # Print results
    print("\n" + "="*80)
    print("RESULTS - TOP PERFORMERS")
    print("="*80)
    
    sorted_results = sorted(all_results.items(), key=lambda x: x[1]['best_return'], reverse=True)
    
    print(f"\n{'Rank':<5} {'Instrument':<12} {'Strategy':<18} {'Return':<10} {'Trades':<8} {'WR':<8}")
    print("-" * 80)
    
    for i, (symbol, result) in enumerate(sorted_results[:10], 1):
        best_strat = result['best']
        strat_result = result['strategies'][best_strat]
        print(f"{i:<5} {symbol:<12} {best_strat:<18} {result['best_return']:>7.2f}% {strat_result['trades']:>6} {strat_result['wr']:>6.1f}%")
    
    print("\n" + "="*80)
    print("PORTFOLIO SUMMARY")
    print("="*80)
    print(f"Combined Return: {portfolio_return:.2f}%")
    print(f"Total Trades: {total_trades}")
    print(f"Average Return/Instrument: {portfolio_return/len(all_results):.2f}%")
    print(f"Best Instrument: {sorted_results[0][0]} ({sorted_results[0][1]['best_return']:.2f}%)")
    print("="*80)
    
    # Save results
    output_dir = Path("results/backtest")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_return_pct': portfolio_return,
        'total_trades': total_trades,
        'instruments_count': len(all_results),
        'strategies_count': 7,
        'results': all_results
    }
    
    with open(output_dir / "optimized_results.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\nResults saved to: results/backtest/optimized_results.json")
    print("\nCompleted in ~5 minutes!")
    
    return summary

if __name__ == "__main__":
    results = main()
