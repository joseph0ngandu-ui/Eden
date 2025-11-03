#!/usr/bin/env python3
"""
Base Strategy Backtest Runner for Eden
Tests 5 core strategies on all 10 instruments with real MT5 data
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SimpleIndicators:
    """Simple technical indicators"""
    
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
    def atr(high, low, close, period=14):
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean()
    
    @staticmethod
    def bollinger_bands(series, period=20, std_dev=2):
        sma = series.rolling(period).mean()
        std = series.rolling(period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower

class BaseStrategies:
    """Core trading strategies"""
    
    def __init__(self, df):
        self.df = df.copy()
        self.signals = pd.Series(0, index=df.index)
        
    def htf_bias_strategy(self):
        """Strategy 1: Higher Timeframe Bias"""
        sma20 = SimpleIndicators.sma(self.df['close'], 20)
        sma50 = SimpleIndicators.sma(self.df['close'], 50)
        rsi = SimpleIndicators.rsi(self.df['close'], 14)
        
        # Buy: Price above both SMAs + RSI > 50
        buy = (self.df['close'] > sma20) & (sma20 > sma50) & (rsi > 50)
        
        # Sell: Price below SMA20 or RSI < 50
        sell = (self.df['close'] < sma20) | (rsi < 50)
        
        signals = pd.Series(0, index=self.df.index)
        signals[buy] = 1
        signals[sell] = -1
        return signals
    
    def breakout_strategy(self):
        """Strategy 2: Simple Breakout"""
        high_20 = self.df['high'].rolling(20).max()
        low_20 = self.df['low'].rolling(20).min()
        
        # Buy on breakout above 20-period high
        buy = self.df['close'] > high_20
        
        # Sell on breakout below 20-period low
        sell = self.df['close'] < low_20
        
        signals = pd.Series(0, index=self.df.index)
        signals[buy] = 1
        signals[sell] = -1
        return signals
    
    def ema_crossover_strategy(self):
        """Strategy 3: EMA Crossover"""
        ema12 = SimpleIndicators.ema(self.df['close'], 12)
        ema26 = SimpleIndicators.ema(self.df['close'], 26)
        
        # Buy when EMA12 crosses above EMA26
        buy = (ema12 > ema26) & (ema12.shift(1) <= ema26.shift(1))
        
        # Sell when EMA12 crosses below EMA26
        sell = (ema12 < ema26) & (ema12.shift(1) >= ema26.shift(1))
        
        signals = pd.Series(0, index=self.df.index)
        signals[buy] = 1
        signals[sell] = -1
        return signals
    
    def bollinger_rsi_strategy(self):
        """Strategy 4: Bollinger Bands + RSI"""
        upper, mid, lower = SimpleIndicators.bollinger_bands(self.df['close'], 20, 2)
        rsi = SimpleIndicators.rsi(self.df['close'], 14)
        
        # Buy: Price near lower band + RSI < 30
        buy = (self.df['close'] < lower) & (rsi < 30)
        
        # Sell: Price near upper band + RSI > 70
        sell = (self.df['close'] > upper) & (rsi > 70)
        
        signals = pd.Series(0, index=self.df.index)
        signals[buy] = 1
        signals[sell] = -1
        return signals
    
    def volume_momentum_strategy(self):
        """Strategy 5: Volume + Momentum"""
        rsi = SimpleIndicators.rsi(self.df['close'], 14)
        vol_ma = self.df['volume'].rolling(20).mean()
        
        # Buy: High volume + RSI > 50
        buy = (self.df['volume'] > vol_ma * 1.5) & (rsi > 50)
        
        # Sell: Volume spike + RSI < 50
        sell = (self.df['volume'] > vol_ma * 1.5) & (rsi < 50)
        
        signals = pd.Series(0, index=self.df.index)
        signals[buy] = 1
        signals[sell] = -1
        return signals

class BacktestSimple:
    """Simple backtest engine"""
    
    def __init__(self, initial_capital=100000, commission=0.001):
        self.capital = initial_capital
        self.commission = commission
        
    def backtest(self, df, signals, strategy_name):
        """Run backtest on signals"""
        trades = []
        position = None
        entry_price = 0
        
        for i in range(1, len(df)):
            signal = signals.iloc[i]
            current_price = df['close'].iloc[i]
            current_time = df['timestamp'].iloc[i]
            
            # Close existing position on opposite signal
            if position and ((position == 1 and signal == -1) or (position == -1 and signal == 1)):
                exit_price = current_price
                pnl = (exit_price - entry_price) * position
                pnl_pct = (pnl / entry_price) * 100 if entry_price > 0 else 0
                
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': current_time,
                    'direction': 'LONG' if position > 0 else 'SHORT',
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct
                })
                position = None
            
            # Open new position
            if signal != 0 and position is None:
                entry_price = current_price
                entry_time = current_time
                position = signal
        
        # Calculate metrics
        if not trades:
            return {
                'strategy': strategy_name,
                'trades': 0,
                'net_pnl': 0,
                'return_pct': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'avg_win': 0,
                'avg_loss': 0
            }
        
        trades_df = pd.DataFrame(trades)
        wins = trades_df[trades_df['pnl'] > 0]
        losses = trades_df[trades_df['pnl'] < 0]
        
        gross_profit = wins['pnl'].sum() if len(wins) > 0 else 0
        gross_loss = abs(losses['pnl'].sum()) if len(losses) > 0 else 0
        
        return {
            'strategy': strategy_name,
            'trades': len(trades),
            'net_pnl': trades_df['pnl'].sum(),
            'return_pct': (trades_df['pnl'].sum() / self.capital) * 100,
            'win_rate': (len(wins) / len(trades) * 100) if len(trades) > 0 else 0,
            'profit_factor': (gross_profit / gross_loss) if gross_loss > 0 else 0,
            'avg_win': (gross_profit / len(wins)) if len(wins) > 0 else 0,
            'avg_loss': (gross_loss / len(losses)) if len(losses) > 0 else 0,
            'winning_trades': len(wins),
            'losing_trades': len(losses)
        }

def load_instrument_m1_data(symbol):
    """Load M1 data for instrument"""
    data_dir = Path("data/mt5_feeds")
    csv_file = data_dir / f"{symbol}_M1.csv"
    
    if not csv_file.exists():
        return None
    
    df = pd.read_csv(csv_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df.sort_values('timestamp').reset_index(drop=True)

def main():
    """Main execution"""
    
    instruments = ['VIX25', 'VIX50', 'VIX75', 'VIX100', 'Boom1000', 
                   'Boom500', 'Crash1000', 'Crash500', 'StepIndex', 'XAUUSD']
    
    strategies = ['HTF Bias', 'Breakout', 'EMA Crossover', 'Bollinger RSI', 'Volume Momentum']
    
    print("\n" + "="*80)
    print("EDEN BASE STRATEGY BACKTEST")
    print("="*80)
    print(f"Instruments: {len(instruments)}")
    print(f"Strategies: {len(strategies)}")
    print(f"Total combinations: {len(instruments) * len(strategies)}")
    print("="*80 + "\n")
    
    all_results = {}
    portfolio_summary = {
        'total_net_pnl': 0,
        'total_return_pct': 0,
        'total_trades': 0,
        'instruments_analyzed': 0,
        'best_instrument': None,
        'best_instrument_return': -float('inf'),
        'results_by_instrument': {},
        'results_by_strategy': {}
    }
    
    backtester = BacktestSimple()
    
    # Run backtest for each instrument-strategy combination
    for instrument in instruments:
        print(f"\n{instrument}:")
        df = load_instrument_m1_data(instrument)
        
        if df is None or len(df) < 50:
            print(f"  Skipped - insufficient data")
            continue
        
        portfolio_summary['instruments_analyzed'] += 1
        instrument_results = []
        best_instrument_result = None
        best_return = -float('inf')
        
        strategies_obj = BaseStrategies(df)
        
        # Test each strategy
        for i, strategy_name in enumerate(strategies):
            # Get signals based on strategy
            if i == 0:  # HTF Bias
                signals = strategies_obj.htf_bias_strategy()
            elif i == 1:  # Breakout
                signals = strategies_obj.breakout_strategy()
            elif i == 2:  # EMA Crossover
                signals = strategies_obj.ema_crossover_strategy()
            elif i == 3:  # Bollinger RSI
                signals = strategies_obj.bollinger_rsi_strategy()
            else:  # Volume Momentum
                signals = strategies_obj.volume_momentum_strategy()
            
            # Run backtest
            result = backtester.backtest(df, signals, strategy_name)
            instrument_results.append(result)
            
            # Track best for this instrument
            if result['return_pct'] > best_return:
                best_return = result['return_pct']
                best_instrument_result = result
            
            # Track strategy-level performance
            if strategy_name not in portfolio_summary['results_by_strategy']:
                portfolio_summary['results_by_strategy'][strategy_name] = {
                    'total_return': 0,
                    'count': 0,
                    'best_instrument': None,
                    'best_return': -float('inf')
                }
            
            portfolio_summary['results_by_strategy'][strategy_name]['total_return'] += result['return_pct']
            portfolio_summary['results_by_strategy'][strategy_name]['count'] += 1
            
            if result['return_pct'] > portfolio_summary['results_by_strategy'][strategy_name]['best_return']:
                portfolio_summary['results_by_strategy'][strategy_name]['best_return'] = result['return_pct']
                portfolio_summary['results_by_strategy'][strategy_name]['best_instrument'] = instrument
            
            # Print result
            print(f"  {strategy_name:18} | Return: {result['return_pct']:7.2f}% | " \
                  f"Trades: {result['trades']:3d} | WR: {result['win_rate']:5.1f}% | " \
                  f"PF: {result['profit_factor']:5.2f}")
        
        # Store instrument results
        portfolio_summary['results_by_instrument'][instrument] = {
            'best_strategy': best_instrument_result['strategy'],
            'best_return': best_return,
            'all_results': instrument_results
        }
        
        # Update portfolio totals
        portfolio_summary['total_net_pnl'] += best_instrument_result['net_pnl']
        portfolio_summary['total_return_pct'] += best_return
        portfolio_summary['total_trades'] += best_instrument_result['trades']
        
        if best_return > portfolio_summary['best_instrument_return']:
            portfolio_summary['best_instrument_return'] = best_return
            portfolio_summary['best_instrument'] = instrument
    
    # Print summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    print("\nBest Performing Instruments:")
    sorted_instruments = sorted(portfolio_summary['results_by_instrument'].items(),
                               key=lambda x: x[1]['best_return'], reverse=True)
    for i, (instrument, result) in enumerate(sorted_instruments[:5], 1):
        print(f"  {i}. {instrument:10} | Strategy: {result['best_strategy']:18} | Return: {result['best_return']:7.2f}%")
    
    print("\nBest Performing Strategies:")
    for strategy in strategies:
        if strategy in portfolio_summary['results_by_strategy']:
            stats = portfolio_summary['results_by_strategy'][strategy]
            avg_return = stats['total_return'] / stats['count']
            print(f"  {strategy:18} | Avg Return: {avg_return:7.2f}% | Best: {stats['best_instrument']} ({stats['best_return']:6.2f}%)")
    
    print("\n" + "="*80)
    print("PORTFOLIO TOTALS (Best strategy per instrument)")
    print("="*80)
    print(f"Total Instruments Analyzed: {portfolio_summary['instruments_analyzed']}")
    print(f"Combined Net PnL: ${portfolio_summary['total_net_pnl']:,.2f}")
    print(f"Combined Return %: {portfolio_summary['total_return_pct']:.2f}%")
    print(f"Average Return per Instrument: {portfolio_summary['total_return_pct'] / portfolio_summary['instruments_analyzed']:.2f}%")
    print(f"Total Trades: {portfolio_summary['total_trades']}")
    print(f"Best Instrument: {portfolio_summary['best_instrument']} ({portfolio_summary['best_instrument_return']:.2f}%)")
    print("="*80)
    
    # Save results
    output_dir = Path("results/backtest")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "base_strategies_results.json", 'w') as f:
        # Convert to serializable format
        serializable_results = {
            'timestamp': datetime.now().isoformat(),
            'summary': portfolio_summary,
            'instruments_analyzed': list(portfolio_summary['results_by_instrument'].keys()),
            'total_combinations_tested': len(instruments) * len(strategies)
        }
        json.dump(serializable_results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_dir}/base_strategies_results.json")
    
    return portfolio_summary

if __name__ == "__main__":
    results = main()
