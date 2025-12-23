#!/usr/bin/env python3
"""
Real MT5 Data Backtester
Tests strategies with actual historical data from MT5
"""

import sys
sys.path.insert(0, ".")

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import MetaTrader5 as mt5
import yaml
import json
from typing import Dict, List, Tuple, Optional

class MT5RealBacktester:
    def __init__(self):
        self.commission_per_lot = 7.0  # $7 commission
        self.account_balance = 10000.0
        self.results = {}
        
    def get_mt5_data(self, symbol: str, timeframe: int, bars: int = 10000) -> Optional[pd.DataFrame]:
        """Get real historical data from MT5"""
        print(f"📊 Fetching {bars} bars of {symbol} data...")
        
        if not mt5.initialize():
            print("❌ MT5 initialization failed")
            return None
        
        try:
            # Get historical data
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
            
            if rates is None or len(rates) == 0:
                print(f"❌ No data available for {symbol}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            print(f"✅ Got {len(df)} bars from {df.index[0]} to {df.index[-1]}")
            return df
            
        except Exception as e:
            print(f"❌ Error getting data for {symbol}: {e}")
            return None
        finally:
            mt5.shutdown()
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        # Moving averages
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['ema_20'] = df['close'].ewm(span=20).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # ATR
        df['tr'] = np.maximum(df['high'] - df['low'], 
                             np.maximum(abs(df['high'] - df['close'].shift(1)),
                                       abs(df['low'] - df['close'].shift(1))))
        df['atr'] = df['tr'].rolling(14).mean()
        
        # Support/Resistance levels
        df['resistance'] = df['high'].rolling(20).max()
        df['support'] = df['low'].rolling(20).min()
        
        return df
    
    def mean_reversion_strategy(self, df: pd.DataFrame) -> List[Dict]:
        """Test mean reversion strategy with real data"""
        trades = []
        position = None
        
        for i in range(50, len(df)):  # Start after indicators are calculated
            current = df.iloc[i]
            
            # Entry conditions: Price far from 20 SMA + RSI extreme
            if position is None:
                distance_from_sma = abs(current['close'] - current['sma_20'])
                atr_distance = distance_from_sma / current['atr']
                
                # Long signal: Price below SMA + RSI oversold
                if (current['close'] < current['sma_20'] and 
                    atr_distance > 1.5 and 
                    current['rsi'] < 30):
                    
                    position = {
                        'type': 'long',
                        'entry_price': current['close'],
                        'entry_time': current.name,
                        'stop_loss': current['close'] - (current['atr'] * 2),
                        'take_profit': current['sma_20']
                    }
                
                # Short signal: Price above SMA + RSI overbought  
                elif (current['close'] > current['sma_20'] and 
                      atr_distance > 1.5 and 
                      current['rsi'] > 70):
                    
                    position = {
                        'type': 'short',
                        'entry_price': current['close'],
                        'entry_time': current.name,
                        'stop_loss': current['close'] + (current['atr'] * 2),
                        'take_profit': current['sma_20']
                    }
            
            # Exit conditions
            elif position is not None:
                exit_trade = False
                exit_price = current['close']
                exit_reason = 'time'
                
                if position['type'] == 'long':
                    if current['high'] >= position['take_profit']:
                        exit_price = position['take_profit']
                        exit_reason = 'tp'
                        exit_trade = True
                    elif current['low'] <= position['stop_loss']:
                        exit_price = position['stop_loss']
                        exit_reason = 'sl'
                        exit_trade = True
                
                elif position['type'] == 'short':
                    if current['low'] <= position['take_profit']:
                        exit_price = position['take_profit']
                        exit_reason = 'tp'
                        exit_trade = True
                    elif current['high'] >= position['stop_loss']:
                        exit_price = position['stop_loss']
                        exit_reason = 'sl'
                        exit_trade = True
                
                # Time-based exit (max 50 bars)
                bars_in_trade = i - df.index.get_loc(position['entry_time'])
                if bars_in_trade >= 50:
                    exit_trade = True
                    exit_reason = 'time'
                
                if exit_trade:
                    # Calculate P&L
                    if position['type'] == 'long':
                        pips = (exit_price - position['entry_price']) * 10000
                    else:
                        pips = (position['entry_price'] - exit_price) * 10000
                    
                    # Account for commission (7 pips)
                    net_pips = pips - 7
                    
                    trades.append({
                        'entry_time': position['entry_time'],
                        'exit_time': current.name,
                        'type': position['type'],
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'pips': pips,
                        'net_pips': net_pips,
                        'exit_reason': exit_reason,
                        'profit_usd': net_pips * 1  # $1 per pip for 0.01 lot
                    })
                    
                    position = None
        
        return trades
    
    def breakout_retest_strategy(self, df: pd.DataFrame) -> List[Dict]:
        """Test breakout retest strategy with real data"""
        trades = []
        position = None
        
        for i in range(50, len(df)):
            current = df.iloc[i]
            prev = df.iloc[i-1]
            
            if position is None:
                # Look for breakout above resistance
                if (current['close'] > current['resistance'] and 
                    prev['close'] <= prev['resistance'] and
                    current['volume'] > df['tick_volume'].rolling(20).mean().iloc[i]):
                    
                    # Wait for retest
                    for j in range(i+1, min(i+10, len(df))):
                        retest_bar = df.iloc[j]
                        if retest_bar['low'] <= current['resistance']:
                            position = {
                                'type': 'long',
                                'entry_price': current['resistance'],
                                'entry_time': retest_bar.name,
                                'stop_loss': current['resistance'] - (current['atr'] * 1.5),
                                'take_profit': current['resistance'] + (current['atr'] * 2)
                            }
                            break
                
                # Look for breakout below support
                elif (current['close'] < current['support'] and 
                      prev['close'] >= prev['support'] and
                      current['volume'] > df['tick_volume'].rolling(20).mean().iloc[i]):
                    
                    # Wait for retest
                    for j in range(i+1, min(i+10, len(df))):
                        retest_bar = df.iloc[j]
                        if retest_bar['high'] >= current['support']:
                            position = {
                                'type': 'short',
                                'entry_price': current['support'],
                                'entry_time': retest_bar.name,
                                'stop_loss': current['support'] + (current['atr'] * 1.5),
                                'take_profit': current['support'] - (current['atr'] * 2)
                            }
                            break
            
            # Exit logic (similar to mean reversion)
            elif position is not None:
                exit_trade = False
                exit_price = current['close']
                exit_reason = 'time'
                
                if position['type'] == 'long':
                    if current['high'] >= position['take_profit']:
                        exit_price = position['take_profit']
                        exit_reason = 'tp'
                        exit_trade = True
                    elif current['low'] <= position['stop_loss']:
                        exit_price = position['stop_loss']
                        exit_reason = 'sl'
                        exit_trade = True
                
                elif position['type'] == 'short':
                    if current['low'] <= position['take_profit']:
                        exit_price = position['take_profit']
                        exit_reason = 'tp'
                        exit_trade = True
                    elif current['high'] >= position['stop_loss']:
                        exit_price = position['stop_loss']
                        exit_reason = 'sl'
                        exit_trade = True
                
                # Time exit
                bars_in_trade = i - df.index.get_loc(position['entry_time'])
                if bars_in_trade >= 30:
                    exit_trade = True
                
                if exit_trade:
                    if position['type'] == 'long':
                        pips = (exit_price - position['entry_price']) * 10000
                    else:
                        pips = (position['entry_price'] - exit_price) * 10000
                    
                    net_pips = pips - 7  # Commission
                    
                    trades.append({
                        'entry_time': position['entry_time'],
                        'exit_time': current.name,
                        'type': position['type'],
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'pips': pips,
                        'net_pips': net_pips,
                        'exit_reason': exit_reason,
                        'profit_usd': net_pips * 1
                    })
                    
                    position = None
        
        return trades
    
    def analyze_trades(self, trades: List[Dict], strategy_name: str) -> Dict:
        """Analyze trade results"""
        if not trades:
            return {
                'strategy': strategy_name,
                'total_trades': 0,
                'win_rate': 0,
                'avg_winner': 0,
                'avg_loser': 0,
                'net_profit': 0,
                'monthly_return': 0
            }
        
        df_trades = pd.DataFrame(trades)
        
        # Basic stats
        total_trades = len(trades)
        winners = df_trades[df_trades['net_pips'] > 0]
        losers = df_trades[df_trades['net_pips'] <= 0]
        
        win_rate = len(winners) / total_trades if total_trades > 0 else 0
        avg_winner = winners['net_pips'].mean() if len(winners) > 0 else 0
        avg_loser = losers['net_pips'].mean() if len(losers) > 0 else 0
        
        total_pips = df_trades['net_pips'].sum()
        total_profit = df_trades['profit_usd'].sum()
        
        # Calculate monthly return
        days_tested = (df_trades['exit_time'].max() - df_trades['entry_time'].min()).days
        monthly_return = (total_profit / self.account_balance) * (30 / days_tested) * 100 if days_tested > 0 else 0
        
        return {
            'strategy': strategy_name,
            'total_trades': total_trades,
            'winners': len(winners),
            'losers': len(losers),
            'win_rate': win_rate,
            'avg_winner': avg_winner,
            'avg_loser': avg_loser,
            'total_pips': total_pips,
            'net_profit': total_profit,
            'monthly_return': monthly_return,
            'profit_factor': (winners['net_pips'].sum() / abs(losers['net_pips'].sum())) if len(losers) > 0 and losers['net_pips'].sum() < 0 else float('inf'),
            'max_drawdown': self.calculate_drawdown(df_trades),
            'days_tested': days_tested
        }
    
    def calculate_drawdown(self, df_trades: pd.DataFrame) -> float:
        """Calculate maximum drawdown"""
        df_trades['cumulative'] = df_trades['profit_usd'].cumsum()
        df_trades['running_max'] = df_trades['cumulative'].expanding().max()
        df_trades['drawdown'] = df_trades['cumulative'] - df_trades['running_max']
        max_drawdown = abs(df_trades['drawdown'].min())
        return (max_drawdown / self.account_balance) * 100
    
    def run_comprehensive_backtest(self):
        """Run backtest on multiple symbols and strategies"""
        symbols = ['EURUSDm', 'GBPUSDm', 'USDJPYm', 'USDCADm']
        timeframe = mt5.TIMEFRAME_M5
        
        print("🚀 STARTING COMPREHENSIVE MT5 BACKTEST")
        print("=" * 60)
        
        all_results = []
        
        for symbol in symbols:
            print(f"\n📊 Testing {symbol}...")
            
            # Get data
            df = self.get_mt5_data(symbol, timeframe, 5000)  # ~17 days of M5 data
            
            if df is None:
                continue
            
            # Calculate indicators
            df = self.calculate_indicators(df)
            
            # Test strategies
            strategies = {
                'Mean Reversion': self.mean_reversion_strategy,
                'Breakout Retest': self.breakout_retest_strategy
            }
            
            for strategy_name, strategy_func in strategies.items():
                print(f"  🧪 Testing {strategy_name}...")
                
                trades = strategy_func(df)
                analysis = self.analyze_trades(trades, f"{strategy_name} - {symbol}")
                
                if analysis['total_trades'] > 0:
                    all_results.append(analysis)
                    
                    print(f"    Trades: {analysis['total_trades']}")
                    print(f"    Win Rate: {analysis['win_rate']:.1%}")
                    print(f"    Net Profit: ${analysis['net_profit']:.0f}")
                    print(f"    Monthly Return: {analysis['monthly_return']:.1f}%")
        
        return all_results
    
    def save_results(self, results: List[Dict]):
        """Save backtest results"""
        # Save detailed results
        with open('real_backtest_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Create summary
        summary = {
            'backtest_date': datetime.now().isoformat(),
            'total_strategies_tested': len(results),
            'profitable_strategies': len([r for r in results if r['monthly_return'] > 0]),
            'best_strategy': max(results, key=lambda x: x['monthly_return']) if results else None,
            'average_monthly_return': np.mean([r['monthly_return'] for r in results]) if results else 0,
            'results': results
        }
        
        with open('backtest_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\n✅ Results saved to real_backtest_results.json")
        return summary

def main():
    backtester = MT5RealBacktester()
    
    # Run comprehensive backtest
    results = backtester.run_comprehensive_backtest()
    
    if results:
        # Save results
        summary = backtester.save_results(results)
        
        print(f"\n📊 REAL BACKTEST SUMMARY")
        print("=" * 40)
        print(f"Strategies Tested: {summary['total_strategies_tested']}")
        print(f"Profitable Strategies: {summary['profitable_strategies']}")
        print(f"Average Monthly Return: {summary['average_monthly_return']:.1f}%")
        
        if summary['best_strategy']:
            best = summary['best_strategy']
            print(f"\n🏆 BEST STRATEGY: {best['strategy']}")
            print(f"  Win Rate: {best['win_rate']:.1%}")
            print(f"  Monthly Return: {best['monthly_return']:.1f}%")
            print(f"  Total Trades: {best['total_trades']}")
            print(f"  Profit Factor: {best['profit_factor']:.2f}")
        
        print(f"\n✅ Real backtest complete with actual MT5 data!")
        
    else:
        print("\n❌ No backtest results generated")
        print("Check MT5 connection and data availability")

if __name__ == "__main__":
    main()
