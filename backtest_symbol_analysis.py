#!/usr/bin/env python3
"""
Per-Symbol Backtest Analysis
Analyzes each symbol separately and ranks by profitability, win rate, and consistency.
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from typing import Dict, List, Tuple

class SymbolBacktestAnalyzer:
    """Analyze individual symbol performance."""
    
    SYMBOLS = [
        'Volatility 75 Index',
        'Boom 500 Index',
        'Crash 500 Index',
        'Volatility 100 Index',
        'Boom 1000 Index',
        'Step Index'
    ]
    
    def __init__(self):
        self.results = {}
        self.rankings = []
    
    def connect_mt5(self) -> bool:
        """Connect to MT5."""
        print("Connecting to MT5...", end=" ")
        if not mt5.initialize():
            print(f"[FAILED] {mt5.last_error()}")
            return False
        print("[OK]")
        return True
    
    def fetch_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch M5 OHLCV data for symbol."""
        try:
            rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M5, start_date, end_date)
            if rates is None or len(rates) == 0:
                return None
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df = df[['time', 'open', 'high', 'low', 'close', 'tick_volume']].copy()
            df.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
            return df.sort_values('time').reset_index(drop=True)
        except:
            return None
    
    def calculate_ma(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate moving average."""
        return df['close'].rolling(period).mean()
    
    def generate_signals(self, df: pd.DataFrame) -> List[Tuple[int, str]]:
        """Generate MA(3,10) crossover signals."""
        df['ma3'] = self.calculate_ma(df, 3)
        df['ma10'] = self.calculate_ma(df, 10)
        df = df.dropna()
        
        signals = []
        for i in range(1, len(df)):
            prev_cross = df['ma3'].iloc[i-1] <= df['ma10'].iloc[i-1]
            curr_cross = df['ma3'].iloc[i] > df['ma10'].iloc[i]
            if prev_cross and curr_cross:
                signals.append((i, 'BUY'))
        
        return signals
    
    def simulate_trades(self, df: pd.DataFrame, signals: List[Tuple], initial_capital: float = 100) -> Dict:
        """Simulate trades and calculate metrics."""
        balance = initial_capital
        trades = []
        monthly_pnl = {}
        
        for sig_idx, sig_type in signals:
            entry_time = df['time'].iloc[sig_idx]
            entry_price = df['close'].iloc[sig_idx]
            month_key = entry_time.strftime('%Y-%m')
            
            # 5-bar hold
            exit_idx = min(sig_idx + 5, len(df) - 1)
            exit_price = df['close'].iloc[exit_idx]
            
            # Simple position sizing: risk 1% per trade
            risk = balance * 0.01
            price_move_ratio = (exit_price - entry_price) / entry_price if entry_price > 0 else 0
            profit = risk * price_move_ratio
            
            trades.append({
                'entry_time': entry_time,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'profit': profit,
                'bars_held': exit_idx - sig_idx
            })
            
            balance += profit
            
            if month_key not in monthly_pnl:
                monthly_pnl[month_key] = 0
            monthly_pnl[month_key] += profit
        
        # Calculate metrics
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_profit': 0,
                'monthly_pnl': monthly_pnl,
                'monthly_consistency_std': 0,
                'final_balance': balance
            }
        
        winning_trades = sum(1 for t in trades if t['profit'] > 0)
        win_rate = (winning_trades / len(trades) * 100) if trades else 0
        total_profit = sum(t['profit'] for t in trades)
        
        # Monthly consistency
        monthly_returns = [v for v in monthly_pnl.values()]
        monthly_std = np.std(monthly_returns) if len(monthly_returns) > 1 else 0
        
        return {
            'total_trades': len(trades),
            'win_rate': win_rate,
            'winning_trades': winning_trades,
            'losing_trades': len(trades) - winning_trades,
            'total_profit': total_profit,
            'final_balance': balance,
            'monthly_pnl': monthly_pnl,
            'monthly_consistency_std': monthly_std,
            'avg_win': np.mean([t['profit'] for t in trades if t['profit'] > 0]) if winning_trades > 0 else 0,
            'avg_loss': np.mean([t['profit'] for t in trades if t['profit'] < 0]) if (len(trades) - winning_trades) > 0 else 0,
        }
    
    def analyze_symbol(self, symbol: str, start_date: datetime, end_date: datetime) -> Dict:
        """Analyze single symbol."""
        print(f"\nAnalyzing {symbol}...", end=" ")
        
        # Fetch data
        df = self.fetch_data(symbol, start_date, end_date)
        if df is None:
            print("[NO DATA]")
            return None
        
        # Generate signals
        signals = self.generate_signals(df)
        if not signals:
            print("[NO SIGNALS]")
            return None
        
        # Simulate trades
        metrics = self.simulate_trades(df, signals)
        print(f"[OK] {metrics['total_trades']} trades, Win Rate: {metrics['win_rate']:.1f}%")
        
        return {
            'symbol': symbol,
            'metrics': metrics,
            'data_bars': len(df)
        }
    
    def rank_symbols(self, results: List[Dict]) -> List[Dict]:
        """Rank symbols by profitability and consistency."""
        if not results:
            return []
        
        ranked = []
        for r in results:
            m = r['metrics']
            rank_score = (
                m['total_profit'] * 0.4 +  # 40% profit
                m['win_rate'] * 0.4 -  # 40% win rate (higher is better)
                (m['monthly_consistency_std'] * 0.2)  # 20% consistency (lower std is better)
            )
            
            ranked.append({
                'symbol': r['symbol'],
                'rank_score': rank_score,
                'total_profit': m['total_profit'],
                'win_rate': m['win_rate'],
                'total_trades': m['total_trades'],
                'monthly_std': m['monthly_consistency_std'],
                'final_balance': m['final_balance'],
                'avg_win': m['avg_win'],
                'avg_loss': m['avg_loss'],
                'profitable': m['total_profit'] > 0,
                'consistent': m['monthly_consistency_std'] < 25,
            })
        
        # Sort by rank score descending
        ranked.sort(key=lambda x: x['rank_score'], reverse=True)
        
        for i, r in enumerate(ranked, 1):
            r['rank'] = i
        
        return ranked
    
    def run_analysis(self, start_date: datetime = None, end_date: datetime = None):
        """Run full symbol analysis."""
        if not start_date:
            start_date = datetime(2025, 1, 1)
        if not end_date:
            end_date = datetime(2025, 10, 31)
        
        if not self.connect_mt5():
            return
        
        print(f"\n{'='*80}")
        print(f"SYMBOL BACKTEST ANALYSIS")
        print(f"Period: {start_date.date()} to {end_date.date()}")
        print(f"{'='*80}\n")
        
        results = []
        for symbol in self.SYMBOLS:
            result = self.analyze_symbol(symbol, start_date, end_date)
            if result:
                results.append(result)
        
        # Rank symbols
        self.rankings = self.rank_symbols(results)
        
        # Print rankings
        print(f"\n{'='*80}")
        print(f"SYMBOL RANKINGS")
        print(f"{'='*80}\n")
        
        print(f"{'Rank':<6} {'Symbol':<25} {'PnL':<12} {'Win%':<10} {'Consistency':<12} {'Profitable':<12}")
        print(f"{'-'*80}")
        
        for r in self.rankings:
            status = "YES" if r['profitable'] else "NO"
            consistency = "GOOD" if r['consistent'] else "POOR"
            print(f"{r['rank']:<6} {r['symbol']:<25} ${r['total_profit']:>10.2f} {r['win_rate']:>8.1f}% {consistency:<12} {status:<12}")
        
        # Save results
        output = {
            'timestamp': datetime.now().isoformat(),
            'period_start': start_date.isoformat(),
            'period_end': end_date.isoformat(),
            'analysis_date': datetime.now().isoformat(),
            'rankings': self.rankings,
            'summary': {
                'total_symbols_tested': len(self.rankings),
                'profitable_symbols': sum(1 for r in self.rankings if r['profitable']),
                'consistent_symbols': sum(1 for r in self.rankings if r['consistent']),
                'top_performer': self.rankings[0]['symbol'] if self.rankings else None,
                'top_pnl': self.rankings[0]['total_profit'] if self.rankings else 0,
            }
        }
        
        with open('symbol_rankings.json', 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            for ranking in output['rankings']:
                ranking['profitable'] = bool(ranking['profitable'])
                ranking['consistent'] = bool(ranking['consistent'])
            json.dump(output, f, indent=2, default=str)
        
        print(f"\n[OK] Results saved to symbol_rankings.json")
        
        mt5.shutdown()
        
        return output

if __name__ == '__main__':
    analyzer = SymbolBacktestAnalyzer()
    analyzer.run_analysis()
