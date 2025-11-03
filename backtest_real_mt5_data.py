#!/usr/bin/env python3
"""
Real MT5 6-Month Backtest using live broker connection
Connects directly to MT5 terminal and pulls real market data
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

class RealMT5Backtest:
    """Live MT5 data backtest"""
    
    SYMBOLS = ['VIX75', 'VIX100', 'VIX50', 'VIX25', 'StepIndex', 
               'Boom1000', 'Crash1000', 'Boom500', 'Crash500', 'XAUUSD']
    
    def __init__(self):
        self.initial_capital = 100
        self.balance = self.initial_capital
        self.monthly_results = []
        self.all_trades = []
        
    def connect_mt5(self):
        """Connect to MT5"""
        print("Connecting to MT5...", end=" ")
        if not mt5.initialize():
            print(f"✗ Failed: {mt5.last_error()}")
            return False
        print("✓ Connected")
        return True
    
    def get_risk_tier(self, balance):
        """Get risk tier based on balance"""
        if balance >= 10000:
            return 0.05
        elif balance >= 5000:
            return 0.08
        elif balance >= 1000:
            return 0.10
        elif balance >= 500:
            return 0.15
        else:
            return 0.20
    
    def fetch_symbol_data(self, symbol, start_date, end_date):
        """Fetch real data from MT5"""
        try:
            rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M5, start_date, end_date)
            if rates is None or len(rates) == 0:
                return None
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df = df[['time', 'open', 'high', 'low', 'close']].copy()
            return df.sort_values('time').reset_index(drop=True)
        except:
            return None
    
    def calculate_signals(self, df):
        """Calculate MA(3,10) signals"""
        df['ma3'] = df['close'].rolling(3).mean()
        df['ma10'] = df['close'].rolling(10).mean()
        df = df.dropna()
        
        signals = []
        for i in range(len(df)):
            if i > 0:
                prev_cross = df['ma3'].iloc[i-1] <= df['ma10'].iloc[i-1]
                curr_cross = df['ma3'].iloc[i] > df['ma10'].iloc[i]
                if prev_cross and curr_cross:
                    signals.append((i, 1))  # Buy signal
        
        return df, signals
    
    def simulate_trades(self, df, signals, risk_tier):
        """Simulate trades from signals"""
        trades = []
        position_size = self.balance * risk_tier
        
        for sig_idx, sig_type in signals:
            entry_price = df['close'].iloc[sig_idx]
            
            # Find exit: 5 bars later or end of data
            exit_idx = min(sig_idx + 5, len(df) - 1)
            exit_price = df['close'].iloc[exit_idx]
            
            profit_pips = (exit_price - entry_price) * 100
            profit = (profit_pips / 100) * position_size
            
            trades.append({
                'entry': entry_price,
                'exit': exit_price,
                'profit': profit,
                'duration': exit_idx - sig_idx
            })
            
            self.balance += profit
        
        return trades
    
    def run_backtest(self, start_date, end_date):
        """Run backtest from start_date to end_date"""
        if not self.connect_mt5():
            return
        
        print(f"\n{'='*80}")
        print(f"REAL MT5 BACKTEST - Risk Ladder Strategy")
        print(f"Period: {start_date.date()} to {end_date.date()}")
        print(f"{'='*80}\n")
        
        # Calculate months between start and end dates
        months = []
        current = start_date
        month_names = ['January', 'February', 'March', 'April', 'May', 'June',
                      'July', 'August', 'September', 'October', 'November', 'December']
        
        while current < end_date:
            month_name = month_names[current.month - 1]
            year = current.year
            if current.month == 12:
                next_month = current.replace(year=year+1, month=1, day=1)
            else:
                next_month = current.replace(month=current.month+1, day=1)
            month_end = min(next_month, end_date)
            
            months.append((f"{month_name} {year}", current, month_end))
            current = next_month
            
            if current >= end_date:
                break
        
        for month_name, month_start, month_end in months:
            month_start_balance = self.balance
            risk_tier = self.get_risk_tier(self.balance)
            
            print(f"\n{month_name}")
            print(f"  Opening Balance: ${month_start_balance:,.2f}")
            print(f"  Risk Tier: {risk_tier*100:.0f}%")
            
            month_trades = 0
            symbols_processed = 0
            
            for symbol in self.SYMBOLS:
                df = self.fetch_symbol_data(symbol, month_start, month_end)
                
                if df is not None and len(df) >= 20:
                    df, signals = self.calculate_signals(df)
                    if signals:
                        trades = self.simulate_trades(df, signals, risk_tier)
                        month_trades += len(trades)
                        self.all_trades.extend(trades)
                        symbols_processed += 1
            
            month_end_balance = self.balance
            month_profit = month_end_balance - month_start_balance
            month_return = (month_profit / month_start_balance * 100) if month_start_balance > 0 else 0
            
            print(f"  Symbols Processed: {symbols_processed}/{len(self.SYMBOLS)}")
            print(f"  Trades Executed: {month_trades}")
            print(f"  Closing Balance: ${month_end_balance:,.2f}")
            print(f"  Monthly Return: {month_return:+.2f}%")
            
            self.monthly_results.append({
                'month': month_name,
                'start': month_start_balance,
                'end': month_end_balance,
                'profit': month_profit,
                'return': month_return,
                'trades': month_trades,
                'symbols': symbols_processed
            })
        
        mt5.shutdown()
        self.print_summary()
    
    def print_summary(self):
        """Print backtest summary"""
        print(f"\n{'='*80}")
        print(f"BACKTEST SUMMARY")
        print(f"{'='*80}\n")
        
        print(f"{'Month':<20} {'Start':>15} {'End':>15} {'Profit':>15} {'Return':>10}")
        print(f"{'-'*80}")
        
        for result in self.monthly_results:
            print(f"{result['month']:<20} ${result['start']:>14,.2f} ${result['end']:>14,.2f} ${result['profit']:>14,.2f} {result['return']:>9.2f}%")
        
        final_balance = self.balance
        total_profit = final_balance - self.initial_capital
        total_return = (total_profit / self.initial_capital * 100) if self.initial_capital > 0 else 0
        multiplier = final_balance / self.initial_capital if self.initial_capital > 0 else 0
        
        print(f"{'-'*80}")
        print(f"\n{'FINAL RESULTS':<40}")
        print(f"  Starting Capital:     ${self.initial_capital:,.2f}")
        print(f"  Final Balance:        ${final_balance:,.2f}")
        print(f"  Total Profit:         ${total_profit:,.2f}")
        print(f"  Total Return:         {total_return:.2f}%")
        print(f"  Multiplier:           {multiplier:.2f}x")
        print(f"  Total Trades:         {len(self.all_trades)}")
        
        # Calculate win rate
        if self.all_trades:
            winning = sum(1 for t in self.all_trades if t['profit'] > 0)
            losing = sum(1 for t in self.all_trades if t['profit'] < 0)
            win_rate = (winning / len(self.all_trades) * 100) if self.all_trades else 0
            
            print(f"  Win Rate:             {win_rate:.1f}%")
            print(f"  Winning Trades:       {winning}")
            print(f"  Losing Trades:        {losing}")
        
        # Save results
        results = {
            'timestamp': datetime.now().isoformat(),
            'initial_capital': self.initial_capital,
            'final_balance': final_balance,
            'total_profit': total_profit,
            'total_return': total_return,
            'multiplier': multiplier,
            'total_trades': len(self.all_trades),
            'monthly_results': self.monthly_results
        }
        
        output_file = 'backtest_results_real_mt5.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Results saved to {output_file}")


def main():
    print("=" * 80)
    print("REAL MT5 DATA BACKTEST - 10 MONTHS")
    print("=" * 80)
    print()
    print("Requirements:")
    print("  • MetaTrader 5 must be running")
    print("  • Account must be logged in")
    print("  • All symbols must be available on your broker")
    print()
    
    start_date = datetime(2025, 1, 1)
    end_date = datetime(2025, 10, 31)
    
    backtester = RealMT5Backtest()
    backtester.run_backtest(start_date, end_date)


if __name__ == '__main__':
    main()
