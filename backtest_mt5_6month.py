#!/usr/bin/env python3
"""
Real MT5 6-Month Backtest with Risk Ladder Position Sizing
Uses actual MT5 data for Aug 1, 2025 - Jan 31, 2026
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json

class RiskLadderBacktester:
    """Risk Ladder backtest using real MT5 data"""
    
    def __init__(self, initial_capital=100):
        self.initial_capital = initial_capital
        self.balance = initial_capital
        self.trades = []
        self.monthly_balances = []
        
    def get_risk_tier(self, balance):
        """Get risk tier based on capital level"""
        if balance >= 10000:
            return 0.05  # 5%
        elif balance >= 5000:
            return 0.08  # 8%
        elif balance >= 1000:
            return 0.10  # 10%
        elif balance >= 500:
            return 0.15  # 15%
        else:
            return 0.20  # 20%
    
    def fetch_mt5_data(self, symbol, start_date, end_date, timeframe):
        """Fetch real data from MT5"""
        if not mt5.initialize():
            print(f"MT5 init failed: {mt5.last_error()}")
            return None
        
        try:
            rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
            if rates is None:
                print(f"No data for {symbol}")
                return None
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df = df[['time', 'open', 'high', 'low', 'close', 'tick_volume']].copy()
            df.rename(columns={'tick_volume': 'volume'}, inplace=True)
            return df.sort_values('time').reset_index(drop=True)
        finally:
            mt5.shutdown()
    
    def calculate_ma_signals(self, df):
        """Calculate MA(3,10) crossover signals"""
        df['ma3'] = df['close'].rolling(3).mean()
        df['ma10'] = df['close'].rolling(10).mean()
        
        # Buy signal: MA3 crosses above MA10
        df['signal'] = 0
        df.loc[df['ma3'] > df['ma10'], 'signal'] = 1
        
        return df
    
    def simulate_trading(self, df, risk_tier):
        """Simulate trades with given risk tier"""
        df = self.calculate_ma_signals(df)
        df = df.dropna()
        
        monthly_trades = []
        in_trade = False
        entry_price = 0
        entry_idx = 0
        
        for idx, row in df.iterrows():
            # Entry logic
            if row['signal'] == 1 and not in_trade:
                in_trade = True
                entry_price = row['close']
                entry_idx = idx
            
            # Exit logic: hold for 5 bars
            elif in_trade and (idx - entry_idx >= 5 or idx == len(df) - 1):
                exit_price = row['close']
                profit = (exit_price - entry_price) * 100  # Normalized
                
                # Apply position sizing
                position_size = self.balance * risk_tier
                actual_profit = (profit / 100) * position_size
                
                monthly_trades.append({
                    'entry': entry_price,
                    'exit': exit_price,
                    'profit': actual_profit,
                    'profit_pct': (profit / 100),
                    'duration': idx - entry_idx
                })
                
                self.balance += actual_profit
                in_trade = False
        
        return monthly_trades
    
    def run_6month_backtest(self, symbols, start_date):
        """Run 6-month backtest"""
        print(f"\n{'='*80}")
        print(f"6-MONTH REAL MT5 BACKTEST - Risk Ladder Strategy")
        print(f"{'='*80}")
        print(f"Starting Capital: ${self.initial_capital:,.2f}")
        print(f"Symbols: {', '.join(symbols)}")
        print()
        
        months_data = [
            ('August 2025', start_date, start_date + timedelta(days=31)),
            ('September 2025', start_date + timedelta(days=31), start_date + timedelta(days=61)),
            ('October 2025', start_date + timedelta(days=61), start_date + timedelta(days=92)),
            ('November 2025', start_date + timedelta(days=92), start_date + timedelta(days=122)),
            ('December 2025', start_date + timedelta(days=122), start_date + timedelta(days=153)),
            ('January 2026', start_date + timedelta(days=153), start_date + timedelta(days=184)),
        ]
        
        for month_name, month_start, month_end in months_data:
            month_start_balance = self.balance
            risk_tier = self.get_risk_tier(self.balance)
            
            print(f"\n{month_name}")
            print(f"  Opening Balance: ${month_start_balance:,.2f}")
            print(f"  Risk Tier: {risk_tier*100:.0f}%")
            
            monthly_trades = []
            for symbol in symbols:
                try:
                    df = self.fetch_mt5_data(symbol, month_start, month_end, mt5.TIMEFRAME_M5)
                    if df is None or len(df) < 20:
                        continue
                    
                    trades = self.simulate_trading(df, risk_tier)
                    monthly_trades.extend(trades)
                except Exception as e:
                    print(f"    Error processing {symbol}: {e}")
                    continue
            
            month_end_balance = self.balance
            month_profit = month_end_balance - month_start_balance
            month_return = (month_profit / month_start_balance * 100) if month_start_balance > 0 else 0
            
            self.monthly_balances.append({
                'month': month_name,
                'start': month_start_balance,
                'end': month_end_balance,
                'profit': month_profit,
                'return': month_return,
                'trades': len(monthly_trades)
            })
            
            print(f"  Trades: {len(monthly_trades)}")
            print(f"  Closing Balance: ${month_end_balance:,.2f}")
            print(f"  Monthly Return: {month_return:+.2f}%")
        
        self._print_summary()
    
    def _print_summary(self):
        """Print backtest summary"""
        print(f"\n{'='*80}")
        print(f"6-MONTH SUMMARY")
        print(f"{'='*80}")
        
        print(f"\n{'Month':<20} {'Start':<15} {'End':<15} {'Return':<10} {'Trades':<8}")
        print(f"{'-'*80}")
        
        total_trades = 0
        for data in self.monthly_balances:
            print(f"{data['month']:<20} ${data['start']:>13,.2f} ${data['end']:>13,.2f} {data['return']:>8.2f}% {data['trades']:>7}")
            total_trades += data['trades']
        
        final_balance = self.balance
        total_profit = final_balance - self.initial_capital
        total_return = (total_profit / self.initial_capital * 100)
        multiplier = final_balance / self.initial_capital
        
        print(f"{'-'*80}")
        print(f"{'TOTAL':<20} ${self.initial_capital:>13,.2f} ${final_balance:>13,.2f} {total_return:>8.2f}% {total_trades:>7}")
        
        print(f"\n{'FINAL RESULTS':<40}")
        print(f"  Starting Capital: ${self.initial_capital:,.2f}")
        print(f"  Final Balance: ${final_balance:,.2f}")
        print(f"  Total Profit: ${total_profit:,.2f}")
        print(f"  Total Return: {total_return:.2f}%")
        print(f"  Multiplier: {multiplier:.2f}x")
        
        # Save results
        results = {
            'initial_capital': self.initial_capital,
            'final_balance': final_balance,
            'total_profit': total_profit,
            'total_return': total_return,
            'multiplier': multiplier,
            'total_trades': total_trades,
            'monthly': self.monthly_balances
        }
        
        output_path = Path(__file__).parent / 'backtest_results_6month_real.json'
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nResults saved to: {output_path}")


def main():
    """Run the backtest"""
    # Symbols per rules
    symbols = ['VIX75', 'VIX100', 'VIX50', 'VIX25', 'StepIndex', 'Boom1000', 'Crash1000', 'Boom500', 'Crash500', 'XAUUSD']
    
    start_date = datetime(2025, 8, 1)
    
    backtester = RiskLadderBacktester(initial_capital=100)
    backtester.run_6month_backtest(symbols, start_date)


if __name__ == '__main__':
    main()
