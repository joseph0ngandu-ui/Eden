#!/usr/bin/env python3
"""
Export real MT5 data for backtest analysis.
This script connects to running MT5 and exports OHLC data.
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json

class MT5DataExporter:
    """Export and analyze real MT5 data"""
    
    SYMBOLS = ['VIX75', 'VIX100', 'VIX50', 'VIX25', 'StepIndex', 'Boom1000', 
               'Crash1000', 'Boom500', 'Crash500', 'XAUUSD']
    TIMEFRAME = mt5.TIMEFRAME_M5
    
    def __init__(self):
        self.data_dir = Path('mt5_data')
        self.data_dir.mkdir(exist_ok=True)
        
    def initialize_mt5(self):
        """Initialize MT5 connection"""
        if not mt5.initialize():
            print(f"MT5 initialization failed: {mt5.last_error()}")
            return False
        print("✓ MT5 initialized successfully")
        return True
    
    def fetch_symbol_data(self, symbol, start_date, end_date, bars=100000):
        """Fetch data for a symbol"""
        try:
            print(f"  Fetching {symbol}...", end=" ")
            
            # Try to get rates by date range
            rates = mt5.copy_rates_range(symbol, self.TIMEFRAME, start_date, end_date)
            
            if rates is None or len(rates) == 0:
                print(f"✗ No data")
                return None
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df = df[['time', 'open', 'high', 'low', 'close', 'tick_volume', 'real_volume']]
            df = df.sort_values('time').reset_index(drop=True)
            
            print(f"✓ {len(df)} bars")
            return df
            
        except Exception as e:
            print(f"✗ Error: {e}")
            return None
    
    def export_6month_data(self, start_date=None):
        """Export 6 months of data for all symbols"""
        if not self.initialize_mt5():
            return
        
        if start_date is None:
            start_date = datetime(2025, 8, 1)
        
        end_date = start_date + timedelta(days=184)
        
        print(f"\n{'='*70}")
        print(f"Exporting MT5 Data: {start_date.date()} to {end_date.date()}")
        print(f"{'='*70}\n")
        
        all_data = {}
        summary = {
            'period': f"{start_date.date()} to {end_date.date()}",
            'symbols': {},
            'total_bars': 0
        }
        
        for symbol in self.SYMBOLS:
            df = self.fetch_symbol_data(symbol, start_date, end_date)
            
            if df is not None and len(df) > 0:
                all_data[symbol] = df
                summary['symbols'][symbol] = {
                    'bars': len(df),
                    'start': df['time'].min().isoformat(),
                    'end': df['time'].max().isoformat(),
                    'first_close': float(df['close'].iloc[0]),
                    'last_close': float(df['close'].iloc[-1]),
                    'high': float(df['high'].max()),
                    'low': float(df['low'].min()),
                }
                summary['total_bars'] += len(df)
                
                # Save individual symbol data
                csv_path = self.data_dir / f"{symbol}_M5.csv"
                df.to_csv(csv_path, index=False)
                print(f"    → Saved to {csv_path.name}")
        
        mt5.shutdown()
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"EXPORT SUMMARY")
        print(f"{'='*70}\n")
        print(f"Total Symbols: {len(all_data)}/{len(self.SYMBOLS)}")
        print(f"Total Bars: {summary['total_bars']:,}")
        print()
        
        for symbol, data in summary['symbols'].items():
            print(f"{symbol:12} {data['bars']:>8,} bars | {data['start'][-8:]} to {data['end'][-8:]}")
        
        # Save summary
        summary_path = self.data_dir / 'export_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✓ Summary saved to {summary_path.name}")
        
        return all_data, summary


class RiskLadderBacktester:
    """Run backtest using exported MT5 data"""
    
    def __init__(self, data_dir='mt5_data'):
        self.data_dir = Path(data_dir)
        self.initial_capital = 100
        self.balance = initial_capital
        
    def get_risk_tier(self, balance):
        """Determine risk tier"""
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
    
    def calculate_ma_signals(self, df):
        """MA(3,10) crossover signals"""
        df = df.copy()
        df['ma3'] = df['close'].rolling(3).mean()
        df['ma10'] = df['close'].rolling(10).mean()
        df['signal'] = 0
        df.loc[df['ma3'] > df['ma10'], 'signal'] = 1
        return df
    
    def simulate_trades(self, df, risk_tier):
        """Simulate trades"""
        df = self.calculate_ma_signals(df).dropna()
        
        trades = []
        in_trade = False
        entry_price = 0
        entry_idx = 0
        
        for idx, row in df.iterrows():
            if row['signal'] == 1 and not in_trade:
                in_trade = True
                entry_price = row['close']
                entry_idx = idx
            
            elif in_trade and (idx - entry_idx >= 5 or idx == len(df) - 1):
                exit_price = row['close']
                profit = (exit_price - entry_price) * 100
                position_size = self.balance * risk_tier
                actual_profit = (profit / 100) * position_size
                
                trades.append({
                    'profit': actual_profit,
                    'duration': idx - entry_idx
                })
                
                self.balance += actual_profit
                in_trade = False
        
        return trades
    
    def run_backtest(self):
        """Run full 6-month backtest"""
        print(f"\n{'='*70}")
        print(f"REAL MT5 DATA BACKTEST - Risk Ladder")
        print(f"{'='*70}\n")
        
        months = [
            ('August 2025', 31),
            ('September 2025', 30),
            ('October 2025', 31),
            ('November 2025', 30),
            ('December 2025', 31),
            ('January 2026', 31),
        ]
        
        results = []
        
        for month_name, days in months:
            month_start = self.balance
            risk_tier = self.get_risk_tier(self.balance)
            
            print(f"{month_name}")
            print(f"  Opening: ${month_start:,.2f} | Tier: {risk_tier*100:.0f}%")
            
            total_trades = 0
            
            # Load and process each symbol's data
            for csv_file in self.data_dir.glob("*_M5.csv"):
                try:
                    df = pd.read_csv(csv_file)
                    df['time'] = pd.to_datetime(df['time'])
                    
                    # Slice by month (approximate)
                    trades = self.simulate_trades(df, risk_tier)
                    total_trades += len(trades)
                    
                except Exception as e:
                    continue
            
            month_end = self.balance
            month_profit = month_end - month_start
            month_return = (month_profit / month_start * 100) if month_start > 0 else 0
            
            print(f"  Closing: ${month_end:,.2f} | Return: {month_return:+.1f}%")
            print(f"  Trades: {total_trades}")
            print()
            
            results.append({
                'month': month_name,
                'start': month_start,
                'end': month_end,
                'profit': month_profit,
                'return': month_return,
                'trades': total_trades
            })
        
        # Print final summary
        print(f"{'='*70}")
        print(f"6-MONTH SUMMARY")
        print(f"{'='*70}\n")
        
        final_balance = self.balance
        total_profit = final_balance - self.initial_capital
        total_return = (total_profit / self.initial_capital * 100)
        multiplier = final_balance / self.initial_capital
        
        print(f"Starting Capital:  ${self.initial_capital:,.2f}")
        print(f"Final Balance:     ${final_balance:,.2f}")
        print(f"Total Profit:      ${total_profit:,.2f}")
        print(f"Total Return:      {total_return:.2f}%")
        print(f"Multiplier:        {multiplier:.2f}x")
        
        return results


def main():
    print("=" * 70)
    print("MT5 DATA EXPORT & BACKTEST TOOL")
    print("=" * 70)
    print()
    print("Make sure MT5 is open and connected to your broker.")
    print()
    
    # Step 1: Export data
    exporter = MT5DataExporter()
    all_data, summary = exporter.export_6month_data()
    
    if summary['total_bars'] > 0:
        print(f"\n✓ Successfully exported {summary['total_bars']:,} bars from {len(all_data)} symbols")
        
        # Step 2: Run backtest
        input("\nPress Enter to start backtest...")
        backtester = RiskLadderBacktester()
        backtester.run_backtest()
    else:
        print("\n✗ No data exported. Ensure MT5 is running and connected.")


if __name__ == '__main__':
    main()
