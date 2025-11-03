#!/usr/bin/env python3
"""
Optimized Backtest v1.2 - With Signal Filters and Exit Logic V2
Tests MA(3,10) strategy with advanced entry filters and adaptive exits
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from typing import Dict, List, Tuple

# Import optimization modules
import sys
sys.path.insert(0, 'src')
from signal_filter import SmartSignalGenerator, SignalConfig
from exit_logic import ExitManager, ExitConfig

class OptimizedBacktest:
    """Backtest with filters and advanced exit logic."""
    
    SYMBOLS = [
        'Volatility 75 Index',
        'Boom 500 Index',
        'Crash 500 Index',
        'Volatility 100 Index',
        'Boom 1000 Index',
        'Step Index'
    ]
    
    def __init__(self):
        self.initial_capital = 100
        self.balance = self.initial_capital
        self.monthly_results = []
        self.all_trades = []
        
        # Initialize signal generator with filters
        signal_config = SignalConfig(
            enable_volume_filter=True,
            enable_adx_filter=True,
            enable_bb_filter=True,
            volume_ma_period=20,
            volume_threshold_ratio=1.0,
            adx_period=14,
            adx_threshold=20.0,
            bb_period=20,
            bb_std_dev=2.0,
            bb_entry_zone=0.3
        )
        self.signal_generator = SmartSignalGenerator(fast_ma=3, slow_ma=10, config=signal_config)
        
        # Initialize exit manager
        exit_config = ExitConfig(
            min_hold_bars=3,
            max_hold_bars=4,
            breakeven_move_ratio=0.8,
            min_reward_ratio=1.5,
            max_reward_ratio=2.0,
            atr_period=14,
            trailing_stop_enable=True,
            use_momentum_exit=True
        )
        self.exit_manager = ExitManager(config=exit_config)
    
    def connect_mt5(self) -> bool:
        """Connect to MT5."""
        print("Connecting to MT5...", end=" ")
        if not mt5.initialize():
            print(f"[FAILED]")
            return False
        print("[OK]")
        return True
    
    def fetch_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch M5 OHLCV data."""
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
    
    def get_risk_tier(self, balance):
        """Get risk tier based on balance."""
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
    
    def simulate_trades_with_exit_logic(self, df: pd.DataFrame, signals: List[Tuple], risk_tier: float) -> List:
        """Simulate trades with advanced exit logic."""
        trades = []
        open_trades = []
        
        for signal_idx, sig_type in signals:
            entry_price = df['close'].iloc[signal_idx]
            entry_time = df['time'].iloc[signal_idx]
            
            # Calculate stop loss and take profit
            atr = self.exit_manager.calculate_atr(df.iloc[:signal_idx+1] if signal_idx < len(df) else df)
            current_atr = atr.iloc[signal_idx] if signal_idx < len(atr) else 0.1
            
            stop_loss = self.exit_manager.calculate_stop_loss(entry_price, current_atr, 'BUY')
            risk = entry_price - stop_loss
            take_profit = entry_price + (risk * 1.5)
            
            # Determine hold bars based on momentum
            hold_bars = self.exit_manager.determine_hold_bars(df, signal_idx)
            
            # Position sizing with risk tier
            position_size = self.balance * risk_tier
            
            # Store open trade
            open_trades.append({
                'entry_idx': signal_idx,
                'entry_price': entry_price,
                'entry_time': entry_time,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'hold_bars': hold_bars,
                'position_size': position_size,
                'direction': 'BUY',
                'initial_risk': risk
            })
        
        # Process exits
        for bar_idx in range(len(df)):
            closed = []
            for trade in open_trades:
                bars_held = bar_idx - trade['entry_idx']
                
                if bars_held > 0:
                    current_price = df['close'].iloc[bar_idx]
                    
                    # Update trailing stop
                    updated_stop = self.exit_manager.update_trailing_stop(
                        current_price,
                        trade['entry_price'],
                        trade['stop_loss'],
                        trade['initial_risk'],
                        'BUY'
                    )
                    trade['stop_loss'] = updated_stop
                    
                    # Check exit conditions
                    should_exit, exit_reason = self.exit_manager.check_exit_condition(
                        current_price,
                        trade['entry_price'],
                        trade['take_profit'],
                        trade['stop_loss'],
                        bars_held,
                        trade['hold_bars'],
                        'BUY'
                    )
                    
                    if should_exit:
                        profit_ratio = (current_price - trade['entry_price']) / trade['entry_price']
                        profit = trade['position_size'] * profit_ratio
                        
                        trades.append({
                            'entry_price': trade['entry_price'],
                            'entry_time': trade['entry_time'],
                            'exit_price': current_price,
                            'exit_idx': bar_idx,
                            'bars_held': bars_held,
                            'profit': profit,
                            'exit_reason': exit_reason
                        })
                        closed.append(trade)
                        self.balance += profit
            
            # Remove closed trades
            for trade in closed:
                open_trades.remove(trade)
        
        return trades
    
    def run_backtest(self, start_date: datetime = None, end_date: datetime = None):
        """Run optimized backtest."""
        if not start_date:
            start_date = datetime(2025, 1, 1)
        if not end_date:
            end_date = datetime(2025, 10, 31)
        
        if not self.connect_mt5():
            return
        
        print(f"\n{'='*80}")
        print(f"OPTIMIZED BACKTEST v1.2 - With Filters & Exit Logic V2")
        print(f"Period: {start_date.date()} to {end_date.date()}")
        print(f"{'='*80}\n")
        
        # Calculate months
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
                df = self.fetch_data(symbol, month_start, month_end)
                
                if df is not None and len(df) >= 50:
                    # Generate filtered signals
                    signals = self.signal_generator.generate_signals(df, use_filters=True)
                    
                    if signals:
                        trades = self.simulate_trades_with_exit_logic(df, signals, risk_tier)
                        month_trades += len(trades)
                        self.all_trades.extend(trades)
                        symbols_processed += 1
            
            month_end_balance = self.balance
            month_profit = month_end_balance - month_start_balance
            month_return = (month_profit / month_start_balance * 100) if month_start_balance > 0 else 0
            
            print(f"  Symbols Processed: {symbols_processed}/6")
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
        
        # Print summary
        self.print_summary()
        mt5.shutdown()
    
    def print_summary(self):
        """Print backtest summary."""
        print(f"\n{'='*80}")
        print(f"OPTIMIZED BACKTEST SUMMARY")
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
        
        if self.all_trades:
            winning = sum(1 for t in self.all_trades if t['profit'] > 0)
            losing = sum(1 for t in self.all_trades if t['profit'] < 0)
            win_rate = (winning / len(self.all_trades) * 100) if self.all_trades else 0
            
            print(f"  Win Rate:             {win_rate:.1f}%")
            print(f"  Winning Trades:       {winning}")
            print(f"  Losing Trades:        {losing}")
            
            # Get signal filter stats
            stats = self.signal_generator.get_signal_quality_report()
            print(f"\n  FILTER STATISTICS:")
            print(f"    Total MA Signals:    {stats['total_ma_signals_detected']}")
            print(f"    Confirmed Signals:   {stats['confirmed_signals']}")
            print(f"    Confirmation Rate:   {stats['confirmation_rate']}")
        
        # Save results
        results = {
            'timestamp': datetime.now().isoformat(),
            'backtest_version': 'v1.2-optimized',
            'initial_capital': self.initial_capital,
            'final_balance': float(final_balance),
            'total_profit': float(total_profit),
            'total_return': float(total_return),
            'multiplier': float(multiplier),
            'total_trades': len(self.all_trades),
            'monthly_results': self.monthly_results
        }
        
        output_file = 'backtest_results_v1.2_optimized.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n[OK] Results saved to {output_file}")

if __name__ == '__main__':
    backtest = OptimizedBacktest()
    backtest.run_backtest()
