#!/usr/bin/env python3
"""
Individual Strategy Backtest
Tests each strategy separately to identify profitability and tune parameters.
Target: 13% monthly return, <2% daily DD, <10% max DD
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import MetaTrader5 as mt5
    import pandas as pd
    import numpy as np
except ImportError as e:
    print(f"❌ Missing package: {e}")
    sys.exit(1)

# Strategy definitions with their target symbols
STRATEGIES = {
    'Pro_Trend_Follower': {
        'symbols': ['EURUSDm', 'GBPUSDm', 'USDJPYm', 'AUDJPYm', 'XAUUSDm'],
        'method': 'trend_follower'
    },
    'Pro_Mean_Reversion': {
        'symbols': ['EURUSDm', 'GBPUSDm', 'AUDJPYm'],
        'method': 'mean_reversion'
    },
    'Pro_RSI_Momentum': {
        'symbols': ['EURUSDm', 'GBPUSDm', 'XAUUSDm'],
        'method': 'rsi_momentum'
    },
    'Pro_Asian_Fade': {
        'symbols': ['USDJPYm', 'AUDJPYm'],
        'method': 'asian_fade'
    },
    'Pro_Volatility_Expansion': {
        'symbols': ['EURUSDm', 'GBPUSDm', 'USDJPYm'],
        'method': 'volatility_expansion'
    },
    'Pro_Overlap_Scalper': {
        'symbols': ['EURUSDm', 'GBPUSDm'],
        'method': 'overlap_scalper'
    },
    'Pro_Gold_Breakout': {
        'symbols': ['XAUUSDm'],
        'method': 'gold_london_breakout'
    },
    # NEW: US Index Strategies
    'Pro_Index_Momentum': {
        'symbols': ['US30m', 'US500m', 'USTECm'],
        'method': 'index_momentum_trend'
    },
    'Pro_Index_NY_Breakout': {
        'symbols': ['US30m', 'US500m', 'USTECm'],
        'method': 'index_ny_breakout'
    },
    'Pro_Index_Mean_Reversion': {
        'symbols': ['US30m', 'US500m', 'USTECm'],
        'method': 'index_mean_reversion'
    }
}

def initialize_mt5():
    mt5_path = r"C:\Program Files\MetaTrader 5 Terminal\terminal64.exe"
    if not mt5.initialize(path=mt5_path):
        print(f"❌ MT5 init failed: {mt5.last_error()}")
        return False
    return True

def fetch_data(symbol: str, months: int = 3):
    if not mt5.symbol_select(symbol, True):
        return None
    end = datetime.now()
    start = end - timedelta(days=months * 30)
    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M5, start, end)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df

def backtest_strategy(strategy_name: str, strategy_config: dict, months: int = 3):
    """Backtest a single strategy on its target symbols."""
    from trading.pro_strategies import ProStrategyEngine
    
    print(f"\n{'='*50}")
    print(f"  {strategy_name}")
    print(f"{'='*50}")
    
    engine = ProStrategyEngine()
    strategy_method = getattr(engine, strategy_config['method'])
    
    # Fetch data for all symbols
    all_data = {}
    for symbol in strategy_config['symbols']:
        df = fetch_data(symbol, months)
        if df is not None and len(df) > 1000:
            all_data[symbol] = df
            print(f"  ✓ {symbol}: {len(df)} bars")
        else:
            print(f"  ⚠ {symbol}: insufficient data")
    
    if not all_data:
        return None
    
    # Simulate trading
    initial_equity = 100000
    equity = initial_equity
    trades = []
    open_positions = {}
    risk_per_trade = 0.01  # 1% risk per trade
    
    # Get common timeline
    all_times = set()
    for df in all_data.values():
        all_times.update(df.index)
    all_times = sorted(list(all_times))
    
    print(f"  Simulating {len(all_times)} time points...")
    
    for t_idx, current_time in enumerate(all_times):
        if t_idx < 500: continue  # Skip warmup
        
        # Close positions (TP/SL)
        for symbol, pos in list(open_positions.items()):
            if symbol not in all_data or current_time not in all_data[symbol].index:
                continue
            bar = all_data[symbol].loc[current_time]
            
            hit_tp = (pos['dir'] == 'LONG' and bar['high'] >= pos['tp']) or \
                     (pos['dir'] == 'SHORT' and bar['low'] <= pos['tp'])
            hit_sl = (pos['dir'] == 'LONG' and bar['low'] <= pos['sl']) or \
                     (pos['dir'] == 'SHORT' and bar['high'] >= pos['sl'])
            
            if hit_tp or hit_sl:
                exit_price = pos['tp'] if hit_tp else pos['sl']
                pnl_price = (exit_price - pos['entry']) if pos['dir'] == 'LONG' else (pos['entry'] - exit_price)
                
                risk_amount = initial_equity * risk_per_trade
                sl_dist = abs(pos['entry'] - pos['sl'])
                pos_size = risk_amount / sl_dist if sl_dist > 0 else 0
                pnl = pos_size * pnl_price
                
                equity += pnl
                trades.append({
                    'time': current_time,
                    'symbol': symbol,
                    'pnl': pnl,
                    'equity': equity,
                    'result': 'WIN' if hit_tp else 'LOSS'
                })
                del open_positions[symbol]
        
        # Generate new signals (max 3 open positions)
        if len(open_positions) >= 3:
            continue
            
        for symbol in all_data.keys():
            if symbol in open_positions or current_time not in all_data[symbol].index:
                continue
            
            df_symbol = all_data[symbol]
            idx = df_symbol.index.get_loc(current_time)
            if idx < 200:
                continue
            
            window = df_symbol.iloc[:idx+1]
            
            try:
                signal = strategy_method(window, symbol)
                if signal:
                    open_positions[symbol] = {
                        'dir': signal.direction,
                        'entry': signal.entry_price,
                        'tp': signal.tp,
                        'sl': signal.sl
                    }
            except Exception as e:
                pass  # Skip errors silently
    
    if len(trades) == 0:
        print("  ❌ No trades generated")
        return None
    
    # Calculate metrics
    df_trades = pd.DataFrame(trades)
    final_equity = df_trades['equity'].iloc[-1]
    total_return = ((final_equity - initial_equity) / initial_equity) * 100
    monthly_return = total_return / months
    
    wins = len(df_trades[df_trades['result'] == 'WIN'])
    win_rate = (wins / len(trades)) * 100
    
    # Drawdown
    peak = initial_equity
    max_dd = 0
    for eq in df_trades['equity']:
        if eq > peak:
            peak = eq
        dd = ((peak - eq) / peak) * 100
        max_dd = max(max_dd, dd)
    
    # Profit factor
    gross_profit = df_trades[df_trades['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(df_trades[df_trades['pnl'] < 0]['pnl'].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else 0
    
    metrics = {
        'strategy': strategy_name,
        'trades': len(trades),
        'win_rate': win_rate,
        'monthly_return': monthly_return,
        'max_dd': max_dd,
        'profit_factor': pf,
        'final_equity': final_equity
    }
    
    # Print results
    status = "✓" if monthly_return > 2 and pf > 1.2 else "⚠"
    print(f"\n  {status} Results:")
    print(f"     Trades: {len(trades)}")
    print(f"     Win Rate: {win_rate:.1f}%")
    print(f"     Monthly Return: {monthly_return:.2f}%")
    print(f"     Max Drawdown: {max_dd:.2f}%")
    print(f"     Profit Factor: {pf:.2f}")
    
    return metrics

def main():
    print("\n" + "="*60)
    print("  INDIVIDUAL STRATEGY BACKTEST")
    print("  Testing each strategy separately for optimization")
    print("="*60)
    
    if not initialize_mt5():
        return
    
    results = []
    
    for name, config in STRATEGIES.items():
        metrics = backtest_strategy(name, config, months=3)
        if metrics:
            results.append(metrics)
    
    mt5.shutdown()
    
    # Summary
    print("\n" + "="*60)
    print("  SUMMARY - Strategy Rankings")
    print("="*60)
    
    if results:
        # Sort by monthly return
        results.sort(key=lambda x: x['monthly_return'], reverse=True)
        
        total_estimated_return = 0
        print(f"\n{'Strategy':<25} {'Monthly%':>10} {'MaxDD%':>8} {'WinRate':>8} {'PF':>6} {'Status':>8}")
        print("-" * 70)
        
        for r in results:
            status = "KEEP" if r['monthly_return'] > 2 and r['profit_factor'] > 1.0 else "REVIEW"
            print(f"{r['strategy']:<25} {r['monthly_return']:>10.2f} {r['max_dd']:>8.2f} {r['win_rate']:>7.1f}% {r['profit_factor']:>6.2f} {status:>8}")
            if status == "KEEP":
                total_estimated_return += r['monthly_return'] * 0.15  # Assume 15% allocation each
        
        print("-" * 70)
        print(f"Estimated Portfolio Monthly Return: {total_estimated_return:.2f}%")
        print(f"Target: 13%+ monthly")
        
        # Save results
        output_file = PROJECT_ROOT / 'reports' / 'individual_strategy_results.json'
        output_file.parent.mkdir(exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n✓ Results saved to: {output_file}")
    else:
        print("  ❌ No strategies produced valid results")

if __name__ == '__main__':
    main()
