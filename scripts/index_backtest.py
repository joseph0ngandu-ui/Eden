#!/usr/bin/env python3
"""
US Index Strategy Backtest
Tests index-specific strategies on US30m, US500m, USTECm
Target: 13% monthly return, <4.5% daily DD, <9.5% max DD
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

# Index Strategy definitions
INDEX_STRATEGIES = {
    'Pro_Index_NY_Breakout': {
        'symbols': ['US30m', 'US500m', 'USTECm'],
        'method': 'index_ny_breakout'
    },
    'Pro_Index_Momentum': {
        'symbols': ['US30m', 'US500m', 'USTECm'],
        'method': 'index_momentum_trend'
    },
    'Pro_Index_Mean_Reversion': {
        'symbols': ['US30m', 'US500m', 'USTECm'],
        'method': 'index_mean_reversion'
    }
}

# Performance targets
MIN_MONTHLY_RETURN = 13.0
MAX_DAILY_DD = 4.5
MAX_OVERALL_DD = 9.5

def initialize_mt5():
    mt5_path = r"C:\Program Files\MetaTrader 5 Terminal\terminal64.exe"
    if not mt5.initialize(path=mt5_path):
        print(f"❌ MT5 init failed: {mt5.last_error()}")
        return False
    print("✓ MT5 connected")
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
    """Backtest a single index strategy on its target symbols."""
    from trading.pro_strategies import ProStrategyEngine
    
    print(f"\n{'='*60}")
    print(f"  {strategy_name}")
    print(f"{'='*60}")
    
    engine = ProStrategyEngine()
    strategy_method = getattr(engine, strategy_config['method'])
    
    # Fetch data for all symbols
    all_data = {}
    for symbol in strategy_config['symbols']:
        df = fetch_data(symbol, months)
        if df is not None and len(df) > 500:
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
    risk_per_trade = 0.005  # 0.5% risk per trade (balanced for returns + DD control)
    last_trade_time = {}  # Cooldown tracking
    
    # Get common timeline
    all_times = set()
    for df in all_data.values():
        all_times.update(df.index)
    all_times = sorted(list(all_times))
    
    print(f"  Simulating {len(all_times)} time points...")
    
    # Track daily equity for DD calculation
    daily_equity = {}
    
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
        
        # Track daily equity
        day_key = current_time.date()
        if day_key not in daily_equity:
            daily_equity[day_key] = {'open': equity, 'high': equity, 'low': equity, 'close': equity}
        else:
            daily_equity[day_key]['high'] = max(daily_equity[day_key]['high'], equity)
            daily_equity[day_key]['low'] = min(daily_equity[day_key]['low'], equity)
            daily_equity[day_key]['close'] = equity
        
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
    
    # Overall Drawdown
    peak = initial_equity
    max_dd = 0
    for eq in df_trades['equity']:
        if eq > peak:
            peak = eq
        dd = ((peak - eq) / peak) * 100
        max_dd = max(max_dd, dd)
    
    # Daily Drawdown
    max_daily_dd = 0
    for day, vals in daily_equity.items():
        if vals['open'] > 0:
            day_dd = ((vals['high'] - vals['low']) / vals['high']) * 100
            max_daily_dd = max(max_daily_dd, day_dd)
    
    # Profit factor
    gross_profit = df_trades[df_trades['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(df_trades[df_trades['pnl'] < 0]['pnl'].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else 0
    
    # Check if meets criteria
    meets_criteria = (
        monthly_return >= MIN_MONTHLY_RETURN and
        max_daily_dd < MAX_DAILY_DD and
        max_dd < MAX_OVERALL_DD and
        pf > 1.0
    )
    
    metrics = {
        'strategy': strategy_name,
        'trades': len(trades),
        'win_rate': win_rate,
        'monthly_return': monthly_return,
        'max_dd': max_dd,
        'max_daily_dd': max_daily_dd,
        'profit_factor': pf,
        'final_equity': final_equity,
        'meets_criteria': meets_criteria
    }
    
    # Print results
    status = "✓ PASS" if meets_criteria else "✗ REVIEW"
    print(f"\n  {status}")
    print(f"     Trades: {len(trades)}")
    print(f"     Win Rate: {win_rate:.1f}%")
    print(f"     Monthly Return: {monthly_return:.2f}% (target: ≥{MIN_MONTHLY_RETURN}%)")
    print(f"     Max Daily DD: {max_daily_dd:.2f}% (target: <{MAX_DAILY_DD}%)")
    print(f"     Max Overall DD: {max_dd:.2f}% (target: <{MAX_OVERALL_DD}%)")
    print(f"     Profit Factor: {pf:.2f}")
    
    return metrics

def main():
    print("\n" + "="*60)
    print("  US INDEX STRATEGY BACKTEST")
    print(f"  Target: {MIN_MONTHLY_RETURN}% monthly | <{MAX_DAILY_DD}% daily DD | <{MAX_OVERALL_DD}% max DD")
    print("="*60)
    
    if not initialize_mt5():
        return
    
    results = []
    passing_strategies = []
    
    for name, config in INDEX_STRATEGIES.items():
        metrics = backtest_strategy(name, config, months=3)
        if metrics:
            results.append(metrics)
            if metrics['meets_criteria']:
                passing_strategies.append(name)
    
    mt5.shutdown()
    
    # Summary
    print("\n" + "="*60)
    print("  SUMMARY - Index Strategy Rankings")
    print("="*60)
    
    if results:
        # Sort by monthly return
        results.sort(key=lambda x: x['monthly_return'], reverse=True)
        
        print(f"\n{'Strategy':<30} {'Monthly%':>10} {'MaxDD%':>8} {'DailyDD%':>10} {'PF':>6} {'Status':>10}")
        print("-" * 80)
        
        for r in results:
            status = "PASS ✓" if r['meets_criteria'] else "REVIEW"
            print(f"{r['strategy']:<30} {r['monthly_return']:>10.2f} {r['max_dd']:>8.2f} {r['max_daily_dd']:>10.2f} {r['profit_factor']:>6.2f} {status:>10}")
        
        print("-" * 80)
        
        if passing_strategies:
            print(f"\n✓ Strategies passing all criteria: {len(passing_strategies)}")
            for name in passing_strategies:
                print(f"  • {name}")
            print("\n💰 RECOMMENDATION: Add these to portfolio for optimization")
        else:
            print("\n⚠ No strategies meet all criteria yet - may need parameter tuning")
        
        # Save results
        output_file = PROJECT_ROOT / 'reports' / 'index_strategy_results.json'
        output_file.parent.mkdir(exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump({
                'test_date': datetime.now().isoformat(),
                'results': results,
                'passing_strategies': passing_strategies
            }, f, indent=2, default=str)
        print(f"\n✓ Results saved to: {output_file}")
    else:
        print("  ❌ No strategies produced valid results")

if __name__ == '__main__':
    main()
