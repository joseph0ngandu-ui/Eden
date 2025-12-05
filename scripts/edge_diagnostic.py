#!/usr/bin/env python3
"""
EDGE DIAGNOSTIC - Find the REAL Edge
Tests each strategy in isolation with NO risk scaling to find true edge.
This answers: "Does this strategy actually make money?"
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

# Strategy configurations
STRATEGIES = {
    'asian_fade': {
        'symbols': ['USDJPYm', 'AUDJPYm'],
        'jpy_pair': True,  # Important for pip value
    },
    'mean_reversion': {
        'symbols': ['EURUSDm', 'GBPUSDm', 'AUDJPYm'],
        'jpy_pair': False,
    },
    'overlap_scalper': {
        'symbols': ['EURUSDm', 'GBPUSDm'],
        'jpy_pair': False,
    },
    'gold_london_breakout': {
        'symbols': ['XAUUSDm'],
        'jpy_pair': False,
    },
    'trend_follower': {
        'symbols': ['EURUSDm', 'GBPUSDm', 'USDJPYm'],
        'jpy_pair': False,
    },
    'index_momentum_trend': {
        'symbols': ['US30m', 'US500m', 'USTECm'],
        'jpy_pair': False,
    },
}

def initialize_mt5():
    mt5_path = r"C:\Program Files\MetaTrader 5 Terminal\terminal64.exe"
    if not mt5.initialize(path=mt5_path):
        print(f"❌ MT5 init failed: {mt5.last_error()}")
        return False
    print("✓ MT5 connected")
    return True

def fetch_data(symbol: str, months: int = 6):
    """Fetch 6 months of data for proper statistical significance"""
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

def test_strategy_edge(strategy_name: str, config: dict, test_months: int = 6):
    """
    Test a strategy with FLAT RISK (1R per trade) to find true edge.
    No scaling, no circuit breakers. Pure strategy performance.
    """
    from trading.pro_strategies import ProStrategyEngine
    
    print(f"\n{'='*60}")
    print(f"  EDGE TEST: {strategy_name}")
    print(f"{'='*60}")
    
    engine = ProStrategyEngine()
    strategy_method = getattr(engine, strategy_name)
    
    # Fetch data
    all_data = {}
    for symbol in config['symbols']:
        df = fetch_data(symbol, test_months)
        if df is not None and len(df) > 5000:
            all_data[symbol] = df
            print(f"  ✓ {symbol}: {len(df)} bars ({len(df)/(12*24*22):.1f} months)")
        else:
            print(f"  ⚠ {symbol}: insufficient data")
    
    if not all_data:
        return None
    
    # Get common timeline
    all_times = set()
    for df in all_data.values():
        all_times.update(df.index)
    all_times = sorted(list(all_times))
    
    print(f"  Testing {len(all_times)} time points (~{len(all_times)/(12*24*22):.1f} months)")
    
    # SIMULATE WITH FLAT 1R RISK
    initial_equity = 100000
    equity = initial_equity
    risk_per_trade = 0.01  # 1% flat risk - NO SCALING
    
    trades = []
    wins = 0
    losses = 0
    open_positions = {}
    
    # Track R multiples
    r_multiples = []
    
    for t_idx, current_time in enumerate(all_times):
        if t_idx < 500:
            continue
        
        # Close positions
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
                
                # Calculate R multiple (how many R did we win/lose)
                sl_dist = abs(pos['entry'] - pos['sl'])
                if sl_dist > 0:
                    pnl_dist = abs(exit_price - pos['entry'])
                    r_mult = pnl_dist / sl_dist if hit_tp else -1.0
                    r_multiples.append(r_mult)
                    
                    # PnL in dollars (1R = 1% of equity = $1000)
                    pnl = initial_equity * risk_per_trade * r_mult
                else:
                    pnl = 0
                    r_mult = 0
                
                equity += pnl
                
                if hit_tp:
                    wins += 1
                else:
                    losses += 1
                
                trades.append({
                    'time': current_time,
                    'symbol': symbol,
                    'direction': pos['dir'],
                    'entry': pos['entry'],
                    'exit': exit_price,
                    'result': 'WIN' if hit_tp else 'LOSS',
                    'r_multiple': r_mult,
                    'pnl': pnl,
                    'equity': equity
                })
                del open_positions[symbol]
        
        # Generate new signals (max 2 concurrent per strategy)
        if len(open_positions) >= 2:
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
                        'sl': signal.sl,
                        'time': current_time
                    }
            except Exception as e:
                pass
    
    if len(trades) == 0:
        print(f"  ❌ No trades generated")
        return None
    
    # Calculate metrics
    total_trades = wins + losses
    win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
    
    avg_win_r = np.mean([r for r in r_multiples if r > 0]) if any(r > 0 for r in r_multiples) else 0
    avg_loss_r = abs(np.mean([r for r in r_multiples if r < 0])) if any(r < 0 for r in r_multiples) else 1
    
    # Expectancy (R per trade)
    expectancy = (win_rate/100 * avg_win_r) - ((1 - win_rate/100) * avg_loss_r)
    
    # Profit factor
    gross_win = sum(r for r in r_multiples if r > 0)
    gross_loss = abs(sum(r for r in r_multiples if r < 0))
    profit_factor = gross_win / gross_loss if gross_loss > 0 else 0
    
    # Total return
    total_return = ((equity - initial_equity) / initial_equity) * 100
    monthly_return = total_return / test_months if test_months > 0 else 0
    
    # MAX DRAWDOWN
    peak = initial_equity
    max_dd = 0
    for trade in trades:
        if trade['equity'] > peak:
            peak = trade['equity']
        dd = ((peak - trade['equity']) / peak) * 100
        max_dd = max(max_dd, dd)
    
    # Print results
    print(f"\n  📊 EDGE ANALYSIS:")
    print(f"     Total Trades:     {total_trades}")
    print(f"     Win Rate:         {win_rate:.1f}%")
    print(f"     Avg Winner:       {avg_win_r:.2f}R")
    print(f"     Avg Loser:        {avg_loss_r:.2f}R")
    print(f"     EXPECTANCY:       {expectancy:.3f}R per trade")
    print(f"     Profit Factor:    {profit_factor:.2f}")
    print(f"     Monthly Return:   {monthly_return:.2f}% (at 1% risk)")
    print(f"     Max Drawdown:     {max_dd:.2f}%")
    
    # Verdict
    if expectancy > 0.3 and profit_factor > 1.2:
        print(f"\n  ✅ STRONG EDGE - Worth trading")
    elif expectancy > 0.1 and profit_factor > 1.1:
        print(f"\n  ⚠️ WEAK EDGE - Trade with caution")
    else:
        print(f"\n  ❌ NO EDGE - Do not trade")
    
    return {
        'strategy': strategy_name,
        'trades': total_trades,
        'win_rate': win_rate,
        'avg_win_r': avg_win_r,
        'avg_loss_r': avg_loss_r,
        'expectancy': expectancy,
        'profit_factor': profit_factor,
        'monthly_return': monthly_return,
        'max_dd': max_dd
    }

def calculate_optimal_portfolio(results):
    """Calculate optimal allocation based on edge strength"""
    print(f"\n{'='*60}")
    print("  OPTIMAL PORTFOLIO ALLOCATION")
    print(f"{'='*60}")
    
    # Filter strategies with positive edge
    profitable = [r for r in results if r['expectancy'] > 0 and r['profit_factor'] > 1.0]
    
    if not profitable:
        print("  ❌ No strategies with positive edge found!")
        return
    
    # Score each strategy: expectancy * trades_per_month (opportunity)
    for r in profitable:
        trades_per_month = r['trades'] / 6  # Assuming 6 months of data
        r['score'] = r['expectancy'] * min(trades_per_month, 50)  # Cap trades
        r['trades_per_month'] = trades_per_month
    
    # Sort by score
    profitable.sort(key=lambda x: x['score'], reverse=True)
    
    # Allocate based on score
    total_score = sum(r['score'] for r in profitable)
    
    print(f"\n  Strategy Allocations (by edge strength):\n")
    print(f"  {'Strategy':<25} {'Expect':<8} {'PF':<6} {'Score':<8} {'Alloc':<8}")
    print(f"  {'-'*55}")
    
    allocations = {}
    for r in profitable:
        alloc = (r['score'] / total_score) * 100 if total_score > 0 else 0
        allocations[r['strategy']] = alloc
        print(f"  {r['strategy']:<25} {r['expectancy']:.3f}R  {r['profit_factor']:.2f}   {r['score']:.2f}     {alloc:.1f}%")
    
    # Estimate combined monthly return
    combined_monthly = sum(r['monthly_return'] * (allocations[r['strategy']]/100) for r in profitable)
    print(f"\n  📈 Estimated Combined Monthly Return: {combined_monthly:.2f}%")
    
    # Risk recommendation
    if combined_monthly > 20:
        print(f"  💡 Can use 1.5-2% risk per trade to hit 13%+ monthly")
    elif combined_monthly > 10:
        print(f"  💡 Need 1.5% risk per trade to hit 13%+ monthly")
    elif combined_monthly > 5:
        print(f"  💡 Need 2.5% risk per trade - RISKY")
    else:
        print(f"  ⚠️ Strategies lack sufficient edge for 13% target")
    
    return allocations

def main():
    print("\n" + "="*60)
    print("  EDGE DIAGNOSTIC - Finding True Strategy Profitability")
    print("  NO scaling, NO circuit breakers - Pure strategy edge")
    print("="*60)
    
    if not initialize_mt5():
        return
    
    results = []
    
    for name, config in STRATEGIES.items():
        result = test_strategy_edge(name, config, test_months=6)
        if result:
            results.append(result)
    
    mt5.shutdown()
    
    if results:
        # Optimal allocation
        allocations = calculate_optimal_portfolio(results)
        
        # Summary
        print(f"\n{'='*60}")
        print("  SUMMARY - Strategy Edge Results")
        print(f"{'='*60}")
        
        results.sort(key=lambda x: x['expectancy'], reverse=True)
        
        print(f"\n  {'Strategy':<25} {'Expect':<10} {'PF':<8} {'WinRate':<10} {'Verdict':<12}")
        print(f"  {'-'*65}")
        
        for r in results:
            if r['expectancy'] > 0.3 and r['profit_factor'] > 1.2:
                verdict = "✅ STRONG"
            elif r['expectancy'] > 0.1 and r['profit_factor'] > 1.1:
                verdict = "⚠️ WEAK"
            else:
                verdict = "❌ NO EDGE"
            
            print(f"  {r['strategy']:<25} {r['expectancy']:.3f}R     {r['profit_factor']:.2f}    {r['win_rate']:.1f}%      {verdict}")
        
        # Save results
        output_file = PROJECT_ROOT / 'reports' / 'edge_diagnostic.json'
        output_file.parent.mkdir(exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump({
                'test_date': datetime.now().isoformat(),
                'test_months': 6,
                'risk_per_trade': '1%',
                'results': results,
                'allocations': allocations
            }, f, indent=2, default=str)
        print(f"\n  ✓ Results saved to: {output_file}")
    else:
        print("\n  ❌ No strategies produced valid results")

if __name__ == '__main__':
    main()
