#!/usr/bin/env python3
"""
ML Impact Backtest - Compare performance with and without ML risk adjustment
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading.pro_strategies import ProStrategyEngine
from trading.gold_strategy import GoldMomentumStrategy, GoldConfig
from trading.ml_risk_manager import MLRiskManager

def fetch_data(symbol: str, months: int = 3):
    path = r"C:\Program Files\MetaTrader 5 Terminal\terminal64.exe"
    if not mt5.initialize(path=path): return None
    if not mt5.symbol_select(symbol, True): return None
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=months * 30)
    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M5, start_date, end_date)
    mt5.shutdown()
    
    if rates is None or len(rates) == 0: return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df

def run_backtest(use_ml: bool = False):
    """Run backtest with or without ML."""
    print(f"\n{'='*80}")
    print(f"Running backtest {'WITH' if use_ml else 'WITHOUT'} ML Risk Adjustment")
    print(f"{'='*80}\n")
    
    # Fetch data
    symbols = {
        'EURUSD': {'strategy': 'pro', 'base_risk': 0.15},
        'GBPUSD': {'strategy': 'pro', 'base_risk': 0.15},
        'USDJPY': {'strategy': 'pro', 'base_risk': 0.15},
        'XAUUSD': {'strategy': 'gold', 'base_risk': 0.5}
    }
    
    data = {}
    for symbol in symbols.keys():
        df = fetch_data(symbol)
        if df is not None:
            data[symbol] = df
            print(f"Loaded {symbol}: {len(df)} bars")
    
    if len(data) < 2:
        print("Insufficient data")
        return None
    
    # Initialize strategies and ML manager
    pro_engine = ProStrategyEngine()
    gold_strategy = GoldMomentumStrategy(GoldConfig())
    ml_manager = MLRiskManager(enable_ml=use_ml) if use_ml else None
    
    # Portfolio simulation
    equity = 100000.0
    initial_equity = 100000.0
    closed_trades = []
    open_trades = {}
    trade_id = 0
    
    # Get common time range
    all_times = set()
    for df in data.values():
        all_times.update(df.index)
    all_times = sorted(list(all_times))
    
    print(f"Simulating {len(all_times)} time points...\n")
    
    for idx, current_time in enumerate(all_times):
        if idx < 1000:
            continue
        
        # Close trades
        for tid, trade in list(open_trades.items()):
            symbol = trade['symbol']
            if symbol not in data or current_time not in data[symbol].index:
                continue
            
            bar = data[symbol].loc[current_time]
            
            hit_tp = (trade['dir'] == 'LONG' and bar['high'] >= trade['tp']) or \
                     (trade['dir'] == 'SHORT' and bar['low'] <= trade['tp'])
            hit_sl = (trade['dir'] == 'LONG' and bar['low'] <= trade['sl']) or \
                     (trade['dir'] == 'SHORT' and bar['high'] >= trade['sl'])
            
            if hit_tp or hit_sl:
                exit_price = trade['tp'] if hit_tp else trade['sl']
                pnl_price = (exit_price - trade['entry']) if trade['dir'] == 'LONG' else (trade['entry'] - exit_price)
                
                risk_amount = equity * (trade['risk_pct'] / 100.0)
                sl_dist = abs(trade['entry'] - trade['sl'])
                
                if sl_dist > 0:
                    pos_size = risk_amount / sl_dist
                    pnl = pos_size * pnl_price
                else:
                    pnl = 0
                
                equity += pnl
                closed_trades.append({
                    'time': current_time,
                    'symbol': symbol,
                    'pnl': pnl,
                    'equity': equity,
                    'result': 'TP' if hit_tp else 'SL'
                })
                del open_trades[tid]
        
        # Generate new signals
        if len(open_trades) >= 7:
            continue
        
        for symbol, config in symbols.items():
            if symbol not in data or current_time not in data[symbol].index:
                continue
            if len(open_trades) >= 7:
                break
            
            # Get window
            df_symbol = data[symbol]
            window_idx = df_symbol.index.get_loc(current_time)
            if window_idx < 600:
                continue
            
            window = df_symbol.iloc[:window_idx+1].reset_index()
            
            # Get signal
            signal = None
            if config['strategy'] == 'gold':
                signal = gold_strategy.generate_signal(window.set_index('time'))
            else:
                signal = pro_engine.evaluate_live(window, symbol)
            
            if signal:
                # Extract signal data
                if hasattr(signal, 'entry_price'):
                    entry = signal.entry_price
                    tp = signal.tp
                    sl = signal.sl
                    direction = signal.direction
                else:
                    entry = signal['entry_price']
                    tp = signal['tp_price']
                    sl = signal['sl_price']
                    direction = signal['direction']
                
                # Determine risk
                base_risk = config['base_risk']
                if use_ml and ml_manager:
                    # ML adjusts risk
                    risk_pct = ml_manager.adjust_risk(
                        window.set_index('time'), 
                        base_risk, 
                        direction, 
                        symbol
                    )
                else:
                    risk_pct = base_risk
                
                # Open trade
                trade_id += 1
                open_trades[trade_id] = {
                    'symbol': symbol,
                    'dir': direction,
                    'entry': entry,
                    'tp': tp,
                    'sl': sl,
                    'risk_pct': risk_pct
                }
    
    # Results
    if not closed_trades:
        print("No trades")
        return None
    
    df_trades = pd.DataFrame(closed_trades)
    total_return = ((equity - initial_equity) / initial_equity) * 100
    
    peak = initial_equity
    max_dd = 0
    for t in closed_trades:
        e = t['equity']
        if e > peak: peak = e
        dd = (peak - e) / peak * 100
        max_dd = max(max_dd, dd)
    
    wins = df_trades[df_trades['result'] == 'TP']
    win_rate = (len(wins) / len(df_trades)) * 100
    
    return {
        'ml_enabled': use_ml,
        'trades': len(df_trades),
        'return': total_return,
        'max_dd': max_dd,
        'win_rate': win_rate
    }

if __name__ == "__main__":
    # Run both backtests
    print("Fetching data...")
    
    result_baseline = run_backtest(use_ml=False)
    result_ml = run_backtest(use_ml=True)
    
    if not result_baseline or not result_ml:
        print("Backtest failed")
        exit(1)
    
    # Compare
    print(f"\n\n{'='*80}")
    print("RESULTS COMPARISON")
    print(f"{'='*80}")
    print(f"{'Metric':<20} {'Baseline':<15} {'With ML':<15} {'Change':<15}")
    print("-" * 80)
    
    trades_change = result_ml['trades'] - result_baseline['trades']
    return_change = result_ml['return'] - result_baseline['return']
    dd_change = result_ml['max_dd'] - result_baseline['max_dd']
    wr_change = result_ml['win_rate'] - result_baseline['win_rate']
    
    print(f"{'Trades':<20} {result_baseline['trades']:<15} {result_ml['trades']:<15} {trades_change:+.0f}")
    print(f"{'Return (%)':<20} {result_baseline['return']:<15.2f} {result_ml['return']:<15.2f} {return_change:+.2f}%")
    print(f"{'MaxDD (%)':<20} {result_baseline['max_dd']:<15.2f} {result_ml['max_dd']:<15.2f} {dd_change:+.2f}%")
    print(f"{'Win Rate (%)':<20} {result_baseline['win_rate']:<15.1f} {result_ml['win_rate']:<15.1f} {wr_change:+.1f}%")
    
    print(f"\n{'='*80}")
    
    # Verdict
    if return_change > 2 and dd_change < 1:
        print("✅ VERDICT: ML IMPROVES profitability with acceptable drawdown increase")
        print("   RECOMMENDATION: Enable ML")
    elif return_change > 0 and dd_change > 2:
        print("⚠️  VERDICT: ML increases return but also increases drawdown significantly")
        print("   RECOMMENDATION: Consider enabling only if you can tolerate higher DD")
    elif return_change < -2:
        print("❌ VERDICT: ML REDUCES profitability")
        print("   RECOMMENDATION: Keep ML disabled")
    else:
        print("➡️  VERDICT: ML has minimal impact (neutral)")
        print("   RECOMMENDATION: Keep ML disabled (not worth the complexity)")
