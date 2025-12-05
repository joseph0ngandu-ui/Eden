#!/usr/bin/env python3
"""
Optimized Portfolio Backtest - Targets: 13% monthly, <4.5% daily DD, <9.5% max DD
"""

import sys
from pathlib import Path
import json
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import MetaTrader5 as mt5
    import pandas as pd
    import numpy as np
    from trading.pro_strategies import ProStrategyEngine
    from trading.ml_portfolio_optimizer import PortfolioMLOptimizer
except ImportError as e:
    print(f"❌ Missing: {e}")
    sys.exit(1)

# HARD TARGETS
MIN_MONTHLY_RETURN = 13.0
MAX_DAILY_DD = 4.5
MAX_OVERALL_DD = 9.5

def initialize_mt5():
    mt5_path = r"C:\Program Files\MetaTrader 5 Terminal\terminal64.exe"
    if not mt5.initialize(path=mt5_path):
        print(f"❌ MT5 init failed")
        return False
    print("✓ MT5 connected")
    return True

def fetch_data(symbol, months=3):
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

def run_optimized_backtest():
    print("\n" + "="*60)
    print("  3-STRATEGY PORTFOLIO 6-MONTH TEST")
    print(f"  Targets: {MIN_MONTHLY_RETURN}% mo | <{MAX_DAILY_DD}% daily DD | <{MAX_OVERALL_DD}% max DD")
    print("="*60)
    
    # 3-Strategy Portfolio symbols
    symbols = ['USDJPYm', 'AUDJPYm', 'EURUSDm', 'GBPUSDm']
    
    data = {}
    print("\nFetching data...")
    for sym in symbols:
        df = fetch_data(sym, 6)  # Extended to 6 months
        if df is not None and len(df) > 1000:
            data[sym] = df
            print(f"  ✓ {sym}: {len(df)} bars")
    
    if not data:
        return None
    
    # Common timeline
    all_times = sorted(set().union(*[set(df.index) for df in data.values()]))
    print(f"\nTotal bars: {len(all_times)}")
    
    engine = ProStrategyEngine()
    optimizer = PortfolioMLOptimizer()
    
    # Trading simulation
    initial_equity = 100000
    equity = initial_equity
    trades = []
    open_positions = {}
    
    # DD Controls - 3-STRATEGY PORTFOLIO
    MAX_DAILY_LOSS = 2.0  # Tight
    MAX_POSITIONS = 2     # Both JPY pairs
    BASE_RISK = 0.75      # Portfolio best config
    
    current_day = None
    day_start_equity = initial_equity
    daily_blocked = False
    peak = initial_equity
    
    print("\nRunning simulation...")
    
    for t_idx, current_time in enumerate(all_times):
        if t_idx < 1000:
            continue
        
        # Progress
        if t_idx % (len(all_times) // 10) == 0:
            pct = t_idx / len(all_times) * 100
            dd = max(0, (peak - equity) / peak * 100)
            print(f"  [{pct:.0f}%] Equity: ${equity:.0f} | DD: {dd:.1f}% | Trades: {len(trades)}")
        
        # Daily reset
        day = current_time.date()
        if day != current_day:
            current_day = day
            day_start_equity = equity
            daily_blocked = False
        
        # Peak tracking
        if equity > peak:
            peak = equity
        
        # Daily loss check
        if not daily_blocked:
            day_pnl = equity - day_start_equity
            if day_pnl < 0:
                day_loss = abs(day_pnl / day_start_equity) * 100
                if day_loss >= MAX_DAILY_LOSS:
                    daily_blocked = True
                    # Close all positions
                    for sym, pos in list(open_positions.items()):
                        if current_time in data[sym].index:
                            bar = data[sym].loc[current_time]
                            exit_p = bar['close']
                            pnl_p = (exit_p - pos['entry']) if pos['dir'] == 'LONG' else (pos['entry'] - exit_p)
                            risk_amt = equity * (pos['risk'] / 100.0)
                            # Size is already calculated correctly
                            pnl = pos['size'] * pnl_p
                            
                            # CURRENCY CONVERSION (Approximate)
                            if 'JPY' in sym:
                                pnl = pnl / exit_p
                                
                            equity += pnl
                            trades.append({'time': current_time, 'pnl': pnl, 'equity': equity, 'result': 'LOSS'})
                        del open_positions[sym]
                    continue
        
        # Close positions (TP/SL)
        for sym, pos in list(open_positions.items()):
            if sym not in data or current_time not in data[sym].index:
                continue
            bar = data[sym].loc[current_time]
            hit_tp = (pos['dir'] == 'LONG' and bar['high'] >= pos['tp']) or \
                     (pos['dir'] == 'SHORT' and bar['low'] <= pos['tp'])
            hit_sl = (pos['dir'] == 'LONG' and bar['low'] <= pos['sl']) or \
                     (pos['dir'] == 'SHORT' and bar['high'] >= pos['sl'])
            
            if hit_tp or hit_sl:
                exit_p = pos['tp'] if hit_tp else pos['sl']
                pnl_p = (exit_p - pos['entry']) if pos['dir'] == 'LONG' else (pos['entry'] - exit_p)
                risk_amt = equity * (pos['risk'] / 100.0)
                # Use stored size
                pnl = pos['size'] * pnl_p
                
                # CURRENCY CONVERSION (Approximate)
                if 'JPY' in sym:
                    pnl = pnl / exit_p
                    
                equity += pnl
                trades.append({'time': current_time, 'pnl': pnl, 'equity': equity, 'result': 'WIN' if hit_tp else 'LOSS'})
                del open_positions[sym]
        
        # New signals
        if len(open_positions) >= MAX_POSITIONS or daily_blocked:
            continue
        
        current_dd = (peak - equity) / peak * 100 if equity < peak else 0
        
        for sym in data.keys():
            if sym in open_positions or current_time not in data[sym].index:
                continue
            
            df_sym = data[sym]
            idx = df_sym.index.get_loc(current_time)
            if idx < 600:
                continue
            
            window = df_sym.iloc[:idx+1]
            signal = engine.evaluate_live(window, sym)
            
            if signal:
                # DD-adjusted risk
                alloc = optimizer.get_allocation({}, current_dd, [signal.strategy])
                weight = alloc.get(signal.strategy, 0.0)
                
                if weight > 0:
                    day_dd = abs((equity - day_start_equity) / day_start_equity) * 100 if equity < day_start_equity else 0
                    risk = optimizer.calculate_position_size(signal.strategy, BASE_RISK, weight, equity, day_dd)
                    
                    if risk > 0.01:
                        # SIZE CALCULATION (Corrected for JPY)
                        # Risk($) = Equity * (Risk%/100)
                        # Size(Units) = Risk($) * ExchangeRate / SL_Dist
                        # For USDJPY, ExchangeRate approx Price
                        
                        risk_amt = equity * (risk / 100.0)
                        sl_dist = abs(signal.entry_price - signal.sl)
                        
                        size = 0
                        if sl_dist > 0:
                            if 'JPY' in sym:
                                size = (risk_amt * signal.entry_price) / sl_dist
                            else:
                                size = risk_amt / sl_dist
                        
                        open_positions[sym] = {
                            'dir': signal.direction,
                            'entry': signal.entry_price,
                            'tp': signal.tp,
                            'sl': signal.sl,
                            'risk': risk,
                            'size': size
                        }
    
    if not trades:
        print("❌ No trades")
        return None
    
    # Metrics
    df_trades = pd.DataFrame(trades)
    final = df_trades['equity'].iloc[-1]
    total_ret = (final - initial_equity) / initial_equity * 100
    monthly_ret = total_ret / 3
    
    # Max DD
    max_dd = 0
    peak_v = initial_equity
    for eq in df_trades['equity']:
        if eq > peak_v:
            peak_v = eq
        dd = (peak_v - eq) / peak_v * 100
        max_dd = max(max_dd, dd)
    
    # Daily DD
    daily_dds = []
    current_day = None
    d_peak = initial_equity
    d_min = initial_equity
    for _, row in df_trades.iterrows():
        day = row['time'].date()
        if day != current_day:
            if current_day:
                daily_dds.append((d_peak - d_min) / d_peak * 100 if d_peak > 0 else 0)
            current_day = day
            d_peak = d_min = row['equity']
        else:
            d_peak = max(d_peak, row['equity'])
            d_min = min(d_min, row['equity'])
    if d_peak > 0:
        daily_dds.append((d_peak - d_min) / d_peak * 100)
    max_daily = max(daily_dds) if daily_dds else 0
    
    wins = len(df_trades[df_trades['result'] == 'WIN'])
    win_rate = wins / len(trades) * 100
    
    gross_profit = df_trades[df_trades['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(df_trades[df_trades['pnl'] < 0]['pnl'].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else 0
    
    meets = monthly_ret >= MIN_MONTHLY_RETURN and max_daily < MAX_DAILY_DD and max_dd < MAX_OVERALL_DD
    
    metrics = {
        'total_trades': len(trades),
        'monthly_return_pct': monthly_ret,
        'max_daily_dd_pct': max_daily,
        'max_drawdown_pct': max_dd,
        'win_rate_pct': win_rate,
        'profit_factor': pf,
        'final_equity': final,
        'meets_criteria': meets
    }
    
    print(f"\n{'Results':-^60}")
    print(f"Monthly Return: {monthly_ret:.2f}% (target: ≥{MIN_MONTHLY_RETURN}%) {'✓' if monthly_ret >= MIN_MONTHLY_RETURN else '✗'}")
    print(f"Max Daily DD: {max_daily:.2f}% (target: <{MAX_DAILY_DD}%) {'✓' if max_daily < MAX_DAILY_DD else '✗'}")
    print(f"Max Overall DD: {max_dd:.2f}% (target: <{MAX_OVERALL_DD}%) {'✓' if max_dd < MAX_OVERALL_DD else '✗'}")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Profit Factor: {pf:.2f}")
    print(f"Final Equity: ${final:.0f}")
    print(f"\n{'✓ PASSES ALL TARGETS' if meets else '✗ FAILS TARGETS'}")
    
    # Save
    with open(PROJECT_ROOT / 'reports' / 'optimized_backtest.json', 'w') as f:
        json.dump({'date': datetime.now().isoformat(), 'metrics': metrics}, f, indent=2, default=str)
    
    return metrics

if __name__ == '__main__':
    if initialize_mt5():
        run_optimized_backtest()
        mt5.shutdown()
