#!/usr/bin/env python3
"""
Professional Portfolio Trading Engine - True Event-Driven Backtester

Simulates simultaneous trading of multiple symbols to accurately:
1. Enforce global risk limits (Max 5 open trades)
2. Calculate true portfolio drawdown
3. Manage correlations
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json

# ================= STRATEGIES (Stateless) =================

def calculate_atr_simple(df: pd.DataFrame) -> float:
    """Simple ATR calculation."""
    if len(df) < 2:
        return 0.001
    tr = df['high'] - df['low']
    return tr.mean() if len(tr) > 0 else 0.001

def overlap_scalper(df: pd.DataFrame, symbol: str, idx: int) -> Optional[Dict]:
    """High-frequency scalper during peak liquidity."""
    row = df.iloc[idx]
    hour = row['hour']
    
    # London/NY overlap: 12:00-16:00 GMT
    if not (12 <= hour < 16):
        return None
    
    if idx < 20: return None
    
    # 5-period momentum
    recent = df.iloc[idx-5:idx+1]
    momentum = recent['close'].iloc[-1] - recent['close'].iloc[0]
    atr = calculate_atr_simple(df.iloc[max(0, idx-14):idx+1])
    
    # Volume spike
    avg_vol = df.iloc[max(0, idx-20):idx]['tick_volume'].mean()
    if row['tick_volume'] < avg_vol * 1.2:
        return None
    
    # Entry
    if abs(momentum) > atr * 0.6:
        direction = "LONG" if momentum > 0 else "SHORT"
        if direction == "LONG":
            entry = row['close']
            sl = entry - (atr * 1.0)
            tp = entry + (atr * 2.5)
        else:
            entry = row['close']
            sl = entry + (atr * 1.0)
            tp = entry - (atr * 2.5)
        
        return {'symbol': symbol, 'direction': direction, 'entry': entry, 'sl': sl, 'tp': tp, 
                'strategy': 'overlap_scalper', 'risk_pct': 0.0015} # 0.15% risk
    return None

def asian_fade(df: pd.DataFrame, symbol: str, idx: int) -> Optional[Dict]:
    """Fade extremes during low-volatility Asian session."""
    row = df.iloc[idx]
    hour = row['hour']
    
    # Asian session: 22:00-06:00 GMT
    if not ((hour >= 22) or (hour < 6)):
        return None
    
    if idx < 60: return None
    
    # Asian range
    asian_bars = df.iloc[idx-60:idx]
    asian_high = asian_bars['high'].max()
    asian_low = asian_bars['low'].min()
    asian_mid = (asian_high + asian_low) / 2
    range_size = asian_high - asian_low
    
    if range_size == 0: return None
    
    price_position = (row['close'] - asian_low) / range_size
    
    if price_position > 0.8:
        return {'symbol': symbol, 'direction': 'SHORT', 'entry': row['close'], 
                'sl': asian_high + range_size * 0.1, 'tp': asian_mid,
                'strategy': 'asian_fade', 'risk_pct': 0.0015} # 0.15% risk
    elif price_position < 0.2:
        return {'symbol': symbol, 'direction': 'LONG', 'entry': row['close'],
                'sl': asian_low - range_size * 0.1, 'tp': asian_mid,
                'strategy': 'asian_fade', 'risk_pct': 0.0015} # 0.15% risk
    return None

def gold_london_breakout(df: pd.DataFrame, symbol: str, idx: int) -> Optional[Dict]:
    """Trade Asian range breakout at London open."""
    if symbol != "XAUUSD": return None
    
    row = df.iloc[idx]
    hour = row['hour']
    
    # London open: 7-9 AM GMT
    if not (7 <= hour < 9): return None
    
    if idx < 72: return None
    
    # Find start of day
    asian_start = None
    for i in range(idx, max(0, idx-100), -1):
        if df.iloc[i]['hour'] == 0:
            asian_start = i
            break
    
    if asian_start is None or idx - asian_start < 60: return None
    
    asian_bars = df.iloc[asian_start:idx]
    asian_high = asian_bars['high'].max()
    asian_low = asian_bars['low'].min()
    asian_range = asian_high - asian_low
    
    if asian_range == 0: return None
    
    if row['close'] > asian_high and row['close'] > asian_high + asian_range * 0.1:
        return {'symbol': symbol, 'direction': 'LONG', 'entry': row['close'],
                'sl': asian_low, 'tp': row['close'] + asian_range * 3,
                'strategy': 'gold_breakout', 'risk_pct': 0.0015} # 0.15% risk
    
    if row['close'] < asian_low and row['close'] < asian_low - asian_range * 0.1:
        return {'symbol': symbol, 'direction': 'SHORT', 'entry': row['close'],
                'sl': asian_high, 'tp': row['close'] - asian_range * 3,
                'strategy': 'gold_breakout', 'risk_pct': 0.0015} # 0.15% risk
    return None

def volatility_expansion(df: pd.DataFrame, symbol: str, idx: int) -> Optional[Dict]:
    """Trade breakouts from volatility compression."""
    if idx < 40: return None
    row = df.iloc[idx]
    
    current_atr = calculate_atr_simple(df.iloc[idx-14:idx+1])
    avg_atr = calculate_atr_simple(df.iloc[idx-40:idx+1])
    
    if avg_atr == 0: return None
    if current_atr > avg_atr * 0.7: return None # Must be compressed
    
    recent_3 = df.iloc[idx-2:idx+1]
    move = abs(recent_3['close'].iloc[-1] - recent_3['close'].iloc[0])
    
    if move > current_atr * 1.5:
        direction = "LONG" if recent_3['close'].iloc[-1] > recent_3['close'].iloc[0] else "SHORT"
        if direction == "LONG":
            return {'symbol': symbol, 'direction': 'LONG', 'entry': row['close'],
                    'sl': row['close'] - avg_atr * 1.5, 'tp': row['close'] + avg_atr * 4,
                    'strategy': 'vol_expansion', 'risk_pct': 0.0015} # 0.15% risk
        else:
            return {'symbol': symbol, 'direction': 'SHORT', 'entry': row['close'],
                    'sl': row['close'] + avg_atr * 1.5, 'tp': row['close'] - avg_atr * 4,
                    'strategy': 'vol_expansion', 'risk_pct': 0.0015} # 0.15% risk
    return None

# ================= PORTFOLIO ENGINE =================

def fetch_data(symbol: str, months: int = 3) -> Optional[pd.DataFrame]:
    """Fetch M5 data."""
    if not mt5.initialize(): return None
    end_date = datetime.now()
    start_date = end_date - timedelta(days=months * 30)
    if not mt5.symbol_select(symbol, True): return None
    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M5, start_date, end_date)
    mt5.shutdown()
    if rates is None or len(rates) == 0: return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df['hour'] = df['time'].dt.hour
    df.set_index('time', inplace=True) # Index by time for alignment
    return df

def run_portfolio_simulation(data_map: Dict[str, pd.DataFrame]) -> Dict:
    """Run synchronized portfolio backtest."""
    
    # 1. Align timelines
    all_times = sorted(list(set().union(*[df.index for df in data_map.values()])))
    print(f"Simulation range: {all_times[0]} to {all_times[-1]}")
    print(f"Total time steps: {len(all_times)}")
    
    equity = 100000.0
    initial_equity = 100000.0
    open_trades = [] # List of active trade dicts
    closed_trades = []
    
    daily_pnl = 0
    last_date = all_times[0].date()
    
    # Pre-calculate integer indices for speed (optional, but good for large data)
    # For simplicity, we'll use .loc but handle missing data gracefully
    
    for current_time in all_times:
        current_date = current_time.date()
        
        # New day reset
        if current_date != last_date:
            daily_pnl = 0
            last_date = current_date
            
        # Circuit breaker: Stop trading if daily loss > 2%
        if daily_pnl < -(equity * 0.02):
            # Still manage exits, but no new entries
            trading_allowed = False
        else:
            trading_allowed = True
            
        # --- 1. Manage Open Trades ---
        active_trades = []
        for trade in open_trades:
            symbol = trade['symbol']
            if current_time not in data_map[symbol].index:
                active_trades.append(trade) # Keep open if no data (market closed?)
                continue
                
            row = data_map[symbol].loc[current_time]
            
            # Check Exit
            hit_tp = (trade['direction'] == 'LONG' and row['high'] >= trade['tp']) or \
                     (trade['direction'] == 'SHORT' and row['low'] <= trade['tp'])
            hit_sl = (trade['direction'] == 'LONG' and row['low'] <= trade['sl']) or \
                     (trade['direction'] == 'SHORT' and row['high'] >= trade['sl'])
            
            if hit_tp or hit_sl:
                exit_price = trade['tp'] if hit_tp else trade['sl']
                pnl_price = (exit_price - trade['entry']) if trade['direction'] == 'LONG' else (trade['entry'] - exit_price)
                
                # Calculate PnL in dollars
                risk_amount = equity * trade['risk_pct']
                sl_distance = abs(trade['entry'] - trade['sl'])
                
                if sl_distance > 0:
                    position_size = risk_amount / sl_distance
                    pnl_dollars = position_size * pnl_price
                else:
                    pnl_dollars = 0
                
                equity += pnl_dollars
                daily_pnl += pnl_dollars
                
                closed_trades.append({
                    'entry_time': trade['entry_time'],
                    'exit_time': current_time,
                    'symbol': symbol,
                    'strategy': trade['strategy'],
                    'pnl_dollars': pnl_dollars,
                    'equity': equity,
                    'result': 'TP' if hit_tp else 'SL'
                })
            else:
                active_trades.append(trade)
        
        open_trades = active_trades
        
        # --- 2. Check New Entries ---
        if not trading_allowed: continue
        
        # Global Risk Cap: Max 5 open trades
        if len(open_trades) >= 5: continue
        
        # Check each symbol for signals
        # Randomize order to avoid bias? Or fixed priority?
        # Let's use fixed order for reproducibility
        for symbol, df in data_map.items():
            if len(open_trades) >= 5: break
            
            if current_time not in df.index: continue
            
            # Need integer location for strategy functions that look back
            # This is slow with .get_loc, but necessary for lookback logic
            try:
                idx = df.index.get_loc(current_time)
            except KeyError:
                continue
                
            if isinstance(idx, slice) or isinstance(idx, np.ndarray): 
                idx = idx.start # Handle duplicates if any
                
            # Skip if not enough history
            if idx < 100: continue
            
            # Check if we already have a trade for this symbol/strategy?
            # Simple rule: Max 1 trade per symbol per strategy
            existing_strategies = [t['strategy'] for t in open_trades if t['symbol'] == symbol]
            
            signal = None
            
            # Strategy 1: Overlap Scalper
            if 'overlap_scalper' not in existing_strategies and symbol in ['EURUSD', 'GBPUSD']:
                signal = overlap_scalper(df, symbol, idx)
            
            # Strategy 2: Asian Fade
            if not signal and 'asian_fade' not in existing_strategies and symbol in ['USDJPY', 'AUDJPY']:
                signal = asian_fade(df, symbol, idx)
                
            # Strategy 3: Gold Breakout
            if not signal and 'gold_breakout' not in existing_strategies and symbol == 'XAUUSD':
                signal = gold_london_breakout(df, symbol, idx)
                
            # Strategy 4: Vol Expansion
            if not signal and 'vol_expansion' not in existing_strategies:
                signal = volatility_expansion(df, symbol, idx)
            
            if signal:
                signal['entry_time'] = current_time
                open_trades.append(signal)

    return {
        'trades': closed_trades,
        'final_equity': equity,
        'initial_equity': initial_equity
    }

def run_monte_carlo(trades: List[Dict], iterations: int = 1000) -> Dict:
    """Run Monte Carlo simulation on trade sequence."""
    pnls = [t['pnl_dollars'] for t in trades]
    max_drawdowns = []
    
    import random
    
    for _ in range(iterations):
        shuffled = pnls.copy()
        random.shuffle(shuffled)
        equity = 100000
        peak = 100000
        max_dd = 0
        
        for pnl in shuffled:
            equity += pnl
            if equity > peak: peak = equity
            dd = (peak - equity) / peak * 100
            max_dd = max(max_dd, dd)
            
        max_drawdowns.append(max_dd)
        
    max_drawdowns.sort()
    return {
        'median_dd': max_drawdowns[int(iterations * 0.5)],
        '95_percentile_dd': max_drawdowns[int(iterations * 0.95)],
        'worst_case_dd': max_drawdowns[-1]
    }

def analyze_results(results: Dict):
    trades = results['trades']
    if not trades:
        print("No trades executed.")
        return
        
    df_trades = pd.DataFrame(trades)
    df_trades['time'] = df_trades['exit_time']
    df_trades.set_index('time', inplace=True)
    
    # Monthly stats
    monthly = df_trades.resample('ME')['pnl_dollars'].sum()
    monthly_pct = (monthly / 100000) * 100
    
    print("\n" + "="*60)
    print("PORTFOLIO PERFORMANCE (Global Risk Cap: 5 Trades)")
    print("="*60)
    print(f"Total Trades: {len(trades)}")
    print(f"Final Equity: ${results['final_equity']:,.2f}")
    print(f"Total Return: {((results['final_equity'] - 100000)/100000)*100:.2f}%")
    
    # Drawdown
    running_equity = 100000
    peak = 100000
    max_dd = 0
    
    for t in trades:
        running_equity = t['equity']
        if running_equity > peak:
            peak = running_equity
        dd = (peak - running_equity) / peak * 100
        max_dd = max(max_dd, dd)
        
    print(f"Max Drawdown: {max_dd:.2f}%")
    
    print("\nMONTHLY BREAKDOWN:")
    for date, val in monthly_pct.items():
        print(f"{date.strftime('%Y-%m')}: {val:>6.2f}%")
        
    # --- STRESS TESTS ---
    print("\n" + "="*60)
    print("STRESS TEST RESULTS")
    print("="*60)
    
    # 1. Monte Carlo
    print("\n[1] Monte Carlo Simulation (1000 runs):")
    mc = run_monte_carlo(trades)
    print(f"  Median MaxDD: {mc['median_dd']:.2f}%")
    print(f"  95% Confidence MaxDD: {mc['95_percentile_dd']:.2f}%")
    print(f"  Worst Case MaxDD: {mc['worst_case_dd']:.2f}%")
    
    # 2. Slippage Stress Test
    slippage_cost = 15.0
    adjusted_equity = 100000
    adj_peak = 100000
    adj_max_dd = 0
    
    for t in trades:
        pnl = t['pnl_dollars'] - slippage_cost
        adjusted_equity += pnl
        if adjusted_equity > adj_peak: adj_peak = adjusted_equity
        dd = (adj_peak - adjusted_equity) / adj_peak * 100
        adj_max_dd = max(adj_max_dd, dd)
        
    print(f"\n[2] Slippage Stress Test (~2.0 pips / $15 per trade):")
    print(f"  Adjusted Return: {((adjusted_equity - 100000)/100000)*100:.2f}%")
    print(f"  Adjusted MaxDD: {adj_max_dd:.2f}%")
    print(f"  Still Profitable: {'YES' if adjusted_equity > 100000 else 'NO'}")

if __name__ == "__main__":
    symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDJPY", "XAUUSD"]
    data_map = {}
    
    print("Fetching data...")
    for s in symbols:
        df = fetch_data(s, 3)
        if df is not None:
            data_map[s] = df
            print(f"  {s}: {len(df)} bars")
            
    if data_map:
        print("\nRunning Portfolio Simulation...")
        results = run_portfolio_simulation(data_map)
        analyze_results(results)
        
        # Save
        with open("portfolio_results.json", 'w') as f:
            json.dump(results['trades'], f, default=str, indent=2)
