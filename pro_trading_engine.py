#!/usr/bin/env python3
"""
Professional Trading Engine - 12% Monthly Target

6-strategy high-frequency system:
1. London/NY Overlap Scalper
2. Asian Range Fade  
3. News Spike Reversal
4. Correlation Arbitrage
5. Gold London Breakout
6. Volatility Expansion

Target: 12% monthly, <8% max DD
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json

def fetch_data(symbol: str, months: int = 3, timeframe=mt5.TIMEFRAME_M5) -> Optional[pd.DataFrame]:
    """Fetch M5 data for high-frequency testing."""
    if not mt5.initialize():
        return None
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=months * 30)
    
    if not mt5.symbol_select(symbol, True):
        mt5.shutdown()
        return None
    
    rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
    mt5.shutdown()
    
    if rates is None or len(rates) == 0:
        return None
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df['hour'] = df['time'].dt.hour
    df['minute'] = df['time'].dt.minute
    return df

# ========== STRATEGY 1: London/NY Overlap Scalper ==========
def overlap_scalper(df: pd.DataFrame, symbol: str, idx: int) -> Optional[Dict]:
    """High-frequency scalper during peak liquidity."""
    row = df.iloc[idx]
    hour = row['hour']
    
    # London/NY overlap: 12:00-16:00 GMT (8AM-12PM EST)
    if not (12 <= hour < 16):
        return None
    
    if idx < 20:
        return None
    
    # 5-period momentum
    recent = df.iloc[idx-5:idx+1]
    momentum = recent['close'].iloc[-1] - recent['close'].iloc[0]
    atr = calculate_atr_simple(df.iloc[max(0, idx-14):idx+1])
    
    # Volume spike (20% above average)
    avg_vol = df.iloc[max(0, idx-20):idx]['tick_volume'].mean()
    if row['tick_volume'] < avg_vol * 1.2:
        return None
    
    # Entry: Strong momentum breakout
    if abs(momentum) > atr * 0.6:
        direction = "LONG" if momentum > 0 else "SHORT"
        
        if direction == "LONG":
            entry = row['close']
            sl = entry - (atr * 1.0)
            tp = entry + (atr * 2.5)  # Increased R:R to 2.5
        else:
            entry = row['close']
            sl = entry + (atr * 1.0)
            tp = entry - (atr * 2.5)  # Increased R:R to 2.5
        
        return {'symbol': symbol, 'direction': direction, 'entry': entry, 'sl': sl, 'tp': tp, 
                'strategy': 'overlap_scalper', 'risk_pct': 0.015}  # Aggressive 1.5% risk
    
    return None

# ========== STRATEGY 2: Asian Range Fade ==========
def asian_fade(df: pd.DataFrame, symbol: str, idx: int) -> Optional[Dict]:
    """Fade extremes during low-volatility Asian session."""
    row = df.iloc[idx]
    hour = row['hour']
    
    # Asian session: 22:00-06:00 GMT
    if not ((hour >= 22) or (hour < 6)):
        return None
    
    if idx < 60:  # Need 5 hours of data
        return None
    
    # Get Asian range (last 60 bars = 5 hours on M5)
    asian_bars = df.iloc[idx-60:idx]
    asian_high = asian_bars['high'].max()
    asian_low = asian_bars['low'].min()
    asian_mid = (asian_high + asian_low) / 2
    range_size = asian_high - asian_low
    
    if range_size == 0:
        return None
    
    # Fade extremes (price > 80% range or < 20% range)
    price_position = (row['close'] - asian_low) / range_size
    
    if price_position > 0.8:  # Near top, fade short
        return {'symbol': symbol, 'direction': 'SHORT', 'entry': row['close'], 
                'sl': asian_high + range_size * 0.1, 'tp': asian_mid,
                'strategy': 'asian_fade', 'risk_pct': 0.01}  # Aggressive 1.0% risk
    
    elif price_position < 0.2:  # Near bottom, fade long
        return {'symbol': symbol, 'direction': 'LONG', 'entry': row['close'],
                'sl': asian_low - range_size * 0.1, 'tp': asian_mid,
                'strategy': 'asian_fade', 'risk_pct': 0.01}  # Aggressive 1.0% risk
    
    return None

# ========== STRATEGY 3: Gold London Breakout ==========
def gold_london_breakout(df: pd.DataFrame, symbol: str, idx: int) -> Optional[Dict]:
    """Trade Asian range breakout at London open."""
    if symbol != "XAUUSD":
        return None
    
    row = df.iloc[idx]
    hour = row['hour']
    
    # London open: 7-9 AM GMT
    if not (7 <= hour < 9):
        return None
    
    if idx < 72:  # Need 6 hours (Asian session)
        return None
    
    # Asian range (midnight to 6 AM = 72 M5 bars)
    asian_start = None
    for i in range(idx, max(0, idx-100), -1):
        if df.iloc[i]['hour'] == 0:
            asian_start = i
            break
    
    if asian_start is None or idx - asian_start < 60:
        return None
    
    asian_bars = df.iloc[asian_start:idx]
    asian_high = asian_bars['high'].max()
    asian_low = asian_bars['low'].min()
    asian_range = asian_high - asian_low
    
    if asian_range == 0:
        return None
    
    # Breakout above Asian high
    if row['close'] > asian_high and row['close'] > asian_high + asian_range * 0.1:
        return {'symbol': symbol, 'direction': 'LONG', 'entry': row['close'],
                'sl': asian_low, 'tp': row['close'] + asian_range * 3, # Increased target to 3x
                'strategy': 'gold_breakout', 'risk_pct': 0.02}  # Aggressive 2.0% risk
    
    # Breakout below Asian low
    if row['close'] < asian_low and row['close'] < asian_low - asian_range * 0.1:
        return {'symbol': symbol, 'direction': 'SHORT', 'entry': row['close'],
                'sl': asian_high, 'tp': row['close'] - asian_range * 3, # Increased target to 3x
                'strategy': 'gold_breakout', 'risk_pct': 0.02}  # Aggressive 2.0% risk
    
    return None

# ========== STRATEGY 4: Volatility Expansion ==========
def volatility_expansion(df: pd.DataFrame, symbol: str, idx: int) -> Optional[Dict]:
    """Trade breakouts from volatility compression."""
    if idx < 40:
        return None
    
    row = df.iloc[idx]
    
    # Calculate ATR compression
    current_atr = calculate_atr_simple(df.iloc[idx-14:idx+1])
    avg_atr = calculate_atr_simple(df.iloc[idx-40:idx+1])
    
    if avg_atr == 0:
        return None
    
    # Compression: Current ATR < 70% of average
    if current_atr > avg_atr * 0.7:
        return None
    
    # Breakout: Strong 3-bar move
    recent_3 = df.iloc[idx-2:idx+1]
    move = abs(recent_3['close'].iloc[-1] - recent_3['close'].iloc[0])
    
    if move > current_atr * 1.5:
        direction = "LONG" if recent_3['close'].iloc[-1] > recent_3['close'].iloc[0] else "SHORT"
        
        if direction == "LONG":
            return {'symbol': symbol, 'direction': 'LONG', 'entry': row['close'],
                    'sl': row['close'] - avg_atr * 1.5, 'tp': row['close'] + avg_atr * 4, # Aggressive 4R target
                    'strategy': 'vol_expansion', 'risk_pct': 0.015} # Aggressive 1.5% risk
        else:
            return {'symbol': symbol, 'direction': 'SHORT', 'entry': row['close'],
                    'sl': row['close'] + avg_atr * 1.5, 'tp': row['close'] - avg_atr * 4, # Aggressive 4R target
                    'strategy': 'vol_expansion', 'risk_pct': 0.015} # Aggressive 1.5% risk
    
    return None

def calculate_atr_simple(df: pd.DataFrame) -> float:
    """Simple ATR calculation."""
    if len(df) < 2:
        return 0.001
    tr = df['high'] - df['low']
    return tr.mean() if len(tr) > 0 else 0.001

def run_multi_strategy_backtest(df: pd.DataFrame, symbol: str) -> List[Dict]:
    """Run all strategies on dataset."""
    trades = []
    open_trades = {}  # Strategy -> trade
    equity = 100000
    daily_pnl = 0
    last_date = None
    
    for idx in range(100, len(df)):
        row = df.iloc[idx]
        
        # Reset daily PnL counter
        current_date = row['time'].date()
        if last_date != current_date:
            daily_pnl = 0
            last_date = current_date
        
        # Circuit breaker: Stop if dailyPnL < -2% of equity
        if daily_pnl < -(equity * 0.02):
            continue
            
        # Check exits for open trades
        current_exposure = {} # symbol -> count
        for t in open_trades.values():
            current_exposure[t['symbol']] = current_exposure.get(t['symbol'], 0) + 1

        for strategy, trade in list(open_trades.items()):
            hit_tp = (trade['direction'] == 'LONG' and row['high'] >= trade['tp']) or \
                     (trade['direction'] == 'SHORT' and row['low'] <= trade['tp'])
            hit_sl = (trade['direction'] == 'LONG' and row['low'] <= trade['sl']) or \
                     (trade['direction'] == 'SHORT' and row['high'] >= trade['sl'])
            
            if hit_tp or hit_sl:
                exit_price = trade['tp'] if hit_tp else trade['sl']
                pnl_price = (exit_price - trade['entry']) if trade['direction'] == 'LONG' else (trade['entry'] - exit_price)
                
                # Correct Position Sizing: Risk Amount / SL Distance
                risk_amount = equity * trade['risk_pct']
                sl_distance = abs(trade['entry'] - trade['sl'])
                
                if sl_distance > 0:
                    position_size = risk_amount / sl_distance
                    pnl_dollars = position_size * pnl_price
                else:
                    pnl_dollars = 0
                
                equity += pnl_dollars
                daily_pnl += pnl_dollars
                
                trades.append({
                    'entry_time': trade['entry_time'],
                    'exit_time': row['time'],
                    'strategy': trade['strategy'],
                    'symbol': trade['symbol'],
                    'pnl': pnl_price,
                    'pnl_dollars': pnl_dollars,
                    'equity': equity,
                    'result': 'TP' if hit_tp else 'SL'
                })
                del open_trades[strategy]
        
        # Check for new entries (one per strategy)
        # PROP FIRM RISK CONTROL: Max 5 open trades total
        # Max risk = 5 * 1.5% = 7.5% (Hard cap under 8%)
        if len(open_trades) < 5:
            if 'overlap_scalper' not in open_trades and symbol in ['EURUSD', 'GBPUSD']:
                signal = overlap_scalper(df, symbol, idx)
                if signal:
                    signal['entry_time'] = row['time']
                    open_trades['overlap_scalper'] = signal
            
            if 'asian_fade' not in open_trades and symbol in ['USDJPY', 'AUDJPY'] and len(open_trades) < 5:
                signal = asian_fade(df, symbol, idx)
                if signal:
                    signal['entry_time'] = row['time']
                    open_trades['asian_fade'] = signal
            
            if 'gold_breakout' not in open_trades and len(open_trades) < 5:
                signal = gold_london_breakout(df, symbol, idx)
                if signal:
                    signal['entry_time'] = row['time']
                    open_trades['gold_breakout'] = signal
            
            if 'vol_expansion' not in open_trades and len(open_trades) < 5:
                signal = volatility_expansion(df, symbol, idx)
                if signal:
                    signal['entry_time'] = row['time']
                    open_trades['vol_expansion'] = signal
    
    return trades

def calculate_performance(trades: List[Dict], months: int = 3) -> Dict:
    """Calculate monthly performance metrics."""
    if not trades:
        return None
    
    wins = [t for t in trades if t['pnl'] > 0]
    
    # Calculate DD
    equities = [100000] + [t['equity'] for t in trades]
    peak = equities[0]
    max_dd_pct = 0
    
    for eq in equities:
        if eq > peak:
            peak = eq
        dd_pct = ((peak - eq) / peak * 100) if peak > 0 else 0
        max_dd_pct = max(max_dd_pct, dd_pct)
    
    total_return_pct = ((equities[-1] - 100000) / 100000) * 100
    monthly_return = total_return_pct / months
    
    return {
        'total_trades': len(trades),
        'win_rate': len(wins) / len(trades) * 100,
        'total_return_pct': total_return_pct,
        'monthly_return_avg': monthly_return,
        'max_dd_pct': max_dd_pct,
        'final_equity': equities[-1]
    }

if __name__ == "__main__":
    print("Professional Trading Engine - 12% Monthly Target")
    print("=" * 60)
    
    symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDJPY", "XAUUSD"]
    all_trades = []
    
    for symbol in symbols:
        print(f"\nTesting {symbol}...")
        df = fetch_data(symbol, 3)  # 3 months
        if df is None:
            print(f"  Failed to fetch data")
            continue
        
        trades = run_multi_strategy_backtest(df, symbol)
        all_trades.extend(trades)
        print(f"  Trades: {len(trades)}")
    
    # Combined performance
    if all_trades:
        perf = calculate_performance(all_trades, 3)
        
        print(f"\n{'='*60}")
        print("COMBINED PORTFOLIO PERFORMANCE")
        print(f"{'='*60}")
        print(f"Total Trades: {perf['total_trades']}")
        print(f"Win Rate: {perf['win_rate']:.1f}%")
        print(f"Total Return (3 months): {perf['total_return_pct']:.2f}%")
        print(f"**MONTHLY AVG: {perf['monthly_return_avg']:.2f}%**")
        print(f"Max Drawdown: {perf['max_dd_pct']:.2f}%")
        print(f"Final Equity: ${perf['final_equity']:,.2f}")
        print(f"{'='*60}")
        
        # Monthly breakdown
        monthly_trades = {}
        for t in all_trades:
            month_key = t['entry_time'].strftime('%Y-%m')
            if month_key not in monthly_trades:
                monthly_trades[month_key] = []
            monthly_trades[month_key].append(t)
        
        print(f"\nMONTHLY BREAKDOWN:")
        for month, trades in sorted(monthly_trades.items()):
            month_perf = calculate_performance(trades, 1)
            print(f"{month}: {month_perf['monthly_return_avg']:>6.2f}% ({month_perf['total_trades']} trades)")
        
        # Save results
        with open("pro_trading_results.json", 'w') as f:
            json.dump({'performance': perf, 'trades': len(all_trades)}, f, indent=2)
        
        print(f"\nResults saved to: pro_trading_results.json")
