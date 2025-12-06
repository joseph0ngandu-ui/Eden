#!/usr/bin/env python3
"""
FINAL CREATIVE APPROACH
Range-based mean reversion + volatility contraction patterns
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
    print(f"❌ {e}")
    sys.exit(1)

def init_mt5():
    if not mt5.initialize(path=r"C:\Program Files\MetaTrader 5 Terminal\terminal64.exe"):
        return False
    print("✓ MT5 connected")
    return True

def fetch(symbol, tf, months=12):
    if not mt5.symbol_select(symbol, True):
        return None
    rates = mt5.copy_rates_range(symbol, tf, datetime.now() - timedelta(days=months*30), datetime.now())
    if rates is None: return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df

def test_hourly_range_reversion(df, symbol):
    """
    HOURLY RANGE REVERSION
    Entry: When price hits hourly high/low, fade the move
    Filter: Only when RSI confirms extremes
    """
    print(f"\n  Hourly Range Reversion ({symbol})...")
    
    trades = []
    
    # Resample to hourly for range calc
    hourly = df.resample('1h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'})
    hourly = hourly.dropna()
    
    for i in range(20, len(hourly) - 2):
        h = hourly.iloc[i]
        prev_h = hourly.iloc[i-1]
        
        high = h['high']
        low = h['low']
        rng = high - low
        
        if rng == 0:
            continue
        
        avg_rng = (hourly['high'] - hourly['low']).iloc[i-20:i].mean()
        
        # Only trade when current hour range is significant
        if rng < avg_rng * 0.8:
            continue
        
        # Calculate hourly RSI from closes
        delta = hourly['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = (100 - (100 / (1 + rs))).iloc[i]
        
        # Entry conditions (end of current hour)
        # LONG: Price at hourly low + RSI < 30
        # SHORT: Price at hourly high + RSI > 70
        
        close = h['close']
        entry = hourly.iloc[i+1]['open']  # Next hour open
        
        if close <= low + rng * 0.1 and rsi < 35:
            direction = 'LONG'
            sl = low - avg_rng * 0.3
            tp = low + rng * 0.8
        elif close >= high - rng * 0.1 and rsi > 65:
            direction = 'SHORT'
            sl = high + avg_rng * 0.3
            tp = high - rng * 0.8
        else:
            continue
        
        # Check result in next hour
        next_h = hourly.iloc[i+1]
        sl_dist = abs(entry - sl)
        if sl_dist == 0:
            continue
        
        if direction == 'LONG':
            hit_tp = next_h['high'] >= tp
            hit_sl = next_h['low'] <= sl
            if hit_tp and not hit_sl:
                r_mult = (tp - entry) / sl_dist
            elif hit_sl:
                r_mult = -1.0
            else:
                r_mult = (next_h['close'] - entry) / sl_dist
        else:
            hit_tp = next_h['low'] <= tp
            hit_sl = next_h['high'] >= sl
            if hit_tp and not hit_sl:
                r_mult = (entry - tp) / sl_dist
            elif hit_sl:
                r_mult = -1.0
            else:
                r_mult = (entry - next_h['close']) / sl_dist
        
        trades.append({'r': r_mult})
    
    return analyze(trades, "Hourly Range Rev")

def test_volatility_squeeze(df, symbol):
    """
    VOLATILITY SQUEEZE BREAKOUT
    Low volatility periods followed by breakout
    """
    print(f"\n  Volatility Squeeze ({symbol})...")
    
    trades = []
    
    for i in range(50, len(df) - 5):
        bar = df.iloc[i]
        
        # Current range
        curr_range = bar['high'] - bar['low']
        
        # Average range over last 20 bars
        avg_range = (df['high'] - df['low']).iloc[i-20:i].mean()
        
        # Historical volatility (std of closes)
        vol = df['close'].iloc[i-20:i].pct_change().std()
        avg_vol = df['close'].iloc[i-50:i].pct_change().std()
        
        # Squeeze condition: current vol < 60% of average
        if vol > avg_vol * 0.6:
            continue
        
        # Wait for breakout: bar closes outside 20-bar high/low
        high_20 = df['high'].iloc[i-20:i].max()
        low_20 = df['low'].iloc[i-20:i].min()
        
        entry = df.iloc[i+1]['open']  # Next bar
        atr = avg_range
        
        if bar['close'] > high_20:
            direction = 'LONG'
            sl = low_20
            tp = entry + (entry - sl) * 1.5
        elif bar['close'] < low_20:
            direction = 'SHORT'
            sl = high_20
            tp = entry - (sl - entry) * 1.5
        else:
            continue
        
        sl_dist = abs(entry - sl)
        if sl_dist == 0:
            continue
        
        # Check result over next 5 bars
        future = df.iloc[i+1:i+6]
        if len(future) < 3:
            continue
        
        if direction == 'LONG':
            hit_tp = future['high'].max() >= tp
            hit_sl = future['low'].min() <= sl
            if hit_tp and not hit_sl:
                r_mult = 1.5
            elif hit_sl:
                r_mult = -1.0
            else:
                r_mult = (future.iloc[-1]['close'] - entry) / sl_dist
        else:
            hit_tp = future['low'].min() <= tp
            hit_sl = future['high'].max() >= sl
            if hit_tp and not hit_sl:
                r_mult = 1.5
            elif hit_sl:
                r_mult = -1.0
            else:
                r_mult = (entry - future.iloc[-1]['close']) / sl_dist
        
        trades.append({'r': r_mult})
    
    return analyze(trades, "Vol Squeeze")

def test_gap_fill(df, symbol):
    """
    GAP FILL STRATEGY (H1)
    Fade overnight gaps that tend to fill during session
    """
    print(f"\n  Gap Fill ({symbol})...")
    
    trades = []
    
    for i in range(50, len(df) - 5):
        bar = df.iloc[i]
        prev_bar = df.iloc[i-1]
        
        hour = bar.name.hour
        
        # Only look for gaps at session opens (7AM, 12PM)
        if hour not in [7, 12]:
            continue
        
        # Calculate gap
        gap = bar['open'] - prev_bar['close']
        atr = (df['high'] - df['low']).iloc[i-20:i].mean()
        
        if atr == 0:
            continue
        
        # Gap must be significant (>0.5 ATR)
        if abs(gap) < atr * 0.5:
            continue
        
        entry = df.iloc[i+1]['open']  # Entry on next bar
        
        # Fade the gap
        if gap > 0:  # Gap up - expect fill down
            direction = 'SHORT'
            sl = bar['high'] + atr * 0.3
            tp = prev_bar['close']  # Gap fill target
        else:  # Gap down - expect fill up
            direction = 'LONG'
            sl = bar['low'] - atr * 0.3
            tp = prev_bar['close']
        
        sl_dist = abs(entry - sl)
        if sl_dist == 0:
            continue
        
        # Check over next 4 bars (1 hour for M15)
        future = df.iloc[i+1:i+5]
        if len(future) < 2:
            continue
        
        if direction == 'LONG':
            hit_tp = future['high'].max() >= tp
            hit_sl = future['low'].min() <= sl
            if hit_tp and not hit_sl:
                r_mult = (tp - entry) / sl_dist
            elif hit_sl:
                r_mult = -1.0
            else:
                r_mult = (future.iloc[-1]['close'] - entry) / sl_dist
        else:
            hit_tp = future['low'].min() <= tp
            hit_sl = future['high'].max() >= sl
            if hit_tp and not hit_sl:
                r_mult = (entry - tp) / sl_dist
            elif hit_sl:
                r_mult = -1.0
            else:
                r_mult = (entry - future.iloc[-1]['close']) / sl_dist
        
        trades.append({'r': r_mult})
    
    return analyze(trades, "Gap Fill")

def analyze(trades, name):
    if len(trades) < 20:
        print(f"    ⚠ Only {len(trades)} trades")
        return None
    
    split = int(len(trades) * 0.6)
    test = trades[split:]
    
    r_mults = [t['r'] for t in test]
    wins = sum(1 for r in r_mults if r > 0)
    total = len(r_mults)
    win_rate = wins / total * 100
    exp = np.mean(r_mults)
    
    gross_win = sum(r for r in r_mults if r > 0)
    gross_loss = abs(sum(r for r in r_mults if r < 0))
    pf = gross_win / gross_loss if gross_loss > 0 else 0
    
    print(f"    Test: {total} trades, WR {win_rate:.1f}%, Exp {exp:.3f}R, PF {pf:.2f}")
    
    if exp > 0.15 and pf > 1.2:
        print(f"    ✅ EDGE FOUND!")
        return {'name': name, 'trades': total, 'win_rate': win_rate, 'expectancy': exp, 'pf': pf}
    elif exp > 0:
        print(f"    ⚠️ Weak edge")
        return {'name': name, 'trades': total, 'win_rate': win_rate, 'expectancy': exp, 'pf': pf}
    else:
        print(f"    ❌ No edge")
        return None

def main():
    print("\n" + "="*50)
    print("  FINAL CREATIVE APPROACH")
    print("  Range + Volatility + Gap patterns")
    print("="*50)
    
    if not init_mt5():
        return
    
    symbols = ['EURUSDm', 'GBPUSDm', 'USDJPYm', 'XAUUSDm']
    results = {}
    
    for sym in symbols:
        print(f"\n{'='*40}")
        print(f"  {sym}")
        print(f"{'='*40}")
        
        # H1 data for range strategies
        df_h1 = fetch(sym, mt5.TIMEFRAME_H1)
        # M15 for gap fill
        df_m15 = fetch(sym, mt5.TIMEFRAME_M15)
        
        if df_h1 is None or df_m15 is None:
            continue
        
        print(f"  H1: {len(df_h1)} bars | M15: {len(df_m15)} bars")
        
        # Test strategies
        for test_func, df in [
            (test_hourly_range_reversion, df_h1),
            (test_volatility_squeeze, df_h1),
            (test_gap_fill, df_m15)
        ]:
            result = test_func(df, sym)
            if result:
                key = f"{sym}_{result['name']}"
                results[key] = result
    
    mt5.shutdown()
    
    # Summary
    print(f"\n{'='*50}")
    print("  FINAL RESULTS")
    print(f"{'='*50}")
    
    if results:
        for k, v in sorted(results.items(), key=lambda x: x[1]['expectancy'], reverse=True):
            marker = "✅" if v['expectancy'] > 0.15 else "⚠️"
            print(f"  {marker} {k}: Exp {v['expectancy']:.3f}R, PF {v['pf']:.2f}, WR {v['win_rate']:.1f}%")
    else:
        print("  ❌ No viable strategies found")
    
    output = PROJECT_ROOT / 'reports' / 'creative_final_results.json'
    with open(output, 'w') as f:
        json.dump({'date': datetime.now().isoformat(), 'results': results}, f, indent=2)
    print(f"\n  ✓ Saved: {output}")

if __name__ == '__main__':
    main()
