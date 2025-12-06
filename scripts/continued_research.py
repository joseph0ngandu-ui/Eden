#!/usr/bin/env python3
"""
CONTINUED RESEARCH - MORE STRATEGIES
Testing: RSI Divergence, MACD Momentum, Trend Strength, Multi-Indicator
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

def fetch(symbol, tf, months=24):
    if not mt5.symbol_select(symbol, True):
        return None
    rates = mt5.copy_rates_range(symbol, tf, datetime.now() - timedelta(days=months*30), datetime.now())
    if rates is None: return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    return 100 - (100 / (1 + gain/loss))

def calc_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast).mean()
    ema_slow = series.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal).mean()
    return macd, signal_line

def test_rsi_divergence(df, symbol):
    """
    RSI DIVERGENCE
    Bullish: Price makes lower low, RSI makes higher low
    Bearish: Price makes higher high, RSI makes lower high
    """
    print(f"\n  RSI Divergence ({symbol})...")
    
    rsi = calc_rsi(df['close'])
    trades = []
    
    for i in range(50, len(df) - 5):
        # Check for swing low/high in last 20 bars
        window = 20
        if i < window:
            continue
        
        price_slice = df['close'].iloc[i-window:i]
        rsi_slice = rsi.iloc[i-window:i]
        
        curr_price = df['close'].iloc[i]
        curr_rsi = rsi.iloc[i]
        prev_price = df['close'].iloc[i-1]
        prev_rsi = rsi.iloc[i-1]
        
        # Find recent lows
        price_low = price_slice.min()
        rsi_at_price_low = rsi_slice.iloc[price_slice.argmin()]
        
        # Bullish divergence: new price low but RSI higher
        if curr_price <= price_low * 1.002 and curr_rsi > rsi_at_price_low + 5:
            if prev_rsi < 35:  # Oversold confirmation
                entry = df.iloc[i+1]['open']
                atr = (df['high'] - df['low']).iloc[i-14:i].mean()
                sl = curr_price - atr * 1.0
                tp = entry + (entry - sl) * 2.0
                
                future = df.iloc[i+1:i+10]
                if len(future) < 5:
                    continue
                
                sl_dist = abs(entry - sl)
                if sl_dist == 0:
                    continue
                
                hit_tp = future['high'].max() >= tp
                hit_sl = future['low'].min() <= sl
                
                if hit_tp and not hit_sl:
                    r_mult = 2.0
                elif hit_sl:
                    r_mult = -1.0
                else:
                    r_mult = (future.iloc[-1]['close'] - entry) / sl_dist
                
                trades.append({'r': r_mult})
    
    return analyze(trades, "RSI Divergence", symbol)

def test_macd_momentum(df, symbol):
    """
    MACD MOMENTUM
    Entry when MACD crosses signal with momentum confirmation
    """
    print(f"\n  MACD Momentum ({symbol})...")
    
    macd, signal = calc_macd(df['close'])
    rsi = calc_rsi(df['close'])
    trades = []
    
    for i in range(50, len(df) - 5):
        curr_macd = macd.iloc[i]
        prev_macd = macd.iloc[i-1]
        curr_sig = signal.iloc[i]
        prev_sig = signal.iloc[i-1]
        curr_rsi = rsi.iloc[i]
        
        atr = (df['high'] - df['low']).iloc[i-14:i].mean()
        if atr == 0:
            continue
        
        entry = df.iloc[i+1]['open']
        
        # Bullish crossover with momentum
        if prev_macd <= prev_sig and curr_macd > curr_sig:
            if curr_rsi > 50 and curr_rsi < 70:  # Trending but not overbought
                sl = df['low'].iloc[i-5:i].min() - atr * 0.3
                tp = entry + (entry - sl) * 2.0
                direction = 'LONG'
            else:
                continue
        # Bearish crossover
        elif prev_macd >= prev_sig and curr_macd < curr_sig:
            if curr_rsi < 50 and curr_rsi > 30:
                sl = df['high'].iloc[i-5:i].max() + atr * 0.3
                tp = entry - (sl - entry) * 2.0
                direction = 'SHORT'
            else:
                continue
        else:
            continue
        
        sl_dist = abs(entry - sl)
        if sl_dist == 0:
            continue
        
        future = df.iloc[i+1:i+10]
        if len(future) < 5:
            continue
        
        if direction == 'LONG':
            hit_tp = future['high'].max() >= tp
            hit_sl = future['low'].min() <= sl
            if hit_tp and not hit_sl:
                r_mult = 2.0
            elif hit_sl:
                r_mult = -1.0
            else:
                r_mult = (future.iloc[-1]['close'] - entry) / sl_dist
        else:
            hit_tp = future['low'].min() <= tp
            hit_sl = future['high'].max() >= sl
            if hit_tp and not hit_sl:
                r_mult = 2.0
            elif hit_sl:
                r_mult = -1.0
            else:
                r_mult = (entry - future.iloc[-1]['close']) / sl_dist
        
        trades.append({'r': r_mult})
    
    return analyze(trades, "MACD Momentum", symbol)

def test_trend_strength(df, symbol):
    """
    TREND STRENGTH
    Trade strong trends only (ADX-like approach)
    """
    print(f"\n  Trend Strength ({symbol})...")
    
    # Calculate trend strength via directional movement
    plus_dm = df['high'].diff()
    minus_dm = -df['low'].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    tr = df['high'] - df['low']
    atr = tr.rolling(14).mean()
    
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr)
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(14).mean()
    
    trades = []
    
    for i in range(50, len(df) - 5):
        curr_adx = adx.iloc[i]
        curr_plus = plus_di.iloc[i]
        curr_minus = minus_di.iloc[i]
        
        if curr_adx < 25:  # Only trade strong trends
            continue
        
        atr_val = atr.iloc[i]
        if atr_val == 0:
            continue
        
        entry = df.iloc[i+1]['open']
        
        # Strong uptrend
        if curr_plus > curr_minus and curr_adx > 30:
            sl = df['low'].iloc[i-5:i].min() - atr_val * 0.5
            tp = entry + (entry - sl) * 2.5
            direction = 'LONG'
        # Strong downtrend
        elif curr_minus > curr_plus and curr_adx > 30:
            sl = df['high'].iloc[i-5:i].max() + atr_val * 0.5
            tp = entry - (sl - entry) * 2.5
            direction = 'SHORT'
        else:
            continue
        
        sl_dist = abs(entry - sl)
        if sl_dist == 0:
            continue
        
        future = df.iloc[i+1:i+15]
        if len(future) < 5:
            continue
        
        if direction == 'LONG':
            hit_tp = future['high'].max() >= tp
            hit_sl = future['low'].min() <= sl
            if hit_tp and not hit_sl:
                r_mult = 2.5
            elif hit_sl:
                r_mult = -1.0
            else:
                r_mult = (future.iloc[-1]['close'] - entry) / sl_dist
        else:
            hit_tp = future['low'].min() <= tp
            hit_sl = future['high'].max() >= sl
            if hit_tp and not hit_sl:
                r_mult = 2.5
            elif hit_sl:
                r_mult = -1.0
            else:
                r_mult = (entry - future.iloc[-1]['close']) / sl_dist
        
        trades.append({'r': r_mult})
    
    return analyze(trades, "Trend Strength", symbol)

def test_multi_confluence(df, symbol):
    """
    MULTI-INDICATOR CONFLUENCE
    Only trade when RSI, MACD, and EMA all align
    """
    print(f"\n  Multi Confluence ({symbol})...")
    
    rsi = calc_rsi(df['close'])
    macd, signal = calc_macd(df['close'])
    ema20 = df['close'].ewm(span=20).mean()
    ema50 = df['close'].ewm(span=50).mean()
    
    trades = []
    
    for i in range(60, len(df) - 5):
        curr_rsi = rsi.iloc[i]
        curr_macd = macd.iloc[i]
        curr_sig = signal.iloc[i]
        curr_ema20 = ema20.iloc[i]
        curr_ema50 = ema50.iloc[i]
        curr_close = df['close'].iloc[i]
        
        atr = (df['high'] - df['low']).iloc[i-14:i].mean()
        if atr == 0:
            continue
        
        entry = df.iloc[i+1]['open']
        
        # BULLISH CONFLUENCE: RSI > 50, MACD > Signal, EMA20 > EMA50, Price > EMA20
        bullish_rsi = 50 < curr_rsi < 70
        bullish_macd = curr_macd > curr_sig
        bullish_ema = curr_ema20 > curr_ema50
        bullish_price = curr_close > curr_ema20
        
        # BEARISH CONFLUENCE
        bearish_rsi = 30 < curr_rsi < 50
        bearish_macd = curr_macd < curr_sig
        bearish_ema = curr_ema20 < curr_ema50
        bearish_price = curr_close < curr_ema20
        
        if bullish_rsi and bullish_macd and bullish_ema and bullish_price:
            sl = df['low'].iloc[i-5:i].min() - atr * 0.3
            tp = entry + (entry - sl) * 2.0
            direction = 'LONG'
        elif bearish_rsi and bearish_macd and bearish_ema and bearish_price:
            sl = df['high'].iloc[i-5:i].max() + atr * 0.3
            tp = entry - (sl - entry) * 2.0
            direction = 'SHORT'
        else:
            continue
        
        sl_dist = abs(entry - sl)
        if sl_dist == 0:
            continue
        
        future = df.iloc[i+1:i+10]
        if len(future) < 5:
            continue
        
        if direction == 'LONG':
            hit_tp = future['high'].max() >= tp
            hit_sl = future['low'].min() <= sl
            if hit_tp and not hit_sl:
                r_mult = 2.0
            elif hit_sl:
                r_mult = -1.0
            else:
                r_mult = (future.iloc[-1]['close'] - entry) / sl_dist
        else:
            hit_tp = future['low'].min() <= tp
            hit_sl = future['high'].max() >= sl
            if hit_tp and not hit_sl:
                r_mult = 2.0
            elif hit_sl:
                r_mult = -1.0
            else:
                r_mult = (entry - future.iloc[-1]['close']) / sl_dist
        
        trades.append({'r': r_mult})
    
    return analyze(trades, "Multi Confluence", symbol)

def analyze(trades, name, symbol):
    if len(trades) < 30:
        print(f"    ⚠ Only {len(trades)} trades")
        return None
    
    split = int(len(trades) * 0.6)
    test = trades[split:]
    
    r_mults = [t['r'] for t in test]
    wins = sum(1 for r in r_mults if r > 0)
    total = len(r_mults)
    win_rate = wins / total * 100 if total > 0 else 0
    exp = np.mean(r_mults) if r_mults else 0
    
    gross_win = sum(r for r in r_mults if r > 0)
    gross_loss = abs(sum(r for r in r_mults if r < 0))
    pf = gross_win / gross_loss if gross_loss > 0 else 0
    
    print(f"    Test: {total} trades, WR {win_rate:.1f}%, Exp {exp:.3f}R, PF {pf:.2f}")
    
    if exp > 0.15 and pf > 1.2:
        print(f"    ✅ STRONG EDGE")
        return {'name': name, 'symbol': symbol, 'trades': total, 'win_rate': win_rate, 'exp': exp, 'pf': pf, 'verdict': 'STRONG'}
    elif exp > 0:
        print(f"    ⚠️ WEAK")
        return {'name': name, 'symbol': symbol, 'trades': total, 'win_rate': win_rate, 'exp': exp, 'pf': pf, 'verdict': 'WEAK'}
    else:
        print(f"    ❌ NONE")
        return None

def main():
    print("\n" + "="*60)
    print("  CONTINUED RESEARCH - MORE STRATEGIES")
    print("  RSI Divergence | MACD | Trend Strength | Confluence")
    print("="*60)
    
    if not init_mt5():
        return
    
    # Test on multiple symbols including indices
    symbols = ['EURUSDm', 'GBPUSDm', 'USDJPYm', 'XAUUSDm', 'US30m', 'USTECm']
    results = {}
    
    for sym in symbols:
        print(f"\n{'='*50}")
        print(f"  {sym}")
        print(f"{'='*50}")
        
        df = fetch(sym, mt5.TIMEFRAME_H1, months=24)
        
        if df is None or len(df) < 1000:
            print(f"  ⚠ Insufficient data")
            continue
        
        print(f"  Loaded {len(df)} bars")
        
        for test_func in [test_rsi_divergence, test_macd_momentum, test_trend_strength, test_multi_confluence]:
            result = test_func(df, sym)
            if result:
                key = f"{sym}_{result['name']}"
                results[key] = result
    
    mt5.shutdown()
    
    # Summary
    print(f"\n{'='*60}")
    print("  RESEARCH RESULTS")
    print(f"{'='*60}")
    
    strong = []
    weak = []
    
    for k, v in sorted(results.items(), key=lambda x: x[1]['exp'], reverse=True):
        if v['verdict'] == 'STRONG':
            print(f"✅ {k}: Exp {v['exp']:.3f}R, PF {v['pf']:.2f}")
            strong.append(k)
        else:
            print(f"⚠️ {k}: Exp {v['exp']:.3f}R, PF {v['pf']:.2f}")
            weak.append(k)
    
    if not results:
        print("  ❌ No viable strategies found")
    
    output = PROJECT_ROOT / 'reports' / 'continued_research.json'
    with open(output, 'w') as f:
        json.dump({'date': datetime.now().isoformat(), 'results': results, 
                   'strong': strong, 'weak': weak}, f, indent=2, default=str)
    print(f"\n  ✓ Saved: {output}")

if __name__ == '__main__':
    main()
