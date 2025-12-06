#!/usr/bin/env python3
"""
ORIGINAL STRATEGIES - Synthesized from Research Insights

Key learnings applied:
1. JPY pairs show strongest edge (carry dynamics, less HFT)
2. Volatility contraction → breakout works
3. Range reversion at extremes with RSI confirmation
4. H1 timeframe optimal
5. Simple rules > complex multi-indicator
6. Next bar entry only (no lookahead)
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

# ============================================================================
# STRATEGY 1: QUIET BEFORE STORM
# ============================================================================
def quiet_before_storm(df, symbol):
    """
    ORIGINAL: Quiet Before Storm
    
    Insight: Markets contract before expanding. When volatility falls to 
    unusually low levels AND price breaks out of compression, big moves follow.
    
    Unique twist: Combine vol contraction with candle body strength.
    - Vol must be <50% of 20-period average
    - Price must break 10-bar high/low
    - Breakout candle body must be >1.5x average body
    """
    print(f"\n  Quiet Before Storm ({symbol})...")
    
    trades = []
    
    for i in range(25, len(df) - 8):
        bar = df.iloc[i]
        
        # Volatility measure (True Range)
        tr = df['high'] - df['low']
        curr_vol = tr.iloc[i]
        avg_vol = tr.iloc[i-20:i].mean()
        
        if avg_vol == 0:
            continue
        
        # Check for compression: last 5 bars have low vol
        recent_vol = tr.iloc[i-5:i].mean()
        compressed = recent_vol < avg_vol * 0.5
        
        if not compressed:
            continue
        
        # Check for breakout with strong candle
        high_10 = df['high'].iloc[i-10:i].max()
        low_10 = df['low'].iloc[i-10:i].min()
        
        body = abs(bar['close'] - bar['open'])
        avg_body = abs(df['close'] - df['open']).iloc[i-20:i].mean()
        strong_body = body > avg_body * 1.5
        
        entry = df.iloc[i+1]['open']
        
        # Bullish breakout
        if bar['close'] > high_10 and bar['close'] > bar['open'] and strong_body:
            sl = df['low'].iloc[i-5:i].min() - avg_vol * 0.2
            tp = entry + (entry - sl) * 2.5
            direction = 'LONG'
        # Bearish breakout
        elif bar['close'] < low_10 and bar['close'] < bar['open'] and strong_body:
            sl = df['high'].iloc[i-5:i].max() + avg_vol * 0.2
            tp = entry - (sl - entry) * 2.5
            direction = 'SHORT'
        else:
            continue
        
        sl_dist = abs(entry - sl)
        if sl_dist == 0:
            continue
        
        # Check result over next 8 bars
        future = df.iloc[i+1:i+9]
        if len(future) < 4:
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
    
    return analyze(trades, "Quiet Before Storm", symbol)

# ============================================================================
# STRATEGY 2: EXHAUSTION REVERSAL
# ============================================================================
def exhaustion_reversal(df, symbol):
    """
    ORIGINAL: Exhaustion Reversal
    
    Insight: Strong trends exhaust themselves. Look for trend exhaustion
    signals combined with extreme positioning.
    
    Unique approach:
    - 3+ consecutive candles in same direction
    - Final candle has longest wick (rejection)
    - RSI at extreme (>75 or <25)
    - Fade with tight stop above/below exhaustion candle
    """
    print(f"\n  Exhaustion Reversal ({symbol})...")
    
    # RSI calculation
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + gain/loss))
    
    trades = []
    
    for i in range(20, len(df) - 5):
        bars = [df.iloc[i-j] for j in range(4, -1, -1)]  # Last 5 bars
        curr_rsi = rsi.iloc[i]
        
        atr = (df['high'] - df['low']).iloc[i-14:i].mean()
        if atr == 0:
            continue
        
        # Check for bullish exhaustion (bearish signal)
        bullish_run = all(bars[j]['close'] > bars[j]['open'] for j in range(4))
        if bullish_run and curr_rsi > 75:
            # Check for rejection wick
            final_bar = bars[4]
            upper_wick = final_bar['high'] - max(final_bar['open'], final_bar['close'])
            body = abs(final_bar['close'] - final_bar['open'])
            
            if upper_wick > body * 1.5:  # Strong rejection
                entry = df.iloc[i+1]['open']
                sl = bars[4]['high'] + atr * 0.2
                tp = entry - (sl - entry) * 2.0
                direction = 'SHORT'
                
                sl_dist = abs(entry - sl)
                if sl_dist > 0:
                    future = df.iloc[i+1:i+6]
                    if len(future) >= 3:
                        hit_tp = future['low'].min() <= tp
                        hit_sl = future['high'].max() >= sl
                        if hit_tp and not hit_sl:
                            r_mult = 2.0
                        elif hit_sl:
                            r_mult = -1.0
                        else:
                            r_mult = (entry - future.iloc[-1]['close']) / sl_dist
                        trades.append({'r': r_mult})
        
        # Check for bearish exhaustion (bullish signal)
        bearish_run = all(bars[j]['close'] < bars[j]['open'] for j in range(4))
        if bearish_run and curr_rsi < 25:
            final_bar = bars[4]
            lower_wick = min(final_bar['open'], final_bar['close']) - final_bar['low']
            body = abs(final_bar['close'] - final_bar['open'])
            
            if lower_wick > body * 1.5:  # Strong rejection
                entry = df.iloc[i+1]['open']
                sl = bars[4]['low'] - atr * 0.2
                tp = entry + (entry - sl) * 2.0
                direction = 'LONG'
                
                sl_dist = abs(entry - sl)
                if sl_dist > 0:
                    future = df.iloc[i+1:i+6]
                    if len(future) >= 3:
                        hit_tp = future['high'].max() >= tp
                        hit_sl = future['low'].min() <= sl
                        if hit_tp and not hit_sl:
                            r_mult = 2.0
                        elif hit_sl:
                            r_mult = -1.0
                        else:
                            r_mult = (future.iloc[-1]['close'] - entry) / sl_dist
                        trades.append({'r': r_mult})
    
    return analyze(trades, "Exhaustion Reversal", symbol)

# ============================================================================
# STRATEGY 3: MOMENTUM PAUSE CONTINUATION
# ============================================================================
def momentum_pause(df, symbol):
    """
    ORIGINAL: Momentum Pause Continuation
    
    Insight: Strong moves pause before continuing. The pause is a 
    consolidation not a reversal.
    
    Unique filters:
    - Need initial strong move (>2 ATR in 3 bars)
    - Then 2-3 small range bars (pause)
    - Resume when bar breaks pause high/low in trend direction
    - Tight stop at pause extreme
    """
    print(f"\n  Momentum Pause ({symbol})...")
    
    trades = []
    
    for i in range(30, len(df) - 6):
        atr = (df['high'] - df['low']).iloc[i-14:i].mean()
        if atr == 0:
            continue
        
        # Check for initial strong up move (bars -7 to -4)
        move_start = df['close'].iloc[i-7]
        move_end = df['close'].iloc[i-4]
        move = move_end - move_start
        
        strong_up = move > atr * 2
        strong_down = move < -atr * 2
        
        if not (strong_up or strong_down):
            continue
        
        # Check for pause (bars -3 to -1): small range bars
        pause_bars = [df.iloc[i-j] for j in range(3, 0, -1)]
        pause_ranges = [b['high'] - b['low'] for b in pause_bars]
        avg_pause_range = np.mean(pause_ranges)
        
        is_pause = avg_pause_range < atr * 0.6
        if not is_pause:
            continue
        
        # Current bar: check for breakout of pause
        bar = df.iloc[i]
        pause_high = max(b['high'] for b in pause_bars)
        pause_low = min(b['low'] for b in pause_bars)
        
        entry = df.iloc[i+1]['open']
        
        # Continuation of up move
        if strong_up and bar['close'] > pause_high:
            sl = pause_low - atr * 0.2
            tp = entry + (entry - sl) * 2.0
            direction = 'LONG'
        # Continuation of down move
        elif strong_down and bar['close'] < pause_low:
            sl = pause_high + atr * 0.2
            tp = entry - (sl - entry) * 2.0
            direction = 'SHORT'
        else:
            continue
        
        sl_dist = abs(entry - sl)
        if sl_dist == 0:
            continue
        
        future = df.iloc[i+1:i+7]
        if len(future) < 3:
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
    
    return analyze(trades, "Momentum Pause", symbol)

# ============================================================================
# STRATEGY 4: RANGE EXTREME SNAP
# ============================================================================
def range_extreme_snap(df, symbol):
    """
    ORIGINAL: Range Extreme Snap
    
    Insight: Price tends to snap back from daily range extremes,
    especially when combined with intraday exhaustion.
    
    Unique approach:
    - Calculate rolling 24-bar (daily) high/low
    - When price hits extreme AND reverses strongly
    - Fade with target at mid-range
    """
    print(f"\n  Range Extreme Snap ({symbol})...")
    
    trades = []
    
    for i in range(30, len(df) - 5):
        bar = df.iloc[i]
        prev_bar = df.iloc[i-1]
        
        # 24-bar range (approximately 1 day on H1)
        range_high = df['high'].iloc[i-24:i].max()
        range_low = df['low'].iloc[i-24:i].min()
        range_mid = (range_high + range_low) / 2
        range_size = range_high - range_low
        
        atr = (df['high'] - df['low']).iloc[i-14:i].mean()
        if atr == 0 or range_size == 0:
            continue
        
        entry = df.iloc[i+1]['open']
        
        # At upper extreme + bearish reversal candle
        at_high = prev_bar['high'] >= range_high * 0.998
        bearish_reversal = (bar['close'] < bar['open'] and 
                          bar['close'] < prev_bar['low'] and
                          (bar['open'] - bar['close']) > atr * 0.5)
        
        if at_high and bearish_reversal:
            sl = range_high + atr * 0.3
            tp = range_mid
            direction = 'SHORT'
            
            sl_dist = abs(entry - sl)
            if sl_dist > 0:
                future = df.iloc[i+1:i+9]
                if len(future) >= 4:
                    hit_tp = future['low'].min() <= tp
                    hit_sl = future['high'].max() >= sl
                    if hit_tp and not hit_sl:
                        r_mult = (entry - tp) / sl_dist
                    elif hit_sl:
                        r_mult = -1.0
                    else:
                        r_mult = (entry - future.iloc[-1]['close']) / sl_dist
                    trades.append({'r': r_mult})
        
        # At lower extreme + bullish reversal candle
        at_low = prev_bar['low'] <= range_low * 1.002
        bullish_reversal = (bar['close'] > bar['open'] and 
                           bar['close'] > prev_bar['high'] and
                           (bar['close'] - bar['open']) > atr * 0.5)
        
        if at_low and bullish_reversal:
            sl = range_low - atr * 0.3
            tp = range_mid
            direction = 'LONG'
            
            sl_dist = abs(entry - sl)
            if sl_dist > 0:
                future = df.iloc[i+1:i+9]
                if len(future) >= 4:
                    hit_tp = future['high'].max() >= tp
                    hit_sl = future['low'].min() <= sl
                    if hit_tp and not hit_sl:
                        r_mult = (tp - entry) / sl_dist
                    elif hit_sl:
                        r_mult = -1.0
                    else:
                        r_mult = (future.iloc[-1]['close'] - entry) / sl_dist
                    trades.append({'r': r_mult})
    
    return analyze(trades, "Range Extreme Snap", symbol)

# ============================================================================
# STRATEGY 5: TRIPLE CANDLE BREAKOUT
# ============================================================================
def triple_candle_breakout(df, symbol):
    """
    ORIGINAL: Triple Candle Breakout
    
    Insight: Three inside bars followed by breakout = stored energy release.
    
    Pattern:
    - Bar 1: Large range (mother)
    - Bar 2: Inside bar (contained)
    - Bar 3: Smaller inside bar (further compression)
    - Bar 4: Breakout candle closes outside mother range
    """
    print(f"\n  Triple Candle Breakout ({symbol})...")
    
    trades = []
    
    for i in range(10, len(df) - 5):
        mother = df.iloc[i-3]
        inside1 = df.iloc[i-2]
        inside2 = df.iloc[i-1]
        breakout = df.iloc[i]
        
        atr = (df['high'] - df['low']).iloc[i-14:i].mean()
        if atr == 0:
            continue
        
        # Check pattern
        inside1_valid = (inside1['high'] < mother['high'] and 
                        inside1['low'] > mother['low'])
        inside2_valid = (inside2['high'] < inside1['high'] and 
                        inside2['low'] > inside1['low'])
        
        if not (inside1_valid and inside2_valid):
            continue
        
        # Mother bar must be significant
        mother_range = mother['high'] - mother['low']
        if mother_range < atr * 0.8:
            continue
        
        entry = df.iloc[i+1]['open']
        
        # Bullish breakout
        if breakout['close'] > mother['high']:
            sl = inside2['low'] - atr * 0.1
            tp = entry + (entry - sl) * 2.5
            direction = 'LONG'
        # Bearish breakout
        elif breakout['close'] < mother['low']:
            sl = inside2['high'] + atr * 0.1
            tp = entry - (sl - entry) * 2.5
            direction = 'SHORT'
        else:
            continue
        
        sl_dist = abs(entry - sl)
        if sl_dist == 0:
            continue
        
        future = df.iloc[i+1:i+8]
        if len(future) < 4:
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
    
    return analyze(trades, "Triple Candle Breakout", symbol)

def analyze(trades, name, symbol):
    if len(trades) < 20:
        print(f"    ⚠ Only {len(trades)} trades")
        return None
    
    split = int(len(trades) * 0.6)
    train = trades[:split]
    test = trades[split:]
    
    for period, t_list in [('Train', train), ('Test', test)]:
        r_mults = [t['r'] for t in t_list]
        wins = sum(1 for r in r_mults if r > 0)
        total = len(r_mults)
        win_rate = wins / total * 100 if total > 0 else 0
        exp = np.mean(r_mults) if r_mults else 0
        gross_win = sum(r for r in r_mults if r > 0)
        gross_loss = abs(sum(r for r in r_mults if r < 0))
        pf = gross_win / gross_loss if gross_loss > 0 else 0
        
        print(f"    {period}: {total} trades, WR {win_rate:.1f}%, Exp {exp:.3f}R, PF {pf:.2f}")
    
    # Use test results for verdict
    r_mults = [t['r'] for t in test]
    exp = np.mean(r_mults)
    gross_win = sum(r for r in r_mults if r > 0)
    gross_loss = abs(sum(r for r in r_mults if r < 0))
    pf = gross_win / gross_loss if gross_loss > 0 else 0
    
    if exp > 0.15 and pf > 1.2:
        print(f"    ✅ STRONG EDGE")
        verdict = 'STRONG'
    elif exp > 0:
        print(f"    ⚠️ WEAK")
        verdict = 'WEAK'
    else:
        print(f"    ❌ NONE")
        verdict = 'NONE'
    
    return {
        'name': name, 'symbol': symbol,
        'total_trades': len(trades),
        'test_trades': len(test),
        'exp': exp, 'pf': pf,
        'verdict': verdict
    }

def main():
    print("\n" + "="*60)
    print("  ORIGINAL STRATEGIES FROM RESEARCH INSIGHTS")
    print("  5 Novel Approaches | 24-Month Backtest")
    print("="*60)
    
    if not init_mt5():
        return
    
    # Focus on JPY pairs (best performers) plus control
    symbols = ['USDJPYm', 'AUDJPYm', 'EURUSDm', 'GBPUSDm', 'XAUUSDm']
    results = {}
    
    for sym in symbols:
        print(f"\n{'='*50}")
        print(f"  {sym} - H1")
        print(f"{'='*50}")
        
        df = fetch(sym, mt5.TIMEFRAME_H1, months=24)
        
        if df is None or len(df) < 1000:
            print(f"  ⚠ Insufficient data")
            continue
        
        print(f"  Loaded {len(df)} bars")
        
        for strategy in [quiet_before_storm, exhaustion_reversal, 
                        momentum_pause, range_extreme_snap, triple_candle_breakout]:
            result = strategy(df, sym)
            if result:
                key = f"{sym}_{result['name']}"
                results[key] = result
    
    mt5.shutdown()
    
    # Summary
    print(f"\n{'='*60}")
    print("  ORIGINAL STRATEGIES - RESULTS")
    print(f"{'='*60}")
    
    strong = []
    weak = []
    
    for k, v in sorted(results.items(), key=lambda x: x[1]['exp'], reverse=True):
        marker = "✅" if v['verdict'] == 'STRONG' else ("⚠️" if v['verdict'] == 'WEAK' else "❌")
        print(f"{marker} {k}: Exp {v['exp']:.3f}R, PF {v['pf']:.2f}, Trades: {v['test_trades']}")
        
        if v['verdict'] == 'STRONG':
            strong.append(k)
        elif v['verdict'] == 'WEAK':
            weak.append(k)
    
    output = PROJECT_ROOT / 'reports' / 'original_strategies.json'
    with open(output, 'w') as f:
        json.dump({'date': datetime.now().isoformat(), 'results': results,
                   'strong': strong, 'weak': weak}, f, indent=2, default=str)
    print(f"\n  ✓ Saved: {output}")
    
    if strong:
        print(f"\n  🎯 STRONG EDGE: {', '.join(strong)}")
    if weak:
        print(f"  ⚠️ WEAK EDGE: {', '.join(weak)}")

if __name__ == '__main__':
    main()
