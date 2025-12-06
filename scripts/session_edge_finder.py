#!/usr/bin/env python3
"""
SESSION-BASED EDGE FINDER
Focus on session timing patterns which have known institutional edge
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

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

def fetch(symbol, months=12):
    if not mt5.symbol_select(symbol, True):
        return None
    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M15, 
                                  datetime.now() - timedelta(days=months*30), datetime.now())
    if rates is None: return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df

def test_london_killzone(df, symbol):
    """
    LONDON KILLZONE (07:00-09:00 GMT)
    Fade the Asia session range at London open
    """
    print(f"\n  London Killzone ({symbol})...")
    
    trades = []
    
    for i in range(100, len(df) - 5):
        bar = df.iloc[i]
        hour = bar.name.hour
        
        # Only trade 07:00-09:00 GMT (London Open)
        if hour not in [7, 8]:
            continue
        
        # Get Asia range (00:00-07:00)
        asia_start = bar.name.replace(hour=0, minute=0)
        asia_end = bar.name.replace(hour=7, minute=0)
        asia = df[(df.index >= asia_start) & (df.index < asia_end)]
        
        if len(asia) < 20:
            continue
        
        asia_high = asia['high'].max()
        asia_low = asia['low'].min()
        asia_mid = (asia_high + asia_low) / 2
        asia_range = asia_high - asia_low
        
        atr = (df['high'] - df['low']).iloc[i-14:i].mean()
        if atr == 0 or asia_range < atr * 0.5:
            continue
        
        # FADE: If price breaks above Asia high, SHORT (reversal likely)
        # Entry at NEXT bar open
        next_bar = df.iloc[i+1] if i+1 < len(df) else None
        exit_bar = df.iloc[i+4] if i+4 < len(df) else None  # 1 hour hold
        
        if next_bar is None or exit_bar is None:
            continue
        
        entry = next_bar['open']
        
        # Fade high breakout
        if bar['close'] > asia_high:
            direction = 'SHORT'
            sl = asia_high + atr * 0.5
            tp = asia_mid
        # Fade low breakout
        elif bar['close'] < asia_low:
            direction = 'LONG'
            sl = asia_low - atr * 0.5
            tp = asia_mid
        else:
            continue
        
        # Calculate result
        sl_dist = abs(entry - sl)
        if sl_dist == 0:
            continue
        
        # Check if hit TP or SL within next 4 bars (1 hour)
        future = df.iloc[i+1:i+5]
        
        if direction == 'LONG':
            hit_tp = future['high'].max() >= tp
            hit_sl = future['low'].min() <= sl
            if hit_tp and not hit_sl:
                r_mult = (tp - entry) / sl_dist
            elif hit_sl:
                r_mult = -1.0
            else:
                # Exit at last bar close
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
        
        trades.append({'r': r_mult, 'win': r_mult > 0})
    
    return analyze_trades(trades, "London Killzone")

def test_ny_momentum(df, symbol):
    """
    NY SESSION MOMENTUM (13:00-16:00 GMT)
    Ride the momentum of first NY hour move
    """
    print(f"\n  NY Momentum ({symbol})...")
    
    trades = []
    
    for i in range(100, len(df) - 8):
        bar = df.iloc[i]
        hour = bar.name.hour
        
        # Signal at 13:00 (after first hour of NY)
        if hour != 13 or bar.name.minute != 0:
            continue
        
        # Get first NY hour (12:00-13:00)
        ny_start = bar.name.replace(hour=12, minute=0)
        ny_first = df[(df.index >= ny_start) & (df.index < bar.name)]
        
        if len(ny_first) < 4:
            continue
        
        ny_open = ny_first.iloc[0]['open']
        ny_close = ny_first.iloc[-1]['close']
        ny_range = ny_first['high'].max() - ny_first['low'].min()
        
        atr = (df['high'] - df['low']).iloc[i-14:i].mean()
        if atr == 0:
            continue
        
        # Only trade if first hour had strong move (>0.5 ATR)
        first_move = abs(ny_close - ny_open)
        if first_move < atr * 0.5:
            continue
        
        # Entry at next bar open
        next_bar = df.iloc[i+1] if i+1 < len(df) else None
        if next_bar is None:
            continue
        
        entry = next_bar['open']
        
        # Follow first hour direction
        if ny_close > ny_open:
            direction = 'LONG'
            sl = ny_first['low'].min() - atr * 0.2
            tp = entry + (entry - sl) * 2.0
        else:
            direction = 'SHORT'
            sl = ny_first['high'].max() + atr * 0.2
            tp = entry - (sl - entry) * 2.0
        
        sl_dist = abs(entry - sl)
        if sl_dist == 0:
            continue
        
        # Check result over next 2 hours (8 x 15min bars)
        future = df.iloc[i+1:i+9]
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
        
        trades.append({'r': r_mult, 'win': r_mult > 0})
    
    return analyze_trades(trades, "NY Momentum")

def test_session_bias(df, symbol):
    """
    SESSION BIAS STRATEGY
    Trade in direction of London bias during NY session
    """
    print(f"\n  Session Bias ({symbol})...")
    
    trades = []
    
    for i in range(200, len(df) - 8):
        bar = df.iloc[i]
        hour = bar.name.hour
        
        # Trade entry at 14:00-15:00 GMT
        if hour not in [14, 15]:
            continue
        
        # Get London session (07:00-12:00)
        london_start = bar.name.replace(hour=7, minute=0)
        london_end = bar.name.replace(hour=12, minute=0)
        london = df[(df.index >= london_start) & (df.index < london_end)]
        
        if len(london) < 15:
            continue
        
        london_open = london.iloc[0]['open']
        london_close = london.iloc[-1]['close']
        london_high = london['high'].max()
        london_low = london['low'].min()
        
        atr = (df['high'] - df['low']).iloc[i-14:i].mean()
        if atr == 0:
            continue
        
        # Calculate London bias
        london_move = london_close - london_open
        if abs(london_move) < atr * 0.8:  # Need strong bias
            continue
        
        # Current price relative to London range
        current = bar['close']
        
        # Entry at next bar
        next_bar = df.iloc[i+1] if i+1 < len(df) else None
        if next_bar is None:
            continue
        
        entry = next_bar['open']
        
        # Strong London up + pullback = LONG
        if london_move > 0 and current < london_close:
            direction = 'LONG'
            sl = london_low - atr * 0.3
            tp = london_high + atr * 0.5
        # Strong London down + pullback = SHORT
        elif london_move < 0 and current > london_close:
            direction = 'SHORT'
            sl = london_high + atr * 0.3
            tp = london_low - atr * 0.5
        else:
            continue
        
        sl_dist = abs(entry - sl)
        if sl_dist == 0:
            continue
        
        # Check result
        future = df.iloc[i+1:i+9]
        if len(future) < 4:
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
        
        trades.append({'r': r_mult, 'win': r_mult > 0})
    
    return analyze_trades(trades, "Session Bias")

def analyze_trades(trades, name):
    if len(trades) < 30:
        print(f"    ⚠ Only {len(trades)} trades - insufficient")
        return None
    
    # Split 60/40 for train/test
    split = int(len(trades) * 0.6)
    train = trades[:split]
    test = trades[split:]
    
    # Test period metrics (what matters)
    wins = sum(1 for t in test if t['win'])
    total = len(test)
    win_rate = wins / total * 100
    
    r_mults = [t['r'] for t in test]
    expectancy = np.mean(r_mults)
    
    gross_win = sum(r for r in r_mults if r > 0)
    gross_loss = abs(sum(r for r in r_mults if r < 0))
    pf = gross_win / gross_loss if gross_loss > 0 else 0
    
    print(f"    Test: {total} trades, WR {win_rate:.1f}%, Exp {expectancy:.3f}R, PF {pf:.2f}")
    
    if expectancy > 0.15 and pf > 1.2:
        print(f"    ✅ STRONG EDGE")
        return {'strategy': name, 'trades': total, 'win_rate': win_rate, 
                'expectancy': expectancy, 'profit_factor': pf}
    elif expectancy > 0:
        print(f"    ⚠️ WEAK EDGE")
        return {'strategy': name, 'trades': total, 'win_rate': win_rate,
                'expectancy': expectancy, 'profit_factor': pf}
    else:
        print(f"    ❌ NO EDGE")
        return None

def main():
    print("\n" + "="*50)
    print("  SESSION-BASED EDGE FINDER")
    print("  Institutional timing patterns")
    print("="*50)
    
    if not init_mt5():
        return
    
    symbols = ['EURUSDm', 'GBPUSDm', 'USDJPYm']
    all_results = {}
    
    for sym in symbols:
        print(f"\n{'='*40}")
        print(f"  {sym}")
        print(f"{'='*40}")
        
        df = fetch(sym)
        if df is None:
            continue
        
        print(f"  Loaded {len(df)} bars")
        
        # Test each strategy
        for test_func in [test_london_killzone, test_ny_momentum, test_session_bias]:
            result = test_func(df, sym)
            if result:
                key = f"{sym}_{result['strategy']}"
                all_results[key] = result
    
    mt5.shutdown()
    
    # Summary
    print(f"\n{'='*50}")
    print("  BEST RESULTS")
    print(f"{'='*50}")
    
    if all_results:
        sorted_results = sorted(all_results.items(), key=lambda x: x[1]['expectancy'], reverse=True)
        for k, v in sorted_results:
            marker = "✅" if v['expectancy'] > 0.15 else "⚠️"
            print(f"  {marker} {k}: Exp {v['expectancy']:.3f}R, PF {v['profit_factor']:.2f}")
    else:
        print("  ❌ No viable strategies found")
    
    # Save
    output = PROJECT_ROOT / 'reports' / 'session_edge_results.json'
    with open(output, 'w') as f:
        json.dump({'date': datetime.now().isoformat(), 'results': all_results}, f, indent=2)
    print(f"\n  ✓ Saved: {output}")

if __name__ == '__main__':
    main()
