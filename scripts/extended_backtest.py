#!/usr/bin/env python3
"""
EXTENDED BACKTEST - 24 MONTHS DATA
Focus on promising strategies with larger sample size
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
    """Fetch extended data - 24 months"""
    if not mt5.symbol_select(symbol, True):
        return None
    rates = mt5.copy_rates_range(symbol, tf, datetime.now() - timedelta(days=months*30), datetime.now())
    if rates is None: return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df

def test_volatility_squeeze(df, symbol):
    """Vol Squeeze - the promising strategy"""
    print(f"\n  Volatility Squeeze ({symbol})...")
    
    trades = []
    
    for i in range(50, len(df) - 5):
        bar = df.iloc[i]
        
        # Current range
        curr_range = bar['high'] - bar['low']
        avg_range = (df['high'] - df['low']).iloc[i-20:i].mean()
        
        # Volatility squeeze detection
        vol = df['close'].iloc[i-20:i].pct_change().std()
        avg_vol = df['close'].iloc[i-50:i].pct_change().std()
        
        if vol > avg_vol * 0.6:
            continue
        
        # Breakout detection
        high_20 = df['high'].iloc[i-20:i].max()
        low_20 = df['low'].iloc[i-20:i].min()
        
        entry = df.iloc[i+1]['open']
        
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
        
        trades.append({'r': r_mult, 'time': bar.name})
    
    return analyze(trades, "Vol Squeeze", symbol)

def test_hourly_range_reversion(df, symbol):
    """Hourly Range Reversion"""
    print(f"\n  Hourly Range Reversion ({symbol})...")
    
    trades = []
    hourly = df.resample('1h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    
    for i in range(20, len(hourly) - 2):
        h = hourly.iloc[i]
        high, low = h['high'], h['low']
        rng = high - low
        
        if rng == 0:
            continue
        
        avg_rng = (hourly['high'] - hourly['low']).iloc[i-20:i].mean()
        if rng < avg_rng * 0.8:
            continue
        
        # RSI calculation
        delta = hourly['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi = (100 - (100 / (1 + gain/loss))).iloc[i]
        
        close = h['close']
        entry = hourly.iloc[i+1]['open']
        
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
        
        trades.append({'r': r_mult, 'time': h.name})
    
    return analyze(trades, "Hourly Range Rev", symbol)

def analyze(trades, name, symbol):
    if len(trades) < 30:
        print(f"    ⚠ Only {len(trades)} trades")
        return None
    
    # 60/40 train/test split
    split = int(len(trades) * 0.6)
    train_trades = trades[:split]
    test_trades = trades[split:]
    
    results = {}
    
    for period, t_list in [('TRAIN', train_trades), ('TEST', test_trades)]:
        r_mults = [t['r'] for t in t_list]
        wins = sum(1 for r in r_mults if r > 0)
        total = len(r_mults)
        win_rate = wins / total * 100 if total > 0 else 0
        exp = np.mean(r_mults) if r_mults else 0
        
        gross_win = sum(r for r in r_mults if r > 0)
        gross_loss = abs(sum(r for r in r_mults if r < 0))
        pf = gross_win / gross_loss if gross_loss > 0 else 0
        
        results[period] = {'trades': total, 'win_rate': win_rate, 'exp': exp, 'pf': pf}
    
    train = results['TRAIN']
    test = results['TEST']
    
    print(f"    TRAIN: {train['trades']} trades, WR {train['win_rate']:.1f}%, Exp {train['exp']:.3f}R, PF {train['pf']:.2f}")
    print(f"    TEST:  {test['trades']} trades, WR {test['win_rate']:.1f}%, Exp {test['exp']:.3f}R, PF {test['pf']:.2f}")
    
    # Check consistency between train and test
    if test['exp'] > 0.1 and test['pf'] > 1.15:
        print(f"    ✅ STRONG EDGE VERIFIED")
        verdict = "STRONG"
    elif test['exp'] > 0:
        print(f"    ⚠️ WEAK EDGE")
        verdict = "WEAK"
    else:
        print(f"    ❌ NO EDGE")
        verdict = "NONE"
    
    return {
        'strategy': name,
        'symbol': symbol,
        'total_trades': len(trades),
        'train': train,
        'test': test,
        'verdict': verdict
    }

def main():
    print("\n" + "="*60)
    print("  EXTENDED BACKTEST - 24 MONTHS DATA")
    print("  Focus on promising Vol Squeeze & Hourly Range strategies")
    print("="*60)
    
    if not init_mt5():
        return
    
    symbols = ['EURUSDm', 'GBPUSDm', 'USDJPYm', 'AUDJPYm', 'XAUUSDm']
    results = {}
    
    for sym in symbols:
        print(f"\n{'='*50}")
        print(f"  {sym} - 24 MONTH DATA")
        print(f"{'='*50}")
        
        # H1 data for both strategies
        df_h1 = fetch(sym, mt5.TIMEFRAME_H1, months=24)
        
        if df_h1 is None or len(df_h1) < 1000:
            print(f"  ⚠ Insufficient data")
            continue
        
        print(f"  Loaded {len(df_h1)} H1 bars (~{len(df_h1)/24/22:.1f} months)")
        
        # Test strategies
        for test_func in [test_volatility_squeeze, test_hourly_range_reversion]:
            result = test_func(df_h1, sym)
            if result:
                key = f"{sym}_{result['strategy']}"
                results[key] = result
    
    mt5.shutdown()
    
    # Summary
    print(f"\n{'='*60}")
    print("  FINAL RESULTS (24 MONTH EXTENDED BACKTEST)")
    print(f"{'='*60}")
    
    # Sort by test expectancy
    strong = []
    weak = []
    
    for k, v in sorted(results.items(), key=lambda x: x[1]['test']['exp'], reverse=True):
        test = v['test']
        line = f"  {k:<30} Test: {test['trades']:>4} trades, Exp {test['exp']:.3f}R, PF {test['pf']:.2f}"
        
        if v['verdict'] == 'STRONG':
            print(f"✅ {line}")
            strong.append(k)
        elif v['verdict'] == 'WEAK':
            print(f"⚠️ {line}")
            weak.append(k)
        else:
            print(f"❌ {line}")
    
    # Save results
    output = PROJECT_ROOT / 'reports' / 'extended_backtest_24mo.json'
    with open(output, 'w') as f:
        json.dump({
            'date': datetime.now().isoformat(),
            'months': 24,
            'results': results,
            'strong_edge': strong,
            'weak_edge': weak
        }, f, indent=2, default=str)
    
    print(f"\n  ✓ Saved: {output}")
    
    if strong:
        print(f"\n  🎯 STRONG EDGE STRATEGIES: {', '.join(strong)}")
    if weak:
        print(f"  ⚠️ WEAK EDGE STRATEGIES: {', '.join(weak)}")

if __name__ == '__main__':
    main()
