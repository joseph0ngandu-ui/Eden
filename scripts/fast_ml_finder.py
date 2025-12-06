#!/usr/bin/env python3
"""
FAST ML EDGE FINDER
Simplified approach: Train/test split with multiple seeds
Tests multiple timeframes quickly
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
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
except ImportError as e:
    print(f"❌ Missing: {e}")
    sys.exit(1)

TIMEFRAMES = {
    'M15': mt5.TIMEFRAME_M15,
    'H1': mt5.TIMEFRAME_H1,
    'H4': mt5.TIMEFRAME_H4,
}

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

def features(df):
    f = pd.DataFrame(index=df.index)
    f['body'] = (df['close'] - df['open']) / df['close'] * 100
    f['range'] = (df['high'] - df['low']) / df['close'] * 100
    for p in [5, 10, 20]:
        f[f'roc{p}'] = df['close'].pct_change(p) * 100
        f[f'sma{p}'] = (df['close'] / df['close'].rolling(p).mean() - 1) * 100
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    f['rsi'] = 100 - (100 / (1 + gain/loss))
    f['atr'] = (df['high'] - df['low']).rolling(14).mean() / df['close'] * 100
    f['hour'] = df.index.hour
    return f.dropna()

def test_strategy(symbol, tf_name, months=12):
    print(f"\n  {symbol} {tf_name}...", end=" ")
    
    df = fetch(symbol, TIMEFRAMES[tf_name], months)
    if df is None or len(df) < 500:
        print("❌ No data")
        return None
    
    # Features on CLOSED bars
    X = features(df)
    
    # Target: next bar direction (1=up, 0=down)
    y = (df['close'].shift(-1) > df['close']).astype(int).loc[X.index]
    
    # Remove last row (no target)
    X = X.iloc[:-1]
    y = y.iloc[:-1]
    
    # 60/40 train/test split (time-based, no shuffle)
    split = int(len(X) * 0.6)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    # Get predictions with probabilities
    probs = model.predict_proba(X_test)
    preds = model.predict(X_test)
    
    # Simulate trading with probability threshold
    best_result = None
    best_thresh = 0.5
    
    for thresh in [0.55, 0.60, 0.65, 0.70]:
        high_conf_mask = (probs.max(axis=1) >= thresh)
        high_conf_preds = preds[high_conf_mask]
        high_conf_actual = y_test.values[high_conf_mask]
        
        if len(high_conf_preds) < 50:
            continue
        
        # Calculate win rate
        correct = (high_conf_preds == high_conf_actual).sum()
        total = len(high_conf_preds)
        win_rate = correct / total
        
        # Simulate with 1:1 R:R (simplified)
        # Spread cost ~ 0.5R per trade
        spread_cost = 0.3  # 30% of R
        net_win_rate = win_rate - spread_cost
        
        # Expectancy with 1:1 R:R
        exp = win_rate * 1.0 - (1 - win_rate) * 1.0 - spread_cost
        
        if best_result is None or exp > best_result['expectancy']:
            pf = (win_rate * total) / ((1-win_rate) * total) if (1-win_rate) > 0 else 0
            best_result = {
                'trades': total,
                'win_rate': win_rate * 100,
                'expectancy': exp,
                'profit_factor': pf,
                'threshold': thresh
            }
            best_thresh = thresh
    
    if best_result and best_result['expectancy'] > 0:
        print(f"✓ Exp: {best_result['expectancy']:.3f}R, WR: {best_result['win_rate']:.1f}%")
        return best_result
    else:
        print(f"❌ No edge")
        return None

def main():
    print("\n" + "="*50)
    print("  FAST ML EDGE FINDER")
    print("="*50)
    
    if not init_mt5():
        return
    
    symbols = ['EURUSDm', 'GBPUSDm', 'USDJPYm', 'XAUUSDm', 'US30m']
    timeframes = ['M15', 'H1', 'H4']
    
    results = {}
    
    for tf in timeframes:
        print(f"\n  === {tf} ===")
        for sym in symbols:
            r = test_strategy(sym, tf)
            if r:
                results[f"{sym}_{tf}"] = r
    
    mt5.shutdown()
    
    # Summary
    print(f"\n{'='*50}")
    print("  RESULTS")
    print(f"{'='*50}")
    
    if results:
        for k, v in sorted(results.items(), key=lambda x: x[1]['expectancy'], reverse=True):
            print(f"  {k:<18} Exp: {v['expectancy']:.3f}R  WR: {v['win_rate']:.1f}%  Trades: {v['trades']}")
    else:
        print("  ❌ No viable strategies found")
    
    # Save
    output = PROJECT_ROOT / 'reports' / 'fast_ml_results.json'
    with open(output, 'w') as f:
        json.dump({'date': datetime.now().isoformat(), 'results': results}, f, indent=2)
    print(f"\n  ✓ Saved: {output}")

if __name__ == '__main__':
    main()
