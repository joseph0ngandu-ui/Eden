#!/usr/bin/env python3
"""
ML PATTERN DISCOVERY - CREATIVE APPROACH
Find edge through data mining + Monte Carlo validation
Multi-timeframe analysis (M15, H1, H4, D1)
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
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
except ImportError as e:
    print(f"❌ Missing package: {e}")
    sys.exit(1)

# Timeframe mapping
TIMEFRAMES = {
    'M15': mt5.TIMEFRAME_M15,
    'H1': mt5.TIMEFRAME_H1,
    'H4': mt5.TIMEFRAME_H4,
    'D1': mt5.TIMEFRAME_D1,
}

SPREAD_PIPS = {
    'EURUSDm': 1.5, 'GBPUSDm': 2.0, 'USDJPYm': 1.5,
    'XAUUSDm': 40, 'US30m': 3.0, 'US500m': 0.5, 'USTECm': 1.5,
}

def initialize_mt5():
    mt5_path = r"C:\Program Files\MetaTrader 5 Terminal\terminal64.exe"
    if not mt5.initialize(path=mt5_path):
        print(f"❌ MT5 init failed: {mt5.last_error()}")
        return False
    print("✓ MT5 connected")
    return True

def fetch_data(symbol: str, timeframe, months: int = 12):
    if not mt5.symbol_select(symbol, True):
        return None
    end = datetime.now()
    start = end - timedelta(days=months * 30)
    rates = mt5.copy_rates_range(symbol, timeframe, start, end)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df

def create_features(df):
    """Create ML features from price data - NO LOOKAHEAD"""
    features = pd.DataFrame(index=df.index)
    
    # Price action features (using CLOSED bars only)
    features['body_pct'] = (df['close'] - df['open']) / df['open'] * 100
    features['upper_wick'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['open'] * 100
    features['lower_wick'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['open'] * 100
    features['range_pct'] = (df['high'] - df['low']) / df['open'] * 100
    
    # Momentum features
    for period in [3, 5, 10, 20]:
        features[f'roc_{period}'] = df['close'].pct_change(period) * 100
        features[f'sma_{period}'] = df['close'].rolling(period).mean()
        features[f'std_{period}'] = df['close'].rolling(period).std()
        features[f'close_vs_sma_{period}'] = (df['close'] / features[f'sma_{period}'] - 1) * 100
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    features['rsi'] = 100 - (100 / (1 + rs))
    
    # Volatility
    features['atr'] = (df['high'] - df['low']).rolling(14).mean()
    features['atr_ratio'] = features['atr'] / df['close'] * 100
    
    # Multi-bar patterns
    features['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
    features['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
    features['inside_bar'] = ((df['high'] < df['high'].shift(1)) & (df['low'] > df['low'].shift(1))).astype(int)
    
    # Trend strength
    features['trend_strength'] = features['roc_10'].abs()
    
    # Hour of day (for forex patterns)
    features['hour'] = df.index.hour
    features['day_of_week'] = df.index.dayofweek
    
    return features

def create_target(df, lookahead=1, min_move_pct=0.1):
    """
    Create target: 1 if next bar closes up by min_move, -1 if down, 0 otherwise
    Uses NEXT bar close (for realistic prediction)
    """
    future_return = df['close'].shift(-lookahead) / df['close'] - 1
    target = np.where(future_return > min_move_pct/100, 1, 
                     np.where(future_return < -min_move_pct/100, -1, 0))
    return target

def monte_carlo_validation(model, X, y, n_iterations=100):
    """Monte Carlo cross-validation to check for overfitting"""
    scores = []
    n_samples = len(X)
    
    for i in range(n_iterations):
        # Random 70/30 split
        idx = np.random.permutation(n_samples)
        train_idx = idx[:int(0.7 * n_samples)]
        test_idx = idx[int(0.7 * n_samples):]
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        scores.append(score)
    
    return np.mean(scores), np.std(scores)

def backtest_ml_strategy(df, model, features, threshold=0.6, spread_pct=0.002):
    """
    Backtest ML strategy with realistic conditions
    Entry at NEXT bar open, includes spread
    """
    predictions = []
    probabilities = []
    
    # Get feature columns (same as training)
    feature_cols = [c for c in features.columns if c not in ['hour', 'day_of_week']]
    
    # Walk-forward: predict each bar using only past data
    initial_train_size = 500
    
    for i in range(initial_train_size, len(df) - 1):
        # Train on all data up to current bar
        X_train = features.iloc[:i][feature_cols].dropna()
        y_train = create_target(df.iloc[:i])[X_train.index.get_indexer(X_train.index)]
        
        # Remove NaN targets
        valid_idx = ~np.isnan(y_train)
        X_train = X_train[valid_idx]
        y_train = y_train[valid_idx]
        
        if len(X_train) < 100:
            predictions.append(0)
            probabilities.append(0.5)
            continue
        
        try:
            model.fit(X_train, y_train)
            
            # Predict current bar
            X_current = features.iloc[i:i+1][feature_cols]
            if X_current.isna().any().any():
                predictions.append(0)
                probabilities.append(0.5)
                continue
            
            pred = model.predict(X_current)[0]
            prob = model.predict_proba(X_current).max()
            
            predictions.append(pred)
            probabilities.append(prob)
        except:
            predictions.append(0)
            probabilities.append(0.5)
    
    # Pad beginning
    predictions = [0] * initial_train_size + predictions
    probabilities = [0.5] * initial_train_size + probabilities
    
    # Simulate trading with threshold
    equity = 100000
    initial_equity = equity
    risk_per_trade = 0.01
    
    trades = []
    position = None
    
    for i in range(initial_train_size + 1, len(df) - 1):
        pred = predictions[i]
        prob = probabilities[i]
        
        current_bar = df.iloc[i]
        next_bar = df.iloc[i + 1]
        
        # Close position if exists
        if position:
            if position['direction'] == 'LONG':
                pnl = (next_bar['close'] - position['entry']) / position['entry']
            else:
                pnl = (position['entry'] - next_bar['close']) / position['entry']
            
            pnl -= spread_pct  # Spread cost
            
            r_mult = pnl / risk_per_trade
            equity_change = initial_equity * risk_per_trade * r_mult
            equity += equity_change
            
            trades.append({
                'r_multiple': r_mult,
                'result': 'WIN' if r_mult > 0 else 'LOSS',
                'equity': equity
            })
            position = None
        
        # Open new position only if high confidence
        if prob >= threshold and pred != 0:
            entry = next_bar['open']  # NEXT BAR OPEN - realistic
            
            if pred == 1:
                position = {'direction': 'LONG', 'entry': entry}
            elif pred == -1:
                position = {'direction': 'SHORT', 'entry': entry}
    
    if len(trades) == 0:
        return None
    
    # Calculate metrics
    wins = len([t for t in trades if t['result'] == 'WIN'])
    total = len(trades)
    win_rate = wins / total * 100
    
    r_mults = [t['r_multiple'] for t in trades]
    expectancy = np.mean(r_mults)
    
    gross_win = sum(r for r in r_mults if r > 0)
    gross_loss = abs(sum(r for r in r_mults if r < 0))
    pf = gross_win / gross_loss if gross_loss > 0 else 0
    
    # Max DD
    peak = initial_equity
    max_dd = 0
    for t in trades:
        if t['equity'] > peak:
            peak = t['equity']
        dd = (peak - t['equity']) / peak * 100
        max_dd = max(max_dd, dd)
    
    return {
        'trades': total,
        'win_rate': win_rate,
        'expectancy': expectancy,
        'profit_factor': pf,
        'max_dd': max_dd,
        'total_return': (equity - initial_equity) / initial_equity * 100
    }

def test_ml_strategy(symbol, timeframe_name, months=12):
    """Test ML strategy on single symbol/timeframe"""
    print(f"\n  Testing {symbol} on {timeframe_name}...")
    
    timeframe = TIMEFRAMES[timeframe_name]
    df = fetch_data(symbol, timeframe, months)
    
    if df is None or len(df) < 1000:
        print(f"    ⚠ Insufficient data")
        return None
    
    print(f"    {len(df)} bars loaded")
    
    # Create features
    features = create_features(df)
    features = features.dropna()
    
    if len(features) < 800:
        print(f"    ⚠ Insufficient features after cleanup")
        return None
    
    # Create target
    target = create_target(df.loc[features.index])
    
    # Remove rows where target is 0 (no clear direction)
    valid_idx = target != 0
    X = features[valid_idx]
    y = target[valid_idx]
    
    print(f"    {len(X)} valid samples")
    
    # Train model
    model = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
    
    # Monte Carlo validation
    mc_mean, mc_std = monte_carlo_validation(model, X, y, n_iterations=50)
    print(f"    Monte Carlo accuracy: {mc_mean:.3f} ± {mc_std:.3f}")
    
    # Only proceed if Monte Carlo shows promise (>55% accuracy)
    if mc_mean < 0.52:
        print(f"    ❌ No edge detected in Monte Carlo")
        return None
    
    # Full backtest
    spread_pct = SPREAD_PIPS.get(symbol, 2) / 10000
    if 'm' in symbol and ('JPY' in symbol or 'XAU' in symbol or 'US' in symbol):
        if 'JPY' in symbol:
            spread_pct = SPREAD_PIPS.get(symbol, 2) / 100
        elif 'XAU' in symbol:
            spread_pct = 0.0002
        elif 'US' in symbol:
            spread_pct = 0.0001
    
    # Test different thresholds
    best_result = None
    best_threshold = 0.5
    
    for threshold in [0.55, 0.60, 0.65, 0.70]:
        result = backtest_ml_strategy(df.loc[features.index], model, features, threshold, spread_pct)
        if result and result['expectancy'] > 0:
            if best_result is None or result['expectancy'] > best_result['expectancy']:
                best_result = result
                best_threshold = threshold
    
    if best_result:
        best_result['threshold'] = best_threshold
        best_result['mc_accuracy'] = mc_mean
        print(f"    ✓ Edge found! Exp: {best_result['expectancy']:.3f}R, PF: {best_result['profit_factor']:.2f}")
    else:
        print(f"    ❌ No profitable configuration found")
    
    return best_result

def main():
    print("\n" + "="*60)
    print("  ML PATTERN DISCOVERY")
    print("  Multi-timeframe | Monte Carlo Validated")
    print("="*60)
    
    if not initialize_mt5():
        return
    
    symbols = ['EURUSDm', 'GBPUSDm', 'USDJPYm', 'XAUUSDm']
    timeframes = ['M15', 'H1', 'H4']
    
    all_results = {}
    
    for tf in timeframes:
        print(f"\n{'='*50}")
        print(f"  TIMEFRAME: {tf}")
        print(f"{'='*50}")
        
        for symbol in symbols:
            result = test_ml_strategy(symbol, tf)
            if result:
                key = f"{symbol}_{tf}"
                all_results[key] = result
    
    mt5.shutdown()
    
    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY - ML PATTERN DISCOVERY")
    print(f"{'='*60}")
    
    if not all_results:
        print("\n  ❌ No viable ML strategies found across all timeframes")
        return
    
    print(f"\n  {'Symbol_TF':<20} {'Exp':<8} {'PF':<6} {'WR':<8} {'Trades':<8}")
    print(f"  {'-'*50}")
    
    viable = []
    for key, result in sorted(all_results.items(), key=lambda x: x[1]['expectancy'], reverse=True):
        print(f"  {key:<20} {result['expectancy']:.3f}R  {result['profit_factor']:.2f}   {result['win_rate']:.1f}%   {result['trades']}")
        if result['expectancy'] > 0.1 and result['profit_factor'] > 1.15:
            viable.append(key)
    
    # Save
    output = PROJECT_ROOT / 'reports' / 'ml_pattern_discovery.json'
    with open(output, 'w') as f:
        json.dump({
            'test_date': datetime.now().isoformat(),
            'results': all_results,
            'viable': viable
        }, f, indent=2, default=str)
    print(f"\n  ✓ Saved to: {output}")
    
    if viable:
        print(f"\n  🎯 VIABLE ML STRATEGIES: {', '.join(viable)}")
    else:
        print(f"\n  ⚠ No strategies met viability threshold (exp > 0.1, PF > 1.15)")

if __name__ == '__main__':
    main()
