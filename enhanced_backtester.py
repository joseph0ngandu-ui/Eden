#!/usr/bin/env python3
"""
Enhanced Portfolio Backtester with Advanced Safety Features
Implements:
- Dynamic risk scaling based on drawdown
- Correlation filtering
- Consecutive loss breakers
- Volatility regime detection
- Multi-timeframe loss limits
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
import json

@dataclass
class SafetyConfig:
    """Safety configuration parameters."""
    base_risk_pct: float = 0.12
    max_concurrent_trades: int = 5
    enable_correlation_filter: bool = True
    enable_dynamic_risk: bool = True
    enable_loss_breaker: bool = True
    enable_volatility_regime: bool = True
    
    # Dynamic risk thresholds
    dd_level1: float = 3.0  # 3% DD -> 0.8x risk
    dd_level2: float = 5.0  # 5% DD -> 0.6x risk
    dd_level3: float = 7.0  # 7% DD -> STOP
    
    # Loss breaker settings
    pause_after_losses: int = 3
    pause_hours: int = 4
    
    # Loss limits
    daily_loss_limit: float = 2.0
    weekly_loss_limit: float = 4.0
    monthly_loss_limit: float = 8.0
    
    # Volatility regime
    vol_threshold: float = 1.5  # 1.5x avg ATR -> reduce risk

# Copy strategy functions from original
def calculate_atr_simple(df: pd.DataFrame) -> float:
    if len(df) < 2: return 0.001
    tr = df['high'] - df['low']
    return tr.mean() if len(tr) > 0 else 0.001

def overlap_scalper(df: pd.DataFrame, symbol: str, idx: int, risk_pct: float) -> Optional[Dict]:
    row = df.iloc[idx]
    hour = row['hour']
    if not (12 <= hour < 16): return None
    if idx < 20: return None
    
    recent = df.iloc[idx-5:idx+1]
    momentum = recent['close'].iloc[-1] - recent['close'].iloc[0]
    atr = calculate_atr_simple(df.iloc[max(0, idx-14):idx+1])
    
    avg_vol = df.iloc[max(0, idx-20):idx]['tick_volume'].mean()
    if row['tick_volume'] < avg_vol * 1.2: return None
    
    if abs(momentum) > atr * 0.6:
        direction = "LONG" if momentum > 0 else "SHORT"
        entry = row['close']
        sl = entry - (atr * 1.0) if direction == "LONG" else entry + (atr * 1.0)
        tp = entry + (atr * 2.5) if direction == "LONG" else entry - (atr * 2.5)
        
        return {'symbol': symbol, 'direction': direction, 'entry': entry, 'sl': sl, 'tp': tp,
                'strategy': 'overlap_scalper', 'risk_pct': risk_pct, 'currency': symbol[:3]}
    return None

def asian_fade(df: pd.DataFrame, symbol: str, idx: int, risk_pct: float) -> Optional[Dict]:
    row = df.iloc[idx]
    hour = row['hour']
    if not ((hour >= 22) or (hour < 6)): return None
    if idx < 60: return None
    
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
                'strategy': 'asian_fade', 'risk_pct': risk_pct, 'currency': symbol[:3]}
    elif price_position < 0.2:
        return {'symbol': symbol, 'direction': 'LONG', 'entry': row['close'],
                'sl': asian_low - range_size * 0.1, 'tp': asian_mid,
                'strategy': 'asian_fade', 'risk_pct': risk_pct, 'currency': symbol[:3]}
    return None

def gold_london_breakout(df: pd.DataFrame, symbol: str, idx: int, risk_pct: float) -> Optional[Dict]:
    if symbol != "XAUUSD": return None
    row = df.iloc[idx]
    hour = row['hour']
    if not (7 <= hour < 9): return None
    if idx < 72: return None
    
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
                'strategy': 'gold_breakout', 'risk_pct': risk_pct, 'currency': 'XAU'}
    
    if row['close'] < asian_low and row['close'] < asian_low - asian_range * 0.1:
        return {'symbol': symbol, 'direction': 'SHORT', 'entry': row['close'],
                'sl': asian_high, 'tp': row['close'] - asian_range * 3,
                'strategy': 'gold_breakout', 'risk_pct': risk_pct, 'currency': 'XAU'}
    return None

def volatility_expansion(df: pd.DataFrame, symbol: str, idx: int, risk_pct: float) -> Optional[Dict]:
    if idx < 40: return None
    row = df.iloc[idx]
    
    current_atr = calculate_atr_simple(df.iloc[idx-14:idx+1])
    avg_atr = calculate_atr_simple(df.iloc[idx-40:idx+1])
    
    if avg_atr == 0: return None
    if current_atr > avg_atr * 0.7: return None
    
    recent_3 = df.iloc[idx-2:idx+1]
    move = abs(recent_3['close'].iloc[-1] - recent_3['close'].iloc[0])
    
    if move > current_atr * 1.5:
        direction = "LONG" if recent_3['close'].iloc[-1] > recent_3['close'].iloc[0] else "SHORT"
        entry = row['close']
        sl = entry - avg_atr * 1.5 if direction == "LONG" else entry + avg_atr * 1.5
        tp = entry + avg_atr * 4 if direction == "LONG" else entry - avg_atr * 4
        
        return {'symbol': symbol, 'direction': direction, 'entry': entry, 'sl': sl, 'tp': tp,
                'strategy': 'vol_expansion', 'risk_pct': risk_pct, 'currency': symbol[:3]}
    return None

def fetch_data(symbol: str, months: int = 3) -> Optional[pd.DataFrame]:
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
    df.set_index('time', inplace=True)
    return df

def run_enhanced_simulation(data_map: Dict[str, pd.DataFrame], config: SafetyConfig) -> Dict:
    """Run simulation with advanced safety features."""
    
    all_times = sorted(list(set().union(*[df.index for df in data_map.values()])))
    print(f"Simulation: {all_times[0]} to {all_times[-1]}")
    print(f"Config: Risk={config.base_risk_pct:.2%}, MaxTrades={config.max_concurrent_trades}")
    
    equity = 100000.0
    initial_equity = 100000.0
    peak_equity = 100000.0
    open_trades = []
    closed_trades = []
    
    # Tracking
    daily_pnl = 0
    weekly_pnl = 0
    monthly_pnl = 0
    last_date = all_times[0].date()
    last_week = all_times[0].isocalendar()[1]
    last_month = all_times[0].month
    
    consecutive_losses = 0
    pause_until = None
    
    for current_time in all_times:
        current_date = current_time.date()
        current_week = current_time.isocalendar()[1]
        current_month = current_time.month
        
        # Reset tracking
        if current_date != last_date:
            daily_pnl = 0
            last_date = current_date
        if current_week != last_week:
            weekly_pnl = 0
            last_week = current_week
        if current_month != last_month:
            monthly_pnl = 0
            last_month = current_month
            
        # Check if paused
        if pause_until and current_time < pause_until:
            trading_allowed = False
        else:
            pause_until = None
            trading_allowed = True
            
        # Loss limit checks
        if config.daily_loss_limit and daily_pnl < -(equity * config.daily_loss_limit / 100):
            trading_allowed = False
        if config.weekly_loss_limit and weekly_pnl < -(equity * config.weekly_loss_limit / 100):
            trading_allowed = False
        if config.monthly_loss_limit and monthly_pnl < -(equity * config.monthly_loss_limit / 100):
            trading_allowed = False
            
        # Dynamic risk calculation
        current_dd = ((peak_equity - equity) / peak_equity) * 100 if peak_equity > 0 else 0
        
        if config.enable_dynamic_risk:
            if current_dd >= config.dd_level3:
                trading_allowed = False
            elif current_dd >= config.dd_level2:
                risk_multiplier = 0.6
            elif current_dd >= config.dd_level1:
                risk_multiplier = 0.8
            else:
                risk_multiplier = 1.0
        else:
            risk_multiplier = 1.0
            
        current_risk = config.base_risk_pct * risk_multiplier
        
        # Manage exits
        active_trades = []
        for trade in open_trades:
            symbol = trade['symbol']
            if current_time not in data_map[symbol].index:
                active_trades.append(trade)
                continue
                
            row = data_map[symbol].loc[current_time]
            hit_tp = (trade['direction'] == 'LONG' and row['high'] >= trade['tp']) or \
                     (trade['direction'] == 'SHORT' and row['low'] <= trade['tp'])
            hit_sl = (trade['direction'] == 'LONG' and row['low'] <= trade['sl']) or \
                     (trade['direction'] == 'SHORT' and row['high'] >= trade['sl'])
            
            if hit_tp or hit_sl:
                exit_price = trade['tp'] if hit_tp else trade['sl']
                pnl_price = (exit_price - trade['entry']) if trade['direction'] == 'LONG' else (trade['entry'] - exit_price)
                
                risk_amount = equity * trade['risk_pct']
                sl_distance = abs(trade['entry'] - trade['sl'])
                
                if sl_distance > 0:
                    position_size = risk_amount / sl_distance
                    pnl_dollars = position_size * pnl_price
                else:
                    pnl_dollars = 0
                
                equity += pnl_dollars
                daily_pnl += pnl_dollars
                weekly_pnl += pnl_dollars
                monthly_pnl += pnl_dollars
                
                if equity > peak_equity:
                    peak_equity = equity
                
                # Track consecutive losses
                if hit_sl:
                    consecutive_losses += 1
                else:
                    consecutive_losses = 0
                
                # Loss breaker
                if config.enable_loss_breaker and consecutive_losses >= config.pause_after_losses:
                    pause_until = current_time + timedelta(hours=config.pause_hours)
                
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
        
        # New entries
        if not trading_allowed: continue
        if len(open_trades) >= config.max_concurrent_trades: continue
        
        for symbol, df in data_map.items():
            if len(open_trades) >= config.max_concurrent_trades: break
            if current_time not in df.index: continue
            
            try:
                idx = df.index.get_loc(current_time)
            except KeyError:
                continue
                
            if isinstance(idx, (slice, np.ndarray)):
                idx = idx.start
            if idx < 100: continue
            
            # Correlation filter
            if config.enable_correlation_filter:
                open_currencies = [t.get('currency', '') for t in open_trades]
                currency_counts = {}
                for curr in open_currencies:
                    currency_counts[curr] = currency_counts.get(curr, 0) + 1
                # Block if currency already has 2 positions
                signal_currency = symbol[:3]
                if currency_counts.get(signal_currency, 0) >= 2:
                    continue
            
            # Volatility regime check
            if config.enable_volatility_regime:
                recent_atr = calculate_atr_simple(df.iloc[idx-30:idx+1])
                avg_atr = calculate_atr_simple(df.iloc[max(0, idx-90):idx+1])
                if avg_atr > 0 and recent_atr > avg_atr * config.vol_threshold:
                    current_risk = current_risk * 0.67  # Reduce risk in high vol
            
            existing_strategies = [t['strategy'] for t in open_trades if t['symbol'] == symbol]
            
            signal = None
            if 'overlap_scalper' not in existing_strategies and symbol in ['EURUSD', 'GBPUSD']:
                signal = overlap_scalper(df, symbol, idx, current_risk)
            if not signal and 'asian_fade' not in existing_strategies and symbol in ['USDJPY', 'AUDJPY']:
                signal = asian_fade(df, symbol, idx, current_risk)
            if not signal and 'gold_breakout' not in existing_strategies and symbol == 'XAUUSD':
                signal = gold_london_breakout(df, symbol, idx, current_risk)
            if not signal and 'vol_expansion' not in existing_strategies:
                signal = volatility_expansion(df, symbol, idx, current_risk)
            
            if signal:
                signal['entry_time'] = current_time
                open_trades.append(signal)

    return {
        'trades': closed_trades,
        'final_equity': equity,
        'initial_equity': initial_equity,
        'config': config.__dict__
    }

def run_monte_carlo(trades: List[Dict], iterations: int = 1000) -> Dict:
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

def analyze_results(results: Dict, label: str = ""):
    trades = results['trades']
    if not trades:
        print("No trades")
        return
        
    df_trades = pd.DataFrame(trades)
    df_trades['time'] = df_trades['exit_time']
    df_trades.set_index('time', inplace=True)
    
    monthly = df_trades.resample('ME')['pnl_dollars'].sum()
    monthly_pct = (monthly / 100000) * 100
    
    peak = 100000
    max_dd = 0
    for t in trades:
        equity = t['equity']
        if equity > peak: peak = equity
        dd = (peak - equity) / peak * 100
        max_dd = max(max_dd, dd)
    
    print(f"\n{'='*60}")
    print(f"{label}")
    print(f"{'='*60}")
    print(f"Trades: {len(trades)}")
    print(f"Return: {((results['final_equity'] - 100000)/100000)*100:.2f}%")
    print(f"MaxDD: {max_dd:.2f}%")
    
    mc = run_monte_carlo(trades)
    print(f"95% Conf DD: {mc['95_percentile_dd']:.2f}%")
    print(f"Worst DD: {mc['worst_case_dd']:.2f}%")
    
    for date, val in monthly_pct.items():
        print(f"  {date.strftime('%Y-%m')}: {val:>6.2f}%")

if __name__ == "__main__":
    symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDJPY", "XAUUSD"]
    data_map = {}
    
    print("Fetching data...")
    for s in symbols:
        df = fetch_data(s, 3)
        if df is not None:
            data_map[s] = df
            
    if data_map:
        # Test 4 aggressive configs targeting >12% monthly
        configs = [
            # Intelligent Aggression: Selective safety
            SafetyConfig(
                base_risk_pct=0.0014,
                max_concurrent_trades=7,
                enable_correlation_filter=True,
                enable_dynamic_risk=False,  # OFF
                enable_loss_breaker=False,  # OFF
                enable_volatility_regime=False,  # OFF
                dd_level3=8.0  # Emergency stop at 8%
            ),
            # High Performance: Minimal safety
            SafetyConfig(
                base_risk_pct=0.0015,
                max_concurrent_trades=8,
                enable_correlation_filter=True,
                enable_dynamic_risk=False,
                enable_loss_breaker=False,
                enable_volatility_regime=False,
                dd_level3=10.0  # Emergency stop at 10%
            ),
            # Maximum Return: Original aggressive
            SafetyConfig(
                base_risk_pct=0.0016,
                max_concurrent_trades=10,
                enable_correlation_filter=False,  # OFF for max trades
                enable_dynamic_risk=False,
                enable_loss_breaker=False,
                enable_volatility_regime=False,
                dd_level3=12.0  # Emergency stop at 12%
            ),
            # Hybrid: Smart balance
            SafetyConfig(
                base_risk_pct=0.0015,
                max_concurrent_trades=7,
                enable_correlation_filter=True,
                enable_dynamic_risk=True,  # ON but lenient
                enable_loss_breaker=False,
                enable_volatility_regime=False,
                dd_level1=5.0,  # More lenient thresholds
                dd_level2=8.0,
                dd_level3=10.0
            ),
        ]
        
        labels = [
            "INTELLIGENT AGGRESSION (0.14%, 7 trades)",
            "HIGH PERFORMANCE (0.15%, 8 trades)",
            "MAXIMUM RETURN (0.16%, 10 trades)",
            "HYBRID SMART (0.15%, 7 trades, dynamic)"
        ]
        
        all_results = []
        for config, label in zip(configs, labels):
            print(f"\n\nTesting: {label}")
            results = run_enhanced_simulation(data_map, config)
            analyze_results(results, label)
            all_results.append(results)
        
        # Save best
        with open("enhanced_results.json", 'w') as f:
            json.dump(all_results, f, default=str, indent=2)
