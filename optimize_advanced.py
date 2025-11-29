#!/usr/bin/env python3
"""
Advanced ICT Optimizer - High Returns, Low Drawdown

Target: +12% returns over 6 months with <8% max drawdown
- Multiple entry logic variations
- XAUUSD support
- Gold-specific strategy
- Dynamic position sizing
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading.ict_strategies import ICTStrategyBot, Bar
from trading.models import Trade

def fetch_data(symbol: str, months: int = 6, timeframe=mt5.TIMEFRAME_M5) -> Optional[pd.DataFrame]:
    """Fetch historical data."""
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
    return df

def run_enhanced_backtest(symbol: str, df: pd.DataFrame, params: Dict) -> Dict:
    """Run backtest with enhanced entry logic."""
    bot = ICTStrategyBot()
    trades = []
    open_trade = None
    
    # Dynamic position sizing
    equity = 100000
    base_risk = params.get('risk_pct', 0.01)
    consecutive_wins = 0
    consecutive_losses = 0
    
    for idx, row in df.iterrows():
        bar = Bar(
            time=row['time'],
            open=row['open'],
            high=row['high'],
            low=row['low'],
            close=row['close'],
            volume=row['tick_volume']
        )
        
        bot.bars.append(bar)
        bot.engine.update_structure(bot.bars)
        
        if open_trade is None and consecutive_losses < 3:
            # Get signal with enhanced filters
            strategy_type = params.get('strategy', 'silver_bullet')
            
            if strategy_type == 'silver_bullet':
                signal = bot.run_2023_silver_bullet(bar, symbol)
            elif strategy_type == 'unicorn':
                signal = bot.run_2024_unicorn(bar, symbol)
            elif strategy_type == 'gold_asian_breakout':
                signal = run_gold_strategy(bot, bar, symbol, params)
            else:
                signal = None
            
            if signal:
                # Apply enhanced filters
                if not apply_entry_filters(bot, bar, signal, params):
                    continue
                
                # Dynamic position sizing
                current_risk = base_risk
                if consecutive_wins >= 3:
                    current_risk *= 1.5  # Increase after winning streak
                elif consecutive_losses >= 2:
                    current_risk *= 0.5  # Reduce after losses
                
                # Modify TP/SL
                rr = params.get('rr_ratio', 2.5)
                risk = abs(signal.entry_price - signal.sl)
                
                # Dynamic SL based on ATR
                if params.get('dynamic_sl', False):
                    atr = calculate_atr(df.iloc[max(0, idx-14):idx+1])
                    if signal.direction == "LONG":
                        signal.sl = signal.entry_price - (atr * 1.5)
                        signal.tp = signal.entry_price + (atr * rr * 1.5)
                    else:
                        signal.sl = signal.entry_price + (atr * 1.5)
                        signal.tp = signal.entry_price - (atr * rr * 1.5)
                else:
                    if signal.direction == "LONG":
                        signal.tp = signal.entry_price + (risk * rr)
                    else:
                        signal.tp = signal.entry_price - (risk * rr)
                
                signal.risk_pct = current_risk
                open_trade = signal
                open_trade.entry_time = bar.time
        else:
            if open_trade:
                # Trailing stop logic
                if params.get('trailing_stop', False):
                    apply_trailing_stop(open_trade, bar)
                
                # Check exit
                hit_tp = False
                hit_sl = False
                
                if open_trade.direction == "LONG":
                    hit_tp = bar.high >= open_trade.tp
                    hit_sl = bar.low <= open_trade.sl
                else:
                    hit_tp = bar.low <= open_trade.tp
                    hit_sl = bar.high >= open_trade.sl
                
                if hit_tp or hit_sl:
                    exit_price = open_trade.tp if hit_tp else open_trade.sl
                    pnl = (exit_price - open_trade.entry_price) if open_trade.direction == "LONG" else (open_trade.entry_price - exit_price)
                    
                    # Update equity
                    risk_pct = getattr(open_trade, 'risk_pct', base_risk)
                    pnl_dollars = pnl * 100000 * (equity * risk_pct / abs(open_trade.entry_price - open_trade.sl if open_trade.entry_price != open_trade.sl else 1))
                    equity += pnl_dollars
                    
                    # Track streaks
                    if pnl > 0:
                        consecutive_wins += 1
                        consecutive_losses = 0
                    else:
                        consecutive_losses += 1
                        consecutive_wins = 0
                    
                    trades.append({
                        'entry_time': open_trade.entry_time,
                        'exit_time': bar.time,
                        'direction': open_trade.direction,
                        'pnl': pnl,
                        'pnl_dollars': pnl_dollars,
                        'equity': equity,
                        'result': 'TP' if hit_tp else 'SL'
                    })
                    open_trade = None
    
    return calculate_metrics(trades, 100000)

def apply_entry_filters(bot, bar, signal, params) -> bool:
    """Apply enhanced entry filters."""
    # Volume filter
    if params.get('volume_filter', False):
        if len(bot.bars) >= 20:
            recent_volume = [b.volume for b in bot.bars[-20:]]
            avg_vol = np.mean(recent_volume if recent_volume else [1])
            if bar.volume < avg_vol * 1.2:  # Require 20% above average
                return False
    
    # Session filter
    if params.get('session_filter', False):
        hour = bar.time.hour
        # London (3-5 AM) or NY (10-11 AM, 14-15 PM)
        if not ((3 <= hour <= 5) or (10 <= hour <= 11) or (14 <= hour <= 15)):
            return False
    
    return True

def apply_trailing_stop(trade, bar):
    """Apply trailing stop at +1R."""
    if trade.direction == "LONG":
        risk = trade.entry_price - trade.sl
        if bar.close >= trade.entry_price + risk:  # +1R reached
            new_sl = trade.entry_price  # Move to breakeven
            trade.sl = max(trade.sl, new_sl)
    else:
        risk = trade.sl - trade.entry_price
        if bar.close <= trade.entry_price - risk:
            new_sl = trade.entry_price
            trade.sl = min(trade.sl, new_sl)

def run_gold_strategy(bot, bar, symbol, params) -> Optional[Trade]:
    """Gold-specific strategy: Asian range breakout."""
    hour = bar.time.hour
    
    # Asian session: 0-6 AM
    if 0 <= hour < 6:
        return None  # Building range
    
    # London open: 7-9 AM - trade breakouts
    if 7 <= hour < 9:
        if len(bot.bars) < 50:
            return None
        
        # Get Asian range (last 6 hours)
        asian_bars = bot.bars[-30:]  # ~30 M5 bars = 6 hours
        asian_high = max(b.high for b in asian_bars)
        asian_low = min(b.low for b in asian_bars)
        
        # Breakout above Asian high
        if bar.close > asian_high:
            sl = asian_low
            tp = bar.close + (2 * (asian_high - asian_low))
            return Trade(symbol, "LONG", bar.close, tp, sl, 0.7, 0, bar.time, 0, strategy="Gold_AsianBreakout")
        
        # Breakout below Asian low
        if bar.close < asian_low:
            sl = asian_high
            tp = bar.close - (2 * (asian_high - asian_low))
            return Trade(symbol, "SHORT", bar.close, tp, sl, 0.7, 0, bar.time, 0, strategy="Gold_AsianBreakout")
    
    return None

def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Calculate ATR."""
    if len(df) < 2:
        return 0.001
    
    high = df['high']
    low = df['low']
    close = df['close']
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(min(period, len(df)), min_periods=1).mean().iloc[-1]
    return float(atr) if not pd.isna(atr) else 0.001

def calculate_metrics(trades: List[Dict], initial_capital: float) -> Dict:
    """Calculate performance metrics."""
    if not trades or len(trades) < 10:
        return None
    
    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] < 0]
    
    # Drawdown
    equities = [initial_capital] + [t['equity'] for t in trades]
    peak = equities[0]
    max_dd_pct = 0
    
    for eq in equities:
        if eq > peak:
            peak = eq
        dd_pct = ((peak - eq) / peak * 100) if peak > 0 else 0
        max_dd_pct = max(max_dd_pct, dd_pct)
    
    # Returns
    total_return_pct = ((equities[-1] - initial_capital) / initial_capital) * 100
    win_rate = len(wins) / len(trades) * 100
    profit_factor = abs(sum(t['pnl'] for t in wins) / sum(t['pnl'] for t in losses)) if losses and sum(t['pnl'] for t in losses) != 0 else 0
    
    # Calmar
    calmar = (total_return_pct / max_dd_pct) if max_dd_pct > 0 else 0
    
    return {
        'total_trades': len(trades),
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'total_return_pct': total_return_pct,
        'max_dd_pct': max_dd_pct,
        'calmar': calmar,
        'final_equity': equities[-1]
    }

if __name__ == "__main__":
    print("Advanced ICT Optimizer - High Returns Edition")
    print("=" * 60)
    
    symbols = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]
    
    # Enhanced parameter grid
    param_variants = [
        # Silver Bullet variations
        {'strategy': 'silver_bullet', 'rr_ratio': 2.5, 'risk_pct': 0.01, 'volume_filter': True, 'session_filter': True},
        {'strategy': 'silver_bullet', 'rr_ratio': 3.0, 'risk_pct': 0.015, 'dynamic_sl': True, 'trailing_stop': True},
        {'strategy': 'silver_bullet', 'rr_ratio': 3.5, 'risk_pct': 0.02, 'volume_filter': True, 'trailing_stop': True},
        
        # Unicorn variations  
        {'strategy': 'unicorn', 'rr_ratio': 2.5, 'risk_pct': 0.01, 'session_filter': True},
        {'strategy': 'unicorn', 'rr_ratio': 3.0, 'risk_pct': 0.015, 'dynamic_sl': True},
        {'strategy': 'unicorn', 'rr_ratio': 3.5, 'risk_pct': 0.02, 'volume_filter': True, 'trailing_stop': True},
        
        # Gold strategy
        {'strategy': 'gold_asian_breakout', 'rr_ratio': 2.0, 'risk_pct': 0.015},
        {'strategy': 'gold_asian_breakout', 'rr_ratio': 2.5, 'risk_pct': 0.02},
    ]
    
    all_results = {}
    
    for symbol in symbols:
        print(f"\n{'-'*60}")
        print(f"Processing {symbol}")
        print(f"{'-'*60}")
        
        df = fetch_data(symbol, 6)
        if df is None:
            print(f"Failed to fetch data for {symbol}")
            continue
        
        print(f"Fetched {len(df)} M5 bars")
        
        for i, params in enumerate(param_variants):
            # Skip non-gold strategies on XAUUSD
            if symbol == "XAUUSD" and params['strategy'] != 'gold_asian_breakout':
                continue
            
            # Skip gold strategy on forex
            if symbol != "XAUUSD" and params['strategy'] == 'gold_asian_breakout':
                continue
            
            metrics = run_enhanced_backtest(symbol, df, params)
            
            if metrics and metrics['max_dd_pct'] < 8.0:
                key = f"{params['strategy']}_{symbol}_{i}"
                all_results[key] = {
                    'params': params,
                    **metrics
                }
                print(f"  {params['strategy'][:15]:15} | Return: {metrics['total_return_pct']:>6.2f}% | DD: {metrics['max_dd_pct']:>5.2f}% | Calmar: {metrics['calmar']:>5.2f}")
    
    # Save results
    with open("advanced_optimization_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Find best by return
    if all_results:
        best = max(all_results.items(), key=lambda x: x[1]['total_return_pct'])
        print(f"\n{'='*60}")
        print(f"BEST RESULT: {best[0]}")
        print(f"Return: {best[1]['total_return_pct']:.2f}% | DD: {best[1]['max_dd_pct']:.2f}% | Calmar: {best[1]['calmar']:.2f}")
        print(f"Params: {best[1]['params']}")
        print(f"{'='*60}")
    
    print(f"\nResults saved to: advanced_optimization_results.json")
