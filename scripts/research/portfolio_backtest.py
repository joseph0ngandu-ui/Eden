#!/usr/bin/env python3
"""
COMPREHENSIVE PORTFOLIO BACKTEST
Simulates ALL active strategies running in parallel on their respective pairs/timeframes.

Strategies:
1. Index Volatility Expansion (M15) - US30, USTEC, US500
2. Gold Spread Hunter (M15) - XAUUSD
3. Forex Volatility Squeeze (M5) - EURUSD, USDJPY
4. Momentum Continuation (D1) - USDCAD, EURUSD, EURJPY, CADJPY
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta, time as dt_time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

@dataclass
class Trade:
    symbol: str
    strategy: str
    direction: str
    entry_price: float
    sl: float
    tp: float
    entry_time: pd.Timestamp
    risk_pct: float  # Actual risk % used

@dataclass
class TradeResult:
    trade: Trade
    exit_price: float
    exit_time: pd.Timestamp
    pnl_r: float  # R-multiple
    pnl_pct: float  # Actual % P&L based on risk

def calculate_atr(df, period=14):
    h, l, c = df['high'], df['low'], df['close'].shift(1)
    tr = pd.concat([h-l, (h-c).abs(), (l-c).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

# ============================================================
# STRATEGY 1: INDEX VOLATILITY EXPANSION (M15)
# ============================================================
def strategy_index_vol_expansion(df: pd.DataFrame, symbol: str) -> Optional[Trade]:
    """Bollinger squeeze breakout on indices during NY session."""
    if 'US' not in symbol: return None
    if len(df) < 100: return None
    
    bar = df.iloc[-1]
    hour = pd.to_datetime(bar.name).hour
    
    # NY Session only (13:00-20:00 Server)
    if not (13 <= hour <= 20): return None
    
    closes = df['close']
    sma_20 = closes.rolling(20).mean()
    std_20 = closes.rolling(20).std()
    upper = sma_20 + 2 * std_20
    lower = sma_20 - 2 * std_20
    bandwidth = (upper - lower) / sma_20
    avg_bw = bandwidth.rolling(50).mean()
    
    # Squeeze check
    if bandwidth.iloc[-1] >= avg_bw.iloc[-1] * 0.8: return None
    
    recent_squeeze = False
    for i in range(1, 4):
        if bandwidth.iloc[-i-1] < avg_bw.iloc[-i-1] * 0.8:
            recent_squeeze = True
            break
    if not recent_squeeze: return None
    
    price = bar['close']
    atr = calculate_atr(df).iloc[-1]
    ema_50 = closes.ewm(span=50).mean().iloc[-1]
    
    if pd.isna(atr) or atr == 0: return None
    
    long_breakout = price > upper.iloc[-1] and price > ema_50
    short_breakout = price < lower.iloc[-1] and price < ema_50
    
    if long_breakout:
        sl = sma_20.iloc[-1]
        risk = price - sl
        tp = price + risk * 1.5
        return Trade(symbol, "Index_VolExpansion", "LONG", price, sl, tp, bar.name, 0.75)
    if short_breakout:
        sl = sma_20.iloc[-1]
        risk = sl - price
        tp = price - risk * 1.5
        return Trade(symbol, "Index_VolExpansion", "SHORT", price, sl, tp, bar.name, 0.75)
    return None

# ============================================================
# STRATEGY 2: GOLD SPREAD HUNTER (M15)
# ============================================================
def strategy_gold_spread_hunter(df: pd.DataFrame, symbol: str) -> Optional[Trade]:
    """Momentum strategy for Gold during low-spread periods."""
    if 'XAU' not in symbol: return None
    if len(df) < 100: return None
    
    bar = df.iloc[-1]
    hour = pd.to_datetime(bar.name).hour
    
    # London/NY overlap (best spreads)
    if not (8 <= hour <= 16): return None
    
    atr = calculate_atr(df).iloc[-1]
    if pd.isna(atr) or atr == 0: return None
    
    ema_20 = df['close'].ewm(span=20).mean()
    ema_50 = df['close'].ewm(span=50).mean()
    
    price = bar['close']
    
    # Momentum: EMA cross + price above both
    bull = ema_20.iloc[-1] > ema_50.iloc[-1] and price > ema_20.iloc[-1]
    bear = ema_20.iloc[-1] < ema_50.iloc[-1] and price < ema_20.iloc[-1]
    
    # Trend confirmation
    if bull and ema_20.iloc[-2] <= ema_50.iloc[-2]:  # Fresh cross
        sl = price - 1.5 * atr
        tp = price + 2.0 * atr
        return Trade(symbol, "Gold_SpreadHunter", "LONG", price, sl, tp, bar.name, 0.50)
    if bear and ema_20.iloc[-2] >= ema_50.iloc[-2]:
        sl = price + 1.5 * atr
        tp = price - 2.0 * atr
        return Trade(symbol, "Gold_SpreadHunter", "SHORT", price, sl, tp, bar.name, 0.50)
    return None

# ============================================================
# STRATEGY 3: FOREX VOLATILITY SQUEEZE (M5)
# ============================================================
def strategy_forex_vol_squeeze(df: pd.DataFrame, symbol: str) -> Optional[Trade]:
    """Volatility squeeze breakout for EUR/JPY pairs."""
    if 'EUR' not in symbol and 'JPY' not in symbol: return None
    if 'GBP' in symbol: return None  # Excluded
    if len(df) < 100: return None
    
    bar = df.iloc[-1]
    
    closes = df['close']
    sma_20 = closes.rolling(20).mean()
    std_20 = closes.rolling(20).std()
    upper = sma_20 + 2 * std_20
    lower = sma_20 - 2 * std_20
    bandwidth = (upper - lower) / sma_20
    avg_bw = bandwidth.rolling(50).mean()
    
    if bandwidth.iloc[-1] >= avg_bw.iloc[-1] * 0.7: return None
    
    price = bar['close']
    atr = calculate_atr(df).iloc[-1]
    if pd.isna(atr) or atr == 0: return None
    
    if price > upper.iloc[-1]:
        sl = sma_20.iloc[-1]
        risk = price - sl
        tp = price + risk * 1.2
        return Trade(symbol, "Forex_VolSqueeze", "LONG", price, sl, tp, bar.name, 0.25)
    if price < lower.iloc[-1]:
        sl = sma_20.iloc[-1]
        risk = sl - price
        tp = price - risk * 1.2
        return Trade(symbol, "Forex_VolSqueeze", "SHORT", price, sl, tp, bar.name, 0.25)
    return None

# ============================================================
# STRATEGY 4: MOMENTUM CONTINUATION (D1)
# ============================================================
def strategy_momentum_continuation(df: pd.DataFrame, symbol: str) -> Optional[Trade]:
    """Enter pullback after strong D1 candle."""
    if symbol not in ['USDCADm', 'EURUSDm', 'EURJPYm', 'CADJPYm']: return None
    if len(df) < 20: return None
    
    daily_ranges = df['high'] - df['low']
    adr = daily_ranges.rolling(14).mean()
    
    if len(df) < 2: return None
    yesterday = df.iloc[-2]
    today = df.iloc[-1]
    
    if pd.isna(adr.iloc[-2]) or adr.iloc[-2] == 0: return None
    
    yest_range = yesterday['high'] - yesterday['low']
    if yest_range < 1.3 * adr.iloc[-2]: return None
    
    bullish = yesterday['close'] > yesterday['open']
    entry = today['open']
    atr = calculate_atr(df).iloc[-1] if not pd.isna(calculate_atr(df).iloc[-1]) else adr.iloc[-1]
    
    if bullish:
        sl = yesterday['low']
        risk = entry - sl
        if risk <= 0: return None
        tp = entry + risk * 1.5
        return Trade(symbol, "Momentum_Continuation", "LONG", entry, sl, tp, today.name, 0.50)
    else:
        sl = yesterday['high']
        risk = sl - entry
        if risk <= 0: return None
        tp = entry - risk * 1.5
        return Trade(symbol, "Momentum_Continuation", "SHORT", entry, sl, tp, today.name, 0.50)

def simulate_trade_outcome(trade: Trade, future_df: pd.DataFrame) -> TradeResult:
    """Simulate trade outcome using future price data."""
    exit_price = trade.entry_price
    exit_time = trade.entry_time
    
    for idx, row in future_df.iterrows():
        if trade.direction == "LONG":
            if row['low'] <= trade.sl:
                exit_price = trade.sl
                exit_time = idx
                break
            if row['high'] >= trade.tp:
                exit_price = trade.tp
                exit_time = idx
                break
        else:  # SHORT
            if row['high'] >= trade.sl:
                exit_price = trade.sl
                exit_time = idx
                break
            if row['low'] <= trade.tp:
                exit_price = trade.tp
                exit_time = idx
                break
        exit_price = row['close']
        exit_time = idx
    
    risk_amt = abs(trade.entry_price - trade.sl)
    if trade.direction == "LONG":
        pnl = exit_price - trade.entry_price
    else:
        pnl = trade.entry_price - exit_price
    
    pnl_r = pnl / risk_amt if risk_amt > 0 else 0
    pnl_pct = pnl_r * trade.risk_pct
    
    return TradeResult(trade, exit_price, exit_time, pnl_r, pnl_pct)

def run_portfolio_backtest():
    try:
        import MetaTrader5 as mt5
    except ImportError:
        print("MetaTrader5 not available")
        return
    
    if not mt5.initialize(path=r"C:\Program Files\MetaTrader 5 Terminal\terminal64.exe"):
        print("MT5 init failed")
        return
    
    print("="*70)
    print("  COMPREHENSIVE PORTFOLIO BACKTEST")
    print("  All Active Strategies in Parallel")
    print("="*70)
    
    end = datetime.now()
    start = end - timedelta(days=90)
    
    # Define strategy configurations
    strategies = [
        {"name": "Index_VolExpansion", "symbols": ["US30m", "USTECm", "US500m"], "tf": mt5.TIMEFRAME_M15, "func": strategy_index_vol_expansion},
        {"name": "Gold_SpreadHunter", "symbols": ["XAUUSDm"], "tf": mt5.TIMEFRAME_M15, "func": strategy_gold_spread_hunter},
        {"name": "Forex_VolSqueeze", "symbols": ["EURUSDm", "USDJPYm"], "tf": mt5.TIMEFRAME_M5, "func": strategy_forex_vol_squeeze},
        {"name": "Momentum_Continuation", "symbols": ["USDCADm", "EURUSDm", "EURJPYm", "CADJPYm"], "tf": mt5.TIMEFRAME_D1, "func": strategy_momentum_continuation},
    ]
    
    all_results: List[TradeResult] = []
    strategy_results: Dict[str, List[TradeResult]] = {}
    
    for strat in strategies:
        strat_name = strat["name"]
        strategy_results[strat_name] = []
        
        for symbol in strat["symbols"]:
            rates = mt5.copy_rates_range(symbol, strat["tf"], start, end)
            if rates is None or len(rates) < 100:
                continue
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            # Iterate through bars
            lookback = 100 if strat["tf"] != mt5.TIMEFRAME_D1 else 20
            future_bars = 200 if strat["tf"] != mt5.TIMEFRAME_D1 else 5
            
            for i in range(lookback, len(df) - future_bars):
                window = df.iloc[:i+1]
                signal = strat["func"](window, symbol)
                
                if signal:
                    future = df.iloc[i+1:i+1+future_bars]
                    result = simulate_trade_outcome(signal, future)
                    all_results.append(result)
                    strategy_results[strat_name].append(result)
    
    mt5.shutdown()
    
    # Sort all results by time
    all_results.sort(key=lambda x: x.trade.entry_time)
    
    # Calculate per-strategy metrics
    print("\n" + "="*70)
    print("  STRATEGY BREAKDOWN")
    print("="*70)
    
    total_r = 0
    total_pct = 0
    
    for strat_name, results in strategy_results.items():
        if not results:
            print(f"\n{strat_name}: No trades")
            continue
        
        r_vals = [r.pnl_r for r in results]
        pct_vals = [r.pnl_pct for r in results]
        wins = len([r for r in r_vals if r > 0])
        
        strat_r = sum(r_vals)
        strat_pct = sum(pct_vals)
        total_r += strat_r
        total_pct += strat_pct
        
        print(f"\n{strat_name}:")
        print(f"  Trades: {len(results)} | WR: {wins/len(results)*100:.1f}%")
        print(f"  Total R: {strat_r:.1f}R | Equity %: {strat_pct:.2f}%")
    
    # Calculate combined equity curve
    print("\n" + "="*70)
    print("  COMBINED PORTFOLIO")
    print("="*70)
    
    if not all_results:
        print("No trades generated")
        return
    
    # Equity curve
    equity = [100.0]  # Start at 100%
    for r in all_results:
        equity.append(equity[-1] * (1 + r.pnl_pct / 100))
    
    equity = np.array(equity)
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak * 100
    max_dd = np.max(drawdown)
    
    final_equity = equity[-1]
    total_return = (final_equity - 100)
    
    print(f"\n  Total Trades: {len(all_results)}")
    print(f"  Total R: {total_r:.1f}R")
    print(f"  Total Return: {total_return:.2f}%")
    print(f"  Max Drawdown: {max_dd:.2f}%")
    
    # Monthly breakdown
    print("\n  Monthly Performance:")
    monthly = {}
    for r in all_results:
        month = r.trade.entry_time.strftime("%Y-%m")
        if month not in monthly:
            monthly[month] = {"r": 0, "pct": 0, "trades": 0}
        monthly[month]["r"] += r.pnl_r
        monthly[month]["pct"] += r.pnl_pct
        monthly[month]["trades"] += 1
    
    for month, data in sorted(monthly.items()):
        print(f"    {month}: {data['trades']} trades | {data['r']:.1f}R | {data['pct']:.2f}%")
    
    # FundedNext Validation
    print("\n" + "="*70)
    print("  FUNDEDNEXT VALIDATION")
    print("="*70)
    
    passed = True
    
    if max_dd > 9.5:
        print(f"  ❌ Max DD: {max_dd:.2f}% > 9.5%")
        passed = False
    else:
        print(f"  ✅ Max DD: {max_dd:.2f}% < 9.5%")
    
    # Check daily DD (simplified)
    if max_dd > 4.5:
        print(f"  ⚠️ Note: Peak DD {max_dd:.2f}% may exceed daily 5% if concentrated")
    else:
        print(f"  ✅ DD within daily limits")
    
    avg_monthly = total_return / 3 if len(monthly) > 0 else 0
    if avg_monthly >= 13:
        print(f"  ✅ Avg Monthly: {avg_monthly:.1f}% >= 13%")
    else:
        print(f"  ⚠️ Avg Monthly: {avg_monthly:.1f}% < 13% target")
    
    print(f"\n  VERDICT: {'✅ PORTFOLIO VALIDATED' if passed else '❌ NEEDS ADJUSTMENT'}")

if __name__ == "__main__":
    run_portfolio_backtest()
