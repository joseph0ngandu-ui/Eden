#!/usr/bin/env python3
"""
Backtest RSI+SMA strategy on MT5 historical data for a recent window.

- Connects to an installed MetaTrader 5 terminal via MetaTrader5 Python module
- Fetches historical bars for a symbol and timeframe
- Simulates simple long/short entries with SL/TP or flip-on-opposite-signal exits
- Prints summary and saves CSV of trades if requested

Usage examples:
  python backtest_vix100.py --days 7 --timeframe M1
  python backtest_vix100.py --days 7 --symbol "Volatility 100 Index" --rsi-period 14 --sma-period 20 \
      --rsi-overbought 70 --rsi-oversold 30 --sl-pips 50 --tp-pips 100
"""
import argparse
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

import sys
from pathlib import Path

# Ensure local modules can import if needed
sys.path.insert(0, str(Path(__file__).parent))

try:
    import MetaTrader5 as mt5
except Exception as e:
    print("MetaTrader5 module not found. Install with: pip install MetaTrader5")
    raise

import pandas as pd
import numpy as np


def parse_timeframe(tf_str: str) -> int:
    tf_str = tf_str.upper()
    mapping = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
    }
    if tf_str not in mapping:
        raise ValueError(f"Unsupported timeframe: {tf_str}")
    return mapping[tf_str]


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def connect_and_resolve_symbol(primary: str, alternatives: List[str]) -> Optional[str]:
    if not mt5.initialize():
        print(f"MT5 initialization failed: {mt5.last_error()}")
        return None
    sym = primary
    info = mt5.symbol_info(sym)
    if info is None:
        for alt in alternatives:
            info = mt5.symbol_info(alt)
            if info is not None:
                sym = alt
                print(f"Using alternative symbol: {sym}")
                break
    if info is None:
        print(f"None of the symbols found: {[primary] + alternatives}")
        return None
    if not info.visible:
        mt5.symbol_select(sym, True)
    return sym


def fetch_rates(symbol: str, timeframe: int, start: datetime, end: datetime) -> Optional[pd.DataFrame]:
    rates = mt5.copy_rates_range(symbol, timeframe, start, end)
    if rates is None:
        print(f"Failed to fetch rates: {mt5.last_error()}")
        return None
    df = pd.DataFrame(rates)
    if df.empty:
        print("No data returned for the requested range.")
        return None
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df


def simulate(df: pd.DataFrame,
             symbol: str,
             rsi_period: int,
             sma_period: int,
             rsi_overbought: float,
             rsi_oversold: float,
             sl_pips: float,
             tp_pips: float) -> Dict[str, Any]:
    # Indicators
    df = df.copy()
    df['rsi'] = rsi(df['close'], rsi_period)
    df['sma'] = df['close'].rolling(window=sma_period).mean()

    # Get point size
    sinfo = mt5.symbol_info(symbol)
    if sinfo is None:
        raise RuntimeError(f"symbol_info returned None for {symbol}")
    point = sinfo.point

    # Warmup index
    start_idx = max(rsi_period, sma_period) + 1
    if len(df) <= start_idx:
        return {"trades": [], "stats": {"message": "Insufficient data after warmup"}}

    open_pos = None  # dict with keys: side, entry_price, entry_time, sl, tp
    trades: List[Dict[str, Any]] = []

    def signal(row) -> str:
        if np.isnan(row['rsi']) or np.isnan(row['sma']):
            return "WAIT"
        if row['rsi'] < rsi_oversold and row['close'] < row['sma']:
            return "BUY"
        if row['rsi'] > rsi_overbought and row['close'] > row['sma']:
            return "SELL"
        return "WAIT"

    rows = df.reset_index(drop=True)

    for i in range(start_idx, len(rows)):
        row = rows.iloc[i]
        cur_time = row['time']
        close = float(row['close'])
        high = float(row['high'])
        low = float(row['low'])
        sig = signal(row)

        # Manage open position
        if open_pos is not None:
            side = open_pos['side']
            # Conservative: assume SL hit before TP if both touched same bar
            if side == 'LONG':
                if low <= open_pos['sl']:
                    exit_px = open_pos['sl']
                    pnl = exit_px - open_pos['entry_price']
                    trades.append({**open_pos, "exit_price": exit_px, "exit_time": cur_time, "reason": "SL", "pnl": pnl})
                    open_pos = None
                elif high >= open_pos['tp']:
                    exit_px = open_pos['tp']
                    pnl = exit_px - open_pos['entry_price']
                    trades.append({**open_pos, "exit_price": exit_px, "exit_time": cur_time, "reason": "TP", "pnl": pnl})
                    open_pos = None
            else:  # SHORT
                if high >= open_pos['sl']:
                    exit_px = open_pos['sl']
                    pnl = open_pos['entry_price'] - exit_px
                    trades.append({**open_pos, "exit_price": exit_px, "exit_time": cur_time, "reason": "SL", "pnl": pnl})
                    open_pos = None
                elif low <= open_pos['tp']:
                    exit_px = open_pos['tp']
                    pnl = open_pos['entry_price'] - exit_px
                    trades.append({**open_pos, "exit_price": exit_px, "exit_time": cur_time, "reason": "TP", "pnl": pnl})
                    open_pos = None

            # Flip on opposite signal if still open
            if open_pos is not None:
                if (open_pos['side'] == 'LONG' and sig == 'SELL') or (open_pos['side'] == 'SHORT' and sig == 'BUY'):
                    exit_px = close
                    pnl = (exit_px - open_pos['entry_price']) if open_pos['side'] == 'LONG' else (open_pos['entry_price'] - exit_px)
                    trades.append({**open_pos, "exit_price": exit_px, "exit_time": cur_time, "reason": "Flip", "pnl": pnl})
                    open_pos = None

        # Enter new position if none open
        if open_pos is None:
            if sig == 'BUY':
                entry = close
                sl = entry - (sl_pips * point)
                tp = entry + (tp_pips * point)
                open_pos = {"side": "LONG", "entry_price": entry, "entry_time": cur_time, "sl": sl, "tp": tp}
            elif sig == 'SELL':
                entry = close
                sl = entry + (sl_pips * point)
                tp = entry - (tp_pips * point)
                open_pos = {"side": "SHORT", "entry_price": entry, "entry_time": cur_time, "sl": sl, "tp": tp}

    # Close any open position at last close
    if open_pos is not None:
        last_row = rows.iloc[-1]
        close_px = float(last_row['close'])
        pnl = (close_px - open_pos['entry_price']) if open_pos['side'] == 'LONG' else (open_pos['entry_price'] - close_px)
        trades.append({**open_pos, "exit_price": close_px, "exit_time": last_row['time'], "reason": "EOD", "pnl": pnl})
        open_pos = None

    total_pnl = float(np.nansum([t['pnl'] for t in trades]))
    wins = sum(1 for t in trades if t['pnl'] > 0)
    losses = sum(1 for t in trades if t['pnl'] <= 0)
    stats = {
        "trades": len(trades),
        "wins": wins,
        "losses": losses,
        "win_rate": (wins / len(trades)) * 100 if trades else 0.0,
        "total_pnl_points": total_pnl / point if point else total_pnl,
        "point": point,
    }
    return {"trades": trades, "stats": stats}


def main():
    parser = argparse.ArgumentParser(description="Backtest RSI+SMA strategy on MT5 data")
    parser.add_argument("--symbol", default="Volatility 100 Index", help="Primary symbol name")
    parser.add_argument("--alt-symbols", nargs="*", default=["VIX100", "Volatility100", "Vol100", "VIX 100"], help="Fallback symbols to try")
    parser.add_argument("--days", type=int, default=7, help="Number of days back from now")
    parser.add_argument("--timeframe", default="M1", choices=["M1", "M5", "M15", "M30", "H1", "H4", "D1"], help="Bar timeframe")
    parser.add_argument("--rsi-period", type=int, default=14)
    parser.add_argument("--sma-period", type=int, default=20)
    parser.add_argument("--rsi-overbought", type=float, default=70)
    parser.add_argument("--rsi-oversold", type=float, default=30)
    parser.add_argument("--sl-pips", type=float, default=50)
    parser.add_argument("--tp-pips", type=float, default=100)
    parser.add_argument("--csv", type=str, default=None, help="Optional path to save trades CSV")

    args = parser.parse_args()

    symbol = connect_and_resolve_symbol(args.symbol, args.alt_symbols)
    if not symbol:
        return

    try:
        tf = parse_timeframe(args.timeframe)
        end = datetime.utcnow()
        start = end - timedelta(days=args.days)
        df = fetch_rates(symbol, tf, start, end)
        if df is None:
            return
        res = simulate(df, symbol, args.rsi_period, args.sma_period, args.rsi_overbought, args.rsi_oversold, args.sl_pips, args.tp_pips)
        trades = res["trades"]
        stats = res["stats"]

        print("\nBacktest Summary")
        print("================")
        print(f"Symbol: {symbol}  Timeframe: {args.timeframe}  Window: {args.days}d  Bars: {len(df)}")
        print(f"Trades: {stats['trades']}  Wins: {stats['wins']}  Losses: {stats['losses']}  Win%: {stats['win_rate']:.2f}%")
        print(f"Total PnL: {stats['total_pnl_points']:.2f} points (point={stats['point']})")

        if args.csv and trades:
            tdf = pd.DataFrame(trades)
            tdf.to_csv(args.csv, index=False)
            print(f"Saved trades to {args.csv}")
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
