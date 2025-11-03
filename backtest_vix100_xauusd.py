#!/usr/bin/env python3
"""
Eden VIX 100 & XAUUSD Backtest
Real MT5 data with multi-timeframe optimization
Focus: VIX 100 (primary), XAUUSD (secondary) - NO FOREX TRADING
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import json
import logging

sys.path.insert(0, str(Path(__file__).parent / "worker" / "python"))

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Eden symbols - Synthetic Indices ONLY (NO FOREX except XAUUSD)
SYMBOLS = {
    "VIX 25": "VIX25",
    "VIX 50": "VIX50",
    "VIX 75": "VIX75",
    "VIX 100": "VIX100",
    "Boom 1000": "BOOM1000",
    "Boom 500": "BOOM500",
    "Crash 1000": "CRASH1000",
    "Crash 500": "CRASH500",
    "Step Index": "STEPINDEX",
    "XAUUSD": "GOLD",
}

TIMEFRAMES = ["M1", "M5", "15M", "1H", "4H"]


def fetch_mt5_data(symbol: str, timeframe: str, days_back: int = 14) -> Optional[pd.DataFrame]:
    """Fetch real data from MetaTrader 5"""
    try:
        import MetaTrader5 as mt5
        
        if not mt5.initialize():
            logger.error(f"MT5 initialization failed: {mt5.last_error()}")
            return None
        
        # Symbol mapping
        symbol_map = {
            "Volatility 100 Index": "Volatility 100 Index",
            "XAUUSD": "XAUUSD",
        }
        
        mt5_symbol = symbol_map.get(symbol, symbol)
        
        # Timeframe mapping
        tf_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "15M": mt5.TIMEFRAME_M15,
            "1H": mt5.TIMEFRAME_H1,
            "4H": mt5.TIMEFRAME_H4,
            "1D": mt5.TIMEFRAME_D1,
        }
        
        tf_const = tf_map.get(timeframe)
        if not tf_const:
            logger.warning(f"Unknown timeframe: {timeframe}")
            return None
        
        # Fetch data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        logger.info(f"Fetching {symbol} {timeframe} from MT5: {start_date.date()} to {end_date.date()}")
        
        rates = mt5.copy_rates_range(mt5_symbol, tf_const, start_date, end_date)
        
        if rates is None or len(rates) == 0:
            logger.warning(f"No MT5 data for {symbol} {timeframe}: {mt5.last_error()}")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(rates)
        df["datetime"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.set_index("datetime")
        
        # Normalize columns
        df = df.rename(columns={"tick_volume": "volume"})
        df = df[["open", "high", "low", "close", "volume"]].copy()
        
        logger.info(f"✓ Fetched {len(df)} bars for {symbol} {timeframe}")
        return df
        
    except ImportError:
        logger.error("MetaTrader5 module not available - install: pip install MetaTrader5")
        return None
    except Exception as e:
        logger.error(f"MT5 fetch failed: {e}")
        return None
    finally:
        try:
            import MetaTrader5 as mt5
            mt5.shutdown()
        except:
            pass


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators"""
    df = df.copy()
    
    # RSI
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))
    
    # SMA
    df["sma_20"] = df["close"].rolling(20).mean()
    df["sma_50"] = df["close"].rolling(50).mean()
    
    # ATR
    high_low = df["high"] - df["low"]
    high_close = abs(df["high"] - df["close"].shift())
    low_close = abs(df["low"] - df["close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    
    # MACD
    ema12 = df["close"].ewm(span=12).mean()
    ema26 = df["close"].ewm(span=26).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    
    return df


def generate_signals(df: pd.DataFrame, rsi_buy: float = 30, rsi_sell: float = 70) -> pd.DataFrame:
    """Generate trading signals - simplified for real data"""
    df = df.copy()
    df["signal"] = 0
    df["confidence"] = 0.0
    
    # Buy: RSI oversold (< 35) + MACD bullish
    buy = (
        (df["rsi"] < rsi_buy) &
        (df["macd"] > df["macd_signal"])
    )
    df.loc[buy, "signal"] = 1
    df.loc[buy, "confidence"] = 0.65
    
    # Sell: RSI overbought (> 65) + MACD bearish
    sell = (
        (df["rsi"] > rsi_sell) &
        (df["macd"] < df["macd_signal"])
    )
    df.loc[sell, "signal"] = -1
    df.loc[sell, "confidence"] = 0.65
    
    return df


def run_backtest(symbol: str, df: pd.DataFrame, starting_cash: float = 100000) -> Dict:
    """Run backtest"""
    results = {
        "symbol": symbol,
        "starting_cash": starting_cash,
        "ending_cash": starting_cash,
        "total_trades": 0,
        "winning_trades": 0,
        "net_pnl": 0.0,
        "net_pnl_pct": 0.0,
        "max_drawdown": 0.0,
        "sharpe_ratio": 0.0,
        "win_rate": 0.0,
        "trades": []
    }
    
    equity = starting_cash
    position = None
    equity_curve = [equity]
    
    for ts, row in df.iterrows():
        if pd.isna(row.get("signal")):
            continue
        
        signal = row["signal"]
        price = row["close"]
        atr = row.get("atr", price * 0.02)
        
        if atr <= 0:
            continue
        
        # Risk 2% per trade
        risk = equity * 0.02
        qty = risk / max(atr, 0.0001)
        
        # Process signals
        if signal == 1 and position is None:
            position = ("long", qty, price, ts)
        
        elif signal == -1 and position is None:
            position = ("short", qty, price, ts)
        
        elif position and signal != 0:
            side, q, entry_px, entry_ts = position
            exit_px = price
            
            # Calculate PnL
            if side == "long":
                pnl = (exit_px - entry_px) * q
            else:
                pnl = (entry_px - exit_px) * q
            
            # Apply 0.5% commission each side
            pnl -= q * entry_px * 0.005 + q * exit_px * 0.005
            
            equity += pnl
            equity_curve.append(equity)
            
            results["total_trades"] += 1
            if pnl > 0:
                results["winning_trades"] += 1
            
            results["trades"].append({
                "entry_time": entry_ts.isoformat(),
                "exit_time": ts.isoformat(),
                "side": side,
                "entry_price": float(entry_px),
                "exit_price": float(exit_px),
                "pnl": float(pnl)
            })
            
            position = None
            if signal == 1:
                position = ("long", qty, price, ts)
            elif signal == -1:
                position = ("short", qty, price, ts)
    
    # Metrics
    results["ending_cash"] = equity
    results["net_pnl"] = equity - starting_cash
    results["net_pnl_pct"] = (results["net_pnl"] / starting_cash) * 100
    
    if results["total_trades"] > 0:
        results["win_rate"] = results["winning_trades"] / results["total_trades"]
    
    # Max drawdown
    peak = equity_curve[0]
    max_dd = 0
    for e in equity_curve:
        if e < peak:
            dd = (peak - e) / peak
            max_dd = max(max_dd, dd)
    results["max_drawdown"] = max_dd
    
    # Sharpe
    if len(equity_curve) > 1:
        returns = pd.Series(equity_curve).pct_change().dropna()
        if len(returns) > 0 and returns.std() > 0:
            results["sharpe_ratio"] = (returns.mean() / returns.std()) * (252 ** 0.5)
    
    return results


def main():
    logger.info(f"\n{'#'*70}")
    logger.info(f"# EDEN VIX 100 & XAUUSD BACKTEST - REAL MT5 DATA")
    logger.info(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"{'#'*70}\n")
    
    results_dir = Path("results/vix100_xauusd_backtest")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    for symbol in SYMBOLS.keys():
        logger.info(f"\n{'='*70}")
        logger.info(f"SYMBOL: {symbol} ({SYMBOLS[symbol]})")
        logger.info(f"{'='*70}")
        
        symbol_results = {}
        
        # Test multiple timeframes
        for timeframe in ["1H", "4H"]:
            logger.info(f"\n--- Timeframe: {timeframe} ---")
            
            # Fetch real MT5 data
            df = fetch_mt5_data(symbol, timeframe, days_back=14)
            
            if df is None or len(df) < 50:
                logger.warning(f"Insufficient data for {symbol} {timeframe}")
                continue
            
            # Calculate indicators
            df = calculate_indicators(df)
            
            # Generate signals
            df = generate_signals(df, rsi_buy=30, rsi_sell=70)
            signal_count = (df["signal"] != 0).sum()
            logger.info(f"Data points: {len(df)} | Signals: {signal_count}")
            
            # Run backtest
            result = run_backtest(symbol, df)
            
            logger.info(
                f"PnL: {result['net_pnl']:+.2f} ({result['net_pnl_pct']:+.2f}%) | "
                f"Trades: {result['total_trades']} | "
                f"Win: {result['win_rate']:.1%} | "
                f"DD: {result['max_drawdown']:.1%} | "
                f"Sharpe: {result['sharpe_ratio']:.2f}"
            )
            
            symbol_results[timeframe] = result
            
            # Save per-timeframe results
            tf_file = results_dir / f"{symbol}_{timeframe}_results.json"
            with open(tf_file, "w") as f:
                json.dump(result, f, indent=2, default=str)
        
        all_results[symbol] = symbol_results
    
    # Summary
    logger.info(f"\n{'#'*70}")
    logger.info(f"# SUMMARY")
    logger.info(f"{'#'*70}\n")
    
    for symbol, timeframes in all_results.items():
        logger.info(f"\n{symbol}:")
        for tf, result in timeframes.items():
            logger.info(
                f"  {tf:4} | PnL: {result['net_pnl']:+8.2f} ({result['net_pnl_pct']:+6.2f}%) | "
                f"Trades: {result['total_trades']:3d} | "
                f"Win: {result['win_rate']:5.1%} | "
                f"Sharpe: {result['sharpe_ratio']:6.2f}"
            )
    
    # Save summary
    summary_file = results_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "symbols": list(SYMBOLS.keys()),
            "results": all_results
        }, f, indent=2, default=str)
    
    logger.info(f"\n✓ Results saved to {results_dir}")


if __name__ == "__main__":
    main()
