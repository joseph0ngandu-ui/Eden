#!/usr/bin/env python3
"""
Volatility Burst v1.3 Backtest Harness

Backtests the Volatility Burst strategy against all trading symbols
from January to October 2025.
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import json

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from volatility_burst_enhanced import VolatilityBurst


class VBBacktester:
    """Backtest Volatility Burst strategy."""
    
    def __init__(self, config_path: str = "config/volatility_burst.yml"):
        self.config_path = config_path
        self.vb = VolatilityBurst(config_path)
        self.results = {}
    
    def fetch_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch OHLCV data from MT5."""
        if not mt5.initialize():
            print(f"MT5 initialization failed")
            return None
        
        try:
            rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M5, start_date, end_date)
            if rates is None:
                print(f"No data for {symbol}")
                return None
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            return df.sort_values('time').reset_index(drop=True)
        
        finally:
            mt5.shutdown()
    
    def backtest_symbol(self, symbol: str, start_date: datetime, end_date: datetime, param_overrides: dict | None = None):
        """Run backtest on a single symbol."""
        print(f"\nBacktesting {symbol}...", end=" ")
        
        df = self.fetch_data(symbol, start_date, end_date)
        if df is None or len(df) < 50:
            print("FAILED (insufficient data)")
            return None
        
        # Reset strategy state
        self.vb = VolatilityBurst(self.config_path)
        if param_overrides:
            self.vb.update_params(**param_overrides)
        
        positions = []
        trades = []
        
        # Pre-generate signals once (much faster)
        df.attrs['symbol'] = symbol
        signals = self.vb.generate_signals(df)
        signals_by_idx = {t.bar_index: t for t in signals}
        
        # Process each bar
        for idx in range(len(df)):
            df_slice = df.iloc[:idx+1]
            
            # Manage open positions
            actions = self.vb.manage_position(df_slice, symbol)
            for action in actions:
                if action["action"] == "close":
                    pos = positions.pop()
                    exit_price = action.get("price", df_slice.iloc[-1]["close"])
                    profit = (exit_price - pos["entry_price"]) if pos["direction"] == "LONG" else (pos["entry_price"] - exit_price)
                    trades.append({
                        "entry_time": pos["entry_time"],
                        "exit_time": df_slice.iloc[-1]["time"],
                        "entry_price": pos["entry_price"],
                        "exit_price": exit_price,
                        "direction": pos["direction"],
                        "profit": profit,
                        "r_value": action.get("r_value", 0),
                        "confidence": pos.get("confidence", None),
                        "entry_atr": pos.get("atr", None)
                    })
            
            # Open new trade if signal at this bar and no open position
            if symbol not in self.vb.open_positions and idx in signals_by_idx:
                trade = signals_by_idx[idx]
                self.vb.on_trade_open(trade)
                positions.append({
                    "entry_time": trade.entry_time,
                    "entry_price": trade.entry_price,
                    "direction": trade.direction,
                    "confidence": trade.confidence
                })
        
        # Close any remaining open positions
        if positions:
            df_last = df.iloc[-1]
            for pos in positions:
                profit = df_last["close"] - pos["entry_price"]
                trades.append({
                    "entry_time": pos["entry_time"],
                    "exit_time": df_last["time"],
                    "entry_price": pos["entry_price"],
                    "exit_price": df_last["close"],
                    "direction": pos["direction"],
                    "profit": profit,
                    "r_value": 0,
                    "confidence": pos.get("confidence", None),
                    "entry_atr": pos.get("atr", None)
                })
        
        # Calculate statistics
        if not trades:
            print("NO TRADES")
            return None
        
        profits = [t["profit"] for t in trades]
        winning = [p for p in profits if p > 0]
        losing = [p for p in profits if p < 0]
        
        total_pnl = sum(profits)
        win_rate = len(winning) / len(trades) * 100 if trades else 0
        avg_win = np.mean(winning) if winning else 0
        avg_loss = np.mean(losing) if losing else 0
        profit_factor = sum(winning) / abs(sum(losing)) if losing else 0
        
        stats = {
            "symbol": symbol,
            "total_trades": len(trades),
            "winning_trades": len(winning),
            "losing_trades": len(losing),
            "total_pnl": total_pnl,
            "pnl_percent": (total_pnl / 100) * 100,  # Normalized to $100 capital
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "max_profit": max(profits) if profits else 0,
            "min_profit": min(profits) if profits else 0
        }
        
        status = "PROFITABLE [+]" if total_pnl > 0 else "UNPROFITABLE [-]"
        print(f"{len(trades)} trades | PnL: ${total_pnl:+.2f} ({stats['pnl_percent']:+.1f}%) | WR: {win_rate:.1f}% {status}")
        
        self.results[symbol] = {
            "stats": stats,
            "trades": trades
        }
        
        return stats
    
    def run(self, symbols: list, start_date: datetime, end_date: datetime):
        """Run backtest on multiple symbols."""
        print("="*100)
        print("VOLATILITY BURST v1.3 BACKTEST")
        print("="*100)
        print(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        conf = self.vb.config
        print(f"Config: BB(period={conf['indicators']['bollinger_bands']['period']}, std={conf['indicators']['bollinger_bands']['std_dev']}), "
              f"KC(mult={conf['indicators']['keltner_channels']['multiplier']}), ATR(period={conf['indicators']['atr']['period']}), "
              f"Conf ≥ {conf['risk']['confidence_threshold']}")
        
        all_stats = []
        for symbol in symbols:
            stats = self.backtest_symbol(symbol, start_date, end_date)
            if stats:
                all_stats.append(stats)
        
        # Summary
        print("\n" + "="*100)
        print("SUMMARY")
        print("="*100)
        
        if all_stats:
            df_summary = pd.DataFrame(all_stats)
            print(f"\n{'Symbol':<30} | {'Trades':<8} | {'PnL':<12} | {'WR%':<8} | {'PF':<6} | Status")
            print("-"*100)
            
            total_trades = 0
            total_pnl = 0
            total_wins = 0
            total_losses = 0
            
            for _, row in df_summary.iterrows():
                status = "PROFITABLE" if row["total_pnl"] > 0 else "UNPROFITABLE"
                print(f"{row['symbol']:<30} | {row['total_trades']:<8d} | ${row['total_pnl']:<11.2f} | {row['win_rate']:<7.1f}% | {row['profit_factor']:<5.2f} | {status}")
                total_trades += row["total_trades"]
                total_pnl += row["total_pnl"]
                total_wins += row["winning_trades"]
                total_losses += row["losing_trades"]
            
            print("-"*100)
            total_wr = total_wins / total_trades * 100 if total_trades > 0 else 0
            print(f"{'TOTAL':<30} | {total_trades:<8d} | ${total_pnl:<11.2f} | {total_wr:<7.1f}% | {'':5} | {'PROFITABLE' if total_pnl > 0 else 'UNPROFITABLE'}")
            
            print(f"\nTotal PnL: ${total_pnl:,.2f}")
            print(f"Return on $100k capital: {(total_pnl/100000)*100:.2f}%")
            print("="*100)
        
        return all_stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Volatility Burst v1.3 Backtest")
    parser.add_argument("--dry-run", action="store_true", help="Validate script without running full backtest")
    args = parser.parse_args()

    symbols = [
        "Volatility 75 Index",
        "Volatility 100 Index",
        "Boom 1000 Index",
        "Boom 500 Index",
        "Crash 500 Index",
        "Step Index"
    ]
    
    backtester = VBBacktester("config/volatility_burst.yml")
    if args.dry_run:
        print("Dry run OK - configuration loaded and symbols set.")
        sys.exit(0)

    start = datetime(2025, 1, 1)
    end = datetime(2025, 10, 31)
    
    results = backtester.run(symbols, start, end)
    
    # Save results
    output_file = "reports/vb_v1.3_backtest_results.json"
    Path("reports").mkdir(exist_ok=True)
    
    with open(output_file, "w") as f:
        json.dump(backtester.results, f, indent=2, default=str)
    
    print(f"\nResults saved to {output_file}")
