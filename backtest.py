#!/usr/bin/env python3
"""
Production Backtest Runner

Usage:
    python backtest.py [--start DATE] [--end DATE] [--symbols SYMBOL ...]
    
Examples:
    python backtest.py --start 2025-08-01 --end 2025-10-31
    python backtest.py --symbols "Volatility 75 Index" "Volatility 100 Index"
"""

import sys
import argparse
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from backtest_engine import BacktestEngine, print_backtest_report


DEFAULT_SYMBOLS = [
    "Volatility 75 Index",
    "Volatility 100 Index",
    "Volatility 50 Index",
    "Volatility 25 Index",
    "Step Index",
    "Boom 1000 Index",
    "Crash 1000 Index",
    "Boom 500 Index",
    "Crash 500 Index",
    "XAUUSD",
]

DEFAULT_START = datetime(2025, 8, 1)
DEFAULT_END = datetime(2025, 10, 31)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run backtest using MA(3,10) strategy on M5 timeframe",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python backtest.py
  python backtest.py --start 2025-08-01 --end 2025-10-31
  python backtest.py --symbols "Volatility 75 Index" "Volatility 100 Index"
        """
    )
    
    parser.add_argument(
        "--start",
        type=str,
        default="2025-08-01",
        help="Backtest start date (YYYY-MM-DD). Default: 2025-08-01"
    )
    
    parser.add_argument(
        "--end",
        type=str,
        default="2025-10-31",
        help="Backtest end date (YYYY-MM-DD). Default: 2025-10-31"
    )
    
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=DEFAULT_SYMBOLS,
        help="Symbols to backtest. Default: all supported symbols"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Parse dates
    try:
        start_date = datetime.strptime(args.start, "%Y-%m-%d")
        end_date = datetime.strptime(args.end, "%Y-%m-%d")
    except ValueError as e:
        print(f"Error parsing dates: {e}")
        print("Expected format: YYYY-MM-DD")
        sys.exit(1)
    
    if start_date >= end_date:
        print("Error: Start date must be before end date")
        sys.exit(1)
    
    # Initialize engine
    engine = BacktestEngine()
    
    print(f"\n{'='*100}")
    print(f"BACKTEST - MA(3,10) Strategy | M5 Timeframe")
    print(f"{'='*100}")
    print(f"\nPeriod: {start_date.date()} to {end_date.date()}")
    print(f"Symbols: {len(args.symbols)}")
    print(f"\nStrategy Configuration:")
    print(f"  Fast MA: {engine.FAST_MA_PERIOD}")
    print(f"  Slow MA: {engine.SLOW_MA_PERIOD}")
    print(f"  Hold Duration: {engine.HOLD_BARS} bars")
    print(f"\n{'='*100}\n")
    
    # Run backtest
    results = engine.backtest_multiple(args.symbols, start_date, end_date)
    
    # Print report
    print_backtest_report(results)
    
    print(f"\n{'='*100}")
    print("✓ Backtest complete")
    print(f"{'='*100}\n")


if __name__ == "__main__":
    main()
