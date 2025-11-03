#!/usr/bin/env python3
"""
Production Live Trading Bot Runner

Usage:
    python trade.py [--symbols SYMBOL ...] [--interval SECONDS]
    
Examples:
    python trade.py
    python trade.py --symbols "Volatility 75 Index" "Volatility 100 Index"
    python trade.py --interval 300
"""

import sys
import argparse
import signal
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from trading_bot import TradingBot


DEFAULT_SYMBOLS = [
    "Volatility 75 Index",    # $1,229,078 ✓ PRIMARY DRIVER
    "Volatility 100 Index",   # $28,027 ✓
    "Boom 1000 Index",        # $17,731 ✓
    "Boom 500 Index",         # $87,321 ✓
    "Crash 500 Index",        # $36,948 ✓
    "XAUUSD",                 # $23,681 ✓
]

DEFAULT_INTERVAL = 300  # 5 minutes


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Start live trading bot using MA(3,10) strategy on M5 timeframe",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python trade.py
  python trade.py --symbols "Volatility 75 Index" "Volatility 100 Index"
  python trade.py --interval 300
        """
    )
    
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=DEFAULT_SYMBOLS,
        help="Symbols to trade. Default: all supported symbols"
    )
    
    parser.add_argument(
        "--interval",
        type=int,
        default=DEFAULT_INTERVAL,
        help="Check interval in seconds. Default: 300 (5 minutes)"
    )
    
    parser.add_argument(
        "--account",
        type=int,
        help="MT5 account ID (optional)"
    )
    
    parser.add_argument(
        "--password",
        type=str,
        help="MT5 password (optional)"
    )
    
    parser.add_argument(
        "--server",
        type=str,
        help="MT5 server name (optional)"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    print(f"\n{'='*100}")
    print(f"LIVE TRADING BOT - MA(3,10) Strategy | M5 Timeframe")
    print(f"{'='*100}")
    print(f"\nConfiguration:")
    print(f"  Entry: MA(3) crosses above MA(10)")
    print(f"  Exit: Fixed 5-bar hold duration")
    print(f"  Check Interval: {args.interval} seconds")
    print(f"  Symbols: {len(args.symbols)}")
    print(f"\nSymbols to trade:")
    for symbol in args.symbols:
        print(f"  • {symbol}")
    print(f"\n{'='*100}\n")
    
    # Initialize bot
    bot = TradingBot(
        symbols=args.symbols,
        account_id=args.account,
        password=args.password,
        server=args.server
    )
    
    # Handle signals for graceful shutdown
    def signal_handler(sig, frame):
        print("\n\n✓ Shutdown signal received, closing positions...")
        bot.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start bot
    try:
        bot.start(check_interval=args.interval)
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
