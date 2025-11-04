#!/usr/bin/env python3
"""
Run live trading bot from terminal with combined strategy (per config/strategy.yml).
Shows MT5 connection status and trading enabled status.
"""
from pathlib import Path
from datetime import datetime
import argparse
import MetaTrader5 as mt5

from src.trading_bot import TradingBot
from src.config_loader import ConfigLoader


def main():
    ap = argparse.ArgumentParser(description='Run live trading bot')
    ap.add_argument('--config', type=str, default='config/strategy.yml')
    args = ap.parse_args()

    cfg = ConfigLoader(args.config)
    symbols = cfg.get_trading_symbols()
    bot = TradingBot(symbols=symbols, config_path=args.config)
    bot.start(check_interval=300)


if __name__ == '__main__':
    main()
