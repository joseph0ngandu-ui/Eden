#!/usr/bin/env python3
"""
Bot Runner

Starts the TradingBot with environment-aware shadow/live mode
based on strategies gating (data/strategies.json).
"""

import os
import json
import time
from pathlib import Path

# Ensure trading modules are importable when running as a script
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trading.trading_bot import TradingBot

DATA_DIR = PROJECT_ROOT / 'data'
STRATEGIES_FILE = DATA_DIR / 'strategies.json'


def get_shadow_mode_from_gating() -> bool:
    """Return True if we should run in PAPER mode based on strategies gating."""
    try:
        if STRATEGIES_FILE.exists():
            data = json.load(open(STRATEGIES_FILE, 'r'))
            # If any active strategy is LIVE, run LIVE; else PAPER
            for s in data.values():
                if s.get('is_active') and s.get('mode', 'PAPER') == 'LIVE':
                    return False
    except Exception:
        pass
    return True  # default PAPER


def main():
    while True:
        try:
            shadow = get_shadow_mode_from_gating()
            os.environ['EDEN_SHADOW'] = '1' if shadow else '0'

            # Start bot
            # Symbols and params come from TradingBot ConfigLoader defaults or config.yaml
            bot = TradingBot(symbols=None, config_path=str(PROJECT_ROOT / 'config' / 'config.yaml'))
            bot.start(check_interval=300)
        except Exception as e:
            # Backoff and retry on failure
            print(f"CRITICAL ERROR: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(30)
        finally:
            # If bot exits (stop or error), sleep then re-evaluate gating and restart
            time.sleep(10)


if __name__ == '__main__':
    main()
