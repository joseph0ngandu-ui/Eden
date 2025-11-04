#!/usr/bin/env python3
"""
Portfolio backtest runner with --dry-run support.
Wraps portfolio_compound_vb.py to validate environment or execute a default run.
"""

import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from portfolio_compound_vb import run_portfolio_compounding

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Portfolio runner")
    parser.add_argument("--dry-run", action="store_true", help="Validate without running")
    parser.add_argument("--start", type=float, default=100.0, help="Starting equity")
    args = parser.parse_args()

    if args.dry_run:
        print("Dry run OK - portfolio runner available.")
        sys.exit(0)

    run_portfolio_compounding(args.start)

if __name__ == "__main__":
    main()
