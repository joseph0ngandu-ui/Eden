#!/usr/bin/env python3
"""
Batch portfolio compounding runs for VB v1.3 per-symbol best params.
Runs for start equities: $10, $50, $100.
"""

import sys
from pathlib import Path
from datetime import datetime
import json
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from portfolio_compound_vb import run_portfolio_compounding

if __name__ == '__main__':
    for start in [10.0, 50.0, 100.0]:
        run_portfolio_compounding(start)
