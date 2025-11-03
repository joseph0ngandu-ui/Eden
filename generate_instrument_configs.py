#!/usr/bin/env python3
"""
Generate per-instrument configuration files for Eden multi-instrument trading system
"""

import json
import os
from pathlib import Path

# Define base configuration
INSTRUMENT_POOL = [
    "VIX75", "VIX100", "VIX50", "VIX25",
    "StepIndex", "Boom1000", "Crash1000", "Boom500", "Crash500",
    "XAUUSD"
]

CONFIG_DIR = Path("configs/instruments")
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

# Load strategies template
strategies_template_path = CONFIG_DIR / "strategies_template.json"
with open(strategies_template_path) as f:
    strategies_template = json.load(f)

# Generate per-instrument configurations
for instrument in INSTRUMENT_POOL:
    config = {
        "symbol": instrument,
        "description": f"Eden multi-instrument config for {instrument}",
        "data": {
            "resolution": "1m",
            "source": "MetaTrader5",
            "fallback_source": "historical_cache"
        },
        "risk": {
            "max_daily_risk_pct": 2.0,
            "max_concurrent_trades": 2,
            "stop_loss_pct": 2.0,
            "take_profit_ratio": 2.0
        },
        "strategies": {}
    }
    
    # Add appropriate strategies based on instrument type
    if instrument == "XAUUSD":
        # Gold uses gold-specific strategies
        config["strategies"].update(strategies_template["gold_strategies"])
        config["instrument_type"] = "commodity"
    else:
        # All VIX and indices use ICT strategies
        config["strategies"].update(strategies_template["ict_strategies"])
        config["instrument_type"] = "synthetic_index"
    
    # Add ML strategy to all instruments
    config["strategies"]["ML_Adaptive"] = strategies_template["ml_strategy"]
    
    # Write configuration file
    config_path = CONFIG_DIR / f"{instrument}.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✓ Created configuration for {instrument} -> {config_path}")

# Create master configuration
master_config = {
    "instruments": INSTRUMENT_POOL,
    "total_instruments": len(INSTRUMENT_POOL),
    "description": "Eden multi-instrument trading system - Synthetic indices + Gold",
    "trading_rules": {
        "NO_FOREX": True,
        "INSTRUMENTS_ONLY": "VIX25,VIX50,VIX75,VIX100,StepIndex,Boom500,Boom1000,Crash500,Crash1000,XAUUSD"
    },
    "optimization": {
        "strategy": "optuna",
        "initial_trials": 80,
        "full_trials": 300,
        "parallel_workers": 8,
        "seed": 42
    },
    "git_tagging": {
        "tag_prefix": "eden/multi-instrument-profitable-",
        "auto_commit": True
    }
}

master_path = CONFIG_DIR / "master_config.json"
with open(master_path, 'w') as f:
    json.dump(master_config, f, indent=2)

print(f"\n✓ Created master configuration -> {master_path}")
print(f"\n✅ Generated configurations for {len(INSTRUMENT_POOL)} instruments")
print(f"   Instruments: {', '.join(INSTRUMENT_POOL)}")
print(f"   Total strategies per instrument: ~8-9 (including ML)")
