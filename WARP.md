# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Environment Setup

**Python Version:** 3.11+ (as specified in README.md)

**Installation:**
```bash
# Install dependencies
pip install -r requirements.txt
pip install -r worker/python/requirements.txt

# Set Python path (required for imports)
# Linux/macOS:
export PYTHONPATH="$(pwd)/worker/python:$PYTHONPATH"
# Windows PowerShell:
$env:PYTHONPATH = "$(Get-Location)\\worker\\python;$env:PYTHONPATH"
```

## Common Commands

### Install/Build
```bash
# Install core dependencies
pip install -r requirements.txt
pip install -r worker/python/requirements.txt

# Install dev/QA tools
pip install black flake8 pytest
```

### Lint and Format
```bash
# Format code (Black)
black .

# Lint code (Flake8) 
flake8 worker/python/eden --count --select=E9,F63,F7,F82 --show-source --statistics
```

### Tests
```bash
# Run all tests
$env:PYTHONPATH = "$(Get-Location)\\worker\\python"; python -m pytest worker/python/eden/tests/ -q

# Run single test file
python -m pytest worker/python/eden/tests/test_data_loader.py -q

# Run specific test
python -m pytest worker/python/eden/tests/test_backtest_engine.py::test_backtest_engine_basic_flow -q
```

### Run Applications

#### Eden CLI (Multi-Asset Trading System)
```bash
# Set PYTHONPATH first
$env:PYTHONPATH = "$(Get-Location)\\worker\\python"

# Initialize workspace
python -c "from eden.cli import main; main()" --init

# Run backtest
python -c "from eden.cli import main; main()" --run-backtest --config config.yml

# Train ML models
python -c "from eden.cli import main; main()" --train --config config.yml

# Strategy discovery
python -c "from eden.cli import main; main()" --discover

# Paper trading
python -c "from eden.cli import main; main()" --paper-trade --config config.yml
```

#### VIX 100 Trading Bot (Standalone)
```bash
# Run the VIX100 bot (requires MetaTrader 5 installed)
python eden_vix100_bot.py
```

#### Scripts (Backtesting & Analysis)
```bash
# Set PYTHONPATH first
$env:PYTHONPATH = "$(Get-Location)\\worker\\python"

# Run MVP pipeline (backtests + optimization)
python scripts/run_mvp.py --instrument "Volatility 100 Index" --strategies ict,momentum

# Run optimized configuration for micro accounts
python scripts/run_optimal.py --days-back 7

# Run comprehensive backtests
python scripts/run_backtests.py
```

#### Eden Advanced CLI (New)
```bash
# Advanced Phase 3 ML ICT trading system
.\eden.bat --phase3 --mvp --output-dir results/phase3_ml_ict/ \
  --start-date 2025-10-07 --end-date 2025-10-14 \
  --ml-enabled --ml-ict-filter --dynamic-risk-per-trade 0.02 \
  --ml-threshold 0.6 --mt5-online \
  --store-local-data results/local_data/vix100.csv \
  --use-local-data-if-available --debug --verbose \
  --safe-mode --auto-retry --retrain-if-stuck \
  --timeout 2700 --monitor-cpu --monitor-subprocesses \
  --log-file results/phase3_ml_ict/run.log

# Extensive optimization with multiple strategies
.\eden.bat --phase3 --mvp --output-dir results/phase3_ml_optim/ \
  --start-date 2025-10-07 --end-date 2025-10-14 \
  --ml-enabled --ml-ict-filter --ml-extensive-optimization \
  --ml-threshold 0.6 --grid-optimization \
  --backtest-strategies ict,momentum,mean_reversion,price_action,ml_generated \
  --dynamic-risk-per-trade 0.02 --min-trade-value 0.5 \
  --verify-strategy-functionality --mt5-online \
  --store-local-data results/local_data/vix100.csv \
  --use-local-data-if-available --debug --verbose \
  --safe-mode --auto-retry --retrain-if-stuck \
  --timeout 7200 --monitor-cpu --monitor-subprocesses \
  --log-file results/phase3_ml_optim/run.log

# Show all available options
.\eden.bat --help
```

## Architecture Overview

This repository contains two main trading systems:

### 1. Standalone VIX 100 Bot (`eden_vix100_bot.py`)
- Direct MetaTrader 5 integration for VIX 100 trading
- Self-contained with RSI/SMA strategies
- Configuration via `config.yaml`
- Runs independently of the Eden system

### 2. Eden Multi-Asset System (`worker/python/eden/`)
- Comprehensive backtesting engine with ML capabilities
- Multi-timeframe strategy framework (ICT, momentum, mean reversion, ML-generated)
- CLI orchestrates: Data loading → Feature engineering → Strategy execution → Analysis
- Supports paper trading, live trading (via MT5/CCXT), and strategy optimization
- Results saved to `results/` directory with metrics, trades, and equity curves

**Key Flow:** 
```
Data Sources (MT5/Yahoo/CSV) → Feature Pipeline → Strategy Registry → Backtest Engine → Analyzer → Results
```

The CLI (`eden.cli`) coordinates these components, while standalone scripts provide specific workflows like optimized backtesting.

## Configuration Notes

- **VIX100 Bot:** Edit `config.yaml` for trading parameters, risk settings, and MT5 connection details
- **Eden System:** Uses `worker/python/config.yml` as default, supports `--config` override
- **Sample Data:** Located in `worker/python/eden/data/sample_data/` for offline testing
- **MetaTrader 5:** Required for live data and trading (both systems support fallback to historical data)
