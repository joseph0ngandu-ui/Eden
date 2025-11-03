# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Prerequisites

**Python Version:** 3.11+ required

**Supported Trading Symbols:** VIX75, VIX100, VIX50, VIX25, StepIndex, Boom1000, Crash1000, Boom500, Crash500, XAUUSD
- Note: VIX100, StepIndex, and Boom1000 show negative performance in recent optimizations; focus on VIX75, Crash1000, XAUUSD

**Installation:**
```bash
# Install core dependencies
python -m pip install -r requirements.txt

# Install Python worker dependencies  
python -m pip install -r worker/python/requirements.txt

# Set Python path for imports (Windows PowerShell)
$env:PYTHONPATH = "$(Get-Location)\worker\python;$env:PYTHONPATH"
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
# Set PYTHONPATH first (required for all tests)
$env:PYTHONPATH = "$(Get-Location)\\worker\\python"

# Run all tests
python -m pytest worker/python/eden/tests/ -q

# Run with verbose output
python -m pytest worker/python/eden/tests/ -v

# Run single test file
python -m pytest worker/python/eden/tests/test_data_loader.py -q

# Run specific test
python -m pytest worker/python/eden/tests/test_backtest_engine.py::test_backtest_engine_basic_flow -q

# Run tests with coverage
python -m pytest worker/python/eden/tests/ --cov=eden --cov-report=html

# Run only failed tests
python -m pytest --lf
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

# Quick backtest with UI
$env:PYTHONPATH = "$(Get-Location)\\worker\\python"; python -c "from eden.ui_components import main; main()"
```

## Architecture Overview

### Two Main Trading Systems

**1. Standalone VIX 100 Bot** (`eden_vix100_bot.py`)
- Direct MetaTrader 5 integration for VIX 100 trading
- Self-contained with RSI/SMA strategies
- Configuration via `config.yaml`
- Runs independently

**2. Eden Multi-Asset System** (`worker/python/eden/`)
- Comprehensive backtesting engine with ML capabilities
- Multi-timeframe strategy framework (ICT, momentum, mean reversion, ML-generated)
- CLI orchestrates: Data → Features → Strategies → Backtest → Analysis
- Supports paper trading, live trading (MT5/CCXT), optimization
- Results saved to `results/` with JSON metrics, CSV trades, equity curve PNGs

### Core Data Flow
```
Data Sources (MT5/Yahoo/CSV) 
  → DataLoader (yfinance, MTFDataFetcher) 
  → Feature Pipeline (build_feature_pipeline, build_mtf_features)
  → Strategy Registry (ICTStrategy, MomentumStrategy, etc.)
  → BacktestEngine.run()
  → Analyzer → Results (JSON, CSV, PNG)
```

### Key Modules

**Data Layer** (`worker/python/eden/data/`)
- `loader.py`: Multi-provider data fetching (yfinance, MT5, dukascopy, alpha_vantage, stooq) with caching
- `mtf_fetcher.py`: Multi-timeframe data fetcher for VIX100 via MT5
- `transforms.py`: Timeframe normalization, OHLCV resampling

**Strategy Layer** (`worker/python/eden/strategies/`)
- `base.py`: StrategyBase interface
- `ict.py`: ICT strategy with multi-timeframe bias
- `momentum.py`: Momentum strategy
- `mean_reversion.py`: Mean reversion strategy
- `price_action.py`: Price action strategy
- `ml_generated.py`: ML-generated strategies

**Backtesting** (`worker/python/eden/backtest/`)
- `engine.py`: Core engine with dynamic risk sizing (equity-based position sizing)
- `analyzer.py`: Performance metrics (Sharpe, drawdown, win rate, trades)
- `monte_carlo.py`: Bootstrap simulation
- `walkforward.py`: Walk-forward optimization

**Risk Management** (`worker/python/eden/risk/`)
- `risk_manager.py`: Controls, position limits, daily loss limits
- `position_sizing.py`: Dynamic sizing based on equity and stop distance

**ML Components** (`worker/python/eden/ml/`)
- `discovery.py`: Strategy discovery via genetic algorithm
- `selector.py`: Automated strategy selection
- `lstm_model.py`: LSTM models for prediction
- `ppo_agent.py`: Reinforcement learning agent
- `pipeline.py`: ML training pipeline

**CLI & Config**
- `cli.py`: Command-line interface for all operations
- `config.py`: Pydantic config model with YAML loading
- `config_manager.py`: Advanced config management with profiles

## Configuration System

**Profile-based Configuration:**
- `config/default.yaml` → `config/profiles/current.yaml` → `config/profiles/rapid-optimized-2025-10-22.yaml`
- Switch profiles by editing `config/profiles/current.yaml`

**Default Settings** (from `config.yml`):
```yaml
symbols: [XAUUSD, EURUSD, US30, NAS100, GBPUSD]
timeframe: 1D
start: 2018-01-01
end: 2023-12-31
starting_cash: 100000
commission_bps: 1.0
slippage_bps: 1.0
strategy: ensemble
```

**Environment Variables:**
- `PYTHONPATH`: Required to include `worker/python` directory
- `EDEN_LOG_LEVEL`: Override logging level
- `EDEN_LIVE=1`: Enable live trading mode
- `EDEN_CONFIRM_LIVE=1`: Auto-confirm live trading without prompt
- `MT5_LOGIN`, `MT5_PASSWORD`, `MT5_SERVER`: MetaTrader 5 credentials

## Data Sources & Caching

**Data Providers** (priority order):
1. MetaTrader 5 (preferred when available)
2. Yahoo Finance (yfinance) - most reliable for backtesting
3. Dukascopy (FX data)
4. Alpha Vantage (FX pairs with API key)
5. Stooq (daily indices)

**Caching Strategy:**
- Raw cache: `data/cache/<symbol>_<timeframe>_<start>_<end>.csv` (by date range)
- Layered cache: `data/layered/<SYMBOL>_<TIMEFRAME>.csv` (cumulative across runs)
- Sample data fallback: `worker/python/eden/data/sample_data/`

## Performance Optimization Notes

**Optimal Configuration** (from OPTIMIZATION_SUMMARY.md):
- Primary: ML Generated M1 (70% allocation) - Expected +15%/month, Sharpe 0.92
- Secondary: Momentum M5 (30% allocation) - Expected +8%/month, Sharpe 2.08
- Starting Balance: $15 (micro account optimized)
- Risk per Trade: 2% of equity with $0.50 minimum
- Max Drawdown Alert: 25%, Emergency Stop: 50%

**Dynamic Risk Sizing:**
- Position size = (equity * risk_per_order_fraction) / stop_distance * confidence * growth_mult
- Parameters in BacktestEngine: `per_order_risk_fraction=0.02`, `min_trade_value=0.50`, `growth_factor=0.5`

## Quick Reference

**Key Documentation Files:**
- `QUICKSTART.md`: Optimized strategy configuration (+776% average return)
- `OPTIMIZATION_SUMMARY.md`: Detailed performance analysis and optimal configurations
- `MULTI_INSTRUMENT_SYSTEM.md`: Multi-asset trading system overview
- `SETUP_GUIDE.md`: Detailed installation and setup instructions

**Important Notes:**
- Always verify strategies on demo accounts before live trading
- The optimal Bollinger Bands strategy (Period=18, StdDev=1.5, RSI=30) achieved +8,175% improvement
- Focus on high-performing symbols: VIX75, Crash1000, XAUUSD, Crash500, Boom500
