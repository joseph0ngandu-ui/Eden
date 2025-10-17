# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

Project overview
- Eden is an extensible algorithmic trading system. Core Python package lives in worker/python/eden. Most dev commands should be run from worker/python so imports resolve.

Common commands
- Install deps
  - cd worker/python
  - python -m pip install -r requirements.txt
- Run GUI (PySide6 dashboard)
  - cd worker/python
  - python Eden.py
- CLI workflow (backtest/train/paper/live)
  - cd worker/python
  - python -m eden.cli --init            # creates config.yml and .env.example
  - python -m eden.cli --run-backtest --config config.yml
  - python -m eden.cli --run-backtest --config config.yml --ci-short   # faster/backoff for short runs
  - python -m eden.cli --train --config config.yml
  - python -m eden.cli --paper-trade --config config.yml
  - python -m eden.cli --live-trade --confirm    # requires EDEN_LIVE=1 and (flag or EDEN_CONFIRM_LIVE=1)
- Utility scripts (run from repo root or worker/python)
  - python scripts/run_backtests.py --execution-tfs M1,M5 --htf-tfs 15M,1H,4H,1D --strategies ict,mean_reversion,momentum,price_action,ml_generated,ensemble --output-dir results
  - python scripts/run_mvp.py --execution-tfs M1,M5 --htf-tfs 15M,1H,4H,1D --output-dir results
- Tests
  - cd worker/python
  - pytest -q
  - Run one test: pytest eden/tests/test_backtest_engine.py::test_backtest_engine_basic_flow -q
- Lint/format
  - cd worker/python
  - flake8 eden
  - black --check .
  - Auto-format: black .
- Build distributables (Windows-focused)
  - cd worker/python
  - python build_setup.py
  - Outputs: dist/Eden Setup.exe and dist/Eden_Portable.zip

Key architecture (big picture)
- CLI and entrypoints
  - worker/python/Eden.py is the unified entrypoint. With no args it tries to launch the GUI; with args it delegates to the CLI in eden/cli.py.
  - eden/cli.py wires the pipeline: config -> data -> features -> strategies -> backtest -> analysis -> results. It also exposes ML discovery/prune/retune utilities for experimentation.
- Configuration and logging
  - eden/config.py defines EdenConfig (Pydantic). eden/logging_conf.py initializes logging to logs/eden.log and stdout. `--init` scaffolds config.yml and .env.example in the CWD.
- Data layer
  - eden/data/loader.py provides DataLoader.get_ohlcv with layered providers (MT5/yfinance/AlphaVantage/Stooq) and CSV caching under data/cache plus a layered store under data/layered. Symbol normalization/mapping is handled via eden/config/symbol_map.yaml.
  - scripts may also use eden/data/mtf_fetcher.py to pull multiple MT5 timeframes specifically for “Volatility 100 Index”.
- Feature engineering
  - eden/features/feature_pipeline.py builds a base indicator set (EMA/RSI/MACD/ATR/VWAP) and merges ICT concepts (fair value gaps, liquidity sweeps, order blocks). build_mtf_features() aligns higher-timeframe features to the base timeframe index for MTF context.
  - eden/features/htf_ict_bias.py computes higher-timeframe directional bias (EMA slopes, BOS, liquidity sweeps) and helpers used by scripts.
- Strategy layer (signals API)
  - Strategies subclass eden/strategies/base.py and must implement on_data(df) -> DataFrame with columns: timestamp, side (“buy”/“sell”), confidence. Optional columns used by the engine: stop_price, tp_price, atr, tag.
  - Included strategies: ICT, MeanReversion, Momentum, PriceAction, MLGenerated; an “ensemble” mode aggregates multiple.
- Backtesting and analysis
  - eden/backtest/engine.py simulates orders with slippage/commission and dynamic risk sizing (per_order_risk_fraction, min_trade_value, growth_factor). It consumes features + strategy signals and tracks equity/trades.
  - eden/backtest/analyzer.py computes metrics (e.g., Sharpe, drawdown) and can save equity curves. CLI saves metrics to results/metrics.json (or examples/results in CI/quick paths) and trades to CSV.
- Optimization and ML
  - eden/optimize/optimizer.py runs a small grid-search over parametric rule-based strategies, with JSONL caching of tried params.
  - eden/ml/pipeline.py demonstrates a lightweight RF training flow on sample data, persisting a model and writing train_metrics.json. Discovery/selector modules can be used for dynamic strategy selection where available.
- Execution interfaces
  - eden/execution provides PaperBroker by default; live trading requires MetaTrader 5 and guarded env variables (EDEN_LIVE/EDEN_CONFIRM_LIVE). CLI safely falls back to paper when live isn’t available.
- Tests
  - eden/tests contains unit and integration tests that exercise the data loader, feature pipeline, strategies, and backtest engine using bundled sample CSVs.

Notes and tips specific to this repo
- Run commands from worker/python when interacting with the eden package directly. Scripts under scripts/ add worker/python to sys.path internally, so they can be run from the repo root.
- Results are written under results/ by scripts and CLI (or examples/results for minimal/CI flows). Check metrics.json, trades.csv, and equity_curve.png.
- For live trading, MT5 must be available; otherwise the system will log a fallback to PaperBroker.
