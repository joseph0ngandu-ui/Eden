# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

Project overview
- This repository currently contains a standalone MetaTrader 5-based VIX 100 trading bot and early scaffolding for a broader, multi-asset “Eden” system. The bot logic is in eden_vix100_bot.py. Additional configuration and documentation live under docs/ and worker/python/.

Common commands
- Create a virtual environment
  - Windows (PowerShell): py -3.11 -m venv .venv; .\.venv\Scripts\Activate
  - macOS/Linux: python3.11 -m venv .venv; source .venv/bin/activate
- Install dependencies
  - Core bot: pip install -r requirements.txt
  - Extended worker stack (optional): pip install -r worker/python/requirements.txt
- Lint and format
  - black .
  - flake8
- Run tests
  - All tests: pytest -q
  - Single test: pytest tests/test_module.py::TestClass::test_case
- Run the bot
  - python eden_vix100_bot.py
- Follow logs
  - Windows: Get-Content .\eden_vix100_bot.log -Wait
  - macOS/Linux: tail -f eden_vix100_bot.log

Configuration
- Primary runtime configuration is declared in config.yaml at the repo root. It includes trading parameters (symbol, risk, trading hours), strategy settings (RSI/SMA thresholds), MT5 settings, logging/monitoring, safety, and development flags. The current bot implementation sets many defaults in code and does not yet load this file; if behavior diverges from config.yaml, defer to values in eden_vix100_bot.py until config loading is implemented.

High-level architecture
- eden_vix100_bot.py
  - VIX100TradingBot encapsulates:
    - connect_mt5(): initialize the MT5 terminal and resolve the VIX 100 symbol (tries fallbacks like VIX100/Vol100)
    - get_market_data(): retrieve recent bars via mt5.copy_rates_from_pos into pandas
    - analyze_market(): compute RSI and SMA, emit BUY/SELL/WAIT with reasoning
    - place_order(): construct MT5 orders with SL/TP based on pip value
    - manage_positions(): log open positions (placeholder for more logic)
    - start()/run_trading_cycle()/stop(): main loop with 60s cadence and graceful shutdown
  - Logging goes to eden_vix100_bot.log and console.
- Configuration and scaffolding
  - config.yaml (root): comprehensive runtime settings.
  - worker/python/config.yml: default multi-asset backtest-style settings (paper broker, ensemble strategy) for the broader Eden stack.
  - worker/python/eden/config/symbol_map.yaml: broker-specific symbol aliases (e.g., EURUSD -> EURUSDm for certain brokers).
  - docs/README.md: describes the intended broader Eden pipeline (Data -> Features -> Strategies -> Backtester -> Analyzer/ML -> Execution) and CLI invocations like python -m eden.cli; those entry points are not present in this repository.

Notes and caveats
- MetaTrader5 must be installed and available; the bot logs and exits if the module or terminal are unavailable.
- TA-Lib is listed with talib-binary as a fallback in requirements.txt; this helps installation on Windows.
- docs/README.md references bootstrap.ps1 and setup.sh, which are not present in this repository; use the virtual environment + pip commands above instead.
- No tests are currently present; pytest commands are provided for when tests are added.

Conventions
- Python 3.11+ is assumed (per README). Use a virtual environment. There is no build step required to run the bot.