# Eden Trading Bot

A modular, auditable trading research system focused on compounding small accounts into large balances using high R:R strategies, adaptive risk, and portfolio diversification.

Core
- Strategies: Volatility Burst v1.3 (VB), Moving Average v1.2 (MA)
- Pipelines: UltraSmall Mode (V75-only), Portfolio compounding (per-symbol optimization), Optimizers
- Markets: Volatility 75/100, Boom 500/1000, Crash 500, Step Index, XAUUSD

Quickstart
- Python 3.12+, MT5 installed and logged in

Setup
- pip install -r requirements.txt (if present)
- Configure: config/volatility_burst.yml, config/ma_v1_2.yml

Commands
- python scripts/backtest_volatility_burst.py --dry-run
- python scripts/backtest_portfolio_runner.py --dry-run
- python scripts/ultra_small_mode.py

Docs
- docs/ARCHITECTURE.md
- docs/STRATEGIES.md
- docs/RISK_MANAGEMENT.md
- docs/BACKTESTING.md
- docs/CONFIGURATION.md

Status
- Tag: v1.3-progress (dea06be)
- Reports in reports/

Disclaimer
Trading involves significant risk. Past performance does not guarantee future results.
