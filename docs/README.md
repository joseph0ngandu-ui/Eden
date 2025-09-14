# Eden Trading System (Eden Bot)

Eden does not revolve around ICT; it incorporates ICT plus other profitable strategies, and uses ML to identify, learn and prune strategies. Eden is an extensible, multi-asset algorithmic trading platform with paper trading by default and optional live trading via guarded environment variables.

Quickstart
- Windows: powershell -ExecutionPolicy Bypass -File .\\bootstrap.ps1
- macOS/Linux: ./setup.sh

Core commands
- python -m eden.cli --init
- python -m eden.cli --run-backtest --config config.yml
- python -m eden.cli --train --config config.yml
- python -m eden.cli --discover  # ML-based strategy discovery (saves to models/strategies_db and updates registry)
- python -m eden.cli --prune     # prune underperforming strategies
- python -m eden.cli --retune <strategy_id>  # retune a discovered strategy
- python -m eden.cli --paper-trade --config config.yml
- python -m eden.cli --live-trade --confirm

Safety
- Live trading requires EDEN_LIVE=1 and explicit confirmation via --confirm or EDEN_CONFIRM_LIVE=1.
- Missing brokers or packages fall back to paper trading with informative logs.

Architecture
Data -> Features -> Strategy Discovery (ML + rule-based) -> Backtest -> Execution (Paper/Live) -> Registry -> Prune

OS support
- Windows, macOS, Linux supported. See docs/developer_guide.md and docs/architecture.md.

Windows notes
- Install Python 3.11 (64-bit) from python.org.
- Ensure pip installs binary wheels: if pandas build fails, install the "Microsoft C++ Build Tools" or use a Conda environment.
- Run: powershell -ExecutionPolicy Bypass -File .\bootstrap.ps1

CI
- GitHub Actions runs lint + tests + a short integration backtest using sample_data.

Architecture diagram (ASCII)

+------------------+     +----------------+     +----------------+     +----------------+     +------------------+     +-----------+
| Data (Yahoo/CSV) | --> |   Features     | --> |  Strategies    | --> |   Backtester   | --> |   Analyzer/ML    | --> | Execution |
+------------------+     +----------------+     +----------------+     +----------------+     +------------------+     +-----------+
                                             \                                                        |
                                              \-> Strategy Search (Optuna/Evolutionary) --------------/

