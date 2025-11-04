# Architecture

High-level components:
- src/volatility_burst_enhanced.py: VB v1.3 strategy (BB+KC squeeze, confidence, ATR exits)
- src/ma_v1_2.py: MA v1.2 (MA(3,10) with ATR-based exits)
- scripts/: backtests, optimizers, portfolio/ultra-small runners
- config/: YAML strategy configs
- reports/: JSON/CSV outputs

Data flow:
- Fetch MT5 data → generate signals → simulate execution with TP/SL/trailing → collect trades → compute stats → save reports

Extensibility:
- Add strategies in src/, expose generate_signals(), on_trade_open(), manage_position()
- Compose pipelines in scripts/ with per-symbol optimization and compounding simulators
