#!/usr/bin/env bash
set -euo pipefail
# Use cached sample data and run backtest for two strategies
python -m eden.cli --init
python -m eden.cli --run-backtest --config config.yml
