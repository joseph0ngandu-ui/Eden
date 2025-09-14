#!/usr/bin/env bash
set -euo pipefail
# Paper trade example (no live connections)
python -m eden.cli --init
python -m eden.cli --paper-trade --config config.yml
