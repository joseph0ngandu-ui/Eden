from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
# Ensure 'eden' package root is importable when run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

MODULES = [
    'eden.ml.stageA_liquidity_detector',
    'eden.ml.stageB_continuation_predictor',
    'eden.ml.stageC_combiner',
    'eden.ml.meta_learning_controller',
    'eden.ml.regime_detector',
    'eden.ml.trainer',
    'eden.rl.ppo_controller',
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=str, required=True)
    args = p.parse_args()
    status = {}
    for m in MODULES:
        try:
            __import__(m)
            status[m] = 'success'
        except Exception as e:
            status[m] = f'error: {e}'
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(status, indent=2))


if __name__ == '__main__':
    main()