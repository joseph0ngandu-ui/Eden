from __future__ import annotations
import argparse
import json
from pathlib import Path


def train_stage(stage: str, save_dir: Path, train_days: int, eval_days: int, seed: int) -> dict:
    """Placeholder trainer: writes metadata and returns mock metrics."""
    save_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "stage": stage,
        "train_days": train_days,
        "eval_days": eval_days,
        "seed": seed,
        "auc": 0.65 if stage == 'A' else 0.6,
        "acc": 0.58 if stage == 'B' else 0.55,
        "calibration": 0.9 if stage == 'C' else 1.0,
    }
    (save_dir / 'metrics.json').write_text(json.dumps(meta, indent=2))
    return meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', type=str, default='all')
    p.add_argument('--train-window-days', type=int, default=21)
    p.add_argument('--eval-window-days', type=int, default=7)
    p.add_argument('--save-dir', type=str, default='results/phase3/models')
    p.add_argument('--seed', type=int, default=20251017)
    args = p.parse_args()

    root = Path(args.save_dir)
    root.mkdir(parents=True, exist_ok=True)
    out = {}
    stages = ['A','B','C'] if args.stage in ('all','ALL') else [args.stage]
    for s in stages:
        out[s] = train_stage(s, root / s, args.train_window_days, args.eval_window_days, args.seed)
    (root / 'train_summary.json').write_text(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()