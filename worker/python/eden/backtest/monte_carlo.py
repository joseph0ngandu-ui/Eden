from __future__ import annotations
import numpy as np
from typing import List


def monte_carlo_drawdowns(trade_pnls: List[float], runs: int = 1000, seed: int = 42):
    rng = np.random.default_rng(seed)
    dds = []
    for _ in range(runs):
        seq = rng.choice(trade_pnls, size=len(trade_pnls), replace=True)
        ec = 100000 + np.cumsum(seq)
        peak = np.maximum.accumulate(ec)
        dd = (ec - peak) / peak
        dds.append(dd.min())
    return {
        "dd_mean": float(np.mean(dds)),
        "dd_5pct": float(np.quantile(dds, 0.05)),
        "dd_95pct": float(np.quantile(dds, 0.95)),
    }
