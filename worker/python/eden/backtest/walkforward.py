from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import pandas as pd


@dataclass
class WalkForwardSplit:
    train_start: str
    train_end: str
    test_start: str
    test_end: str


def make_walkforward_splits(start: str, end: str, window: int = 252, step: int = 63) -> List[WalkForwardSplit]:
    # Simple trading-days approximations on daily index
    idx = pd.date_range(start, end, freq='D', tz='UTC')
    splits: List[WalkForwardSplit] = []
    i = 0
    while i + window*2 < len(idx):
        tr_start = idx[i]
        tr_end = idx[i + window]
        te_start = tr_end
        te_end = idx[i + window*2]
        splits.append(WalkForwardSplit(
            train_start=str(tr_start.date()),
            train_end=str(tr_end.date()),
            test_start=str(te_start.date()),
            test_end=str(te_end.date()),
        ))
        i += step
    return splits


def run_walkforward(df: pd.DataFrame, build_signals_fn, engine_factory, start: str, end: str) -> List[dict]:
    splits = make_walkforward_splits(start, end)
    results = []
    for s in splits:
        tr = df.loc[s.train_start:s.train_end]
        te = df.loc[s.test_start:s.test_end]
        if tr.empty or te.empty:
            continue
        # Assume build_signals_fn takes df and returns signals for test segment using training info from tr
        signals = build_signals_fn(tr, te)
        engine = engine_factory()
        trades = engine.run(te, signals, symbol='WF', risk_manager=None)
        from .analyzer import Analyzer
        metrics = Analyzer(trades).metrics()
        results.append({"split": s.__dict__, "metrics": metrics})
    return results
