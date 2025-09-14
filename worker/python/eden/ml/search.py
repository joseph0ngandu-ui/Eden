from __future__ import annotations
import random
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from typing import Dict, Any, List

import pandas as pd

from ..backtest.engine import BacktestEngine
from ..features.feature_pipeline import build_feature_pipeline
from ..strategies.mean_reversion import MeanReversionStrategy
from ..strategies.momentum import MomentumStrategy


@dataclass
class Candidate:
    params: Dict[str, Any]
    score: float


def _evaluate_candidate(args):
    df, strategy_name, params = args
    if strategy_name == 'mean_reversion':
        strat = MeanReversionStrategy()
    else:
        strat = MomentumStrategy()
    feat = build_feature_pipeline(df)
    sig = strat.on_data(feat)
    eng = BacktestEngine(starting_cash=100000)
    trades = eng.run(feat, sig, symbol='SEARCH', risk_manager=None)
    from ..backtest.analyzer import Analyzer
    metrics = Analyzer(trades).metrics()
    return Candidate(params=params, score=metrics.get('net_pnl', 0.0))


def random_search(df: pd.DataFrame, budget: int = 10, parallel: bool = True) -> List[Candidate]:
    jobs = []
    for _ in range(budget):
        strategy_name = random.choice(['mean_reversion', 'momentum'])
        params = {"dummy": random.random()}
        jobs.append((df, strategy_name, params))
    if parallel and (not (os.getenv('CI') == 'true')):
        with Pool(max(1, min(2, cpu_count()//2))) as p:
            results = p.map(_evaluate_candidate, jobs)
    else:
        results = list(map(_evaluate_candidate, jobs))
    return sorted(results, key=lambda c: c.score, reverse=True)