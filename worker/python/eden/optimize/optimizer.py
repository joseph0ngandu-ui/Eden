from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Iterable, Tuple, Optional

import pandas as pd

from eden.backtest.engine import BacktestEngine
from eden.backtest.analyzer import Analyzer
from eden.features.feature_pipeline import build_feature_pipeline, build_mtf_features
from eden.strategies.parametric import RuleBasedParamStrategy


@dataclass
class ParamCache:
    path: Path

    def __post_init__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("")

    def key(self, symbol: str, timeframe: str, params: Dict[str, Any]) -> str:
        # Stable key independent of dict order
        return json.dumps(
            {
                "symbol": symbol.upper(),
                "timeframe": timeframe.upper(),
                "params": params,
            },
            sort_keys=True,
        )

    def load_seen(self) -> set[str]:
        seen: set[str] = set()
        try:
            with self.path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        seen.add(
                            self.key(
                                rec.get("symbol", ""),
                                rec.get("timeframe", ""),
                                rec.get("params", {}),
                            )
                        )
                    except Exception:
                        continue
        except Exception:
            pass
        return seen

    def append(self, symbol: str, timeframe: str, params: Dict[str, Any], score: float):
        rec = {
            "symbol": symbol.upper(),
            "timeframe": timeframe.upper(),
            "params": params,
            "score": score,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")


def param_grid() -> Iterable[Dict[str, Any]]:
    # Simple bounded grid for quick search; can be extended
    buy_rsi_max_list = [25, 30, 35, 40]
    sell_rsi_min_list = [60, 65, 70, 75]
    use_ema_cross_list = [True, False]
    macd_hist_bias_list = [-0.1, 0.0, 0.1]
    for a in buy_rsi_max_list:
        for b in sell_rsi_min_list:
            for c in use_ema_cross_list:
                for d in macd_hist_bias_list:
                    yield {
                        "buy_rsi_max": a,
                        "sell_rsi_min": b,
                        "use_ema_cross": c,
                        "macd_hist_bias": d,
                    }


def _score_from_metrics(metrics: Dict[str, float]) -> float:
    return float(metrics.get("sharpe", 0.0)) * 1000.0 + float(
        metrics.get("net_pnl", 0.0)
    )


def evaluate_params_feat(
    feat: pd.DataFrame, symbol: str, params: Dict[str, Any]
) -> Tuple[float, Dict[str, float]]:
    strat = RuleBasedParamStrategy(params=params)
    sig = strat.on_data(feat)
    eng = BacktestEngine(starting_cash=100000.0)
    trades = eng.run(feat, sig, symbol=symbol.upper(), risk_manager=None)
    an = Analyzer(trades)
    metrics = an.metrics()
    score = _score_from_metrics(metrics)
    return score, metrics


def evaluate_params(
    df: pd.DataFrame, symbol: str, timeframe: str, params: Dict[str, Any]
) -> Tuple[float, Dict[str, float]]:
    # Build features (include higher timeframe context for robustness)
    extras = (
        ["1D", "1W", "1MO"]
        if timeframe.upper() in ("M1", "5M", "15M", "1H", "4H")
        else ["1W", "1MO"]
    )
    try:
        feat = build_mtf_features(df, timeframe, extras)
    except Exception:
        feat = build_feature_pipeline(df)
    return evaluate_params_feat(feat, symbol, params)


def run_grid_search(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    budget: int = 40,
    cache_path: Optional[Path] = None,
    trials_out: Optional[Path] = None,
    precomputed_feat: Optional[pd.DataFrame] = None,
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    cache = ParamCache(cache_path or Path("data/cache/opt_cache.jsonl"))
    seen = cache.load_seen()
    best_params: Dict[str, Any] = {}
    best_metrics: Dict[str, float] = {}
    best_score = float("-inf")

    if trials_out:
        trials_out.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    for params in param_grid():
        if total >= max(1, budget):
            break
        key = cache.key(symbol, timeframe, params)
        if key in seen:
            continue
        if precomputed_feat is not None:
            score, metrics = evaluate_params_feat(precomputed_feat, symbol, params)
        else:
            score, metrics = evaluate_params(df, symbol, timeframe, params)
        cache.append(symbol, timeframe, params, score)
        seen.add(key)
        total += 1
        if trials_out:
            with trials_out.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "symbol": symbol.upper(),
                            "timeframe": timeframe.upper(),
                            "params": params,
                            "score": score,
                            "metrics": metrics,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
        if score > best_score:
            best_score = score
            best_params = params
            best_metrics = metrics

    return best_params, best_metrics
