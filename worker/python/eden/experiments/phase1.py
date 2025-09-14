from __future__ import annotations
import logging
import os
import time
import uuid
from dataclasses import dataclass
from typing import Dict, Any, Optional

import pandas as pd

from ..data.loader import DataLoader
from ..features.feature_pipeline import build_feature_pipeline, build_mtf_features
from ..strategies.base import StrategyBase
from ..strategies.ict import ICTStrategy
from ..strategies.mean_reversion import MeanReversionStrategy
from ..strategies.momentum import MomentumStrategy
from ..strategies.price_action import PriceActionStrategy

log = logging.getLogger("eden.experiments.phase1")

@dataclass
class Phase1Config:
    symbols: list[str]
    timeframe: str
    start: str
    end: str
    min_confidence: float = 0.0


def run_dynamic_selection_preview(cfg: Phase1Config) -> Dict[str, Any]:
    """Lightweight preview run for dynamic selection and reversal gating."""
    dl = DataLoader()
    selection: Dict[str, list[str]] = {}
    for sym in cfg.symbols:
        df = dl.get_ohlcv(sym, cfg.timeframe, cfg.start, cfg.end, prefer_mt5=True)
        if df is None or df.empty:
            continue
        extras = ["1H", "4H", "1D"] if cfg.timeframe not in ("1D", "1W", "1MO") else []
        df_feat = build_mtf_features(df, cfg.timeframe, extras) if extras else build_feature_pipeline(df)
        from ..ml.selector import select_strategies_for_symbol
        try:
            chosen = select_strategies_for_symbol(sym, cfg.timeframe, df_feat)
            selection[sym] = [s.name for s in chosen]
        except Exception:
            selection[sym] = ["ict", "momentum"]
    return {"selection": selection}
