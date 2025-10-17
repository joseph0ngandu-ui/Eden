from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple

import math

@dataclass
class StageAConfig:
    fvg_weight: float = 0.3
    ob_weight: float = 0.2
    sweep_weight: float = 0.4
    wick_weight: float = 0.1


def sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def run_stageA_row(row: Dict, cfg: StageAConfig = StageAConfig()) -> Tuple[float, Dict]:
    """Heuristic Stage-A liquidity/displacement probability.
    Uses available HTF/MF features; no heavy training to keep runtime light.
    Returns (P_liquidity, features_dict).
    """
    fvg_bull = float(row.get('HTF_FVG_BULL', 0))
    fvg_bear = float(row.get('HTF_FVG_BEAR', 0))
    ob_bull = float(row.get('HTF_OB_COUNT_BULL', 0))
    ob_bear = float(row.get('HTF_OB_COUNT_BEAR', 0))
    sweep_high = float(row.get('HTF_RECENT_SWEEP_HIGH', 0))
    sweep_low = float(row.get('HTF_RECENT_SWEEP_LOW', 0))

    # Wick metrics if present
    upper_wick = float(max(0.0, row.get('high', 0) - max(row.get('open', 0), row.get('close', 0))))
    lower_wick = float(max(0.0, min(row.get('open', 0), row.get('close', 0)) - row.get('low', 0)))

    score = 0.0
    score += cfg.fvg_weight * (fvg_bull + fvg_bear)
    score += cfg.ob_weight * (ob_bull + ob_bear)
    score += cfg.sweep_weight * (sweep_high + sweep_low)
    score += cfg.wick_weight * (upper_wick + lower_wick)

    p = sigmoid(score)
    feats = {
        'fvg': fvg_bull + fvg_bear,
        'ob': ob_bull + ob_bear,
        'sweep': sweep_high + sweep_low,
        'wick': upper_wick + lower_wick,
    }
    return p, feats