from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import math


@dataclass
class VolatilityConfig:
    short_atr_col: str = "atr_14"
    # Preference order for higher timeframe ATR columns produced by build_mtf_features
    long_atr_cols_priority: tuple[str, ...] = ("1H_atr_14", "4H_atr_14", "1D_atr_14")
    cap_min: float = 0.5
    cap_max: float = 3.0
    conservative_threshold: float = 2.2
    conservative_size_scale: float = 0.5
    conservative_sl_scale: float = 1.5


def _safe_get_numeric(row, key: str, default: float = float("nan")) -> float:
    try:
        v = row.get(key)
        if v is None:
            return default
        # coerce to float
        return float(v)
    except Exception:
        return default


def compute_volatility_factor(df_row: dict, cfg: VolatilityConfig = VolatilityConfig()) -> float:
    """Compute volatility factor as ATR_short / ATR_long (HTF), capped to [cap_min, cap_max].

    If no HTF ATR is available, returns 1.0.
    """
    atr_s = _safe_get_numeric(df_row, cfg.short_atr_col, default=float("nan"))
    atr_l = float("nan")
    for col in cfg.long_atr_cols_priority:
        v = _safe_get_numeric(df_row, col, default=float("nan"))
        if math.isfinite(v) and v > 0:
            atr_l = v
            break
    if not (math.isfinite(atr_s) and math.isfinite(atr_l) and atr_l > 0):
        return 1.0
    vf = atr_s / atr_l
    # Cap
    vf = max(cfg.cap_min, min(cfg.cap_max, vf))
    return float(vf)


def adjust_stop_and_size(base_stop_dist: float, base_size: float, volatility_factor: float, cfg: VolatilityConfig = VolatilityConfig(), max_position_limit: Optional[float] = None) -> Tuple[float, float, dict]:
    """Return (adjusted_stop_dist, adjusted_position_size, meta).

    Rules:
    - adjusted_stop = base_stop_dist * volatility_factor
    - adjusted_size = base_size / volatility_factor
    - If volatility_factor > conservative_threshold => size *= conservative_size_scale; stop *= conservative_sl_scale
    - Respect optional max_position_limit
    """
    stop = float(base_stop_dist) * float(volatility_factor)
    size = float(base_size) / max(1e-12, float(volatility_factor))
    conservative = False
    if volatility_factor > cfg.conservative_threshold:
        size *= cfg.conservative_size_scale
        stop *= cfg.conservative_sl_scale
        conservative = True
    if max_position_limit is not None:
        size = min(size, float(max_position_limit))
    return stop, size, {"conservative": conservative}