from __future__ import annotations
import logging
from typing import List

from ..strategies.base import StrategyBase
from ..strategies.ict import ICTStrategy
from ..strategies.mean_reversion import MeanReversionStrategy
from ..strategies.momentum import MomentumStrategy
from ..strategies.price_action import PriceActionStrategy

log = logging.getLogger("eden.ml.selector")

# Simple placeholder selector: choose strategies based on volatility/trend proxies.
# In Phase 2 this can be replaced with a trained classifier.


def _is_trending(df) -> bool:
    try:
        close = df["close"]
        ma_fast = close.rolling(20).mean()
        ma_slow = close.rolling(50).mean()
        trend = (ma_fast - ma_slow).abs().mean()
        return bool(trend > close.median() * 0.0005)
    except Exception:
        return True


def _is_high_vol(df) -> bool:
    try:
        pass

        ret = df["close"].pct_change().dropna()
        vol = ret.rolling(50).std().iloc[-1]
        return bool(vol > ret.std())
    except Exception:
        return False


def select_strategies_for_symbol(symbol: str, timeframe: str, df) -> List[StrategyBase]:
    trending = _is_trending(df)
    high_vol = _is_high_vol(df)
    log.info(
        "Selector %s %s -> trending=%s high_vol=%s",
        symbol,
        timeframe,
        trending,
        high_vol,
    )

    out: List[StrategyBase] = []
    if trending:
        out.append(MomentumStrategy())
    else:
        out.append(MeanReversionStrategy())

    # ICT works well across regimes with FVG/sweep overlays
    out.append(ICTStrategy())

    # Add price action in low vol or if we lack diversity
    if not high_vol or len(out) < 2:
        out.append(PriceActionStrategy())

    return out
