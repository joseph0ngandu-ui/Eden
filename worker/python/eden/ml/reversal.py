from __future__ import annotations
import logging
import pandas as pd

log = logging.getLogger("eden.ml.reversal")

# Phase 1: Lightweight reversal scoring placeholder
# Adds/updates 'confidence' column based on simple reversal criteria:
# - Divergence between short/long MAs
# - Large wick detection (potential sweep)
# - Distance from recent swing extremes


def score_reversals(df_feat: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    if signals is None or signals.empty:
        return signals
    try:
        # Prepare helpers
        close = df_feat["close"]
        ma_fast = close.rolling(10).mean()
        ma_slow = close.rolling(40).mean()
        div = ma_fast - ma_slow
        (df_feat["high"] - df_feat["low"]).rolling(20).mean()
        upper = df_feat["high"].rolling(50).max()
        lower = df_feat["low"].rolling(50).min()

        conf = []
        for _, row in signals.iterrows():
            ts = row["timestamp"]
            side = row["side"]
            c = float(row.get("confidence", 0.5))
            # Base score
            score = c
            # Divergence: if going against recent MA spread, increase score (reversal)
            d = div.loc[:ts].iloc[-1] if ts in div.index else 0.0
            if side == "sell" and d > 0:
                score += 0.1
            if side == "buy" and d < 0:
                score += 0.1
            # Distance to extremes: closer to band edges increases reversal chance
            try:
                hi = float(upper.loc[:ts].iloc[-1]) if ts in upper.index else None
                lo = float(lower.loc[:ts].iloc[-1]) if ts in lower.index else None
                px = float(close.loc[ts])
                if side == "sell" and hi:
                    score += max(0.0, min(0.2, (hi - px) / max(1e-6, hi) * 2))
                if side == "buy" and lo:
                    score += max(0.0, min(0.2, (px - lo) / max(1e-6, lo) * 2))
            except Exception:
                pass
            conf.append(min(0.99, max(0.0, score)))
        signals = signals.copy()
        signals["confidence"] = conf
        return signals
    except Exception as e:
        log.warning("reversal scoring error: %s", e)
        return signals
