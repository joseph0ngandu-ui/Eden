from __future__ import annotations
from typing import Tuple
from pathlib import Path
import pandas as pd


def detect_regime(df: pd.DataFrame) -> str:
    """Classify current regime based on ATR percentile and simple proxies.
    Expects df with atr_14 and optionally rolling volatility columns.
    """
    if df is None or df.empty:
        return 'normal'
    atr = df.get('atr_14')
    if atr is None or atr.empty:
        return 'normal'
    q = atr.rank(pct=True).iloc[-1]
    if q < 0.25:
        return 'low_vol'
    if q < 0.6:
        return 'normal'
    if q < 0.85:
        return 'high_vol'
    return 'mania'


def write_regime_timeline(df: pd.DataFrame, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if df is None or df.empty:
        out_csv.write_text("")
        return
    regime = df[['atr_14']].copy()
    # rolling percentile approx
    regime['atr_pct'] = regime['atr_14'].rank(pct=True)
    regime['regime_tag'] = regime['atr_pct'].apply(lambda x: 'low_vol' if x < 0.25 else ('normal' if x < 0.6 else ('high_vol' if x < 0.85 else 'mania')))
    regime.reset_index().to_csv(out_csv, index=False)