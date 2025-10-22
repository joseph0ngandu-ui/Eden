import numpy as np
import pandas as pd
from eden.features.indicators import (
    compute_fair_value_gaps,
    detect_liquidity_sweeps,
    identify_order_blocks,
)


def test_indicators_on_synthetic():
    idx = pd.date_range("2020-01-01", periods=10, freq="D", tz="UTC")
    df = pd.DataFrame(
        {
            "open": np.arange(10) + 1,
            "high": np.arange(10) + 2,
            "low": np.arange(10),
            "close": np.arange(10) + 1.5,
            "volume": np.ones(10),
        },
        index=idx,
    )
    fvg = compute_fair_value_gaps(df)
    sweeps = detect_liquidity_sweeps(df)
    obs = identify_order_blocks(df)
    assert "fvg_bull" in fvg.columns
    assert "sweep_high" in sweeps.columns
    assert "ob_bull" in obs.columns
