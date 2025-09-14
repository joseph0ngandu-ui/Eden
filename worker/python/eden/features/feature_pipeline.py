import pandas as pd
from .indicators import atr, rsi, macd, ema, sma, vwap, compute_fair_value_gaps, detect_liquidity_sweeps, identify_order_blocks
from eden.data.transforms import resample_ohlcv, timeframe_to_rule


def build_feature_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['ema_50'] = ema(df['close'], 50)
    df['ema_200'] = ema(df['close'], 200)
    df['sma_20'] = sma(df['close'], 20)
    df['rsi_14'] = rsi(df['close'], 14)
    macd_line, signal_line, hist = macd(df['close'])
    df['macd'] = macd_line
    df['macd_signal'] = signal_line
    df['macd_hist'] = hist
    df['atr_14'] = atr(df, 14)
    df['vwap_20'] = vwap(df, 20)

    fvg = compute_fair_value_gaps(df)
    sweeps = detect_liquidity_sweeps(df)
    obs = identify_order_blocks(df)

    out = pd.concat([df, fvg, sweeps, obs], axis=1)
    out = out.ffill().bfill().dropna(subset=["open","high","low","close","volume"]).copy()
    return out


def build_mtf_features(df: pd.DataFrame, base_tf: str, extra_tfs: list[str]) -> pd.DataFrame:
    """Build features including higher timeframe context aligned to base timeframe index."""
    base = build_feature_pipeline(df)
    frames = [base]
    for tf in extra_tfs:
        try:
            agg = resample_ohlcv(df, tf)
            feats = build_feature_pipeline(agg)
            prefix = tf.upper() + "_"
            feats = feats.add_prefix(prefix)
            aligned = feats.reindex(base.index, method='ffill')
            frames.append(aligned)
        except Exception:
            continue
    out = pd.concat(frames, axis=1)
    out = out.ffill().bfill().dropna(subset=["open","high","low","close","volume"]).copy()
    return out
    return out
