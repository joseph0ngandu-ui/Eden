"""
HTF-ICT Bias Module
Implements higher timeframe bias calculation with ICT liquidity concepts
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def calculate_ema_slope(series: pd.Series, period: int, lookback: int = 5) -> pd.Series:
    """Calculate EMA and its slope over lookback periods"""
    ema = series.ewm(span=period, adjust=False).mean()
    slope = ema.diff(lookback) / lookback
    return slope


def detect_swing_points(df: pd.DataFrame, lookback: int = 10) -> Tuple[pd.Series, pd.Series]:
    """
    Detect swing highs and lows
    Returns: (swing_highs, swing_lows) as boolean Series
    """
    high = df['high']
    low = df['low']
    
    swing_high = pd.Series(False, index=df.index)
    swing_low = pd.Series(False, index=df.index)
    
    for i in range(lookback, len(df) - lookback):
        # Swing high: highest point in window
        if high.iloc[i] == high.iloc[i-lookback:i+lookback+1].max():
            swing_high.iloc[i] = True
        
        # Swing low: lowest point in window
        if low.iloc[i] == low.iloc[i-lookback:i+lookback+1].min():
            swing_low.iloc[i] = True
    
    return swing_high, swing_low


def detect_break_of_structure(df: pd.DataFrame, swing_lookback: int = 20) -> pd.Series:
    """
    Detect Break of Structure (BOS)
    Returns: Series with values {1: bullish BOS, -1: bearish BOS, 0: no BOS}
    """
    swing_highs, swing_lows = detect_swing_points(df, lookback=5)
    
    bos = pd.Series(0, index=df.index, dtype=int)
    
    # Track recent swing points
    recent_high = df['high'].rolling(swing_lookback).max()
    recent_low = df['low'].rolling(swing_lookback).min()
    
    # Bullish BOS: price breaks above recent swing high
    bullish_break = (df['close'] > recent_high.shift(1)) & (df['close'].shift(1) <= recent_high.shift(2))
    
    # Bearish BOS: price breaks below recent swing low
    bearish_break = (df['close'] < recent_low.shift(1)) & (df['close'].shift(1) >= recent_low.shift(2))
    
    bos[bullish_break] = 1
    bos[bearish_break] = -1
    
    return bos


def detect_fair_value_gaps(df: pd.DataFrame, min_gap_pips: float = 5.0) -> pd.DataFrame:
    """
    Detect Fair Value Gaps (FVG)
    Returns DataFrame with fvg_bull, fvg_bear columns
    """
    result = pd.DataFrame(index=df.index)
    result['fvg_bull'] = 0
    result['fvg_bear'] = 0
    result['fvg_bull_size'] = 0.0
    result['fvg_bear_size'] = 0.0
    
    # Bullish FVG: gap between candle[i-2].low and candle[i].high, with candle[i-1] not filling it
    for i in range(2, len(df)):
        gap_up = df['low'].iloc[i] - df['high'].iloc[i-2]
        gap_down = df['low'].iloc[i-2] - df['high'].iloc[i]
        
        # Bullish FVG
        if gap_up > min_gap_pips:
            # Check if middle candle doesn't fill the gap
            if df['low'].iloc[i-1] > df['high'].iloc[i-2]:
                result['fvg_bull'].iloc[i] = 1
                result['fvg_bull_size'].iloc[i] = gap_up
        
        # Bearish FVG
        if gap_down > min_gap_pips:
            if df['high'].iloc[i-1] < df['low'].iloc[i-2]:
                result['fvg_bear'].iloc[i] = 1
                result['fvg_bear_size'].iloc[i] = gap_down
    
    return result


def detect_order_blocks(df: pd.DataFrame, volume_threshold: float = 1.5) -> pd.DataFrame:
    """
    Detect Order Blocks (OB) - 3-candle engulfing patterns with volume spike
    Returns DataFrame with ob_bull, ob_bear columns
    """
    result = pd.DataFrame(index=df.index)
    result['ob_bull'] = 0
    result['ob_bear'] = 0
    result['ob_bull_price'] = np.nan
    result['ob_bear_price'] = np.nan
    
    # Calculate average volume for threshold
    if 'volume' in df.columns:
        avg_volume = df['volume'].rolling(20).mean()
    else:
        # If no volume, use True Range as proxy
        avg_volume = pd.Series(1.0, index=df.index)
        volume_threshold = 1.0
    
    for i in range(3, len(df)):
        current_vol = df['volume'].iloc[i] if 'volume' in df.columns else 1.0
        
        # Bullish OB: strong bullish candle with volume spike
        body_size = df['close'].iloc[i] - df['open'].iloc[i]
        if (body_size > 0 and 
            df['close'].iloc[i] > df['high'].iloc[i-1] and
            current_vol > avg_volume.iloc[i] * volume_threshold):
            result['ob_bull'].iloc[i] = 1
            result['ob_bull_price'].iloc[i] = df['low'].iloc[i]
        
        # Bearish OB: strong bearish candle with volume spike
        body_size = df['open'].iloc[i] - df['close'].iloc[i]
        if (body_size > 0 and 
            df['close'].iloc[i] < df['low'].iloc[i-1] and
            current_vol > avg_volume.iloc[i] * volume_threshold):
            result['ob_bear'].iloc[i] = 1
            result['ob_bear_price'].iloc[i] = df['high'].iloc[i]
    
    return result


def detect_liquidity_sweeps(df: pd.DataFrame, atr_multiplier: float = 1.5) -> pd.DataFrame:
    """
    Detect liquidity sweeps - long wicks that reverse quickly
    Returns DataFrame with sweep_high, sweep_low columns
    """
    result = pd.DataFrame(index=df.index)
    result['sweep_high'] = 0
    result['sweep_low'] = 0
    result['sweep_high_size'] = 0.0
    result['sweep_low_size'] = 0.0
    
    # Calculate ATR
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift(1))
    low_close = abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(14).mean()
    
    for i in range(1, len(df)):
        upper_wick = df['high'].iloc[i] - max(df['open'].iloc[i], df['close'].iloc[i])
        lower_wick = min(df['open'].iloc[i], df['close'].iloc[i]) - df['low'].iloc[i]
        
        # High liquidity sweep
        if upper_wick > atr.iloc[i] * atr_multiplier:
            # Check if price closed back inside prior body
            prior_high = max(df['open'].iloc[i-1], df['close'].iloc[i-1])
            if df['close'].iloc[i] < prior_high:
                result['sweep_high'].iloc[i] = 1
                result['sweep_high_size'].iloc[i] = upper_wick
        
        # Low liquidity sweep
        if lower_wick > atr.iloc[i] * atr_multiplier:
            prior_low = min(df['open'].iloc[i-1], df['close'].iloc[i-1])
            if df['close'].iloc[i] > prior_low:
                result['sweep_low'].iloc[i] = 1
                result['sweep_low_size'].iloc[i] = lower_wick
    
    return result


def calculate_htf_bias(
    df_15m: pd.DataFrame,
    df_1h: pd.DataFrame,
    df_4h: pd.DataFrame,
    df_1d: Optional[pd.DataFrame] = None,
    ema_weight: float = 0.4,
    bos_weight: float = 0.3,
    liquidity_weight: float = 0.3
) -> pd.DataFrame:
    """
    Calculate HTF bias combining trend, structure, and liquidity
    Returns DataFrame with HTF_BIAS column: {-1: bearish, 0: neutral, 1: bullish}
    """
    # Calculate components for each timeframe
    htf_scores = []
    
    for df_htf, name in [(df_1h, '1H'), (df_4h, '4H')]:
        if df_htf is None or df_htf.empty:
            continue
        
        # Trend component (EMA slopes)
        ema50_slope = calculate_ema_slope(df_htf['close'], 50, lookback=3)
        ema200_slope = calculate_ema_slope(df_htf['close'], 200, lookback=5)
        trend_score = np.sign(ema50_slope + ema200_slope * 0.5)
        
        # Structure component (BOS)
        bos = detect_break_of_structure(df_htf)
        bos_score = bos.rolling(5).sum() / 5  # Average recent BOS
        
        # Liquidity component
        sweeps = detect_liquidity_sweeps(df_htf)
        liquidity_score = (sweeps['sweep_low'] - sweeps['sweep_high']).rolling(5).sum() / 5
        
        # Weighted combination
        htf_score = (
            trend_score * ema_weight +
            bos_score * bos_weight +
            liquidity_score * liquidity_weight
        )
        
        htf_scores.append(htf_score)
    
    # Average across timeframes
    if htf_scores:
        combined_score = sum(htf_scores) / len(htf_scores)
    else:
        combined_score = pd.Series(0, index=df_15m.index if df_15m is not None else df_1h.index)
    
    # Convert to discrete bias
    result = pd.DataFrame(index=combined_score.index)
    result['HTF_BIAS'] = 0
    result.loc[combined_score > 0.2, 'HTF_BIAS'] = 1
    result.loc[combined_score < -0.2, 'HTF_BIAS'] = -1
    result['HTF_BIAS_STRENGTH'] = abs(combined_score)
    
    return result


def compute_micro_features(df: pd.DataFrame, atr_multiplier: float = 1.2) -> pd.DataFrame:
    """
    Compute micro-timeframe features (M1/M5)
    Returns DataFrame with micro FVG, OB, and sweep features
    """
    result = pd.DataFrame(index=df.index)
    
    # Micro FVGs
    fvg = detect_fair_value_gaps(df, min_gap_pips=2.0)
    result = pd.concat([result, fvg], axis=1)
    
    # Micro Order Blocks
    ob = detect_order_blocks(df, volume_threshold=1.3)
    result = pd.concat([result, ob], axis=1)
    
    # Micro Liquidity Sweeps
    sweeps = detect_liquidity_sweeps(df, atr_multiplier=atr_multiplier)
    result = pd.concat([result, sweeps], axis=1)
    
    # Micro imbalance (consecutive candles in same direction)
    result['micro_imbalance_bull'] = (
        (df['close'] > df['open']).astype(int).rolling(3).sum() >= 2
    ).astype(int)
    
    result['micro_imbalance_bear'] = (
        (df['close'] < df['open']).astype(int).rolling(3).sum() >= 2
    ).astype(int)
    
    return result


def detect_liquidity_features(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Comprehensive liquidity feature detection
    Returns dict of liquidity-related features
    """
    features = {}
    
    # FVG features
    fvg = detect_fair_value_gaps(df)
    features['fvg_bull'] = fvg['fvg_bull']
    features['fvg_bear'] = fvg['fvg_bear']
    features['fvg_bull_size'] = fvg['fvg_bull_size']
    features['fvg_bear_size'] = fvg['fvg_bear_size']
    
    # Order Block features
    ob = detect_order_blocks(df)
    features['ob_bull'] = ob['ob_bull']
    features['ob_bear'] = ob['ob_bear']
    
    # Liquidity Sweep features
    sweeps = detect_liquidity_sweeps(df)
    features['sweep_high'] = sweeps['sweep_high']
    features['sweep_low'] = sweeps['sweep_low']
    
    # Structure features
    features['bos'] = detect_break_of_structure(df)
    
    return features


def build_htf_context(df_1h: pd.DataFrame, df_4h: pd.DataFrame, lookback: int = 10) -> pd.DataFrame:
    """Build additional HTF context features used for strict gating.
    Returns a DataFrame indexed like df_1h with columns:
      - HTF_FVG_BULL, HTF_FVG_BEAR (0/1)
      - HTF_OB_COUNT_BULL, HTF_OB_COUNT_BEAR (ints over lookback)
      - HTF_RECENT_SWEEP_HIGH, HTF_RECENT_SWEEP_LOW (0/1 over last lookback)
    """
    idx = df_1h.index if df_1h is not None and not df_1h.empty else df_4h.index
    out = pd.DataFrame(index=idx)
    try:
        fvg1 = detect_fair_value_gaps(df_1h)
        ob1 = detect_order_blocks(df_1h)
        sw1 = detect_liquidity_sweeps(df_1h)
    except Exception:
        fvg1 = pd.DataFrame(index=idx); ob1 = pd.DataFrame(index=idx); sw1 = pd.DataFrame(index=idx)
    try:
        fvg4 = detect_fair_value_gaps(df_4h)
        ob4 = detect_order_blocks(df_4h)
        sw4 = detect_liquidity_sweeps(df_4h)
    except Exception:
        fvg4 = pd.DataFrame(index=idx); ob4 = pd.DataFrame(index=idx); sw4 = pd.DataFrame(index=idx)

    bull_fvg = (fvg1.get('fvg_bull', 0).fillna(0) + fvg4.get('fvg_bull', 0).fillna(0))
    bear_fvg = (fvg1.get('fvg_bear', 0).fillna(0) + fvg4.get('fvg_bear', 0).fillna(0))
    out['HTF_FVG_BULL'] = (bull_fvg > 0).astype(int)
    out['HTF_FVG_BEAR'] = (bear_fvg > 0).astype(int)

    ob_bull = (ob1.get('ob_bull', 0).fillna(0) + ob4.get('ob_bull', 0).fillna(0)).rolling(lookback).sum()
    ob_bear = (ob1.get('ob_bear', 0).fillna(0) + ob4.get('ob_bear', 0).fillna(0)).rolling(lookback).sum()
    out['HTF_OB_COUNT_BULL'] = ob_bull.fillna(0).astype(int)
    out['HTF_OB_COUNT_BEAR'] = ob_bear.fillna(0).astype(int)

    sw_high = (sw1.get('sweep_high', 0).fillna(0) + sw4.get('sweep_high', 0).fillna(0)).rolling(lookback).max()
    sw_low = (sw1.get('sweep_low', 0).fillna(0) + sw4.get('sweep_low', 0).fillna(0)).rolling(lookback).max()
    out['HTF_RECENT_SWEEP_HIGH'] = (sw_high > 0).astype(int)
    out['HTF_RECENT_SWEEP_LOW'] = (sw_low > 0).astype(int)
    return out


def align_htf_to_execution_tf(
    df_exec: pd.DataFrame,
    htf_bias: pd.DataFrame,
    method: str = 'ffill'
) -> pd.DataFrame:
    """
    Align HTF bias features to execution timeframe index
    """
    # Reindex HTF features to match execution TF
    aligned = htf_bias.reindex(df_exec.index, method=method)
    
    # Fill any remaining NaNs
    aligned = aligned.fillna(0)
    
    return aligned
