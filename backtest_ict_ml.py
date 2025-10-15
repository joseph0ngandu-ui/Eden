#!/usr/bin/env python3
"""
ICT + ML Backtester over MT5 historical data with parameter optimization.

- Detects simplified ICT "liquidity grab" events on the execution timeframe
- Computes multi-timeframe confluences (M15/M30/H1/H4 bias + exec RSI/SMA)
- Trains a simple ML classifier to produce a confidence score for each signal
- Filters trades by min confluences, min confidence, and target RR
- Uses Optuna to optimize parameters; saves best to config.yaml under strategy.ict_ml

Examples:
  python backtest_ict_ml.py --days 7 --exec-timeframe M5 --optimize --trials 20 \
      --csv results/ict_ml_trades.csv --json results/ict_ml_summary.json --save-best-config config.yaml

Notes:
- Requires MetaTrader5 installed and running (same Windows user) with history for the symbol.
- Designed to run without TA-Lib; only pandas/numpy/sklearn/optuna/pyyaml are required.
"""
import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys

# Ensure local import path
sys.path.insert(0, str(Path(__file__).parent))

try:
    import MetaTrader5 as mt5
except Exception:
    print("MetaTrader5 module not found. Install with: pip install MetaTrader5")
    raise

import numpy as np
import pandas as pd

try:
    import optuna
except Exception:
    optuna = None

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split
except Exception:
    LogisticRegression = None

try:
    import yaml
except Exception:
    yaml = None


TF_MAP = None  # populated after mt5 import


def tf_const(tf: str) -> int:
    t = tf.upper()
    mapping = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
    }
    if t not in mapping:
        raise ValueError(f"Unsupported timeframe: {tf}")
    return mapping[t]


def connect_and_resolve_symbol(primary: str, alternatives: List[str]) -> Optional[str]:
    if not mt5.initialize():
        print(f"MT5 initialization failed: {mt5.last_error()}")
        return None
    sym = primary
    info = mt5.symbol_info(sym)
    if info is None:
        for alt in alternatives:
            info = mt5.symbol_info(alt)
            if info is not None:
                sym = alt
                print(f"Using alternative symbol: {sym}")
                break
    if info is None:
        print(f"None of the symbols found: {[primary] + alternatives}")
        return None
    if not info.visible:
        mt5.symbol_select(sym, True)
    return sym


def fetch_rates(symbol: str, timeframe: str, start: datetime, end: datetime) -> Optional[pd.DataFrame]:
    tfc = tf_const(timeframe)
    rates = mt5.copy_rates_range(symbol, tfc, start, end)
    if rates is None:
        print(f"Failed to fetch rates for {timeframe}: {mt5.last_error()}")
        return None
    df = pd.DataFrame(rates)
    if df.empty:
        print(f"No data returned for {timeframe}")
        return None
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df = df.sort_values('time').reset_index(drop=True)
    return df


# --- Indicators/utilities ---

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr1 = (df['high'] - df['low']).abs()
    tr2 = (df['high'] - df['close'].shift(1)).abs()
    tr3 = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()

def zscore(series: pd.Series, window: int = 20) -> pd.Series:
    mean = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std(ddof=0)
    return (series - mean) / std.replace(0, np.nan)

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=period).mean()
    rs = gain / loss
    out = 100 - (100 / (1 + rs))
    return out


def sma(series: pd.Series, period: int = 20) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()


def swing_points(high: pd.Series, low: pd.Series, lookback: int) -> Tuple[pd.Series, pd.Series]:
    # True at bar i if it's a local swing high/low
    sh = (high.shift(1).rolling(lookback, min_periods=1).max().shift(-1) < high) & \
         (high.shift(-1).rolling(lookback, min_periods=1).max().shift(1) < high)
    sl = (low.shift(1).rolling(lookback, min_periods=1).min().shift(-1) > low) & \
         (low.shift(-1).rolling(lookback, min_periods=1).min().shift(1) > low)
    # Clean NaNs to False
    return sh.fillna(False), sl.fillna(False)


def last_swing_values(df: pd.DataFrame, lookback: int) -> Tuple[pd.Series, pd.Series]:
    sh_flag, sl_flag = swing_points(df['high'], df['low'], lookback)
    df = df.copy()
    df['swing_high_val'] = np.where(sh_flag, df['high'], np.nan)
    df['swing_low_val'] = np.where(sl_flag, df['low'], np.nan)
    prev_sh = df['swing_high_val'].ffill().shift(1)
    prev_sl = df['swing_low_val'].ffill().shift(1)
    return prev_sh, prev_sl


def liquidity_grab_flags(df: pd.DataFrame, prev_sh: pd.Series, prev_sl: pd.Series) -> Tuple[pd.Series, pd.Series]:
    # Bearish grab: sweep above prev swing high and close back below
    bear = (df['high'] > prev_sh) & (df['close'] < prev_sh)
    # Bullish grab: sweep below prev swing low and close back above
    bull = (df['low'] < prev_sl) & (df['close'] > prev_sl)
    return bear.fillna(False), bull.fillna(False)


def fvg_zones(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Three-candle FVGs.
    # Bullish FVG when high[n-2] < low[n]
    bull_upper = df['low']
    bull_lower = df['high'].shift(2)
    bull = (bull_lower.notna()) & (bull_lower < bull_upper)
    bull_df = pd.DataFrame({
        'time': df['time'],
        'bull_fvg_lower': bull_lower.where(bull),
        'bull_fvg_upper': bull_upper.where(bull)
    })
    # Bearish FVG when low[n-2] > high[n]
    bear_lower = df['low'].shift(2)
    bear_upper = df['high']
    bear = (bear_lower.notna()) & (bear_lower > bear_upper)
    bear_df = pd.DataFrame({
        'time': df['time'],
        'bear_fvg_lower': bear_lower.where(bear),
        'bear_fvg_upper': bear_upper.where(bear)
    })
    return bull_df, bear_df


def bos_choch(df: pd.DataFrame, prev_sh: pd.Series, prev_sl: pd.Series) -> Tuple[pd.Series, pd.Series]:
    bos_up = (df['close'] > prev_sh).fillna(False)
    bos_dn = (df['close'] < prev_sl).fillna(False)
    # CHoCH: change in direction vs rolling prior BOS
    prior_trend_up = bos_up.shift(1).rolling(5, min_periods=1).max().fillna(False)
    prior_trend_dn = bos_dn.shift(1).rolling(5, min_periods=1).max().fillna(False)
    choch_up = bos_up & prior_trend_dn
    choch_dn = bos_dn & prior_trend_up
    return choch_up, choch_dn


def displacement_flags(df: pd.DataFrame, atr_series: pd.Series, k: float = 1.5) -> Tuple[pd.Series, pd.Series]:
    rng = (df['high'] - df['low']).abs()
    disp = (rng > k * atr_series)
    up = disp & (df['close'] > df['open'])
    dn = disp & (df['close'] < df['open'])
    return up.fillna(False), dn.fillna(False)


def volume_spike(df: pd.DataFrame, z: float = 2.0) -> pd.Series:
    if 'tick_volume' not in df.columns:
        return pd.Series(False, index=df.index)
    zv = zscore(df['tick_volume'], 50)
    return (zv > z).fillna(False)


def wick_ratio(df: pd.DataFrame) -> pd.Series:
    body = (df['close'] - df['open']).abs()
    upper = (df['high'] - df[['close', 'open']].max(axis=1))
    lower = (df[['close', 'open']].min(axis=1) - df['low'])
    rng = (df['high'] - df['low']).replace(0, np.nan)
    wr = (upper + lower) / rng
    return wr.fillna(0)


# --- Feature engineering across timeframes ---

def merge_asof_features(exec_df: pd.DataFrame, ctx_df: pd.DataFrame, suffix: str, 
                        add_rsi: bool = True, add_sma_: bool = True, sma_period: int = 20) -> pd.DataFrame:
    cdf = ctx_df[['time', 'close']].copy()
    if add_rsi:
        cdf[f'rsi_{suffix}'] = rsi(ctx_df['close']).values
    if add_sma_:
        cdf[f'sma_{suffix}'] = sma(ctx_df['close'], sma_period).values
    # We only need time + derived features; ensure no duplicates
    cdf = cdf[['time'] + [c for c in cdf.columns if c != 'time' and c != 'close']]
    out = pd.merge_asof(exec_df.sort_values('time'), cdf.sort_values('time'), on='time', direction='backward')
    return out


@dataclass
class Params:
    min_confluences: int = 2
    rr: float = 2.5
    confidence_threshold: float = 0.8
    rsi_period: int = 14
    sma_period: int = 20
    swing_lookback: int = 5
    bias_timeframes: Tuple[str, str] = ("H1", "H4")
    # Risk/drawdown control knobs
    cooldown_bars_after_loss: int = 0
    stop_after_losses: int = 0
    pause_bars_after_streak: int = 0
    # Entry gating
    require_sweep: bool = False
    require_bias_align: bool = False


def confluence_counts(df: pd.DataFrame, params: Params) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
    # Base exec features
    rsi_e = rsi(df['close'], params.rsi_period)
    sma_e = sma(df['close'], params.sma_period)
    wr = wick_ratio(df)
    atr_e = atr(df, max(14, params.sma_period // 2))
    prev_sh, prev_sl = last_swing_values(df, params.swing_lookback)
    bear_grab, bull_grab = liquidity_grab_flags(df, prev_sh, prev_sl)
    choch_up, choch_dn = bos_choch(df, prev_sh, prev_sl)
    disp_up, disp_dn = displacement_flags(df, atr_e, k=1.5)

    # FVG zones and proximity/fill
    bull_fvg_df, bear_fvg_df = fvg_zones(df)
    df_merge = df[['time', 'open', 'high', 'low', 'close']].copy()
    df_merge = df_merge.merge(bull_fvg_df, on='time', how='left').merge(bear_fvg_df, on='time', how='left')
    bull_fvg_fill = ((df_merge['low'] <= df_merge['bull_fvg_upper']) & (df_merge['close'] > df_merge['bull_fvg_lower'])).fillna(False)
    bear_fvg_fill = ((df_merge['high'] >= df_merge['bear_fvg_lower']) & (df_merge['close'] < df_merge['bear_fvg_upper'])).fillna(False)

    # Volume spike
    vol_spike = volume_spike(df)

    # Bias from multiple HTFs (merged earlier as columns rsi_<tf>, sma_<tf>)
    def bias_cols(tf: str) -> Tuple[str, str]:
        return f'rsi_{tf}', f'sma_{tf}'

    tf1, tf2 = params.basis_tfs if hasattr(params, 'basis_tfs') else params.bias_timeframes
    rsi_tf1, sma_tf1 = bias_cols(tf1)
    rsi_tf2, sma_tf2 = bias_cols(tf2)

    # Gracefully fallback if HTF features are missing
    sma_tf1_series = df[sma_tf1] if sma_tf1 in df.columns else sma_e
    sma_tf2_series = df[sma_tf2] if sma_tf2 in df.columns else sma_e

    # Bias as close vs SMA sign replicated by last merged values
    bias_buy_tf1 = (df['close'] > sma_tf1_series)
    bias_sell_tf1 = (df['close'] < sma_tf1_series)
    bias_buy_tf2 = (df['close'] > sma_tf2_series)
    bias_sell_tf2 = (df['close'] < sma_tf2_series)

    # Confluences for BUY (ICT + PA + bias)
    conf_buy = (
        bull_grab.astype(int) +                 # liquidity sweep long
        bull_fvg_fill.astype(int) +             # FVG fill
        choch_up.astype(int) +                  # CHoCH upwards
        disp_up.astype(int) +                   # displacement up
        (wr > 0.5).astype(int) +               # wick rejection
        (rsi_e < 35).astype(int) +             # near-oversold
        (df['close'] > sma_e).astype(int) +    # above SMA
        vol_spike.astype(int) +                # reaction after volume expansion
        bias_buy_tf1.astype(int) + bias_buy_tf2.astype(int)
    )

    # Confluences for SELL
    conf_sell = (
        bear_grab.astype(int) +
        bear_fvg_fill.astype(int) +
        choch_dn.astype(int) +
        disp_dn.astype(int) +
        (wr > 0.5).astype(int) +
        (rsi_e > 65).astype(int) +
        (df['close'] < sma_e).astype(int) +
        vol_spike.astype(int) +
        bias_sell_tf1.astype(int) + bias_sell_tf2.astype(int)
    )

    feats = df.copy()
    feats['rsi_e'] = rsi_e
    feats['sma_e'] = sma_e
    feats['atr_e'] = atr_e
    feats['wr'] = wr
    feats['prev_swing_high'] = prev_sh
    feats['prev_swing_low'] = prev_sl
    feats['bear_grab'] = bear_grab
    feats['bull_grab'] = bull_grab
    feats['bull_fvg_fill'] = bull_fvg_fill
    feats['bear_fvg_fill'] = bear_fvg_fill
    feats['choch_up'] = choch_up
    feats['choch_dn'] = choch_dn
    feats['disp_up'] = disp_up
    feats['disp_dn'] = disp_dn

    return conf_buy, conf_sell, feats


def simulate_trades(df: pd.DataFrame, conf_buy: pd.Series, conf_sell: pd.Series, feats: pd.DataFrame,
                    params: Params, point: float,
                    use_ml: bool = True,
                    train_frac: float = 0.6,
                    return_dataset: bool = False,
                    force_ml: bool = False):
    # Build candidate entries from confluences
    entries = []
    # Bias alignment helpers
    sma_h1 = feats['sma_H1'] if 'sma_H1' in feats.columns else feats['sma_e']
    sma_h4 = feats['sma_H4'] if 'sma_H4' in feats.columns else feats['sma_e']
    for i in range(len(df)):
        cb = int(conf_buy.iat[i])
        cs = int(conf_sell.iat[i])
        close_i = float(df['close'].iat[i])
        # BUY gating
        if cb >= params.min_confluences:
            if params.require_sweep and not bool(feats['bull_grab'].iat[i]):
                pass
            else:
                if params.require_bias_align:
                    if not (close_i > float(sma_h1.iat[i]) and close_i > float(sma_h4.iat[i])):
                        pass
                    else:
                        entries.append((i, 'BUY'))
                else:
                    entries.append((i, 'BUY'))
        # SELL gating
        elif cs >= params.min_confluences:
            if params.require_sweep and not bool(feats['bear_grab'].iat[i]):
                pass
            else:
                if params.require_bias_align:
                    if not (close_i < float(sma_h1.iat[i]) and close_i < float(sma_h4.iat[i])):
                        pass
                    else:
                        entries.append((i, 'SELL'))
                else:
                    entries.append((i, 'SELL'))

    if not entries:
        empty_stats = {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "total_pnl_points": 0.0,
            "point": point,
            "ml_used": False,
            "confidence_threshold": params.confidence_threshold,
            "rr": params.rr,
            "min_confluences": params.min_confluences,
        }
        return {"trades": [], "stats": empty_stats}, None

    # For each entry, define SL/TP using adaptive risk with partials and BE
    def trade_outcome(i: int, side: str):
        row = df.iloc[i]
        entry = float(row['close'])
        atr_val = float(feats['atr_e'].iat[i]) if not np.isnan(feats['atr_e'].iat[i]) else float((df['high']-df['low']).rolling(14).mean().iat[i])
        if side == 'BUY':
            swing_sl = feats['prev_swing_low'].iat[i]
            base_sl = entry - 1.0 * atr_val
            sl = float(swing_sl) if not np.isnan(swing_sl) else base_sl
            risk = max(entry - sl, 0.5 * atr_val)
            tp1 = entry + 1.0 * risk
            tp2 = entry + params.rr * risk
        else:
            swing_sl = feats['prev_swing_high'].iat[i]
            base_sl = entry + 1.0 * atr_val
            sl = float(swing_sl) if not np.isnan(swing_sl) else base_sl
            risk = max(sl - entry, 0.5 * atr_val)
            tp1 = entry - 1.0 * risk
            tp2 = entry - params.rr * risk
        # walk-forward with scaled TP and BE on TP1
        be_moved = False
        for j in range(i + 1, len(df)):
            hi = float(df['high'].iat[j])
            lo = float(df['low'].iat[j])
            if side == 'BUY':
                # SL/TP order: SL first if both touched
                if lo <= sl:
                    pnl = sl - entry
                    return {"i": i, "side": side, "entry": entry, "sl": sl, "tp": tp2, "exit": sl, "reason": "SL", "pnl": pnl}
                if hi >= tp1 and not be_moved:
                    sl = entry  # move to BE
                    be_moved = True
                if hi >= tp2:
                    # scaled TP: assume 50% at tp1, 50% at tp2 if tp1 hit; else full at tp2
                    if be_moved:
                        pnl = 0.5 * (tp1 - entry) + 0.5 * (tp2 - entry)
                    else:
                        pnl = tp2 - entry
                    return {"i": i, "side": side, "entry": entry, "sl": sl, "tp": tp2, "exit": tp2, "reason": "TP2", "pnl": pnl}
            else:
                if hi >= sl:
                    pnl = entry - sl
                    return {"i": i, "side": side, "entry": entry, "sl": sl, "tp": tp2, "exit": sl, "reason": "SL", "pnl": pnl}
                if lo <= tp1 and not be_moved:
                    sl = entry
                    be_moved = True
                if lo <= tp2:
                    if be_moved:
                        pnl = 0.5 * (entry - tp1) + 0.5 * (entry - tp2)
                    else:
                        pnl = entry - tp2
                    return {"i": i, "side": side, "entry": entry, "sl": sl, "tp": tp2, "exit": tp2, "reason": "TP2", "pnl": pnl}
        # EOD close
        last_close = float(df['close'].iat[-1])
        pnl = (last_close - entry) if side == 'BUY' else (entry - last_close)
        return {"i": i, "side": side, "entry": entry, "sl": sl, "tp": tp2, "exit": last_close, "reason": "EOD", "pnl": pnl}

    # Build dataset for ML: features at entry bar, label 1 if TP before SL
    rows = []
    trades_seq = []
    for idx, side in entries:
        outcome = trade_outcome(idx, side)
        if outcome is None:
            continue
        trades_seq.append(outcome)
        feat_row = {
            "side": 1 if side == 'BUY' else -1,
            "rsi_e": float(feats['rsi_e'].iat[idx]) if not np.isnan(feats['rsi_e'].iat[idx]) else 50.0,
            "wr": float(feats['wr'].iat[idx]),
            "dist_to_sma": float(df['close'].iat[idx] - feats['sma_e'].iat[idx]) if not np.isnan(feats['sma_e'].iat[idx]) else 0.0,
            "rsi_M15": float(feats.get('rsi_M15', pd.Series([np.nan])).iat[idx]) if 'rsi_M15' in feats else 50.0,
            "rsi_M30": float(feats.get('rsi_M30', pd.Series([np.nan])).iat[idx]) if 'rsi_M30' in feats else 50.0,
            "rsi_H1": float(feats.get('rsi_H1', pd.Series([np.nan])).iat[idx]) if 'rsi_H1' in feats else 50.0,
            "rsi_H4": float(feats.get('rsi_H4', pd.Series([np.nan])).iat[idx]) if 'rsi_H4' in feats else 50.0,
            "conf_buy": int(conf_buy.iat[idx]),
            "conf_sell": int(conf_sell.iat[idx]),
            # Label a trade as 1 (win) if its realized PnL is positive, else 0 (loss)
            "label": 1 if float(outcome.get('pnl', 0.0)) > 0 else 0,
        }
        rows.append(feat_row)

    if not rows:
        empty_stats = {
            "trades": len(trades_seq),
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "total_pnl_points": 0.0,
            "point": point,
            "ml_used": False,
            "confidence_threshold": params.confidence_threshold,
            "rr": params.rr,
            "min_confluences": params.min_confluences,
        }
        return {"trades": trades_seq, "stats": empty_stats}, None

    df_ds = pd.DataFrame(rows).dropna()

    clf = None
    ds_info = {"ml": False, "threshold": params.confidence_threshold}
    if use_ml and LogisticRegression is not None and (len(df_ds) >= 30 or force_ml):
        X = df_ds.drop(columns=['label'])
        y = df_ds['label']
        if len(df_ds) >= 5:
            ts = max(0.3, min(0.4, len(df_ds) / 200))
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts, shuffle=False)
        else:
            X_train, y_train = X, y
            X_test, y_test = X.iloc[0:0], y.iloc[0:0]
        clf = Pipeline([
            ('scaler', StandardScaler(with_mean=False)),
            ('lr', LogisticRegression(max_iter=2000, class_weight='balanced'))
        ])
        try:
            clf.fit(X_train, y_train)
            ds_info["ml"] = True
        except Exception:
            clf = None

    # Final pass: filter trades by ML confidence threshold with cooldown and streak control
    filtered = []
    consec_losses = 0
    max_consec_losses = 0
    cooldown_until = -1
    for outcome in trades_seq:
        idx = outcome['i']
        if cooldown_until != -1 and idx < cooldown_until:
            continue
        side = outcome['side']
        # Compute conf for this entry
        conf = 1.0
        if clf is not None:
            feat = {
                "side": 1 if side == 'BUY' else -1,
                "rsi_e": float(feats['rsi_e'].iat[idx]) if not np.isnan(feats['rsi_e'].iat[idx]) else 50.0,
                "wr": float(feats['wr'].iat[idx]),
                "dist_to_sma": float(df['close'].iat[idx] - feats['sma_e'].iat[idx]) if not np.isnan(feats['sma_e'].iat[idx]) else 0.0,
                "rsi_M15": float(feats.get('rsi_M15', pd.Series([np.nan])).iat[idx]) if 'rsi_M15' in feats else 50.0,
                "rsi_M30": float(feats.get('rsi_M30', pd.Series([np.nan])).iat[idx]) if 'rsi_M30' in feats else 50.0,
                "rsi_H1": float(feats.get('rsi_H1', pd.Series([np.nan])).iat[idx]) if 'rsi_H1' in feats else 50.0,
                "rsi_H4": float(feats.get('rsi_H4', pd.Series([np.nan])).iat[idx]) if 'rsi_H4' in feats else 50.0,
                "conf_buy": int(conf_buy.iat[idx]),
                "conf_sell": int(conf_sell.iat[idx]),
            }
            vec = pd.DataFrame([feat])
            try:
                prob = clf.predict_proba(vec)[:, 1][0]
                conf = float(prob)
            except Exception:
                conf = 1.0
        outcome['confidence'] = conf
        if conf >= params.confidence_threshold:
            filtered.append(outcome)
            if float(outcome.get('pnl', 0.0)) > 0:
                consec_losses = 0
            else:
                consec_losses += 1
                max_consec_losses = max(max_consec_losses, consec_losses)
                # Apply cooldown after loss
                if params.cooldown_bars_after_loss > 0:
                    cooldown_until = idx + int(params.cooldown_bars_after_loss)
                # Pause after hitting loss streak threshold
                if params.stop_after_losses > 0 and consec_losses >= int(params.stop_after_losses):
                    if params.pause_bars_after_streak > 0:
                        cooldown_until = max(cooldown_until, idx + int(params.pause_bars_after_streak))
                    consec_losses = 0

    total_pnl = float(np.nansum([t['pnl'] for t in filtered]))
    wins = sum(1 for t in filtered if t['pnl'] > 0)
    losses = sum(1 for t in filtered if t['pnl'] <= 0)
    trades_ct = len(filtered)
    win_rate = (wins / trades_ct) * 100 if trades_ct else 0.0
    total_points = total_pnl / point if point else total_pnl
    # Risk-aware score: favor profit, penalize losing streaks, encourage higher win rate
    score = total_points + 20.0 * (win_rate - 50.0) - 200.0 * max(0, max_consec_losses - 2) - 0.5 * max(0, 10 - trades_ct)
    stats = {
        "trades": trades_ct,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "total_pnl_points": total_points,
        "point": point,
        "ml_used": ds_info["ml"],
        "ml_rows": int(len(df_ds)) if 'df_ds' in locals() else 0,
        "confidence_threshold": params.confidence_threshold,
        "rr": params.rr,
        "min_confluences": params.min_confluences,
        "max_consecutive_losses": int(max_consec_losses),
        "score": float(score),
    }

    if return_dataset:
        return {"trades": filtered, "stats": stats}, df_ds
    return {"trades": filtered, "stats": stats}, None


def build_exec_frame(symbol: str, exec_tf: str, ctx_tfs: List[str], start: datetime, end: datetime) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    dfs: Dict[str, pd.DataFrame] = {}
    for tf in [exec_tf] + ctx_tfs:
        d = fetch_rates(symbol, tf, start, end)
        if d is None:
            raise RuntimeError(f"No data for {tf}")
        dfs[tf] = d
    base = dfs[exec_tf].copy()
    # merge context features
    out = base.copy()
    for tf in ctx_tfs:
        cdf = dfs[tf]
        # derive features on ctx and merge
        feats = merge_asof_features(base[['time', 'open', 'high', 'low', 'close']], cdf, suffix=tf)
        # Attach only derived columns
        derived_cols = [c for c in feats.columns if c.startswith('rsi_') or c.startswith('sma_')]
        for c in derived_cols:
            out[c] = feats[c]
    return out, dfs


def optimize_params(exec_df: pd.DataFrame, params: Params, point: float, n_trials: int = 20) -> Params:
    if optuna is None:
        print("Optuna not installed; skipping optimization")
        return params

    def objective(trial: optuna.trial.Trial) -> float:
        p = Params(
            min_confluences=trial.suggest_int('min_confs', 2, 8),
            rr=trial.suggest_float('rr', 2.0, 6.0, step=0.25),
            confidence_threshold=trial.suggest_float('conf_th', 0.7, 0.99, step=0.05),
            rsi_period=trial.suggest_int('rsi_period', 10, 20),
            sma_period=trial.suggest_int('sma_period', 10, 50),
            swing_lookback=trial.suggest_int('swing_lookback', 3, 12),
            bias_timeframes=params.bias_timeframes,
            cooldown_bars_after_loss=trial.suggest_int('cooldown_loss_bars', 0, 30),
            stop_after_losses=trial.suggest_int('stop_after_losses', 0, 3),
            pause_bars_after_streak=trial.suggest_int('pause_bars_after_streak', 0, 60),
            require_sweep=trial.suggest_categorical('require_sweep', [False, True]),
            require_bias_align=trial.suggest_categorical('require_bias_align', [False, True]),
        )
        conf_buy, conf_sell, feats = confluence_counts(exec_df, p)
        res, _ = simulate_trades(exec_df, conf_buy, conf_sell, feats, p, point, use_ml=True, force_ml=True)
        stats = res['stats']
        # Prefer the built-in risk-aware score; fallback to simple profit metric
        score = float(stats.get('score', stats.get('total_pnl_points', 0.0)))
        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    bp = study.best_params
    best = Params(
        min_confluences=bp.get('min_confs', params.min_confluences),
        rr=bp.get('rr', params.rr),
        confidence_threshold=bp.get('conf_th', params.confidence_threshold),
        rsi_period=bp.get('rsi_period', params.rsi_period),
        sma_period=bp.get('sma_period', params.sma_period),
        swing_lookback=bp.get('swing_lookback', params.swing_lookback),
        bias_timeframes=params.bias_timeframes,
        cooldown_bars_after_loss=bp.get('cooldown_loss_bars', getattr(params, 'cooldown_bars_after_loss', 0)),
        stop_after_losses=bp.get('stop_after_losses', getattr(params, 'stop_after_losses', 0)),
        pause_bars_after_streak=bp.get('pause_bars_after_streak', getattr(params, 'pause_bars_after_streak', 0)),
        require_sweep=bp.get('require_sweep', getattr(params, 'require_sweep', False)),
        require_bias_align=bp.get('require_bias_align', getattr(params, 'require_bias_align', False)),
    )
    return best


def save_best_to_config(config_path: Path, params: Params, exec_tf: str, ctx_tfs: List[str]):
    if yaml is None:
        print("PyYAML not installed; skipping config update")
        return
    data = {}
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
        except Exception:
            data = {}
    # Ensure structure
    if 'strategy' not in data:
        data['strategy'] = {}
    data['strategy']['ict_ml'] = {
        'exec_timeframe': exec_tf,
        'context_timeframes': ctx_tfs,
        'min_confluences': int(params.min_confluences),
        'rr': float(params.rr),
        'confidence_threshold': float(params.confidence_threshold),
        'rsi_period': int(params.rsi_period),
        'sma_period': int(params.sma_period),
        'swing_lookback': int(params.swing_lookback),
        'cooldown_bars_after_loss': int(getattr(params, 'cooldown_bars_after_loss', 0)),
        'stop_after_losses': int(getattr(params, 'stop_after_losses', 0)),
        'pause_bars_after_streak': int(getattr(params, 'pause_bars_after_streak', 0)),
        'require_sweep': bool(getattr(params, 'require_sweep', False)),
        'require_bias_align': bool(getattr(params, 'require_bias_align', False)),
        'updated_at': datetime.utcnow().isoformat() + 'Z',
    }
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, sort_keys=False)
    print(f"Saved best params to {config_path} under strategy.ict_ml")


def main():
    parser = argparse.ArgumentParser(description="ICT + ML backtester over MT5 data")
    parser.add_argument("--symbol", default="Volatility 100 Index")
    parser.add_argument("--alt-symbols", nargs="*", default=["VIX100", "Volatility100", "Vol100", "VIX 100"])
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--exec-timeframe", default="M5", choices=["M1", "M5", "M15", "M30", "H1", "H4", "D1"])
    parser.add_argument("--optimize", action='store_true')
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--continuous", action='store_true', help="Run repeated optimization rounds until convergence")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--improve-epsilon", type=float, default=0.5)
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--json", type=str, default=None)
    parser.add_argument("--save-best-config", type=str, default=None)
    parser.add_argument("--force-ml", action='store_true')

    # Defaults inspired by user's last run
    parser.add_argument("--min-confluences", type=int, default=2)
    parser.add_argument("--rr", type=float, default=2.5)
    parser.add_argument("--confidence", type=float, default=0.8)

    args = parser.parse_args()

    symbol = connect_and_resolve_symbol(args.symbol, args.alt_symbols)
    if not symbol:
        return

    try:
        end = datetime.utcnow()
        start = end - timedelta(days=args.days)

        exec_tf = args.exec_timeframe
        # Use all major TFs except the execution TF for context, per ICT multi-timeframe analysis
        ctx_tfs = [tf for tf in ["M1", "M5", "M15", "M30", "H1", "H4"] if tf != exec_tf]

        exec_df, all_dfs = build_exec_frame(symbol, exec_tf, ctx_tfs, start, end)
        sinfo = mt5.symbol_info(symbol)
        if sinfo is None:
            print("symbol_info is None")
            return
        point = sinfo.point

        base_params = Params(
            min_confluences=args.min_confluences,
            rr=args.rr,
            confidence_threshold=args.confidence,
            bias_timeframes=("H1", "H4"),
        )

        # Attach confluence features and simulate
        conf_buy, conf_sell, feats = confluence_counts(exec_df, base_params)

        # Optional optimization
        best_params = base_params
        if args.optimize:
            if optuna is None:
                print("Optuna not available; skipping optimization")
            else:
                if args.continuous:
                    best_score = -1e9
                    for r in range(args.rounds):
                        cand = optimize_params(exec_df, best_params, point, args.trials)
                        conf_buy_tmp, conf_sell_tmp, feats_tmp = confluence_counts(exec_df, cand)
                        res_tmp, _ = simulate_trades(exec_df, conf_buy_tmp, conf_sell_tmp, feats_tmp, cand, point, use_ml=True, force_ml=True)
                        score_tmp = float(res_tmp['stats'].get('score', res_tmp['stats'].get('total_pnl_points', 0.0)))
                        print(f"Round {r+1}: score={score_tmp:.2f} cand={cand}")
                        if score_tmp > best_score + args.improve_epsilon:
                            best_score = score_tmp
                            best_params = cand
                            conf_buy, conf_sell, feats = conf_buy_tmp, conf_sell_tmp, feats_tmp
                        else:
                            print("No significant improvement; stopping continuous loop.")
                            break
                else:
                    best_params = optimize_params(exec_df, base_params, point, args.trials)
                    print("Best params:", best_params)
                    # Recompute features with optimized params
                    conf_buy, conf_sell, feats = confluence_counts(exec_df, best_params)

        res, ds = simulate_trades(exec_df, conf_buy, conf_sell, feats, best_params, point, use_ml=True, return_dataset=True, force_ml=args.force_ml)
        stats = res['stats']
        print("\nICT + ML Backtest Summary")
        print("=========================")
        print(f"Symbol: {symbol}  Exec TF: {exec_tf}  Window: {args.days}d  Bars: {len(exec_df)}")
        print(f"Trades: {stats['trades']}  Wins: {stats['wins']}  Losses: {stats['losses']}  Win%: {stats['win_rate']:.2f}%  Max L-streak: {stats.get('max_consecutive_losses', 0)}")
        print(f"Total PnL: {stats['total_pnl_points']:.2f} points (point={stats['point']})  Score: {stats.get('score', 0.0):.2f}")
        print(f"Params: min_confs={best_params.min_confluences} rr={best_params.rr} conf>={best_params.confidence_threshold}")
        print(f"ML used: {stats['ml_used']}  conf_th={stats['confidence_threshold']}")

        # Save artifacts
        out_trades = args.csv
        out_json = args.json
        if out_trades:
            Path(out_trades).parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(res['trades']).to_csv(out_trades, index=False)
            print(f"Saved trades to {out_trades}")
        if out_json:
            Path(out_json).parent.mkdir(parents=True, exist_ok=True)
            import json
            with open(out_json, 'w', encoding='utf-8') as f:
                json.dump({"stats": stats, "params": best_params.__dict__}, f, indent=2)
            print(f"Saved summary to {out_json}")

        if args.save_best_config:
            save_best_to_config(Path(args.save_best_config), best_params, exec_tf, ctx_tfs)
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
