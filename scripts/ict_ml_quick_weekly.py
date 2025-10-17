#!/usr/bin/env python3
"""
Quick ICT + ML weekly optimizer on cached 5m intraday data (time-constrained 5–10 minutes).
- Strategies: ICT base + ML (best-known config from prior runs; persist feature order; retrain weekly)
- HTF frames: 15M, 1H, 4H
- ML thresholds: use last successful ml_threshold from prior quick run; enable adaptive weighting
- Killzones: relaxed (allow entries outside strict LON/NY)
- Filters: anomaly filter OFF; HTF bias applied as weight (not required for entry)
- Risk-Reward: stop_atr=1.0, tp_atr=3.0 for ICT
- Output: results/ict_ml_quick_best_iteration.json
- Select best iteration (Sharpe first, then PnL) within time window; stop as soon as 5–10 minutes reached
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

# Ensure eden package is importable
ROOT = Path(__file__).resolve().parents[1]
EDEN_PY = ROOT / 'worker' / 'python'
if str(EDEN_PY) not in sys.path:
    sys.path.insert(0, str(EDEN_PY))

import warnings
warnings.filterwarnings("ignore")

from eden.data.loader import DataLoader
from eden.features.feature_pipeline import build_mtf_features
from eden.strategies.ict import ICTStrategy
from eden.backtest.engine import BacktestEngine
from eden.backtest.analyzer import Analyzer

# Optional ML pipeline utilities
try:
    from eden.ml.pipeline import create_features_for_ml, get_feature_alignment
except Exception:
    create_features_for_ml = None
    get_feature_alignment = None

import pandas as pd

SYMBOL = "Volatility 100 Index"
BASE_TF = "5M"
HTF_TFS = ["15M", "1H", "4H"]

MIN_TIME_SEC = 300   # 5 minutes (soft minimum)
TIME_LIMIT_SEC = 600 # 10 minutes (hard cap)

OUT_FILE = Path("results/ict_ml_quick_best_iteration.json")
FEAT_ALIGN_PATH = Path("models/feature_alignment.json")
MODEL_PATH = Path("models/weekly_rf.joblib")


def _load_last_ml_threshold() -> float:
    # Try ict_quick_best_iteration first
    for p in [Path("results/ict_quick_best_iteration.json"), Path("results/ict_ml_quick_best_iteration.json")]:
        try:
            if p.exists():
                data = json.loads(p.read_text())
                thr = data.get("applied_thresholds", {}).get("ml_threshold")
                if isinstance(thr, (int, float)) and 0 < thr < 1:
                    return float(thr)
        except Exception:
            pass
    return 0.6


def _load_cached_5m() -> Optional[pd.DataFrame]:
    dl = DataLoader(cache_dir=Path("data/cache"))
    # Strictly avoid provider calls; read layered/local cache only
    layered = Path("data/layered") / f"{SYMBOL.upper()}_{BASE_TF.upper()}.csv"
    if layered.exists():
        try:
            return dl.load_csv(layered)
        except Exception:
            pass
    # common aliases
    for name in ["VIX100_5M.csv", "VOLATILITY100_5M.csv", "VOL100_5M.csv"]:
        p = Path("data/layered") / name
        if p.exists():
            try:
                return dl.load_csv(p)
            except Exception:
                continue
    # Try any csv cache shard without network
    try:
        cache_dir = Path("data/cache")
        candidates = list(cache_dir.glob("*.csv"))
        for c in candidates:
            try:
                df = dl.load_csv(c)
                if df is not None and not df.empty:
                    return df
            except Exception:
                continue
    except Exception:
        pass
    return None


def _compute_bias_weights(df: pd.DataFrame) -> pd.Series:
    # Lightweight HTF bias influence: favor trend direction, but do not gate
    ema200 = df.get('ema_200')
    close = df.get('close')
    if ema200 is None or close is None:
        return pd.Series(1.0, index=df.index)
    trend = (close > ema200).astype(int)
    # map: aligned 1.0, against 0.9
    return trend.map({1: 1.0, 0: 0.9}).astype(float)


def _ml_week_signals(df_week: pd.DataFrame, ml_threshold: float) -> pd.DataFrame:
    import numpy as np
    # Build aligned features
    if create_features_for_ml is None:
        return pd.DataFrame(columns=["timestamp", "side", "confidence"])  # ML not available
    try:
        # Use persisted alignment if available
        if FEAT_ALIGN_PATH.exists():
            try:
                align_cols = json.loads(FEAT_ALIGN_PATH.read_text())
            except Exception:
                align_cols = None
        else:
            align_cols = None
        if align_cols is None and get_feature_alignment is not None:
            align_cols = list(get_feature_alignment(df_week))
            # uniquify
            seen, order = set(), []
            for c in align_cols:
                if c not in seen:
                    seen.add(c); order.append(c)
            FEAT_ALIGN_PATH.parent.mkdir(parents=True, exist_ok=True)
            FEAT_ALIGN_PATH.write_text(json.dumps(order, indent=2))
            align_cols = order
        X, y = create_features_for_ml(df_week, align_cols) if align_cols else create_features_for_ml(df_week)
        if X is None or len(X) < 20:
            return pd.DataFrame(columns=["timestamp", "side", "confidence"])
        # Train light model weekly
        from sklearn.ensemble import RandomForestClassifier
        import joblib
        model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
        model.fit(X.fillna(0.0), y)
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            joblib.dump(model, MODEL_PATH)
        except Exception:
            pass
        prob = model.predict_proba(X.fillna(0.0))[:, 1]
        idx = X.index
        # Confidence based on distance from 0.5
        conf = np.abs(prob - 0.5) * 2.0
        # Directional signals
        sides = np.where(prob >= 0.5, 'buy', 'sell')
        out = pd.DataFrame({
            'timestamp': idx,
            'side': sides,
            'confidence': conf
        })
        # Apply ML threshold on confidence (not prob directly)
        out = out[out['confidence'] >= float(ml_threshold)]
        # Reindex to DataFrame columns
        out = out[['timestamp', 'side', 'confidence']]
        return out.reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=["timestamp", "side", "confidence"])


def _score_strategy(df_week: pd.DataFrame, signals: pd.DataFrame) -> Tuple[float, float]:
    if signals is None or signals.empty:
        return (0.0, 0.0)
    eng = BacktestEngine(
        starting_cash=15.0,
        commission_bps=1.0,
        slippage_bps=1.0,
        per_order_risk_fraction=0.02,
        min_trade_value=0.50,
        growth_factor=0.5,
    )
    trades = eng.run(df_week, signals, symbol='VIX100', risk_manager=None)
    an = Analyzer(trades or [], starting_cash=15.0)
    m = an.metrics()
    sharpe = float(m.get('sharpe', 0.0) or 0.0)
    pnl = float(m.get('net_pnl', 0.0) or 0.0)
    return (max(0.0, sharpe), pnl)


def main():
    t0 = time.time()
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    df5 = _load_cached_5m()
    if df5 is None or df5.empty:
        print("ERROR: No cached 5m data found for 'Volatility 100 Index'. Ensure data/layered/*_5M.csv exists.")
        sys.exit(2)

    feats = build_mtf_features(df5, BASE_TF, HTF_TFS)
    bias_w = _compute_bias_weights(feats)

    # Weekly grouping
    feats_week = feats.copy()
    feats_week['__week__'] = feats_week.index.to_period('W').astype(str)
    grouped = feats_week.groupby('__week__')

    last_ml_thr = _load_last_ml_threshold()

    best = None
    best_key = (-9999.0, -1e18)  # (sharpe, pnl)
    evaluated = 0

    for wk, dfw in grouped:
        if time.time() - t0 >= TIME_LIMIT_SEC:
            break
        df_week = dfw.drop(columns=['__week__'], errors='ignore')
        if len(df_week) < 50:
            continue
        try:
            # ICT (relaxed killzones; bias used as weight)
            ict = ICTStrategy(
                require_htf_bias=False,
                ml_fallback_on_neutral=True,
                stop_atr_multiplier=1.0,
                tp_atr_multiplier=3.0,
                min_confidence=last_ml_thr,
                killzones_enabled=False,
                killzones="london,ny",
                london_window="07-11",
                ny_window="12-17",
            )
            ict_sig = ict.on_data(df_week)
            if ict_sig is None:
                ict_sig = pd.DataFrame(columns=['timestamp','side','confidence'])

            # ML signals (weekly retrain, persisted feature order if available)
            ml_sig = _ml_week_signals(df_week, last_ml_thr)

            # Apply bias weighting to confidence (non-gating)
            if not ict_sig.empty:
                ict_conf = ict_sig.set_index('timestamp')['confidence']
                aligned = bias_w.reindex(ict_conf.index, method='ffill').fillna(1.0)
                ict_sig['confidence'] = (ict_conf * aligned).clip(0.0, 0.99).values
            if not ml_sig.empty:
                ml_conf = ml_sig.set_index('timestamp')['confidence']
                aligned = bias_w.reindex(ml_conf.index, method='ffill').fillna(1.0)
                ml_sig['confidence'] = (ml_conf * aligned).clip(0.0, 0.99).values

            # Adaptive weighting between ICT and ML
            ict_sh, ict_pnl = _score_strategy(df_week, ict_sig)
            ml_sh, ml_pnl = _score_strategy(df_week, ml_sig)
            # Composite weights
            w_ict = max(0.1, ict_sh * 0.6 + (1.0 if ict_pnl > 0 else 0.0) * 0.4)
            w_ml  = max(0.1, ml_sh  * 0.6 + (1.0 if ml_pnl  > 0 else 0.0) * 0.4)
            s = w_ict + w_ml
            w_ict /= s; w_ml /= s

            # Combine signals by scaling confidence
            import numpy as np
            parts = []
            if not ict_sig.empty:
                a = ict_sig.copy(); a['strategy'] = 'ict'; a['confidence'] = (a['confidence'] * w_ict).clip(0.0, 0.99)
                parts.append(a)
            if not ml_sig.empty:
                b = ml_sig.copy(); b['strategy'] = 'ml'; b['confidence'] = (b['confidence'] * w_ml).clip(0.0, 0.99)
                parts.append(b)
            if not parts:
                evaluated += 1
                if evaluated >= 3 and (time.time() - t0) >= MIN_TIME_SEC:
                    break
                continue
            combined = pd.concat(parts, ignore_index=True)

            # Backtest combined
            eng = BacktestEngine(
                starting_cash=15.0,
                commission_bps=1.0,
                slippage_bps=1.0,
                per_order_risk_fraction=0.02,
                min_trade_value=0.50,
                growth_factor=0.5,
            )
            trades = eng.run(df_week, combined, symbol='VIX100', risk_manager=None)
            an = Analyzer(trades or [], starting_cash=15.0)
            m = an.metrics()

            sharpe = float(m.get('sharpe', 0.0) or 0.0)
            pnl = float(m.get('net_pnl', 0.0) or 0.0)
            key = (sharpe, pnl)
            if key > best_key:
                best_key = key
                best = {
                    "week": wk,
                    "pnl": pnl,
                    "pnl_pct": float(m.get('equity_growth_pct', 0.0) or 0.0),
                    "max_drawdown_pct": float(m.get('max_drawdown_pct', 0.0) or 0.0),
                    "trades": int(m.get('trades', 0) or 0),
                    "applied_thresholds": {"ml_threshold": float(last_ml_thr)},
                    "killzone_variant": "relaxed",
                    "htf_frames": ",".join(HTF_TFS),
                    "strategy_weights": {"ict": round(w_ict, 3), "ml": round(w_ml, 3)},
                    "ml_config_used": MODEL_PATH.name if MODEL_PATH.exists() else "trained_weekly_rf",
                    "duration_sec": round(time.time() - t0, 2),
                    "symbol": SYMBOL,
                    "base_timeframe": BASE_TF,
                }
            evaluated += 1
        except Exception:
            evaluated += 1
            continue
        # Soft stop when we have some iterations and exceeded 5 minutes
        if evaluated >= 3 and (time.time() - t0) >= MIN_TIME_SEC:
            break

    if best is None:
        print("No viable weekly iteration produced within time budget.")
        OUT_FILE.write_text(json.dumps({"status":"no_results","time_sec": round(time.time()-t0,2)}, indent=2))
        sys.exit(0)

    # Console output
    print("=== ICT+ML Quick Best Weekly Iteration ===")
    print(f"Week: {best['week']}")
    print(f"PnL: ${best['pnl']:.2f} ({best['pnl_pct']:.2f}%)  |  MaxDD: {best['max_drawdown_pct']:.2f}%  |  Trades: {best['trades']}")
    print(f"Threshold: ml_threshold={best['applied_thresholds']['ml_threshold']:.2f}  |  Killzones: {best['killzone_variant']}")
    print(f"HTF: {best['htf_frames']}  |  Weights: {best['strategy_weights']}  |  ML: {best['ml_config_used']}")

    OUT_FILE.write_text(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
