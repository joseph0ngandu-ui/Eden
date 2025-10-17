#!/usr/bin/env python3
"""
ICT + ML weekly optimizer in 5-minute bursts using cached 5m VIX100 data.
- ICT base + ML (weekly retrain, persisted feature order)
- HTF: 15M, 1H, 4H
- Adaptive weighting across bursts (start {'ict':0.2,'ml':0.8})
- Relaxed killzones (can toggle LON06-11 / NY13-17 per burst)
- RR: stop_atr=1.0, tp_atr in {3.0, 3.5}
- Anomaly filter OFF; HTF bias applied as weight (non-gating)
- ml_threshold ∈ {0.30,0.35,0.40,0.45}
- Cap trades per burst to avoid overtrading
- Outputs:
  * results/ict_ml_best_iteration.json (best across all bursts)
  * results/ict_ml_cycle_summaries/burst_XXX_summary.json (each burst)
Constraints: cached data only; persist feature order; avoid high resource usage.
"""
from __future__ import annotations
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, List

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
import numpy as np

SYMBOL = "Volatility 100 Index"
BASE_TF = "5M"
HTF_TFS = ["15M", "1H", "4H"]

# Files
OUT_BEST = Path("results/ict_ml_best_iteration.json")
OUT_DIR_SUM = Path("results/ict_ml_cycle_summaries")
FEAT_ALIGN_PATH = Path("models/feature_alignment.json")
MODEL_PATH = Path("models/weekly_rf.joblib")

ML_THRESHOLDS = [0.30, 0.35, 0.40, 0.45]
TP_ATR_OPTIONS = [3.0, 3.5]
KZ_VARIANTS = [
    {"enabled": False, "label": "relaxed"},
    {"enabled": True,  "label": "LON06-11_NY13-17", "lon":"06-11", "ny":"13-17"},
]

@dataclass
class Iteration:
    week: str
    pnl: float
    pnl_pct: float
    max_dd_pct: float
    trades: int
    ml_threshold: float
    tp_atr: float
    killzone: str
    htf_frames: str
    weights: Dict[str,float]
    ml_model: str


def _load_cached_5m() -> Optional[pd.DataFrame]:
    dl = DataLoader(cache_dir=Path("data/cache"))
    # Strictly avoid provider calls; read layered/local cache only
    layered = Path("data/layered") / f"{SYMBOL.upper()}_{BASE_TF.upper()}.csv"
    if layered.exists():
        try:
            return dl.load_csv(layered)
        except Exception:
            pass
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
    ema200 = df.get('ema_200')
    close = df.get('close')
    if ema200 is None or close is None:
        return pd.Series(1.0, index=df.index)
    trend = (close > ema200).astype(int)
    return trend.map({1: 1.0, 0: 0.9}).astype(float)


def _ensure_alignment(df: pd.DataFrame) -> List[str]:
    if FEAT_ALIGN_PATH.exists():
        try:
            cols = json.loads(FEAT_ALIGN_PATH.read_text())
            if isinstance(cols, list) and cols:
                return cols
        except Exception:
            pass
    if get_feature_alignment is None:
        return []
    cols = list(get_feature_alignment(df))
    seen, order = set(), []
    for c in cols:
        if c not in seen:
            seen.add(c); order.append(c)
    FEAT_ALIGN_PATH.parent.mkdir(parents=True, exist_ok=True)
    FEAT_ALIGN_PATH.write_text(json.dumps(order, indent=2))
    return order


def _ml_week_signals(df_week: pd.DataFrame, ml_threshold: float, align_cols: List[str]) -> pd.DataFrame:
    if create_features_for_ml is None:
        return pd.DataFrame(columns=["timestamp","side","confidence"])
    try:
        X, y = create_features_for_ml(df_week, align_cols) if align_cols else create_features_for_ml(df_week)
        if X is None or len(X) < 20:
            return pd.DataFrame(columns=["timestamp","side","confidence"])
        from sklearn.ensemble import RandomForestClassifier
        import joblib
        model = RandomForestClassifier(n_estimators=120, max_depth=10, random_state=42, n_jobs=-1)
        model.fit(X.fillna(0.0), y)
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            joblib.dump(model, MODEL_PATH)
        except Exception:
            pass
        prob = model.predict_proba(X.fillna(0.0))[:, 1]
        conf = np.abs(prob - 0.5) * 2.0
        sides = np.where(prob >= 0.5, 'buy', 'sell')
        out = pd.DataFrame({'timestamp': X.index, 'side': sides, 'confidence': conf})
        out = out[out['confidence'] >= float(ml_threshold)]
        return out.reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=["timestamp","side","confidence"])


def _score(df_week: pd.DataFrame, signals: pd.DataFrame, max_trades: int) -> Tuple[float,float,Dict]:
    if signals is None or signals.empty:
        return (0.0, 0.0, {"trades":0})
    # cap trades by highest confidence
    sig = signals.sort_values('confidence', ascending=False).head(max_trades).copy()
    eng = BacktestEngine(
        starting_cash=15.0,
        commission_bps=1.0,
        slippage_bps=1.0,
        per_order_risk_fraction=0.02,
        min_trade_value=0.50,
        growth_factor=0.5,
    )
    trades = eng.run(df_week, sig, symbol='VIX100', risk_manager=None)
    an = Analyzer(trades or [], starting_cash=15.0)
    m = an.metrics()
    return (float(m.get('sharpe',0.0) or 0.0), float(m.get('net_pnl',0.0) or 0.0), m)


def _run_burst(burst_idx: int, feats: pd.DataFrame, bias_w: pd.Series, init_weights: Dict[str,float],
               max_seconds: int, max_trades: int) -> Tuple[Dict, Iteration, Dict[str,float]]:
    t0 = time.time()
    # weekly groups
    wk_df = feats.copy(); wk_df['__week__'] = wk_df.index.to_period('W').astype(str)
    grouped = wk_df.groupby('__week__')

    best_key = (-9999.0, -1e18)  # (sharpe, pnl)
    best_iter: Optional[Iteration] = None

    # per-burst adjustable params
    kz = KZ_VARIANTS[burst_idx % len(KZ_VARIANTS)]
    align_cols = _ensure_alignment(feats)

    # cycle summary
    summary = {
        "burst": burst_idx,
        "kz": kz["label"],
        "start_time": time.time(),
        "evaluations": [],
    }

    w_ict = float(init_weights.get('ict', 0.2)); w_ml = float(init_weights.get('ml', 0.8))
    # normalize
    s = max(1e-9, w_ict + w_ml); w_ict /= s; w_ml /= s

    for wk, dfw in grouped:
        if time.time() - t0 >= max_seconds:
            break
        df_week = dfw.drop(columns=['__week__'], errors='ignore')
        if len(df_week) < 50:
            continue
        # try a small grid per week until time up
        for thr in ML_THRESHOLDS:
            for tp_atr in TP_ATR_OPTIONS:
                if time.time() - t0 >= max_seconds:
                    break
                try:
                    # ICT
                    ict = ICTStrategy(
                        require_htf_bias=False,
                        ml_fallback_on_neutral=True,
                        stop_atr_multiplier=1.0,
                        tp_atr_multiplier=float(tp_atr),
                        min_confidence=float(thr),
                        killzones_enabled=bool(kz.get('enabled', False)),
                        killzones="london,ny",
                        london_window=kz.get('lon','06-11'),
                        ny_window=kz.get('ny','13-17'),
                    )
                    ict_sig = ict.on_data(df_week)
                    if ict_sig is None:
                        ict_sig = pd.DataFrame(columns=['timestamp','side','confidence'])
                    # ML
                    ml_sig = _ml_week_signals(df_week, float(thr), align_cols)
                    # bias weighting
                    if not ict_sig.empty:
                        c = ict_sig.set_index('timestamp')['confidence']
                        a = bias_w.reindex(c.index, method='ffill').fillna(1.0)
                        ict_sig['confidence'] = (c * a).clip(0.0,0.99).values
                    if not ml_sig.empty:
                        c = ml_sig.set_index('timestamp')['confidence']
                        a = bias_w.reindex(c.index, method='ffill').fillna(1.0)
                        ml_sig['confidence'] = (c * a).clip(0.0,0.99).values
                    # adaptive pre-weight score
                    ict_sh, ict_pnl, _ = _score(df_week, ict_sig, max_trades)
                    ml_sh,  ml_pnl,  _ = _score(df_week, ml_sig,  max_trades)
                    # update internal weights within burst slightly toward better performer
                    base_ict, base_ml = w_ict, w_ml
                    if (ict_pnl > ml_pnl):
                        w_ict = min(0.9, w_ict + 0.05); w_ml = 1.0 - w_ict
                    elif (ml_pnl > ict_pnl):
                        w_ml = min(0.9, w_ml + 0.05); w_ict = 1.0 - w_ml
                    # combine
                    parts = []
                    if not ict_sig.empty:
                        a = ict_sig.copy(); a['strategy'] = 'ict'; a['confidence'] = (a['confidence'] * w_ict).clip(0,0.99)
                        parts.append(a)
                    if not ml_sig.empty:
                        b = ml_sig.copy(); b['strategy'] = 'ml'; b['confidence'] = (b['confidence'] * w_ml).clip(0,0.99)
                        parts.append(b)
                    if not parts:
                        continue
                    combined = pd.concat(parts, ignore_index=True)
                    sh, pnl, metrics = _score(df_week, combined, max_trades)
                    key = (sh, pnl)
                    evaluation = {
                        "week": wk,
                        "thr": thr,
                        "tp_atr": tp_atr,
                        "weights_before": {"ict": round(base_ict,3), "ml": round(base_ml,3)},
                        "weights_after": {"ict": round(w_ict,3), "ml": round(w_ml,3)},
                        "pnl": pnl,
                        "sharpe": sh,
                        "trades": int(metrics.get('trades',0) or 0),
                        "dd": float(metrics.get('max_drawdown_pct',0.0) or 0.0)
                    }
                    summary["evaluations"].append(evaluation)
                    if key > best_key:
                        best_key = key
                        best_iter = Iteration(
                            week=wk,
                            pnl=pnl,
                            pnl_pct=float(metrics.get('equity_growth_pct',0.0) or 0.0),
                            max_dd_pct=float(metrics.get('max_drawdown_pct',0.0) or 0.0),
                            trades=int(metrics.get('trades',0) or 0),
                            ml_threshold=float(thr),
                            tp_atr=float(tp_atr),
                            killzone=kz["label"],
                            htf_frames=",".join(HTF_TFS),
                            weights={"ict": round(w_ict,3), "ml": round(w_ml,3)},
                            ml_model=MODEL_PATH.name if MODEL_PATH.exists() else "weekly_rf.joblib",
                        )
                except Exception:
                    continue
        # small early stop for the week if many evals
        if len(summary["evaluations"]) >= 12 and (time.time() - t0) > max_seconds * 0.5:
            break

    # fallback if none
    if best_iter is None:
        best_iter = Iteration(week="none", pnl=0.0, pnl_pct=0.0, max_dd_pct=0.0, trades=0,
                              ml_threshold=0.35, tp_atr=3.0, killzone=kz["label"],
                              htf_frames=",".join(HTF_TFS), weights={"ict":w_ict, "ml":w_ml},
                              ml_model=MODEL_PATH.name if MODEL_PATH.exists() else "weekly_rf.joblib")

    # Decide next-burst global weight adj rules based on best pnl
    next_weights = {"ict": w_ict, "ml": w_ml}
    pnl_ratio = best_iter.pnl / 15.0 if 15.0 > 0 else 0.0
    if pnl_ratio > 0.5:
        # favor ML a bit
        next_weights['ml'] = min(0.95, next_weights['ml'] + 0.05)
        next_weights['ict'] = 1.0 - next_weights['ml']
    elif best_iter.pnl < 0:
        # reduce ML, relax thresholds next time implicitly by exploring lower thr in set
        next_weights['ml'] = max(0.2, next_weights['ml'] - 0.1)
        next_weights['ict'] = 1.0 - next_weights['ml']

    # finalize summary
    summary["end_time"] = time.time()
    summary["best"] = {
        "week": best_iter.week,
        "pnl": best_iter.pnl,
        "pnl_pct": best_iter.pnl_pct,
        "dd": best_iter.max_dd_pct,
        "trades": best_iter.trades,
        "thr": best_iter.ml_threshold,
        "tp_atr": best_iter.tp_atr,
        "kz": best_iter.killzone,
        "weights": best_iter.weights,
        "ml_model": best_iter.ml_model
    }

    return summary, best_iter, next_weights


def main():
    import argparse
    parser = argparse.ArgumentParser(description="ICT+ML weekly optimizer in bursts (cached data)")
    parser.add_argument("--bursts", type=int, default=2, help="Number of 5-minute bursts")
    parser.add_argument("--burst-seconds", type=int, default=300, help="Seconds per burst (default 300)")
    parser.add_argument("--max-trades", type=int, default=300, help="Max trades per burst evaluation")
    args = parser.parse_args()

    OUT_DIR_SUM.mkdir(parents=True, exist_ok=True)
    OUT_BEST.parent.mkdir(parents=True, exist_ok=True)

    df5 = _load_cached_5m()
    if df5 is None or df5.empty:
        print("ERROR: No cached 5m data found for 'Volatility 100 Index'. Ensure data/layered/*_5M.csv exists.")
        sys.exit(2)

    feats = build_mtf_features(df5, BASE_TF, HTF_TFS)
    bias_w = _compute_bias_weights(feats)

    global_best_key = (-9999.0, -1e18)
    global_best: Optional[Iteration] = None
    weights = {"ict": 0.2, "ml": 0.8}

    for i in range(1, args.bursts + 1):
        start = time.time()
        summary, best_iter, weights = _run_burst(i, feats, bias_w, weights, args.burst_seconds, args.max_trades)
        # write summary
        (OUT_DIR_SUM / f"burst_{i:03d}_summary.json").write_text(json.dumps(summary, indent=2, default=str))
        key = (best_iter.pnl / max(1e-9, abs(best_iter.max_dd_pct)+1.0), best_iter.pnl)  # heuristic
        if key > global_best_key:
            global_best_key = key
            global_best = best_iter
        # Soft resource pacing
        if time.time() - start < 2.0:
            time.sleep(1.0)

    if global_best is None:
        OUT_BEST.write_text(json.dumps({"status":"no_results"}, indent=2))
        print("No best iteration found.")
        return

    result = {
        "week": global_best.week,
        "pnl": global_best.pnl,
        "pnl_pct": global_best.pnl_pct,
        "max_drawdown_pct": global_best.max_dd_pct,
        "trades": global_best.trades,
        "applied_thresholds": {"ml_threshold": global_best.ml_threshold},
        "killzone_variant": global_best.killzone,
        "htf_frames": global_best.htf_frames,
        "strategy_weights": global_best.weights,
        "ml_model_used": global_best.ml_model,
        "symbol": SYMBOL,
        "base_timeframe": BASE_TF,
    }
    OUT_BEST.write_text(json.dumps(result, indent=2))

    # Console report
    print("=== ICT+ML Best Weekly Iteration (bursts) ===")
    print(f"Week: {result['week']}")
    print(f"PnL: ${result['pnl']:.2f} ({result['pnl_pct']:.2f}%)  |  MaxDD: {result['max_drawdown_pct']:.2f}%  |  Trades: {result['trades']}")
    print(f"Threshold: ml_threshold={result['applied_thresholds']['ml_threshold']:.2f}  |  Killzones: {result['killzone_variant']}")
    print(f"HTF: {result['htf_frames']}  |  Weights: {result['strategy_weights']}  |  ML: {result['ml_model_used']}")


if __name__ == "__main__":
    main()
