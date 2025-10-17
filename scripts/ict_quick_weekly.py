#!/usr/bin/env python3
"""
Quick ICT-only weekly optimizer on cached 5m VIX100 data.
- Time limit: 5 minutes (hard stop)
- Sweep: ml_threshold in {0.35, 0.40, 0.45}; killzones variants: 
  A) LON 07-11 / NY 12-17, B) LON 06-11 / NY 13-17
- Anomaly filter OFF; RR: ATR stop 1.0 / TP 3.0
- HTF frames: M15, 1H, 4H
- Offline only: use cached data (no network)
- ML: do not train; if a prior model exists, record it but do not retrain
- Output best weekly iteration only to console and results/ict_quick_best_iteration.json
"""
from __future__ import annotations
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

# Ensure eden package is importable
ROOT = Path(__file__).resolve().parents[1]
EDEN_PY = ROOT / 'worker' / 'python'
if str(EDEN_PY) not in sys.path:
    sys.path.insert(0, str(EDEN_PY))

from eden.data.loader import DataLoader
from eden.features.feature_pipeline import build_mtf_features
from eden.strategies.ict import ICTStrategy
from eden.backtest.engine import BacktestEngine
from eden.backtest.analyzer import Analyzer

SYMBOL = "Volatility 100 Index"
BASE_TF = "5M"
HTF_TFS = ["15M", "1H", "4H"]
TIME_LIMIT_SEC = 300

ML_THRESHOLDS = [0.35, 0.40, 0.45]
KILLZONE_VARIANTS = [
    ("07-11", "12-17", "LON07-11_NY12-17"),
    ("06-11", "13-17", "LON06-11_NY13-17"),
]

@dataclass
class WeeklyResult:
    week: str
    pnl: float
    pnl_pct: float
    max_drawdown_pct: float
    trades: int
    ml_threshold: float
    killzone_label: str
    htf_frames: str
    strategy_weights: Dict[str, float]
    ml_config_used: str


def _load_cached_5m() -> Optional['pd.DataFrame']:
    import pandas as pd
    dl = DataLoader(cache_dir=Path("data/cache"))

    # Try direct cache retrieval (no network)
    df = dl.get_ohlcv(SYMBOL, BASE_TF, "2000-01-01", "2100-01-01", allow_network=False, force_refresh=False, prefer_mt5=False)
    if df is not None and not df.empty:
        return df

    # Fallback to layered store
    layered = Path("data/layered") / f"{SYMBOL.upper()}_{BASE_TF.upper()}.csv"
    if layered.exists():
        try:
            return dl.load_csv(layered)
        except Exception:
            pass

    # Last resort: common aliases
    for name in ["VIX100_5M.csv", "VOLATILITY100_5M.csv", "VOL100_5M.csv"]:
        p = Path("data/layered") / name
        if p.exists():
            try:
                return dl.load_csv(p)
            except Exception:
                continue
    return None


def _week_key(ts) -> str:
    dt = pd.to_datetime(ts, utc=True)
    year, week, _ = dt.isocalendar()
    # Get start/end of the ISO week (Mon-Sun) in dataset timezone (UTC)
    start = (dt - pd.to_timedelta(dt.weekday(), unit='D')).normalize()
    end = start + pd.Timedelta(days=6, hours=23, minutes=59, seconds=59)
    return f"{int(year)}-W{int(week):02d}", start, end


def _best_known_ml_config() -> str:
    # Record presence of previously saved models without training
    models_dir = Path("models")
    if not models_dir.exists():
        return "none"
    candidates = list(models_dir.glob("*.joblib"))
    if candidates:
        return f"{candidates[0].name} (pre-existing)"
    return "none"


def main():
    global pd
    import pandas as pd

    t0 = time.time()
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "ict_quick_best_iteration.json"

    df5 = _load_cached_5m()
    if df5 is None or df5.empty:
        print("ERROR: No cached 5m data found for 'Volatility 100 Index'. Ensure data/layered/*_5M.csv exists.")
        sys.exit(2)

    # Build features with MTF (no anomaly filter)
    feats = build_mtf_features(df5, BASE_TF, HTF_TFS)

    # Partition by week
    weekly_map: Dict[str, Tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]] = {}
    for ts in feats.index:
        wk, start, end = _week_key(ts)
        if wk not in weekly_map:
            weekly_map[wk] = ([], start, end)  # will replace list with df later
    # Efficient weekly grouping
    feats_week = feats.copy()
    feats_week['__week__'] = feats_week.index.to_period('W').astype(str)
    grouped = feats_week.groupby('__week__')

    best: Optional[WeeklyResult] = None

    ml_config = _best_known_ml_config()

    # Sweep across parameter grid and weeks until time limit
    for ml_thr in ML_THRESHOLDS:
        for lon_win, ny_win, kz_label in KILLZONE_VARIANTS:
            if time.time() - t0 >= TIME_LIMIT_SEC:
                break
            # Configure ICT strategy
            ict = ICTStrategy(
                require_htf_bias=True,
                ml_fallback_on_neutral=True,
                stop_atr_multiplier=1.0,
                tp_atr_multiplier=3.0,
                min_confidence=ml_thr,
                killzones_enabled=True,
                killzones="london,ny",
                london_window=lon_win,
                ny_window=ny_win,
            )

            for wk, dfw in grouped:
                if time.time() - t0 >= TIME_LIMIT_SEC:
                    break
                df_week = dfw.drop(columns=['__week__'], errors='ignore')
                if len(df_week) < 10:
                    continue
                try:
                    sig = ict.on_data(df_week)
                    if sig is None or sig.empty:
                        continue
                    # Run backtest with micro-account style sizing
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
                    pnl = float(m.get('net_pnl', 0.0))
                    pnl_pct = float(m.get('equity_growth_pct', 0.0))
                    dd = float(m.get('max_drawdown_pct', 0.0))
                    ntr = int(m.get('trades', 0))

                    curr = WeeklyResult(
                        week=wk,
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        max_drawdown_pct=dd,
                        trades=ntr,
                        ml_threshold=ml_thr,
                        killzone_label=kz_label,
                        htf_frames=",".join(HTF_TFS),
                        strategy_weights={"ict": 1.0},
                        ml_config_used=ml_config,
                    )

                    # Select best by Sharpe (if available) then PnL
                    best_key = (float(m.get('sharpe', -9999.0)), pnl)
                    if best is None:
                        best = curr
                        best._sel_key = best_key  # type: ignore
                    else:
                        prev_key = getattr(best, '_sel_key', (-9999.0, -1e18))  # type: ignore
                        if best_key > prev_key:
                            best = curr
                            best._sel_key = best_key  # type: ignore
                except Exception:
                    continue

    if best is None:
        print("No trades generated in the allotted time.")
        data = {"status": "no_results", "reason": "no_signals_or_data", "time_sec": time.time() - t0}
        out_file.write_text(json.dumps(data, indent=2))
        sys.exit(0)

    # Prepare output
    result_obj = {
        "week": best.week,
        "pnl": best.pnl,
        "pnl_pct": best.pnl_pct,
        "max_drawdown_pct": best.max_drawdown_pct,
        "trades": best.trades,
        "applied_thresholds": {"ml_threshold": best.ml_threshold},
        "killzone_variant": best.killzone_label,
        "htf_frames": best.htf_frames,
        "strategy_weights": best.strategy_weights,
        "ml_config_used": best.ml_config_used,
        "duration_sec": round(time.time() - t0, 2),
        "symbol": SYMBOL,
        "base_timeframe": BASE_TF,
    }

    # Console output (concise)
    print("=== ICT Quick Best Weekly Iteration ===")
    print(f"Week: {result_obj['week']}")
    print(f"PnL: ${result_obj['pnl']:.2f} ({result_obj['pnl_pct']:.2f}%)  |  MaxDD: {result_obj['max_drawdown_pct']:.2f}%  |  Trades: {result_obj['trades']}")
    print(f"Threshold: ml_threshold={result_obj['applied_thresholds']['ml_threshold']:.2f}  |  Killzones: {result_obj['killzone_variant']}")
    print(f"HTF: {result_obj['htf_frames']}  |  Weights: {result_obj['strategy_weights']}  |  ML: {result_obj['ml_config_used']}")

    out_file.write_text(json.dumps(result_obj, indent=2))

if __name__ == "__main__":
    main()
