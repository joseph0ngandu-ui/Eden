from __future__ import annotations
import json
from pathlib import Path
import pandas as pd


def update_meta(decision_log: Path, out_path: Path, window_trades: int = 50) -> dict:
    """Compute simple correlations and adjust weight hints.
    Writes to out_path (JSON) and returns dict.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not decision_log.exists():
        meta = {"status": "no_decisions"}
        out_path.write_text(json.dumps(meta, indent=2))
        return meta
    try:
        df = pd.read_csv(decision_log)
    except Exception:
        meta = {"status": "read_error"}
        out_path.write_text(json.dumps(meta, indent=2))
        return meta

    df_tail = df.tail(max(1, window_trades))
    # Placeholder: compute mean probabilities
    meta = {
        "mean_stageA_P": float(df_tail.get('stageA_P', pd.Series(dtype=float)).dropna().mean() or 0.0),
        "mean_stageB_P": float(df_tail.get('stageB_P', pd.Series(dtype=float)).dropna().mean() or 0.0),
        "mean_stageC_P": float(df_tail.get('stageC_P', pd.Series(dtype=float)).dropna().mean() or 0.0),
        "update_window": int(window_trades),
    }
    out_path.write_text(json.dumps(meta, indent=2))
    return meta