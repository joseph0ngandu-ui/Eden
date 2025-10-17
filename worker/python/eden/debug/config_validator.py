from __future__ import annotations
import json
from pathlib import Path

REQUIRED_KEYS = ['instrument','execution_tfs','htf_tfs','starting_cash']


def validate(run_config_path: Path, out_error: Path) -> bool:
    try:
        cfg = json.loads(run_config_path.read_text())
    except Exception as e:
        out_error.parent.mkdir(parents=True, exist_ok=True)
        out_error.write_text(f"run_config read error: {e}")
        return False
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        out_error.parent.mkdir(parents=True, exist_ok=True)
        out_error.write_text(f"Missing keys: {missing}")
        return False
    return True