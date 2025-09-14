import json
from pathlib import Path
from typing import Any, Dict
import joblib


def save_model(model, meta: Dict[str, Any], models_dir: Path = Path("models")) -> Path:
    models_dir.mkdir(parents=True, exist_ok=True)
    name = meta.get("name", "model")
    path = models_dir / f"{name}.joblib"
    joblib.dump(model, path)
    # update registry
    reg = models_dir / "registry.json"
    registry = []
    if reg.exists():
        registry = json.loads(reg.read_text())
    registry.append(meta)
    reg.write_text(json.dumps(registry, indent=2))
    return path


def save_json(data: Dict[str, Any], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
