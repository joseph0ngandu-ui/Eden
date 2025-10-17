from __future__ import annotations
from pathlib import Path
import json

def try_train_ppo(models_dir: Path, error_out: Path) -> bool:
    models_dir.mkdir(parents=True, exist_ok=True)
    try:
        import stable_baselines3  # noqa: F401
    except Exception as e:
        msg = f"PPO unavailable: {e}"
        error_out.parent.mkdir(parents=True, exist_ok=True)
        error_out.write_text(((error_out.read_text() if error_out.exists() else '') + "\n" + msg))
        return False
    # Placeholder: in constrained environments we skip heavy training
    (models_dir / 'README.txt').write_text('PPO training placeholder - integrate SB3 here')
    return True


def fallback_controller(decision: dict) -> dict:
    """Deterministic fallback policy adjustments."""
    out = decision.copy()
    dd = float(decision.get('dd_pct', 0.0))
    vol = float(decision.get('volatility_factor', 1.0))
    is_ml = str(decision.get('strategy','')).lower().startswith('ml')
    scale = 1.0
    if dd > 0.08:
        scale *= 0.5
    if vol > 2.0 and is_ml:
        out['pause'] = True
    out['risk_scale'] = scale
    return out