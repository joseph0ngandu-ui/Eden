from __future__ import annotations
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import optuna  # type: ignore

from ..cli import run_backtest
from ..utils.storage import init_db, insert_run, insert_metrics

log = logging.getLogger("eden.optimize.sweeps")

# Phase 1: Parameter sweep scaffold using Optuna

DEFAULT_OBJECTIVE = "sharpe"


def _objective_from_metrics(metrics: Dict[str, float], objective: str) -> float:
    if objective == "sharpe":
        return float(metrics.get("sharpe", 0.0))
    if objective == "profit_factor":
        return float(metrics.get("profit_factor", 0.0))
    if objective == "expectancy":
        return float(metrics.get("expectancy", 0.0))
    # Higher is better by default
    return float(metrics.get(objective, 0.0))


def run_parameter_sweep(
    symbols: list[str],
    timeframe: str,
    strategy: str,
    budget: int = 50,
    objective: str = DEFAULT_OBJECTIVE,
    constraints: Optional[Dict[str, float]] = None,
    search_space: Optional[Dict[str, Dict[str, Any]]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    min_confidence: float = 0.0,
) -> Dict[str, Any]:
    """Run a parameter sweep and return best params + trials summary."""
    init_db()
    constraints = constraints or {}
    search_space = search_space or {
        "sl_bars": {"type": "int", "low": 5, "high": 50},
        "tp_rr": {"type": "float", "low": 1.0, "high": 4.0},
        "fvg_strength": {"type": "float", "low": 0.0, "high": 1.0},
    }

    def sample_params(trial: optuna.Trial) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        for name, spec in search_space.items():
            t = spec.get("type")
            if t == "int":
                params[name] = trial.suggest_int(
                    name, spec.get("low", 0), spec.get("high", 10)
                )
            elif t == "float":
                params[name] = trial.suggest_float(
                    name, spec.get("low", 0.0), spec.get("high", 1.0)
                )
            elif t == "categorical":
                params[name] = trial.suggest_categorical(name, spec.get("choices", []))
        return params

    def _trial_fn(trial: optuna.Trial) -> float:
        params = sample_params(trial)
        run_id = str(uuid.uuid4())
        try:
            # Compose overrides
            overrides = {
                "symbols": symbols,
                "timeframe": timeframe,
                "strategy": strategy,
                "start": start or "2018-01-01",
                "end": end or "2023-12-31",
                # runtime flags
                "min_confidence": min_confidence,
            }
            overrides.update(params)

            # Execute backtest
            run_backtest(None, ci_short=False, overrides=overrides)

            # Read results
            result_dir = Path("results")
            metrics_path = result_dir / "metrics.json"
            if not metrics_path.exists():
                raise RuntimeError("metrics.json not found")
            import json

            metrics = json.loads(metrics_path.read_text())

            # Constraints filtering
            if "max_drawdown" in constraints:
                dd = float(metrics.get("max_drawdown", 1.0))
                if dd > float(constraints["max_drawdown"]):
                    return -1e9
            if "min_trades" in constraints:
                if int(metrics.get("trade_count", 0)) < int(constraints["min_trades"]):
                    return -1e9

            score = _objective_from_metrics(metrics, objective)
            insert_run(
                run_id,
                {
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "symbol": ",".join(symbols),
                    "timeframe": timeframe,
                    "strategy": strategy,
                    "params": params,
                    "status": "completed",
                    "metrics": metrics,
                    "results_path": str(result_dir),
                    "env_snapshot": {"objective": objective},
                },
            )
            insert_metrics(run_id, metrics)
            return float(score)
        except Exception as e:
            insert_run(
                run_id,
                {
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "symbol": ",".join(symbols),
                    "timeframe": timeframe,
                    "strategy": strategy,
                    "params": params,
                    "status": f"error: {e}",
                    "metrics": {},
                    "results_path": "results",
                    "env_snapshot": {"objective": objective},
                },
            )
            raise

    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(_trial_fn, n_trials=int(budget))

    best = study.best_trial
    trials = [
        {
            "number": t.number,
            "value": t.value,
            "params": t.params,
            "user_attrs": t.user_attrs,
        }
        for t in study.trials
    ]
    return {
        "best_value": best.value,
        "best_params": best.params,
        "trials": trials,
        "objective": objective,
    }
