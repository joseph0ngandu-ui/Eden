#!/usr/bin/env python3
"""
Strategies API Router
"""

from fastapi import APIRouter, HTTPException, Body
from typing import Dict, Any
from pathlib import Path
import json

router = APIRouter(prefix="/strategies", tags=["strategies"])

DATA_DIR = (Path(__file__).resolve().parents[1] / 'data')
DATA_DIR.mkdir(exist_ok=True)
STRATEGIES_FILE = DATA_DIR / 'strategies.json'
VALIDATED_FILE = DATA_DIR / 'validated_strategies.json'


def _load_json(path: Path) -> Dict:
    if path.exists():
        try:
            return json.load(open(path, 'r'))
        except Exception:
            return {}
    return {}


def _save_json(path: Path, data: Dict):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)


@router.get("")
async def list_strategies():
    return _load_json(STRATEGIES_FILE)


@router.get("/validated")
async def list_validated_strategies():
    return _load_json(VALIDATED_FILE)


@router.get("/active")
async def list_active_strategies():
    all_strats = _load_json(STRATEGIES_FILE)
    return {k: v for k, v in all_strats.items() if v.get('is_active')}


@router.put("/{strategy_id}/activate")
async def activate_strategy(strategy_id: str):
    all_strats = _load_json(STRATEGIES_FILE)
    if strategy_id not in all_strats:
        raise HTTPException(status_code=404, detail="Strategy not found")
    validated = _load_json(VALIDATED_FILE)
    if strategy_id not in validated:
        raise HTTPException(status_code=400, detail="Strategy not validated")
    s = all_strats[strategy_id]
    s['is_active'] = True
    # Start in PAPER mode by default until promoted
    s.setdefault('mode', 'PAPER')
    s.setdefault('policy', {'required_paper_trades': 10})
    all_strats[strategy_id] = s
    _save_json(STRATEGIES_FILE, all_strats)
    return {"status": "ok", "strategy_id": strategy_id, "mode": s['mode']}


@router.put("/{strategy_id}/deactivate")
async def deactivate_strategy(strategy_id: str):
    all_strats = _load_json(STRATEGIES_FILE)
    if strategy_id not in all_strats:
        raise HTTPException(status_code=404, detail="Strategy not found")
    all_strats[strategy_id]['is_active'] = False
    _save_json(STRATEGIES_FILE, all_strats)
    return {"status": "ok", "strategy_id": strategy_id}


@router.put("/{strategy_id}/promote")
async def promote_strategy(strategy_id: str):
    all_strats = _load_json(STRATEGIES_FILE)
    if strategy_id not in all_strats:
        raise HTTPException(status_code=404, detail="Strategy not found")
    s = all_strats[strategy_id]
    if not s.get('is_active'):
        raise HTTPException(status_code=400, detail="Strategy not active")
    # Manual promotion to LIVE
    s['mode'] = 'LIVE'
    all_strats[strategy_id] = s
    _save_json(STRATEGIES_FILE, all_strats)
    return {"status": "ok", "strategy_id": strategy_id, "mode": "LIVE"}


@router.patch("/{strategy_id}/policy")
async def set_policy(strategy_id: str, payload: Dict[str, Any] = Body(...)):
    all_strats = _load_json(STRATEGIES_FILE)
    if strategy_id not in all_strats:
        raise HTTPException(status_code=404, detail="Strategy not found")
    s = all_strats[strategy_id]
    policy = s.get('policy', {})
    policy.update(payload or {})
    s['policy'] = policy
    all_strats[strategy_id] = s
    _save_json(STRATEGIES_FILE, all_strats)
    return {"status": "ok", "strategy_id": strategy_id, "policy": policy}
