#!/usr/bin/env python3
"""
Strategies API Router
"""

from fastapi import APIRouter, HTTPException, Body
from typing import Dict, Any, List
from pathlib import Path
import json
from datetime import datetime
from app.models import StrategyItem

router = APIRouter(prefix="/strategies", tags=["strategies"])

DATA_DIR = (Path(__file__).resolve().parents[3] / 'data')
DATA_DIR.mkdir(exist_ok=True)
STRATEGIES_FILE = DATA_DIR / 'strategies.json'
VALIDATED_FILE = DATA_DIR / 'validated_strategies.json'


def _load_json(path: Path) -> Dict:
    print(f"DEBUG: Loading JSON from {path}")
    print(f"DEBUG: Path exists? {path.exists()}")
    print(f"DEBUG: Absolute path: {path.absolute()}")
    if path.exists():
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                print(f"DEBUG: Loaded {len(data)} items")
                return data
        except Exception as e:
            print(f"DEBUG: Error loading JSON: {e}")
            return {"error": str(e), "path": str(path)}
    print("DEBUG: Path does not exist")
    return {"error": "Path does not exist", "path": str(path)}


def _save_json(path: Path, data: Dict):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)


@router.post("")
async def upload_strategy(strategy: Dict[str, Any] = Body(...)):
    """Upload a new strategy configuration (e.g. from Mac app)."""
    strategy_id = strategy.get("id")
    if not strategy_id:
        raise HTTPException(status_code=400, detail="Strategy ID required")
        
    all_strats = _load_json(STRATEGIES_FILE)
    
    # Ensure it's marked as from external source
    strategy["source"] = "mac_app_upload"
    strategy["uploaded_at"] = datetime.utcnow().isoformat()
    
    # Save
    all_strats[strategy_id] = strategy
    _save_json(STRATEGIES_FILE, all_strats)
    
    return {"status": "success", "message": "Strategy uploaded", "id": strategy_id}


@router.get("", response_model=List[StrategyItem])
async def list_strategies():
    data = _load_json(STRATEGIES_FILE)
    if not data or "error" in data:
        return []
    
    strategies = []
    for s_id, s_data in data.items():
        # Ensure ID is present
        if "id" not in s_data:
            s_data["id"] = s_id
            
        strategies.append(s_data)
    return strategies


@router.get("/validated", response_model=List[StrategyItem])
async def list_validated_strategies():
    data = _load_json(VALIDATED_FILE)
    if not data or "error" in data:
        return []
        
    strategies = []
    for s_id, s_data in data.items():
        if "id" not in s_data:
            s_data["id"] = s_id
        strategies.append(s_data)
    return strategies


@router.get("/active", response_model=List[StrategyItem])
async def list_active_strategies():
    all_strats = _load_json(STRATEGIES_FILE)
    if not all_strats or "error" in all_strats:
        return []
        
    strategies = []
    for s_id, s_data in all_strats.items():
        if s_data.get('is_active'):
            if "id" not in s_data:
                s_data["id"] = s_id
            strategies.append(s_data)
    return strategies


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
