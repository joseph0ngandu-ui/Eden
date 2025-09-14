from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any


@dataclass
class StrategyRegistry:
    path: Path = Path("models/registry.json")

    def _read(self) -> List[Dict[str, Any]]:
        if self.path.exists():
            return json.loads(self.path.read_text())
        return []

    def _write(self, items: List[Dict[str, Any]]):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(items, indent=2))

    def register(self, strategy_meta: Dict[str, Any]):
        items = self._read()
        items.append(strategy_meta)
        self._write(items)

    def list_active(self) -> List[Dict[str, Any]]:
        return [x for x in self._read() if x.get("active", True)]

    def deactivate(self, id_: str):
        items = self._read()
        for x in items:
            if x.get("id") == id_:
                x["active"] = False
        self._write(items)

    def prune(self, min_expectancy: float = 0.0):
        items = self._read()
        for x in items:
            if x.get("expectancy", 0.0) < min_expectancy:
                x["active"] = False
        self._write(items)
