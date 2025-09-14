from __future__ import annotations
from dataclasses import dataclass
import pandas as pd


@dataclass
class StrategyBase:
    name: str = "base"

    def on_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return DataFrame with columns: timestamp, side, confidence"""
        raise NotImplementedError

    def params(self) -> dict:
        return {}
