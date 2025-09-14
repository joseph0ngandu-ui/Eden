from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class SKLearnModelWrapper:
    model: Any

    def predict_proba(self, X):
        return self.model.predict_proba(X)


@dataclass
class LightGBMWrapper:
    model: Any

    def predict_proba(self, X):
        # LightGBM binary predict can produce raw scores; convert to pseudo proba via logistic
        import numpy as np
        try:
            proba = self.model.predict(X)
            # If 1D, map to two-class proba
            if proba.ndim == 1:
                p1 = 1 / (1 + np.exp(-proba))
                return np.vstack([1 - p1, p1]).T
            return proba
        except Exception:
            # Fallback uniform probability
            return np.tile([0.5, 0.5], (len(X), 1))