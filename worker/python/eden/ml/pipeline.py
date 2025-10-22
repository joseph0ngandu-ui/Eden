from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from ..features.feature_pipeline import build_feature_pipeline
from ..utils.persistence import save_model


@dataclass
class TrainResult:
    model_path: Path
    metrics: dict


def create_features_for_ml(
    df: pd.DataFrame, feature_alignment=None
) -> Tuple[pd.DataFrame, pd.Series]:
    """Create features for ML training with optional feature alignment"""
    feat = build_feature_pipeline(df)
    # Simple target: next-period return > 0
    ret = feat["close"].pct_change().shift(-1).fillna(0.0)
    y = (ret > 0).astype(int)
    X = feat.drop(columns=["open", "high", "low", "close", "volume"], errors="ignore")

    # Filter out non-numeric columns (datetime, object, etc.)
    numeric_columns = X.select_dtypes(include=[np.number]).columns
    X = X[numeric_columns]

    # Drop duplicate columns before alignment
    if X.columns.duplicated().any():
        X = X.loc[:, ~X.columns.duplicated()]

    # Apply feature alignment if provided (preserve order; fill missing with 0)
    if feature_alignment is not None:
        # De-duplicate alignment while preserving order
        seen = set()
        ordered = []
        for f in feature_alignment:
            if f not in seen:
                seen.add(f)
                ordered.append(f)
        # Reindex columns to exact alignment (missing filled with 0.0)
        X = X.reindex(columns=ordered, fill_value=0.0)

    # Fill any remaining NaN values
    X = X.fillna(0.0)

    return X, y


def get_feature_alignment(df: pd.DataFrame) -> list:
    """Get standard feature alignment for consistent ML training/prediction"""
    feat = build_feature_pipeline(df)
    X = feat.drop(columns=["open", "high", "low", "close", "volume"], errors="ignore")
    numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
    # De-duplicate while preserving order
    seen = set()
    ordered = []
    for f in numeric_columns:
        if f not in seen:
            seen.add(f)
            ordered.append(f)
    return ordered


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_dir: Path = Path("models"),
    use_optuna: bool = True,
) -> TrainResult:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    from sklearn.ensemble import RandomForestClassifier

    X_train, X_test, y_train, y_test = train_test_split(
        X.fillna(0.0), y, test_size=0.25, random_state=42
    )

    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    prob = model.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, prob)) if len(np.unique(y_test)) > 1 else 0.5

    meta = {"name": "sample_model", "type": "RandomForest", "auc": auc}
    path = save_model(model, meta, models_dir=model_dir)
    # Also save as a predictable sample filename for quick checks
    import shutil

    shutil.copy(path, model_dir / "sample_model.joblib")
    return TrainResult(model_path=path, metrics={"auc": auc})


def minimal_train_entry(cfg):
    # lightweight train on XAUUSD sample data
    sample = (
        Path(__file__).resolve().parents[1] / "data" / "sample_data" / "XAUUSD_1D.csv"
    )
    import pandas as pd

    df = pd.read_csv(sample)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.set_index("datetime")
    X, y = create_features_for_ml(df)
    res = train_model(X, y)
    # Ensure results dir exists and save brief metrics
    results_dir = Path("examples/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "train_metrics.json").write_text(json.dumps(res.metrics, indent=2))
