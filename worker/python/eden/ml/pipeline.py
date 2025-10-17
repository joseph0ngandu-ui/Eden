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


def create_features_for_ml(df: pd.DataFrame, feature_alignment=None) -> Tuple[pd.DataFrame, pd.Series]:
    """Create features for ML training with optional feature alignment"""
    feat = build_feature_pipeline(df)
    # Simple target: next-period return > 0
    ret = feat['close'].pct_change().shift(-1).fillna(0.0)
    y = (ret > 0).astype(int)
    X = feat.drop(columns=['open','high','low','close','volume'], errors='ignore')
    
    # Filter out non-numeric columns (datetime, object, etc.)
    numeric_columns = X.select_dtypes(include=[np.number]).columns
    X = X[numeric_columns]
    
    # Apply feature alignment if provided
    if feature_alignment is not None:
        # Align features to match expected feature names
        expected_features = set(feature_alignment)
        current_features = set(X.columns)
        
        # Add missing features as zeros
        missing_features = expected_features - current_features
        for feature in missing_features:
            X[feature] = 0.0
            
        # Remove extra features not in alignment
        extra_features = current_features - expected_features
        if extra_features:
            X = X.drop(columns=list(extra_features), errors='ignore')
            
        # Reorder columns to match alignment
        X = X[sorted(expected_features.intersection(X.columns))]
    
    # Fill any remaining NaN values
    X = X.fillna(0.0)
    
    return X, y


def get_feature_alignment(df: pd.DataFrame) -> list:
    """Get standard feature alignment for consistent ML training/prediction"""
    feat = build_feature_pipeline(df)
    X = feat.drop(columns=['open','high','low','close','volume'], errors='ignore')
    numeric_columns = X.select_dtypes(include=[np.number]).columns
    return sorted(numeric_columns.tolist())


def train_model(X: pd.DataFrame, y: pd.Series, model_dir: Path = Path("models"), use_optuna: bool = True) -> TrainResult:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    from sklearn.ensemble import RandomForestClassifier

    X_train, X_test, y_train, y_test = train_test_split(X.fillna(0.0), y, test_size=0.25, random_state=42)

    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    prob = model.predict_proba(X_test)[:,1]
    auc = float(roc_auc_score(y_test, prob)) if len(np.unique(y_test)) > 1 else 0.5

    meta = {"name": "sample_model", "type": "RandomForest", "auc": auc}
    path = save_model(model, meta, models_dir=model_dir)
    # Also save as a predictable sample filename for quick checks
    import shutil
    shutil.copy(path, model_dir / "sample_model.joblib")
    return TrainResult(model_path=path, metrics={"auc": auc})


def minimal_train_entry(cfg):
    # lightweight train on XAUUSD sample data
    sample = Path(__file__).resolve().parents[1] / "data" / "sample_data" / "XAUUSD_1D.csv"
    import pandas as pd
    df = pd.read_csv(sample)
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    df = df.set_index('datetime')
    X, y = create_features_for_ml(df)
    res = train_model(X, y)
    # Ensure results dir exists and save brief metrics
    results_dir = Path("examples/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "train_metrics.json").write_text(json.dumps(res.metrics, indent=2))
