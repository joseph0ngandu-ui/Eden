from pathlib import Path
from eden.data.loader import DataLoader
from eden.features.feature_pipeline import build_feature_pipeline
from eden.strategies.mean_reversion import MeanReversionStrategy
from eden.strategies.momentum import MomentumStrategy
from eden.strategies.ict import ICTStrategy


def test_strategy_signals_generation():
    p = Path(__file__).resolve().parents[1] / "data" / "sample_data" / "XAUUSD_1D.csv"
    dl = DataLoader()
    df = dl.load_csv(p)
    feat = build_feature_pipeline(df)
    for strat in [MeanReversionStrategy(), MomentumStrategy(), ICTStrategy()]:
        sig = strat.on_data(feat)
        assert set(["timestamp", "side", "confidence"]).issubset(sig.columns)
