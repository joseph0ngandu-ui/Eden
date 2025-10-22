from pathlib import Path
from eden.data.loader import DataLoader
from eden.features.feature_pipeline import build_feature_pipeline
from eden.backtest.engine import BacktestEngine
from eden.strategies.momentum import MomentumStrategy


def test_backtest_engine_basic_flow():
    p = Path(__file__).resolve().parents[1] / "data" / "sample_data" / "EURUSD_1D.csv"
    dl = DataLoader()
    df = dl.load_csv(p)
    feat = build_feature_pipeline(df)
    sig = MomentumStrategy().on_data(feat)
    eng = BacktestEngine(starting_cash=10000)
    trades = eng.run(feat, sig, symbol="EURUSD", risk_manager=None)
    assert isinstance(trades, list)
