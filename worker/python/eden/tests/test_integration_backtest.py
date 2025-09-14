from pathlib import Path
from eden.data.loader import DataLoader
from eden.features.feature_pipeline import build_feature_pipeline
from eden.backtest.engine import BacktestEngine
from eden.backtest.analyzer import Analyzer
from eden.strategies.mean_reversion import MeanReversionStrategy
from eden.strategies.momentum import MomentumStrategy


def test_integration_backtest_end_to_end(tmp_path):
    dl = DataLoader()
    p1 = Path(__file__).resolve().parents[1] / 'data' / 'sample_data' / 'XAUUSD_1D.csv'
    p2 = Path(__file__).resolve().parents[1] / 'data' / 'sample_data' / 'EURUSD_1D.csv'
    df1 = dl.load_csv(p1)
    df2 = dl.load_csv(p2)
    feat1 = build_feature_pipeline(df1)
    feat2 = build_feature_pipeline(df2)

    engine = BacktestEngine(starting_cash=100000)
    sig1 = MomentumStrategy().on_data(feat1)
    sig2 = MeanReversionStrategy().on_data(feat2)
    engine.run(feat1, sig1, symbol='XAUUSD', risk_manager=None)
    engine.run(feat2, sig2, symbol='EURUSD', risk_manager=None)

    out_dir = tmp_path / 'results'
    out_dir.mkdir(parents=True, exist_ok=True)
    an = Analyzer(engine.trades)
    metrics = an.metrics()
    assert 'net_pnl' in metrics
    an.plot_equity_curve(save_path=out_dir / 'equity_curve.png')
    engine.save_trades_csv(out_dir / 'trades.csv')
    assert (out_dir / 'equity_curve.png').exists()
    assert (out_dir / 'trades.csv').exists()