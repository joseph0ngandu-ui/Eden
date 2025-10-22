from pathlib import Path
from eden.data.loader import DataLoader


def test_sample_csv_load_and_columns():
    dl = DataLoader()
    p = Path(__file__).resolve().parents[1] / "data" / "sample_data" / "XAUUSD_1D.csv"
    df = dl.load_csv(p)
    assert set(["open", "high", "low", "close", "volume"]).issubset(df.columns)
    assert df.index.tz is not None


def test_yfinance_fetch_mock_fallback():
    dl = DataLoader()
    df = dl.fetch_yfinance(
        "XAUUSD", "1D", "2018-01-01", "2018-01-31", allow_network=False
    )
    assert df is None
