#!/usr/bin/env python3
"""
Correlation analysis among top instruments
"""
import json
from pathlib import Path
import pandas as pd


def load_results(backtest_dir: str) -> dict:
    path = Path(backtest_dir) / "optimization_summary.json"
    with open(path, "r") as f:
        return json.load(f)


def load_prices(data_dir: str, symbol: str) -> pd.DataFrame:
    return pd.read_parquet(Path(data_dir) / f"{symbol}_clean.parquet")


def compute_returns(df: pd.DataFrame) -> pd.Series:
    return df["close"].pct_change().dropna()


def analyze_correlation(data_dir: str = "data", backtest_dir: str = "backtests", top_n: int = 5):
    results = load_results(backtest_dir)

    # Rank instruments by Sharpe of their default/best strategy
    ranking = []
    for sym, res in results.items():
        if res.get("status") == "failed":
            continue
        strategies = res.get("strategies", {})
        # take first strategy metrics
        if strategies:
            first_key = next(iter(strategies))
            sharpe = strategies[first_key].get("metrics", {}).get("sharpe_ratio", 0.0)
            ranking.append((sym, sharpe))

    top_syms = [s for s, _ in sorted(ranking, key=lambda x: x[1], reverse=True)[:top_n]]

    rets = {}
    for sym in top_syms:
        df = load_prices(data_dir, sym)
        rets[sym] = compute_returns(df)

    # Align indexes
    aligned = pd.concat(rets, axis=1).dropna()
    corr = aligned.corr()

    out_path = Path(backtest_dir) / "correlation_top_instruments.csv"
    corr.to_csv(out_path)
    print(f"Saved correlation matrix to {out_path}")


if __name__ == "__main__":
    analyze_correlation()
