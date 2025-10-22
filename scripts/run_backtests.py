"""
Run individual and ensemble backtests using M1/M5 execution and HTF bias.
Saves per-run trades.csv, metrics.json, and equity_curve.png.
"""

from __future__ import annotations
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import sys

import pandas as pd

# Ensure 'eden' package is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "worker" / "python"))

from eden.data.mtf_fetcher import MTFDataFetcher
from eden.features.htf_ict_bias import (
    calculate_htf_bias,
    align_htf_to_execution_tf,
    compute_micro_features,
)
from eden.features.feature_pipeline import build_feature_pipeline
from eden.backtest.engine import BacktestEngine
from eden.backtest.analyzer import Analyzer

from eden.strategies.ict import ICTStrategy
from eden.strategies.mean_reversion import MeanReversionStrategy
from eden.strategies.momentum import MomentumStrategy
from eden.strategies.price_action import PriceActionStrategy
from eden.strategies.ml_generated import MLGeneratedStrategy


logging.basicConfig(level=logging.INFO)
log = logging.getLogger("scripts.run_backtests")


@dataclass
class BacktestConfig:
    days_back: int = 7
    starting_cash: float = 15.0  # Optimal micro account balance
    commission_bps: float = 1.0
    slippage_bps: float = 1.0
    min_confidence: float = 0.6  # Optimal for ML Generated
    stop_atr_multiplier: float = 1.2  # Optimal ATR stop multiplier
    tp_atr_multiplier: float = 1.5  # Optimal ATR TP multiplier
    # Dynamic risk sizing - optimized for micro accounts
    per_order_risk_fraction: float = 0.02  # 2% risk per trade
    min_trade_value: float = 0.50  # $0.50 minimum for micro accounts
    growth_factor: float = 0.5  # Conservative growth scaling


def _prepare_execution_frames(
    raw: Dict[str, pd.DataFrame], execution_tfs: List[str], htf_tfs: List[str]
) -> Dict[str, pd.DataFrame]:
    """Build features and align HTF bias to execution frames (M1/M5)."""
    # Fetch HTF bias primarily from 1H and 4H; 15M used as reference if available
    df_15m = raw.get("15M")
    df_1h = raw.get("1H")
    df_4h = raw.get("4H")

    if df_1h is None or df_4h is None:
        raise RuntimeError("Missing HTF data (1H/4H) to compute bias")

    htf_bias = calculate_htf_bias(df_15m, df_1h, df_4h)

    out: Dict[str, pd.DataFrame] = {}

    for tf in execution_tfs:
        dfe = raw.get(tf)
        if dfe is None or dfe.empty:
            continue
        # Build base features and micro features
        feat = build_feature_pipeline(dfe)
        micro = compute_micro_features(dfe)
        feat = pd.concat([feat, micro], axis=1)
        # Align bias
        bias = align_htf_to_execution_tf(feat, htf_bias)
        feat = pd.concat([feat, bias], axis=1)
        out[tf] = feat
    return out


def _run_one_strategy(
    symbol: str,
    tf_df: pd.DataFrame,
    strategy_name: str,
    cfg: BacktestConfig,
    out_dir: Path,
):
    """Run a single strategy backtest on a given execution timeframe DataFrame."""
    # Strategy factory
    if strategy_name == "ict":
        strat = ICTStrategy(
            stop_atr_multiplier=cfg.stop_atr_multiplier,
            tp_atr_multiplier=cfg.tp_atr_multiplier,
            min_confidence=cfg.min_confidence,
        )
    elif strategy_name == "mean_reversion":
        strat = MeanReversionStrategy()
    elif strategy_name == "momentum":
        strat = MomentumStrategy()
    elif strategy_name == "price_action":
        strat = PriceActionStrategy()
    elif strategy_name == "ml_generated":
        strat = MLGeneratedStrategy()
    else:
        raise ValueError(f"Unknown strategy {strategy_name}")

    signals = strat.on_data(tf_df)
    if signals is None or signals.empty:
        log.warning("%s produced no signals", strategy_name)
        return None

    eng = BacktestEngine(
        starting_cash=cfg.starting_cash,
        commission_bps=cfg.commission_bps,
        slippage_bps=cfg.slippage_bps,
        per_order_risk_fraction=cfg.per_order_risk_fraction,
        min_trade_value=cfg.min_trade_value,
        growth_factor=cfg.growth_factor,
    )
    trades = eng.run(tf_df, signals, symbol=symbol, risk_manager=None)

    # Save outputs
    run_dir = out_dir / strategy_name
    run_dir.mkdir(parents=True, exist_ok=True)
    eng.save_trades_csv(run_dir / "trades.csv")
    an = Analyzer(trades, starting_cash=cfg.starting_cash)
    metrics = an.metrics()
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    an.plot_equity_curve(save_path=run_dir / "equity_curve.png")

    return metrics


def run_all_backtests(
    cfg: BacktestConfig,
    execution_tfs: List[str],
    htf_tfs: List[str],
    strategies: List[str],
    instrument: str = "Volatility 100 Index",
    start_date: str | None = None,
    end_date: str | None = None,
    output_dir: str = "results",
) -> Dict[str, dict]:
    results_dir = Path(output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1) Fetch data via MT5
    fetcher = MTFDataFetcher(symbol=instrument)
    # Normalize and validate timeframes
    valid_tfs = {"M1", "M5", "15M", "1H", "4H", "1D"}
    all_tfs = list({tf.strip().upper() for tf in (execution_tfs + htf_tfs)})
    all_tfs = [tf for tf in all_tfs if tf in valid_tfs]
    if start_date and end_date:
        from datetime import datetime as _dt

        sd = _dt.fromisoformat(start_date)
        ed = _dt.fromisoformat(end_date)
        raw = fetcher.fetch_all_timeframes_range(
            timeframes=all_tfs, start_date=sd, end_date=ed, use_cache=True
        )
    else:
        raw = fetcher.fetch_all_timeframes(
            timeframes=all_tfs, days_back=cfg.days_back, use_cache=True
        )
    fetcher.shutdown()
    if not raw:
        raise RuntimeError("Failed to fetch any data")

    # 2) Prepare execution frames with features and bias
    exec_frames = _prepare_execution_frames(raw, execution_tfs, htf_tfs)
    if not exec_frames:
        raise RuntimeError("No execution frames available")

    symbol = "VIX100"
    all_metrics: Dict[str, dict] = {}

    for tf, df_exec in exec_frames.items():
        out_dir = results_dir / f"backtests_{tf}"
        out_dir.mkdir(exist_ok=True)
        for strat in strategies:
            if strat == "ensemble":
                continue
            log.info("Running %s on %s ...", strat, tf)
            m = _run_one_strategy(symbol, df_exec, strat, cfg, out_dir)
            if m:
                all_metrics[f"{strat}_{tf}"] = m

    # Simple ensemble (deterministic): merge signals by union, weight confidence
    try:
        for tf, df_exec in exec_frames.items():
            out_dir = results_dir / f"backtests_{tf}"
            # Build signals from each strategy
            sigs: List[pd.DataFrame] = []
            for strat in strategies:
                if strat == "ensemble":
                    continue
                s = None
                # Recompute signals
                if strat == "ict":
                    s = ICTStrategy(
                        stop_atr_multiplier=cfg.stop_atr_multiplier,
                        tp_atr_multiplier=cfg.tp_atr_multiplier,
                        min_confidence=cfg.min_confidence,
                    ).on_data(df_exec)
                elif strat == "mean_reversion":
                    s = MeanReversionStrategy().on_data(df_exec)
                elif strat == "momentum":
                    s = MomentumStrategy().on_data(df_exec)
                elif strat == "price_action":
                    s = PriceActionStrategy().on_data(df_exec)
                elif strat == "ml_generated":
                    s = MLGeneratedStrategy().on_data(df_exec)
                if s is not None and not s.empty:
                    sigs.append(s)
            if sigs:
                all_sigs = pd.concat(sigs, ignore_index=True)
                # Aggregate by timestamp+side: average confidence
                if not all_sigs.empty:
                    grouped = (
                        all_sigs.groupby(["timestamp", "side"])["confidence"]
                        .mean()
                        .reset_index()
                    )
                    eng = BacktestEngine(
                        starting_cash=cfg.starting_cash,
                        commission_bps=cfg.commission_bps,
                        slippage_bps=cfg.slippage_bps,
                        per_order_risk_fraction=cfg.per_order_risk_fraction,
                        min_trade_value=cfg.min_trade_value,
                        growth_factor=cfg.growth_factor,
                    )
                    trades = eng.run(df_exec, grouped, symbol=symbol, risk_manager=None)
                    run_dir = out_dir / "ensemble"
                    run_dir.mkdir(exist_ok=True)
                    eng.save_trades_csv(run_dir / "trades.csv")
                    an = Analyzer(trades, starting_cash=cfg.starting_cash)
                    metrics = an.metrics()
                    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
                    an.plot_equity_curve(save_path=run_dir / "equity_curve.png")
                    all_metrics[f"ensemble_{tf}"] = metrics
    except Exception as e:
        log.warning("Ensemble run failed: %s", e)

    # Save summary
    (results_dir / "metrics.json").write_text(json.dumps(all_metrics, indent=2))
    return all_metrics


if __name__ == "__main__":
    run_all_backtests(days_back=7)
