import argparse
import json
import logging
import os
from pathlib import Path
from typing import Optional

from .config import EdenConfig, load_config
from .logging_conf import configure_logging
from .backtest.engine import BacktestEngine
from .backtest.analyzer import Analyzer
from .data.loader import DataLoader
from .features.feature_pipeline import build_feature_pipeline, build_mtf_features
import sys
from .strategies.base import StrategyBase
from .strategies.ict import ICTStrategy
from .strategies.mean_reversion import MeanReversionStrategy
from .strategies.momentum import MomentumStrategy
from .strategies.price_action import PriceActionStrategy
from .strategies.ml_generated import MLGeneratedStrategy
from .execution.paper_broker import PaperBroker
from .execution.mt5_broker import MT5Broker, is_mt5_available
from .risk.risk_manager import RiskManager


def _default_results_dir() -> Path:
    # Prefer results/ if present, otherwise examples/results for backwards compatibility
    base = Path("results").absolute()
    if base.exists():
        return base
    return Path("examples/results").absolute()


def init_workspace():
    root = Path.cwd()
    for d in [
        root / "data/cache",
        root / "models",
        root / "results",
        root / "logs",
        root / "examples/results",
    ]:
        d.mkdir(parents=True, exist_ok=True)
    # Create default config if missing
    cfg = root / "config.yml"
    if not cfg.exists():
        cfg.write_text(
            """
# Eden default config
symbols: ["XAUUSD", "EURUSD", "US30", "NAS100", "GBPUSD"]
timeframe: "1D"
start: "2018-01-01"
end: "2023-12-31"
starting_cash: 100000
commission_bps: 1.0
slippage_bps: 1.0
broker: "paper"
strategy: "ensemble"
log_level: "INFO"
            """.strip()
        )
    # .env example
    env_example = root / ".env.example"
    if not env_example.exists():
        env_example.write_text(
            """
# Copy to .env to configure secrets
EDEN_LIVE=0
EDEN_CONFIRM_LIVE=0
MT5_LOGIN=
MT5_PASSWORD=
MT5_SERVER=
            """.strip()
        )
    print("Workspace initialized. Edit config.yml and run commands.")


def _make_strategy(name: str, cfg: EdenConfig) -> list[StrategyBase]:
    name = (name or "").lower()
    if name in ("ict",):
        return [ICTStrategy()]
    if name in ("mean_reversion", "mr"):
        return [MeanReversionStrategy()]
    if name in ("momentum", "mom"):
        return [MomentumStrategy()]
    if name in ("price_action", "pa"):
        return [PriceActionStrategy()]
    if name in ("ml", "ml_generated"):
        return [MLGeneratedStrategy()]
    if name in ("ensemble", "default", "all"):
        return [
            ICTStrategy(),
            MeanReversionStrategy(),
            MomentumStrategy(),
            PriceActionStrategy(),
            MLGeneratedStrategy(),
        ]
    logging.getLogger(__name__).warning("Unknown strategy '%s', falling back to ensemble", name)
    return [ICTStrategy(), MeanReversionStrategy(), MomentumStrategy()]


def _bundled_sample_dir() -> Path:
    # Find sample_data when bundled with PyInstaller (sys._MEIPASS) or in source tree
    if hasattr(sys, "_MEIPASS"):
        return Path(getattr(sys, "_MEIPASS")) / "eden" / "data" / "sample_data"
    return Path(__file__).parent / "data" / "sample_data"


def run_backtest(config_path: Optional[str], ci_short: bool = False, overrides: Optional[dict] = None):
    cfg = load_config(config_path)
    # Apply runtime overrides from caller (e.g., UI)
    runtime = overrides or {}
    if overrides:
        try:
            from pydantic import ValidationError  # type: ignore
            cfg = EdenConfig(**{**cfg.model_dump(), **overrides})
        except Exception:
            cfg = EdenConfig(**{**cfg.model_dump(), **overrides})
    configure_logging(cfg.log_level)
    logger = logging.getLogger("eden.cli")
    logger.info("Starting backtest")

    dl = DataLoader(cache_dir=Path("data/cache"))
    symbols = cfg.symbols[:2] if (ci_short or os.getenv("CI") == "true") else cfg.symbols

    # Load data for each symbol (prefer MT5 when possible)
    data_map = {}
    for sym in symbols:
        df = dl.get_ohlcv(
            sym,
            cfg.timeframe,
            cfg.start,
            cfg.end,
            allow_network=not (os.getenv("CI") == "true"),
            prefer_mt5=True,  # default to MT5 source where possible
        )
        # Opportunistic prefetch of base+htf for caching when network is allowed
        try:
            if os.getenv("CI") != "true":
                from .data.transforms import normalize_timeframe  # type: ignore
                base = normalize_timeframe(cfg.timeframe)
                ladder = ["M1", "5M", "15M", "1H", "4H", "1D", "1W", "1MO"]
                # Pull one notch up for cache warm if available
                if base in ladder:
                    idx = ladder.index(base)
                    for extra in ladder[idx + 1 : idx + 3]:
                        _ = dl.get_ohlcv(sym, extra, cfg.start, cfg.end, allow_network=True, prefer_mt5=True)
        except Exception:
            pass
        if df is None or df.empty:
            # fallback to sample CSV
            sample = _bundled_sample_dir() / f"{sym}_{cfg.timeframe}.csv"
            if sample.exists():
                df = dl.load_csv(sample)
            else:
                logger.warning("No data available for %s; skipping", sym)
                continue
        data_map[sym] = df

    # Strategy selection: support dynamic selection via runtime flag or strategy == "dynamic"
    dynamic_strategy = bool(runtime.get("dynamic_strategy")) or (str(cfg.strategy).lower() in ("dynamic", "auto"))
    strategies: list[StrategyBase]
    if dynamic_strategy:
        try:
            from .ml.selector import select_strategies_for_symbol  # type: ignore
        except Exception:
            select_strategies_for_symbol = None

    engine = BacktestEngine(
        starting_cash=cfg.starting_cash,
        commission_bps=cfg.commission_bps,
        slippage_bps=cfg.slippage_bps,
    )

    risk = RiskManager()

    import pandas as pd
    all_trades = []

    def _choose_extra_tfs(tf: str) -> list[str]:
        """Return higher timeframes to include for multi-timeframe context by default.
        We scale up from the base timeframe to higher ones in this ladder:
        M1 < 5M < 15M < 1H < 4H < 1D < 1W < 1MO
        """
        try:
            from .data.transforms import normalize_timeframe  # type: ignore
        except Exception:
            normalize_timeframe = lambda s: (s or "").strip().upper()
        base = normalize_timeframe(tf)
        ladder = ["M1", "5M", "15M", "1H", "4H", "1D", "1W", "1MO"]
        rank = {t: i for i, t in enumerate(ladder)}
        if base not in rank:
            # Safe default: include top-down higher frames
            return ["1D", "1W", "1MO"]
        return [t for t in ladder if rank[t] > rank[base]]

    min_confidence = float(runtime.get("min_confidence", 0.0))

    for sym, df in data_map.items():
        # Build multi-timeframe features by default
        extras = _choose_extra_tfs(cfg.timeframe)
        if extras:
            df_feat = build_mtf_features(df, cfg.timeframe, extras)
        else:
            df_feat = build_feature_pipeline(df)

        # Strategy list resolution
        if dynamic_strategy and 'close' in df_feat.columns:
            try:
                selected = select_strategies_for_symbol(sym, cfg.timeframe, df_feat)
                strategies = selected if selected else _make_strategy(cfg.strategy, cfg)
                logger.info("Dynamic strategy for %s: %s", sym, ",".join([s.name for s in strategies]))
            except Exception as e:
                logger.warning("Dynamic strategy selection failed for %s: %s", sym, e)
                strategies = _make_strategy(cfg.strategy, cfg)
        else:
            strategies = _make_strategy(cfg.strategy, cfg)

        # Aggregate signals from all strategies
        signals_list: list[pd.DataFrame] = []
        for strat in strategies:
            try:
                sigs = strat.on_data(df_feat)
            except Exception as e:
                logger.warning("Strategy %s failed: %s", strat.name, e)
                continue
            if sigs is None or sigs.empty:
                continue
            # Ensure required cols
            if 'confidence' not in sigs.columns:
                sigs['confidence'] = 0.5
            if min_confidence > 0:
                sigs = sigs[sigs['confidence'] >= min_confidence]
            sigs["strategy"] = strat.name
            signals_list.append(sigs)
        if not signals_list:
            continue
        signals = pd.concat(signals_list, ignore_index=True)

        # Optional reversal scoring gate
        if runtime.get("use_reversal_scorer"):
            try:
                from .ml.reversal import score_reversals  # type: ignore
                signals = score_reversals(df_feat, signals)
                if min_confidence > 0:
                    signals = signals[signals['confidence'] >= min_confidence]
            except Exception as e:
                logger.warning("Reversal scoring failed: %s", e)

        trades = engine.run(df_feat, signals, symbol=sym, risk_manager=risk)
        all_trades.extend(trades)

    analyzer = Analyzer(all_trades)
    metrics = analyzer.metrics()

    results_dir = _default_results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    analyzer.plot_equity_curve(save_path=results_dir / "equity_curve.png")
    engine.save_trades_csv(results_dir / "trades.csv")

    logger.info("Backtest complete. Metrics saved to %s", results_dir)


def run_train(config_path: Optional[str]):
    cfg = load_config(config_path)
    configure_logging(cfg.log_level)
    # For brevity, call into ML pipeline minimal path
    from .ml.pipeline import minimal_train_entry
    minimal_train_entry(cfg)


def run_paper(config_path: Optional[str]):
    cfg = load_config(config_path)
    configure_logging(cfg.log_level)
    logger = logging.getLogger("eden.cli")

    broker = PaperBroker(slippage_bps=cfg.slippage_bps)
    logger.info("Paper trading session bootstrap complete. In tests we do not open network connections.")


def run_live(config_path: Optional[str], confirm: bool):
    cfg = load_config(config_path)
    configure_logging(cfg.log_level)
    logger = logging.getLogger("eden.cli")

    if os.getenv("EDEN_LIVE") != "1":
        logger.warning("EDEN_LIVE is not set to 1; refusing to run live. Falling back to paper broker.")
        return run_paper(config_path)
    if not (confirm or os.getenv("EDEN_CONFIRM_LIVE") == "1"):
        logger.error("Live trading requires --confirm or EDEN_CONFIRM_LIVE=1. Aborting.")
        return

    if is_mt5_available():
        broker = MT5Broker.from_env()
        logger.info("Live trading on MT5 started in safe mode.")
    else:
        logger.warning("MT5 not available; falling back to PaperBroker.")
        broker = PaperBroker()
    # Here we would wire strategies to live stream; left as safe no-op for CI.


def main():
    parser = argparse.ArgumentParser(description="Eden Trading System CLI")
    parser.add_argument("--init", "-i", action="store_true")
    parser.add_argument("--run-backtest", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--paper-trade", action="store_true")
    parser.add_argument("--live-trade", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--ci-short", action="store_true")
    # ML strategy discovery and pruning
    parser.add_argument("--discover", action="store_true", help="Run ML-based strategy discovery")
    parser.add_argument("--prune", action="store_true", help="Prune underperforming strategies")
    parser.add_argument("--retune", type=str, default=None, help="Retune a specific strategy by id")
    args = parser.parse_args()

    if args.init:
        init_workspace()
        return
    if args.run_backtest:
        return run_backtest(args.config, ci_short=args.ci_short)
    if args.train:
        return run_train(args.config)
    if args.paper_trade:
        return run_paper(args.config)
    if args.live_trade:
        return run_live(args.config, confirm=args.confirm)

    if args.discover or args.prune or args.retune:
        # Load one sample dataset and run discovery/prune/retune
        from .data.loader import DataLoader
        dl = DataLoader()
        sample = Path(__file__).parent / "data" / "sample_data" / "XAUUSD_1D.csv"
        df = dl.load_csv(sample)
        from .ml.discovery import StrategyDiscovery
        sd = StrategyDiscovery()
        if args.discover:
            sd.discover_strategies(df, generations=2, population_size=10, elite_size=3, min_trades=3, min_sharpe=0.0)
        if args.prune:
            sd.prune_underperforming(min_expectancy=0.0)
        if args.retune:
            sd.retune_strategy(args.retune, df, iterations=5)
        return

    parser.print_help()
