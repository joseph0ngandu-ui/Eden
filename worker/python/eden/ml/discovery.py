from __future__ import annotations
import json
import random
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

import numpy as np
import pandas as pd

from ..strategies.parametric import RuleBasedParamStrategy
from ..backtest.engine import BacktestEngine
from ..backtest.analyzer import Analyzer
from ..features.feature_pipeline import build_feature_pipeline
from ..utils.persistence import save_json
from .strategy_registry import StrategyRegistry


def generate_random_params() -> Dict[str, Any]:
    """Generate random strategy parameters within reasonable bounds."""
    return {
        "buy_rsi_max": float(random.uniform(20, 40)),
        "sell_rsi_min": float(random.uniform(60, 80)),
        "use_ema_cross": random.choice([True, False]),
        "macd_hist_bias": float(random.uniform(-0.01, 0.01)),
        "atr_mult_stop": float(random.uniform(1.5, 3.0)),
        "risk_frac": float(random.uniform(0.005, 0.02)),
    }


def mutate_params(params: Dict[str, Any], mutation_rate: float = 0.3) -> Dict[str, Any]:
    """Mutate existing parameters for genetic algorithm."""
    new_params = params.copy()
    for key, val in params.items():
        if random.random() < mutation_rate:
            if isinstance(val, bool):
                new_params[key] = not val
            elif isinstance(val, float):
                if "rsi" in key:
                    new_params[key] = float(
                        np.clip(val + random.uniform(-5, 5), 10, 90)
                    )
                else:
                    new_params[key] = float(val * random.uniform(0.8, 1.2))
    return new_params


def crossover_params(p1: Dict[str, Any], p2: Dict[str, Any]) -> Dict[str, Any]:
    """Genetic crossover of two parent parameter sets."""
    child = {}
    for key in p1.keys():
        child[key] = p1[key] if random.random() < 0.5 else p2.get(key, p1[key])
    return child


def evaluate_strategy(
    df: pd.DataFrame, params: Dict[str, Any], symbol: str = "DISCOVER"
) -> Dict[str, float]:
    """Run backtest and return performance metrics."""
    strat = RuleBasedParamStrategy(
        name=f"gen_{hash(str(params)) % 10000}", params=params
    )
    feat = build_feature_pipeline(df)
    signals = strat.on_data(feat)
    engine = BacktestEngine(starting_cash=100000)
    trades = engine.run(feat, signals, symbol=symbol, risk_manager=None)
    metrics = Analyzer(trades).metrics()

    # Add extra metrics for better selection
    n_trades = metrics.get("trades", 0)
    if n_trades == 0:
        expectancy = 0.0
        win_rate = 0.0
    else:
        wins = [t.pnl for t in trades if t.pnl > 0]
        losses = [abs(t.pnl) for t in trades if t.pnl < 0]
        win_rate = len(wins) / max(1, n_trades)
        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = np.mean(losses) if losses else 0.0
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

    return {
        "net_pnl": metrics.get("net_pnl", 0.0),
        "sharpe": metrics.get("sharpe", 0.0),
        "max_dd": metrics.get("max_dd", 0.0),
        "trades": n_trades,
        "expectancy": expectancy,
        "win_rate": win_rate,
    }


class StrategyDiscovery:
    def __init__(
        self, data_dir: Path = Path("data/cache"), models_dir: Path = Path("models")
    ):
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.strategies_db = models_dir / "strategies_db"
        self.strategies_db.mkdir(parents=True, exist_ok=True)
        self.registry = StrategyRegistry(path=models_dir / "registry.json")
        self.log = logging.getLogger("eden.ml.discovery")

    def discover_strategies(
        self,
        df: pd.DataFrame,
        generations: int = 5,
        population_size: int = 20,
        elite_size: int = 5,
        mutation_rate: float = 0.3,
        min_trades: int = 5,
        min_sharpe: float = 0.5,
        symbol: str | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Genetic algorithm to discover profitable strategy parameters.
        """
        self.log.info("Starting strategy discovery with %d generations", generations)

        # Split data for in-sample and out-of-sample validation
        split_idx = int(len(df) * 0.7)
        train_df = df.iloc[:split_idx]
        val_df = df.iloc[split_idx:]

        # Initialize population
        population = [
            {"params": generate_random_params(), "metrics": None}
            for _ in range(population_size)
        ]

        best_strategies = []

        for gen in range(generations):
            self.log.info("Generation %d/%d", gen + 1, generations)

            # Evaluate population on training data
            for candidate in population:
                if candidate["metrics"] is None:
                    candidate["metrics"] = evaluate_strategy(
                        train_df, candidate["params"]
                    )

            # Sort by composite score
            population.sort(
                key=lambda x: self._fitness_score(x["metrics"]), reverse=True
            )

            # Select elites
            elites = population[:elite_size]

            # Out-of-sample validation for top candidates
            for elite in elites[:3]:
                val_metrics = evaluate_strategy(
                    val_df, elite["params"], symbol="VALIDATION"
                )
                if self._is_viable(val_metrics, min_trades, min_sharpe):
                    strategy_meta = {
                        "id": f"discovered_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}",
                        "params": elite["params"],
                        "train_metrics": elite["metrics"],
                        "val_metrics": val_metrics,
                        "generation": gen,
                        "discovered_at": datetime.now().isoformat(),
                        "symbol": symbol,
                    }
                    best_strategies.append(strategy_meta)
                    self._save_strategy(strategy_meta)

            # Create next generation
            new_population = elites.copy()

            while len(new_population) < population_size:
                if random.random() < 0.7:  # Crossover
                    p1 = random.choice(elites)
                    p2 = random.choice(population[: population_size // 2])
                    child_params = crossover_params(p1["params"], p2["params"])
                else:  # Mutation
                    parent = random.choice(elites)
                    child_params = mutate_params(parent["params"], mutation_rate)

                new_population.append({"params": child_params, "metrics": None})

            population = new_population

        self.log.info(
            "Discovery complete. Found %d viable strategies", len(best_strategies)
        )
        return best_strategies

    def prune_underperforming(
        self, lookback_days: int = 30, min_expectancy: float = 0.0
    ):
        """
        Prune strategies that have been underperforming in recent backtests.
        """
        self.log.info("Pruning underperforming strategies")
        active = self.registry.list_active()

        for strategy_meta in active:
            if (
                strategy_meta.get("val_metrics", {}).get("expectancy", 0.0)
                < min_expectancy
            ):
                self.registry.deactivate(strategy_meta["id"])
                self.log.info(
                    "Deactivated strategy %s (expectancy: %.2f)",
                    strategy_meta["id"],
                    strategy_meta.get("val_metrics", {}).get("expectancy", 0.0),
                )

    def retune_strategy(
        self, strategy_id: str, df: pd.DataFrame, iterations: int = 10
    ) -> Dict[str, Any]:
        """
        Re-tune an existing strategy's parameters using fresh data.
        """
        # Load existing strategy
        strategy_path = self.strategies_db / f"{strategy_id}.json"
        if not strategy_path.exists():
            self.log.warning("Strategy %s not found", strategy_id)
            return {}

        strategy_meta = json.loads(strategy_path.read_text())
        base_params = strategy_meta["params"]

        best_params = base_params.copy()
        best_metrics = evaluate_strategy(df, best_params)

        for _ in range(iterations):
            candidate_params = mutate_params(base_params, mutation_rate=0.2)
            candidate_metrics = evaluate_strategy(df, candidate_params)

            if self._fitness_score(candidate_metrics) > self._fitness_score(
                best_metrics
            ):
                best_params = candidate_params
                best_metrics = candidate_metrics

        # Update strategy with new parameters
        strategy_meta["params"] = best_params
        strategy_meta["retuned_at"] = datetime.now().isoformat()
        strategy_meta["retune_metrics"] = best_metrics

        self._save_strategy(strategy_meta)
        self.log.info("Retuned strategy %s", strategy_id)
        return strategy_meta

    def _fitness_score(self, metrics: Dict[str, float]) -> float:
        """Composite fitness score for ranking strategies."""
        if metrics is None:
            return -1000.0
        score = (
            metrics.get("net_pnl", 0.0) * 0.3
            + metrics.get("sharpe", 0.0) * 1000 * 0.3
            + metrics.get("expectancy", 0.0) * 100 * 0.2
            + metrics.get("win_rate", 0.0) * 100 * 0.1
            - abs(metrics.get("max_dd", 0.0)) * 100 * 0.1
        )
        # Penalize strategies with too few trades
        if metrics.get("trades", 0) < 5:
            score *= 0.1
        return score

    def _is_viable(
        self, metrics: Dict[str, float], min_trades: int, min_sharpe: float
    ) -> bool:
        """Check if strategy meets minimum viability criteria."""
        return (
            metrics.get("trades", 0) >= min_trades
            and metrics.get("sharpe", 0.0) >= min_sharpe
            and metrics.get("expectancy", 0.0) > 0
            and metrics.get("max_dd", 0.0) > -0.3
        )

    def _save_strategy(self, strategy_meta: Dict[str, Any]):
        """Save strategy to disk and register it."""
        strategy_id = strategy_meta["id"]
        sym = strategy_meta.get("symbol")
        fname = f"{sym}_{strategy_id}.json" if sym else f"{strategy_id}.json"
        strategy_path = self.strategies_db / fname
        save_json(strategy_meta, strategy_path)

        # Register in the main registry
        registry_entry = {
            "id": strategy_id,
            "name": f"Discovered_{strategy_id[:8]}",
            "type": "RuleBasedParam",
            "active": True,
            "symbol": sym,
            "expectancy": strategy_meta.get("val_metrics", {}).get("expectancy", 0.0),
            "sharpe": strategy_meta.get("val_metrics", {}).get("sharpe", 0.0),
            "path": str(strategy_path),
        }
        self.registry.register(registry_entry)

    def load_best_strategies(self, top_n: int = 5) -> List[RuleBasedParamStrategy]:
        """Load the top N performing strategies from the registry."""
        active = self.registry.list_active()
        active.sort(key=lambda x: x.get("expectancy", 0.0), reverse=True)

        strategies = []
        for meta in active[:top_n]:
            if "path" in meta:
                strategy_data = json.loads(Path(meta["path"]).read_text())
                strat = RuleBasedParamStrategy(
                    name=meta["name"], params=strategy_data["params"]
                )
                strategies.append(strat)

        return strategies


def run_continuous_discovery(df: pd.DataFrame, cycles: int = 3):
    """
    Run discovery, pruning, and retuning in cycles to continuously improve.
    """
    discovery = StrategyDiscovery()

    for cycle in range(cycles):
        logging.info(f"Discovery cycle {cycle + 1}/{cycles}")

        # Discover new strategies
        new_strategies = discovery.discover_strategies(
            df,
            generations=3,
            population_size=10,
            elite_size=3,
            min_trades=5,
            min_sharpe=0.3,
        )

        # Prune underperforming strategies
        discovery.prune_underperforming(min_expectancy=0.0)

        # Retune top strategies
        active = discovery.registry.list_active()
        for strategy_meta in active[:3]:
            if "id" in strategy_meta:
                discovery.retune_strategy(strategy_meta["id"], df, iterations=5)

    # Return the best strategies after all cycles
    return discovery.load_best_strategies(top_n=5)
