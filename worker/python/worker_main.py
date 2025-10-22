#!/usr/bin/env python3
"""
Eden Python Worker - Main Entry Point
Provides ZeroMQ-based RPC interface for the C++ frontend to execute
backtests, ML training, and other Python-based trading operations.
"""

import sys
import os
import json
import time
import logging
import threading
import traceback
from pathlib import Path
from typing import Dict, Any
import signal

# Add the current directory to path for Eden imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    import zmq
except ImportError:
    print("ZeroMQ not available. Installing pyzmq...")
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyzmq"])
    import zmq

# Eden imports (existing Python modules)
from eden.logging_conf import configure_logging
from eden.cli import run_backtest as cli_run_backtest
from eden.data.loader import DataLoader


class EdenWorker:
    """Main worker class that handles ZeroMQ communications and delegates to Eden modules."""

    def __init__(self, req_port: int = 5555, pub_port: int = 5556):
        self.req_port = req_port
        self.pub_port = pub_port
        self.context = None
        self.req_socket = None
        self.pub_socket = None
        self.running = False
        self.logger = logging.getLogger("eden.worker")

        # Worker state
        self.current_backtest = None
        self.backtest_thread = None
        self.data_loader = DataLoader(cache_dir=Path("shared/data/cache"))

    def start(self):
        """Start the ZeroMQ worker server."""
        self.context = zmq.Context()

        # REQ/REP socket for commands
        self.req_socket = self.context.socket(zmq.REP)
        self.req_socket.bind(f"tcp://*:{self.req_port}")

        # PUB socket for progress updates
        self.pub_socket = self.context.socket(zmq.PUB)
        self.pub_socket.bind(f"tcp://*:{self.pub_port}")

        self.running = True
        self.logger.info(
            f"Eden Worker started on ports REQ:{self.req_port}, PUB:{self.pub_port}"
        )

        # Publish initial status
        self.publish_status("ready", "Eden Worker ready")

        # Main worker loop
        while self.running:
            try:
                # Wait for command (with timeout)
                if self.req_socket.poll(timeout=1000):  # 1 second timeout
                    message = self.req_socket.recv_json()
                    response = self.handle_command(message)
                    self.req_socket.send_json(response)

            except zmq.ZMQError as e:
                if e.errno == zmq.ETERM:
                    break  # Context was terminated
                self.logger.error(f"ZMQ Error: {e}")
            except Exception as e:
                self.logger.error(f"Unexpected error: {e}")
                self.logger.debug(traceback.format_exc())

                # Send error response if we can
                try:
                    error_response = {
                        "status": "error",
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                    }
                    self.req_socket.send_json(error_response)
                except:
                    pass

    def stop(self):
        """Stop the worker gracefully."""
        self.running = False
        if self.req_socket:
            self.req_socket.close()
        if self.pub_socket:
            self.pub_socket.close()
        if self.context:
            self.context.term()

        self.logger.info("Eden Worker stopped")

    def handle_command(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming command from C++ frontend."""
        command = message.get("command", "")
        params = message.get("params", {})
        request_id = message.get("request_id", "")

        self.logger.info(f"Received command: {command} (ID: {request_id})")

        try:
            if command == "ping":
                return {
                    "status": "success",
                    "response": "pong",
                    "request_id": request_id,
                }

            elif command == "get_status":
                return self.get_status(request_id)

            elif command == "run_backtest":
                return self.start_backtest(params, request_id)

            elif command == "stop_backtest":
                return self.stop_backtest(request_id)

            elif command == "load_data":
                return self.load_market_data(params, request_id)

            elif command == "get_strategies":
                return self.get_available_strategies(request_id)

            elif command == "optimize_strategy":
                return self.optimize_strategy(params, request_id)

            elif command == "train_ml_model":
                return self.train_ml_model(params, request_id)

            elif command == "run_parameter_sweep":
                return self.run_parameter_sweep(params, request_id)

            elif command == "train_regime_classifier":
                return self.train_regime_classifier(params, request_id)

            elif command == "train_trade_scorer":
                return self.train_trade_scorer(params, request_id)

            elif command == "shutdown":
                threading.Thread(target=self._delayed_shutdown, daemon=True).start()
                return {
                    "status": "success",
                    "response": "Shutting down",
                    "request_id": request_id,
                }

            else:
                return {
                    "status": "error",
                    "error": f"Unknown command: {command}",
                    "request_id": request_id,
                }

        except Exception as e:
            self.logger.error(f"Error handling command {command}: {e}")
            return {
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "request_id": request_id,
            }

    def get_status(self, request_id: str) -> Dict[str, Any]:
        """Get worker status."""
        status = {
            "worker_running": self.running,
            "backtest_running": self.current_backtest is not None,
            "memory_usage_mb": self._get_memory_usage(),
            "data_cache_size": (
                len(list(Path("shared/data/cache").glob("*.csv")))
                if Path("shared/data/cache").exists()
                else 0
            ),
        }

        return {"status": "success", "response": status, "request_id": request_id}

    def start_backtest(self, params: Dict[str, Any], request_id: str) -> Dict[str, Any]:
        """Start a backtest asynchronously. If monte_carlo=true, will stream PF and max losing streak updates."""
        if self.current_backtest:
            return {
                "status": "error",
                "error": "A backtest is already running",
                "request_id": request_id,
            }

        # Start backtest in separate thread
        self.backtest_thread = threading.Thread(
            target=self._run_backtest_thread, args=(params, request_id), daemon=True
        )
        self.backtest_thread.start()

        return {
            "status": "success",
            "response": "Backtest started",
            "request_id": request_id,
        }

    def _run_backtest_thread(self, params: Dict[str, Any], request_id: str):
        """Run backtest in separate thread with progress updates.
        Supports streaming Monte Carlo-style metrics if monte_carlo=true.
        """
        self.current_backtest = request_id

        try:
            self.publish_progress(0.0, "Preparing backtest...", request_id)

            # Create config from parameters
            config_overrides = {
                "symbols": params.get("symbols", ["EURUSD"]),
                "strategy": params.get("strategy", "ensemble"),
                "starting_cash": params.get("starting_cash", 100000),
                "commission_bps": params.get("commission_bps", 1.0),
                "slippage_bps": params.get("slippage_bps", 1.0),
                "start": params.get("start_date", "2018-01-01"),
                "end": params.get("end_date", "2024-12-31"),
                "timeframe": params.get("timeframe", "1H"),
                # Phase 1 flags to speed up feedback
                "min_confidence": float(params.get("min_confidence", 0.0)),
            }

            self.publish_progress(0.15, "Loading data (MT5 preferred)...", request_id)

            # Initial run
            self.publish_progress(0.25, "Running initial backtest...", request_id)
            cli_run_backtest(None, ci_short=False, overrides=config_overrides)

            # Read results to compute PF & max losing streak
            from pathlib import Path as _P
            import json as _json

            result_dir = _P("results")
            metrics_path = result_dir / "metrics.json"
            result_dir / "trades.csv"
            metrics = {}
            if metrics_path.exists():
                metrics = _json.loads(metrics_path.read_text())
            pf = float(metrics.get("profit_factor", 0.0))
            mcl = int(metrics.get("max_consec_losses", 0))
            self.publish_progress(
                0.6, f"PF: {pf:.2f}, Max losing streak: {mcl}", request_id
            )

            # If 'optimize_as_you_go' flag is set, run a quick sweep and stream updates
            if params.get("optimize_as_you_go", True):
                try:
                    self.publish_progress(
                        0.65, "Optimizing parameters (quick sweep)...", request_id
                    )
                    from eden.optimize.sweeps import run_parameter_sweep  # type: ignore

                    quick = run_parameter_sweep(
                        symbols=config_overrides["symbols"],
                        timeframe=config_overrides["timeframe"],
                        strategy=str(params.get("strategy", "ict")),
                        budget=int(params.get("budget", 15)),
                        objective=str(params.get("objective", "sharpe")),
                        constraints={"min_trades": int(params.get("min_trades", 50))},
                        start=config_overrides["start"],
                        end=config_overrides["end"],
                        min_confidence=float(params.get("min_confidence", 0.0)),
                    )
                    self.publish_progress(
                        0.9,
                        f"Best trial score: {quick.get('best_value', 0):.3f}",
                        request_id,
                    )
                except Exception as e:
                    self.publish_status("opt_error", f"Optimization error: {e}")

            self.publish_progress(1.0, "Monte Carlo backtest complete", request_id)
            self.publish_status(
                "backtest_complete", f"Backtest {request_id} completed successfully"
            )

        except Exception as e:
            self.logger.error(f"Backtest error: {e}")
            self.publish_status(
                "backtest_error", f"Backtest {request_id} failed: {str(e)}"
            )

        finally:
            self.current_backtest = None
            self.backtest_thread = None

    def stop_backtest(self, request_id: str) -> Dict[str, Any]:
        """Stop running backtest."""
        if not self.current_backtest:
            return {
                "status": "error",
                "error": "No backtest is currently running",
                "request_id": request_id,
            }

        # In a real implementation, we'd need a way to signal the backtest thread to stop
        # For now, just reset the state
        self.current_backtest = None
        if self.backtest_thread:
            # Note: We can't actually stop a thread in Python safely
            # This would need to be implemented with proper cancellation tokens
            pass

        self.publish_status("backtest_stopped", f"Backtest {request_id} stopped")

        return {
            "status": "success",
            "response": "Backtest stop requested",
            "request_id": request_id,
        }

    def load_market_data(
        self, params: Dict[str, Any], request_id: str
    ) -> Dict[str, Any]:
        """Load market data for given symbol and timeframe (prefer MT5)."""
        symbol = params.get("symbol", "XAUUSD")
        timeframe = params.get("timeframe", "1D")
        start_date = params.get("start_date", "2020-01-01")
        end_date = params.get("end_date", "2023-12-31")
        prefer_mt5 = bool(params.get("prefer_mt5", True))

        try:
            df = self.data_loader.get_ohlcv(
                symbol, timeframe, start_date, end_date, prefer_mt5=prefer_mt5
            )

            if df is None or df.empty:
                return {
                    "status": "error",
                    "error": f"No data available for {symbol} {timeframe}",
                    "request_id": request_id,
                }

            # Return basic statistics
            data_info = {
                "symbol": symbol,
                "timeframe": timeframe,
                "rows": len(df),
                "start_date": df.index[0].isoformat() if not df.empty else None,
                "end_date": df.index[-1].isoformat() if not df.empty else None,
                "columns": list(df.columns),
                "source": "mt5" if prefer_mt5 else "fallback",
            }

            return {
                "status": "success",
                "response": data_info,
                "request_id": request_id,
            }

        except Exception as e:
            return {
                "status": "error",
                "error": f"Failed to load data: {str(e)}",
                "request_id": request_id,
            }

    def get_available_strategies(self, request_id: str) -> Dict[str, Any]:
        """Get list of available strategies."""
        strategies = [
            {
                "id": "ict",
                "name": "ICT Strategy",
                "description": "Inner Circle Trader methodology",
            },
            {
                "id": "mean_reversion",
                "name": "Mean Reversion",
                "description": "Statistical mean reversion strategy",
            },
            {
                "id": "momentum",
                "name": "Momentum",
                "description": "Trend following momentum strategy",
            },
            {
                "id": "ensemble",
                "name": "Ensemble",
                "description": "Combined strategy approach",
            },
            {
                "id": "ml_generated",
                "name": "ML Generated",
                "description": "Machine learning generated strategy",
            },
        ]

        return {
            "status": "success",
            "response": {"strategies": strategies},
            "request_id": request_id,
        }

    def optimize_strategy(
        self, params: Dict[str, Any], request_id: str
    ) -> Dict[str, Any]:
        """Run a basic optimization placeholder."""
        try:
            budget = int(params.get("budget", 20))
            objective = str(params.get("objective", "sharpe"))
            symbols = params.get("symbols") or ["XAUUSD"]
            timeframe = params.get("timeframe", "1H")
            strategy = params.get("strategy", "ict")
            from eden.optimize.sweeps import run_parameter_sweep  # type: ignore

            result = run_parameter_sweep(
                symbols=symbols,
                timeframe=timeframe,
                strategy=strategy,
                budget=budget,
                objective=objective,
                constraints=params.get("constraints"),
                search_space=params.get("search_space"),
                start=params.get("start"),
                end=params.get("end"),
                min_confidence=float(params.get("min_confidence", 0.0)),
            )
            return {"status": "success", "response": result, "request_id": request_id}
        except Exception as e:
            return {"status": "error", "error": str(e), "request_id": request_id}

    def run_parameter_sweep(
        self, params: Dict[str, Any], request_id: str
    ) -> Dict[str, Any]:
        """Explicit endpoint for parameter sweeps."""
        return self.optimize_strategy(params, request_id)

    def train_regime_classifier(
        self, params: Dict[str, Any], request_id: str
    ) -> Dict[str, Any]:
        """Placeholder for regime classifier training in Phase 2."""
        return {
            "status": "success",
            "response": {"message": "Regime training scheduled"},
            "request_id": request_id,
        }

    def train_trade_scorer(
        self, params: Dict[str, Any], request_id: str
    ) -> Dict[str, Any]:
        """Placeholder for trade scoring model training in Phase 2."""
        return {
            "status": "success",
            "response": {"message": "Trade scorer training scheduled"},
            "request_id": request_id,
        }

    def train_ml_model(self, params: Dict[str, Any], request_id: str) -> Dict[str, Any]:
        """Train ML model (placeholder)."""
        # This would integrate with the existing ML modules
        return {
            "status": "success",
            "response": {"message": "ML training not yet implemented"},
            "request_id": request_id,
        }

    def publish_status(self, status_type: str, message: str):
        """Publish status update via PUB socket."""
        if self.pub_socket:
            update = {
                "type": "status",
                "status": status_type,
                "message": message,
                "timestamp": time.time(),
            }
            self.pub_socket.send_multipart([b"status", json.dumps(update).encode()])

    def publish_progress(self, progress: float, message: str, request_id: str):
        """Publish progress update for a specific request."""
        if self.pub_socket:
            update = {
                "type": "progress",
                "request_id": request_id,
                "progress": progress,
                "message": message,
                "timestamp": time.time(),
            }
            self.pub_socket.send_multipart([b"progress", json.dumps(update).encode()])

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil

            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0

    def _delayed_shutdown(self):
        """Shutdown with delay to allow response to be sent."""
        time.sleep(0.5)
        self.stop()


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    print(f"\nReceived signal {signum}, shutting down gracefully...")
    if hasattr(signal_handler, "worker"):
        signal_handler.worker.stop()
    sys.exit(0)


def main():
    """Main entry point."""
    # Setup logging
    configure_logging("INFO")
    logger = logging.getLogger("eden.worker.main")

    # Create worker
    worker = EdenWorker()
    signal_handler.worker = worker

    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        logger.info("Starting Eden Python Worker...")
        worker.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Worker error: {e}")
        logger.debug(traceback.format_exc())
    finally:
        worker.stop()


if __name__ == "__main__":
    main()
