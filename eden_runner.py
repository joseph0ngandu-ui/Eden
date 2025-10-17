#!/usr/bin/env python3
"""
Eden Advanced CLI Runner
Comprehensive trading system runner with ML, ICT, and advanced monitoring capabilities
"""
import argparse
import json
import logging
import os
import sys
import time
import psutil
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

# Add worker/python to Python path
sys.path.insert(0, str(Path(__file__).parent / "worker" / "python"))

def setup_logging(log_file: Optional[str] = None, debug: bool = False, verbose: bool = False) -> logging.Logger:
    """Setup comprehensive logging"""
    level = logging.DEBUG if debug else (logging.INFO if verbose else logging.WARNING)
    
    handlers = [logging.StreamHandler()]
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    return logging.getLogger('eden_runner')

class SystemMonitor:
    """Monitor system resources and subprocesses"""
    
    def __init__(self, monitor_cpu: bool = False, monitor_subprocesses: bool = False, notify_on_stall: bool = False):
        self.monitor_cpu = monitor_cpu
        self.monitor_subprocesses = monitor_subprocesses
        self.notify_on_stall = notify_on_stall
        self.running = False
        self.logger = logging.getLogger('eden_runner.monitor')
        
    def start_monitoring(self):
        """Start system monitoring in background thread"""
        if not (self.monitor_cpu or self.monitor_subprocesses):
            return
            
        self.running = True
        monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        monitor_thread.start()
        self.logger.info("System monitoring started")
        
    def stop_monitoring(self):
        """Stop system monitoring"""
        self.running = False
        self.logger.info("System monitoring stopped")
        
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                if self.monitor_cpu:
                    cpu_percent = psutil.cpu_percent(interval=1)
                    memory = psutil.virtual_memory()
                    self.logger.debug(f"CPU: {cpu_percent}%, Memory: {memory.percent}%")
                    
                    if cpu_percent > 90 or memory.percent > 90:
                        self.logger.warning(f"High resource usage - CPU: {cpu_percent}%, Memory: {memory.percent}%")
                        
                if self.monitor_subprocesses:
                    current_process = psutil.Process()
                    children = current_process.children(recursive=True)
                    if children:
                        self.logger.debug(f"Active subprocesses: {len(children)}")
                        
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                
            time.sleep(5)  # Monitor every 5 seconds

class EdenRunner:
    """Advanced Eden trading system runner"""
    
    def __init__(self, args):
        self.args = args
        self.logger = setup_logging(args.log_file, args.debug, args.verbose)
        self.monitor = SystemMonitor(args.monitor_cpu, args.monitor_subprocesses, args.notify_on_stall)
        self.start_time = time.time()
        
    def validate_mt5_connection(self) -> bool:
        """Validate MT5 connection if required"""
        if not self.args.mt5_online:
            return True
            
        try:
            import MetaTrader5 as mt5
            if not mt5.initialize():
                self.logger.error(f"MT5 initialization failed: {mt5.last_error()}")
                return False
            self.logger.info("MT5 connection validated")
            return True
        except ImportError:
            self.logger.error("MetaTrader5 module not available")
            return False
            
    def setup_data_storage(self):
        """Setup local data storage paths"""
        if self.args.store_local_data:
            data_path = Path(self.args.store_local_data)
            data_path.parent.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Data storage path: {data_path}")
            
    def check_timeout(self):
        """Check if timeout has been exceeded"""
        if self.args.timeout and (time.time() - self.start_time) > self.args.timeout:
            self.logger.error(f"Timeout exceeded ({self.args.timeout}s)")
            return True
        return False
    
    def create_git_checkpoint(self, stage: str, message_suffix: str = ""):
        """Create a git checkpoint if enabled"""
        if not self.args.git_checkpoints:
            return
            
        try:
            import subprocess
            from datetime import datetime
            
            # Check if we're in a git repository
            result = subprocess.run(['git', 'rev-parse', '--git-dir'], 
                                   capture_output=True, text=True, cwd='.')
            if result.returncode != 0:
                self.logger.warning("Not in a git repository, skipping checkpoint")
                return
            
            # Add all changes
            subprocess.run(['git', 'add', '.'], cwd='.', capture_output=True)
            
            # Create commit message
            if self.args.git_commit_message:
                # Replace $(date ...) pattern with actual date
                import re
                date_pattern = r"\$\(date \+'([^']+)'\)"
                message = self.args.git_commit_message
                
                def replace_date(match):
                    format_str = match.group(1)
                    # Convert to Python datetime format
                    py_format = format_str.replace('%Y', '%Y').replace('%m', '%m').replace('%d', '%d')
                    py_format = py_format.replace('%H', '%H').replace('%M', '%M').replace('%S', '%S')
                    return datetime.now().strftime(py_format)
                
                message = re.sub(date_pattern, replace_date, message)
            else:
                message = f"Eden checkpoint: {stage} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                
            if message_suffix:
                message += f" - {message_suffix}"
            
            # Create commit
            result = subprocess.run(['git', 'commit', '-m', message], 
                                   cwd='.', capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info(f"🔄 Git checkpoint created: {stage}")
                self.logger.debug(f"Commit message: {message}")
            else:
                self.logger.warning(f"Git commit failed: {result.stderr.strip()}")
                
        except Exception as e:
            self.logger.error(f"Git checkpoint failed: {e}")
            if self.args.debug:
                import traceback
                traceback.print_exc()
    
    def _setup_strategies(self, df_features):
        """Setup strategies based on configuration"""
        strategies = []
        
        # Import strategy modules
        from eden.strategies.ict import ICTStrategy
        from eden.strategies.ml_generated import MLGeneratedStrategy
        from eden.strategies.momentum import MomentumStrategy
        from eden.strategies.mean_reversion import MeanReversionStrategy
        from eden.strategies.price_action import PriceActionStrategy
        
        # Parse strategy list if provided
        if self.args.backtest_strategies:
            strategy_names = [s.strip().lower() for s in self.args.backtest_strategies.split(',')]
            self.logger.info(f"Requested strategies: {strategy_names}")
        else:
            strategy_names = []
            
        # ML model training/loading if ML is enabled
        if self.args.ml_enabled or 'ml_generated' in strategy_names:
            self.logger.info("Setting up ML components")
            
            # Train or retrain model if needed
            model_path = Path("models/sample_model.joblib")
            if self.args.retrain_if_stuck or not model_path.exists():
                self.logger.info("Training ML model")
                X, y = self._train_ml_model(df_features, model_path)
            
        # Add strategies based on configuration
        if self.args.ml_enabled or 'ml_generated' in strategy_names:
            ml_strategy = MLGeneratedStrategy()
            strategies.append(ml_strategy)
            
        if self.args.ml_ict_filter or 'ict' in strategy_names:
            self.logger.info("Adding ICT strategy")
            ict_strategy = ICTStrategy(
                min_confidence=self.args.ml_threshold,
                stop_atr_multiplier=1.2,
                tp_atr_multiplier=1.5
            )
            strategies.append(ict_strategy)
            
        if 'momentum' in strategy_names:
            self.logger.info("Adding Momentum strategy")
            strategies.append(MomentumStrategy())
            
        if 'mean_reversion' in strategy_names:
            self.logger.info("Adding Mean Reversion strategy")
            strategies.append(MeanReversionStrategy())
            
        if 'price_action' in strategy_names:
            self.logger.info("Adding Price Action strategy")
            strategies.append(PriceActionStrategy())
            
        # Fallback to default if no strategies configured
        if not strategies:
            self.logger.warning("No strategies configured, adding default momentum strategy")
            strategies.append(MomentumStrategy())
            
        return strategies
    
    def _train_ml_model(self, df_features, model_path):
        """Train ML model with extensive optimization if requested"""
        from eden.ml.pipeline import create_features_for_ml
        X, y = create_features_for_ml(df_features)
        # Create models directory if it doesn't exist
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Train model directly
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score
        from sklearn.ensemble import RandomForestClassifier
        import joblib
        import numpy as np
        
        X_train, X_test, y_train, y_test = train_test_split(X.fillna(0.0), y, test_size=0.25, random_state=42)
        
        if self.args.ml_extensive_optimization:
            self.logger.info("Running extensive ML optimization")
            # Use more sophisticated parameters for extensive optimization
            model = RandomForestClassifier(
                n_estimators=200, 
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
        else:
            model = RandomForestClassifier(n_estimators=50, random_state=42)
            
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_test)[:,1]
        auc = float(roc_auc_score(y_test, prob)) if len(np.unique(y_test)) > 1 else 0.5
        
        # Save model directly
        joblib.dump(model, model_path)
        self.logger.info(f"Model trained with AUC: {auc:.3f}")
        return X, y
    
    def _verify_strategies(self, strategies, df_features):
        """Verify that all strategies can generate signals"""
        self.logger.info("Verifying strategy functionality...")
        verified_strategies = []
        
        for strategy in strategies:
            try:
                signals = strategy.on_data(df_features)
                if signals is not None and not signals.empty:
                    self.logger.info(f"✅ Strategy {strategy.name} verified: {len(signals)} potential signals")
                    verified_strategies.append(strategy)
                else:
                    self.logger.warning(f"⚠️ Strategy {strategy.name} produced no signals")
                    if not self.args.auto_retry:
                        verified_strategies.append(strategy)  # Include anyway if not auto-retry
            except Exception as e:
                self.logger.error(f"❌ Strategy {strategy.name} failed verification: {e}")
                if self.args.debug:
                    import traceback
                    traceback.print_exc()
                    
        if not verified_strategies:
            self.logger.error("No strategies passed verification")
            from eden.strategies.momentum import MomentumStrategy
            verified_strategies.append(MomentumStrategy())
            self.logger.info("Added fallback Momentum strategy")
            
        return verified_strategies
    
    def _run_grid_optimization(self, df_features, strategies, output_dir):
        """Run grid optimization on strategy parameters"""
        self.logger.info("Running grid optimization...")
        
        try:
            from eden.optimize.optimizer import run_grid_search
            
            optimization_results = []
            for strategy in strategies:
                if hasattr(strategy, 'params') and strategy.params():
                    self.logger.info(f"Optimizing strategy: {strategy.name}")
                    best_params, best_metrics = run_grid_search(
                        df_features, 
                        "VIX100", 
                        "M1", 
                        budget=20,  # Reduced budget for faster execution
                        precomputed_feat=df_features
                    )
                    optimization_results.append({
                        'strategy': strategy.name,
                        'best_params': best_params,
                        'best_metrics': best_metrics
                    })
                    self.logger.info(f"Best params for {strategy.name}: {best_params}")
                    
            # Save optimization results
            if optimization_results:
                import json
                opt_file = output_dir / "optimization_results.json"
                with open(opt_file, 'w') as f:
                    json.dump(optimization_results, f, indent=2)
                self.logger.info(f"Optimization results saved to: {opt_file}")
                    
        except Exception as e:
            self.logger.error(f"Grid optimization failed: {e}")
            if self.args.debug:
                import traceback
                traceback.print_exc()
        
    def run_phase3_ml_ict(self):
        """Run Phase 3 ML-enabled ICT strategy"""
        self.logger.info("Starting Phase 3 ML ICT Run")
        
        # Create initial checkpoint
        self.create_git_checkpoint("phase3_start", "Starting Phase 3 ML ICT execution")
        
        # Import Eden components
        from eden.data.loader import DataLoader
        from eden.features.feature_pipeline import build_mtf_features
        from eden.backtest.engine import BacktestEngine
        from eden.backtest.analyzer import Analyzer
        from eden.strategies.ict import ICTStrategy
        from eden.strategies.ml_generated import MLGeneratedStrategy
        from eden.ml.pipeline import train_model, create_features_for_ml
        
        # Setup output directory
        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data loading with local cache support
        dl = DataLoader(cache_dir=Path("data/cache"))
        
        # Check for local data first
        local_data_path = None
        if self.args.use_local_data_if_available and self.args.store_local_data:
            local_data_path = Path(self.args.store_local_data)
            
        if local_data_path and local_data_path.exists():
            self.logger.info(f"Using local data: {local_data_path}")
            df = dl.load_csv(local_data_path)
        else:
            self.logger.info("Fetching fresh data")
            df = dl.get_ohlcv(
                "Volatility 100 Index",
                "M1",
                self.args.start_date,
                self.args.end_date,
                allow_network=True,
                prefer_mt5=self.args.mt5_online
            )
            
            if df is None or df.empty:
                raise RuntimeError("Failed to load market data")
                
            # Store locally if requested
            if self.args.store_local_data:
                store_path = Path(self.args.store_local_data)
                store_path.parent.mkdir(parents=True, exist_ok=True)
                df_reset = df.reset_index()
                df_reset.to_csv(store_path, index=False)
                self.logger.info(f"Data stored to: {store_path}")
        
        # Build multi-timeframe features
        self.logger.info("Building multi-timeframe features")
        htf_timeframes = ["M5", "M15", "1H", "4H"]
        df_features = build_mtf_features(df, "M1", htf_timeframes)
        
        # Checkpoint after data processing
        self.create_git_checkpoint("data_processed", "Data loaded and features built")
        
        # Strategy selection and setup
        strategies = self._setup_strategies(df_features)
        
        # Verify strategy functionality if requested
        if self.args.verify_strategy_functionality:
            strategies = self._verify_strategies(strategies, df_features)
            
        # Checkpoint after strategy setup
        self.create_git_checkpoint("strategies_ready", f"Strategies configured: {[s.name for s in strategies]}")
        
        # Run grid optimization if requested
        if self.args.grid_optimization:
            self._run_grid_optimization(df_features, strategies, output_dir)
            # Checkpoint after optimization
            self.create_git_checkpoint("optimization_complete", "Grid optimization completed")
        
        # Setup backtest engine with dynamic risk
        engine = BacktestEngine(
            starting_cash=15.0,  # Micro account default
            per_order_risk_fraction=self.args.dynamic_risk_per_trade,
            min_trade_value=self.args.min_trade_value,
            commission_bps=1.0,
            slippage_bps=1.0
        )
        
        # Generate and combine signals
        all_signals = []
        for strategy in strategies:
            self.logger.info(f"Running strategy: {strategy.name}")
            
            if self.check_timeout():
                break
                
            try:
                signals = strategy.on_data(df_features)
                if signals is not None and not signals.empty:
                    # Apply ML threshold filtering
                    if hasattr(signals, 'confidence'):
                        signals = signals[signals['confidence'] >= self.args.ml_threshold]
                    signals['strategy'] = strategy.name
                    all_signals.append(signals)
                    self.logger.info(f"Generated {len(signals)} signals from {strategy.name}")
            except Exception as e:
                self.logger.error(f"Strategy {strategy.name} failed: {e}")
                if self.args.auto_retry:
                    self.logger.info(f"Retrying strategy {strategy.name}")
                    time.sleep(1)
                    try:
                        signals = strategy.on_data(df_features)
                        if signals is not None and not signals.empty:
                            signals['strategy'] = strategy.name
                            all_signals.append(signals)
                    except Exception as e2:
                        self.logger.error(f"Retry failed for {strategy.name}: {e2}")
        
        if not all_signals:
            self.logger.error("No signals generated from any strategy")
            return
            
        # Combine all signals
        import pandas as pd
        combined_signals = pd.concat(all_signals, ignore_index=True)
        self.logger.info(f"Total combined signals: {len(combined_signals)}")
        
        # Run backtest
        self.logger.info("Running backtest")
        trades = engine.run(df_features, combined_signals, symbol="VIX100")
        
        # Analysis and results
        analyzer = Analyzer(trades, starting_cash=15.0)
        metrics = analyzer.metrics()
        
        # Save results
        results_file = output_dir / "metrics.json"
        with open(results_file, 'w') as f:
            json.dump(metrics, f, indent=2)
            
        trades_file = output_dir / "trades.csv"
        engine.save_trades_csv(trades_file)
        
        equity_file = output_dir / "equity_curve.png"
        analyzer.plot_equity_curve(save_path=equity_file)
        
        # Log summary
        self.logger.info("=" * 60)
        self.logger.info("PHASE 3 ML ICT RESULTS")
        self.logger.info("=" * 60)
        self.logger.info(f"Net PnL: ${metrics.get('net_pnl', 0):.2f}")
        self.logger.info(f"Total Trades: {metrics.get('trades', 0)}")
        self.logger.info(f"Win Rate: {metrics.get('win_rate', 0):.1f}%")
        self.logger.info(f"Sharpe Ratio: {metrics.get('sharpe', 0):.2f}")
        self.logger.info(f"Max Drawdown: {metrics.get('max_drawdown_pct', 0):.1f}%")
        self.logger.info(f"Results saved to: {output_dir}")
        self.logger.info("=" * 60)
        
        # Final checkpoint with results
        self.create_git_checkpoint("phase3_complete", f"Phase 3 completed - PnL: ${metrics.get('net_pnl', 0):.2f}, Trades: {metrics.get('trades', 0)}")
        
    def run(self):
        """Main runner execution"""
        try:
            self.monitor.start_monitoring()
            
            if self.args.safe_mode:
                self.logger.info("Running in safe mode")
                
            # Validate MT5 if needed
            if self.args.mt5_online and not self.validate_mt5_connection():
                if not self.args.auto_retry:
                    raise RuntimeError("MT5 connection failed")
                self.logger.warning("MT5 connection failed, continuing without MT5")
                
            # Setup data storage
            self.setup_data_storage()
            
            # Run the main phase 3 process
            if self.args.phase3:
                self.run_phase3_ml_ict()
            else:
                self.logger.error("No valid phase specified")
                return 1
                
            self.logger.info(f"Execution completed in {time.time() - self.start_time:.1f}s")
            return 0
            
        except Exception as e:
            self.logger.error(f"Execution failed: {e}")
            if self.args.debug:
                import traceback
                traceback.print_exc()
            return 1
        finally:
            self.monitor.stop_monitoring()

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="Eden Advanced Trading System Runner")
    
    # Phase selection
    parser.add_argument("--phase3", action="store_true", help="Run Phase 3 ML ICT strategy")
    parser.add_argument("--mvp", action="store_true", help="Run MVP configuration")
    
    # Data and time parameters
    parser.add_argument("--start-date", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--incremental-days", action="store_true", help="Process data incrementally")
    
    # ML and strategy parameters
    parser.add_argument("--ml-enabled", action="store_true", help="Enable ML components")
    parser.add_argument("--ml-ict-filter", action="store_true", help="Apply ML ICT filtering")
    parser.add_argument("--ml-extensive-optimization", action="store_true", help="Enable extensive ML optimization")
    parser.add_argument("--ml-threshold", type=float, default=0.6, help="ML confidence threshold")
    parser.add_argument("--grid-optimization", action="store_true", help="Enable grid optimization")
    parser.add_argument("--backtest-strategies", type=str, help="Comma-separated list of strategies to backtest")
    parser.add_argument("--dynamic-risk-per-trade", type=float, default=0.02, help="Dynamic risk percentage per trade")
    parser.add_argument("--min-trade-value", type=float, default=0.5, help="Minimum trade value")
    parser.add_argument("--verify-strategy-functionality", action="store_true", help="Verify all strategies work before execution")
    
    # Data management
    parser.add_argument("--mt5-online", action="store_true", help="Use MT5 for live data")
    parser.add_argument("--store-local-data", type=str, help="Path to store local data")
    parser.add_argument("--use-local-data-if-available", action="store_true", help="Use local data if available")
    
    # Output and logging
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    parser.add_argument("--log-file", type=str, help="Log file path")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    # System management
    parser.add_argument("--safe-mode", action="store_true", help="Run in safe mode")
    parser.add_argument("--auto-retry", action="store_true", help="Auto-retry on failures")
    parser.add_argument("--retrain-if-stuck", action="store_true", help="Retrain models if needed")
    parser.add_argument("--timeout", type=int, help="Timeout in seconds")
    
    # Monitoring
    parser.add_argument("--monitor-cpu", action="store_true", help="Monitor CPU usage")
    parser.add_argument("--monitor-subprocesses", action="store_true", help="Monitor subprocesses")
    parser.add_argument("--notify-on-stall", action="store_true", help="Notify on system stalls")
    
    # Version control
    parser.add_argument("--git-checkpoints", action="store_true", help="Create git checkpoints during execution")
    parser.add_argument("--git-commit-message", type=str, help="Custom git commit message template")
    
    args = parser.parse_args()
    
    # Install required packages if missing
    try:
        import psutil
    except ImportError:
        print("Installing required package: psutil")
        os.system("pip install psutil")
        import psutil
    
    runner = EdenRunner(args)
    return runner.run()

if __name__ == "__main__":
    sys.exit(main())