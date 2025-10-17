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
        from eden.ml.pipeline import create_features_for_ml, get_feature_alignment
        
        # Get feature alignment for consistent training/prediction
        if self.args.ml_fix_features:
            feature_alignment = get_feature_alignment(df_features)
            self.logger.info(f"Using feature alignment with {len(feature_alignment)} features")
            X, y = create_features_for_ml(df_features, feature_alignment)
            
            # Save feature alignment for prediction consistency
            alignment_path = model_path.parent / "feature_alignment.json"
            import json
            with open(alignment_path, 'w') as f:
                json.dump(feature_alignment, f)
        else:
            X, y = create_features_for_ml(df_features)
        
        # Create models directory if it doesn't exist
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Train model directly
        from sklearn.model_selection import train_test_split, GridSearchCV
        from sklearn.metrics import roc_auc_score
        from sklearn.ensemble import RandomForestClassifier
        import joblib
        import numpy as np
        
        X_train, X_test, y_train, y_test = train_test_split(X.fillna(0.0), y, test_size=0.25, random_state=42)
        
        if self.args.ml_optimization:
            self.logger.info("🔬 Running advanced ML hyperparameter optimization")
            
            # Check if we have enough data for cross-validation
            min_class_size = min(sum(y_train == 0), sum(y_train == 1))
            
            if min_class_size >= 3:  # Minimum for 3-fold CV
                # Define parameter grid for optimization
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [6, 8, 10, None],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2],
                    'max_features': ['sqrt', 'log2']
                }
                
                # Use GridSearchCV for hyperparameter optimization
                rf = RandomForestClassifier(random_state=42, n_jobs=-1)
                cv_folds = min(3, min_class_size)
                grid_search = GridSearchCV(
                    rf, param_grid, cv=cv_folds, scoring='roc_auc', 
                    n_jobs=-1, verbose=1
                )
                
                grid_search.fit(X_train, y_train)
                model = grid_search.best_estimator_
                
                self.logger.info(f"Best parameters: {grid_search.best_params_}")
                self.logger.info(f"Best CV score: {grid_search.best_score_:.3f}")
            else:
                self.logger.warning(f"Insufficient data for CV (min class size: {min_class_size}), using default parameters")
                model = RandomForestClassifier(
                    n_estimators=100, 
                    max_depth=8,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    random_state=42,
                    n_jobs=-1
                )
            
        elif self.args.ml_extensive_optimization:
            self.logger.info("Running extensive ML optimization")
            # Use more sophisticated parameters for extensive optimization
            model = RandomForestClassifier(
                n_estimators=200, 
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        else:
            model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            
        if not self.args.ml_optimization:  # Grid search already fits the model
            model.fit(X_train, y_train)
            
        prob = model.predict_proba(X_test)[:,1]
        auc = float(roc_auc_score(y_test, prob)) if len(np.unique(y_test)) > 1 else 0.5
        
        # Save model directly
        joblib.dump(model, model_path)
        self.logger.info(f"🎯 Model trained with AUC: {auc:.3f}")
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
    
    def _generate_ml_strategies(self, df_features):
        """Generate new strategies using ML techniques"""
        if not self.args.ml_strategy_generation:
            return []
            
        self.logger.info("🧠 Generating ML-based strategies...")
        generated_strategies = []
        
        try:
            # Import strategy discovery components
            from eden.ml.discovery import StrategyDiscovery
            from eden.ml.strategy_registry import StrategyRegistry
            
            # Initialize discovery engine
            discovery = StrategyDiscovery()
            registry = StrategyRegistry()
            
            # Generate strategies with conservative parameters for production
            self.logger.info("Running strategy discovery algorithm...")
            discovered = discovery.discover_strategies(
                df_features, 
                generations=3,  # Reduced for faster execution
                population_size=8,
                elite_size=2,
                min_trades=2,
                min_sharpe=0.0
            )
            
            # Register and convert to strategy objects
            for strategy_meta in discovered:
                registry.register(strategy_meta)
                # Create a dynamic strategy wrapper
                strategy_obj = self._create_dynamic_strategy(strategy_meta)
                if strategy_obj:
                    generated_strategies.append(strategy_obj)
                    
            self.logger.info(f"✅ Generated {len(generated_strategies)} ML strategies")
            
        except ImportError:
            self.logger.warning("ML strategy discovery components not available")
        except Exception as e:
            self.logger.error(f"ML strategy generation failed: {e}")
            if self.args.debug:
                import traceback
                traceback.print_exc()
                
        return generated_strategies
    
    def _create_dynamic_strategy(self, strategy_meta):
        """Create a dynamic strategy from ML-generated metadata"""
        try:
            from eden.strategies.base import StrategyBase
            import pandas as pd
            
            class DynamicMLStrategy(StrategyBase):
                def __init__(self, meta):
                    self.name = f"ml_dynamic_{meta.get('id', 'unknown')}"
                    self.meta = meta
                    self.params_dict = meta.get('params', {})
                    
                def on_data(self, df):
                    # Simple ML-generated signal logic based on metadata
                    signals = []
                    try:
                        # Extract signal parameters from metadata
                        buy_threshold = self.params_dict.get('buy_threshold', 0.3)
                        sell_threshold = self.params_dict.get('sell_threshold', 0.7)
                        
                        # Use RSI as base signal (can be enhanced with ML model)
                        rsi = df.get('rsi_14', pd.Series(index=df.index, dtype=float))
                        
                        for ts in df.index:
                            if ts in rsi.index and pd.notna(rsi[ts]):
                                rsi_val = rsi[ts]
                                if rsi_val < (buy_threshold * 100):  # Convert to RSI scale
                                    signals.append({
                                        "timestamp": ts,
                                        "side": "buy",
                                        "confidence": min(0.8, 1 - (rsi_val / 100))
                                    })
                                elif rsi_val > (sell_threshold * 100):
                                    signals.append({
                                        "timestamp": ts,
                                        "side": "sell",
                                        "confidence": min(0.8, rsi_val / 100)
                                    })
                    except Exception as e:
                        self.logger.error(f"Dynamic strategy signal generation failed: {e}")
                        
                    return pd.DataFrame(signals, columns=["timestamp", "side", "confidence"])
                
                def params(self):
                    return self.params_dict
                    
            return DynamicMLStrategy(strategy_meta)
            
        except Exception as e:
            self.logger.error(f"Dynamic strategy creation failed: {e}")
            return None
    
    def _fetch_comprehensive_data(self, symbol, start_date, end_date, timeframes):
        """Fetch comprehensive historical data across multiple timeframes"""
        if not self.args.fetch_full_historical_data:
            return None
            
        self.logger.info(f"📊 Fetching comprehensive historical data from {start_date} to {end_date}")
        
        try:
            from eden.data.loader import DataLoader
            import yfinance as yf
            import pandas as pd
            from datetime import datetime
            
            dl = DataLoader(cache_dir=Path("data/comprehensive_cache"))
            
            # Try multiple data sources for comprehensive coverage
            data_sources = ['yfinance', 'mt5', 'stooq']
            
            for source in data_sources:
                try:
                    if source == 'yfinance':
                        # Map VIX100 to available ticker
                        ticker_map = {
                            'Volatility 100 Index': '^VIX',  # Use VIX as proxy
                            'VIX100': '^VIX'
                        }
                        ticker = ticker_map.get(symbol, '^VIX')
                        
                        self.logger.info(f"Fetching {symbol} data via yfinance ({ticker})...")
                        stock = yf.Ticker(ticker)
                        
                        # Get multiple timeframes
                        periods = {
                            '1d': start_date,
                            '1h': '2y',  # Last 2 years for intraday
                            '5m': '60d'  # Last 60 days for minute data
                        }
                        
                        df_daily = stock.history(start=start_date, end=end_date, interval='1d')
                        if not df_daily.empty:
                            # Convert to expected format
                            df_daily = df_daily.rename(columns={
                                'Open': 'open', 'High': 'high', 'Low': 'low', 
                                'Close': 'close', 'Volume': 'volume'
                            })
                            df_daily.index = pd.to_datetime(df_daily.index)
                            
                            self.logger.info(f"✅ Fetched {len(df_daily)} daily records from yfinance")
                            return df_daily
                            
                except Exception as e:
                    self.logger.warning(f"Failed to fetch from {source}: {e}")
                    continue
                    
            # Fallback to generating synthetic data
            self.logger.warning("All data sources failed, generating synthetic data...")
            return self._generate_synthetic_data(symbol, start_date, end_date)
            
        except Exception as e:
            self.logger.error(f"Comprehensive data fetch failed: {e}")
            return None
    
    def _generate_synthetic_data(self, symbol, start_date, end_date):
        """Generate synthetic market data for testing"""
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        self.logger.info("📈 Generating synthetic market data for comprehensive testing...")
        
        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)
        
        # Generate daily data points
        dates = pd.date_range(start=start, end=end, freq='D')
        np.random.seed(42)  # For reproducible results
        
        # Generate realistic price movements
        initial_price = 12.50
        returns = np.random.normal(0.0005, 0.02, len(dates))  # 0.05% daily return with 2% volatility
        prices = [initial_price]
        
        for ret in returns:
            new_price = prices[-1] * (1 + ret)
            prices.append(max(0.1, new_price))  # Ensure positive prices
            
        prices = prices[1:]  # Remove initial price
        
        # Generate OHLC data
        data = []
        for i, (date, close) in enumerate(zip(dates, prices)):
            high = close * (1 + abs(np.random.normal(0, 0.01)))
            low = close * (1 - abs(np.random.normal(0, 0.01)))
            open_price = prices[i-1] if i > 0 else close
            volume = max(1000, int(np.random.normal(2000, 500)))
            
            data.append({
                'open': open_price,
                'high': max(high, close, open_price),
                'low': min(low, close, open_price),
                'close': close,
                'volume': volume
            })
            
        df = pd.DataFrame(data, index=dates)
        self.logger.info(f"✅ Generated {len(df)} synthetic data points")
        return df
    
    def _apply_feature_refinement(self, df_features):
        """Apply advanced feature refinement techniques"""
        if not self.args.feature_refinement:
            return df_features
            
        self.logger.info("🔬 Applying feature refinement techniques...")
        refinement_features = [f.strip().lower() for f in self.args.feature_refinement.split(',')]
        
        try:
            import pandas as pd
            import numpy as np
            
            refined_features = df_features.copy()
            
            if 'liquidity_sweep' in refinement_features:
                self.logger.info("Refining liquidity sweep detection...")
                # Enhanced liquidity sweep detection
                if 'high' in refined_features.columns and 'low' in refined_features.columns:
                    refined_features['liquidity_sweep_strength'] = (
                        (refined_features['high'] - refined_features['low']) / 
                        refined_features['close'] * 100
                    )
                    
            if 'fvg' in refinement_features:
                self.logger.info("Refining Fair Value Gap detection...")
                # Enhanced FVG detection
                if 'close' in refined_features.columns:
                    refined_features['fvg_momentum'] = (
                        refined_features['close'].pct_change().rolling(3).mean()
                    )
                    
            if 'htf_bias' in refinement_features:
                self.logger.info("Refining Higher Timeframe bias...")
                # Enhanced HTF bias calculation
                for col in ['1H_close', '4H_close', '1D_close']:
                    if col in refined_features.columns:
                        refined_features[f'{col}_trend'] = (
                            refined_features[col] > refined_features[col].rolling(20).mean()
                        ).astype(int)
                        
            self.logger.info(f"✅ Applied {len(refinement_features)} feature refinements")
            return refined_features
            
        except Exception as e:
            self.logger.error(f"Feature refinement failed: {e}")
            return df_features
    
    def _optimize_strategy_thresholds(self, strategies, df_features):
        """Optimize strategy confidence thresholds for maximum profitability"""
        if not self.args.strategy_threshold_optimizer:
            return strategies
            
        self.logger.info("⚡ Optimizing strategy confidence thresholds...")
        
        try:
            from eden.backtest.engine import BacktestEngine
            from eden.backtest.analyzer import Analyzer
            
            optimized_strategies = []
            
            for strategy in strategies:
                best_threshold = 0.6  # Default
                best_sharpe = -999
                
                # Test different thresholds
                test_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                
                for threshold in test_thresholds:
                    try:
                        # Create test signals with modified confidence threshold
                        signals = strategy.on_data(df_features)
                        if signals is not None and not signals.empty:
                            # Filter by threshold
                            test_signals = signals[signals.get('confidence', 1.0) >= threshold].copy()
                            
                            if len(test_signals) > 5:  # Minimum trades for valid test
                                # Quick backtest
                                engine = BacktestEngine(starting_cash=15.0)
                                trades = engine.run(df_features, test_signals, symbol="VIX100")
                                
                                if trades:
                                    analyzer = Analyzer(trades)
                                    metrics = analyzer.metrics()
                                    sharpe = metrics.get('sharpe', -999)
                                    
                                    if sharpe > best_sharpe:
                                        best_sharpe = sharpe
                                        best_threshold = threshold
                                        
                    except Exception:
                        continue
                        
                # Apply optimized threshold
                if hasattr(strategy, 'min_confidence'):
                    strategy.min_confidence = best_threshold
                    
                self.logger.info(f"Strategy {strategy.name}: optimized threshold = {best_threshold:.2f} (Sharpe: {best_sharpe:.2f})")
                optimized_strategies.append(strategy)
                
            return optimized_strategies
            
        except Exception as e:
            self.logger.error(f"Strategy threshold optimization failed: {e}")
            return strategies
    
    def _apply_adaptive_weighting(self, strategies, df_features):
        """Apply adaptive weighting based on historical performance"""
        if not (self.args.adaptive_strategy_weighting or self.args.ml_strategy_weighting):
            return strategies
            
        self.logger.info("⚖️ Applying adaptive strategy weighting...")
        
        try:
            from eden.backtest.engine import BacktestEngine
            from eden.backtest.analyzer import Analyzer
            
            # Calculate performance weights for each strategy
            strategy_weights = {}
            
            for strategy in strategies:
                try:
                    signals = strategy.on_data(df_features)
                    if signals is not None and not signals.empty:
                        # Quick performance test
                        engine = BacktestEngine(starting_cash=15.0)
                        trades = engine.run(df_features, signals, symbol="VIX100")
                        
                        if trades:
                            analyzer = Analyzer(trades)
                            metrics = analyzer.metrics()
                            
                            # Calculate composite score
                            sharpe = max(0, metrics.get('sharpe', 0))
                            profit_factor = max(0, metrics.get('profit_factor', 0))
                            win_rate = max(0, metrics.get('win_rate', 0) / 100)
                            net_pnl = metrics.get('net_pnl', 0)
                            
                            # Enhanced weighting for ML strategies
                            if self.args.ml_strategy_weighting and 'ml' in strategy.name.lower():
                                # Give ML strategies bonus weight if they perform well
                                ml_bonus = 0.3 if (sharpe > 0.5 and net_pnl > 0) else 0.0
                                score = (sharpe * 0.5 + profit_factor * 0.25 + win_rate * 0.25 + ml_bonus)
                            else:
                                # Standard weighting for other strategies
                                score = (sharpe * 0.4 + profit_factor * 0.3 + win_rate * 0.3)
                                
                            strategy_weights[strategy.name] = max(0.1, score)  # Minimum 10% weight
                        else:
                            strategy_weights[strategy.name] = 0.1
                    else:
                        strategy_weights[strategy.name] = 0.1
                        
                except Exception:
                    strategy_weights[strategy.name] = 0.1
                    
            # Normalize weights
            total_weight = sum(strategy_weights.values())
            if total_weight > 0:
                for name in strategy_weights:
                    strategy_weights[name] /= total_weight
                    
            # Apply weights to strategies
            for strategy in strategies:
                weight = strategy_weights.get(strategy.name, 0.2)
                # Store weight for signal adjustment
                strategy.adaptive_weight = weight
                self.logger.info(f"✅ Strategy {strategy.name}: weight = {weight:.3f}")
                
            return strategies
            
        except Exception as e:
            self.logger.error(f"Adaptive weighting failed: {e}")
            return strategies
    
    def _apply_daily_adaptive_tuning(self, df_features):
        """Apply daily adaptive ML tuning"""
        if not self.args.ml_daily_adaptive_tuning:
            return
            
        self.logger.info("📅 Applying daily adaptive ML tuning...")
        
        try:
            import pandas as pd
            from datetime import datetime, timedelta
            
            # Split data by days for adaptive tuning
            df_features['date'] = pd.to_datetime(df_features.index).date
            unique_dates = sorted(df_features['date'].unique())
            
            # Use recent 30 days for adaptive tuning
            recent_dates = unique_dates[-30:] if len(unique_dates) >= 30 else unique_dates
            
            for date in recent_dates[-5:]:  # Tune on last 5 days
                daily_data = df_features[df_features['date'] == date]
                if len(daily_data) > 0:
                    # Retrain ML model on daily data
                    try:
                        from eden.ml.pipeline import create_features_for_ml
                        X, y = create_features_for_ml(daily_data)
                        
                        if len(X) > 10:  # Minimum samples
                            # Quick daily model update
                            from sklearn.ensemble import RandomForestClassifier
                            import joblib
                            
                            model = RandomForestClassifier(n_estimators=50, random_state=42)
                            model.fit(X.fillna(0), y)
                            
                            # Save daily adapted model
                            models_dir = Path("models")
                            models_dir.mkdir(exist_ok=True)
                            daily_model_path = models_dir / f"daily_adapted_{date}.joblib"
                            joblib.dump(model, daily_model_path)
                            
                    except Exception as e:
                        self.logger.debug(f"Daily tuning for {date} failed: {e}")
                        
            self.logger.info("✅ Daily adaptive tuning completed")
            
        except Exception as e:
            self.logger.error(f"Daily adaptive tuning failed: {e}")
    
    def _apply_auto_ml_evolution(self, strategies, df_features):
        """Apply automatic ML strategy evolution"""
        if not self.args.auto_ml_strategy_evolution:
            return strategies
            
        self.logger.info("🧬 Applying automatic ML strategy evolution...")
        
        try:
            # Evolve strategies based on performance feedback
            evolved_strategies = strategies.copy()
            
            # Generate new strategy variants
            from eden.strategies.base import StrategyBase
            import pandas as pd
            import random
            
            class EvolvedStrategy(StrategyBase):
                def __init__(self, base_strategy, mutation_params):
                    self.name = f"evolved_{base_strategy.name}_{random.randint(1000,9999)}"
                    self.base_strategy = base_strategy
                    self.mutation_params = mutation_params
                    
                def on_data(self, df):
                    # Get base signals
                    base_signals = self.base_strategy.on_data(df)
                    if base_signals is None or base_signals.empty:
                        return base_signals
                        
                    # Apply evolutionary mutations
                    evolved_signals = base_signals.copy()
                    
                    # Mutate confidence levels
                    confidence_multiplier = self.mutation_params.get('confidence_mult', 1.0)
                    if 'confidence' in evolved_signals.columns:
                        evolved_signals['confidence'] *= confidence_multiplier
                        evolved_signals['confidence'] = evolved_signals['confidence'].clip(0.1, 0.95)
                        
                    return evolved_signals
                    
            # Create evolved variants of top performing strategies
            for strategy in strategies[:3]:  # Top 3 strategies
                try:
                    # Create mutations
                    mutations = [
                        {'confidence_mult': 1.1},  # More confident
                        {'confidence_mult': 0.9},  # Less confident
                        {'confidence_mult': 1.2},  # Much more confident
                    ]
                    
                    for mutation in mutations:
                        evolved = EvolvedStrategy(strategy, mutation)
                        evolved_strategies.append(evolved)
                        
                except Exception as e:
                    self.logger.debug(f"Evolution failed for {strategy.name}: {e}")
                    
            self.logger.info(f"✅ Generated {len(evolved_strategies) - len(strategies)} evolved strategies")
            return evolved_strategies
            
        except Exception as e:
            self.logger.error(f"Auto ML evolution failed: {e}")
            return strategies
            
    def _apply_htf_ict_bias(self, signals, df_features):
        """Apply higher timeframe ICT bias to all strategy entries"""
        self.logger.info("🔍 Applying HTF ICT bias to signals...")
        
        try:
            import pandas as pd
            
            # Calculate HTF bias indicators
            df_htf = df_features.copy()
            
            # Higher timeframe trend bias (using 4H equivalent)
            df_htf['htf_trend'] = (df_htf['close'] > df_htf['ema_200']).astype(int)
            df_htf['htf_momentum'] = df_htf['rsi_14'] > 50
            df_htf['htf_bias_score'] = (df_htf['htf_trend'] + df_htf['htf_momentum']) / 2
            
            # Apply bias filter to signals
            if not signals.empty:
                # Merge HTF bias with signals
                signals_with_bias = signals.merge(
                    df_htf[['htf_bias_score']], 
                    left_index=True, right_index=True, how='left'
                )
                
                # Filter signals based on HTF bias
                bias_threshold = 0.5
                biased_signals = signals_with_bias[
                    signals_with_bias['htf_bias_score'] >= bias_threshold
                ].copy()
                
                self.logger.info(f"✅ HTF bias applied: {len(signals)} -> {len(biased_signals)} signals")
                return biased_signals
            
            return signals
            
        except Exception as e:
            self.logger.error(f"HTF ICT bias application failed: {e}")
            return signals
    
    def _postprocess_comprehensive_metrics(self, metrics, trades, output_dir):
        """Compute comprehensive performance metrics"""
        self.logger.info("📊 Computing comprehensive performance metrics...")
        
        try:
            import pandas as pd
            import json
            
            # Enhanced metrics computation
            comprehensive_metrics = metrics.copy()
            
            if trades:
                trades_df = pd.DataFrame(trades)
                
                # Risk-adjusted returns
                daily_returns = trades_df.groupby(trades_df['entry_time'].dt.date)['pnl'].sum()
                if len(daily_returns) > 1:
                    comprehensive_metrics['volatility'] = daily_returns.std()
                    comprehensive_metrics['risk_adjusted_return'] = (
                        comprehensive_metrics.get('net_pnl', 0) / comprehensive_metrics['volatility']
                        if comprehensive_metrics['volatility'] > 0 else 0
                    )
                
                # Strategy contribution analysis
                if 'strategy' in trades_df.columns:
                    strategy_pnl = trades_df.groupby('strategy')['pnl'].sum()
                    comprehensive_metrics['strategy_contributions'] = strategy_pnl.to_dict()
                
                # HTF alignment metrics
                comprehensive_metrics['avg_trade_duration'] = (
                    (trades_df['exit_time'] - trades_df['entry_time']).mean().total_seconds() / 3600
                    if 'exit_time' in trades_df.columns else 0
                )
                
            # Save comprehensive metrics
            comp_metrics_file = output_dir / "comprehensive_metrics.json"
            with open(comp_metrics_file, 'w') as f:
                json.dump(comprehensive_metrics, f, indent=2, default=str)
                
            self.logger.info(f"✅ Comprehensive metrics saved to: {comp_metrics_file}")
            
        except Exception as e:
            self.logger.error(f"Comprehensive metrics computation failed: {e}")
    
    def _generate_enhanced_equity_curve(self, analyzer, output_dir):
        """Generate enhanced equity curve visualization"""
        self.logger.info("📈 Generating enhanced equity curve...")
        
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            
            # Create enhanced plot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Main equity curve
            equity_curve = analyzer.equity_curve
            ax1.plot(equity_curve.index, equity_curve['equity'], 'b-', linewidth=2, label='Equity')
            ax1.fill_between(equity_curve.index, equity_curve['equity'], alpha=0.3)
            ax1.set_title('Enhanced Equity Curve', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Portfolio Value ($)')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Drawdown plot
            drawdown = analyzer.drawdown_curve
            ax2.fill_between(drawdown.index, drawdown['drawdown'], 0, 
                           color='red', alpha=0.5, label='Drawdown')
            ax2.set_title('Drawdown Analysis')
            ax2.set_ylabel('Drawdown (%)')
            ax2.set_xlabel('Time')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # Format x-axis
            for ax in [ax1, ax2]:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            plt.tight_layout()
            
            # Save enhanced plot
            enhanced_plot_file = output_dir / "enhanced_equity_curve.png"
            plt.savefig(enhanced_plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"✅ Enhanced equity curve saved to: {enhanced_plot_file}")
            
        except Exception as e:
            self.logger.error(f"Enhanced equity curve generation failed: {e}")
    
    def _export_optimization_results(self, output_dir):
        """Export optimization results to JSON"""
        self.logger.info("💾 Exporting optimization results...")
        
        try:
            import json
            
            opt_results_file = output_dir / "optimization_results.json"
            with open(opt_results_file, 'w') as f:
                json.dump(self.optimization_results, f, indent=2, default=str)
                
            self.logger.info(f"✅ Optimization results exported to: {opt_results_file}")
            
        except Exception as e:
            self.logger.error(f"Optimization results export failed: {e}")
    
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
        
        # Enhanced data loading with comprehensive historical data support
        dl = DataLoader(cache_dir=Path("data/cache"))
        
        # Try comprehensive historical data fetch first
        if self.args.fetch_full_historical_data:
            timeframes = self.args.multi_timeframe.split(',') if self.args.multi_timeframe else ['M1', 'M5', '15M', '1H', '4H']
            df = self._fetch_comprehensive_data("Volatility 100 Index", self.args.start_date, self.args.end_date, timeframes)
        else:
            df = None
            
        # Check for local data if comprehensive fetch failed
        if df is None or df.empty:
            local_data_path = None
            if self.args.use_local_data_if_available and self.args.store_local_data:
                local_data_path = Path(self.args.store_local_data)
                
            if local_data_path and local_data_path.exists():
                self.logger.info(f"Using local data: {local_data_path}")
                df = dl.load_csv(local_data_path)
            else:
                self.logger.info("Fetching fresh data via standard methods")
                df = dl.get_ohlcv(
                    "Volatility 100 Index",
                    "M1",
                    self.args.start_date,
                    self.args.end_date,
                    allow_network=True,
                    prefer_mt5=self.args.mt5_online
                )
                
        if df is None or df.empty:
            self.logger.error("All data sources failed, cannot proceed")
            raise RuntimeError("Failed to load market data from any source")
            
        # Store locally if requested and not already stored
        if self.args.store_local_data and not (Path(self.args.store_local_data).exists()):
            store_path = Path(self.args.store_local_data)
            store_path.parent.mkdir(parents=True, exist_ok=True)
            df_reset = df.reset_index()
            df_reset.to_csv(store_path, index=False)
            self.logger.info(f"Data stored to: {store_path}")
        
        # Build multi-timeframe features
        self.logger.info("Building multi-timeframe features")
        # Use configured timeframes if available
        if self.args.multi_timeframe:
            htf_timeframes = [tf.strip() for tf in self.args.multi_timeframe.split(',') if tf.strip() != 'M1']
        else:
            htf_timeframes = ["M5", "M15", "1H", "4H"]
        df_features = build_mtf_features(df, "M1", htf_timeframes)
        
        # Apply feature refinement
        df_features = self._apply_feature_refinement(df_features)
        
        # Apply daily adaptive tuning
        self._apply_daily_adaptive_tuning(df_features)
        
        # Checkpoint after data processing
        self.create_git_checkpoint("data_processed", "Data loaded and features built")
        
        # Strategy selection and setup
        strategies = self._setup_strategies(df_features)
        
        # Generate ML strategies if requested
        if self.args.ml_strategy_generation:
            ml_generated_strategies = self._generate_ml_strategies(df_features)
            strategies.extend(ml_generated_strategies)
            self.logger.info(f"Added {len(ml_generated_strategies)} ML-generated strategies")
            
        # Apply advanced optimizations
        strategies = self._optimize_strategy_thresholds(strategies, df_features)
        strategies = self._apply_adaptive_weighting(strategies, df_features) 
        strategies = self._apply_auto_ml_evolution(strategies, df_features)
        
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
        
        # Apply HTF ICT bias if enabled
        if self.args.htf_ict_bias:
            self._apply_htf_ict_bias(combined_signals, df_features)
        
        # Comprehensive postprocessing if enabled
        if self.args.postprocess_metrics:
            self._postprocess_comprehensive_metrics(metrics, trades, output_dir)
            
        # Generate equity curve plot if enabled
        if self.args.equity_curve_plot:
            self._generate_enhanced_equity_curve(analyzer, output_dir)
            
        # Export optimization results if enabled
        if self.args.optimization_results_json and hasattr(self, 'optimization_results'):
            self._export_optimization_results(output_dir)
        
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
    parser.add_argument("--ml-strategy-generation", action="store_true", help="Enable ML-based strategy generation")
    parser.add_argument("--ml-daily-adaptive-tuning", action="store_true", help="Enable daily adaptive ML tuning")
    parser.add_argument("--ml-threshold", type=float, default=0.6, help="ML confidence threshold")
    parser.add_argument("--grid-optimization", action="store_true", help="Enable grid optimization")
    parser.add_argument("--backtest-strategies", type=str, help="Comma-separated list of strategies to backtest")
    parser.add_argument("--dynamic-risk-per-trade", type=float, default=0.02, help="Dynamic risk percentage per trade")
    parser.add_argument("--min-trade-value", type=float, default=0.5, help="Minimum trade value")
    parser.add_argument("--verify-strategy-functionality", action="store_true", help="Verify all strategies work before execution")
    
    # Advanced optimization features
    parser.add_argument("--feature-refinement", type=str, help="Comma-separated list of features to refine (e.g., liquidity_sweep,fvg,htf_bias)")
    parser.add_argument("--adaptive-strategy-weighting", action="store_true", help="Enable adaptive strategy weighting based on performance")
    parser.add_argument("--risk-reward-tuning", action="store_true", help="Enable risk-reward ratio tuning")
    parser.add_argument("--strategy-threshold-optimizer", action="store_true", help="Optimize strategy confidence thresholds")
    parser.add_argument("--daily-performance-feedback", action="store_true", help="Enable daily performance feedback loop")
    parser.add_argument("--auto-ml-strategy-evolution", action="store_true", help="Enable automatic ML strategy evolution")
    
    # New ML feature fix and optimization parameters
    parser.add_argument("--ml-fix-features", action="store_true", help="Automatically align ML features with strategy pipeline")
    parser.add_argument("--ml-optimization", action="store_true", help="Run advanced hyperparameter tuning for ML models")
    parser.add_argument("--ml-strategy-weighting", action="store_true", help="Dynamically adjust strategy contribution based on performance")
    parser.add_argument("--htf-ict-bias", action="store_true", help="Apply HTF bias to all strategy entries")
    parser.add_argument("--postprocess-metrics", action="store_true", help="Compute comprehensive performance metrics")
    parser.add_argument("--equity-curve-plot", action="store_true", help="Generate equity curve visualization")
    parser.add_argument("--optimization-results-json", action="store_true", help="Export optimization results to JSON")
    
    # Data management
    parser.add_argument("--mt5-online", action="store_true", help="Use MT5 for live data")
    parser.add_argument("--store-local-data", type=str, help="Path to store local data")
    parser.add_argument("--use-local-data-if-available", action="store_true", help="Use local data if available")
    parser.add_argument("--fetch-full-historical-data", action="store_true", help="Fetch comprehensive historical data")
    parser.add_argument("--multi-timeframe", type=str, help="Comma-separated list of timeframes to analyze")
    
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