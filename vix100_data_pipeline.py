#!/usr/bin/env python3
"""
VIX100 Advanced Data Pipeline System
====================================

Automated data acquisition, preprocessing, and storage system for VIX100.
Handles continuous 24/7 data collection, real-time preprocessing, and
intelligent data labeling for machine learning.

Features:
- Continuous tick data collection
- Multi-timeframe candlestick generation
- Automated pattern labeling
- Data quality validation
- Rolling window storage
- Anomaly detection in data streams

Author: Eden AI System
Version: 1.0
Date: October 13, 2025
"""

import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import sqlite3
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
import threading
from dataclasses import dataclass
from collections import deque
import time

logger = logging.getLogger(__name__)

@dataclass
class DataQualityMetrics:
    """Data quality assessment"""
    missing_ticks: int
    data_gaps: List[Tuple[datetime, datetime]]
    anomaly_count: int
    quality_score: float
    timestamp: datetime

class VIX100TickCollector:
    """Real-time VIX100 tick data collector"""
    
    def __init__(self, symbol: str = "Volatility 100 Index", buffer_size: int = 10000):
        self.symbol = symbol
        self.buffer_size = buffer_size
        self.tick_buffer = deque(maxlen=buffer_size)
        self.is_collecting = False
        self.last_tick_time = None
        self.collection_thread = None
        
    def start_collection(self):
        """Start real-time tick collection"""
        if self.is_collecting:
            return
            
        self.is_collecting = True
        self.collection_thread = threading.Thread(target=self._collect_ticks)
        self.collection_thread.daemon = True
        self.collection_thread.start()
        logger.info("Started VIX100 tick collection")
    
    def stop_collection(self):
        """Stop tick collection"""
        self.is_collecting = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
        logger.info("Stopped VIX100 tick collection")
    
    def _collect_ticks(self):
        """Internal tick collection loop"""
        while self.is_collecting:
            try:
                # Get latest tick
                tick = mt5.symbol_info_tick(self.symbol)
                if tick:
                    tick_data = {
                        'timestamp': datetime.fromtimestamp(tick.time),
                        'bid': tick.bid,
                        'ask': tick.ask,
                        'spread': tick.ask - tick.bid,
                        'volume': getattr(tick, 'volume', 1),
                        'flags': getattr(tick, 'flags', 0)
                    }
                    
                    self.tick_buffer.append(tick_data)
                    self.last_tick_time = tick_data['timestamp']
                
                time.sleep(0.1)  # 100ms interval
                
            except Exception as e:
                logger.error(f"Error collecting tick: {e}")
                time.sleep(1)
    
    def get_recent_ticks(self, count: int = 100) -> List[Dict]:
        """Get recent tick data"""
        return list(self.tick_buffer)[-count:]
    
    def get_tick_statistics(self) -> Dict:
        """Get tick collection statistics"""
        if not self.tick_buffer:
            return {}
            
        ticks = list(self.tick_buffer)
        spreads = [t['spread'] for t in ticks]
        
        return {
            'total_ticks': len(ticks),
            'avg_spread': np.mean(spreads),
            'min_spread': np.min(spreads),
            'max_spread': np.max(spreads),
            'last_tick_time': self.last_tick_time,
            'ticks_per_minute': self._calculate_tick_rate()
        }
    
    def _calculate_tick_rate(self) -> float:
        """Calculate ticks per minute"""
        if len(self.tick_buffer) < 2:
            return 0.0
            
        ticks = list(self.tick_buffer)
        time_span = (ticks[-1]['timestamp'] - ticks[0]['timestamp']).total_seconds()
        
        if time_span > 0:
            return (len(ticks) / time_span) * 60
        return 0.0

class VIX100CandlestickGenerator:
    """Generate multi-timeframe candlesticks from tick data"""
    
    def __init__(self):
        self.timeframes = {
            'M1': 60,
            'M5': 300,
            'M15': 900,
            'M30': 1800,
            'H1': 3600,
            'H4': 14400,
            'D1': 86400
        }
        self.candle_builders = {tf: CandleBuilder(interval) for tf, interval in self.timeframes.items()}
    
    def process_tick(self, tick_data: Dict) -> Dict[str, Optional[Dict]]:
        """Process tick and generate candles if complete"""
        completed_candles = {}
        
        for tf, builder in self.candle_builders.items():
            candle = builder.add_tick(tick_data)
            completed_candles[tf] = candle
            
        return completed_candles
    
    def get_current_candles(self) -> Dict[str, Dict]:
        """Get current incomplete candles"""
        return {tf: builder.get_current_candle() for tf, builder in self.candle_builders.items()}

class CandleBuilder:
    """Build candlesticks from tick data"""
    
    def __init__(self, interval_seconds: int):
        self.interval = interval_seconds
        self.current_candle = None
        self.current_period_start = None
    
    def add_tick(self, tick_data: Dict) -> Optional[Dict]:
        """Add tick to candle, return completed candle if period ended"""
        timestamp = tick_data['timestamp']
        price = (tick_data['bid'] + tick_data['ask']) / 2
        
        # Calculate period start
        period_start = self._get_period_start(timestamp)
        
        # Check if we need to start a new candle
        if self.current_period_start != period_start:
            completed_candle = self.current_candle
            self._start_new_candle(timestamp, price, tick_data['volume'])
            self.current_period_start = period_start
            return completed_candle
        
        # Update current candle
        if self.current_candle:
            self._update_candle(price, tick_data['volume'])
            
        return None
    
    def _get_period_start(self, timestamp: datetime) -> datetime:
        """Get the start of the period for this timestamp"""
        total_seconds = int(timestamp.timestamp())
        period_seconds = (total_seconds // self.interval) * self.interval
        return datetime.fromtimestamp(period_seconds)
    
    def _start_new_candle(self, timestamp: datetime, price: float, volume: int):
        """Start a new candle"""
        self.current_candle = {
            'timestamp': self.current_period_start or self._get_period_start(timestamp),
            'open': price,
            'high': price,
            'low': price,
            'close': price,
            'volume': volume,
            'tick_count': 1
        }
    
    def _update_candle(self, price: float, volume: int):
        """Update current candle with new tick"""
        self.current_candle['high'] = max(self.current_candle['high'], price)
        self.current_candle['low'] = min(self.current_candle['low'], price)
        self.current_candle['close'] = price
        self.current_candle['volume'] += volume
        self.current_candle['tick_count'] += 1
    
    def get_current_candle(self) -> Optional[Dict]:
        """Get current incomplete candle"""
        return self.current_candle.copy() if self.current_candle else None

class VIX100DataLabeler:
    """Automated pattern and behavior labeling for ML training"""
    
    def __init__(self):
        self.volatility_thresholds = {
            'low': 0.001,
            'medium': 0.005,
            'high': 0.02,
            'extreme': 0.05
        }
    
    def label_volatility_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """Label volatility events in the data"""
        df = df.copy()
        
        # Calculate volatility metrics
        df['price_change'] = df['close'].pct_change()
        df['volatility'] = df['price_change'].rolling(20).std()
        
        # Label volatility levels
        conditions = [
            df['volatility'] <= self.volatility_thresholds['low'],
            df['volatility'] <= self.volatility_thresholds['medium'],
            df['volatility'] <= self.volatility_thresholds['high'],
            df['volatility'] <= self.volatility_thresholds['extreme']
        ]
        choices = ['low', 'medium', 'high', 'extreme']
        df['volatility_level'] = np.select(conditions, choices, default='extreme')
        
        return df
    
    def label_market_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Label market patterns for strategy development"""
        df = df.copy()
        
        # Trend patterns
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        
        df['trend'] = np.where(
            (df['close'] > df['sma_20']) & (df['sma_20'] > df['sma_50']), 'uptrend',
            np.where(
                (df['close'] < df['sma_20']) & (df['sma_20'] < df['sma_50']), 'downtrend',
                'sideways'
            )
        )
        
        # Volatility patterns
        df['bb_upper'] = df['sma_20'] + df['close'].rolling(20).std() * 2
        df['bb_lower'] = df['sma_20'] - df['close'].rolling(20).std() * 2
        df['bb_squeeze'] = (df['bb_upper'] - df['bb_lower']) / df['sma_20']
        
        # Label compression/expansion
        squeeze_threshold = df['bb_squeeze'].rolling(50).quantile(0.2)
        df['compression'] = df['bb_squeeze'] < squeeze_threshold
        
        # Breakout patterns
        df['breakout_up'] = (df['close'] > df['bb_upper']) & (df['close'].shift(1) <= df['bb_upper'].shift(1))
        df['breakout_down'] = (df['close'] < df['bb_lower']) & (df['close'].shift(1) >= df['bb_lower'].shift(1))
        
        return df
    
    def label_trade_outcomes(self, df: pd.DataFrame, forward_periods: int = 50) -> pd.DataFrame:
        """Label future price movements for ML training"""
        df = df.copy()
        
        # Future returns at different horizons
        for periods in [5, 10, 20, forward_periods]:
            df[f'future_return_{periods}'] = df['close'].pct_change(periods).shift(-periods)
            
            # Binary labels for classification
            df[f'profitable_{periods}'] = (df[f'future_return_{periods}'] > 0).astype(int)
            
            # Risk-adjusted returns (considering volatility)
            vol_period = df['close'].rolling(periods).std().shift(-periods)
            df[f'risk_adjusted_return_{periods}'] = df[f'future_return_{periods}'] / vol_period
        
        return df

class VIX100DataQualityMonitor:
    """Monitor and ensure data quality"""
    
    def __init__(self, max_gap_minutes: int = 5):
        self.max_gap_minutes = max_gap_minutes
        self.quality_history = deque(maxlen=100)
    
    def assess_quality(self, df: pd.DataFrame) -> DataQualityMetrics:
        """Assess data quality and return metrics"""
        if df.empty:
            return DataQualityMetrics(0, [], 0, 0.0, datetime.now())
        
        # Check for missing data
        time_diff = df.index.to_series().diff()
        expected_interval = time_diff.median()
        large_gaps = time_diff > expected_interval * 3
        gap_count = large_gaps.sum()
        
        # Find data gaps
        gaps = []
        for i, is_gap in enumerate(large_gaps):
            if is_gap and i > 0:
                gap_start = df.index[i-1]
                gap_end = df.index[i]
                gaps.append((gap_start, gap_end))
        
        # Detect price anomalies
        price_changes = df['close'].pct_change()
        price_std = price_changes.std()
        anomalies = abs(price_changes) > price_std * 5
        anomaly_count = anomalies.sum()
        
        # Calculate quality score
        gap_penalty = min(gap_count / len(df) * 100, 50)
        anomaly_penalty = min(anomaly_count / len(df) * 100, 30)
        quality_score = max(100 - gap_penalty - anomaly_penalty, 0)
        
        metrics = DataQualityMetrics(
            missing_ticks=gap_count,
            data_gaps=gaps,
            anomaly_count=anomaly_count,
            quality_score=quality_score,
            timestamp=datetime.now()
        )
        
        self.quality_history.append(metrics)
        return metrics
    
    def get_quality_trend(self) -> Dict:
        """Get data quality trend over time"""
        if not self.quality_history:
            return {}
        
        scores = [m.quality_score for m in self.quality_history]
        return {
            'current_score': scores[-1],
            'avg_score': np.mean(scores),
            'min_score': np.min(scores),
            'max_score': np.max(scores),
            'trend': 'improving' if len(scores) > 1 and scores[-1] > scores[-5] else 'declining',
            'total_assessments': len(scores)
        }

class VIX100AdvancedPipeline:
    """Complete VIX100 data pipeline system"""
    
    def __init__(self, data_dir: str = "vix100_data", symbol: str = "Volatility 100 Index"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.symbol = symbol
        
        # Initialize components
        self.tick_collector = VIX100TickCollector(symbol)
        self.candle_generator = VIX100CandlestickGenerator()
        self.data_labeler = VIX100DataLabeler()
        self.quality_monitor = VIX100DataQualityMonitor()
        
        # Database setup
        self.db_path = self.data_dir / "vix100_pipeline.db"
        self.init_database()
        
        # Pipeline state
        self.is_running = False
        self.processed_candles = 0
        
    def init_database(self):
        """Initialize pipeline database"""
        with sqlite3.connect(self.db_path) as conn:
            # Raw tick data
            conn.execute("""
                CREATE TABLE IF NOT EXISTS raw_ticks (
                    timestamp TEXT PRIMARY KEY,
                    bid REAL,
                    ask REAL,
                    spread REAL,
                    volume INTEGER,
                    flags INTEGER
                )
            """)
            
            # Processed candles with labels
            conn.execute("""
                CREATE TABLE IF NOT EXISTS labeled_candles (
                    timestamp TEXT,
                    timeframe TEXT,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    tick_count INTEGER,
                    volatility_level TEXT,
                    trend TEXT,
                    compression BOOLEAN,
                    breakout_up BOOLEAN,
                    breakout_down BOOLEAN,
                    future_return_5 REAL,
                    future_return_10 REAL,
                    future_return_20 REAL,
                    future_return_50 REAL,
                    profitable_5 INTEGER,
                    profitable_10 INTEGER,
                    profitable_20 INTEGER,
                    profitable_50 INTEGER,
                    PRIMARY KEY (timestamp, timeframe)
                )
            """)
            
            # Data quality metrics
            conn.execute("""
                CREATE TABLE IF NOT EXISTS quality_metrics (
                    timestamp TEXT PRIMARY KEY,
                    missing_ticks INTEGER,
                    anomaly_count INTEGER,
                    quality_score REAL,
                    gaps_data TEXT
                )
            """)
    
    async def start_pipeline(self):
        """Start the complete data pipeline"""
        logger.info("Starting VIX100 Advanced Data Pipeline")
        
        # Connect to MT5
        if not mt5.initialize():
            logger.error(f"MT5 initialization failed: {mt5.last_error()}")
            return False
        
        # Start tick collection
        self.tick_collector.start_collection()
        self.is_running = True
        
        # Start main processing loop
        await self._run_pipeline()
    
    async def _run_pipeline(self):
        """Main pipeline processing loop"""
        while self.is_running:
            try:
                # Process recent ticks into candles
                recent_ticks = self.tick_collector.get_recent_ticks(1000)
                
                if recent_ticks:
                    # Store raw ticks
                    await self._store_ticks(recent_ticks)
                    
                    # Generate candles
                    for tick in recent_ticks[-100:]:  # Process last 100 ticks
                        completed_candles = self.candle_generator.process_tick(tick)
                        
                        # Process completed candles
                        for timeframe, candle in completed_candles.items():
                            if candle:
                                await self._process_completed_candle(candle, timeframe)
                
                await asyncio.sleep(10)  # Process every 10 seconds
                
            except Exception as e:
                logger.error(f"Pipeline error: {e}")
                await asyncio.sleep(30)
    
    async def _store_ticks(self, ticks: List[Dict]):
        """Store raw tick data"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                for tick in ticks[-50:]:  # Store last 50 ticks
                    conn.execute("""
                        INSERT OR REPLACE INTO raw_ticks 
                        (timestamp, bid, ask, spread, volume, flags)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        tick['timestamp'].isoformat(),
                        tick['bid'],
                        tick['ask'],
                        tick['spread'],
                        tick['volume'],
                        tick['flags']
                    ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error storing ticks: {e}")
    
    async def _process_completed_candle(self, candle: Dict, timeframe: str):
        """Process and label completed candle"""
        try:
            # Get recent candles for context
            recent_candles = await self._get_recent_candles(timeframe, 100)
            
            if len(recent_candles) < 50:
                return  # Need enough history for labeling
            
            # Add current candle
            df = pd.DataFrame(recent_candles + [candle])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # Apply labeling
            df = self.data_labeler.label_volatility_events(df)
            df = self.data_labeler.label_market_patterns(df)
            df = self.data_labeler.label_trade_outcomes(df)
            
            # Quality assessment
            quality = self.quality_monitor.assess_quality(df)
            
            # Store labeled candle
            await self._store_labeled_candle(df.iloc[-1], timeframe)
            
            # Store quality metrics
            await self._store_quality_metrics(quality)
            
            self.processed_candles += 1
            logger.debug(f"Processed {timeframe} candle #{self.processed_candles}")
            
        except Exception as e:
            logger.error(f"Error processing candle: {e}")
    
    async def _get_recent_candles(self, timeframe: str, count: int) -> List[Dict]:
        """Get recent candles from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT timestamp, open, high, low, close, volume, tick_count
                    FROM labeled_candles 
                    WHERE timeframe = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (timeframe, count))
                
                candles = []
                for row in cursor.fetchall():
                    candles.append({
                        'timestamp': row[0],
                        'open': row[1],
                        'high': row[2],
                        'low': row[3],
                        'close': row[4],
                        'volume': row[5],
                        'tick_count': row[6]
                    })
                
                return list(reversed(candles))  # Return in chronological order
        except Exception as e:
            logger.error(f"Error getting recent candles: {e}")
            return []
    
    async def _store_labeled_candle(self, candle_row: pd.Series, timeframe: str):
        """Store labeled candle in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO labeled_candles (
                        timestamp, timeframe, open, high, low, close, volume, tick_count,
                        volatility_level, trend, compression, breakout_up, breakout_down,
                        future_return_5, future_return_10, future_return_20, future_return_50,
                        profitable_5, profitable_10, profitable_20, profitable_50
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    candle_row.name.isoformat(),
                    timeframe,
                    candle_row.get('open', 0),
                    candle_row.get('high', 0),
                    candle_row.get('low', 0),
                    candle_row.get('close', 0),
                    candle_row.get('volume', 0),
                    candle_row.get('tick_count', 0),
                    candle_row.get('volatility_level', ''),
                    candle_row.get('trend', ''),
                    candle_row.get('compression', False),
                    candle_row.get('breakout_up', False),
                    candle_row.get('breakout_down', False),
                    candle_row.get('future_return_5', None),
                    candle_row.get('future_return_10', None),
                    candle_row.get('future_return_20', None),
                    candle_row.get('future_return_50', None),
                    candle_row.get('profitable_5', None),
                    candle_row.get('profitable_10', None),
                    candle_row.get('profitable_20', None),
                    candle_row.get('profitable_50', None)
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error storing labeled candle: {e}")
    
    async def _store_quality_metrics(self, quality: DataQualityMetrics):
        """Store data quality metrics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO quality_metrics 
                    (timestamp, missing_ticks, anomaly_count, quality_score, gaps_data)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    quality.timestamp.isoformat(),
                    quality.missing_ticks,
                    quality.anomaly_count,
                    quality.quality_score,
                    json.dumps([(g[0].isoformat(), g[1].isoformat()) for g in quality.data_gaps])
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error storing quality metrics: {e}")
    
    def stop_pipeline(self):
        """Stop the data pipeline"""
        logger.info("Stopping VIX100 Data Pipeline")
        self.is_running = False
        self.tick_collector.stop_collection()
        mt5.shutdown()
    
    def get_pipeline_stats(self) -> Dict:
        """Get pipeline statistics"""
        tick_stats = self.tick_collector.get_tick_statistics()
        quality_trend = self.quality_monitor.get_quality_trend()
        
        return {
            'processed_candles': self.processed_candles,
            'tick_collection': tick_stats,
            'quality_trend': quality_trend,
            'is_running': self.is_running
        }

if __name__ == "__main__":
    # Example usage
    async def main():
        pipeline = VIX100AdvancedPipeline()
        
        try:
            await pipeline.start_pipeline()
        except KeyboardInterrupt:
            pipeline.stop_pipeline()
            logger.info("Pipeline stopped by user")
    
    asyncio.run(main())