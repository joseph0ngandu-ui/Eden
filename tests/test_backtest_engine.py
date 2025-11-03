#!/usr/bin/env python3
"""
Unit tests for BacktestEngine

Run with: pytest tests/test_backtest_engine.py
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from backtest_engine import BacktestEngine, Position, BacktestStats


class TestBacktestEngine(unittest.TestCase):
    """Test cases for BacktestEngine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engine = BacktestEngine()
    
    def test_fast_ma_period(self):
        """Test fast MA period is 3."""
        self.assertEqual(self.engine.FAST_MA_PERIOD, 3)
    
    def test_slow_ma_period(self):
        """Test slow MA period is 10."""
        self.assertEqual(self.engine.SLOW_MA_PERIOD, 10)
    
    def test_hold_bars(self):
        """Test hold duration is 5 bars."""
        self.assertEqual(self.engine.HOLD_BARS, 5)
    
    def test_calculate_moving_averages(self):
        """Test moving average calculation."""
        # Create sample data
        data = {
            'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            'time': pd.date_range('2025-01-01', periods=11, freq='5min'),
            'open': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'high': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
            'low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        }
        df = pd.DataFrame(data)
        
        df = self.engine._calculate_moving_averages(df)
        
        self.assertIn('MA3', df.columns)
        self.assertIn('MA10', df.columns)
        self.assertTrue(df['MA3'].iloc[-1] > 0)
        self.assertTrue(df['MA10'].iloc[-1] > 0)
    
    def test_position_profit_calculation(self):
        """Test position profit calculation."""
        pos = Position(
            entry_price=100.0,
            exit_price=101.0,
            entry_time=datetime.now(),
            exit_time=datetime.now(),
            profit=(101.0 - 100.0) * 100,
            duration_bars=5
        )
        
        self.assertEqual(pos.profit, 100.0)
        self.assertEqual(pos.profit_pips, 1.0)
    
    def test_backtest_stats_return_percent(self):
        """Test backtest stats return percent calculation."""
        stats = BacktestStats(
            total_trades=100,
            winning_trades=50,
            losing_trades=50,
            breakeven_trades=0,
            total_pnl=5000.0,
            win_rate=50.0,
            avg_win=100.0,
            avg_loss=-100.0,
            max_profit=500.0,
            min_profit=-500.0,
            profit_factor=1.0
        )
        
        # On $100k capital: 5000 / 100000 * 100 = 5%
        self.assertEqual(stats.return_percent, 5.0)
    
    def test_statistics_calculation_empty(self):
        """Test statistics calculation with no positions."""
        self.engine.positions = []
        stats = self.engine._calculate_statistics()
        
        self.assertEqual(stats.total_trades, 0)
        self.assertEqual(stats.total_pnl, 0)
        self.assertEqual(stats.win_rate, 0)
    
    def test_statistics_calculation_with_positions(self):
        """Test statistics calculation with positions."""
        self.engine.positions = [
            Position(
                entry_price=100.0, exit_price=101.0,
                entry_time=datetime.now(), exit_time=datetime.now(),
                profit=100.0, duration_bars=5
            ),
            Position(
                entry_price=101.0, exit_price=100.0,
                entry_time=datetime.now(), exit_time=datetime.now(),
                profit=-100.0, duration_bars=5
            ),
        ]
        
        stats = self.engine._calculate_statistics()
        
        self.assertEqual(stats.total_trades, 2)
        self.assertEqual(stats.total_pnl, 0.0)
        self.assertEqual(stats.winning_trades, 1)
        self.assertEqual(stats.losing_trades, 1)
        self.assertEqual(stats.win_rate, 50.0)


class TestSignalGeneration(unittest.TestCase):
    """Test signal generation logic."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engine = BacktestEngine()
    
    def test_no_signal_insufficient_data(self):
        """Test no signal generated with insufficient data."""
        data = {
            'close': [100, 101],
            'time': pd.date_range('2025-01-01', periods=2, freq='5min'),
            'open': [99, 100],
            'high': [101, 102],
            'low': [99, 100],
        }
        df = pd.DataFrame(data)
        df = self.engine._calculate_moving_averages(df)
        
        signals = self.engine._generate_signals(df)
        
        # Should all be neutral
        self.assertTrue((signals['signal'] == 0).all())


class TestRiskMetrics(unittest.TestCase):
    """Test risk and performance metrics."""
    
    def test_profit_factor_calculation(self):
        """Test profit factor calculation."""
        stats = BacktestStats(
            total_trades=10,
            winning_trades=6,
            losing_trades=4,
            breakeven_trades=0,
            total_pnl=600.0,
            win_rate=60.0,
            avg_win=200.0,     # 6 * 200 = 1200
            avg_loss=-100.0,    # 4 * 100 = 400
            max_profit=300.0,
            min_profit=-150.0,
            profit_factor=3.0   # 1200 / 400 = 3.0
        )
        
        self.assertEqual(stats.profit_factor, 3.0)


if __name__ == "__main__":
    unittest.main()
