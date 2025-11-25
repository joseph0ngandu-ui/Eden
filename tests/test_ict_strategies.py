import unittest
from datetime import datetime, timedelta
import pandas as pd
from trading.ict_strategies import ICTPatternEngine, ICTTimeManager, ICTStrategyBot, Bar, SwingPoint, PDArray

class TestICTStrategies(unittest.TestCase):
    def setUp(self):
        self.engine = ICTPatternEngine()
        self.timer = ICTTimeManager()
        self.bot = ICTStrategyBot()

    def create_bar(self, time_str, open, high, low, close):
        return Bar(
            time=datetime.strptime(time_str, "%Y-%m-%d %H:%M"),
            open=open, high=high, low=low, close=close, volume=100
        )

    def test_swing_high_detection(self):
        # Create a fractal pattern: Low, Higher, Highest, Lower, Lowest
        bars = [
            self.create_bar("2025-01-01 10:00", 10, 11, 9, 10),
            self.create_bar("2025-01-01 10:05", 10, 12, 9, 10),
            self.create_bar("2025-01-01 10:10", 10, 13, 9, 10), # Swing High
            self.create_bar("2025-01-01 10:15", 10, 12, 9, 10),
            self.create_bar("2025-01-01 10:20", 10, 11, 9, 10),
        ]
        
        # Feed bars one by one (simulating live) or all at once
        # The engine checks the last 5 bars.
        self.engine.update_structure(bars)
        
        self.assertEqual(len(self.engine.swing_highs), 1)
        self.assertEqual(self.engine.swing_highs[0].price, 13)
        self.assertEqual(self.engine.swing_highs[0].type, "HIGH")

    def test_fvg_detection(self):
        # Create FVG pattern: Candle A, Gap, Candle C
        bars = [
            self.create_bar("2025-01-01 10:00", 100, 105, 95, 102), # A: High 105
            self.create_bar("2025-01-01 10:05", 103, 110, 103, 109), # B: Gap
            self.create_bar("2025-01-01 10:10", 108, 115, 107, 112), # C: Low 107
        ]
        # Gap between 105 and 107 -> Bullish FVG
        
        fvg = self.engine.detect_fvg(bars)
        self.assertIsNotNone(fvg)
        self.assertEqual(fvg.type, "FVG_BULL")
        self.assertEqual(fvg.bottom, 105)
        self.assertEqual(fvg.top, 107)

    def test_time_manager(self):
        # NY AM SB: 10:00 - 11:00
        t1 = datetime(2025, 1, 1, 10, 30)
        t2 = datetime(2025, 1, 1, 9, 59)
        t3 = datetime(2025, 1, 1, 11, 1)
        
        self.assertTrue(self.timer.is_in_window(t1, "NY_AM_SB"))
        self.assertFalse(self.timer.is_in_window(t2, "NY_AM_SB"))
        self.assertFalse(self.timer.is_in_window(t3, "NY_AM_SB"))

    def test_silver_bullet_signal(self):
        # Setup: In window, FVG formed
        t = datetime(2025, 1, 1, 10, 5) # In window
        
        bars = [
            Bar(t - timedelta(minutes=10), 100, 105, 95, 102, 100),
            Bar(t - timedelta(minutes=5), 103, 110, 103, 109, 100),
            Bar(t, 108, 115, 107, 112, 100),
        ]
        
        # Manually set bars in bot
        self.bot.bars = bars
        
        # Run strategy
        signal = self.bot.run_2023_silver_bullet(bars[-1], "EURUSD")
        
        self.assertIsNotNone(signal)
        self.assertEqual(signal.direction, "LONG")
        self.assertEqual(signal.strategy, "ICT_SilverBullet")

if __name__ == '__main__':
    unittest.main()
