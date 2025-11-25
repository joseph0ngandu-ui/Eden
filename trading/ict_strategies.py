import datetime
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import pandas as pd
import logging
from trading.models import Trade, Position

logger = logging.getLogger("Eden.ICT")

# ==========================================
# 1. CORE DATA STRUCTURES
# ==========================================

@dataclass
class Bar:
    time: datetime.datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

@dataclass
class SwingPoint:
    price: float
    time: datetime.datetime
    type: str # "HIGH" or "LOW"
    is_breaker: bool = False

@dataclass
class PDArray:
    type: str # "FVG_BULL", "FVG_BEAR", "SUSPENSION_BLOCK"
    top: float
    bottom: float
    time: datetime.datetime
    mitigated: bool = False

# ==========================================
# 2. THE EYE: PATTERN RECOGNITION ENGINE
# ==========================================

class ICTPatternEngine:
    def __init__(self):
        self.swing_highs: List[SwingPoint] = []
        self.swing_lows: List[SwingPoint] = []
        self.pd_arrays: List[PDArray] = []

    def update_structure(self, bars: List[Bar]):
        """
        Updates Swing Points (Fractals). 
        Uses the standard 5-candle fractal definition (Bill Williams).
        """
        if len(bars) < 5:
            return

        # Check for Swing High (Middle candle is highest of 5)
        if (bars[-3].high > bars[-4].high and 
            bars[-3].high > bars[-5].high and 
            bars[-3].high > bars[-2].high and 
            bars[-3].high > bars[-1].high):
            
            new_swing = SwingPoint(price=bars[-3].high, time=bars[-3].time, type="HIGH")
            # Avoid duplicates
            if not self.swing_highs or self.swing_highs[-1].time != new_swing.time:
                self.swing_highs.append(new_swing)

        # Check for Swing Low (Middle candle is lowest of 5)
        if (bars[-3].low < bars[-4].low and 
            bars[-3].low < bars[-5].low and 
            bars[-3].low < bars[-2].low and 
            bars[-3].low < bars[-1].low):
            
            new_swing = SwingPoint(price=bars[-3].low, time=bars[-3].time, type="LOW")
            if not self.swing_lows or self.swing_lows[-1].time != new_swing.time:
                self.swing_lows.append(new_swing)

    def detect_fvg(self, bars: List[Bar]) -> Optional[PDArray]:
        """
        Detects if the *latest complete candle* formed a Fair Value Gap.
        Scan: Candle A (bars[-3]), Candle B (bars[-2]), Candle C (bars[-1])
        """
        if len(bars) < 3:
            return None

        candle_a = bars[-3] # 2 bars ago
        candle_c = bars[-1] # Current closed bar

        # Bullish FVG: Candle A High < Candle C Low
        if candle_a.high < candle_c.low:
            return PDArray(type="FVG_BULL", top=candle_c.low, bottom=candle_a.high, time=bars[-2].time)
        
        # Bearish FVG: Candle A Low > Candle C High
        if candle_a.low > candle_c.high:
            return PDArray(type="FVG_BEAR", top=candle_a.low, bottom=candle_c.high, time=bars[-2].time)

        return None

    def detect_suspension_block(self, bars: List[Bar]) -> Optional[PDArray]:
        """
        2025 Concept: Detects a candle isolated by Volume Imbalances.
        Pattern: [Gap] -> [Candle] -> [Gap]
        """
        if len(bars) < 4:
            return None
            
        # Check Gaps around bars[-2]
        gap_before = abs(bars[-2].open - bars[-3].close) > 0.25 # Threshold for "Gap"
        gap_after = abs(bars[-1].open - bars[-2].close) > 0.25
        
        if gap_before and gap_after:
            # The Suspension Block is the body of the isolated candle
            top = max(bars[-2].open, bars[-2].close)
            bottom = min(bars[-2].open, bars[-2].close)
            return PDArray(type="SUSPENSION_BLOCK", top=top, bottom=bottom, time=bars[-2].time)
            
        return None

    def check_breaker(self, current_price: float):
        """
        Checks if a Swing Point has been broken, turning it into a Breaker.
        """
        # Identify Bullish Breaker: A Swing High that was broken to the upside
        for swing in reversed(self.swing_highs):
            if not swing.is_breaker and current_price > swing.price:
                swing.is_breaker = True # Mark as potential support
                return swing
        return None

# ==========================================
# 3. THE CLOCK: TIME & SESSION MANAGER
# ==========================================

class ICTTimeManager:
    def __init__(self):
        # Define Silver Bullet Windows (Hour, Minute) in NY Time
        self.windows = {
            "LONDON_SB": ((3, 0), (4, 0)),
            "NY_AM_SB":  ((10, 0), (11, 0)),
            "NY_PM_SB":  ((14, 0), (15, 0)),
            "VENOM_BOX": ((8, 0), (9, 30)),
            "VENOM_EXEC":((9, 30), (11, 0))
        }
    
    def is_in_window(self, current_time: datetime.datetime, window_key: str) -> bool:
        # Ensure current_time is converted to NY Timezone before passing here
        # For simplicity, assuming input is already NY time or UTC if bot runs in UTC
        # Ideally, use pytz for timezone conversion
        
        start_h, start_m = self.windows[window_key][0]
        end_h, end_m = self.windows[window_key][1]
        
        now_minutes = current_time.hour * 60 + current_time.minute
        start_minutes = start_h * 60 + start_m
        end_minutes = end_h * 60 + end_m
        
        return start_minutes <= now_minutes < end_minutes

# ==========================================
# 4. THE BRAIN: STRATEGY IMPLEMENTATIONS
# ==========================================

class ICTStrategyBot:
    def __init__(self):
        self.engine = ICTPatternEngine()
        self.timer = ICTTimeManager()
        self.bars: List[Bar] = []
        self.open_positions: Dict[str, Position] = {}
        self.daily_trades: Dict[str, int] = {}
        
        # State for Venom Model
        self.venom_high = -float('inf')
        self.venom_low = float('inf')
        self.venom_locked = False
        self.last_processed_time = None

    def _df_to_bars(self, df: pd.DataFrame) -> List[Bar]:
        bars = []
        for row in df.itertuples():
            bars.append(Bar(
                time=row.time,
                open=row.open,
                high=row.high,
                low=row.low,
                close=row.close,
                volume=getattr(row, 'volume', 0)
            ))
        return bars

    def evaluate_live(self, df: pd.DataFrame, symbol: str) -> Optional[Trade]:
        """
        Evaluate live data for entry signal.
        """
        if df.empty:
            return None
            
        # Convert DataFrame to Bars
        current_bars = self._df_to_bars(df)
        
        # Update internal state only if new bar
        last_bar = current_bars[-1]
        if self.last_processed_time != last_bar.time:
            self.bars = current_bars # Keep full history or window?
            # Ideally we append new bars, but for stateless DF input, we might just replace or sync
            # For simplicity in this adapter, we re-run structure update on recent history
            self.engine.update_structure(self.bars)
            self.last_processed_time = last_bar.time
            
            # Run strategies
            # 1. Silver Bullet
            sb_signal = self.run_2023_silver_bullet(last_bar, symbol)
            if sb_signal: return sb_signal
            
            # 2. Unicorn
            unicorn_signal = self.run_2024_unicorn(last_bar, symbol)
            if unicorn_signal: return unicorn_signal
            
            # 3. Venom
            venom_signal = self.run_2025_venom(last_bar, symbol)
            if venom_signal: return venom_signal
            
        return None

    def manage_position(self, df: pd.DataFrame, symbol: str) -> List[Dict]:
        """
        Manage open positions.
        """
        if symbol not in self.open_positions:
            return []
            
        actions = []
        pos = self.open_positions[symbol]
        current_bar = df.iloc[-1]
        
        # Simple TP/SL check for now
        # TODO: Implement advanced management
        
        if pos.direction == "LONG":
            if current_bar['high'] >= pos.tp:
                actions.append({"action": "close", "symbol": symbol, "price": pos.tp, "reason": "tp_hit"})
                del self.open_positions[symbol]
            elif current_bar['low'] <= pos.sl:
                actions.append({"action": "close", "symbol": symbol, "price": pos.sl, "reason": "sl_hit"})
                del self.open_positions[symbol]
                
        elif pos.direction == "SHORT":
            if current_bar['low'] <= pos.tp:
                actions.append({"action": "close", "symbol": symbol, "price": pos.tp, "reason": "tp_hit"})
                del self.open_positions[symbol]
            elif current_bar['high'] >= pos.sl:
                actions.append({"action": "close", "symbol": symbol, "price": pos.sl, "reason": "sl_hit"})
                del self.open_positions[symbol]
                
        return actions
        
    def on_trade_open(self, trade: Trade):
        self.open_positions[trade.symbol] = Position(
            symbol=trade.symbol,
            direction=trade.direction,
            entry_price=trade.entry_price,
            tp=trade.tp,
            sl=trade.sl,
            entry_bar_index=trade.bar_index,
            entry_time=trade.entry_time,
            atr=trade.atr,
            confidence=trade.confidence,
            strategy=trade.strategy
        )

    # --- STRATEGY 1: 2023 SILVER BULLET (Mechanical) ---
    def run_2023_silver_bullet(self, bar: Bar, symbol: str) -> Optional[Trade]:
        if not self.timer.is_in_window(bar.time, "NY_AM_SB"):
            return None

        fvg = self.engine.detect_fvg(self.bars)
        if not fvg: 
            return None

        # Entry Logic
        if fvg.type == "FVG_BULL":
            sl = bar.low
            tp = bar.close + (bar.close - sl) * 2 # 2R
            return Trade(symbol, "LONG", bar.close, tp, sl, 0.8, 0, bar.time, 0, strategy="ICT_SilverBullet")
        elif fvg.type == "FVG_BEAR":
            sl = bar.high
            tp = bar.close - (sl - bar.close) * 2 # 2R
            return Trade(symbol, "SHORT", bar.close, tp, sl, 0.8, 0, bar.time, 0, strategy="ICT_SilverBullet")
        return None

    # --- STRATEGY 2: 2024 UNICORN (Breaker + FVG) ---
    def run_2024_unicorn(self, bar: Bar, symbol: str) -> Optional[Trade]:
        fvg = self.engine.detect_fvg(self.bars)
        if not fvg:
            return None
            
        breaker = self.engine.check_breaker(bar.close)
        
        if fvg.type == "FVG_BULL" and breaker:
            if fvg.bottom <= breaker.price <= fvg.top:
                sl = bar.low
                tp = bar.close + (bar.close - sl) * 3 # 3R
                return Trade(symbol, "LONG", bar.close, tp, sl, 0.9, 0, bar.time, 0, strategy="ICT_Unicorn")
        return None

    # --- STRATEGY 3: 2025 VENOM (Opening Range + Suspension) ---
    def run_2025_venom(self, bar: Bar, symbol: str) -> Optional[Trade]:
        # 1. Build the Box
        if self.timer.is_in_window(bar.time, "VENOM_BOX"):
            self.venom_high = max(self.venom_high, bar.high)
            self.venom_low = min(self.venom_low, bar.low)
            return None

        # 2. Lock the Box
        if not self.venom_locked and bar.time.hour == 9 and bar.time.minute == 30:
            self.venom_locked = True
            logger.info(f"VENOM BOX LOCKED: {self.venom_high} - {self.venom_low}")

        # 3. Execute
        if self.venom_locked and self.timer.is_in_window(bar.time, "VENOM_EXEC"):
            s_block = self.engine.detect_suspension_block(self.bars)
            
            if s_block and bar.close > self.venom_low:
                sl = self.venom_low - 0.0005 # Buffer
                tp = self.venom_high
                return Trade(symbol, "LONG", bar.close, tp, sl, 0.85, 0, bar.time, 0, strategy="ICT_Venom")
        return None
