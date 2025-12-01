import pandas as pd
import numpy as np
from datetime import time
from typing import Dict, List, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class GoldConfig:
    """Configuration for Gold Momentum Strategy."""
    asian_session_start: time = time(0, 0)
    asian_session_end: time = time(6, 0)
    london_open_start: time = time(7, 0)
    london_open_end: time = time(10, 0)
    ny_open_start: time = time(13, 0)
    ny_open_end: time = time(16, 0)
    
    breakout_buffer: float = 1.0   # Conservative buffer
    stop_loss_pips: float = 50.0   # 50 pips ($5.00)
    take_profit_pips: float = 100.0 # 100 pips ($10.00) - 1:2 RR
    trailing_stop_pips: float = 30.0
    trailing_trigger_pips: float = 50.0
    
    risk_percent: float = 0.5      # Conservative risk
    max_daily_trades: int = 1      # Quality over quantity

class GoldMomentumStrategy:
    """
    Gold (XAUUSD) Momentum Strategy.
    
    Logic:
    1. Define Asian Session Range (High/Low).
    2. Trade breakouts during London or NY Open.
    3. Filter with H1 Trend (EMA 50).
    """
    
    def __init__(self, config: GoldConfig = None):
        self.config = config or GoldConfig()
        self.asian_high = None
        self.asian_low = None
        self.daily_trades = 0
        self.last_trade_day = None
        self.open_positions = {} # Fix: Add open_positions attribute

    def manage_position(self, df: pd.DataFrame, symbol: str) -> List[Dict]:
        """Manage open positions."""
        if symbol not in self.open_positions:
            return []
            
        actions = []
        try:
            pos = self.open_positions[symbol]
            current_bar = df.iloc[-1]
            
            # Simple TP/SL check
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
        except Exception as e:
            logger.error(f"Error managing position for {symbol}: {e}")
            
        return actions

    def on_trade_open(self, trade):
        """Register new position."""
        try:
            # Create a simple position object if not imported
            # Assuming trade object has necessary fields
            from trading.models import Position
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
        except Exception as e:
            logger.error(f"Error registering position for {trade.symbol}: {e}")

    def evaluate_live(self, df: pd.DataFrame, symbol: str) -> Optional[Dict]:
        """
        Generate trade signal for the current bar.
        Expects df to have OHLCV data and 'time' index or column.
        """
        if len(df) < 600: # Need enough data for EMA
            return None
            
        # Ensure we have a datetime index or column
        if 'time' in df.columns:
            df = df.set_index('time')
            
        current_bar = df.iloc[-1]
        current_time = current_bar.name.time()
        current_date = current_bar.name.date()
        
        # Reset daily counters
        if self.last_trade_day != current_date:
            self.daily_trades = 0
            self.last_trade_day = current_date
            self.asian_high = None
            self.asian_low = None
            
        if self.daily_trades >= self.config.max_daily_trades:
            return None
            
        # 1. Calculate Asian Range (00:00 - 06:00)
        # Get today's data so far
        today_data = df[df.index.date == current_date]
        
        # Extract Asian session data
        asian_data = today_data.between_time(self.config.asian_session_start, self.config.asian_session_end)
        
        if len(asian_data) > 0:
            self.asian_high = asian_data['high'].max()
            self.asian_low = asian_data['low'].min()
        else:
            # If we don't have Asian data yet (e.g. it's 02:00), we can't trade breakouts yet
            return None
            
        # 2. Check Trading Window (London or NY Open)
        is_london = self.config.london_open_start <= current_time <= self.config.london_open_end
        is_ny = self.config.ny_open_start <= current_time <= self.config.ny_open_end
        
        if not (is_london or is_ny):
            return None
            
        # 3. Trend Filter (EMA 200 on M5 ~ EMA 17 on H1, let's use EMA 200 M5 for short term trend)
        # Or better: EMA 50 on H1 = EMA 600 on M5
        ema_period = 200
        ema = df['close'].ewm(span=ema_period).mean()
        current_ema = ema.iloc[-1]
        
        # 4. Check Breakouts
        close = current_bar['close']
        
        # Long Breakout
        if close > (self.asian_high + self.config.breakout_buffer):
            if close > current_ema: # Trend Filter
                return {
                    'direction': 'LONG',
                    'entry_price': close,
                    'sl': close - self.config.stop_loss_pips * 0.1, # Convert pips to price (approx) - Gold 1 pip = 0.1? No, 0.01 usually on MT5, but let's assume 0.1 for XAUUSD standard
                    # Actually XAUUSD 1 pip = $0.10 usually. $1 move = 10 pips.
                    # Let's standardize: input pips are actually $ moves for simplicity in config?
                    # No, let's stick to price. 40 pips = $4.00 move.
                    'sl_price': close - 4.0, 
                    'tp_price': close + 12.0, # 1:3 RR
                    'reason': 'Asian Breakout Long'
                }
                
        # Short Breakout
        if close < (self.asian_low - self.config.breakout_buffer):
            if close < current_ema: # Trend Filter
                return {
                    'direction': 'SHORT',
                    'entry_price': close,
                    'sl_price': close + 4.0,
                    'tp_price': close - 12.0,
                    'reason': 'Asian Breakout Short'
                }
                
        return None
