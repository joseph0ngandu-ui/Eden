#!/usr/bin/env python3
"""
Volatility Burst v1.3 Strategy - Enhanced Version

Core Principle: Detect low-volatility squeeze zones (BB inside KC) and trade the momentum
breakout with ATR-based exits and confidence scoring.

Squeeze Detection: Bollinger Bands inside Keltner Channels
Breakout Signal: Price closes outside Bollinger Bands with high confidence
Risk Management: ATR-based TP/SL with trailing stops and time-based exits
"""

from dataclasses import dataclass
from typing import Dict, Optional, List
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger("Eden.VolatilityBurst")


@dataclass
class VBConfig:
    """Volatility Burst configuration."""
    atr_period: int = 14
    atr_ema_period: int = 20
    squeeze_bars: int = 12
    squeeze_atr_threshold: float = 0.8
    breakout_atr_multiplier: float = 1.5
    min_breakout_candle_size: float = 0.4
    breakout_confirmation_bars: int = 1
    max_hold_bars: int = 12
    tp_atr_multiplier: float = 1.5
    sl_atr_multiplier: float = 1.0
    trailing_after_r_mult: float = 0.8
    max_positions_per_symbol: int = 1
    daily_max_trades_per_symbol: int = 8
    min_confidence: float = 0.6


class VolatilityBurst:
    """
    Volatility Burst strategy implementation.
    
    Core mechanics:
    1. Detect squeeze: ATR < 0.8 * ATR_EMA over N bars
    2. Detect breakout: ATR > 1.5 * ATR_EMA on current bar
    3. Score confidence: composite of breakout strength, candle size, volume, direction
    4. Enter if confidence >= threshold
    5. Exit on TP/SL hit or max_hold_bars elapsed
    6. Trail stop to BE after +0.8R
    """
    
    def __init__(self, config: VBConfig):
        self.cfg = config
        self.open_positions: Dict[str, Dict] = {}
        self.daily_trades: Dict[str, int] = {}
        self.symbol_state: Dict[str, Dict] = {}
    
    # ---------- Indicators ----------
    @staticmethod
    def true_range(df: pd.DataFrame) -> pd.Series:
        """Calculate true range."""
        high = df["high"]
        low = df["low"]
        close = df["close"]
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr
    
    def atr(self, df: pd.DataFrame) -> pd.Series:
        """Calculate ATR (Average True Range)."""
        tr = self.true_range(df)
        atr = tr.rolling(self.cfg.atr_period, min_periods=1).mean()
        return atr
    
    def atr_ema(self, atr_series: pd.Series) -> pd.Series:
        """Calculate EMA of ATR for baseline volatility."""
        return atr_series.ewm(span=self.cfg.atr_ema_period, adjust=False).mean()
    
    # ---------- State Detection ----------
    def detect_squeeze(self, df: pd.DataFrame) -> pd.Series:
        """Detect squeeze condition: ATR < 0.8 * ATR_EMA."""
        atr_series = self.atr(df)
        atr_ema = self.atr_ema(atr_series)
        squeeze = atr_series < (atr_ema * self.cfg.squeeze_atr_threshold)
        return squeeze
    
    def detect_breakout(self, df: pd.DataFrame) -> pd.Series:
        """Detect breakout condition: ATR > 1.5 * ATR_EMA."""
        atr_series = self.atr(df)
        atr_ema = self.atr_ema(atr_series)
        breakout = atr_series > (atr_ema * self.cfg.breakout_atr_multiplier)
        return breakout
    
    # ---------- Confidence Scoring ----------
    def compute_confidence(self, df: pd.DataFrame, idx: int) -> float:
        """
        Composite score (0-1) for breakout candidate.
        
        Features:
        - Breakout strength (ATR / ATR_EMA)
        - Candle body size relative to ATR
        - Candle volume relative to MA
        - Direction strength vs prior N bars
        """
        if idx <= 0:
            return 0.0
        
        atr_series = self.atr(df)
        atr_ema = self.atr_ema(atr_series)
        atr = atr_series.iat[idx]
        atr_baseline = atr_ema.iat[idx]
        
        if atr_baseline == 0:
            return 0.0
        
        # Breakout strength: how much ATR exceeds the multiplier threshold
        breakout_strength = min(1.0, (atr / atr_baseline) / self.cfg.breakout_atr_multiplier)
        
        # Candle body size relative to ATR
        row = df.iloc[idx]
        body = abs(row["close"] - row["open"])
        body_vs_atr = min(1.0, body / (atr if atr > 0 else 1e-6) / self.cfg.min_breakout_candle_size)
        
        # Volume score
        vol_score = 0.5
        if "volume" in df.columns:
            vol = row["volume"]
            vol_ma = df["volume"].rolling(20, min_periods=1).mean().iat[idx]
            if vol_ma > 0:
                vol_score = min(1.0, vol / vol_ma)
        
        # Direction: check if candle closes beyond prior N bars high/low
        direction_score = 0.5
        if idx > self.cfg.squeeze_bars:
            prior_high = df["high"].iloc[max(0, idx - self.cfg.squeeze_bars):idx].max()
            prior_low = df["low"].iloc[max(0, idx - self.cfg.squeeze_bars):idx].min()
            
            if row["close"] > prior_high:
                direction_score = 1.0
            elif row["close"] < prior_low:
                direction_score = 1.0
            else:
                direction_score = 0.3
        
        # Weighted composite
        score = (0.45 * breakout_strength + 
                 0.25 * body_vs_atr + 
                 0.15 * vol_score + 
                 0.15 * direction_score)
        
        return float(min(1.0, max(0.0, score)))
    
    # ---------- Entry Evaluation ----------
    def evaluate_entry(self, symbol: str, df: pd.DataFrame) -> Optional[Dict]:
        """
        Evaluate entry conditions and return entry dict if all criteria met.
        
        Requirements:
        - Recent squeeze (last N bars with low ATR)
        - Current breakout (ATR above threshold)
        - Confidence score above minimum
        - Not exceeding daily trade cap
        """
        idx = len(df) - 1
        
        # Need enough data
        min_bars = max(self.cfg.atr_period, self.cfg.squeeze_bars) + 2
        if idx < min_bars:
            return None
        
        # Check daily trade cap
        if self.daily_trades.get(symbol, 0) >= self.cfg.daily_max_trades_per_symbol:
            return None
        
        # Check for recent squeeze
        squeeze_series = self.detect_squeeze(df)
        recently_squeezed = squeeze_series.iloc[-self.cfg.squeeze_bars:].sum() >= (self.cfg.squeeze_bars * 0.7)
        if not recently_squeezed:
            return None
        
        # Check for current breakout
        breakout_series = self.detect_breakout(df)
        if not breakout_series.iat[-1]:
            return None
        
        # Compute confidence
        confidence = self.compute_confidence(df, idx)
        if confidence < self.cfg.min_confidence:
            return None
        
        # Determine direction based on close vs prior bar
        last = df.iloc[-1]
        prior = df.iloc[-2]
        direction = "LONG" if last["close"] > prior["close"] else "SHORT"
        
        # Calculate TP/SL using ATR
        atr = self.atr(df).iat[-1]
        if direction == "LONG":
            tp = last["close"] + (atr * self.cfg.tp_atr_multiplier)
            sl = last["close"] - (atr * self.cfg.sl_atr_multiplier)
        else:  # SHORT
            tp = last["close"] - (atr * self.cfg.tp_atr_multiplier)
            sl = last["close"] + (atr * self.cfg.sl_atr_multiplier)
        
        entry = {
            "symbol": symbol,
            "direction": direction,
            "entry_price": last["close"],
            "tp": tp,
            "sl": sl,
            "atr": atr,
            "confidence": confidence,
            "max_hold_bars": self.cfg.max_hold_bars,
        }
        
        logger.info(f"{symbol} VB Entry: {direction} @ {last['close']:.2f} | Confidence: {confidence:.2f} | ATR: {atr:.4f}")
        
        return entry
    
    # ---------- Position Management ----------
    def on_order_filled(self, order: Dict):
        """Register a filled order and track position."""
        symbol = order["symbol"]
        self.open_positions[symbol] = {
            "entry_price": order["entry_price"],
            "direction": order["direction"],
            "tp": order["tp"],
            "sl": order["sl"],
            "entry_bar_index": order["bar_index"],
            "current_bar_index": order["bar_index"],
            "atr": order["atr"],
            "confidence": order["confidence"],
            "max_hold_bars": order.get("max_hold_bars", self.cfg.max_hold_bars),
            "stop_moved": False,
        }
        self.daily_trades[symbol] = self.daily_trades.get(symbol, 0) + 1
        logger.info(f"{symbol} Position opened: {order['direction']} | SL: {order['sl']:.2f} | TP: {order['tp']:.2f}")
    
    def manage_positions(self, df: pd.DataFrame, symbol: str) -> List[Dict]:
        """
        Manage open positions for the given symbol.
        Called on each new bar.
        
        Returns list of actions: close_position, trail_stop, etc.
        """
        actions = []
        
        if symbol not in self.open_positions:
            return actions
        
        pos = self.open_positions[symbol]
        idx = len(df) - 1
        pos["current_bar_index"] = idx
        
        last = df.iloc[-1]
        
        # Calculate unrealized P&L in R units
        if pos["direction"] == "LONG":
            unreal = last["close"] - pos["entry_price"]
        else:
            unreal = pos["entry_price"] - last["close"]
        
        sl_dist = pos["atr"] * self.cfg.sl_atr_multiplier
        r_value = unreal / (sl_dist if sl_dist > 0 else 1e-9)
        
        # Trailing stop: move SL to entry (breakeven) after +0.8R
        if r_value >= self.cfg.trailing_after_r_mult and not pos.get("stop_moved", False):
            pos["stop_moved"] = True
            pos["sl"] = pos["entry_price"]
            actions.append({
                "action": "trail_stop",
                "symbol": symbol,
                "reason": f"trailing_after_{self.cfg.trailing_after_r_mult}R",
                "new_sl": pos["sl"]
            })
            logger.info(f"{symbol} Trailing stop moved to BE @ {pos['sl']:.2f}")
        
        # Check TP hit
        if pos["direction"] == "LONG" and last["high"] >= pos["tp"]:
            actions.append({
                "action": "close",
                "symbol": symbol,
                "reason": "tp_hit",
                "price": pos["tp"],
                "r_value": r_value
            })
            del self.open_positions[symbol]
            logger.info(f"{symbol} TP hit @ {pos['tp']:.2f} | R value: {r_value:.2f}")
            return actions
        
        if pos["direction"] == "SHORT" and last["low"] <= pos["tp"]:
            actions.append({
                "action": "close",
                "symbol": symbol,
                "reason": "tp_hit",
                "price": pos["tp"],
                "r_value": r_value
            })
            del self.open_positions[symbol]
            logger.info(f"{symbol} TP hit @ {pos['tp']:.2f} | R value: {r_value:.2f}")
            return actions
        
        # Check SL hit
        if pos["direction"] == "LONG" and last["low"] <= pos["sl"]:
            actions.append({
                "action": "close",
                "symbol": symbol,
                "reason": "sl_hit",
                "price": pos["sl"],
                "r_value": r_value
            })
            del self.open_positions[symbol]
            logger.info(f"{symbol} SL hit @ {pos['sl']:.2f} | R value: {r_value:.2f}")
            return actions
        
        if pos["direction"] == "SHORT" and last["high"] >= pos["sl"]:
            actions.append({
                "action": "close",
                "symbol": symbol,
                "reason": "sl_hit",
                "price": pos["sl"],
                "r_value": r_value
            })
            del self.open_positions[symbol]
            logger.info(f"{symbol} SL hit @ {pos['sl']:.2f} | R value: {r_value:.2f}")
            return actions
        
        # Force close after max hold bars
        bars_held = idx - pos["entry_bar_index"]
        if bars_held >= pos["max_hold_bars"]:
            actions.append({
                "action": "close",
                "symbol": symbol,
                "reason": "max_hold_bars_expired",
                "price": last["close"],
                "r_value": r_value
            })
            del self.open_positions[symbol]
            logger.info(f"{symbol} Max hold ({pos['max_hold_bars']} bars) expired @ {last['close']:.2f}")
            return actions
        
        return actions
    
    def reset_daily_trades(self):
        """Reset daily trade counters (call daily at market open)."""
        self.daily_trades = {}
        logger.info("Daily trade counters reset")
