#!/usr/bin/env python3
"""
Volatility Burst v1.3 Strategy - Enhanced Version

Core Principle: Detect low-volatility squeeze zones (BB inside KC) and trade the momentum
breakout with ATR-based exits and confidence scoring.

Squeeze Detection: Bollinger Bands inside Keltner Channels
Breakout Signal: Price closes outside Bollinger Bands with high confidence
Risk Management: ATR-based TP/SL with trailing stops and time-based exits
"""

import yaml
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger("Eden.VolatilityBurst")


@dataclass
class Trade:
    """Trade signal object."""
    symbol: str
    direction: str  # "LONG" or "SHORT"
    entry_price: float
    tp: float
    sl: float
    confidence: float
    atr: float
    entry_time: pd.Timestamp
    bar_index: int


@dataclass
class Position:
    """Open position tracking."""
    symbol: str
    direction: str
    entry_price: float
    tp: float
    sl: float
    entry_bar_index: int
    entry_time: pd.Timestamp
    atr: float
    confidence: float
    stop_moved: bool = False


class VolatilityBurst:
    """
    Enhanced Volatility Burst v1.3 Strategy
    
    Detects squeeze conditions using Bollinger Bands inside Keltner Channels,
    then trades breakouts with confidence-based filtering and ATR-based exits.
    """
    
    def __init__(self, config_path: str = None):
        """Initialize strategy with configuration."""
        self.config_path = config_path or "config/volatility_burst.yml"
        self.config = self._load_config()
        self.open_positions: Dict[str, Position] = {}
        self.daily_trades: Dict[str, int] = {}
        self.bars_since_open = 0
        
        # Extract key config values
        self.risk_pct = self.config['risk']['risk_pct']
        self.confidence_threshold = self.config['risk']['confidence_threshold']
        self.max_trades_per_day = self.config['risk']['max_trades_per_day']
        self.max_bars_in_trade = self.config['risk']['max_bars_in_trade']
        self.skip_bars_after_open = self.config['risk']['skip_bars_after_open']
        
        # Indicator periods
        self.atr_period = self.config['indicators']['atr']['period']
        self.bb_period = self.config['indicators']['bollinger_bands']['period']
        self.bb_std = self.config['indicators']['bollinger_bands']['std_dev']
        self.kc_period = self.config['indicators']['keltner_channels']['period']
        self.kc_mult = self.config['indicators']['keltner_channels']['multiplier']
        
        # Entry/Exit
        self.tp_atr_mult = self.config['entry_exit']['tp_atr_multiplier']
        self.sl_atr_mult = self.config['entry_exit']['sl_atr_multiplier']
        self.trail_trigger_r = self.config['entry_exit']['trail_trigger_r']
        
        # Confidence weights and boost
        self.conf_weights = self.config['confidence']
        self.conf_boost_alpha = float(self.config.get('confidence_boost_alpha', 2.5))
        
        logger.info(f"VolatilityBurst v1.3 initialized with config: {self.config_path}")
    
    def update_params(self, **kwargs):
        """Dynamically update key parameters (for optimization)."""
        for k, v in kwargs.items():
            if k == 'confidence_threshold':
                self.confidence_threshold = float(v)
            elif k == 'tp_atr_multiplier':
                self.tp_atr_mult = float(v)
            elif k == 'sl_atr_multiplier':
                self.sl_atr_mult = float(v)
            elif k == 'kc_multiplier':
                self.kc_mult = float(v)
            elif k == 'bb_std':
                self.bb_std = float(v)
            elif k == 'bb_period':
                self.bb_period = int(v)
            elif k == 'atr_period':
                self.atr_period = int(v)
            elif k == 'max_trades_per_day':
                self.max_trades_per_day = int(v)
            elif k == 'skip_bars_after_open':
                self.skip_bars_after_open = int(v)
            elif k == 'confidence_boost_alpha':
                self.conf_boost_alpha = float(v)
            # Add more keys as needed
    
    def _load_config(self) -> Dict:
        """Load strategy configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Config file not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing config: {e}")
            raise
    
    # ---------- Technical Indicators ----------
    
    def calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Average True Range."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        return tr.rolling(self.atr_period, min_periods=1).mean()
    
    def calculate_bollinger_bands(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands (upper, middle, lower)."""
        close = df['close']
        middle = close.rolling(self.bb_period, min_periods=1).mean()
        std = close.rolling(self.bb_period, min_periods=1).std()
        
        upper = middle + (std * self.bb_std)
        lower = middle - (std * self.bb_std)
        
        return upper, middle, lower
    
    def calculate_keltner_channels(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Keltner Channels (upper, middle, lower)."""
        close = df['close']
        atr = self.calculate_atr(df)
        middle = close.ewm(span=self.kc_period).mean()
        
        upper = middle + (atr * self.kc_mult)
        lower = middle - (atr * self.kc_mult)
        
        return upper, middle, lower
    
    def calculate_ema_momentum(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Calculate fast and slow EMA for momentum."""
        close = df['close']
        fast_period = self.config['indicators']['ema_momentum']['fast_period']
        slow_period = self.config['indicators']['ema_momentum']['slow_period']
        
        ema_fast = close.ewm(span=fast_period).mean()
        ema_slow = close.ewm(span=slow_period).mean()
        
        return ema_fast, ema_slow
    
    # ---------- Signal Detection ----------
    
    def detect_squeeze(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect squeeze condition: Bollinger Bands inside Keltner Channels.
        Returns boolean series where True = squeeze active.
        """
        bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(df)
        kc_upper, kc_middle, kc_lower = self.calculate_keltner_channels(df)
        
        # Squeeze when BB upper < KC upper AND BB lower > KC lower
        squeeze = (bb_upper < kc_upper) & (bb_lower > kc_lower)
        
        return squeeze
    
    def detect_breakout(self, df: pd.DataFrame) -> Tuple[pd.Series, str]:
        """
        Detect breakout from squeeze: price closes outside Bollinger Bands.
        Returns (breakout_series, direction) where direction is 'LONG' or 'SHORT'.
        """
        bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(df)
        close = df['close']
        
        # Breakout conditions
        long_breakout = close > bb_upper
        short_breakout = close < bb_lower
        
        # Determine overall breakout and direction
        breakout = long_breakout | short_breakout
        
        # Direction based on last bar
        if len(df) > 0:
            if long_breakout.iloc[-1]:
                direction = "LONG"
            elif short_breakout.iloc[-1]:
                direction = "SHORT"
            else:
                direction = None
        else:
            direction = None
        
        return breakout, direction
    
    def compute_confidence(self, df: pd.DataFrame, idx: int) -> float:
        """
        Compute confidence score (0.0-1.0) for breakout candidate.
        
        Factors:
        - Breakout strength: how far price is from KC bands
        - Body size: candle body vs ATR
        - Volume expansion: volume vs 20-bar MA
        - Momentum: EMA alignment
        """
        if idx < max(self.atr_period, self.bb_period, self.kc_period):
            return 0.0
        
        row = df.iloc[idx]
        atr_series = self.calculate_atr(df)
        atr = atr_series.iloc[idx] if idx < len(atr_series) else atr_series.iloc[-1]
        
        if atr <= 0:
            return 0.0
        
        # 1. Breakout Strength: distance from Keltner Channel
        kc_upper, kc_middle, kc_lower = self.calculate_keltner_channels(df)
        kc_up = kc_upper.iloc[idx] if idx < len(kc_upper) else kc_upper.iloc[-1]
        kc_low = kc_lower.iloc[idx] if idx < len(kc_lower) else kc_lower.iloc[-1]
        
        if row['close'] > kc_up:
            breakout_strength = min(1.0, (row['close'] - kc_up) / atr)
        elif row['close'] < kc_low:
            breakout_strength = min(1.0, (kc_low - row['close']) / atr)
        else:
            breakout_strength = 0.0
        
        # 2. Body Size: candle body relative to ATR
        body = abs(row['close'] - row['open'])
        body_score = min(1.0, body / atr)
        
        # 3. Volume Expansion: volume vs 20-bar MA
        volume_score = 0.5  # default if no volume data
        if 'volume' in df.columns and idx >= 20:
            vol = row['volume']
            vol_ma = df['volume'].iloc[max(0, idx-19):idx+1].mean()
            if vol_ma > 0:
                volume_score = min(1.0, vol / vol_ma)
        
        # 4. Momentum: EMA alignment
        ema_fast, ema_slow = self.calculate_ema_momentum(df)
        if idx < len(ema_fast) and idx < len(ema_slow):
            ema_f = ema_fast.iloc[idx]
            ema_s = ema_slow.iloc[idx]
            momentum_score = min(1.0, abs(ema_f - ema_s) / (row['close'] * 0.01))  # Normalized
        else:
            momentum_score = 0.5
        
        # Weighted composite score
        confidence = (
            self.conf_weights['breakout_strength_weight'] * breakout_strength +
            self.conf_weights['body_size_weight'] * body_score +
            self.conf_weights['volume_expansion_weight'] * volume_score +
            self.conf_weights['momentum_weight'] * momentum_score
        )
        
        # Non-linear boost to push high-confidence events closer to 1.0
        confidence = float(min(1.0, max(0.0, confidence)))
        confidence = 1.0 - pow((1.0 - confidence), self.conf_boost_alpha)
        return float(min(1.0, max(0.0, confidence)))
    
    # ---------- Signal Generation ----------
    
    def generate_signals(self, df: pd.DataFrame) -> List[Trade]:
        """
        Generate trade signals from price data (vectorized).
        
        Steps:
        1. Precompute indicators (ATR, BB, KC, EMA, volume MA)
        2. Build squeeze and breakout masks
        3. Compute confidence score vector
        4. Apply daily limits and skip rules
        5. Emit Trade objects at qualifying bars
        """
        n = len(df)
        if n < max(self.atr_period, self.bb_period, self.kc_period) + 5:
            return []
        
        # Precompute indicators
        atr = self.calculate_atr(df)
        bb_upper, bb_mid, bb_lower = self.calculate_bollinger_bands(df)
        kc_upper, kc_mid, kc_lower = self.calculate_keltner_channels(df)
        ema_fast, ema_slow = self.calculate_ema_momentum(df)
        
        # Masks
        squeeze = (bb_upper < kc_upper) & (bb_lower > kc_lower)
        long_breakout = df['close'] > bb_upper
        short_breakout = df['close'] < bb_lower
        breakout = long_breakout | short_breakout
        
        # Confidence components
        # Breakout strength relative to KC edges
        breakout_dist = pd.Series(0.0, index=df.index)
        breakout_dist = np.where(long_breakout, (df['close'] - kc_upper) / atr.replace(0, np.nan), breakout_dist)
        breakout_dist = np.where(short_breakout, (kc_lower - df['close']) / atr.replace(0, np.nan), breakout_dist)
        breakout_strength = pd.Series(breakout_dist, index=df.index).clip(lower=0, upper=1).fillna(0)
        
        body = (df['close'] - df['open']).abs()
        body_score = (body / atr.replace(0, np.nan)).clip(0, 1).fillna(0)
        
        if 'volume' in df.columns:
            vol_ma20 = df['volume'].rolling(20, min_periods=1).mean()
            volume_score = (df['volume'] / vol_ma20.replace(0, np.nan)).clip(0, 1).fillna(0.5)
        else:
            volume_score = pd.Series(0.5, index=df.index)
        
        momentum_score = ((ema_fast - ema_slow).abs() / (df['close'] * 0.01).replace(0, np.nan)).clip(0, 1).fillna(0.5)
        
        # Weighted confidence + boost
        confidence = (
            self.conf_weights['breakout_strength_weight'] * breakout_strength +
            self.conf_weights['body_size_weight'] * body_score +
            self.conf_weights['volume_expansion_weight'] * volume_score +
            self.conf_weights['momentum_weight'] * momentum_score
        ).clip(0, 1)
        # Apply same non-linear boost as compute_confidence
        confidence = 1.0 - (1.0 - confidence) ** self.conf_boost_alpha
        
        # Recent squeeze (within last 12 bars)
        squeeze_window = 12
        recent_squeeze = squeeze.rolling(window=squeeze_window, min_periods=1).sum().shift(1).fillna(0) > 0
        
        # Skip first N bars after daily session open
        if 'time' in df.columns:
            day_index = df['time'].dt.date
            intraday_idx = day_index.groupby(day_index).cumcount()
            skip_mask = intraday_idx < self.skip_bars_after_open
        else:
            skip_mask = pd.Series([False] * n)
        
        # Final signal mask
        signal_mask = (~skip_mask) & recent_squeeze & breakout & (confidence >= self.confidence_threshold)
        
        # Enforce per-day trade cap (keep first N signals per day)
        if 'time' in df.columns:
            signal_indices = np.where(signal_mask)[0]
            if len(signal_indices) == 0:
                return []
            dates = df['time'].iloc[signal_indices].dt.date
            # count signals per day and cap
            counts = {}
            kept_indices = []
            for i, d in zip(signal_indices, dates):
                c = counts.get(d, 0)
                if c < self.max_trades_per_day:
                    kept_indices.append(i)
                    counts[d] = c + 1
            signal_indices = kept_indices
        else:
            signal_indices = np.where(signal_mask)[0].tolist()
        
        # Emit Trade objects
        trades: List[Trade] = []
        sym = df.attrs.get('symbol', 'UNKNOWN')
        for idx in signal_indices:
            row = df.iloc[idx]
            dirn = 'LONG' if long_breakout.iloc[idx] else 'SHORT'
            atr_val = float(atr.iloc[idx]) if not pd.isna(atr.iloc[idx]) else 0.0
            if atr_val <= 0:
                continue
            entry = float(row['close'])
            if dirn == 'LONG':
                tp = entry + (atr_val * self.tp_atr_mult)
                sl = entry - (atr_val * self.sl_atr_mult)
            else:
                tp = entry - (atr_val * self.tp_atr_mult)
                sl = entry + (atr_val * self.sl_atr_mult)
            trades.append(Trade(
                symbol=sym,
                direction=dirn,
                entry_price=entry,
                tp=float(tp),
                sl=float(sl),
                confidence=float(confidence.iloc[idx]),
                atr=atr_val,
                entry_time=row['time'] if 'time' in df.columns else pd.Timestamp.now(),
                bar_index=int(idx)
            ))
        
        return trades

    def evaluate_live(self, df: pd.DataFrame, symbol: str) -> Optional[Trade]:
        """
        Evaluate live data for entry signal.
        Used in live trading mode.
        """
        # Check daily trade limit
        if self.daily_trades.get(symbol, 0) >= self.max_trades_per_day:
            return None
        
        # Check if position already open
        if symbol in self.open_positions:
            return None
        
        # Generate signals and return the latest one
        df.attrs['symbol'] = symbol
        signals = self.generate_signals(df)
        
        if signals:
            trade = signals[-1]  # Take the most recent signal
            
            # Register the trade
            self.daily_trades[symbol] = self.daily_trades.get(symbol, 0) + 1
            
            # Create position tracking
            position = Position(
                symbol=symbol,
                direction=trade.direction,
                entry_price=trade.entry_price,
                tp=trade.tp,
                sl=trade.sl,
                entry_bar_index=trade.bar_index,
                entry_time=trade.entry_time,
                atr=trade.atr,
                confidence=trade.confidence
            )
            self.open_positions[symbol] = position
            
            return trade
        
        return None
    
    def on_trade_open(self, trade: Trade):
        """Register an opened trade (used by backtests or live)."""
        self.daily_trades[trade.symbol] = self.daily_trades.get(trade.symbol, 0) + 1
        self.open_positions[trade.symbol] = Position(
            symbol=trade.symbol,
            direction=trade.direction,
            entry_price=trade.entry_price,
            tp=trade.tp,
            sl=trade.sl,
            entry_bar_index=trade.bar_index,
            entry_time=trade.entry_time,
            atr=trade.atr,
            confidence=trade.confidence
        )
        logger.info(f"{trade.symbol} Trade opened: {trade.direction} @ {trade.entry_price:.2f} | TP {trade.tp:.2f} | SL {trade.sl:.2f}")

    def manage_position(self, df: pd.DataFrame, symbol: str) -> List[Dict]:
        """
        Manage open position for given symbol.
        Returns list of actions to take.
        """
        if symbol not in self.open_positions:
            return []
        
        actions = []
        pos = self.open_positions[symbol]
        current_idx = len(df) - 1
        current_bar = df.iloc[-1]
        
        # Calculate current P&L in R units
        if pos.direction == "LONG":
            unrealized = current_bar['close'] - pos.entry_price
        else:
            unrealized = pos.entry_price - current_bar['close']
        
        sl_distance = pos.atr * self.sl_atr_mult
        r_value = unrealized / sl_distance if sl_distance > 0 else 0
        
        # 1. Trailing stop to breakeven
        if r_value >= self.trail_trigger_r and not pos.stop_moved:
            pos.stop_moved = True
            pos.sl = pos.entry_price
            actions.append({
                "action": "trail_stop",
                "symbol": symbol,
                "new_sl": pos.sl,
                "reason": f"trailing_to_be_after_{self.trail_trigger_r}R"
            })
            logger.info(f"{symbol} Trailing stop to BE @ {pos.sl:.2f}")
        
        # 2. Check TP hit
        if pos.direction == "LONG" and current_bar['high'] >= pos.tp:
            actions.append({
                "action": "close",
                "symbol": symbol,
                "price": pos.tp,
                "reason": "tp_hit",
                "r_value": r_value
            })
            del self.open_positions[symbol]
            return actions
        
        if pos.direction == "SHORT" and current_bar['low'] <= pos.tp:
            actions.append({
                "action": "close",
                "symbol": symbol,
                "price": pos.tp,
                "reason": "tp_hit",
                "r_value": r_value
            })
            del self.open_positions[symbol]
            return actions
        
        # 3. Check SL hit
        if pos.direction == "LONG" and current_bar['low'] <= pos.sl:
            actions.append({
                "action": "close",
                "symbol": symbol,
                "price": pos.sl,
                "reason": "sl_hit",
                "r_value": r_value
            })
            del self.open_positions[symbol]
            return actions
        
        if pos.direction == "SHORT" and current_bar['high'] >= pos.sl:
            actions.append({
                "action": "close",
                "symbol": symbol,
                "price": pos.sl,
                "reason": "sl_hit",
                "r_value": r_value
            })
            del self.open_positions[symbol]
            return actions
        
        # 4. Time-based exit
        bars_in_trade = current_idx - pos.entry_bar_index
        if bars_in_trade >= self.max_bars_in_trade:
            actions.append({
                "action": "close",
                "symbol": symbol,
                "price": current_bar['close'],
                "reason": "max_time_exit",
                "r_value": r_value
            })
            del self.open_positions[symbol]
            return actions
        
        return actions
    
    def reset_daily_trades(self):
        """Reset daily trade counters."""
        self.daily_trades = {}
        self.bars_since_open = 0
        logger.info("Daily counters reset")
    
    def update_metrics(self, trade_result: Dict):
        """Update strategy metrics (for live trading monitoring)."""
        # This can be extended for live performance tracking
        symbol = trade_result.get('symbol')
        pnl = trade_result.get('pnl', 0)
        r_value = trade_result.get('r_value', 0)
        
        logger.info(f"{symbol} Trade closed: PnL=${pnl:.2f} | R={r_value:.2f}")


def load_strategy_config(config_path: str = None) -> Dict:
    """Load strategy configuration (utility function)."""
    path = config_path or "config/volatility_burst.yml"
    with open(path, 'r') as f:
        return yaml.safe_load(f)