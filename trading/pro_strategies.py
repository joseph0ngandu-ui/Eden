"""
Pro Strategy Engine - Prop Firm Certified
Implements verified strategies:
1. Overlap Scalper (London/NY)
2. Asian Fade (Asian Session)
3. Gold Breakout (London Open)
4. Volatility Expansion (All)
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional, List
import logging
from trading.models import Trade, Position

logger = logging.getLogger(__name__)

class ProStrategyEngine:
    def __init__(self):
        self.strategies = [
            self.overlap_scalper,
            self.asian_fade,
            self.gold_london_breakout,
            self.volatility_expansion
        ]
        self.open_positions: Dict[str, Position] = {}
        
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        try:
            if len(df) < 2: return 0.001
            tr = df['high'] - df['low']
            return tr.rolling(period).mean().iloc[-1] if len(tr) > period else tr.mean()
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return 0.001

    def evaluate_live(self, df: pd.DataFrame, symbol: str) -> Optional[Trade]:
        """Get signal for a symbol based on applicable strategies."""
        if df is None or df.empty:
            logger.warning(f"Empty data for {symbol}")
            return None
            
        if len(df) < 100:
            logger.warning(f"Insufficient data for {symbol}: {len(df)} bars")
            return None
            
        # Run all applicable strategies
        for strategy in self.strategies:
            try:
                signal = strategy(df, symbol)
                if signal:
                    logger.info(f"SIGNAL FOUND: {symbol} via {strategy.__name__}")
                    return signal
            except Exception as e:
                logger.error(f"Strategy error ({strategy.__name__}) on {symbol}: {e}")
                continue
                
        return None

    def manage_position(self, df: pd.DataFrame, symbol: str) -> List[Dict]:
        """Manage open positions."""
        if symbol not in self.open_positions:
            return []
            
        actions = []
        try:
            pos = self.open_positions[symbol]
            current_bar = df.iloc[-1]
            
            # Check TP/SL
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

    def on_trade_open(self, trade: Trade):
        """Register new position."""
        try:
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
            logger.info(f"Position registered: {trade.symbol} ({trade.direction})")
        except Exception as e:
            logger.error(f"Error registering position for {trade.symbol}: {e}")

    def overlap_scalper(self, df: pd.DataFrame, symbol: str) -> Optional[Trade]:
        """London/NY Overlap Scalper (EURUSD, GBPUSD)"""
        if symbol not in ['EURUSD', 'GBPUSD']: return None
        
        row = df.iloc[-1]
        hour = row['time'].hour
        
        # 12:00-16:00 GMT
        if not (12 <= hour < 16): return None
        
        # Momentum
        recent = df.iloc[-6:]
        momentum = recent['close'].iloc[-1] - recent['close'].iloc[0]
        atr = self.calculate_atr(df)
        
        # Volume
        avg_vol = df['tick_volume'].iloc[-21:-1].mean()
        if row['tick_volume'] < avg_vol * 1.2: return None
        
        if abs(momentum) > atr * 0.6:
            direction = "LONG" if momentum > 0 else "SHORT"
            sl_pips = atr * 1.0
            tp_pips = atr * 2.5
            
            entry = row['close']
            sl = entry - sl_pips if direction == "LONG" else entry + sl_pips
            tp = entry + tp_pips if direction == "LONG" else entry - tp_pips
            
            return Trade(
                symbol=symbol,
                direction=direction,
                entry_price=entry,
                tp=tp,
                sl=sl,
                confidence=0.8,
                atr=atr,
                entry_time=row['time'],
                bar_index=0,
                strategy="Pro_OverlapScalper"
            )
        return None

    def asian_fade(self, df: pd.DataFrame, symbol: str) -> Optional[Trade]:
        """Asian Range Fade (USDJPY, AUDJPY)"""
        if symbol not in ['USDJPY', 'AUDJPY']: return None
        
        row = df.iloc[-1]
        hour = row['time'].hour
        
        # 22:00-06:00 GMT
        if not ((hour >= 22) or (hour < 6)): return None
        
        # Asian Range (last 5 hours ~ 60 M5 bars)
        asian_bars = df.iloc[-61:-1]
        if len(asian_bars) < 60: return None
        
        asian_high = asian_bars['high'].max()
        asian_low = asian_bars['low'].min()
        asian_mid = (asian_high + asian_low) / 2
        range_size = asian_high - asian_low
        
        if range_size == 0: return None
        
        price_pos = (row['close'] - asian_low) / range_size
        atr = self.calculate_atr(df)
        
        if price_pos > 0.8: # Fade Short
            return Trade(
                symbol=symbol,
                direction="SHORT",
                entry_price=row['close'],
                sl=asian_high + range_size * 0.1,
                tp=asian_mid,
                confidence=0.8,
                atr=atr,
                entry_time=row['time'],
                bar_index=0,
                strategy="Pro_AsianFade"
            )
        elif price_pos < 0.2: # Fade Long
            return Trade(
                symbol=symbol,
                direction="LONG",
                entry_price=row['close'],
                sl=asian_low - range_size * 0.1,
                tp=asian_mid,
                confidence=0.8,
                atr=atr,
                entry_time=row['time'],
                bar_index=0,
                strategy="Pro_AsianFade"
            )
        return None

    def gold_london_breakout(self, df: pd.DataFrame, symbol: str) -> Optional[Trade]:
        """Gold London Breakout"""
        if symbol != 'XAUUSD': return None
        
        row = df.iloc[-1]
        hour = row['time'].hour
        
        # 07:00-09:00 GMT
        if not (7 <= hour < 9): return None
        
        # Find Asian Range
        # Simplified: Look back 8 hours
        lookback = df.iloc[-96:-1] # ~8 hours M5
        asian_high = lookback[lookback['time'].dt.hour < 6]['high'].max()
        asian_low = lookback[lookback['time'].dt.hour < 6]['low'].min()
        
        if pd.isna(asian_high) or pd.isna(asian_low): return None
        
        asian_range = asian_high - asian_low
        if asian_range == 0: return None
        atr = self.calculate_atr(df)
        
        if row['close'] > asian_high:
            return Trade(
                symbol=symbol,
                direction="LONG",
                entry_price=row['close'],
                sl=asian_low,
                tp=row['close'] + asian_range * 3,
                confidence=0.85,
                atr=atr,
                entry_time=row['time'],
                bar_index=0,
                strategy="Pro_GoldBreakout"
            )
        elif row['close'] < asian_low:
            return Trade(
                symbol=symbol,
                direction="SHORT",
                entry_price=row['close'],
                sl=asian_high,
                tp=row['close'] - asian_range * 3,
                confidence=0.85,
                atr=atr,
                entry_time=row['time'],
                bar_index=0,
                strategy="Pro_GoldBreakout"
            )
        return None

    def volatility_expansion(self, df: pd.DataFrame, symbol: str) -> Optional[Trade]:
        """Volatility Expansion (All Pairs)"""
        row = df.iloc[-1]
        
        current_atr = self.calculate_atr(df, period=14)
        avg_atr = self.calculate_atr(df, period=40)
        
        if avg_atr == 0: return None
        if current_atr > avg_atr * 0.9: return None # Relaxed from 0.7 for VIX
        
        recent_3 = df.iloc[-3:]
        move = abs(recent_3['close'].iloc[-1] - recent_3['close'].iloc[0])
        
        if move > current_atr * 1.5:
            direction = "LONG" if recent_3['close'].iloc[-1] > recent_3['close'].iloc[0] else "SHORT"
            
            sl_pips = avg_atr * 1.5
            tp_pips = avg_atr * 4.0
            
            entry = row['close']
            sl = entry - sl_pips if direction == "LONG" else entry + sl_pips
            tp = entry + tp_pips if direction == "LONG" else entry - tp_pips
            
            return Trade(
                symbol=symbol,
                direction=direction,
                entry_price=entry,
                tp=tp,
                sl=sl,
                confidence=0.75,
                atr=current_atr,
                entry_time=row['time'],
                bar_index=0,
                strategy="Pro_VolExpansion"
            )
        return None
