"""
Pro Strategy Engine - Prop Firm Certified
Tuned for DD Control: Daily < 4.5%, Max < 9.5%

Active Strategies (positive PF, manageable DD):
1. Overlap Scalper (London/NY) - 15.33% monthly, PF 1.10
2. Asian Fade (Asian Session) - 87.99% monthly, PF 1.31
3. Gold Breakout (London Open) - 3.03% monthly, PF 1.57
4. Trend Follower (EMA + RSI) - 3.0% monthly, PF 1.01
5. Mean Reversion (BB + RSI) - 40.32% monthly, PF 1.12

Disabled (negative returns):
- Volatility Expansion: -21.89% monthly, PF 0.88
- RSI Momentum: -15.67% monthly, PF 0.97
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
        # LIVE CONFIGURATION: ASIAN FADE + INDEX MOMENTUM
        self.strategies = [
            self.asian_fade,            # 114% monthly in 6-mo backtest (forex)
            self.index_momentum_trend,  # 12% monthly, PF 1.19 (US indices)
            # DISABLED:
            # self.index_ny_breakout,     # 5.8% monthly, PF 1.15 (lower returns)
            # self.index_mean_reversion,  # 12% monthly, high DD
            # self.mean_reversion,
            # self.overlap_scalper,
            # self.gold_london_breakout,
            # self.trend_follower,
            # self.volatility_expansion,
            # self.rsi_momentum
        ]
        self.open_positions: Dict[str, Position] = {}
        self.cooldown_minutes = 15
        self.last_trade_time = {}
        self.open_positions: Dict[str, Position] = {}
        self.cooldown_minutes = 15
        self.last_trade_time = {}
        
    def adopt_position(self, pos):
        """Adopt an existing MT5 position."""
        try:
            direction = "LONG" if pos.type == 0 else "SHORT" # 0=Buy, 1=Sell
            self.open_positions[pos.symbol] = Position(
                symbol=pos.symbol,
                direction=direction,
                entry_price=pos.price_open,
                tp=pos.tp,
                sl=pos.sl,
                entry_bar_index=0, # Unknown
                entry_time=datetime.fromtimestamp(pos.time),
                atr=0.0, # Unknown
                confidence=0.0, # Unknown
                strategy="Recovered"
            )
            logger.info(f"Adopted position: {pos.symbol} ({direction})")
        except Exception as e:
            logger.error(f"Error adopting position {pos.ticket}: {e}")

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        try:
            if len(df) < 2: 
                # Fallback: 0.2% of price if available, else 0.001
                if not df.empty and 'close' in df.columns:
                    return df['close'].iloc[-1] * 0.002
                return 0.001
            tr = df['high'] - df['low']
            return tr.rolling(period).mean().iloc[-1] if len(tr) > period else tr.mean()
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return 0.001

    def calculate_rsi(self, series: pd.Series, period: int = 14) -> float:
        try:
            delta = series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs.iloc[-1]))
        except Exception:
            return 50.0

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
                if current_bar['high'] >= pos.tp and pos.tp > 0:
                    actions.append({"action": "close", "symbol": symbol, "price": pos.tp, "reason": "tp_hit"})
                    del self.open_positions[symbol]
                elif current_bar['low'] <= pos.sl and pos.sl > 0:
                    actions.append({"action": "close", "symbol": symbol, "price": pos.sl, "reason": "sl_hit"})
                    del self.open_positions[symbol]
                    
            elif pos.direction == "SHORT":
                if current_bar['low'] <= pos.tp and pos.tp > 0:
                    actions.append({"action": "close", "symbol": symbol, "price": pos.tp, "reason": "tp_hit"})
                    del self.open_positions[symbol]
                elif current_bar['high'] >= pos.sl and pos.sl > 0:
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

    # --- EXISTING STRATEGIES ---

    def overlap_scalper(self, df: pd.DataFrame, symbol: str) -> Optional[Trade]:
        """London/NY Overlap Scalper (EURUSD, GBPUSD)"""
        base_symbol = symbol.replace('m', '')
        if base_symbol not in ['EURUSD', 'GBPUSD']: return None
        
        row = df.iloc[-1]
        hour = row.name.hour
        
        # 12:00-16:00 GMT -> approx 14:00-18:00 Broker Time (GMT+2)
        # Let's stick to the original 12-16 logic but acknowledge the timezone risk
        if not (12 <= hour < 16): return None
        
        # Momentum
        recent = df.iloc[-6:]
        momentum = recent['close'].iloc[-1] - recent['close'].iloc[0]
        atr = self.calculate_atr(df)
        
        # Volume
        avg_vol = df['tick_volume'].iloc[-21:-1].mean()
        if row['tick_volume'] < avg_vol * 1.2: return None
        
        if abs(momentum) > atr * 0.8:  # Tighter filter (was 0.6)
            direction = "LONG" if momentum > 0 else "SHORT"
            sl_pips = atr * 0.8  # Tighter SL
            tp_pips = atr * 4.0  # 5:1 R:R for high expectancy
            
            entry = row['close']
            sl = entry - sl_pips if direction == "LONG" else entry + sl_pips
            tp = entry + tp_pips if direction == "LONG" else entry - tp_pips
            
            return Trade(
                symbol=symbol,
                direction=direction,
                entry_price=entry,
                tp=tp,
                sl=sl,
                confidence=0.85,
                atr=atr,
                entry_time=row.name,
                bar_index=0,
                strategy="Pro_OverlapScalper"
            )
        return None

    def asian_fade(self, df: pd.DataFrame, symbol: str) -> Optional[Trade]:
        """Asian Range Fade (USDJPY, AUDJPY)"""
        base_symbol = symbol.replace('m', '')
        if base_symbol not in ['USDJPY', 'AUDJPY']: return None
        
        row = df.iloc[-1]
        hour = row.name.hour
        
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
        
        atr = self.calculate_atr(df)
        if not atr or atr == 0: return None
            
        # SAFETY FILTER: Avoid extremely tight ranges (leads to huge sizing & explosive breakouts)
        # Require range to be at least 2.0x ATR
        if range_size < atr * 2.0: 
            return None
        
        price_pos = (row['close'] - asian_low) / range_size
        
        if price_pos > 0.8: # Original entry threshold
            return Trade(
                symbol=symbol,
                direction="SHORT",
                entry_price=row['close'],
                sl=asian_high + range_size * 0.1,  # Original SL
                tp=asian_mid,                      # Original TP = mean
                confidence=0.8,
                atr=atr,
                entry_time=row.name,
                bar_index=0,
                strategy="Pro_AsianFade"
            )
        elif price_pos < 0.2: # Original entry threshold
            return Trade(
                symbol=symbol,
                direction="LONG",
                entry_price=row['close'],
                sl=asian_low - range_size * 0.1,   # Original SL
                tp=asian_mid,                      # Original TP = mean
                confidence=0.8,
                atr=atr,
                entry_time=row.name,
                bar_index=0,
                strategy="Pro_AsianFade"
            )
        return None

    def gold_london_breakout(self, df: pd.DataFrame, symbol: str) -> Optional[Trade]:
        """Gold London Breakout"""
        if 'XAU' not in symbol: return None
        
        row = df.iloc[-1]
        hour = row.name.hour
        
        # 07:00-09:00 GMT
        if not (7 <= hour < 9): return None
        
        # Find Asian Range
        # Simplified: Look back 8 hours
        lookback = df.iloc[-96:-1] # ~8 hours M5
        asian_high = lookback[lookback.index.hour < 6]['high'].max()
        asian_low = lookback[lookback.index.hour < 6]['low'].min()
        
        if pd.isna(asian_high) or pd.isna(asian_low): return None
        
        asian_range = asian_high - asian_low
        if asian_range == 0: return None
        atr = self.calculate_atr(df)
        
        if row['close'] > asian_high + atr * 0.2:  # Require clear breakout
            sl_dist = row['close'] - asian_low
            return Trade(
                symbol=symbol,
                direction="LONG",
                entry_price=row['close'],
                sl=row['close'] - sl_dist * 0.3,  # Tighter SL (30% of range)
                tp=row['close'] + sl_dist * 1.2,  # 4:1 R:R
                confidence=0.85,
                atr=atr,
                entry_time=row.name,
                bar_index=0,
                strategy="Pro_GoldBreakout"
            )
        elif row['close'] < asian_low - atr * 0.2:  # Require clear breakout
            sl_dist = asian_high - row['close']
            return Trade(
                symbol=symbol,
                direction="SHORT",
                entry_price=row['close'],
                sl=row['close'] + sl_dist * 0.3,  # Tighter SL
                tp=row['close'] - sl_dist * 1.2,  # 4:1 R:R
                confidence=0.85,
                atr=atr,
                entry_time=row.name,
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
                entry_time=row.name,
                bar_index=0,
                strategy="Pro_VolExpansion"
            )
        return None

    # --- NEW STRATEGIES ---

    def trend_follower(self, df: pd.DataFrame, symbol: str) -> Optional[Trade]:
        """EMA Crossover + RSI Trend Follower"""
        if len(df) < 55: return None
        
        row = df.iloc[-1]
        
        ema_20 = df['close'].ewm(span=20).mean().iloc[-1]
        ema_50 = df['close'].ewm(span=50).mean().iloc[-1]
        rsi = self.calculate_rsi(df['close'])
        atr = self.calculate_atr(df)
        
        # Long
        if ema_20 > ema_50 and rsi > 55 and row['close'] > ema_20:
            return Trade(
                symbol=symbol,
                direction="LONG",
                entry_price=row['close'],
                sl=row['close'] - (atr * 1.5),  # Tighter SL
                tp=row['close'] + (atr * 4.5),  # Better R:R (3:1)
                confidence=0.8,
                atr=atr,
                entry_time=row.name,
                bar_index=0,
                strategy="Pro_TrendFollower"
            )
        # Short
        elif ema_20 < ema_50 and rsi < 45 and row['close'] < ema_20:
            return Trade(
                symbol=symbol,
                direction="SHORT",
                entry_price=row['close'],
                sl=row['close'] + (atr * 1.5),  # Tighter SL
                tp=row['close'] - (atr * 4.5),  # Better R:R (3:1)
                confidence=0.8,
                atr=atr,
                entry_time=row.name,
                bar_index=0,
                strategy="Pro_TrendFollower"
            )
        return None

    def mean_reversion(self, df: pd.DataFrame, symbol: str) -> Optional[Trade]:
        """Bollinger Band Mean Reversion"""
        if len(df) < 25: return None
        
        row = df.iloc[-1]
        
        sma_20 = df['close'].rolling(20).mean().iloc[-1]
        std_20 = df['close'].rolling(20).std().iloc[-1]
        bb_upper = sma_20 + (std_20 * 2)
        bb_lower = sma_20 - (std_20 * 2)
        rsi = self.calculate_rsi(df['close'])
        atr = self.calculate_atr(df)
        
        # Long (Oversold)
        if row['close'] < bb_lower and rsi < 30:
            return Trade(
                symbol=symbol,
                direction="LONG",
                entry_price=row['close'],
                sl=row['close'] - (atr * 1.5),
                tp=sma_20, # Target mean
                confidence=0.75,
                atr=atr,
                entry_time=row.name,
                bar_index=0,
                strategy="Pro_MeanReversion"
            )
        # Short (Overbought)
        elif row['close'] > bb_upper and rsi > 70:
            return Trade(
                symbol=symbol,
                direction="SHORT",
                entry_price=row['close'],
                sl=row['close'] + (atr * 1.5),
                tp=sma_20,
                confidence=0.75,
                atr=atr,
                entry_time=row.name,
                bar_index=0,
                strategy="Pro_MeanReversion"
            )
        return None

    def rsi_momentum(self, df: pd.DataFrame, symbol: str) -> Optional[Trade]:
        """RSI Breakout Momentum"""
        if len(df) < 20: return None
        
        row = df.iloc[-1]
        prev_rsi = self.calculate_rsi(df['close'].iloc[:-1])
        curr_rsi = self.calculate_rsi(df['close'])
        atr = self.calculate_atr(df)
        
        # Long Breakout
        if prev_rsi <= 70 and curr_rsi > 70:
            return Trade(
                symbol=symbol,
                direction="LONG",
                entry_price=row['close'],
                sl=row['close'] - (atr * 1.0),
                tp=row['close'] + (atr * 3.0),
                confidence=0.7,
                atr=atr,
                entry_time=row.name,
                bar_index=0,
                strategy="Pro_RSIMomentum"
            )
        # Short Breakdown
        elif prev_rsi >= 30 and curr_rsi < 30:
            return Trade(
                symbol=symbol,
                direction="SHORT",
                entry_price=row['close'],
                sl=row['close'] + (atr * 1.0),
                tp=row['close'] - (atr * 3.0),
                confidence=0.7,
                atr=atr,
                entry_time=row.name,
                bar_index=0,
                strategy="Pro_RSIMomentum"
            )
        return None

    # --- US INDEX STRATEGIES ---

    def index_ny_breakout(self, df: pd.DataFrame, symbol: str) -> Optional[Trade]:
        """US Index NY Open Breakout (Trades Asian range breakout at NY open)"""
        # Only trade US indices
        if symbol not in ['US30m', 'US500m', 'USTECm']: return None
        
        if len(df) < 100: return None
        
        row = df.iloc[-1]
        # Handle both datetime index and integer index with 'time' column
        if hasattr(row.name, 'hour'):
            hour = row.name.hour
        elif 'time' in df.columns:
            hour = pd.to_datetime(df['time'].iloc[-1]).hour
        else:
            return None
        
        # NY Open window: 12:00-15:00 GMT (7:00-10:00 EST)
        if not (12 <= hour < 15): return None
        
        # Asian Range (00:00-06:00 GMT) - last 72 M5 bars roughly
        asian_bars = df.iloc[-85:-13]  # Skip last hour for cleaner range
        if len(asian_bars) < 60: return None
        
        asian_high = asian_bars['high'].max()
        asian_low = asian_bars['low'].min()
        asian_range = asian_high - asian_low
        
        if asian_range == 0: return None
        
        atr = self.calculate_atr(df)
        if not atr or atr == 0: return None
        
        # Require range to be at least 1.5x ATR (filter low volatility days)
        if asian_range < atr * 1.5: return None
        
        # Volume confirmation
        avg_vol = df['tick_volume'].iloc[-25:-1].mean()
        if row['tick_volume'] < avg_vol * 1.1: return None
        
        # Breakout Long
        if row['close'] > asian_high + atr * 0.3:
            return Trade(
                symbol=symbol,
                direction="LONG",
                entry_price=row['close'],
                sl=asian_high - atr * 0.5,  # SL below breakout level
                tp=row['close'] + asian_range * 0.8,  # Target: 80% of range
                confidence=0.8,
                atr=atr,
                entry_time=row.name,
                bar_index=0,
                strategy="Pro_IndexNYBreakout"
            )
        # Breakout Short
        elif row['close'] < asian_low - atr * 0.3:
            return Trade(
                symbol=symbol,
                direction="SHORT",
                entry_price=row['close'],
                sl=asian_low + atr * 0.5,  # SL above breakout level
                tp=row['close'] - asian_range * 0.8,  # Target: 80% of range
                confidence=0.8,
                atr=atr,
                entry_time=row.name,
                bar_index=0,
                strategy="Pro_IndexNYBreakout"
            )
        return None

    def index_momentum_trend(self, df: pd.DataFrame, symbol: str) -> Optional[Trade]:
        """US Index Momentum Trend (EMA crossover with RSI filter)"""
        # Only trade US indices
        if symbol not in ['US30m', 'US500m', 'USTECm']: return None
        
        if len(df) < 60: return None
        
        row = df.iloc[-1]
        # Handle both datetime index and integer index with 'time' column
        if hasattr(row.name, 'hour'):
            hour = row.name.hour
        elif 'time' in df.columns:
            hour = pd.to_datetime(df['time'].iloc[-1]).hour
        else:
            return None
        
        # Active US session hours: 13:00-20:00 GMT
        if not (13 <= hour < 20): return None
        
        ema_20 = df['close'].ewm(span=20).mean().iloc[-1]
        ema_50 = df['close'].ewm(span=50).mean().iloc[-1]
        prev_ema_20 = df['close'].ewm(span=20).mean().iloc[-2]
        prev_ema_50 = df['close'].ewm(span=50).mean().iloc[-2]
        
        rsi = self.calculate_rsi(df['close'])
        atr = self.calculate_atr(df)
        
        if atr == 0: return None
        
        # Trend strength check (EMAs diverging)
        ema_gap = abs(ema_20 - ema_50) / atr
        if ema_gap < 0.3: return None  # Weak trend, skip
        
        # Long: Uptrend + pullback to EMA20 + RSI not overbought
        if ema_20 > ema_50 and prev_ema_20 > prev_ema_50:  # Confirmed uptrend
            # Price near EMA20 (pullback)
            price_to_ema = abs(row['close'] - ema_20) / atr
            if price_to_ema < 0.5 and 40 < rsi < 70:  # Near EMA, not overbought
                return Trade(
                    symbol=symbol,
                    direction="LONG",
                    entry_price=row['close'],
                    sl=row['close'] - atr * 1.5,
                    tp=row['close'] + atr * 3.5,
                    confidence=0.75,
                    atr=atr,
                    entry_time=row.name,
                    bar_index=0,
                    strategy="Pro_IndexMomentum"
                )
        
        # Short: Downtrend + pullback to EMA20 + RSI not oversold
        elif ema_20 < ema_50 and prev_ema_20 < prev_ema_50:  # Confirmed downtrend
            price_to_ema = abs(row['close'] - ema_20) / atr
            if price_to_ema < 0.5 and 30 < rsi < 60:  # Near EMA, not oversold
                return Trade(
                    symbol=symbol,
                    direction="SHORT",
                    entry_price=row['close'],
                    sl=row['close'] + atr * 1.5,
                    tp=row['close'] - atr * 3.5,
                    confidence=0.75,
                    atr=atr,
                    entry_time=row.name,
                    bar_index=0,
                    strategy="Pro_IndexMomentum"
                )
        return None

    def index_mean_reversion(self, df: pd.DataFrame, symbol: str) -> Optional[Trade]:
        """US Index Mean Reversion (Bollinger Band extremes with RSI)"""
        # Only trade US indices
        if symbol not in ['US30m', 'US500m', 'USTECm']: return None
        
        if len(df) < 30: return None
        
        row = df.iloc[-1]
        
        # Calculate Bollinger Bands (2.5 std dev for stronger signals)
        sma_20 = df['close'].rolling(20).mean().iloc[-1]
        std_20 = df['close'].rolling(20).std().iloc[-1]
        bb_upper = sma_20 + (std_20 * 2.5)
        bb_lower = sma_20 - (std_20 * 2.5)
        
        rsi = self.calculate_rsi(df['close'])
        atr = self.calculate_atr(df)
        
        if atr == 0 or std_20 == 0: return None
        
        # Long: Oversold at lower band
        if row['close'] < bb_lower and rsi < 25:
            return Trade(
                symbol=symbol,
                direction="LONG",
                entry_price=row['close'],
                sl=row['close'] - atr * 1.0,
                tp=sma_20,  # Target: return to mean
                confidence=0.7,
                atr=atr,
                entry_time=row.name,
                bar_index=0,
                strategy="Pro_IndexMeanReversion"
            )
        
        # Short: Overbought at upper band
        elif row['close'] > bb_upper and rsi > 75:
            return Trade(
                symbol=symbol,
                direction="SHORT",
                entry_price=row['close'],
                sl=row['close'] + atr * 1.0,
                tp=sma_20,  # Target: return to mean
                confidence=0.7,
                atr=atr,
                entry_time=row.name,
                bar_index=0,
                strategy="Pro_IndexMeanReversion"
            )
        return None
