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
        # OPTIMIZED CONFIGURATION (2025-12-06)
        # Includes original high-edge strategies from research
        self.strategies = [
            # ORIGINAL STRATEGIES - HIGH EDGE (from research)
            self.quiet_before_storm,      # 0.599R GBP, 0.220R Gold - STRONG
            self.triple_candle_breakout,  # 0.445R JPY, 0.228R GBP - STRONG
            self.volatility_squeeze,      # 0.185R JPY - STRONG
            # ===== ACTIVE PORTFOLIO (MULTI-PAIR) =====
            # 1. Volatility Squeeze (Primary: Run on 28 Pairs)
            self.volatility_squeeze,      # +15.5R (6mo) - High Win Rate Trend
            
            # 2. Quiet Before Storm (Sniper: Gold/GBP)
            self.quiet_before_storm,      # +0.5R (6mo) / +0.6R (24mo) - Big Moves
            
            # 3. VWAP Reversion M5 (High Freq: Enabled by User Request)
            self.vwap_reversion_m5,       # High Volume Cash Flow (Monitor Closely!)
            
            # DEPRECATED / FAILED VALIDATION
            # self.asian_fade (Failed 6mo Test: -72R)
            # self.hourly_range_reversion (Failed 6mo Test: -49R)
            # self.triple_candle_breakout (Failed 6mo Test: -3R)
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

    # ===== VERIFIED HIGH-EDGE STRATEGIES (2025-12-06) =====
    # Includes H1 Breakout patterns and M5 High-Freq Mean Reversion

    def quiet_before_storm(self, df: pd.DataFrame, symbol: str) -> Optional[Trade]:
        """
        Quiet Before Storm (Best on GBPUSD, XAUUSD)
        Logic: Vol Contraction -> Strong Breakout
        Edge: 0.599R (GBP), 0.220R (Gold)
        """
        if "GBP" not in symbol and "XAU" not in symbol: return None
        if len(df) < 50: return None
        
        signal = df.iloc[-2]
        current = df.iloc[-1]
        atr = self.calculate_atr(df)
        if atr == 0: return None
        
        tr = df['high'] - df['low']
        avg_vol = tr.iloc[-22:-2].mean()
        recent_vol = tr.iloc[-7:-2].mean()
        
        if recent_vol > avg_vol * 0.6: return None
        
        high_10 = df['high'].iloc[-12:-2].max()
        low_10 = df['low'].iloc[-12:-2].min()
        
        body = abs(signal['close'] - signal['open'])
        avg_body = abs(df['close'] - df['open']).iloc[-22:-2].mean()
        
        # Relaxed body filter for higher frequency
        if body < avg_body * 1.2: return None
        
        # LONG
        if signal['close'] > high_10 and signal['close'] > signal['open']:
            sl = df['low'].iloc[-7:-2].min() - atr * 0.2
            tp = current['close'] + (current['close'] - sl) * 2.5
            return Trade(symbol=symbol, direction="LONG", entry_price=current['close'],
                         sl=sl, tp=tp, confidence=0.90, atr=atr,
                         entry_time=current.name, bar_index=0, strategy="Pro_QuietStorm")
        # SHORT
        if signal['close'] < low_10 and signal['close'] < signal['open']:
            sl = df['high'].iloc[-7:-2].max() + atr * 0.2
            tp = current['close'] - (sl - current['close']) * 2.5
            return Trade(symbol=symbol, direction="SHORT", entry_price=current['close'],
                         sl=sl, tp=tp, confidence=0.90, atr=atr,
                         entry_time=current.name, bar_index=0, strategy="Pro_QuietStorm")
        return None

    def triple_candle_breakout(self, df: pd.DataFrame, symbol: str) -> Optional[Trade]:
        """
        Triple Candle Breakout (Best on JPY pairs, GBP)
        Logic: Mother -> In1 -> In2 -> Breakout
        Edge: 0.445R (USDJPY), 0.228R (GBP)
        """
        if len(df) < 50: return None
        breakout = df.iloc[-2]
        inside2 = df.iloc[-3]
        inside1 = df.iloc[-4]
        mother = df.iloc[-5]
        current = df.iloc[-1]
        atr = self.calculate_atr(df)
        
        mother_range = mother['high'] - mother['low']
        if mother_range < atr * 0.8: return None
        
        if not (inside1['high'] < mother['high'] and inside1['low'] > mother['low']): return None
        if not (inside2['high'] < inside1['high'] and inside2['low'] > inside1['low']): return None
        
        # LONG
        if breakout['close'] > mother['high']:
            sl = inside2['low'] - atr * 0.1
            tp = current['close'] + (current['close'] - sl) * 2.5
            return Trade(symbol=symbol, direction="LONG", entry_price=current['close'],
                         sl=sl, tp=tp, confidence=0.85, atr=atr,
                         entry_time=current.name, bar_index=0, strategy="Pro_TripleCandle")
        # SHORT
        if breakout['close'] < mother['low']:
            sl = inside2['high'] + atr * 0.1
            tp = current['close'] - (sl - current['close']) * 2.5
            return Trade(symbol=symbol, direction="SHORT", entry_price=current['close'],
                         sl=sl, tp=tp, confidence=0.85, atr=atr,
                         entry_time=current.name, bar_index=0, strategy="Pro_TripleCandle")
        return None

    def volatility_squeeze(self, df: pd.DataFrame, symbol: str) -> Optional[Trade]:
        if "JPY" not in symbol: return None
        if len(df) < 60: return None
        signal = df.iloc[-2]
        current = df.iloc[-1]
        atr = self.calculate_atr(df)
        
        closes = df['close']
        vol_20 = closes.iloc[-21:-1].pct_change().std()
        vol_50 = closes.iloc[-51:-1].pct_change().std()
        
        if np.isnan(vol_20) or np.isnan(vol_50): return None
        if vol_20 > vol_50 * 0.6: return None
        
        high_20 = df['high'].iloc[-22:-2].max()
        low_20 = df['low'].iloc[-22:-2].min()
        
        if signal['close'] > high_20:
            sl = low_20
            if (current['close'] - sl) > 2 * atr: sl = current['close'] - 2 * atr
            tp = current['close'] + (current['close'] - sl) * 1.5
            return Trade(symbol=symbol, direction="LONG", entry_price=current['close'],
                         sl=sl, tp=tp, confidence=0.80, atr=atr,
                         entry_time=current.name, bar_index=0, strategy="Pro_VolSqueeze")
        if signal['close'] < low_20:
            sl = high_20
            if (sl - current['close']) > 2 * atr: sl = current['close'] + 2 * atr
            tp = current['close'] - (sl - current['close']) * 1.5
            return Trade(symbol=symbol, direction="SHORT", entry_price=current['close'],
                         sl=sl, tp=tp, confidence=0.80, atr=atr,
                         entry_time=current.name, bar_index=0, strategy="Pro_VolSqueeze")
        return None

    def hourly_range_reversion(self, df: pd.DataFrame, symbol: str) -> Optional[Trade]:
        if "JPY" not in symbol: return None
        if len(df) < 50: return None
        signal = df.iloc[-2]
        current = df.iloc[-1]
        atr = self.calculate_atr(df)
        
        curr_range = signal['high'] - signal['low']
        avg_range = (df['high'] - df['low']).iloc[-22:-2].mean()
        if curr_range < avg_range * 0.8: return None
        
        rsi = self.calculate_rsi(df['close'].iloc[:-1])
        if signal['close'] <= signal['low'] + curr_range * 0.1 and rsi < 35:
            sl = signal['low'] - avg_range * 0.3
            tp = signal['low'] + curr_range * 0.8
            return Trade(symbol=symbol, direction="LONG", entry_price=current['close'],
                         sl=sl, tp=tp, confidence=0.75, atr=atr,
                         entry_time=current.name, bar_index=0, strategy="Pro_HrRangeRev")
        if signal['close'] >= signal['high'] - curr_range * 0.1 and rsi > 65:
            sl = signal['high'] + avg_range * 0.3
            tp = signal['high'] - curr_range * 0.8
            return Trade(symbol=symbol, direction="SHORT", entry_price=current['close'],
                         sl=sl, tp=tp, confidence=0.75, atr=atr,
                         entry_time=current.name, bar_index=0, strategy="Pro_HrRangeRev")
        return None

    def vwap_reversion_m5(self, df: pd.DataFrame, symbol: str) -> Optional[Trade]:
        """
        VWAP Reversion M5 (EURUSD, GBPUSD)
        Edge: 0.17R (High Freq)
        """
        if "EUR" not in symbol and "GBP" not in symbol: return None
        # Using shorter lookback for live optimization
        if len(df) < 100: return None
        
        last_time = df.index[-1]
        today = last_time.date()
        today_data = df[df.index.date == today]
        if len(today_data) < 10: return None
        
        tp = (today_data['high'] + today_data['low'] + today_data['close']) / 3
        pv = tp * today_data['tick_volume']
        vwap = (pv.cumsum() / today_data['tick_volume'].cumsum()).iloc[-1]
        
        atr = self.calculate_atr(df)
        band_dist = atr * 3.0
        row = df.iloc[-1]
        
        if row['close'] > vwap + band_dist:
            sl = row['high'] + atr * 0.5
            return Trade(symbol=symbol, direction="SHORT", entry_price=row['close'],
                         sl=sl, tp=vwap, confidence=0.82, atr=atr,
                         entry_time=row.name, bar_index=0, strategy="Pro_VWAP_M5")
        if row['close'] < vwap - band_dist:
            sl = row['low'] - atr * 0.5
            return Trade(symbol=symbol, direction="LONG", entry_price=row['close'],
                         sl=sl, tp=vwap, confidence=0.82, atr=atr,
                         entry_time=row.name, bar_index=0, strategy="Pro_VWAP_M5")
        return None

    # ===== DEPRECATED STRATEGIES (BELOW) =====
    # kept for reference but not in active list
    # Target: Expectancy >0.3R, PF >1.3, low daily DD

    def supply_demand_reversal(self, df: pd.DataFrame, symbol: str) -> Optional[Trade]:
        """
        Supply/Demand Zone Reversal Strategy
        
        Edge Source: Institutional order flow creates predictable reaction zones
        - Identifies fresh supply/demand zones from strong price moves
        - Waits for price to return to zone with specific confirmation
        - High win rate from mean reversion in zones
        
        Target: 65%+ win rate, 1.5:1 R:R = 0.47R expectancy
        """
        if len(df) < 100: return None
        
        row = df.iloc[-1]
        atr = self.calculate_atr(df)
        if atr == 0: return None
        
        # Find significant swing highs/lows in last 50 bars
        lookback = df.iloc[-50:-1]
        
        # Demand zone: lowest low in lookback with strong bullish follow-through
        demand_low = lookback['low'].min()
        demand_idx = lookback['low'].idxmin()
        
        # Supply zone: highest high in lookback with strong bearish follow-through
        supply_high = lookback['high'].max()
        supply_idx = lookback['high'].idxmax()
        
        # Check if zone is "fresh" (price hasn't returned to it yet)
        bars_since_demand = len(lookback) - lookback.index.get_loc(demand_idx) if demand_idx in lookback.index else 50
        bars_since_supply = len(lookback) - lookback.index.get_loc(supply_idx) if supply_idx in lookback.index else 50
        
        # Zone must be at least 10 bars old but less than 40 bars (still fresh)
        demand_fresh = 10 < bars_since_demand < 40
        supply_fresh = 10 < bars_since_supply < 40
        
        # BUY at demand zone (price touching demand zone from above)
        if demand_fresh and row['low'] <= demand_low + atr * 0.3:
            # Confirmation: RSI oversold or bullish candle pattern
            rsi = self.calculate_rsi(df['close'])
            bullish_candle = row['close'] > row['open'] and (row['close'] - row['open']) > atr * 0.3
            
            if rsi < 35 or bullish_candle:
                sl = demand_low - atr * 0.5
                tp = row['close'] + (row['close'] - sl) * 1.5  # 1.5:1 R:R
                
                return Trade(
                    symbol=symbol,
                    direction="LONG",
                    entry_price=row['close'],
                    sl=sl,
                    tp=tp,
                    confidence=0.85,
                    atr=atr,
                    entry_time=row.name,
                    bar_index=0,
                    strategy="Pro_SupplyDemand"
                )
        
        # SELL at supply zone (price touching supply zone from below)
        if supply_fresh and row['high'] >= supply_high - atr * 0.3:
            rsi = self.calculate_rsi(df['close'])
            bearish_candle = row['close'] < row['open'] and (row['open'] - row['close']) > atr * 0.3
            
            if rsi > 65 or bearish_candle:
                sl = supply_high + atr * 0.5
                tp = row['close'] - (sl - row['close']) * 1.5  # 1.5:1 R:R
                
                return Trade(
                    symbol=symbol,
                    direction="SHORT",
                    entry_price=row['close'],
                    sl=sl,
                    tp=tp,
                    confidence=0.85,
                    atr=atr,
                    entry_time=row.name,
                    bar_index=0,
                    strategy="Pro_SupplyDemand"
                )
        return None

    def session_breakout_retest(self, df: pd.DataFrame, symbol: str) -> Optional[Trade]:
        """
        Session Breakout + Retest Strategy
        
        Edge Source: Session highs/lows are key institutional levels
        - Identifies London/NY session range
        - Waits for breakout then retest of the range
        - Entry on bounce from prior resistance-turned-support (or vice versa)
        
        Target: 60%+ win rate, 2:1 R:R = 0.40R expectancy
        """
        base_symbol = symbol.replace('m', '')
        # Works best on major pairs and gold
        if base_symbol not in ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD']: return None
        
        if len(df) < 100: return None
        
        row = df.iloc[-1]
        hour = row.name.hour if hasattr(row.name, 'hour') else 12
        
        # Only trade during NY session (12:00-18:00 GMT)
        if not (12 <= hour < 18): return None
        
        atr = self.calculate_atr(df)
        if atr == 0: return None
        
        # Get London session range (06:00-12:00 GMT) - approx 72 M5 bars
        london_bars = df.iloc[-85:-13]  # Skip recent bars
        if len(london_bars) < 60: return None
        
        london_high = london_bars['high'].max()
        london_low = london_bars['low'].min()
        london_range = london_high - london_low
        
        # Need minimum range (avoid choppy days)
        if london_range < atr * 1.5: return None
        
        # LONG: Price broke above London high and is now retesting it
        if row['low'] <= london_high <= row['high'] and row['close'] > london_high:
            # Confirmation: price closed back above the level
            if row['close'] > london_high + atr * 0.1:
                sl = london_high - atr * 0.7
                tp = row['close'] + (row['close'] - sl) * 2.0  # 2:1 R:R
                
                return Trade(
                    symbol=symbol,
                    direction="LONG",
                    entry_price=row['close'],
                    sl=sl,
                    tp=tp,
                    confidence=0.80,
                    atr=atr,
                    entry_time=row.name,
                    bar_index=0,
                    strategy="Pro_SessionRetest"
                )
        
        # SHORT: Price broke below London low and is now retesting it
        if row['low'] <= london_low <= row['high'] and row['close'] < london_low:
            if row['close'] < london_low - atr * 0.1:
                sl = london_low + atr * 0.7
                tp = row['close'] - (sl - row['close']) * 2.0  # 2:1 R:R
                
                return Trade(
                    symbol=symbol,
                    direction="SHORT",
                    entry_price=row['close'],
                    sl=sl,
                    tp=tp,
                    confidence=0.80,
                    atr=atr,
                    entry_time=row.name,
                    bar_index=0,
                    strategy="Pro_SessionRetest"
                )
        return None

    def engulfing_momentum(self, df: pd.DataFrame, symbol: str) -> Optional[Trade]:
        """
        High-Probability Engulfing Pattern Strategy
        
        Edge Source: Engulfing patterns at key levels signal reversals
        - Only trades engulfing patterns at swing highs/lows
        - Requires momentum confirmation (EMA alignment)
        - Tight stops, wide targets
        
        Target: 55%+ win rate, 3:1 R:R = 0.40R expectancy
        """
        if len(df) < 60: return None
        
        row = df.iloc[-1]
        prev = df.iloc[-2]
        atr = self.calculate_atr(df)
        if atr == 0: return None
        
        # EMAs for trend context
        ema_20 = df['close'].ewm(span=20).mean().iloc[-1]
        ema_50 = df['close'].ewm(span=50).mean().iloc[-1]
        
        # Bullish Engulfing at swing low
        is_bullish_engulfing = (
            prev['close'] < prev['open'] and  # Previous candle bearish
            row['close'] > row['open'] and     # Current candle bullish
            row['close'] > prev['open'] and    # Closes above prev open
            row['open'] < prev['close'] and    # Opens below prev close
            (row['close'] - row['open']) > (prev['open'] - prev['close']) * 1.2  # 20% bigger body
        )
        
        # Bearish Engulfing at swing high
        is_bearish_engulfing = (
            prev['close'] > prev['open'] and  # Previous candle bullish
            row['close'] < row['open'] and     # Current candle bearish
            row['close'] < prev['open'] and    # Closes below prev open
            row['open'] > prev['close'] and    # Opens above prev close
            (row['open'] - row['close']) > (prev['close'] - prev['open']) * 1.2  # 20% bigger body
        )
        
        # Check if at swing low (lowest low in 20 bars)
        recent_lows = df['low'].iloc[-21:-1]
        at_swing_low = row['low'] <= recent_lows.min() * 1.002  # Within 0.2%
        
        # Check if at swing high (highest high in 20 bars)
        recent_highs = df['high'].iloc[-21:-1]
        at_swing_high = row['high'] >= recent_highs.max() * 0.998  # Within 0.2%
        
        # LONG: Bullish engulfing at swing low with EMA support
        if is_bullish_engulfing and at_swing_low:
            # Extra filter: price near or below lower EMA
            if row['close'] < ema_20 * 1.005:  # Within 0.5% of EMA20
                sl = row['low'] - atr * 0.3
                tp = row['close'] + (row['close'] - sl) * 3.0  # 3:1 R:R
                
                return Trade(
                    symbol=symbol,
                    direction="LONG",
                    entry_price=row['close'],
                    sl=sl,
                    tp=tp,
                    confidence=0.75,
                    atr=atr,
                    entry_time=row.name,
                    bar_index=0,
                    strategy="Pro_Engulfing"
                )
        
        # SHORT: Bearish engulfing at swing high with EMA resistance
        if is_bearish_engulfing and at_swing_high:
            if row['close'] > ema_20 * 0.995:  # Within 0.5% of EMA20
                sl = row['high'] + atr * 0.3
                tp = row['close'] - (sl - row['close']) * 3.0  # 3:1 R:R
                
                return Trade(
                    symbol=symbol,
                    direction="SHORT",
                    entry_price=row['close'],
                    sl=sl,
                    tp=tp,
                    confidence=0.75,
                    atr=atr,
                    entry_time=row.name,
                    bar_index=0,
                    strategy="Pro_Engulfing"
                )
        return None

    def power_three_setup(self, df: pd.DataFrame, symbol: str) -> Optional[Trade]:
        """
        ICT Power of 3 (Accumulation/Manipulation/Distribution)
        
        Edge Source: Smart money creates false moves before true direction
        - Identify accumulation phase (tight range)
        - Wait for manipulation (false breakout)
        - Enter on distribution in opposite direction
        
        Target: 50%+ win rate, 4:1 R:R = 0.50R expectancy
        """
        if len(df) < 80: return None
        
        row = df.iloc[-1]
        hour = row.name.hour if hasattr(row.name, 'hour') else 12
        
        # Best during London open (07:00-09:00 GMT) or NY open (12:00-14:00 GMT)
        is_session_open = (7 <= hour < 9) or (12 <= hour < 14)
        if not is_session_open: return None
        
        atr = self.calculate_atr(df)
        if atr == 0: return None
        
        # Phase 1: Find accumulation (tight range in prior session)
        # Look at last 4 hours (48 M5 bars)
        accumulation = df.iloc[-60:-12]
        if len(accumulation) < 40: return None
        
        acc_high = accumulation['high'].max()
        acc_low = accumulation['low'].min()
        acc_range = acc_high - acc_low
        
        # Range must be relatively tight (less than 2x ATR)
        if acc_range > atr * 2.5: return None
        
        # Phase 2: Check for manipulation (false breakout in last 12 bars)
        manipulation = df.iloc[-12:-1]
        
        # False breakout high (broke above acc_high then closed back inside)
        false_breakout_high = manipulation['high'].max() > acc_high + atr * 0.2
        back_inside_from_high = row['close'] < acc_high
        
        # False breakout low (broke below acc_low then closed back inside)
        false_breakout_low = manipulation['low'].min() < acc_low - atr * 0.2
        back_inside_from_low = row['close'] > acc_low
        
        # Phase 3: Enter distribution (opposite of manipulation)
        
        # LONG: False breakout low, now reversing up
        if false_breakout_low and back_inside_from_low:
            # Confirm with bullish close
            if row['close'] > row['open'] and row['close'] > acc_low + acc_range * 0.3:
                sl = manipulation['low'].min() - atr * 0.2
                tp = row['close'] + (row['close'] - sl) * 4.0  # 4:1 R:R
                
                return Trade(
                    symbol=symbol,
                    direction="LONG",
                    entry_price=row['close'],
                    sl=sl,
                    tp=tp,
                    confidence=0.70,
                    atr=atr,
                    entry_time=row.name,
                    bar_index=0,
                    strategy="Pro_Power3"
                )
        
        # SHORT: False breakout high, now reversing down
        if false_breakout_high and back_inside_from_high:
            if row['close'] < row['open'] and row['close'] < acc_high - acc_range * 0.3:
                sl = manipulation['high'].max() + atr * 0.2
                tp = row['close'] - (sl - row['close']) * 4.0  # 4:1 R:R
                
                return Trade(
                    symbol=symbol,
                    direction="SHORT",
                    entry_price=row['close'],
                    sl=sl,
                    tp=tp,
                    confidence=0.70,
                    atr=atr,
                    entry_time=row.name,
                    bar_index=0,
                    strategy="Pro_Power3"
                )
        return None
