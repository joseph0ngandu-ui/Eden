"""
Pro Strategy Engine - Prop Firm Certified
Tuned for Optimal Risk: H1 Trend (Stable) + M5 Reversion (Cash Flow) + M15 Spread Hunter (Gold) + M15 Index Volatility
Risk Limits: Daily < 4.5%, Max < 9.5%
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
        self.strategies = [
            # 1. Volatility Squeeze (Forex Trend)
            # FILTERED: Only EURUSD, USDJPY. (GBP removed due to losses).
            self.volatility_squeeze,      
            
            # REMOVED: Quiet Before Storm (Failed Audit: -7R)
            # REMOVED: Triple Candle Breakout (Failed Audit: -14R)
            # REMOVED: VWAP Reversion (Excessive Churn: 4000+ trades, commission risk)
            
            # 2. Spread Hunter (Gold Only)
            self.spread_hunter_momentum,  # Cost-Exploiting Momentum (M15)
            
            # 3. Index Volatility Expansion (US30/USTEC/US500)
            self.index_volatility_expansion, # Squeeze Breakout (M15)
            
            # 4. Momentum Continuation (D1) - APPROVED Phase 5
            # Pairs: USDCAD, EURUSD, EURJPY, CADJPY
            self.momentum_continuation,
            
            # London Breakout: RESERVED (GBPCADm +34.7R, DD 10.8R)
        ]
        self.open_positions: Dict[str, Position] = {}
        self.cooldown_minutes = 15
        self.last_trade_time = {}
        
        # State Tracking
        self.spread_history: Dict[str, List[float]] = {}
        self.bandwidth_history: Dict[str, List[float]] = {}
    
    # ... (adopt_position, calculate_atr, etc. remain same) ...

    # [OMITTED METHODS FOR BREVITY - PRESERVE EXISTING]


        
    def adopt_position(self, pos):
        """Adopt an existing MT5 position."""
        try:
            direction = "LONG" if pos.type == 0 else "SHORT"
            self.open_positions[pos.symbol] = Position(
                symbol=pos.symbol,
                direction=direction,
                entry_price=pos.price_open,
                tp=pos.tp,
                sl=pos.sl,
                entry_bar_index=0, 
                entry_time=datetime.fromtimestamp(pos.time),
                atr=0.0, 
                confidence=0.0, 
                strategy="Recovered"
            )
            logger.info(f"Adopted position: {pos.symbol} ({direction})")
        except Exception as e:
            logger.error(f"Error adopting position {pos.ticket}: {e}")

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        try:
            if len(df) < 2: 
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

    def evaluate_live(self, df: pd.DataFrame, symbol: str, timeframe: int = 5) -> Optional[Trade]:
        """
        Get signal for a symbol based on applicable strategies and timeframe.
        - Legacy strategies run on M5.
        - Spread Hunter & Index Vol run on M15.
        """
        if df is None or df.empty: return None
        if len(df) < 100: return None
            
        # Global Rollover Filter
        if hasattr(df.index, 'hour'):
            current_hour = df.index[-1].hour
        else:
            current_hour = pd.to_datetime(df['time'].iloc[-1]).hour
            
        if 21 <= current_hour < 22: return None 
            
        for strategy in self.strategies:
            # === TIMEFRAME ROUTING ===
            if strategy.__name__ == 'spread_hunter_momentum':
                if timeframe != 15: continue
                if 'XAU' not in symbol: continue 
            elif strategy.__name__ == 'index_volatility_expansion':
                if timeframe != 15: continue
                if 'US' not in symbol and 'USTEC' not in symbol: continue 
            elif strategy.__name__ == 'momentum_continuation':
                if timeframe != 1440: continue  # D1 only
                # Filtered pairs
                if symbol not in ['USDCADm', 'EURUSDm', 'EURJPYm', 'CADJPYm']: continue
            # London Breakout routing REMOVED (Strategy Rejected)
            else:
                # All legacy strategies are M5
                if timeframe != 5: continue
                # Restrict legacy strategies from running on Indices (if passed M5)
                if 'US30' in symbol or 'USTEC' in symbol or 'US500' in symbol: continue
            
            # Specific Filter for M5 VWAP
            if strategy.__name__ == 'vwap_reversion_m5':
                 if current_hour >= 20: continue
                 
            try:
                signal = strategy(df, symbol)
                if signal:
                    logger.info(f"SIGNAL FOUND: {symbol} via {strategy.__name__} (TF: {timeframe}m)")
                    return signal
            except Exception as e:
                logger.error(f"Strategy error ({strategy.__name__}) on {symbol}: {e}")
                continue
        return None

    def manage_position(self, df: pd.DataFrame, symbol: str) -> List[Dict]:
        """Manage open positions."""
        if symbol not in self.open_positions: return []
            
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

    # ===== VERIFIED ACTIVE STRATEGIES =====



    def volatility_squeeze(self, df: pd.DataFrame, symbol: str) -> Optional[Trade]:
        # FILTER: Only EURUSD and USDJPY proved profitable in audit.
        if "EUR" not in symbol and "JPY" not in symbol: return None
        # Exclude Indices explicitly (safety)
        if "US30" in symbol or "USTEC" in symbol or "US500" in symbol: return None
        
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



    def spread_hunter_momentum(self, df: pd.DataFrame, symbol: str) -> Optional[Trade]:
        if 'XAU' not in symbol: return None
        if len(df) < 50: return None
        current_bar = df.iloc[-1]
        if 'spread' in df.columns:
            current_spread = current_bar['spread']
        else:
            return None
        if symbol not in self.spread_history:
            self.spread_history[symbol] = []
        self.spread_history[symbol].append(current_spread)
        if len(self.spread_history[symbol]) > 100:
            self.spread_history[symbol].pop(0)
        avg_spread = np.mean(self.spread_history[symbol])
        if current_spread > avg_spread * 0.85:
            return None
        prev = df.iloc[-2]
        atr = self.calculate_atr(df)
        ema_10 = df['close'].ewm(span=10).mean()
        ema_30 = df['close'].ewm(span=30).mean()
        bullish_trend = ema_10.iloc[-1] > ema_30.iloc[-1] and ema_10.iloc[-5] > ema_30.iloc[-5]
        bearish_trend = ema_10.iloc[-1] < ema_30.iloc[-1] and ema_10.iloc[-5] < ema_30.iloc[-5]
        pullback_low = prev['low'] < ema_10.iloc[-2] and current_bar['close'] > ema_10.iloc[-1]
        pullback_high = prev['high'] > ema_10.iloc[-2] and current_bar['close'] < ema_10.iloc[-1]
        if bullish_trend and pullback_low:
            sl = df['low'].iloc[-5:].min() - atr * 0.3
            tp = current_bar['close'] + (current_bar['close'] - sl) * 1.5
            return Trade(symbol=symbol, direction="LONG", entry_price=current_bar['close'],
                         sl=sl, tp=tp, confidence=0.75, atr=atr,
                         entry_time=current_bar.name, bar_index=0, strategy="SpreadHunter_M15")
        if bearish_trend and pullback_high:
            sl = df['high'].iloc[-5:].max() + atr * 0.3
            tp = current_bar['close'] - (sl - current_bar['close']) * 1.5
            return Trade(symbol=symbol, direction="SHORT", entry_price=current_bar['close'],
                         sl=sl, tp=tp, confidence=0.75, atr=atr,
                         entry_time=current_bar.name, bar_index=0, strategy="SpreadHunter_M15")
        return None

    def index_volatility_expansion(self, df: pd.DataFrame, symbol: str) -> Optional[Trade]:
        """Index Volatility Expansion (US30/USTEC) M15"""
        if 'US' not in symbol and 'USTEC' not in symbol: return None
        if len(df) < 50: return None
        
        # Squeeze Logic
        closes = df['close']
        sma_20 = closes.rolling(20).mean()
        std_20 = closes.rolling(20).std()
        upper = sma_20 + (std_20 * 2.0)
        lower = sma_20 - (std_20 * 2.0)
        bandwidth = (upper - lower) / sma_20
        
        current_bw = bandwidth.iloc[-1]
        if symbol not in self.bandwidth_history:
            self.bandwidth_history[symbol] = []
        self.bandwidth_history[symbol].append(current_bw)
        if len(self.bandwidth_history[symbol]) > 200:
            self.bandwidth_history[symbol].pop(0)
            
        avg_bw = np.mean(self.bandwidth_history[symbol])
        
        # Check for recent squeeze
        recent_squeeze = False
        for i in range(1, 4):
            if bandwidth.iloc[-i-1] < avg_bw * 0.8:
                recent_squeeze = True
                break
        if not recent_squeeze: return None # No setup
        
        # Breakout
        current_bar = df.iloc[-1]
        prev_bar = df.iloc[-2]
        current_price = current_bar['close']
        
        # NY Open Filter
        # Assuming df time index or col
        hour = pd.to_datetime(df.index[-1]).hour if hasattr(df.index, 'hour') else pd.to_datetime(current_bar.name).hour
        is_ny = 13 <= hour <= 20
        if not is_ny: return None

        # Trend Filter
        ema_50 = closes.rolling(50).mean().iloc[-1]
        atr = self.calculate_atr(df)
        
        long_breakout = current_price > upper.iloc[-1] and current_price > ema_50
        short_breakout = current_price < lower.iloc[-1] and current_price < ema_50
        
        if long_breakout:
            sl = sma_20.iloc[-1] # Middle Band
            tp = current_price + (current_price - sl) * 1.5
            return Trade(symbol=symbol, direction="LONG", entry_price=current_price,
                         sl=sl, tp=tp, confidence=0.70, atr=atr,
                         entry_time=current_bar.name, bar_index=0, strategy="Index_VolExpansion")
                         
        if short_breakout:
            sl = sma_20.iloc[-1]
            tp = current_price - (sl - current_price) * 1.5
            return Trade(symbol=symbol, direction="SHORT", entry_price=current_price,
                         sl=sl, tp=tp, confidence=0.70, atr=atr,
                         entry_time=current_bar.name, bar_index=0, strategy="Index_VolExpansion")
        return None

    def london_breakout(self, df: pd.DataFrame, symbol: str) -> Optional[Trade]:
        """London Session Breakout (EURUSD/GBPJPY) M15."""
        # Symbol Filter: Only profitable pairs
        if 'EUR' not in symbol and 'GBPJPY' not in symbol: return None
        if 'GBP' in symbol and 'JPY' not in symbol: return None # Exclude GBPUSD
        if len(df) < 100: return None
        
        current = df.iloc[-1]
        timestamp = current.name
        
        # Time Filter: London Entry (08:00 - 11:00 Server Time)
        hour = pd.to_datetime(timestamp).hour
        if not (8 <= hour <= 11): return None
        
        # Define Asian Range (00:00 - 08:00 Server)
        today_date = timestamp.date()
        from datetime import time as dt_time
        range_start = pd.Timestamp.combine(today_date, dt_time(0, 0))
        range_end = pd.Timestamp.combine(today_date, dt_time(8, 0))
        
        range_data = df[(df.index >= range_start) & (df.index < range_end)]
        if len(range_data) < 10: return None
        
        range_high = range_data['high'].max()
        range_low = range_data['low'].min()
        
        # ADR Filter (Prevent over-extended ranges)
        atr = self.calculate_atr(df)
        range_width = range_high - range_low
        if range_width > atr * 3.0: return None # Too wide
        
        # EMA Trend
        ema_50 = df['close'].ewm(span=50).mean()
        trend_up = current['close'] > ema_50.iloc[-1]
        trend_down = current['close'] < ema_50.iloc[-1]
        
        price = current['close']
        
        # Breakout Detection
        if price > range_high and trend_up:
            sl = range_low - atr * 0.3
            risk = price - sl
            tp = price + risk * 1.5
            return Trade(symbol=symbol, direction="LONG", entry_price=price,
                         sl=sl, tp=tp, confidence=0.70, atr=atr,
                         entry_time=timestamp, bar_index=0, strategy="London_Breakout")
        
        if price < range_low and trend_down:
            sl = range_high + atr * 0.3
            risk = sl - price
            tp = price - risk * 1.5
            return Trade(symbol=symbol, direction="SHORT", entry_price=price,
                         sl=sl, tp=tp, confidence=0.70, atr=atr,
                         entry_time=timestamp, bar_index=0, strategy="London_Breakout")
        return None

    def momentum_continuation(self, df: pd.DataFrame, symbol: str) -> Optional[Trade]:
        """
        Momentum Continuation (D1) - APPROVED Phase 5
        Logic: Enter pullback next day after strong D1 candle (>1.3x ADR).
        Pairs: USDCAD, EURUSD, EURJPY, CADJPY
        """
        # Filtered pairs check (also in routing, but double-check)
        if symbol not in ['USDCADm', 'EURUSDm', 'EURJPYm', 'CADJPYm']: return None
        if len(df) < 20: return None
        
        # Calculate ADR (14-day)
        daily_ranges = df['high'] - df['low']
        adr = daily_ranges.rolling(14).mean()
        
        # Yesterday's candle (df.iloc[-2]) vs Today open (df.iloc[-1])
        if len(df) < 2: return None
        yesterday = df.iloc[-2]
        today = df.iloc[-1]
        
        if pd.isna(adr.iloc[-2]) or adr.iloc[-2] == 0: return None
        
        # Strong day check: Range > 1.3x ADR
        yest_range = yesterday['high'] - yesterday['low']
        if yest_range < 1.3 * adr.iloc[-2]: return None
        
        # Direction
        bullish = yesterday['close'] > yesterday['open']
        entry = today['open']
        atr = self.calculate_atr(df)
        
        if bullish:
            sl = yesterday['low']
            risk = entry - sl
            if risk <= 0: return None
            tp = entry + risk * 1.5
            return Trade(symbol=symbol, direction="LONG", entry_price=entry,
                         sl=sl, tp=tp, confidence=0.75, atr=atr,
                         entry_time=today.name, bar_index=0, strategy="Momentum_Continuation")
        else:
            sl = yesterday['high']
            risk = sl - entry
            if risk <= 0: return None
            tp = entry - risk * 1.5
            return Trade(symbol=symbol, direction="SHORT", entry_price=entry,
                         sl=sl, tp=tp, confidence=0.75, atr=atr,
                         entry_time=today.name, bar_index=0, strategy="Momentum_Continuation")

