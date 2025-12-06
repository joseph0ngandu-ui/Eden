#!/usr/bin/env python3
"""
LOOKAHEAD-FREE STRATEGIES
All strategies use CLOSED bars only for signals, entry at NEXT bar open.
Tested with realistic_verification.py methodology.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional, List
import logging
from trading.models import Trade, Position

logger = logging.getLogger(__name__)

class LookaheadFreeStrategies:
    """
    Strategies that don't use lookahead bias.
    All signals based on CLOSED bars, entry at NEXT bar open.
    """
    
    def __init__(self):
        self.strategies = [
            self.three_bar_reversal,
            self.ema_pullback_closed,
            self.inside_bar_breakout,
            self.momentum_continuation,
        ]
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        if len(df) < 2:
            return df['close'].iloc[-1] * 0.002 if not df.empty else 0.001
        tr = df['high'] - df['low']
        return tr.rolling(period).mean().iloc[-1] if len(tr) > period else tr.mean()
    
    def calculate_rsi(self, series: pd.Series, period: int = 14) -> float:
        try:
            delta = series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs.iloc[-1]))
        except:
            return 50.0

    def three_bar_reversal(self, df: pd.DataFrame, symbol: str) -> Optional[Dict]:
        """
        Three Bar Reversal (Morning Star / Evening Star variant)
        
        NO LOOKAHEAD: Uses CLOSED bars [-3], [-2], [-1] only.
        Entry: Next bar open (bar [0])
        
        Bullish: Bar1 bearish, Bar2 small body (indecision), Bar3 bullish close > Bar1 open
        Bearish: Bar1 bullish, Bar2 small body (indecision), Bar3 bearish close < Bar1 open
        """
        if len(df) < 50: return None
        
        # Use previous closed bars only
        bar1 = df.iloc[-3]  # First bar
        bar2 = df.iloc[-2]  # Indecision bar
        bar3 = df.iloc[-1]  # Confirmation bar (JUST CLOSED)
        
        atr = self.calculate_atr(df)
        if atr == 0: return None
        
        bar1_body = abs(bar1['close'] - bar1['open'])
        bar2_body = abs(bar2['close'] - bar2['open'])
        bar3_body = abs(bar3['close'] - bar3['open'])
        
        # Bar2 must be small (indecision)
        if bar2_body > bar1_body * 0.5:
            return None
        
        # BULLISH: Bar1 bearish, Bar3 bullish closing above Bar1 open
        bar1_bearish = bar1['close'] < bar1['open']
        bar3_bullish = bar3['close'] > bar3['open']
        bar3_strong = bar3['close'] > bar1['open']
        
        if bar1_bearish and bar3_bullish and bar3_strong:
            # Entry will be at NEXT bar open
            entry_estimate = bar3['close']  # For SL/TP calculation
            sl = min(bar1['low'], bar2['low'], bar3['low']) - atr * 0.3
            tp = entry_estimate + (entry_estimate - sl) * 2.0  # 2:1 R:R
            
            return {
                'direction': 'LONG',
                'sl': sl,
                'tp': tp,
                'atr': atr,
                'strategy': 'Pro_ThreeBarReversal'
            }
        
        # BEARISH: Bar1 bullish, Bar3 bearish closing below Bar1 open
        bar1_bullish = bar1['close'] > bar1['open']
        bar3_bearish = bar3['close'] < bar3['open']
        bar3_strong = bar3['close'] < bar1['open']
        
        if bar1_bullish and bar3_bearish and bar3_strong:
            entry_estimate = bar3['close']
            sl = max(bar1['high'], bar2['high'], bar3['high']) + atr * 0.3
            tp = entry_estimate - (sl - entry_estimate) * 2.0  # 2:1 R:R
            
            return {
                'direction': 'SHORT',
                'sl': sl,
                'tp': tp,
                'atr': atr,
                'strategy': 'Pro_ThreeBarReversal'
            }
        
        return None

    def ema_pullback_closed(self, df: pd.DataFrame, symbol: str) -> Optional[Dict]:
        """
        EMA Pullback Strategy (Closed Bar Confirmed)
        
        NO LOOKAHEAD: Uses EMA of CLOSED bars, entry when last CLOSED bar
        bounces off EMA.
        
        Entry: Next bar open after bounce confirmation
        """
        if len(df) < 55: return None
        
        # Calculate EMAs on closed data
        ema_20 = df['close'].ewm(span=20).mean()
        ema_50 = df['close'].ewm(span=50).mean()
        
        ema_20_current = ema_20.iloc[-1]
        ema_50_current = ema_50.iloc[-1]
        ema_20_prev = ema_20.iloc[-2]
        ema_50_prev = ema_50.iloc[-2]
        
        bar = df.iloc[-1]  # Last CLOSED bar
        prev_bar = df.iloc[-2]
        
        atr = self.calculate_atr(df)
        if atr == 0: return None
        
        rsi = self.calculate_rsi(df['close'])
        
        # Uptrend: EMA20 > EMA50
        in_uptrend = ema_20_current > ema_50_current and ema_20_prev > ema_50_prev
        
        # Downtrend: EMA20 < EMA50
        in_downtrend = ema_20_current < ema_50_current and ema_20_prev < ema_50_prev
        
        # LONG: Uptrend + previous bar dipped to EMA20 + current bar closed bullish
        if in_uptrend:
            # Previous bar touched EMA20 (pullback)
            touched_ema = prev_bar['low'] <= ema_20.iloc[-2] * 1.002
            # Current bar closed bullish and above EMA20
            bullish_bounce = bar['close'] > bar['open'] and bar['close'] > ema_20_current
            # RSI not overbought
            rsi_ok = 40 < rsi < 70
            
            if touched_ema and bullish_bounce and rsi_ok:
                sl = min(prev_bar['low'], bar['low']) - atr * 0.3
                entry_estimate = bar['close']
                tp = entry_estimate + (entry_estimate - sl) * 2.5  # 2.5:1 R:R
                
                return {
                    'direction': 'LONG',
                    'sl': sl,
                    'tp': tp,
                    'atr': atr,
                    'strategy': 'Pro_EMAPullback'
                }
        
        # SHORT: Downtrend + previous bar spiked to EMA20 + current bar closed bearish
        if in_downtrend:
            touched_ema = prev_bar['high'] >= ema_20.iloc[-2] * 0.998
            bearish_bounce = bar['close'] < bar['open'] and bar['close'] < ema_20_current
            rsi_ok = 30 < rsi < 60
            
            if touched_ema and bearish_bounce and rsi_ok:
                sl = max(prev_bar['high'], bar['high']) + atr * 0.3
                entry_estimate = bar['close']
                tp = entry_estimate - (sl - entry_estimate) * 2.5
                
                return {
                    'direction': 'SHORT',
                    'sl': sl,
                    'tp': tp,
                    'atr': atr,
                    'strategy': 'Pro_EMAPullback'
                }
        
        return None

    def inside_bar_breakout(self, df: pd.DataFrame, symbol: str) -> Optional[Dict]:
        """
        Inside Bar Breakout Strategy
        
        NO LOOKAHEAD: Identifies inside bar pattern from CLOSED bars,
        enters when the bar AFTER inside bar closes outside mother bar range.
        
        Pattern: Mother bar (large) -> Inside bar (contained) -> Breakout bar (closes outside)
        Entry: Next bar open after breakout confirmation
        """
        if len(df) < 30: return None
        
        mother = df.iloc[-3]
        inside = df.iloc[-2]
        breakout = df.iloc[-1]  # Just closed
        
        atr = self.calculate_atr(df)
        if atr == 0: return None
        
        # Check inside bar pattern: inside bar contained within mother bar
        is_inside = (inside['high'] < mother['high']) and (inside['low'] > mother['low'])
        
        if not is_inside:
            return None
        
        mother_range = mother['high'] - mother['low']
        
        # Require meaningful range
        if mother_range < atr * 0.5:
            return None
        
        # LONG: Breakout bar CLOSED above mother high
        if breakout['close'] > mother['high']:
            # Confirm breakout is strong
            if breakout['close'] > mother['high'] + atr * 0.1:
                sl = inside['low'] - atr * 0.2
                entry_estimate = breakout['close']
                tp = entry_estimate + (entry_estimate - sl) * 2.0
                
                return {
                    'direction': 'LONG',
                    'sl': sl,
                    'tp': tp,
                    'atr': atr,
                    'strategy': 'Pro_InsideBreakout'
                }
        
        # SHORT: Breakout bar CLOSED below mother low
        if breakout['close'] < mother['low']:
            if breakout['close'] < mother['low'] - atr * 0.1:
                sl = inside['high'] + atr * 0.2
                entry_estimate = breakout['close']
                tp = entry_estimate - (sl - entry_estimate) * 2.0
                
                return {
                    'direction': 'SHORT',
                    'sl': sl,
                    'tp': tp,
                    'atr': atr,
                    'strategy': 'Pro_InsideBreakout'
                }
        
        return None

    def momentum_continuation(self, df: pd.DataFrame, symbol: str) -> Optional[Dict]:
        """
        Momentum Continuation Strategy
        
        NO LOOKAHEAD: Uses closed bar momentum to identify trend continuation.
        Entry after 3 consecutive bars in same direction with pullback.
        
        Entry: Next bar open after pullback bar closes
        """
        if len(df) < 30: return None
        
        atr = self.calculate_atr(df)
        if atr == 0: return None
        
        # Check momentum: 3 consecutive bullish/bearish bars before pullback
        bars = [df.iloc[-i] for i in range(5, 0, -1)]  # bars[-5] to bars[-1]
        
        # Bullish momentum: 3 bullish bars followed by 1 pullback bar, then bullish continuation
        momentum_bullish = all(bars[i]['close'] > bars[i]['open'] for i in range(3))
        pullback = bars[3]['close'] < bars[3]['open']  # Pullback bar
        continuation = bars[4]['close'] > bars[4]['open']  # Continuation bar (just closed)
        
        if momentum_bullish and pullback and continuation:
            # Confirm continuation is strong
            if bars[4]['close'] > bars[3]['high']:
                sl = bars[3]['low'] - atr * 0.3
                entry_estimate = bars[4]['close']
                tp = entry_estimate + (entry_estimate - sl) * 2.0
                
                return {
                    'direction': 'LONG',
                    'sl': sl,
                    'tp': tp,
                    'atr': atr,
                    'strategy': 'Pro_MomentumCont'
                }
        
        # Bearish momentum: 3 bearish bars followed by 1 pullback bar, then bearish continuation
        momentum_bearish = all(bars[i]['close'] < bars[i]['open'] for i in range(3))
        pullback = bars[3]['close'] > bars[3]['open']
        continuation = bars[4]['close'] < bars[4]['open']
        
        if momentum_bearish and pullback and continuation:
            if bars[4]['close'] < bars[3]['low']:
                sl = bars[3]['high'] + atr * 0.3
                entry_estimate = bars[4]['close']
                tp = entry_estimate - (sl - entry_estimate) * 2.0
                
                return {
                    'direction': 'SHORT',
                    'sl': sl,
                    'tp': tp,
                    'atr': atr,
                    'strategy': 'Pro_MomentumCont'
                }
        
        return None
