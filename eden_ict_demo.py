#!/usr/bin/env python3
"""
Eden ICT Demo System
===================

Demonstration of complete ICT strategy with all confluences:
- Unified ICT Strategy (Liquidity + FVG + OB + OTE + Judas)
- Full top-down analysis (D1→H4→H1→M15)
- Against-bias trading with reversal detection
- Uses synthetic data for demonstration

Author: Eden AI System
Version: Demo 1.0
Date: September 14, 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional

warnings.filterwarnings('ignore')

@dataclass
class MarketRegime:
    regime: str
    confidence: float
    volatility: float
    direction: int

@dataclass  
class HTFBias:
    daily: str
    h4: str
    h1: str
    overall: str
    confidence: float
    reversal_probability: float

@dataclass
class Signal:
    timestamp: datetime
    symbol: str
    side: str
    confidence: float
    strategy_name: str
    strategy_family: str
    entry_price: float
    htf_bias: Optional[str] = None
    against_bias: bool = False
    reversal_signal: bool = False
    confluences: Optional[Dict] = None
    regime: Optional[str] = None

class SyntheticDataGenerator:
    """Generate realistic market data for demo"""
    
    def generate_symbol_data(self, symbol: str, bars: int, timeframe: str) -> pd.DataFrame:
        """Generate synthetic OHLC data with realistic patterns"""
        np.random.seed(hash(symbol + timeframe) % 2**32)
        
        # Base price levels for different symbols
        base_prices = {
            'XAUUSD': 1950.0,
            'EURUSD': 1.0650,
            'GBPUSD': 1.2500,
            'USDJPY': 150.0,
            'USDCHF': 0.9000
        }
        
        base_price = base_prices.get(symbol, 1.0000)
        
        # Generate price movements with trends and reversals
        dates = pd.date_range(
            start=datetime.now() - timedelta(hours=bars),
            periods=bars,
            freq='1H' if timeframe == 'H1' else ('4H' if timeframe == 'H4' else '1D')
        )
        
        # Create price series with realistic patterns
        price_changes = np.random.normal(0, base_price * 0.002, bars)
        
        # Add trending periods
        trend_periods = []
        for i in range(0, bars, random.randint(20, 50)):
            trend_length = min(random.randint(15, 30), bars - i)
            trend_strength = random.choice([1, -1]) * random.uniform(0.0005, 0.002)
            trend_periods.extend([trend_strength] * trend_length)
        
        if len(trend_periods) < bars:
            trend_periods.extend([0] * (bars - len(trend_periods)))
        trend_periods = trend_periods[:bars]
        
        # Combine random walk with trends
        cumulative_changes = np.cumsum(price_changes + trend_periods)
        close_prices = base_price + cumulative_changes
        
        # Generate OHLC from close prices
        data = []
        for i, close in enumerate(close_prices):
            if i == 0:
                open_price = base_price
            else:
                open_price = close_prices[i-1]
            
            volatility = abs(price_changes[i]) * random.uniform(2, 4)
            high = close + volatility * random.uniform(0.3, 0.8)
            low = close - volatility * random.uniform(0.3, 0.8)
            
            # Ensure OHLC logic
            high = max(high, open_price, close)
            low = min(low, open_price, close)
            
            data.append({
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'tick_volume': random.randint(1000, 5000)
            })
        
        df = pd.DataFrame(data, index=dates)
        return self._add_features(df)
    
    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators"""
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Moving averages
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_21'] = df['close'].ewm(span=21).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        df['ema_50'] = df['close'].ewm(span=50).mean()
        df['sma_200'] = df['close'].rolling(window=200).mean()
        
        # ATR
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(window=14).mean()
        
        return df.fillna(method='ffill').fillna(0)

class HTFAnalyzer:
    """Higher timeframe bias analyzer"""
    
    def analyze_full_htf_bias(self, daily_data: pd.DataFrame, h4_data: pd.DataFrame,
                             h1_data: pd.DataFrame, m15_data: pd.DataFrame,
                             m5_data: pd.DataFrame) -> HTFBias:
        """Full top-down analysis"""
        
        daily_bias = self._get_timeframe_bias(daily_data.iloc[-20:], "D1")
        h4_bias = self._get_timeframe_bias(h4_data.iloc[-50:], "H4")
        h1_bias = self._get_timeframe_bias(h1_data.iloc[-100:], "H1")
        
        # Overall bias weighting
        bias_scores = {'bullish': 0, 'bearish': 0, 'neutral': 0}
        weights = {'daily': 0.4, 'h4': 0.3, 'h1': 0.3}
        biases = {'daily': daily_bias, 'h4': h4_bias, 'h1': h1_bias}
        
        for tf, bias in biases.items():
            bias_scores[bias] += weights[tf]
        
        overall_bias = max(bias_scores, key=bias_scores.get)
        confidence = bias_scores[overall_bias]
        
        reversal_prob = self._calculate_reversal_probability(daily_data, h4_data, h1_data)
        
        return HTFBias(
            daily=daily_bias,
            h4=h4_bias,
            h1=h1_bias,
            overall=overall_bias,
            confidence=confidence,
            reversal_probability=reversal_prob
        )
    
    def _get_timeframe_bias(self, data: pd.DataFrame, timeframe: str) -> str:
        """Get bias for specific timeframe"""
        if len(data) < 10:
            return 'neutral'
        
        recent_data = data.iloc[-10:]
        
        # Structure analysis
        hh = recent_data['high'].iloc[-1] > recent_data['high'].iloc[:-1].max()
        hl = recent_data['low'].iloc[-1] > recent_data['low'].iloc[:-1].min()
        ll = recent_data['low'].iloc[-1] < recent_data['low'].iloc[:-1].min()
        lh = recent_data['high'].iloc[-1] < recent_data['high'].iloc[:-1].max()
        
        # EMA alignment
        ema_bullish = (recent_data['ema_12'].iloc[-1] > recent_data['ema_26'].iloc[-1] > 
                      recent_data['ema_50'].iloc[-1])
        ema_bearish = (recent_data['ema_12'].iloc[-1] < recent_data['ema_26'].iloc[-1] < 
                      recent_data['ema_50'].iloc[-1])
        
        # 200 SMA position
        above_200 = recent_data['close'].iloc[-1] > recent_data['sma_200'].iloc[-1]
        below_200 = recent_data['close'].iloc[-1] < recent_data['sma_200'].iloc[-1]
        
        bullish_signals = sum([hh, hl, ema_bullish, above_200])
        bearish_signals = sum([ll, lh, ema_bearish, below_200])
        
        if bullish_signals >= 3:
            return 'bullish'
        elif bearish_signals >= 3:
            return 'bearish'
        else:
            return 'neutral'
    
    def _calculate_reversal_probability(self, daily: pd.DataFrame, h4: pd.DataFrame,
                                      h1: pd.DataFrame) -> float:
        """Calculate reversal probability"""
        reversal_signals = 0
        total_signals = 0
        
        if len(h1) >= 50:
            recent_h1 = h1.iloc[-20:]
            
            # RSI divergence
            if len(recent_h1) >= 10:
                price_trend = recent_h1['close'].iloc[-1] > recent_h1['close'].iloc[-10]
                rsi_trend = recent_h1['rsi_14'].iloc[-1] > recent_h1['rsi_14'].iloc[-10]
                if price_trend != rsi_trend:
                    reversal_signals += 2
            total_signals += 2
            
            # Volume divergence
            if 'tick_volume' in recent_h1.columns and len(recent_h1) >= 15:
                price_higher = recent_h1['close'].iloc[-1] > recent_h1['close'].iloc[-10]
                volume_lower = (recent_h1['tick_volume'].iloc[-5:].mean() < 
                               recent_h1['tick_volume'].iloc[-15:-5].mean())
                if price_higher and volume_lower:
                    reversal_signals += 1
            total_signals += 1
        
        return reversal_signals / max(total_signals, 1)

class ICTStrategy:
    """Complete ICT Strategy with all confluences"""
    
    def __init__(self):
        self.name = "ict_confluence"
        self.family = "ICT"
        self.confidence_threshold = 0.7
        self.params = {
            "min_confluences": 3,
            "liquidity_lookback": 20,
            "fvg_min_size": 0.0001,
            "ob_min_size": 0.0002,
            "ote_fib_levels": [0.618, 0.705, 0.786],
            "judas_sessions": [8, 13],
        }
    
    def generate_signals(self, daily_data: pd.DataFrame, h4_data: pd.DataFrame,
                        h1_data: pd.DataFrame, m15_data: pd.DataFrame, primary_data: pd.DataFrame,
                        htf_bias: HTFBias, regimes: List[MarketRegime]) -> List[Signal]:
        """Generate ICT signals with confluence analysis"""
        signals = []
        
        print(f"   🎯 ICT analyzing {len(primary_data)} H1 bars with full confluence...")
        
        for i in range(200, min(len(primary_data), len(regimes))):
            current_regime = regimes[i]
            current_bar = primary_data.iloc[i]
            timestamp = current_bar.name
            
            # Get corresponding HTF data
            daily_row = self._get_htf_data_at_time(daily_data, timestamp)
            h4_row = self._get_htf_data_at_time(h4_data, timestamp)
            h1_row = self._get_htf_data_at_time(h1_data, timestamp)
            m15_row = self._get_htf_data_at_time(m15_data, timestamp)
            
            if any(row is None for row in [daily_row, h4_row, h1_row, m15_row]):
                continue
            
            # ICT Confluence Analysis
            confluences = self._analyze_ict_confluences(
                primary_data, i, daily_row, h4_row, h1_row, m15_row, current_bar
            )
            
            # Count valid confluences
            confluence_count = sum(1 for conf in confluences.values() if conf.get('valid', False))
            
            if confluence_count >= self.params['min_confluences']:
                # Determine signal direction
                bullish_weight = sum(conf['weight'] for conf in confluences.values()
                                   if conf.get('valid', False) and conf.get('direction') == 'bullish')
                bearish_weight = sum(conf['weight'] for conf in confluences.values()
                                   if conf.get('valid', False) and conf.get('direction') == 'bearish')
                
                if bullish_weight > bearish_weight:
                    side = "buy"
                    confidence = min(bullish_weight / 10.0, 0.95)
                    against_bias = htf_bias.overall == 'bearish'
                elif bearish_weight > bullish_weight:
                    side = "sell"
                    confidence = min(bearish_weight / 10.0, 0.95)
                    against_bias = htf_bias.overall == 'bullish'
                else:
                    continue
                
                # Check if we can trade
                can_trade = (confidence >= self.confidence_threshold or
                           (against_bias and htf_bias.reversal_probability > 0.6))
                
                if can_trade:
                    # Check for reversal signals
                    reversal_signal = (confluences.get('liquidity_sweep', {}).get('valid', False) or
                                     confluences.get('judas_swing', {}).get('valid', False))
                    
                    signal = Signal(
                        timestamp=timestamp,
                        symbol="",
                        side=side,
                        confidence=confidence,
                        strategy_name=self.name,
                        strategy_family=self.family,
                        entry_price=current_bar['close'],
                        htf_bias=htf_bias.overall,
                        against_bias=against_bias,
                        reversal_signal=reversal_signal,
                        confluences=confluences,
                        regime=current_regime.regime
                    )
                    signals.append(signal)
        
        return signals
    
    def _get_htf_data_at_time(self, htf_data: pd.DataFrame, timestamp: datetime) -> Optional[pd.Series]:
        """Get higher timeframe data at specific time"""
        try:
            available_times = htf_data.index[htf_data.index <= timestamp]
            if len(available_times) == 0:
                return None
            return htf_data.loc[available_times[-1]]
        except:
            return None
    
    def _analyze_ict_confluences(self, data: pd.DataFrame, current_idx: int,
                               daily_row: pd.Series, h4_row: pd.Series, h1_row: pd.Series,
                               m15_row: pd.Series, current_bar: pd.Series) -> Dict:
        """Analyze all ICT confluences"""
        confluences = {}
        
        # 1. Liquidity Sweep Analysis
        confluences['liquidity_sweep'] = self._check_liquidity_sweep(data, current_idx)
        
        # 2. Fair Value Gap Analysis
        confluences['fair_value_gap'] = self._check_fair_value_gap(data, current_idx)
        
        # 3. Order Block Analysis
        confluences['order_block'] = self._check_order_block(data, current_idx)
        
        # 4. Optimal Trade Entry Analysis
        confluences['optimal_trade_entry'] = self._check_ote(data, current_idx)
        
        # 5. Judas Swing Analysis
        confluences['judas_swing'] = self._check_judas_swing(data, current_idx, current_bar)
        
        # 6. HTF Structure Confluence
        confluences['htf_structure'] = self._check_htf_structure(daily_row, h4_row, h1_row)
        
        # 7. Session Confluence
        confluences['session'] = self._check_session_confluence(current_bar)
        
        return confluences
    
    def _check_liquidity_sweep(self, data: pd.DataFrame, idx: int) -> Dict:
        """Check for liquidity sweep patterns"""
        if idx < self.params['liquidity_lookback']:
            return {'valid': False, 'direction': None, 'weight': 0}
        
        lookback_data = data.iloc[idx-self.params['liquidity_lookback']:idx]
        current_bar = data.iloc[idx]
        
        recent_high = lookback_data['high'].max()
        recent_low = lookback_data['low'].min()
        
        # Check for sweep above recent high (bearish reversal expected)
        swept_high = current_bar['high'] > recent_high * 1.0005
        closed_below = current_bar['close'] < recent_high * 0.999
        
        # Check for sweep below recent low (bullish reversal expected)
        swept_low = current_bar['low'] < recent_low * 0.9995
        closed_above = current_bar['close'] > recent_low * 1.001
        
        if swept_high and closed_below:
            return {'valid': True, 'direction': 'bearish', 'weight': 3, 'type': 'high_sweep'}
        elif swept_low and closed_above:
            return {'valid': True, 'direction': 'bullish', 'weight': 3, 'type': 'low_sweep'}
        
        return {'valid': False, 'direction': None, 'weight': 0}
    
    def _check_fair_value_gap(self, data: pd.DataFrame, idx: int) -> Dict:
        """Check for Fair Value Gap fills"""
        if idx < 3:
            return {'valid': False, 'direction': None, 'weight': 0}
        
        current_bar = data.iloc[idx]
        prev_bar = data.iloc[idx-1]
        prev2_bar = data.iloc[idx-2]
        
        # Bullish FVG: prev2_high < current_low (gap between)
        bullish_fvg = prev2_bar['high'] < current_bar['low']
        bullish_gap_size = current_bar['low'] - prev2_bar['high']
        
        # Bearish FVG: prev2_low > current_high
        bearish_fvg = prev2_bar['low'] > current_bar['high']
        bearish_gap_size = prev2_bar['low'] - current_bar['high']
        
        if bullish_fvg and bullish_gap_size >= self.params['fvg_min_size']:
            # Check if price is filling the gap
            if (current_bar['low'] <= prev2_bar['high'] + bullish_gap_size/2 and
                current_bar['high'] >= prev2_bar['high']):
                return {'valid': True, 'direction': 'bullish', 'weight': 2, 'gap_size': bullish_gap_size}
        
        elif bearish_fvg and bearish_gap_size >= self.params['fvg_min_size']:
            # Check if price is filling the gap
            if (current_bar['high'] >= prev2_bar['low'] - bearish_gap_size/2 and
                current_bar['low'] <= prev2_bar['low']):
                return {'valid': True, 'direction': 'bearish', 'weight': 2, 'gap_size': bearish_gap_size}
        
        return {'valid': False, 'direction': None, 'weight': 0}
    
    def _check_order_block(self, data: pd.DataFrame, idx: int) -> Dict:
        """Check for Order Block retests"""
        if idx < 20:
            return {'valid': False, 'direction': None, 'weight': 0}
        
        current_bar = data.iloc[idx]
        lookback_data = data.iloc[idx-20:idx]
        
        # Look for strong moves that created order blocks
        for i in range(5, len(lookback_data)-5):
            ob_bar = lookback_data.iloc[i]
            before_bars = lookback_data.iloc[i-5:i]
            after_bars = lookback_data.iloc[i+1:i+6]
            
            # Bullish order block: consolidation followed by strong move up
            if (len(after_bars) > 0 and after_bars['close'].iloc[-1] > ob_bar['close'] * 1.003 and
                before_bars['high'].max() - before_bars['low'].min() < self.params['ob_min_size']):
                
                # Check if current price is retesting the OB
                if (current_bar['low'] <= ob_bar['high'] and current_bar['high'] >= ob_bar['low']):
                    return {'valid': True, 'direction': 'bullish', 'weight': 2, 'ob_price': ob_bar['low']}
            
            # Bearish order block: consolidation followed by strong move down
            elif (len(after_bars) > 0 and after_bars['close'].iloc[-1] < ob_bar['close'] * 0.997 and
                  before_bars['high'].max() - before_bars['low'].min() < self.params['ob_min_size']):
                
                # Check if current price is retesting the OB
                if (current_bar['high'] >= ob_bar['low'] and current_bar['low'] <= ob_bar['high']):
                    return {'valid': True, 'direction': 'bearish', 'weight': 2, 'ob_price': ob_bar['high']}
        
        return {'valid': False, 'direction': None, 'weight': 0}
    
    def _check_ote(self, data: pd.DataFrame, idx: int) -> Dict:
        """Check for Optimal Trade Entry levels"""
        if idx < 50:
            return {'valid': False, 'direction': None, 'weight': 0}
        
        current_bar = data.iloc[idx]
        lookback_data = data.iloc[idx-50:idx]
        
        # Find swing high and low
        swing_high_idx = lookback_data['high'].idxmax()
        swing_low_idx = lookback_data['low'].idxmin()
        
        swing_high = lookback_data.loc[swing_high_idx, 'high']
        swing_low = lookback_data.loc[swing_low_idx, 'low']
        swing_range = swing_high - swing_low
        
        if swing_range < current_bar['atr'] * 2:
            return {'valid': False, 'direction': None, 'weight': 0}
        
        # Check for OTE retracement levels
        for fib_level in self.params['ote_fib_levels']:
            # Determine if we're in uptrend or downtrend
            swing_high_pos = lookback_data.index.get_loc(swing_high_idx)
            swing_low_pos = lookback_data.index.get_loc(swing_low_idx)
            
            if swing_high_pos > swing_low_pos:  # Uptrend
                ote_level = swing_high - (swing_range * fib_level)
                if abs(current_bar['close'] - ote_level) <= swing_range * 0.02:
                    return {'valid': True, 'direction': 'bullish', 'weight': 2, 'ote_level': fib_level}
            else:  # Downtrend
                ote_level = swing_low + (swing_range * fib_level)
                if abs(current_bar['close'] - ote_level) <= swing_range * 0.02:
                    return {'valid': True, 'direction': 'bearish', 'weight': 2, 'ote_level': fib_level}
        
        return {'valid': False, 'direction': None, 'weight': 0}
    
    def _check_judas_swing(self, data: pd.DataFrame, idx: int, current_bar: pd.Series) -> Dict:
        """Check for Judas Swing patterns"""
        hour = current_bar.name.hour
        
        # Check if we're in Judas window
        judas_window = any(session <= hour <= session + 2 for session in self.params['judas_sessions'])
        
        if not judas_window or idx < 20:
            return {'valid': False, 'direction': None, 'weight': 0}
        
        lookback_data = data.iloc[idx-20:idx]
        range_high = lookback_data['high'].max()
        range_low = lookback_data['low'].min()
        
        # False break above range followed by reversal
        if (current_bar['high'] > range_high and
            current_bar['close'] < range_high - (range_high - range_low) * 0.1):
            return {'valid': True, 'direction': 'bearish', 'weight': 3, 'false_break': 'high'}
        
        # False break below range followed by reversal
        elif (current_bar['low'] < range_low and
              current_bar['close'] > range_low + (range_high - range_low) * 0.1):
            return {'valid': True, 'direction': 'bullish', 'weight': 3, 'false_break': 'low'}
        
        return {'valid': False, 'direction': None, 'weight': 0}
    
    def _check_htf_structure(self, daily_row: pd.Series, h4_row: pd.Series, h1_row: pd.Series) -> Dict:
        """Check higher timeframe structure alignment"""
        # Daily structure
        daily_bullish = (daily_row['ema_12'] > daily_row['ema_26'] and
                         daily_row['close'] > daily_row['sma_200'])
        daily_bearish = (daily_row['ema_12'] < daily_row['ema_26'] and
                        daily_row['close'] < daily_row['sma_200'])
        
        # H4 structure
        h4_bullish = h4_row['ema_12'] > h4_row['ema_26']
        h4_bearish = h4_row['ema_12'] < h4_row['ema_26']
        
        # H1 structure
        h1_bullish = h1_row['rsi_14'] < 70 and h1_row['close'] > h1_row['ema_21']
        h1_bearish = h1_row['rsi_14'] > 30 and h1_row['close'] < h1_row['ema_21']
        
        bullish_alignment = sum([daily_bullish, h4_bullish, h1_bullish])
        bearish_alignment = sum([daily_bearish, h4_bearish, h1_bearish])
        
        if bullish_alignment >= 2:
            return {'valid': True, 'direction': 'bullish', 'weight': bullish_alignment}
        elif bearish_alignment >= 2:
            return {'valid': True, 'direction': 'bearish', 'weight': bearish_alignment}
        
        return {'valid': False, 'direction': None, 'weight': 0}
    
    def _check_session_confluence(self, current_bar: pd.Series) -> Dict:
        """Check trading session confluence"""
        hour = current_bar.name.hour
        
        # London session (high activity)
        if 8 <= hour <= 17:
            return {'valid': True, 'direction': 'neutral', 'weight': 1, 'session': 'london'}
        # NY session (high activity)
        elif 13 <= hour <= 22:
            return {'valid': True, 'direction': 'neutral', 'weight': 1, 'session': 'ny'}
        # Asian session (lower activity)
        else:
            return {'valid': True, 'direction': 'neutral', 'weight': 0.5, 'session': 'asian'}

class EdenICTDemo:
    """Eden ICT Demo System"""
    
    def __init__(self):
        self.data_generator = SyntheticDataGenerator()
        self.htf_analyzer = HTFAnalyzer()
        self.ict_strategy = ICTStrategy()
        self.symbols = ['XAUUSD', 'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF']
    
    def run_demo(self):
        """Run ICT demo with synthetic data"""
        print("🎯 Eden ICT Demo System - Complete Strategy Demonstration")
        print("=" * 80)
        print("🎯 Features:")
        print("  • Unified ICT Strategy (Liquidity+FVG+OB+OTE+Judas)")
        print("  • Complete Top-Down Analysis (D1→H4→H1→M15)")
        print("  • Against-Bias Trading with Reversal Detection")
        print("  • HTF Structure Alignment")
        print("  • Session-Based Analysis")
        print(f"📊 Demo Symbols: {', '.join(self.symbols)}")
        print()
        
        all_results = {}
        
        for symbol in self.symbols:
            print(f"🎯 Running ICT Analysis for {symbol}...")
            result = self.analyze_symbol(symbol)
            if result:
                all_results[symbol] = result
        
        self.display_summary(all_results)
        return all_results
    
    def analyze_symbol(self, symbol: str) -> Dict:
        """Analyze symbol with ICT strategy"""
        
        # Generate synthetic data for all timeframes
        daily_data = self.data_generator.generate_symbol_data(symbol, 365, 'D1')
        h4_data = self.data_generator.generate_symbol_data(symbol, 1000, 'H4')
        h1_data = self.data_generator.generate_symbol_data(symbol, 2000, 'H1')
        m15_data = self.data_generator.generate_symbol_data(symbol, 3000, 'M15')
        
        print(f"   📊 Generated data: D1({len(daily_data)}), H4({len(h4_data)}), H1({len(h1_data)}), M15({len(m15_data)})")
        
        # Analyze HTF bias
        htf_bias = self.htf_analyzer.analyze_full_htf_bias(
            daily_data, h4_data, h1_data, m15_data, m15_data
        )
        
        print(f"   📊 HTF Bias: {htf_bias.overall} (confidence: {htf_bias.confidence:.2f})")
        print(f"   🔄 Reversal Probability: {htf_bias.reversal_probability:.2f}")
        
        # Create regimes
        regimes = []
        for i in range(len(h1_data)):
            volatility = h1_data.iloc[i]['atr'] / h1_data.iloc[i]['close']
            if volatility > 0.01:
                regime = 'momentum_burst'
            elif volatility < 0.005:
                regime = 'range'
            else:
                regime = 'trend'
            regimes.append(MarketRegime(regime, 0.7, volatility, 0))
        
        # Generate ICT signals
        signals = self.ict_strategy.generate_signals(
            daily_data, h4_data, h1_data, m15_data, h1_data, htf_bias, regimes
        )
        
        for signal in signals:
            signal.symbol = symbol
        
        print(f"   📡 Generated {len(signals)} ICT confluence signals")
        
        if len(signals) > 0:
            print(f"   🔍 Sample signals:")
            for i, signal in enumerate(signals[:3]):
                confluences = signal.confluences
                valid_confs = [k for k, v in confluences.items() if v.get('valid', False)]
                conf_details = []
                for conf_name in valid_confs:
                    conf_data = confluences[conf_name]
                    if 'type' in conf_data:
                        conf_details.append(f"{conf_name}({conf_data['type']})")
                    elif 'ote_level' in conf_data:
                        conf_details.append(f"{conf_name}({conf_data['ote_level']})")
                    else:
                        conf_details.append(conf_name)
                
                print(f"     • {signal.side.upper()} at {signal.timestamp.strftime('%Y-%m-%d %H:%M')} - "
                      f"Confidence: {signal.confidence:.2f} - "
                      f"HTF: {signal.htf_bias} {'(Against Bias)' if signal.against_bias else ''} - "
                      f"Confluences: {', '.join(conf_details)}")
        
        return {
            'symbol': symbol,
            'signals': len(signals),
            'htf_bias': htf_bias,
            'test_bars': len(h1_data),
            'sample_signals': signals[:5] if signals else []
        }
    
    def display_summary(self, results: Dict):
        """Display demo results summary"""
        print("\n" + "=" * 80)
        print("🎯 EDEN ICT DEMO SYSTEM SUMMARY")
        print("=" * 80)
        
        total_signals = sum(r['signals'] for r in results.values())
        
        print(f"📊 ICT Confluence Strategy Results:")
        print(f"   • Total Signals Generated: {total_signals:,}")
        print(f"   • Average Signals per Symbol: {total_signals/len(results):.1f}")
        
        # Analyze signal characteristics
        against_bias_count = 0
        reversal_count = 0
        all_signals = []
        
        for symbol, result in results.items():
            htf_bias = result['htf_bias']
            print(f"\n📊 {symbol}:")
            print(f"   • Test Bars: {result['test_bars']:,}")
            print(f"   • Signals: {result['signals']}")
            print(f"   • HTF Bias: {htf_bias.overall} ({htf_bias.confidence:.2f})")
            print(f"   • Reversal Probability: {htf_bias.reversal_probability:.2f}")
            
            for signal in result['sample_signals']:
                all_signals.append(signal)
                if signal.against_bias:
                    against_bias_count += 1
                if signal.reversal_signal:
                    reversal_count += 1
        
        print(f"\n🎯 Signal Analysis:")
        if len(all_signals) > 0:
            avg_confidence = sum(s.confidence for s in all_signals) / len(all_signals)
            print(f"   • Average Signal Confidence: {avg_confidence:.2f}")
            print(f"   • Against-Bias Trades: {against_bias_count}")
            print(f"   • Reversal Signals: {reversal_count}")
            
            # Most common confluences
            confluence_count = {}
            for signal in all_signals:
                if signal.confluences:
                    for conf_name, conf_data in signal.confluences.items():
                        if conf_data.get('valid', False):
                            confluence_count[conf_name] = confluence_count.get(conf_name, 0) + 1
            
            print(f"   • Most Common Confluences:")
            for conf, count in sorted(confluence_count.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"     - {conf}: {count} signals")
        
        print(f"\n✅ ICT Demo System with full confluences completed!")
        print(f"💡 This demonstrates the unified ICT approach with:")
        print(f"   • All 5 ICT elements working together")
        print(f"   • Top-down analysis from D1 to M15")
        print(f"   • Smart against-bias trading")
        print(f"   • Session-aware confluence detection")

def main():
    """Main execution"""
    demo = EdenICTDemo()
    demo.run_demo()

if __name__ == "__main__":
    main()