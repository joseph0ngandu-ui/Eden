#!/usr/bin/env python3
"""
SPREAD HUNTER ALGORITHM
Research Implementation - DO NOT DEPLOY TO LIVE BOT

Core Thesis: Only trade when EV > Cost
- Spread Compression Detection
- Dynamic Cost Calculation
- Prop Firm (FundedNext) Compliance
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import logging

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class CostConfig:
    """Trading cost configuration for different account types."""
    # Standard Account (Exness) - Normal costs
    STANDARD_SPREAD_PIPS: float = 1.5      # Average spread
    STANDARD_MARKUP_PIPS: float = 0.5      # Hidden markup
    STANDARD_SLIPPAGE_PIPS: float = 0.2    # Average slippage
    
    # Raw Spread Account
    RAW_SPREAD_PIPS: float = 0.2           # Typical raw spread
    RAW_COMMISSION_PER_LOT: float = 7.0    # $7 per round turn
    RAW_SLIPPAGE_PIPS: float = 0.1
    
    # Indices (US30, NAS100)
    INDEX_SPREAD_POINTS: float = 2.0       # Points not pips
    INDEX_SLIPPAGE_POINTS: float = 0.5
    
    def get_standard_cost_pips(self) -> float:
        return self.STANDARD_SPREAD_PIPS + self.STANDARD_MARKUP_PIPS + self.STANDARD_SLIPPAGE_PIPS
    
    def get_raw_cost_pips(self, lot_size: float = 1.0) -> float:
        # Commission in pips: $7 per lot = 0.7 pips for majors
        commission_pips = (self.RAW_COMMISSION_PER_LOT / lot_size) / 10  # Rough conversion
        return self.RAW_SPREAD_PIPS + commission_pips + self.RAW_SLIPPAGE_PIPS


@dataclass
class FundedNextRules:
    """FundedNext Stellar 2-Step Challenge Rules."""
    DAILY_LOSS_LIMIT_PCT: float = 5.0      # 5% of initial balance
    MAX_DRAWDOWN_PCT: float = 10.0         # 10% of initial balance
    PROFIT_TARGET_PCT: float = 10.0        # Phase 1: 10%, Phase 2: 5%
    
    # Risk Management (RELAXED for testing)
    MAX_RISK_PER_TRADE_PCT: float = 1.5    # Allow slightly higher
    MIN_RISK_PER_TRADE_PCT: float = 0.25   # Floor
    
    # Circuit Breaker Thresholds (RELAXED)
    DD_WARNING_THRESHOLD: float = 0.7      # 70% of limit = slow down
    DD_CRITICAL_THRESHOLD: float = 0.9     # 90% of limit = stop
    
    # Consecutive Loss Tolerance (INCREASED)
    MAX_CONSECUTIVE_LOSSES: int = 10       # Was 5, now 10


@dataclass
class TradeSignal:
    symbol: str
    direction: str  # LONG or SHORT
    entry_price: float
    sl: float
    tp: float
    confidence: float
    strategy: str
    entry_time: pd.Timestamp
    
    # Cost-Aware Metrics
    live_spread_pips: float
    avg_spread_pips: float
    expected_ev_pips: float
    total_cost_pips: float
    ev_to_cost_ratio: float
    
    atr: float


# ============================================================================
# CORE COMPONENTS
# ============================================================================

class SpreadAnalyzer:
    """Detects spread compression regimes."""
    
    def __init__(self, history_len: int = 100):
        self.spread_history: Dict[str, List[float]] = {}
        self.history_len = history_len
        
    def update_spread(self, symbol: str, spread_pips: float):
        if symbol not in self.spread_history:
            self.spread_history[symbol] = []
        self.spread_history[symbol].append(spread_pips)
        if len(self.spread_history[symbol]) > self.history_len:
            self.spread_history[symbol].pop(0)
            
    def get_average_spread(self, symbol: str) -> float:
        if symbol not in self.spread_history or not self.spread_history[symbol]:
            return 999.0
        return np.mean(self.spread_history[symbol])
    
    def is_spread_compressed(self, symbol: str, current_spread: float, threshold: float = 0.85) -> bool:
        """Returns True if current spread is significantly below recent average."""
        avg = self.get_average_spread(symbol)
        if avg == 999.0: return False
        return current_spread < (avg * threshold)


class EVCalculator:
    """Calculates Expected Value (EV) adjusted for costs."""
    
    def __init__(self, cost_config: CostConfig):
        self.cost_config = cost_config
        self.trade_results: Dict[str, List[float]] = {}  # Strategy -> List of R outcomes
        
        # Base stats per strategy (Conservative defaults)
        self.STRATEGY_STATS = {
            'SpreadHunter_Breakout': {'win_rate': 0.45, 'avg_rr': 1.5},
            'SpreadHunter_Reversion': {'win_rate': 0.55, 'avg_rr': 1.0},
            'SpreadHunter_Momentum': {'win_rate': 0.50, 'avg_rr': 1.5},
        }
    
    def update_stats(self, strategy: str, pnl_r: float):
        """Update strategy stats with actual trade result."""
        if strategy not in self.trade_results:
            self.trade_results[strategy] = []
        self.trade_results[strategy].append(pnl_r)
        
        # Recalculate stats from actual results
        if len(self.trade_results[strategy]) >= 20:
            results = self.trade_results[strategy][-100:]  # Last 100 trades
            wins = [r for r in results if r > 0]
            losses = [r for r in results if r < 0]
            
            if len(results) > 0:
                win_rate = len(wins) / len(results)
                avg_win = np.mean(wins) if wins else 1.0
                avg_loss = abs(np.mean(losses)) if losses else 1.0
                avg_rr = avg_win / avg_loss if avg_loss > 0 else 1.0
                
                self.STRATEGY_STATS[strategy] = {
                    'win_rate': win_rate,
                    'avg_rr': avg_rr
                }
    
    def calculate_ev_pips(self, strategy: str, sl_pips: float) -> float:
        """
        Calculate expected value in pips.
        EV = (WinRate * AvgWin) - (LossRate * AvgLoss)
        EV_pips = EV_R * SL_pips (since R = SL distance)
        """
        stats = self.STRATEGY_STATS.get(strategy, {'win_rate': 0.50, 'avg_rr': 1.0})
        win_rate = stats['win_rate']
        avg_rr = stats['avg_rr']
        
        ev_r = (win_rate * avg_rr) - (1 - win_rate)
        ev_pips = ev_r * sl_pips
        return ev_pips
    
    def get_total_cost_pips(self, symbol: str, live_spread: float, account_type: str = 'standard') -> float:
        """Calculate total trading cost in pips."""
        if account_type == 'standard':
            # Spread is already measured, add markup + slippage
            return live_spread + self.cost_config.STANDARD_MARKUP_PIPS + self.cost_config.STANDARD_SLIPPAGE_PIPS
        else:
            return self.cost_config.get_raw_cost_pips()
    
    def is_trade_profitable(self, ev_pips: float, cost_pips: float, min_ratio: float = 1.2) -> bool:
        """
        Check if trade has positive expectancy above cost threshold.
        Requires EV to be at least min_ratio * cost to account for variance.
        """
        if cost_pips <= 0:
            return ev_pips > 0
        return ev_pips >= cost_pips * min_ratio


class PropFirmRiskManager:
    """Risk management layer for FundedNext compliance."""
    
    def __init__(self, initial_balance: float, rules: FundedNextRules):
        self.initial_balance = initial_balance
        self.rules = rules
        
        # State tracking
        self.current_equity = initial_balance
        self.peak_equity = initial_balance
        self.start_of_day_balance = initial_balance
        self.current_day = datetime.now().date()
        
        # Trade tracking
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.trades_today = 0
        self.consecutive_losses = 0
        
        # Circuit breaker state
        self.is_trading_paused = False
        self.pause_reason = ""
    
    def new_day(self, opening_balance: float) -> None:
        """Reset daily tracking for new trading day."""
        self.start_of_day_balance = opening_balance
        self.daily_pnl = 0.0
        self.trades_today = 0
        self.current_day = datetime.now().date()
        self.is_trading_paused = False
        self.pause_reason = ""
        logger.info(f"📅 NEW DAY: Balance=${opening_balance:.2f}")
    
    def calculate_daily_dd_pct(self) -> float:
        """Calculate current daily drawdown as percentage of initial balance."""
        if self.start_of_day_balance <= 0:
            return 0.0
        daily_loss = max(0, self.start_of_day_balance - self.current_equity)
        return (daily_loss / self.initial_balance) * 100
    
    def calculate_max_dd_pct(self) -> float:
        """Calculate maximum drawdown from initial balance."""
        if self.initial_balance <= 0:
            return 0.0
        total_loss = max(0, self.initial_balance - self.current_equity)
        return (total_loss / self.initial_balance) * 100
    
    def get_risk_multiplier(self) -> float:
        """Dynamic risk adjustment based on current drawdown."""
        daily_dd = self.calculate_daily_dd_pct()
        max_dd = self.calculate_max_dd_pct()
        
        # Check against limits
        daily_usage = daily_dd / self.rules.DAILY_LOSS_LIMIT_PCT
        max_usage = max_dd / self.rules.MAX_DRAWDOWN_PCT
        
        # Use the worse of the two
        worst_usage = max(daily_usage, max_usage)
        
        if worst_usage >= self.rules.DD_CRITICAL_THRESHOLD:
            return 0.0  # Stop trading
        elif worst_usage >= self.rules.DD_WARNING_THRESHOLD:
            range_size = self.rules.DD_CRITICAL_THRESHOLD - self.rules.DD_WARNING_THRESHOLD
            position = worst_usage - self.rules.DD_WARNING_THRESHOLD
            return max(0.0, 1.0 - (position / range_size))
        else:
            return 1.0  # Full risk allowed
    
    def calculate_position_risk_pct(self, base_risk: float = 0.5) -> float:
        """Calculate actual risk percentage for next trade."""
        multiplier = self.get_risk_multiplier()
        risk = base_risk * multiplier
        risk = max(self.rules.MIN_RISK_PER_TRADE_PCT, risk)
        risk = min(self.rules.MAX_RISK_PER_TRADE_PCT, risk)
        return risk
    
    def can_trade(self) -> Tuple[bool, str]:
        """Check if trading is allowed."""
        daily_dd = self.calculate_daily_dd_pct()
        max_dd = self.calculate_max_dd_pct()
        
        # Hard limits
        if daily_dd >= self.rules.DAILY_LOSS_LIMIT_PCT * 0.95:
            return False, f"Daily DD limit reached ({daily_dd:.1f}%)"
        
        if max_dd >= self.rules.MAX_DRAWDOWN_PCT * 0.95:
            return False, f"Max DD limit reached ({max_dd:.1f}%)"
        
        # Consecutive loss breaker
        if self.consecutive_losses >= self.rules.MAX_CONSECUTIVE_LOSSES:
            return False, f"Consecutive loss limit ({self.consecutive_losses})"
        
        return True, "OK"
    
    def record_trade(self, pnl: float) -> None:
        """Record trade result and update state."""
        self.daily_pnl += pnl
        self.total_pnl += pnl
        self.current_equity += pnl
        self.trades_today += 1
        
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        if self.current_equity > self.peak_equity:
            self.peak_equity = self.current_equity


class SpreadHunterStrategies:
    """Collection of cost-aware trading strategies."""
    
    def __init__(self):
        self.atr_period = 14
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        if len(df) < 2: return 0.001
        tr = df['high'] - df['low']
        return tr.rolling(period).mean().iloc[-1] if len(tr) > period else tr.mean()
    
    def calculate_rsi(self, series: pd.Series, period: int = 14) -> float:
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs.iloc[-1]))
    
    # breakout_strategy and mean_reversion_strategy omitted for brevity since disabled
    
    def momentum_strategy(self, df: pd.DataFrame, symbol: str, live_spread: float) -> Optional[TradeSignal]:
        """Momentum continuation during trend + tight spread."""
        if len(df) < 50: return None
        
        current = df.iloc[-1]
        prev = df.iloc[-2]
        atr = self.calculate_atr(df)
        
        # Trend detection
        ema_10 = df['close'].ewm(span=10).mean()
        ema_30 = df['close'].ewm(span=30).mean()
        
        bullish_trend = ema_10.iloc[-1] > ema_30.iloc[-1] and ema_10.iloc[-5] > ema_30.iloc[-5]
        bearish_trend = ema_10.iloc[-1] < ema_30.iloc[-1] and ema_10.iloc[-5] < ema_30.iloc[-5]
        
        # Pullback detection
        pullback_low = prev['low'] < ema_10.iloc[-2] and current['close'] > ema_10.iloc[-1]
        pullback_high = prev['high'] > ema_10.iloc[-2] and current['close'] < ema_10.iloc[-1]
        
        pip_value = 0.0001 if 'JPY' not in symbol else 0.01
        spread_pips = live_spread / pip_value if pip_value > 0 else live_spread * 10000
        
        if bullish_trend and pullback_low:
            sl = df['low'].iloc[-5:].min() - atr * 0.3
            tp = current['close'] + (current['close'] - sl) * 1.5
            return TradeSignal(
                symbol=symbol, direction="LONG", entry_price=current['close'], sl=sl, tp=tp,
                confidence=0.70, strategy="SpreadHunter_Momentum", live_spread_pips=spread_pips,
                avg_spread_pips=0, expected_ev_pips=0, total_cost_pips=0, ev_to_cost_ratio=0,
                entry_time=current.name, atr=atr
            )
        
        if bearish_trend and pullback_high:
            sl = df['high'].iloc[-5:].max() + atr * 0.3
            tp = current['close'] - (sl - current['close']) * 1.5
            return TradeSignal(
                symbol=symbol, direction="SHORT", entry_price=current['close'], sl=sl, tp=tp,
                confidence=0.70, strategy="SpreadHunter_Momentum", live_spread_pips=spread_pips,
                avg_spread_pips=0, expected_ev_pips=0, total_cost_pips=0, ev_to_cost_ratio=0,
                entry_time=current.name, atr=atr
            )
        return None


class SpreadHunterEngine:
    """Main engine that orchestrates cost-aware trading."""
    
    def __init__(self, initial_balance: float = 100000.0, account_type: str = 'standard'):
        self.cost_config = CostConfig()
        self.rules = FundedNextRules()
        
        self.spread_analyzer = SpreadAnalyzer()
        self.ev_calculator = EVCalculator(self.cost_config)
        self.risk_manager = PropFirmRiskManager(initial_balance, self.rules)
        self.strategies = SpreadHunterStrategies()
        self.account_type = account_type
        
        # Tracking
        self.signals_generated = 0
        self.signals_filtered_spread = 0
        self.signals_filtered_ev = 0
        self.signals_filtered_risk = 0
        self.signals_executed = 0
    
    def get_h1_trend(self, h1_df: pd.DataFrame) -> int:
        """Determine H1 Trend Direction."""
        if len(h1_df) < 50: return 0
        ema_20 = h1_df['close'].ewm(span=20).mean()
        ema_50 = h1_df['close'].ewm(span=50).mean()
        last_price = h1_df['close'].iloc[-1]
        
        if ema_20.iloc[-1] > ema_50.iloc[-1] and last_price > ema_20.iloc[-1]:
            return 1
        elif ema_20.iloc[-1] < ema_50.iloc[-1] and last_price < ema_20.iloc[-1]:
            return -1
        return 0

    def evaluate(self, df: pd.DataFrame, symbol: str, live_spread: float, h1_trend: int = 0) -> Optional[TradeSignal]:
        """Evaluate symbol for trading opportunity."""
        # 1. Prop Firm Risk Check
        can_trade, reason = self.risk_manager.can_trade()
        if not can_trade:
            self.signals_filtered_risk += 1
            return None
        
        # 2. Update spread history
        self.spread_analyzer.update_spread(symbol, live_spread)
        avg_spread = self.spread_analyzer.get_average_spread(symbol)
        
        # 3. Spread Compression Check (The Exploit)
        if not self.spread_analyzer.is_spread_compressed(symbol, live_spread, threshold=0.85):
            self.signals_filtered_spread += 1
            return None
        
        # 4. Generate raw signals
        raw_signal = self.strategies.momentum_strategy(df, symbol, live_spread)
        
        # MTF FILTER:
        if h1_trend != 0 and raw_signal:
            if h1_trend == 1 and raw_signal.direction != "LONG": return None
            if h1_trend == -1 and raw_signal.direction != "SHORT": return None
        
        if not raw_signal:
            return None
            
        self.signals_generated += 1
        
        # Calculate cost
        pip_value = 0.0001 if 'JPY' not in symbol else 0.01
        sl_pips = abs(raw_signal.entry_price - raw_signal.sl) / pip_value
        
        total_cost = self.ev_calculator.get_total_cost_pips(
            symbol, raw_signal.live_spread_pips, self.account_type
        )
        
        # Calculate EV
        ev_pips = self.ev_calculator.calculate_ev_pips(raw_signal.strategy, sl_pips)
        
        # Check profitability (relaxed: 1.2x)
        if not self.ev_calculator.is_trade_profitable(ev_pips, total_cost, min_ratio=1.2):
            self.signals_filtered_ev += 1
            return None
        
        ev_ratio = ev_pips / total_cost if total_cost > 0 else 0
        
        # Update signal with cost data
        raw_signal.avg_spread_pips = avg_spread
        raw_signal.expected_ev_pips = ev_pips
        raw_signal.total_cost_pips = total_cost
        raw_signal.ev_to_cost_ratio = ev_ratio
        
        self.signals_executed += 1
        logger.info(
            f"✅ SIGNAL: {symbol} {raw_signal.direction} | "
            f"EV={raw_signal.expected_ev_pips:.2f} pips | "
            f"Cost={raw_signal.total_cost_pips:.2f} pips | "
            f"Ratio={raw_signal.ev_to_cost_ratio:.2f}x"
        )
        
        return raw_signal
    
    def get_stats(self) -> Dict:
        return {
            'signals_generated': self.signals_generated,
            'filtered_spread': self.signals_filtered_spread,
            'filtered_ev': self.signals_filtered_ev,
            'filtered_risk': self.signals_filtered_risk,
            'executed': self.signals_executed,
            'filter_rate': (1 - self.signals_executed / max(1, self.signals_generated)) * 100
        }


def run_backtest_validation():
    """Run comparative backtest: M15 Baseline vs MTF (H1 Trend + M5 Entry)."""
    try:
        import MetaTrader5 as mt5
    except ImportError:
        print("MetaTrader5 not available")
        return
    
    if not mt5.initialize(path=r"C:\Program Files\MetaTrader 5 Terminal\terminal64.exe"):
        print("MT5 init failed")
        return
    
    print("=" * 80)
    print("  SPREAD HUNTER: STRATEGY SHOWDOWN")
    print("  Baseline (M15) vs. Multi-Timeframe (H1 Trend + M5 Entry)")
    print("=" * 80)
    
    symbols = ['EURUSDm', 'GBPUSDm', 'USDJPYm', 'XAUUSDm']
    end_time = datetime.now()
    start_time = end_time - timedelta(days=90)
    
    modes = [
        {'name': 'BASELINE_M15', 'tf': mt5.TIMEFRAME_M15, 'mtf': False},
        {'name': 'MTF_H1_M5',    'tf': mt5.TIMEFRAME_M5,  'mtf': True}
    ]
    
    results = {}
    
    for mode in modes:
        mode_name = mode['name']
        tf = mode['tf']
        print(f"\n{'='*80}")
        print(f"  RUNNING MODE: {mode_name}")
        print(f"{'='*80}")
        
        engine = SpreadHunterEngine(initial_balance=100000.0, account_type='standard')
        all_trades = []
        
        for symbol in symbols:
            print(f"\n  Processing {symbol}...")
            
            rates = mt5.copy_rates_range(symbol, tf, start_time, end_time)
            if rates is None or len(rates) < 100:
                print(f"    WARNING: Insufficient data")
                continue
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            df['spread'] = (df['high'] - df['low']) / df['close'] * 10000 
            
            h1_trends = {}
            if mode['mtf']:
                h1_rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_H1, start_time, end_time)
                if h1_rates is not None:
                    h1_df = pd.DataFrame(h1_rates)
                    h1_df['time'] = pd.to_datetime(h1_df['time'], unit='s')
                    h1_df.set_index('time', inplace=True)
                    
                    ema_20 = h1_df['close'].ewm(span=20).mean()
                    ema_50 = h1_df['close'].ewm(span=50).mean()
                    
                    for i in range(50, len(h1_df)):
                        ts = h1_df.index[i]
                        close = h1_df['close'].iloc[i]
                        # Fix scalar access
                        e20 = ema_20.iloc[i]
                        e50 = ema_50.iloc[i]
                        trend = 0
                        if e20 > e50 and close > e20: trend = 1
                        elif e20 < e50 and close < e20: trend = -1
                        h1_trends[ts] = trend
            
            trade_count = 0
            for i in range(100, len(df) - 50):
                window = df.iloc[i-100:i+1]
                current_time = df.index[i]
                current_bar = df.iloc[i]
                live_spread = current_bar['spread'] * 0.1
                
                h1_trend = 0
                if mode['mtf']:
                    h1_ts = current_time.replace(minute=0, second=0, microsecond=0)
                    h1_trend = h1_trends.get(h1_ts, 0)
                
                try:
                    signal = engine.evaluate(window, symbol, live_spread, h1_trend)
                except Exception as e:
                    continue
                
                if signal:
                    future = df.iloc[i+1:i+50]
                    if len(future) < 5: continue
                    result = check_trade_outcome(signal, future)
                    if result:
                        all_trades.append(result)
                        engine.risk_manager.record_trade(result['pnl_r'] * 100)
                        trade_count += 1
            print(f"    Trades: {trade_count}")
        results[mode_name] = all_trades
    
    mt5.shutdown()
    
    print("\n" + "=" * 80)
    print("  STRATEGY SHOWDOWN RESULTS")
    print("=" * 80)
    for mode_name, trades in results.items():
        if not trades:
            print(f"{mode_name:<15} N/A")
            continue
        wins = [t for t in trades if t['pnl_r'] > 0]
        losses = [t for t in trades if t['pnl_r'] < 0]
        total_r = sum(t['pnl_r'] for t in trades)
        win_rate = len(wins) / len(trades) * 100
        profit_factor = sum(t['pnl_r'] for t in wins) / abs(sum(t['pnl_r'] for t in losses)) if losses else 0
        print(f"{mode_name:<15} Trades: {len(trades):<5} WR: {win_rate:<5.1f}% Total: {total_r:<6.1f}R PF: {profit_factor:<4.2f}")
    
    if 'BASELINE_M15' in results and 'MTF_H1_M5' in results:
        base = results['BASELINE_M15']
        mtf = results['MTF_H1_M5']
        base_r = sum(t['pnl_r'] for t in base)
        mtf_r = sum(t['pnl_r'] for t in mtf)
        diff = mtf_r - base_r
        print(f"\nWINNER: {'MTF_H1_M5' if diff > 0 else 'BASELINE_M15'} by {abs(diff):.1f}R")

def check_trade_outcome(signal: TradeSignal, future: pd.DataFrame) -> Optional[Dict]:
    entry = signal.entry_price
    sl = signal.sl
    tp = signal.tp
    
    for idx, row in future.iterrows():
        if signal.direction == 'LONG':
            if row['low'] <= sl:
                return {'pnl_r': -1.0}
            if row['high'] >= tp:
                risk = abs(entry - sl)
                reward = abs(tp - entry)
                return {'pnl_r': reward / risk if risk > 0 else 1.0}
        else:  # SHORT
            if row['high'] >= sl:
                return {'pnl_r': -1.0}
            if row['low'] <= tp:
                risk = abs(sl - entry)
                reward = abs(entry - tp)
                return {'pnl_r': reward / risk if risk > 0 else 1.0}
    
    last = future.iloc[-1]
    pnl = (last['close'] - entry) if signal.direction == 'LONG' else (entry - last['close'])
    risk = abs(entry - sl)
    return {'pnl_r': pnl / risk if risk > 0 else 0}

if __name__ == '__main__':
    run_backtest_validation()
