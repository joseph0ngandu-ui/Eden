#!/usr/bin/env python3
"""
Signal fusion for ICT + Price Action with confidence scoring and kill-zone filtering.
Produces Trade-like entries compatible with existing backtest harnesses.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict
import pandas as pd
import numpy as np

from .ict import ICT
from .price_action import PriceAction


@dataclass
class ICTPAParams:
    atr_period: int = 14
    tp_atr_mult: float = 2.5
    sl_atr_mult: float = 0.8
    min_confidence: float = 0.60
    max_trades_per_day: int = 5
    skip_bars_after_open: int = 0
    killzones: List[str] = None  # e.g., ["LONDON","NY"]
    
    # weights (best-performing observed)
    w_bias: float = 0.25
    w_pa: float = 0.55
    w_kz: float = 0.10


@dataclass
class Trade:
    symbol: str
    direction: str
    entry_price: float
    tp: float
    sl: float
    confidence: float
    atr: float
    entry_time: pd.Timestamp
    bar_index: int


class KillZones:
    @staticmethod
    def score(ts: pd.Timestamp, zones: Optional[List[str]]) -> float:
        if not zones:
            return 0.5
        h = ts.hour
        s = 0.0
        if 'ASIA' in zones and 1 <= h <= 8:
            s = max(s, 0.6)
        if 'LONDON' in zones and 7 <= h <= 11:
            s = max(s, 1.0)
        if 'NY' in zones and 12 <= h <= 16:
            s = max(s, 0.9)
        return s if s > 0 else 0.4


class ICTPASignals:
    def __init__(self, params: Optional[ICTPAParams] = None):
        self.params = params or ICTPAParams(killzones=["LONDON","NY"])
        self.ict = ICT()
        self.pa = PriceAction()

    @staticmethod
    def _atr(df: pd.DataFrame, period: int) -> pd.Series:
        tr1 = df['high'] - df['low']
        tr2 = (df['high'] - df['close'].shift(1)).abs()
        tr3 = (df['low'] - df['close'].shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period, min_periods=1).mean()

    def generate_signals(self, df: pd.DataFrame) -> List[Trade]:
        n = len(df)
        if n < 50:
            return []
        atr = self._atr(df, self.params.atr_period)
        fvg = self.ict.detect_fvg(df)
        ob = self.ict.detect_order_blocks(df)
        sweep = self.ict.detect_liquidity_sweep(df)
        bos = self.pa.detect_bos_choch(df)
        rej = self.pa.detect_rejection(df)
        mom = self.pa.detect_momentum_shift(df)
        bias = self.ict.compute_htf_bias(df)

        trades: List[Trade] = []
        # daily trade cap logic
        daily_counts: Dict[str, int] = {}
        for idx in range(2, n):
            ts = df['time'].iloc[idx] if 'time' in df.columns else None
            if ts is None:
                continue
            day = str(ts.date())
            daily_counts.setdefault(day, 0)
            if daily_counts[day] >= self.params.max_trades_per_day:
                continue

            # PA confirmation vector
            pa_vec = np.array([
                1.0 if bos.iloc[idx] != 0 else 0.0,
                1.0 if rej.iloc[idx] != 0 else 0.0,
                1.0 if mom.iloc[idx] != 0 else 0.0,
            ])
            pa_score = float(pa_vec.mean())  # 0..1

            # ICT context strength
            ict_vec = np.array([
                1.0 if fvg.iloc[idx] != 0 else 0.0,
                1.0 if ob.iloc[idx] != 0 else 0.0,
                1.0 if sweep.iloc[idx] != 0 else 0.0,
            ])
            ict_score = float(ict_vec.mean())  # 0..1

            kz_score = KillZones.score(ts, self.params.killzones)

            confidence = (
                self.params.w_pa * pa_score +
                self.params.w_bias * (bias.strength or 0.0) +
                self.params.w_kz * kz_score
            )
            # Direction decision: align with HTF bias if exists, otherwise PA signal sign
            dirn = None
            if bias.direction:
                dirn = bias.direction
            else:
                sgn = (bos.iloc[idx] + rej.iloc[idx] + mom.iloc[idx])
                dirn = 'LONG' if sgn > 0 else ('SHORT' if sgn < 0 else None)

            if dirn is None or confidence < self.params.min_confidence:
                continue

            entry = float(df['close'].iloc[idx])
            a = float(atr.iloc[idx]) if not pd.isna(atr.iloc[idx]) else 0.0
            if a <= 0:
                continue
            if dirn == 'LONG':
                tp = entry + a * self.params.tp_atr_mult
                sl = entry - a * self.params.sl_atr_mult
            else:
                tp = entry - a * self.params.tp_atr_mult
                sl = entry + a * self.params.sl_atr_mult

            trades.append(Trade(
                symbol=df.attrs.get('symbol', 'UNKNOWN'),
                direction=dirn,
                entry_price=entry,
                tp=float(tp),
                sl=float(sl),
                confidence=float(confidence),
                atr=a,
                entry_time=ts,
                bar_index=idx
            ))
            daily_counts[day] += 1

        return trades

    def manage_position(self, df: pd.DataFrame, open_pos: Dict[str, dict], symbol: str) -> List[Dict]:
        actions: List[Dict] = []
        if symbol not in open_pos:
            return actions
        pos = open_pos[symbol]
        last = df.iloc[-1]
        if pos['direction'] == 'LONG':
            if last['high'] >= pos['tp']:
                actions.append({'action':'close','symbol':symbol,'price':pos['tp'],'reason':'tp_hit','r_value': (pos['tp']-pos['entry_price'])/(pos['atr']*max(1e-9, (pos['sl_mult'])) )})
                del open_pos[symbol]
                return actions
            if last['low'] <= pos['sl']:
                actions.append({'action':'close','symbol':symbol,'price':pos['sl'],'reason':'sl_hit','r_value': -1.0})
                del open_pos[symbol]
                return actions
        else:
            if last['low'] <= pos['tp']:
                actions.append({'action':'close','symbol':symbol,'price':pos['tp'],'reason':'tp_hit','r_value': (pos['entry_price']-pos['tp'])/(pos['atr']*max(1e-9, (pos['sl_mult'])) )})
                del open_pos[symbol]
                return actions
            if last['high'] >= pos['sl']:
                actions.append({'action':'close','symbol':symbol,'price':pos['sl'],'reason':'sl_hit','r_value': -1.0})
                del open_pos[symbol]
                return actions
        # Max bars exit optional: omitted for simplicity
        return actions

class ICTPA:
    """Wrapper compatible with VB-style backtester interface."""
    def __init__(self, config: Optional[dict] = None):
        p = (config or {}).get('params', {})
        self.params = ICTPAParams(**p) if p else ICTPAParams(killzones=["LONDON","NY"])
        self.signals = ICTPASignals(self.params)
        self.open_positions: Dict[str, dict] = {}

    def update_params(self, **kwargs):
        for k,v in kwargs.items():
            if hasattr(self.params, k):
                setattr(self.params, k, v)

    def generate_signals(self, df: pd.DataFrame) -> List[Trade]:
        return self.signals.generate_signals(df)

    def on_trade_open(self, t: Trade):
        self.open_positions[t.symbol] = {
            'entry_price': t.entry_price,
            'direction': t.direction,
            'tp': t.tp,
            'sl': t.sl,
            'atr': t.atr,
            'sl_mult': self.params.sl_atr_mult,
            'entry_time': t.entry_time,
            'bar_index': t.bar_index,
        }

    def manage_position(self, df: pd.DataFrame, symbol: str):
        return self.signals.manage_position(df, self.open_positions, symbol)
