#!/usr/bin/env python3
"""
MA v1.2 Strategy - MA crossover with ATR-based exits
"""

import yaml
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger("Eden.MA_v1_2")

@dataclass
class MATrade:
    symbol: str
    direction: str
    entry_price: float
    tp: float
    sl: float
    atr: float
    entry_time: pd.Timestamp
    bar_index: int

@dataclass
class MAPosition:
    symbol: str
    direction: str
    entry_price: float
    tp: float
    sl: float
    entry_bar_index: int
    entry_time: pd.Timestamp
    atr: float
    stop_moved: bool = False

class MA_v1_2:
    def __init__(self, config_path: str = 'config/ma_v1_2.yml'):
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
        r = self.cfg['risk']
        ie = self.cfg['entry_exit']
        ind = self.cfg['indicators']
        self.max_trades_per_day = r['max_trades_per_day']
        self.max_bars_in_trade = r['max_bars_in_trade']
        self.skip_bars_after_open = r['skip_bars_after_open']
        self.atr_period = ind['atr']['period']
        self.fast = ind['ma']['fast_period']
        self.slow = ind['ma']['slow_period']
        self.tp_mult = ie['tp_atr_multiplier']
        self.sl_mult = ie['sl_atr_multiplier']
        self.trail_r = ie['trail_trigger_r']
        self.open_positions: Dict[str, MAPosition] = {}
        self.daily_trades: Dict[str, int] = {}

    def update_params(self, **kwargs):
        if 'fast_period' in kwargs: self.fast = int(kwargs['fast_period'])
        if 'slow_period' in kwargs: self.slow = int(kwargs['slow_period'])
        if 'tp_atr_multiplier' in kwargs: self.tp_mult = float(kwargs['tp_atr_multiplier'])
        if 'sl_atr_multiplier' in kwargs: self.sl_mult = float(kwargs['sl_atr_multiplier'])

    def atr(self, df: pd.DataFrame) -> pd.Series:
        tr1 = df['high'] - df['low']
        tr2 = (df['high'] - df['close'].shift(1)).abs()
        tr3 = (df['low'] - df['close'].shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(self.atr_period, min_periods=1).mean()

    def generate_signals(self, df: pd.DataFrame) -> List[MATrade]:
        n = len(df)
        if n < max(self.fast, self.slow, self.atr_period) + 5:
            return []
        
        df_ma = df.copy()
        df_ma['ma_fast'] = df_ma['close'].rolling(self.fast, min_periods=1).mean()
        df_ma['ma_slow'] = df_ma['close'].rolling(self.slow, min_periods=1).mean()
        cross_up = (df_ma['ma_fast'] > df_ma['ma_slow']) & (df_ma['ma_fast'].shift(1) <= df_ma['ma_slow'].shift(1))
        cross_down = (df_ma['ma_fast'] < df_ma['ma_slow']) & (df_ma['ma_fast'].shift(1) >= df_ma['ma_slow'].shift(1))
        atr = self.atr(df_ma)

        # Skip early bars per-day if time available
        if 'time' in df.columns:
            day_index = df['time'].dt.date
            intraday_idx = day_index.groupby(day_index).cumcount()
            skip_mask = intraday_idx < self.skip_bars_after_open
        else:
            skip_mask = pd.Series([False]*n)

        longs = cross_up & (~skip_mask)
        shorts = cross_down & (~skip_mask)

        trades: List[MATrade] = []
        sym = df.attrs.get('symbol', 'UNKNOWN')
        for idx in range(n):
            if longs.iloc[idx] or shorts.iloc[idx]:
                dirn = 'LONG' if longs.iloc[idx] else 'SHORT'
                a = float(atr.iloc[idx]) if not pd.isna(atr.iloc[idx]) else 0.0
                if a <= 0: continue
                entry = float(df['close'].iloc[idx])
                if dirn == 'LONG':
                    tp = entry + a * self.tp_mult
                    sl = entry - a * self.sl_mult
                else:
                    tp = entry - a * self.tp_mult
                    sl = entry + a * self.sl_mult
                trades.append(MATrade(
                    symbol=sym,
                    direction=dirn,
                    entry_price=entry,
                    tp=float(tp),
                    sl=float(sl),
                    atr=a,
                    entry_time=df['time'].iloc[idx] if 'time' in df.columns else pd.Timestamp.now(),
                    bar_index=idx
                ))
        return trades

    def on_trade_open(self, t: MATrade):
        self.daily_trades[t.symbol] = self.daily_trades.get(t.symbol, 0) + 1
        self.open_positions[t.symbol] = MAPosition(
            symbol=t.symbol,
            direction=t.direction,
            entry_price=t.entry_price,
            tp=t.tp,
            sl=t.sl,
            entry_bar_index=t.bar_index,
            entry_time=t.entry_time,
            atr=t.atr
        )

    def manage_position(self, df: pd.DataFrame, symbol: str):
        if symbol not in self.open_positions:
            return []
        pos = self.open_positions[symbol]
        last = df.iloc[-1]
        if pos.direction == 'LONG':
            unreal = last['close'] - pos.entry_price
        else:
            unreal = pos.entry_price - last['close']
        r = unreal / (pos.atr * self.sl_mult if pos.atr>0 else 1e-9)
        actions = []
        if r >= self.trail_r and not pos.stop_moved:
            pos.stop_moved = True
            pos.sl = pos.entry_price
            actions.append({'action':'trail_stop','symbol':symbol,'new_sl':pos.sl})
        if pos.direction=='LONG' and last['high']>=pos.tp:
            actions.append({'action':'close','symbol':symbol,'price':pos.tp,'r_value':r})
            del self.open_positions[symbol]
            return actions
        if pos.direction=='SHORT' and last['low']<=pos.tp:
            actions.append({'action':'close','symbol':symbol,'price':pos.tp,'r_value':r})
            del self.open_positions[symbol]
            return actions
        if pos.direction=='LONG' and last['low']<=pos.sl:
            actions.append({'action':'close','symbol':symbol,'price':pos.sl,'r_value':r})
            del self.open_positions[symbol]
            return actions
        if pos.direction=='SHORT' and last['high']>=pos.sl:
            actions.append({'action':'close','symbol':symbol,'price':pos.sl,'r_value':r})
            del self.open_positions[symbol]
            return actions
        idx = len(df)-1
        if idx - pos.entry_bar_index >= self.max_bars_in_trade:
            actions.append({'action':'close','symbol':symbol,'price':last['close'],'r_value':r})
            del self.open_positions[symbol]
            return actions
        return actions
