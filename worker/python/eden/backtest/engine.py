from __future__ import annotations
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from ..utils.types import Trade


@dataclass
class BacktestEngine:
    starting_cash: float = 100000.0
    commission_bps: float = 1.0
    slippage_bps: float = 1.0
    equity: float = field(init=False)
    trades: List[Trade] = field(default_factory=list)
    log: logging.Logger = field(default_factory=lambda: logging.getLogger("eden.backtest.engine"))

    def __post_init__(self):
        self.equity = self.starting_cash
        self.position_qty = 0.0
        self.position_side = None  # 'long' or 'short'
        self.entry_price = 0.0
        self.entry_time = None
        self.trade_log_rows = []

    def _apply_slippage(self, price: float, side: str) -> float:
        slip = price * (self.slippage_bps / 10000.0)
        return price + slip if side == 'buy' else price - slip

    def _apply_commission(self, notional: float) -> float:
        return notional * (self.commission_bps / 10000.0)

    def run(self, df: pd.DataFrame, signals: pd.DataFrame, symbol: str, risk_manager=None) -> List[Trade]:
        # Handle missing or empty signals gracefully
        if signals is None or len(signals) == 0 or 'timestamp' not in signals.columns:
            return self.trades
        sig = signals.sort_values("timestamp").reset_index(drop=True)
        price_series = df['close']
        sig_idx = 0
        sig_rows = sig.to_dict("records")

        for ts, price in price_series.items():
            # Process all signals at this timestamp
            while sig_idx < len(sig_rows) and sig_rows[sig_idx]["timestamp"] <= ts:
                row = sig_rows[sig_idx]
                sig_idx += 1
                side = row["side"]
                conf = float(row.get("confidence", 0.5))
                qty = max(1.0, self.equity * 0.01 / max(price, 1e-6)) * conf  # 1% of equity scaled by confidence
                if risk_manager and not risk_manager.allow_order(symbol, side, qty, price, self.equity):
                    continue
                self._execute_signal(ts, symbol, side, qty, price)

        # Close any open position at end
        if self.position_qty != 0:
            self._close_position(ts, symbol, price)

        return self.trades

    def _execute_signal(self, ts: datetime, symbol: str, side: str, qty: float, price: float):
        px = self._apply_slippage(price, side)
        notional = qty * px
        fee = self._apply_commission(notional)
        if side == 'buy':
            if self.position_side == 'short':
                # close short then open long
                self._close_position(ts, symbol, px)
            if self.position_side != 'long':
                self.entry_time = ts
                self.entry_price = px
                self.position_side = 'long'
                self.position_qty = qty
                self.equity -= fee
        else:  # sell signal
            if self.position_side == 'long':
                self._close_position(ts, symbol, px)
            if self.position_side != 'short':
                self.entry_time = ts
                self.entry_price = px
                self.position_side = 'short'
                self.position_qty = qty
                self.equity -= fee

    def _close_position(self, ts: datetime, symbol: str, exit_price: float):
        if self.position_side is None or self.position_qty == 0:
            return
        side = self.position_side
        qty = self.position_qty
        pnl = 0.0
        if side == 'long':
            pnl = (exit_price - self.entry_price) * qty
        else:
            pnl = (self.entry_price - exit_price) * qty
        fee = self._apply_commission(abs(exit_price * qty))
        pnl -= fee
        self.equity += pnl
        trade = Trade(
            open_time=self.entry_time,
            close_time=ts,
            symbol=symbol,
            side=side,
            qty=float(qty),
            entry_price=float(self.entry_price),
            exit_price=float(exit_price),
            pnl=float(pnl),
            strategy="multi",
        )
        self.trades.append(trade)
        # reset
        self.position_qty = 0.0
        self.position_side = None
        self.entry_price = 0.0
        self.entry_time = None

    def save_trades_csv(self, path: Path):
        import pandas as pd
        rows = [t.__dict__ for t in self.trades]
        df = pd.DataFrame(rows)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
