from __future__ import annotations
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from ..utils.types import Trade


@dataclass
class BacktestEngine:
    starting_cash: float = 100000.0
    commission_bps: float = 1.0
    slippage_bps: float = 1.0
    # Dynamic risk sizing params
    per_order_risk_fraction: float = 0.02  # 2% of equity per trade by default
    min_trade_value: float = 0.50  # minimum dollar risk per trade
    growth_factor: float = 0.5  # smooth compounding exponent
    equity: float = field(init=False)
    trades: List[Trade] = field(default_factory=list)
    log: logging.Logger = field(
        default_factory=lambda: logging.getLogger("eden.backtest.engine")
    )

    def __post_init__(self):
        self.equity = self.starting_cash
        self.position_qty = 0.0
        self.position_side = None  # 'long' or 'short'
        self.entry_price = 0.0
        self.entry_time: datetime | None = None
        self.trade_log_rows = []
        self.open_tag: str | None = None
        self.open_rrr: float = 0.0

    def _apply_slippage(self, price: float, side: str) -> float:
        slip = price * (self.slippage_bps / 10000.0)
        return price + slip if side == "buy" else price - slip

    def _apply_commission(self, notional: float) -> float:
        return notional * (self.commission_bps / 10000.0)

    def run(
        self, df: pd.DataFrame, signals: pd.DataFrame, symbol: str, risk_manager=None
    ) -> List[Trade]:
        # Handle missing or empty signals gracefully
        if signals is None or len(signals) == 0 or "timestamp" not in signals.columns:
            return self.trades
        sig = signals.sort_values("timestamp").reset_index(drop=True)
        price_series = df["close"]
        sig_idx = 0
        sig_rows = sig.to_dict("records")

        for ts, price in price_series.items():
            # Snapshot row at this timestamp for ATR etc.
            df_row = df.loc[ts] if ts in df.index else None
            # Process all signals at this timestamp
            while sig_idx < len(sig_rows) and sig_rows[sig_idx]["timestamp"] <= ts:
                row = sig_rows[sig_idx]
                sig_idx += 1
                side = row["side"]
                conf = float(row.get("confidence", 0.5))

                # Determine stop distance for risk sizing
                stop_price = row.get("stop_price")
                atr_val = float(
                    row.get(
                        "atr",
                        (
                            df_row["atr_14"]
                            if df_row is not None and "atr_14" in df_row
                            else 0.0
                        ),
                    )
                )
                default_stop_dist = max(atr_val, price * 0.005)
                if stop_price is not None and np.isfinite(stop_price):
                    stop_dist = abs(price - float(stop_price))
                    if stop_dist <= 0:
                        stop_dist = default_stop_dist
                else:
                    stop_dist = default_stop_dist

                # Risk per trade dollars
                risk_per_trade = max(
                    self.equity * self.per_order_risk_fraction, self.min_trade_value
                )
                # Base position from risk budget and stop distance
                base_qty = risk_per_trade / max(stop_dist, 1e-8)
                # Smooth growth scaling - ensure no complex numbers
                equity_ratio = self.equity / max(self.starting_cash, 1e-8)
                if self.growth_factor > 0 and equity_ratio > 0:
                    growth_mult = equity_ratio**self.growth_factor
                else:
                    growth_mult = 1.0

                # Ensure all components are real numbers
                growth_mult = (
                    float(np.real(growth_mult))
                    if isinstance(growth_mult, complex)
                    else float(growth_mult)
                )
                conf = (
                    float(np.real(conf)) if isinstance(conf, complex) else float(conf)
                )
                base_qty = (
                    float(np.real(base_qty))
                    if isinstance(base_qty, complex)
                    else float(base_qty)
                )

                qty = max(0.0, base_qty * conf * growth_mult)

                if risk_manager and not risk_manager.allow_order(
                    symbol, side, qty, price, self.equity
                ):
                    continue

                # Compute simple RRR if tp_price is available
                tp_price = row.get("tp_price")
                rrr = 0.0
                if stop_dist > 0 and tp_price is not None and np.isfinite(tp_price):
                    reward = abs(float(tp_price) - price)
                    rrr = float(reward / max(stop_dist, 1e-8)) if reward > 0 else 0.0

                tag = row.get("tag")
                self._execute_signal(ts, symbol, side, qty, price, tag=tag, rrr=rrr)

        # Close any open position at end
        if self.position_qty != 0:
            self._close_position(ts, symbol, price)

        return self.trades

    def _execute_signal(
        self,
        ts: datetime,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        tag: str | None = None,
        rrr: float = 0.0,
    ):
        px = self._apply_slippage(price, side)
        notional = qty * px
        fee = self._apply_commission(notional)
        if side == "buy":
            if self.position_side == "short":
                # close short then open long
                self._close_position(ts, symbol, px)
            if self.position_side != "long":
                self.entry_time = ts
                self.entry_price = px
                self.position_side = "long"
                self.position_qty = qty
                self.equity -= fee
                self.open_tag = tag
                self.open_rrr = rrr
        else:  # sell signal
            if self.position_side == "long":
                self._close_position(ts, symbol, px)
            if self.position_side != "short":
                self.entry_time = ts
                self.entry_price = px
                self.position_side = "short"
                self.position_qty = qty
                self.equity -= fee
                self.open_tag = tag
                self.open_rrr = rrr

    def _close_position(self, ts: datetime, symbol: str, exit_price: float):
        if self.position_side is None or self.position_qty == 0:
            return
        side = self.position_side
        qty = self.position_qty
        pnl = 0.0
        if side == "long":
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
            tag=self.open_tag,
            rrr=float(self.open_rrr),
        )
        self.trades.append(trade)
        # reset
        self.position_qty = 0.0
        self.position_side = None
        self.entry_price = 0.0
        self.entry_time = None
        self.open_tag = None
        self.open_rrr = 0.0

    def save_trades_csv(self, path: Path):
        import pandas as pd

        rows = [t.__dict__ for t in self.trades]
        df = pd.DataFrame(rows)
        df["starting_cash"] = float(self.starting_cash)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
