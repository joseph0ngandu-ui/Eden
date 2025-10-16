from __future__ import annotations
import json
from dataclasses import dataclass
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..utils.metrics import sharpe, max_drawdown, profit_factor, max_consecutive_losses
from ..utils.types import Trade


@dataclass
class Analyzer:
    trades: List[Trade]
    starting_cash: float = 100000.0

    def equity_curve(self) -> pd.Series:
        if not self.trades:
            return pd.Series(dtype=float)
        # cumulative PnL over time
        df = pd.DataFrame([t.__dict__ for t in self.trades]).sort_values("close_time")
        pnl = df["pnl"].cumsum()
        base = float(self.starting_cash)
        equity = base + pnl
        equity.index = df["close_time"]
        return equity

    def metrics(self) -> dict:
        eq = self.equity_curve()
        if eq.empty:
            return {"net_pnl": 0.0, "sharpe": 0.0, "max_dd": 0.0, "trades": 0, "profit_factor": 0.0, "max_consec_losses": 0}
        returns = eq.pct_change().fillna(0.0)
        net_pnl = float(eq.iloc[-1] - eq.iloc[0])
        sh = float(sharpe(returns.values))
        dd, dd_idx = max_drawdown(eq.values)
        # Trade series stats
        tdf = pd.DataFrame([t.__dict__ for t in self.trades])
        pf = float(profit_factor(tdf['pnl'].values)) if not tdf.empty else 0.0
        mcl = int(max_consecutive_losses(tdf['pnl'].values)) if not tdf.empty else 0
        out = {
            "net_pnl": net_pnl,
            "sharpe": sh,
            "max_dd": float(dd),
            "trades": int(len(self.trades)),
            "profit_factor": pf,
            "max_consec_losses": mcl,
        }
        return out

    def plot_equity_curve(self, save_path=None):
        eq = self.equity_curve()
        if eq.empty:
            return
        plt.figure(figsize=(8, 4))
        plt.plot(eq.index, eq.values)
        plt.title("Equity Curve")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
