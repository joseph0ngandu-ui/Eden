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
    # Dynamic risk sizing params
    per_order_risk_fraction: float = 0.02  # 2% of equity per trade by default
    min_trade_value: float = 0.50          # minimum dollar risk per trade
    growth_factor: float = 0.5             # smooth compounding exponent
    # Phase-2 controls
    enable_volatility_normalization: bool = True
    conf_cut_low: float = 0.0
    conf_cut_mid: float = 0.70
    conf_cut_high: float = 0.85
    risk_mult_low: float = 0.35
    risk_mult_mid: float = 0.60
    risk_mult_high: float = 1.00
    htf_strict_mode: bool = False
    volatility_cap: float = 3.0
    decision_log_path: Optional[Path] = None
    enable_stage_pipeline: bool = False
    meta_update_window_trades: int = 50
    controller_enable: bool = False
    equity: float = field(init=False)
    trades: List[Trade] = field(default_factory=list)
    log: logging.Logger = field(default_factory=lambda: logging.getLogger("eden.backtest.engine"))

    def __post_init__(self):
        self.equity = self.starting_cash
        self.max_equity = self.starting_cash
        self.position_qty = 0.0
        self.position_side = None  # 'long' or 'short'
        self.entry_price = 0.0
        self.entry_time: datetime | None = None
        self.trade_log_rows = []
        self.open_tag: str | None = None
        self.open_rrr: float = 0.0
        # Phase-2 open-state telemetry
        self._open_conf: float | None = None
        self._open_risk_mult: float | None = None
        self._open_vol_factor: float | None = None
        self._open_final_risk_usd: float | None = None
        self._open_strategy: str | None = None
        self._open_blocked_htf: bool | None = None
        self._open_ml_override: bool | None = None
        # decision log
        self._decision_log_file = None
        if self.decision_log_path:
            try:
                self.decision_log_path.parent.mkdir(parents=True, exist_ok=True)
                if not self.decision_log_path.exists():
                    self.decision_log_path.write_text("timestamp,reason,strategy,conf,vol_factor,dd_pct,risk_scale,action\n", encoding="utf-8")
            except Exception:
                pass

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

        # Import here to avoid hard dependency where not needed
        try:
            from eden.risk.volatility_adapter import compute_volatility_factor, adjust_stop_and_size, VolatilityConfig
            vol_cfg = VolatilityConfig(cap_max=self.volatility_cap)
        except Exception:
            compute_volatility_factor = None
            adjust_stop_and_size = None
            vol_cfg = None

        for ts, price in price_series.items():
            # Snapshot row at this timestamp for ATR etc.
            df_row = (df.loc[ts] if ts in df.index else None)
            df_row_dict = (df_row.to_dict() if df_row is not None else {})
            # Process all signals at this timestamp
            while sig_idx < len(sig_rows) and sig_rows[sig_idx]["timestamp"] <= ts:
                row = sig_rows[sig_idx]
                sig_idx += 1
                side = row.get("side")
                conf = float(row.get("confidence", 0.5))
                strategy_name = row.get("strategy", "unknown")

                # Strict HTF gating
                blocked_by_htf = False
                ml_override = False
                if self.htf_strict_mode and df_row is not None:
                    try:
                        bias = int(df_row_dict.get("HTF_BIAS", 0))
                        bias_ok = (bias == 1 and side == 'buy') or (bias == -1 and side == 'sell')
                        # Alignment checks (optional columns)
                        if side == 'buy':
                            fvg_ok = bool(int(df_row_dict.get('HTF_FVG_BULL', 0)))
                            ob_ok = int(df_row_dict.get('HTF_OB_COUNT_BULL', 0)) >= 1
                            no_opp_sweep = not bool(int(df_row_dict.get('HTF_RECENT_SWEEP_HIGH', 0)))
                        else:
                            fvg_ok = bool(int(df_row_dict.get('HTF_FVG_BEAR', 0)))
                            ob_ok = int(df_row_dict.get('HTF_OB_COUNT_BEAR', 0)) >= 1
                            no_opp_sweep = not bool(int(df_row_dict.get('HTF_RECENT_SWEEP_LOW', 0)))
                        checks = [bias_ok, (fvg_ok or ob_ok), no_opp_sweep]
                        passed = sum(1 for c in checks if c)
                        # Compute volatility factor for override condition
                        vol_factor_tmp = 1.0
                        if self.enable_volatility_normalization and compute_volatility_factor is not None:
                            vol_factor_tmp = compute_volatility_factor(df_row_dict, vol_cfg)
                        if passed < 2 and not (conf >= 0.95 and vol_factor_tmp < 1.5):
                            blocked_by_htf = True
                    except Exception:
                        pass
                if blocked_by_htf:
                    # Decision log with required columns
                    try:
                        if self.decision_log_path:
                            with self.decision_log_path.open("a", encoding="utf-8") as f:
                                f.write(f"{ts},,,{strategy_name},{int(df_row_dict.get('HTF_BIAS',0))},1,HTF_STRICT,0,0,0,,{1.0},{0.0},block,1,0,\n")
                    except Exception:
                        pass
                    continue

                # Determine stop distance for risk sizing
                stop_price = row.get('stop_price')
                atr_val = float(row.get('atr', df_row_dict.get('atr_14', 0.0)))
                default_stop_dist = max(atr_val, float(price) * 0.005)
                if stop_price is not None and np.isfinite(stop_price):
                    stop_dist = abs(price - float(stop_price))
                    if stop_dist <= 0:
                        stop_dist = default_stop_dist
                else:
                    stop_dist = default_stop_dist

                # Volatility normalization
                vol_factor = 1.0
                if self.enable_volatility_normalization and compute_volatility_factor is not None:
                    try:
                        vol_factor = compute_volatility_factor(df_row_dict, vol_cfg)
                    except Exception:
                        vol_factor = 1.0

                # Confidence-weighted risk mapping
                if conf < self.conf_cut_low:
                    # Skip low-confidence trades
                    try:
                        if self.decision_log_path:
                            with self.decision_log_path.open("a", encoding="utf-8") as f:
                                f.write(f"{ts},CONF_LOW_SKIP,{strategy_name},{conf},{vol_factor},,,skip\n")
                    except Exception:
                        pass
                    continue
                if conf >= self.conf_cut_high:
                    risk_mult = self.risk_mult_high
                elif conf >= self.conf_cut_mid:
                    risk_mult = self.risk_mult_mid
                else:
                    risk_mult = self.risk_mult_low

                # Staged ML pipeline (optional)
                stageA_P = None; stageB_P = None; stageC_P = None; regime_tag = ''
                if self.enable_stage_pipeline:
                    try:
                        from eden.ml.stageA_liquidity_detector import run_stageA_row
                        from eden.ml.stageB_continuation_predictor import run_stageB_row
                        from eden.ml.stageC_combiner import combine_stage_outputs
                        from eden.ml.regime_detector import detect_regime
                        # build row dict for stages
                        stage_row = dict(df_row_dict)
                        stageA_P, _ = run_stageA_row(stage_row)
                        stageB_P, _ = run_stageB_row(stage_row)
                        regime_tag = detect_regime(df.iloc[: df.index.get_loc(ts)+1]) if ts in df.index else 'normal'
                        stageC_P, stageC_risk_mult = combine_stage_outputs(stageA_P or 0.0, stageB_P or 0.0, int(df_row_dict.get('HTF_BIAS',0)), regime_tag, conf)
                    except Exception:
                        stageA_P = stageA_P or 0.0
                        stageB_P = stageB_P or 0.0
                        stageC_P = None
                        stageC_risk_mult = None
                # Deterministic controller
                ctrl_scale = 1.0
                pause_ml = False
                # drawdown
                dd_pct = 0.0 if self.max_equity <= 0 else (self.max_equity - self.equity) / self.max_equity
                if self.controller_enable:
                    if dd_pct > 0.08:
                        ctrl_scale *= 0.5
                    if vol_factor > 2.0 and (strategy_name or '').lower() == 'ml_generated':
                        pause_ml = True
                    # ML recent winrate adjustment
                    try:
                        last_ml = [t for t in self.trades[-50:] if (t.strategy or '').lower() == 'ml_generated']
                        if last_ml:
                            wins = sum(1 for t in last_ml if t.pnl > 0)
                            winrate = wins / max(1, len(last_ml))
                            if winrate < 0.40:
                                if (strategy_name or '').lower() == 'ml_generated':
                                    ctrl_scale *= 0.5
                    except Exception:
                        pass
                if pause_ml:
                    try:
                        if self.decision_log_path:
                            with self.decision_log_path.open("a", encoding="utf-8") as f:
                                f.write(f"{ts},VOL>2_PAUSE_ML,{strategy_name},{conf},{vol_factor},{dd_pct},{ctrl_scale},skip\n")
                    except Exception:
                        pass
                    continue

                # Risk per trade dollars (base)
                base_risk = max(self.equity * self.per_order_risk_fraction, self.min_trade_value)
                # Final risk with confidence + volatility + controller
                # If staged pipeline produced risk multiplier, use it; else use confidence mapping
                if self.enable_stage_pipeline and stageC_P is not None and stageC_risk_mult is not None:
                    risk_mult_eff = float(stageC_risk_mult)
                else:
                    risk_mult_eff = float(risk_mult)
                final_risk_usd = base_risk * risk_mult_eff * ctrl_scale / max(1e-12, vol_factor)
                if final_risk_usd < self.min_trade_value:
                    try:
                        if self.decision_log_path:
                            with self.decision_log_path.open("a", encoding="utf-8") as f:
                                f.write(f"{ts},,,{strategy_name},{int(df_row_dict.get('HTF_BIAS',0))},,,{stageA_P or 0},{stageB_P or 0},{stageC_P or 0},{regime_tag},{vol_factor},{base_risk},{final_risk_usd},skip,1,0,\n")
                    except Exception:
                        pass
                    continue
                final_risk_usd = base_risk * risk_mult * ctrl_scale / max(1e-12, vol_factor)

                # Base position from risk and stop
                base_qty = final_risk_usd / max(stop_dist, 1e-8)

                # Growth scaling
                equity_ratio = self.equity / max(self.starting_cash, 1e-8)
                if self.growth_factor > 0 and equity_ratio > 0:
                    growth_mult = equity_ratio ** self.growth_factor
                else:
                    growth_mult = 1.0
                growth_mult = float(np.real(growth_mult)) if isinstance(growth_mult, complex) else float(growth_mult)

                # Adjust for volatility adapter (stop and max pos)
                adj_stop = stop_dist
                adj_qty = base_qty
                if self.enable_volatility_normalization and adjust_stop_and_size is not None:
                    try:
                        adj_stop, adj_qty, meta = adjust_stop_and_size(stop_dist, base_qty, vol_factor, vol_cfg)
                    except Exception:
                        adj_stop, adj_qty = stop_dist, base_qty

                qty = max(0.0, float(adj_qty) * growth_mult)

                if risk_manager and not risk_manager.allow_order(symbol, side, qty, price, self.equity):
                    continue

                # Compute simple RRR if tp_price is available
                tp_price = row.get('tp_price')
                rrr = 0.0
                if adj_stop > 0 and tp_price is not None and np.isfinite(tp_price):
                    reward = abs(float(tp_price) - price)
                    rrr = float(reward / max(adj_stop, 1e-8)) if reward > 0 else 0.0

                tag = row.get('tag')
                # Stash open-state telemetry
                self._open_conf = conf
                self._open_risk_mult = risk_mult
                self._open_vol_factor = vol_factor
                self._open_final_risk_usd = final_risk_usd
                self._open_strategy = strategy_name
                self._open_ml_override = False  # reserved
                self._open_blocked_htf = False
                # Execute
                self._execute_signal(ts, symbol, side, qty, price, tag=tag, rrr=rrr)
                # Decision log (full columns)
                try:
                    if self.decision_log_path:
                        with self.decision_log_path.open("a", encoding="utf-8") as f:
                            f.write(f"{ts},,,{strategy_name},{int(df_row_dict.get('HTF_BIAS',0))},,,{stageA_P or 0},{stageB_P or 0},{stageC_P or 0},{regime_tag},{vol_factor},{base_risk},{final_risk_usd},enter,0,0,\n")
                except Exception:
                    pass

        # Close any open position at end
        if self.position_qty != 0:
            self._close_position(ts, symbol, price)

        return self.trades

    def _execute_signal(self, ts: datetime, symbol: str, side: str, qty: float, price: float, tag: str | None = None, rrr: float = 0.0):
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
                self.open_tag = tag
                self.open_rrr = rrr
        else:  # sell signal
            if self.position_side == 'long':
                self._close_position(ts, symbol, px)
            if self.position_side != 'short':
                self.entry_time = ts
                self.entry_price = px
                self.position_side = 'short'
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
        if side == 'long':
            pnl = (exit_price - self.entry_price) * qty
        else:
            pnl = (self.entry_price - exit_price) * qty
        fee = self._apply_commission(abs(exit_price * qty))
        pnl -= fee
        self.equity += pnl
        # Track max equity for drawdown controller
        if self.equity > getattr(self, 'max_equity', self.equity):
            self.max_equity = self.equity
        trade = Trade(
            open_time=self.entry_time,
            close_time=ts,
            symbol=symbol,
            side=side,
            qty=float(qty),
            entry_price=float(self.entry_price),
            exit_price=float(exit_price),
            pnl=float(pnl),
            strategy=self._open_strategy or "multi",
            tag=self.open_tag,
            rrr=float(self.open_rrr),
            model_confidence=self._open_conf,
            risk_multiplier=self._open_risk_mult,
            volatility_factor=self._open_vol_factor,
            final_risk_usd=self._open_final_risk_usd,
            blocked_by_htf_strict=self._open_blocked_htf,
            ml_override=self._open_ml_override,
        )
        self.trades.append(trade)
        # reset
        self.position_qty = 0.0
        self.position_side = None
        self.entry_price = 0.0
        self.entry_time = None
        self.open_tag = None
        self.open_rrr = 0.0
        self._open_conf = None
        self._open_risk_mult = None
        self._open_vol_factor = None
        self._open_final_risk_usd = None
        self._open_strategy = None
        self._open_blocked_htf = None
        self._open_ml_override = None

    def save_trades_csv(self, path: Path):
        import pandas as pd
        rows = [t.__dict__ for t in self.trades]
        df = pd.DataFrame(rows)
        df['starting_cash'] = float(self.starting_cash)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
