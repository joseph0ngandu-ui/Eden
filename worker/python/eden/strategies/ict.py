from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Optional

from .base import StrategyBase


class ICTStrategy(StrategyBase):
    name = "ict"

    def __init__(
        self,
        require_htf_bias: bool = True,
        ml_fallback_on_neutral: bool = True,
        stop_atr_multiplier: float = 1.2,
        tp_atr_multiplier: float = 2.0,
        min_confidence: float = 0.6,
        killzones_enabled: bool = False,
        killzones: str = "london,ny",  # comma-separated labels
        london_window: str = "07-10",
        ny_window: str = "12-16",
    ):
        self.require_htf_bias = require_htf_bias
        self.ml_fallback_on_neutral = ml_fallback_on_neutral
        self.stop_atr_multiplier = stop_atr_multiplier
        self.tp_atr_multiplier = tp_atr_multiplier
        self.min_confidence = min_confidence
        self.killzones_enabled = killzones_enabled
        self.killzones = {k.strip().lower() for k in killzones.split(",") if k.strip()}
        self.london_window = london_window
        self.ny_window = ny_window

    def _col_scalar(self, df: pd.DataFrame, name: str, idx: int, default=0):
        col = df.get(name)
        if col is None:
            return default
        if isinstance(col, pd.DataFrame):
            # duplicate column names: take first occurrence
            if col.shape[1] > 0:
                col = col.iloc[:, 0]
            else:
                return default
        try:
            return col.iloc[idx]
        except Exception:
            return default

    def liquidity_sweep_entry(self, df: pd.DataFrame, idx: int) -> Optional[dict]:
        """
        Detect liquidity sweep entry: micro sweep followed by rejection candle
        """
        if idx < 1:
            return None

        # Check for sweep
        sweep_low = self._col_scalar(df, "sweep_low", idx, default=0)
        sweep_high = self._col_scalar(df, "sweep_high", idx, default=0)

        # Get HTF bias
        htf_bias = self._col_scalar(df, "HTF_BIAS", idx, default=0)

        # Check for rejection (close back in direction)
        close = df["close"].iloc[idx]
        open_price = df["open"].iloc[idx]

        # Bullish sweep entry
        if sweep_low and htf_bias >= 0:
            # Rejection candle: closes higher than open after sweeping low
            if close > open_price:
                confidence = 0.75 if htf_bias == 1 else 0.65
                return {
                    "timestamp": df.index[idx],
                    "side": "buy",
                    "confidence": confidence,
                    "tag": "liquidity_sweep",
                    "htf_bias": int(htf_bias),
                }

        # Bearish sweep entry
        if sweep_high and htf_bias <= 0:
            if close < open_price:
                confidence = 0.75 if htf_bias == -1 else 0.65
                return {
                    "timestamp": df.index[idx],
                    "side": "sell",
                    "confidence": confidence,
                    "tag": "liquidity_sweep",
                    "htf_bias": int(htf_bias),
                }

        return None

    def order_block_retest(self, df: pd.DataFrame, idx: int) -> Optional[dict]:
        """
        Detect order block retest with micro-imbalance confirmation
        """
        if idx < 2:
            return None

        ob_bull_series = df.get("ob_bull")
        if isinstance(ob_bull_series, pd.DataFrame):
            ob_bull_series = ob_bull_series.iloc[:, 0]
        if ob_bull_series is None:
            ob_bull_series = pd.Series(0, index=df.index)
        ob_bull = ob_bull_series.iloc[max(0, idx - 2) : idx + 1].sum() > 0

        ob_bear_series = df.get("ob_bear")
        if isinstance(ob_bear_series, pd.DataFrame):
            ob_bear_series = ob_bear_series.iloc[:, 0]
        if ob_bear_series is None:
            ob_bear_series = pd.Series(0, index=df.index)
        ob_bear = ob_bear_series.iloc[max(0, idx - 2) : idx + 1].sum() > 0

        micro_imbalance_bull = self._col_scalar(
            df, "micro_imbalance_bull", idx, default=0
        )
        micro_imbalance_bear = self._col_scalar(
            df, "micro_imbalance_bear", idx, default=0
        )

        htf_bias = self._col_scalar(df, "HTF_BIAS", idx, default=0)

        # Bullish OB retest
        if ob_bull and micro_imbalance_bull and htf_bias >= 0:
            confidence = 0.7 if htf_bias == 1 else 0.6
            return {
                "timestamp": df.index[idx],
                "side": "buy",
                "confidence": confidence,
                "tag": "order_block_retest",
                "htf_bias": int(htf_bias),
            }

        # Bearish OB retest
        if ob_bear and micro_imbalance_bear and htf_bias <= 0:
            confidence = 0.7 if htf_bias == -1 else 0.6
            return {
                "timestamp": df.index[idx],
                "side": "sell",
                "confidence": confidence,
                "tag": "order_block_retest",
                "htf_bias": int(htf_bias),
            }

        return None

    def fvg_entry(self, df: pd.DataFrame, idx: int) -> Optional[dict]:
        """
        Detect FVG entry with HTF alignment
        """
        if idx < 1:
            return None

        fvg_bull = self._col_scalar(df, "fvg_bull", idx, default=0)
        fvg_bear = self._col_scalar(df, "fvg_bear", idx, default=0)

        htf_bias = self._col_scalar(df, "HTF_BIAS", idx, default=0)

        # Check for confirmation candle
        close = df["close"].iloc[idx]
        open_price = df["open"].iloc[idx]

        # Bullish FVG entry
        if fvg_bull and htf_bias >= 0:
            if close > open_price:  # Confirmation
                confidence = 0.72 if htf_bias == 1 else 0.62
                return {
                    "timestamp": df.index[idx],
                    "side": "buy",
                    "confidence": confidence,
                    "tag": "fvg_entry",
                    "htf_bias": int(htf_bias),
                }

        # Bearish FVG entry
        if fvg_bear and htf_bias <= 0:
            if close < open_price:
                confidence = 0.72 if htf_bias == -1 else 0.62
                return {
                    "timestamp": df.index[idx],
                    "side": "sell",
                    "confidence": confidence,
                    "tag": "fvg_entry",
                    "htf_bias": int(htf_bias),
                }

        return None

    def calculate_stop_tp(self, df: pd.DataFrame, idx: int, side: str) -> dict:
        """
        Calculate stop-loss and take-profit levels
        """
        # Calculate ATR
        atr = df.get("atr_14", pd.Series(0, index=df.index))
        if atr.iloc[idx] == 0 or pd.isna(atr.iloc[idx]):
            # Fallback: use recent price range
            atr_val = (
                df["high"].iloc[max(0, idx - 14) : idx + 1].max()
                - df["low"].iloc[max(0, idx - 14) : idx + 1].min()
            ) / 14
        else:
            atr_val = atr.iloc[idx]

        entry_price = df["close"].iloc[idx]

        if side == "buy":
            stop = entry_price - (atr_val * self.stop_atr_multiplier)
            tp = entry_price + (atr_val * self.tp_atr_multiplier)
        else:  # sell
            stop = entry_price + (atr_val * self.stop_atr_multiplier)
            tp = entry_price - (atr_val * self.tp_atr_multiplier)

        return {"stop_price": float(stop), "tp_price": float(tp), "atr": float(atr_val)}

    def _parse_window(self, s: str, default: tuple[int, int]) -> tuple[int, int]:
        try:
            a, b = s.split("-")
            return (int(a), int(b))
        except Exception:
            return default

    def _in_killzone(self, ts) -> bool:
        try:
            t = pd.to_datetime(ts, utc=True).time()
        except Exception:
            return True  # if unknown, don't block
        h = t.hour
        lw = self._parse_window(getattr(self, "london_window", "07-10"), (7, 10))
        nw = self._parse_window(getattr(self, "ny_window", "12-16"), (12, 16))
        in_london = lw[0] <= h <= lw[1]
        in_ny = nw[0] <= h <= nw[1]
        allow = True
        if "london" in self.killzones and "ny" in self.killzones:
            allow = in_london or in_ny
        elif "london" in self.killzones:
            allow = in_london
        elif "ny" in self.killzones:
            allow = in_ny
        return allow

    def on_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate ICT trading signals with HTF bias validation and trade tagging
        """
        signals = []

        # Ensure HTF_BIAS column exists
        if "HTF_BIAS" not in df.columns:
            df["HTF_BIAS"] = 0

        # Iterate through data and check entry conditions
        for idx in range(len(df)):
            # Try each entry template
            signal = None

            # Priority order: liquidity sweep > order block retest > FVG
            signal = self.liquidity_sweep_entry(df, idx)
            if signal is None:
                signal = self.order_block_retest(df, idx)
            if signal is None:
                signal = self.fvg_entry(df, idx)

            # If signal found and meets confidence threshold
            if signal and signal["confidence"] >= self.min_confidence:
                # Kill zone gating
                if self.killzones_enabled and not self._in_killzone(df.index[idx]):
                    continue
                # Calculate stop and TP
                stop_tp = self.calculate_stop_tp(df, idx, signal["side"])
                signal.update(stop_tp)

                signals.append(signal)

        # Convert to DataFrame
        if signals:
            result = pd.DataFrame(signals)
            # Ensure required columns
            required_cols = [
                "timestamp",
                "side",
                "confidence",
                "tag",
                "htf_bias",
                "stop_price",
                "tp_price",
                "atr",
            ]
            for col in required_cols:
                if col not in result.columns:
                    result[col] = 0 if col in ["htf_bias"] else np.nan
            return result
        else:
            return pd.DataFrame(
                columns=[
                    "timestamp",
                    "side",
                    "confidence",
                    "tag",
                    "htf_bias",
                    "stop_price",
                    "tp_price",
                    "atr",
                ]
            )
