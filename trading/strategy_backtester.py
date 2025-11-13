#!/usr/bin/env python3
"""
Simple strategy backtester for MA and RSI strategies.
Uses OHLC data (DataFrame) and returns performance metrics.
"""

from typing import Dict
import pandas as pd
import numpy as np


def backtest_ma(df: pd.DataFrame, fast: int, slow: int, hold_bars: int = 5) -> Dict:
    prices = df.copy()
    prices['MA_fast'] = prices['close'].rolling(fast).mean()
    prices['MA_slow'] = prices['close'].rolling(slow).mean()
    prices.dropna(inplace=True)

    signals = np.where((prices['MA_fast'] > prices['MA_slow']) & (prices['MA_fast'].shift(1) <= prices['MA_slow'].shift(1)), 1, 0)
    entries = np.where(signals==1)[0]

    wins, losses, pnl_list = 0, 0, []

    for idx in entries:
        entry_price = prices['close'].iloc[idx]
        exit_idx = min(idx + hold_bars, len(prices)-1)
        exit_price = prices['close'].iloc[exit_idx]
        pnl = (exit_price - entry_price)
        pnl_list.append(pnl)
        if pnl > 0: wins += 1
        else: losses += 1

    total_trades = len(pnl_list)
    total_profit = sum(p for p in pnl_list if p>0)
    total_loss = abs(sum(p for p in pnl_list if p<=0))

    return _metrics(total_trades, wins, losses, total_profit, total_loss)


def backtest_rsi(df: pd.DataFrame, period: int, oversold: int, overbought: int, hold_bars: int = 5) -> Dict:
    prices = df.copy()
    delta = prices['close'].diff()
    up, down = delta.clip(lower=0), -1*delta.clip(upper=0)
    roll_up = up.rolling(period).mean()
    roll_down = down.rolling(period).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    prices['RSI'] = 100.0 - (100.0 / (1.0 + rs))
    prices.dropna(inplace=True)

    signals = np.where((prices['RSI'] < oversold) & (prices['RSI'].shift(1) >= oversold), 1, 0)
    entries = np.where(signals==1)[0]

    wins, losses, pnl_list = 0, 0, []

    for idx in entries:
        entry_price = prices['close'].iloc[idx]
        exit_idx = min(idx + hold_bars, len(prices)-1)
        exit_price = prices['close'].iloc[exit_idx]
        pnl = (exit_price - entry_price)
        pnl_list.append(pnl)
        if pnl > 0: wins += 1
        else: losses += 1

    total_trades = len(pnl_list)
    total_profit = sum(p for p in pnl_list if p>0)
    total_loss = abs(sum(p for p in pnl_list if p<=0))

    return _metrics(total_trades, wins, losses, total_profit, total_loss)


def _metrics(total_trades, wins, losses, total_profit, total_loss) -> Dict:
    win_rate = wins/total_trades if total_trades>0 else 0.0
    profit_factor = (total_profit/total_loss) if total_loss>0 else 0.0
    net_profit = total_profit - total_loss
    return {
        'total_trades': total_trades,
        'winning_trades': wins,
        'losing_trades': losses,
        'win_rate': win_rate,
        'avg_win': (total_profit/wins if wins>0 else 0.0),
        'avg_loss': (total_loss/losses if losses>0 else 0.0),
        'total_profit': total_profit,
        'total_loss': total_loss,
        'net_profit': net_profit,
        'profit_factor': profit_factor,
        'max_drawdown': 0.0,
        'sharpe_ratio': 0.0,
    }
