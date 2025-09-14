import numpy as np


def sharpe(returns, risk_free=0.0):
    r = np.array(returns)
    if r.std() == 0:
        return 0.0
    return (r.mean() - risk_free) / (r.std() + 1e-12) * np.sqrt(252)


def max_drawdown(equity_curve):
    ec = np.array(equity_curve)
    if ec.size == 0:
        return 0.0, 0
    peak = np.maximum.accumulate(ec)
    dd = (ec - peak) / np.where(peak == 0, 1e-12, peak)
    return float(dd.min()), int(dd.argmin())


def profit_factor(pnls):
    """Gross profit divided by gross loss (absolute). If no losses, return large value."""
    arr = np.array(pnls, dtype=float)
    gross_profit = arr[arr > 0].sum()
    gross_loss = -arr[arr < 0].sum()
    if gross_loss <= 0:
        return float('inf') if gross_profit > 0 else 0.0
    return float(gross_profit / (gross_loss + 1e-12))


def max_consecutive_losses(pnls):
    """Maximum number of losing trades in a row."""
    arr = np.array(pnls, dtype=float)
    max_streak = 0
    cur = 0
    for v in arr:
        if v < 0:
            cur += 1
            if cur > max_streak:
                max_streak = cur
        else:
            cur = 0
    return int(max_streak)
