#!/usr/bin/env python3
"""
Simulate small-account compounding with VB v1.3 (optimized params)

Equity model: PnL per trade = risk_amount * R
Risk ladder:
- < $30: 20%
- $30-$100: 10%
- $100-$500: 5%
- $500-$1000: 3%
- $1000+: 1%
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from backtest_volatility_burst import VBBacktester


def risk_pct(equity: float) -> float:
    if equity < 30:
        return 0.20
    if equity < 100:
        return 0.10
    if equity < 500:
        return 0.05
    if equity < 1000:
        return 0.03
    return 0.01


def run_compounding(start_equity=100.0):
    symbols = [
        "Volatility 75 Index",
        "Crash 500 Index",
        "Boom 1000 Index",
    ]
    start = datetime(2025, 1, 1)
    end = datetime(2025, 10, 31)

    # Best params from optimization
    overrides = {
        'confidence_threshold': 0.7,
        'tp_atr_multiplier': 1.2,
        'sl_atr_multiplier': 1.2,
    }

    bt = VBBacktester("config/volatility_burst.yml")

    all_trades = []
    for sym in symbols:
        stats = bt.backtest_symbol(sym, start, end, param_overrides=overrides)
        if stats is None:
            continue
        # Collect trades with symbol
        trades = bt.results[sym]['trades']
        for t in trades:
            t['symbol'] = sym
        all_trades.extend(trades)

    if not all_trades:
        print("No trades for compounding simulation.")
        return

    # Sort trades chronologically by exit_time
    df = pd.DataFrame(all_trades)
    df['exit_time'] = pd.to_datetime(df['exit_time'])
    df = df.sort_values('exit_time').reset_index(drop=True)

    equity = start_equity
    equity_curve = [equity]
    results = []

    for _, row in df.iterrows():
        r = row.get('r_value', 0.0)
        rp = risk_pct(equity)
        risk_amount = equity * rp
        pnl = risk_amount * float(r)
        equity += pnl
        equity_curve.append(equity)
        results.append({
            'time': row['exit_time'],
            'symbol': row['symbol'],
            'r': r,
            'risk_pct': rp,
            'risk_amount': risk_amount,
            'pnl': pnl,
            'equity': equity,
        })

    df_res = pd.DataFrame(results)
    print(f"\nSMALL ACCOUNT COMPOUNDING (VB v1.3 optimized)\nStart: ${start_equity:.2f} -> End: ${equity:.2f} | Return: {(equity/start_equity-1)*100:.1f}%")
    # Monthly summary
    df_res['month'] = df_res['time'].dt.to_period('M')
    monthly = df_res.groupby('month')['pnl'].sum().to_frame('PnL')
    print("\nMonthly PnL:\n", monthly)

    out_dir = Path('reports')
    out_dir.mkdir(exist_ok=True)
    df_res.to_csv(out_dir / 'vb_v1.3_compounding_results.csv', index=False)
    print("\nSaved detailed results to reports/vb_v1.3_compounding_results.csv")


if __name__ == '__main__':
    run_compounding(100.0)
