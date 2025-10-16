"""
Post-process backtest outputs to compute extended metrics and consolidated reports.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict
import sys

import numpy as np
import pandas as pd

# Ensure 'eden' package is importable if needed
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'worker' / 'python'))


def sharpe_ratio(returns: pd.Series, rf: float = 0.0, periods_per_year: int = 252) -> float:
    r = returns.dropna()
    if r.std() == 0:
        return 0.0
    return float((r.mean() - rf) / (r.std() + 1e-12) * np.sqrt(periods_per_year))


def sortino_ratio(returns: pd.Series, rf: float = 0.0, periods_per_year: int = 252) -> float:
    r = returns.dropna()
    downside = r[r < 0]
    dd = downside.std()
    if dd == 0:
        return 0.0
    return float((r.mean() - rf) / (dd + 1e-12) * np.sqrt(periods_per_year))


def max_drawdown(equity: pd.Series) -> (float, int):
    peak = equity.cummax()
    dd = (equity - peak) / (peak.replace(0, 1e-12))
    if dd.empty:
        return 0.0, 0
    
    dd_min_idx = dd.idxmin()
    dd_max_idx = dd.idxmax()
    
    # Handle datetime vs integer indices
    try:
        duration_days = int((dd_min_idx - dd_max_idx).days)
    except AttributeError:
        # For integer indices, assume they represent some time unit
        duration_days = int(abs(dd_min_idx - dd_max_idx))
    
    return float(dd.min()), duration_days


def expectancy(pnls: pd.Series) -> float:
    if pnls.empty:
        return 0.0
    return float(pnls.mean())


def average_rrr(trades: pd.DataFrame) -> float:
    if 'rrr' in trades.columns:
        s = trades['rrr'].dropna()
        return float(s.mean()) if not s.empty else 0.0
    # Fallback approximate RRR: |avg win| / |avg loss|
    wins = trades[trades['pnl'] > 0]['pnl']
    losses = trades[trades['pnl'] < 0]['pnl']
    if losses.empty:
        return float('inf') if not wins.empty else 0.0
    return float(abs(wins.mean()) / abs(losses.mean()) if not wins.empty else 0.0)


def load_trades_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    # Try parse times
    for c in ['open_time', 'close_time', 'timestamp']:
        if c in df.columns:
            try:
                df[c] = pd.to_datetime(df[c])
            except Exception:
                pass
    return df


def compute_metrics_for_run(run_dir: Path) -> Dict:
    trades = load_trades_csv(run_dir / 'trades.csv')
    if trades.empty:
        return {}
    trades['duration'] = (pd.to_datetime(trades['close_time']) - pd.to_datetime(trades['open_time'])).dt.total_seconds() / 60.0
    # Rebuild equity curve from trades
    start_equity = float(trades.get('starting_cash').iloc[0]) if 'starting_cash' in trades.columns and not trades['starting_cash'].empty else 100000.0
    eq = (trades['pnl'].cumsum() + start_equity)
    if not eq.empty:
        equity_growth_pct = float((eq.iloc[-1] / eq.iloc[0] - 1.0) * 100.0)
    else:
        equity_growth_pct = 0.0
    returns = eq.pct_change().fillna(0.0)

    net_pnl = float(trades['pnl'].sum())
    wins = trades[trades['pnl'] > 0]
    losses = trades[trades['pnl'] < 0]
    win_rate = float(len(wins) / len(trades)) * 100.0 if len(trades) else 0.0
    loss_rate = 100.0 - win_rate

    pf = float(wins['pnl'].sum() / abs(losses['pnl'].sum()) if not losses.empty else float('inf'))
    sh = sharpe_ratio(returns)
    so = sortino_ratio(returns)
    dd, dd_dur = max_drawdown(eq)

    # Trade size vs profitability correlation
    if {'qty','entry_price'}.issubset(trades.columns):
        trade_size = trades['qty'] * trades['entry_price']
        if trade_size.std() > 0 and trades['pnl'].std() > 0:
            size_profit_corr = float(trade_size.corr(trades['pnl']))
        else:
            size_profit_corr = 0.0
    else:
        size_profit_corr = 0.0

    metrics = {
        'net_pnl': net_pnl,
        'profit_factor': pf,
        'win_rate': win_rate,
        'loss_rate': loss_rate,
        'avg_trade_duration_min': float(trades['duration'].mean() if not trades.empty else 0.0),
        'avg_win': float(wins['pnl'].mean() if not wins.empty else 0.0),
        'avg_loss': float(losses['pnl'].mean() if not losses.empty else 0.0),
        'largest_win': float(wins['pnl'].max() if not wins.empty else 0.0),
        'largest_loss': float(losses['pnl'].min() if not losses.empty else 0.0),
        'trades': int(len(trades)),
        'expectancy': expectancy(trades['pnl']),
        'avg_rrr': average_rrr(trades),
        'sharpe': sh,
        'sortino': so,
        'max_drawdown_pct': float(dd * 100.0),
        'max_drawdown_duration_days': dd_dur,
        # Dynamic risk scaling metrics
        'equity_growth_pct': equity_growth_pct,
        'risk_adjusted_return_per_dollar': float(net_pnl / start_equity if start_equity > 0 else 0.0),
        'trade_size_profit_corr': size_profit_corr,
    }

    # Tag-based metrics (liquidity_sweep, order_block_retest, fvg_entry)
    metrics['tags'] = {}
    if 'tag' in trades.columns:
        for tag in ['liquidity_sweep', 'order_block_retest', 'fvg_entry']:
            t = trades[trades['tag'] == tag]
            if not t.empty:
                wins_t = t[t['pnl'] > 0]
                metrics['tags'][tag] = {
                    'count': int(len(t)),
                    'win_rate': float(len(wins_t) / len(t) * 100.0),
                    'avg_pnl': float(t['pnl'].mean()),
                    'total_pnl': float(t['pnl'].sum())
                }

    return metrics


def main(results_root: Path = Path('results')) -> Dict:
    # Iterate over backtest directories (backtests_M1, backtests_M5)
    consolidated: Dict[str, Dict] = {}

    for sub in results_root.glob('backtests_*'):
        for run in ['ict', 'mean_reversion', 'momentum', 'price_action', 'ml_generated', 'ensemble']:
            run_dir = sub / run
            if run_dir.exists():
                m = compute_metrics_for_run(run_dir)
                if m:
                    consolidated[f"{run}_{sub.name.split('_')[-1]}"] = m

    # Save outputs
    (results_root / 'metrics.json').write_text(json.dumps(consolidated, indent=2))

    # Human-readable summary
    summary = []
    for k, v in consolidated.items():
        summary.append({
            'run': k,
            'Net PnL': v.get('net_pnl', 0.0),
            'Sharpe': v.get('sharpe', 0.0),
            'Max DD %': v.get('max_drawdown_pct', 0.0),
            'Win Rate %': v.get('win_rate', 0.0),
            'Trades': v.get('trades', 0)
        })
    (results_root / 'consolidated_report.json').write_text(json.dumps(summary, indent=2))

    # Merge trades for convenience
    merged_rows = []
    for sub in results_root.glob('backtests_*'):
        for run in ['ict', 'mean_reversion', 'momentum', 'price_action', 'ml_generated', 'ensemble']:
            run_dir = sub / run
            if run_dir.exists():
                t = load_trades_csv(run_dir / 'trades.csv')
                if not t.empty:
                    t['run'] = f"{run}_{sub.name.split('_')[-1]}"
                    merged_rows.append(t)
    if merged_rows:
        merged = pd.concat(merged_rows, ignore_index=True)
        merged.to_csv(results_root / 'trades_all.csv', index=False)

    return consolidated


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Postprocess Eden results')
    parser.add_argument('--output-dir', type=str, default='results')
    args = parser.parse_args()
    main(Path(args.output_dir))
