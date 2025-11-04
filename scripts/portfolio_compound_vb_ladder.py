#!/usr/bin/env python3
"""
Portfolio compounding with optimized Risk Ladder using VB v1.3 trades.
- Loads per-symbol best (prefers merged) and filters long-term profitable symbols
- Runs compounding with RiskLadder/PositionSizer and optimizes ladder params on a small grid
- Outputs JSON/CSV summaries under reports/
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import argparse
import json
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from backtest_volatility_burst import VBBacktester
from risk_ladder import RiskLadder, PositionSizer

REPORTS = Path('reports')
REPORTS.mkdir(exist_ok=True)


def load_profitable_symbols(best_path: Path) -> dict:
    with open(best_path, 'r') as f:
        data = json.load(f)
    # If merged, require long-term profitability
    selected = {}
    for sym, info in data.items():
        if 'long_term' in info and isinstance(info['long_term'], dict):
            lt_stats = (info['long_term'] or {}).get('stats', {})
            if lt_stats and float(lt_stats.get('total_pnl', 0) or 0) > 0:
                selected[sym] = info['overrides']
        else:
            stats = info.get('stats', {})
            if stats and float(stats.get('total_pnl', 0) or 0) > 0:
                selected[sym] = info['overrides']
    return selected


def fetch_trades(selected: dict, start: datetime, end: datetime):
    bt = VBBacktester('config/volatility_burst.yml')
    all_trades = []
    per_symbol_stats = {}
    for sym, overrides in selected.items():
        stats = bt.backtest_symbol(sym, start, end, param_overrides=overrides)
        if not stats:
            continue
        per_symbol_stats[sym] = stats
        for t in bt.results[sym]['trades']:
            t['symbol'] = sym
            t['strategy'] = 'VB_v1.3'
            all_trades.append(t)
    df = pd.DataFrame(all_trades)
    if df.empty:
        return pd.DataFrame(), per_symbol_stats
    df['exit_time'] = pd.to_datetime(df['exit_time'])
    df = df.sort_values('exit_time').reset_index(drop=True)
    return df, per_symbol_stats


def compute_max_drawdown(curve):
    peak = -float('inf')
    max_dd = 0.0
    for e in curve:
        peak = max(peak, e)
        dd = (peak - e) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)
    return max_dd


def compound_with_ladder(df: pd.DataFrame, start_equity: float, ladder_args: dict, daily_cap: int = 5):
    if df is None or df.empty:
        return {
            'start': start_equity,
            'end': start_equity,
            'return_pct': 0.0,
            'equity_curve': [start_equity],
            'max_dd_pct': 0.0,
            'trades': []
        }
    rl = RiskLadder(
        initial_balance=start_equity,
        growth_mode_enabled=True,
        high_aggression_below=ladder_args.get('high_aggression_below', 30.0),
        equity_step_size=ladder_args.get('equity_step_size', 50.0),
        equity_step_drawdown_limit=ladder_args.get('equity_step_drawdown_limit', 0.15),
    )
    sizer = PositionSizer(rl, pip_value=ladder_args.get('pip_value', 10.0))
    eq = start_equity
    curve = [eq]
    results = []
    daily_counts = {}

    for _, row in df.iterrows():
        day = pd.to_datetime(row['exit_time']).date()
        daily_counts.setdefault(day, 0)
        if daily_counts[day] >= daily_cap:
            continue
        # Adjust risk if equity step drawdown breached
        risk_pct = rl.get_adjusted_risk_pct()  # percent
        atr = float(row.get('entry_atr') or 0.0)
        # Calculate intended lot to ensure ATR-normalized risk (for reference)
        sizing = sizer.calculate(equity=eq, atr=atr)
        # Use risk_pct to compute risk amount
        risk_amount = eq * (risk_pct / 100.0)
        r = float(row.get('r_value', 0.0))
        pnl = risk_amount * r
        eq += pnl
        curve.append(eq)
        results.append({
            'time': row['exit_time'],
            'symbol': row['symbol'],
            'r': r,
            'atr': atr,
            'risk_pct': risk_pct,
            'lot_size': sizing['lot_size'],
            'pnl': pnl,
            'equity': eq,
            'tier': sizing['tier'],
        })
        daily_counts[day] += 1
        rl.update_balance(eq)
    max_dd = compute_max_drawdown(curve)
    return {
        'start': start_equity,
        'end': eq,
        'return_pct': (eq / start_equity - 1) * 100,
        'equity_curve': curve,
        'max_dd_pct': max_dd * 100,
        'trades': results,
        'ladder': ladder_args,
    }


def optimize_ladder(df: pd.DataFrame, start_equity: float):
    grid = {
        'high_aggression_below': [20.0, 30.0, 50.0],
        'equity_step_size': [25.0, 50.0, 100.0],
        'equity_step_drawdown_limit': [0.10, 0.15, 0.20],
        'pip_value': [10.0],
    }
    keys = list(grid.keys())
    from itertools import product
    best = None
    for combo in product(*[grid[k] for k in keys]):
        args = {k: v for k, v in zip(keys, combo)}
        sim = compound_with_ladder(df, start_equity, args)
        # Score: prioritize higher end equity, penalize DD > 50%
        score = sim['end'] - (0 if sim['max_dd_pct'] <= 50 else (sim['end'] * 0.5))
        if (best is None) or (score > best['score']):
            best = {'score': score, 'sim': sim}
    return best['sim'] if best else compound_with_ladder(df, start_equity, {})


def main():
    ap = argparse.ArgumentParser(description='Portfolio compound with optimized Risk Ladder')
    ap.add_argument('--start-equity', type=float, default=100.0)
    ap.add_argument('--days', type=int, default=14)
    ap.add_argument('--best', type=str, default='', help='Path to per-symbol best (merged preferred)')
    args = ap.parse_args()

    # Resolve best config
    if args.best:
        best_cfg_path = Path(args.best)
    else:
        merged = REPORTS / 'vb_v1.3_per_symbol_best_merged.json'
        best_cfg_path = merged if merged.exists() else (REPORTS / 'vb_v1.3_per_symbol_best.json')

    if not best_cfg_path.exists():
        print(f'Best-config file not found: {best_cfg_path}')
        sys.exit(1)

    selected = load_profitable_symbols(best_cfg_path)
    if not selected:
        print('No profitable symbols found in best-config file.')
        sys.exit(1)

    start = datetime.now() - timedelta(days=args.days)
    end = datetime.now()

    df_trades, per_symbol = fetch_trades(selected, start, end)
    if df_trades.empty:
        print('No trades for live period.')
        sys.exit(0)

    sim = optimize_ladder(df_trades, args.start_equity)

    # Save outputs
    trades_csv = REPORTS / 'live_sim_vb_ladder_trades.csv'
    pd.DataFrame(sim['trades']).to_csv(trades_csv, index=False)
    summary = {
        'timestamp': datetime.now().isoformat(),
        'start_equity': sim['start'],
        'end_equity': sim['end'],
        'return_percent': sim['return_pct'],
        'max_drawdown_percent': sim['max_dd_pct'],
        'ladder': sim['ladder'],
        'period': {'start': start.isoformat(), 'end': end.isoformat()},
        'best_config_used': str(best_cfg_path),
        'selected_symbols': list(selected.keys()),
    }
    out_json = REPORTS / 'live_sim_vb_ladder_summary.json'
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Ladder-optimized live-sim: ${sim['start']:.2f} -> ${sim['end']:.2f} | Return {sim['return_pct']:.1f}% | MaxDD {sim['max_dd_pct']:.1f}%")
    print(f"Ladder args: {sim['ladder']}")
    print(f"Reports written: {trades_csv.name}, {out_json.name}")


if __name__ == '__main__':
    main()
