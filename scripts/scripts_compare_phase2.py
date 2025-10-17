from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def _load_metrics(path: Path) -> dict:
    f = path / 'metrics.json'
    if not f.exists():
        return {}
    try:
        return json.loads(f.read_text())
    except Exception:
        return {}


def _primary(metrics: dict) -> dict:
    # Prefer ensemble_M5 then ensemble_M1 else any
    for k in ['ensemble_M5', 'ensemble_M1']:
        if k in metrics:
            return metrics[k]
    return next(iter(metrics.values())) if metrics else {}


def _load_equity_from_trades(trades_csv: Path) -> pd.Series:
    if not trades_csv.exists():
        return pd.Series(dtype=float)
    t = pd.read_csv(trades_csv)
    if t.empty:
        return pd.Series(dtype=float)
    start_eq = float(t.get('starting_cash').iloc[0]) if 'starting_cash' in t.columns else 10000.0
    eq = t['pnl'].cumsum() + start_eq
    eq.index = pd.RangeIndex(start=0, stop=len(eq))
    return eq


def compare_runs(baseline_dir: Path, phase2_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    base_metrics = _load_metrics(baseline_dir)
    p2_metrics = _load_metrics(phase2_dir)
    base = _primary(base_metrics) or {}
    p2 = _primary(p2_metrics) or {}

    # Load trades for risk analysis
    def _find_trades(d: Path) -> Path:
        cands = [d / 'backtests_M5' / 'ensemble' / 'trades.csv', d / 'backtests_M1' / 'ensemble' / 'trades.csv']
        for c in cands:
            if c.exists():
                return c
        return cands[0]
    t_base = _find_trades(baseline_dir)
    t_p2 = _find_trades(phase2_dir)
    df_base = pd.read_csv(t_base) if t_base.exists() else pd.DataFrame()
    df_p2 = pd.read_csv(t_p2) if t_p2.exists() else pd.DataFrame()
    avg_risk_base = float(df_base.get('final_risk_usd', pd.Series(dtype=float)).dropna().mean()) if not df_base.empty else 0.0
    avg_risk_p2 = float(df_p2.get('final_risk_usd', pd.Series(dtype=float)).dropna().mean()) if not df_p2.empty else 0.0

    # Strict blocks from decision log (phase2 only)
    logs = phase2_dir.parent / 'logs' / 'decision_log.csv'
    pct_blocked = 0.0
    blocked_count = 0
    total_actions = 0
    if logs.exists():
        dlog = pd.read_csv(logs)
        blocked_count = int((dlog['reason'] == 'HTF_STRICT_BLOCK').sum()) if 'reason' in dlog.columns else 0
        total_actions = len(dlog)
        if total_actions > 0:
            pct_blocked = float(blocked_count / total_actions * 100.0)

    # Compute deltas
    report = {
        'baseline': base,
        'phase2': p2,
        'delta': {
            'net_pnl': float(p2.get('net_pnl', 0.0) - base.get('net_pnl', 0.0)),
            'sharpe': float(p2.get('sharpe', 0.0) - base.get('sharpe', 0.0)),
            'max_drawdown_pct': float(p2.get('max_drawdown_pct', 0.0) - base.get('max_drawdown_pct', 0.0)),
            'equity_growth_pct': float(p2.get('equity_growth_pct', 0.0) - base.get('equity_growth_pct', 0.0)),
            'trades': int(p2.get('trades', 0) - base.get('trades', 0)),
            'avg_final_risk_usd_before': avg_risk_base,
            'avg_final_risk_usd_after': avg_risk_p2,
            'htf_strict_block_pct': pct_blocked,
            'htf_strict_block_count': blocked_count,
        }
    }
    (out_dir / 'comparison_report.json').write_text(json.dumps(report, indent=2))

    # Summary txt
    lines = [
        f"Net PnL delta: {report['delta']['net_pnl']:.2f}",
        f"Sharpe delta: {report['delta']['sharpe']:.3f}",
        f"Max Drawdown delta (pct): {report['delta']['max_drawdown_pct']:.2f}",
        f"Equity Growth delta (pct): {report['delta']['equity_growth_pct']:.2f}",
        f"Trades count delta: {report['delta']['trades']}",
        f"Avg risk (before -> after): {avg_risk_base:.4f} -> {avg_risk_p2:.4f}",
        f"HTF_STRICT blocks: {blocked_count} ({pct_blocked:.2f}%)",
    ]
    (out_dir / 'comparison_summary.txt').write_text("\n".join(lines))

    # Equity overlay
    base_trades = baseline_dir / 'backtests_M5' / 'ensemble' / 'trades.csv'
    if not base_trades.exists():
        base_trades = baseline_dir / 'backtests_M1' / 'ensemble' / 'trades.csv'
    p2_trades = phase2_dir / 'backtests_M5' / 'ensemble' / 'trades.csv'
    if not p2_trades.exists():
        p2_trades = phase2_dir / 'backtests_M1' / 'ensemble' / 'trades.csv'

    eq_b = _load_equity_from_trades(base_trades)
    eq_p = _load_equity_from_trades(p2_trades)
    if not eq_b.empty and not eq_p.empty:
        plt.figure(figsize=(10,5))
        plt.plot(eq_b.values, label='Baseline')
        plt.plot(eq_p.values, label='Phase-2')
        plt.title('Equity Curve Comparison (Baseline vs Phase-2)')
        plt.xlabel('Trade Index')
        plt.ylabel('Equity')
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / 'equity_comparison.png')
        plt.close()
