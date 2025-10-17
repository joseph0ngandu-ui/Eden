from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
# Ensure 'eden' package root is importable when run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eden.risk.volatility_adapter import compute_volatility_factor, VolatilityConfig
from eden.features.htf_ict_bias import build_htf_context


def run_tests() -> dict:
    results = {}
    # Vol adapter
    vf = compute_volatility_factor({'atr_14': 10.0, '1H_atr_14': 5.0}, VolatilityConfig())
    results['volatility_factor'] = vf
    results['volatility_ok'] = 1.9 < vf < 2.1
    # HTF context
    import pandas as pd
    idx = pd.date_range('2024-01-01', periods=5, freq='H', tz='UTC')
    df = pd.DataFrame({'open':[1,2,3,4,5],'high':[2,3,4,5,6],'low':[0,1,2,3,4],'close':[1.5,2.5,3.5,4.5,5.5],'volume':[1]*5}, index=idx)
    ctx = build_htf_context(df, df)
    results['htf_ctx_cols'] = list(ctx.columns)
    results['htf_ctx_ok'] = all(c in ctx.columns for c in ['HTF_FVG_BULL','HTF_FVG_BEAR'])
    results['status'] = 'ok' if (results['volatility_ok'] and results['htf_ctx_ok']) else 'fail'
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=str, required=True)
    args = p.parse_args()
    res = run_tests()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()