#!/usr/bin/env python3
"""
Run Eden Complete MT5 optimization in background with overrides.
- Symbols: EURUSDm, GBPUSDm, XAUUSDm, US30m, USTECm
- Date range: last 90 days
- Iterations: 120
Saves results to a timestamped JSON and writes a last_complete_results.json pointer.
"""
import os
import json
import sys
from datetime import datetime, timedelta
from dataclasses import asdict

from eden_complete_mt5_system import EdenCompleteMT5System

def main():
    now = datetime.now()
    ts = now.strftime('%Y%m%d_%H%M%S')
    symbols = ['EURUSDm', 'GBPUSDm', 'XAUUSDm', 'US30m', 'USTECm']
    end_date = now
    start_date = end_date - timedelta(days=90)

    pointer_path = os.path.abspath('last_complete_results.json')

    try:
        system = EdenCompleteMT5System()
        results = system.run_full_optimization(
            max_iterations=120,
            override_symbols=symbols,
            override_start=start_date,
            override_end=end_date,
        )

        if results:
            # Convert monthly_metrics dataclasses to dicts
            json_results = {}
            for k, v in results.items():
                if k == 'monthly_metrics':
                    json_results[k] = {mk: asdict(mv) for mk, mv in v.items()}
                else:
                    json_results[k] = v

            results_file = os.path.abspath(f'eden_complete_mt5_results_{ts}.json')
            with open(results_file, 'w') as f:
                json.dump(json_results, f, indent=2, default=str)

            with open(pointer_path, 'w') as f:
                json.dump({
                    'results_file': results_file,
                    'timestamp': ts,
                    'status': 'success',
                    'symbols': symbols,
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat(),
                    'iterations': 120
                }, f)

            print(f'Results saved to: {results_file}', flush=True)
            sys.exit(0)
        else:
            with open(pointer_path, 'w') as f:
                json.dump({
                    'results_file': None,
                    'timestamp': ts,
                    'status': 'empty',
                    'symbols': symbols,
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat(),
                    'iterations': 120
                }, f)
            print('No results returned.', flush=True)
            sys.exit(2)

    except Exception as e:
        import traceback
        err = str(e)
        with open(pointer_path, 'w') as f:
            json.dump({
                'results_file': None,
                'timestamp': ts,
                'status': 'error',
                'error': err,
                'symbols': symbols,
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'iterations': 120
            }, f)
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
