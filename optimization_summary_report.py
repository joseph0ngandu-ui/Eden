#!/usr/bin/env python3
import json
from pathlib import Path

config = json.load(open('results/backtest/optimal_engine_config.json'))
opt_results = json.load(open('results/backtest/parameter_optimization_50iter.json'))

print('\n' + '='*100)
print('50-ITERATION PARAMETER OPTIMIZATION - FINAL REPORT')
print('='*100)
print(f'\n✅ BEST CONFIGURATION FOUND:')
print(f'  Strategy Name: {config["name"]}')
print(f'  Optimization Iteration: {config["optimization_iteration"]}')
print(f'  Score (Return/Drawdown): {config["optimization_score"]:.4f}')
print(f'\n📊 STRATEGY PARAMETERS:')
print(f'  Bollinger Bands Period: {config["strategy_parameters"]["period"]}')
print(f'  Standard Deviation: {config["strategy_parameters"]["std_dev"]}')
print(f'  RSI Threshold: {config["strategy_parameters"]["rsi_threshold"]}')
print(f'\n💰 PERFORMANCE METRICS:')
print(f'  Average Return: {config["performance_metrics"]["avg_return_pct"]:+.2f}%')
print(f'  Average Drawdown: {config["performance_metrics"]["avg_drawdown_pct"]:.2f}%')
print(f'  Average Win Rate: {config["performance_metrics"]["avg_win_rate_pct"]:.2f}%')
print(f'\n🎯 TOP PERFORMERS:')
instruments = sorted(config['instrument_results'].items(), key=lambda x: x[1]['return_pct'], reverse=True)
for i, (symbol, stats) in enumerate(instruments[:3], 1):
    print(f'  {i}. {symbol}: {stats["return_pct"]:+.2f}% return | {stats["trades"]} trades | {stats["wr"]:.1f}% WR')
print(f'\n📁 SAVED FILES:')
print(f'  ✓ results/backtest/optimal_engine_config.json (DEFAULT ENGINE)')
print(f'  ✓ results/backtest/parameter_optimization_50iter.json (FULL RESULTS)')
print('='*100 + '\n')
