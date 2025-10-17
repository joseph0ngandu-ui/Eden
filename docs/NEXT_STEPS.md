# Next steps for Phase-3

This branch contains Phase-2/3 groundwork and scaffolding for the staged ML pipeline and debugging tools.

What’s done
- Phase-2: volatility normalization, confidence-weighted risk, HTF strict gating, decision logging, tests.
- Phase-3 scaffolding: staged ML (A/B/C), regime detector, meta-learning stub, PPO fallback, trainer stub, debug import-check + unit tests.
- Engine integration hooks and run_mvp orchestration for --phase3.

How to continue (quick commands)
- Install deps: (from repo root)
  - pip install -r worker/python/requirements.txt
- Sanity checks:
  - python worker/python/eden/debug/import_checker.py --output results/phase3/import_status.json
  - python worker/python/eden/debug/unit_tests.py --output results/phase3/unit_tests_report.json
- Phase-3 run (cached data; MT5 optional):
  - $env:PYTHONPATH="$(Resolve-Path worker/python)"; python scripts/run_mvp.py --phase3 --output-dir results/phase3/ --start-date 2025-10-07 --end-date 2025-10-14
- Compare with baseline:
  - python scripts/postprocess_metrics.py --baseline results/phase3/baseline/ --compare results/phase3/phase3_run/ --out results/phase3/comparison_report.json
- Optional grid (compact):
  - python scripts/grid_optimize.py --config results/phase3/run_config.json --output results/phase3/best_config_phase3.json

Targets for fine-tuning
- Resolve pandas chained-assignment warnings in eden/features/htf_ict_bias.py (use .loc and avoid inplace iloc writes).
- Tune risk multipliers and confidence thresholds in BacktestEngine (conf_cut_* and stageC risk mapping).
- Improve Stage B sequence features (add short rolling momentum, volatility bands) and calibrate Stage C combiner.
- Expand decision_log.csv to audit more fields if needed; iterate meta updates via meta_learning_controller.update_meta.
- If compute allows, integrate real PPO training in eden/rl/ppo_controller.py.

Artifacts to inspect
- results/phase2/* (baseline vs phase-2) and results/phase3/* (baseline vs phase-3), including comparison_report.json and equity_comparison.png.

Branch
- phase3-ml-pipeline
