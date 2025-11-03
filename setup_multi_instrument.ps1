# Multi-Instrument Trading Pipeline Setup
# Configures VIX variants + XAUUSD intraday strategies

# Environment Variables
$INSTRUMENT_POOL = 'VIX75,VIX100,VIX50,VIX25,StepIndex,Boom1000,Crash1000,Boom500,Crash500,XAUUSD'
$DATA_PROVIDER = 'your_data_provider'
$DATA_DIR = 'data'
$CONFIG_DIR = 'configs/instruments'
$BACKTEST_DIR = 'backtests'
$REPORT_DIR = 'reports'
$PARALLEL_WORKERS = 8
$OPTUNA_TRIALS_INITIAL = 80
$OPTUNA_TRIALS_FULL = 300
$SEED = 42
$GIT_TAG_PREFIX = 'eden/multi-instrument-'

Write-Host "=== Multi-Instrument Pipeline Setup ===" -ForegroundColor Cyan

# Step 1: Create directory structure
Write-Host "`n[1/7] Creating directory structure..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path $CONFIG_DIR | Out-Null
New-Item -ItemType Directory -Force -Path $DATA_DIR | Out-Null
New-Item -ItemType Directory -Force -Path $BACKTEST_DIR | Out-Null
New-Item -ItemType Directory -Force -Path $REPORT_DIR | Out-Null

# Step 2: Create instruments pool JSON
Write-Host "[2/7] Creating instruments pool configuration..." -ForegroundColor Yellow
$instruments = $INSTRUMENT_POOL -split ','
$poolConfig = @{
    instruments = $instruments
    data_provider = $DATA_PROVIDER
    parallel_workers = $PARALLEL_WORKERS
    seed = $SEED
} | ConvertTo-Json -Depth 10
$poolConfig | Out-File -FilePath "$CONFIG_DIR/instruments_pool.json" -Encoding utf8
Write-Host "  Created: $CONFIG_DIR/instruments_pool.json" -ForegroundColor Green

# Step 3: Generate per-instrument configs
Write-Host "[3/7] Generating per-instrument configurations..." -ForegroundColor Yellow
python -c @"
import json
import os

instruments = '$INSTRUMENT_POOL'.split(',')
cfg_dir = '$CONFIG_DIR'

# VIX template
vix_template = {
    'data': {'resolution': '1m', 'source': '$DATA_PROVIDER'},
    'risk': {
        'risk_per_trade_pct': 0.5,
        'max_daily_risk_pct': 2.0,
        'max_concurrent_trades': 2
    },
    'entry': {
        'method': 'ICT_price_action',
        'killzone': ['07:00', '10:00'],
        'htf_bias_windows': ['1H', '4H']
    },
    'exit': {
        'take_profit': {'type': 'scaled', 'targets': [0.5, 1.0, 1.5]},
        'stop_loss': {'type': 'atr', 'multiplier': 3}
    },
    'optimization': {
        'search_space': {
            'risk_per_trade_pct': [0.1, 1.5],
            'atr_multiplier': [2, 6]
        },
        'objective': 'sharpe_drawdown_penalized',
        'trials_initial': $OPTUNA_TRIALS_INITIAL,
        'trials_full': $OPTUNA_TRIALS_FULL
    },
    'meta': {'seed': $SEED}
}

# Gold strategies
gold_strategies = {
    'breakout_volume': {
        'method': 'BreakoutVolume',
        'timeframe': '15m',
        'volume_confirmation': True,
        'stop_loss_at_swing': True,
        'take_profit_risk_ratio': 2.0
    },
    'false_breakout': {
        'method': 'FalseBreakoutMR',
        'timeframe': '5m',
        'stop_loss': 1.5,
        'take_profit': 1.0,
        'exit_end_of_day': True
    },
    'ema_trend': {
        'method': 'EMA_Crossover',
        'fast_period': 20,
        'slow_period': 50,
        'trend_filter': 'RSI14',
        'stop_loss_at_ATR': True,
        'risk_per_trade_pct': 1.0
    },
    'bollinger_rsi': {
        'method': 'BollingerRSI',
        'BB_period': 20,
        'BB_std': 2,
        'RSI_period': 14,
        'divergence_filter': True,
        'stop_loss_at_band': True
    }
}

for symbol in instruments:
    config_path = os.path.join(cfg_dir, f'{symbol}.json')
    
    if symbol == 'XAUUSD':
        cfg = {
            'symbol': 'XAUUSD',
            'strategies': gold_strategies,
            'data': {'resolution': '5m', 'source': '$DATA_PROVIDER'},
            'risk': {'max_daily_risk_pct': 2.0},
            'meta': {'seed': $SEED}
        }
    else:
        cfg = vix_template.copy()
        cfg['symbol'] = symbol
    
    with open(config_path, 'w') as f:
        json.dump(cfg, f, indent=2)
    print(f'  Created config: {symbol}.json')
"@

Write-Host "[4/7] Setup complete! Next steps:" -ForegroundColor Yellow
Write-Host "  - Run data validation: python scripts/data_validation.py" -ForegroundColor Gray
Write-Host "  - Run optimization: python scripts/multi_instrument_optimizer.py" -ForegroundColor Gray
Write-Host "  - Generate reports: python scripts/generate_reports.py" -ForegroundColor Gray

Write-Host "`n=== Configuration Summary ===" -ForegroundColor Cyan
Write-Host "  Instruments: $($instruments.Count)" -ForegroundColor White
Write-Host "  Config directory: $CONFIG_DIR" -ForegroundColor White
Write-Host "  Parallel workers: $PARALLEL_WORKERS" -ForegroundColor White
Write-Host "  Optuna trials: $OPTUNA_TRIALS_INITIAL (initial), $OPTUNA_TRIALS_FULL (full)" -ForegroundColor White
