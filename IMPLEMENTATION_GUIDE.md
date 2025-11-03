# Eden Trading System - Implementation Guide

## New Features Implemented

### 1. Trade Journal Exporter ✓
**File**: `src/trade_journal.py`

Exports each trade to `logs/trade_history.csv` for analytics.

**Features**:
- Automatic CSV export of all closed trades
- Trade metadata: timestamp, symbol, type, entry/exit prices, PnL, commission, slippage
- Trade statistics calculation (win rate, profit factor, avg win/loss)
- Persistent trade history loading on startup

**Usage**:
```python
from src.trade_journal import TradeJournal

journal = TradeJournal(log_dir="logs")

# Add a trade
journal.add_trade(
    symbol="Volatility 75 Index",
    trade_type="BUY",
    entry_price=100.50,
    entry_time=datetime.now(),
    exit_price=101.20,
    exit_time=datetime.now(),
    volume=1.0
)

# Export to CSV
journal.export_csv()

# Get statistics
stats = journal.get_statistics()
journal.print_summary()
```

---

### 2. Configurable Risk Cap ✓
**File**: `src/health_monitor.py` + Updated `config/strategy.yml`

Auto-disables live trading when max drawdown is breached.

**Configuration** (in `strategy.yml`):
```yaml
risk_management:
  position_size: 1.0
  max_concurrent_positions: 10
  max_drawdown_percent: 10.0  # Auto-disable at 10% drawdown
  max_daily_loss_percent: 5.0  # Daily loss limit
```

**Features**:
- Tracks peak balance and current drawdown percentage
- Real-time risk level monitoring (SAFE, WARNING, CRITICAL)
- Auto-disables trading at max drawdown
- Callback system for risk state changes
- Health status reporting

**Usage in TradingBot**:
```python
# Automatically integrated in TradingBot.__init__()
bot = TradingBot(symbols=["Volatility 75 Index"])

# Monitor via callback
def on_health_change(health_status, risk_level):
    if risk_level == RiskLevel.CRITICAL:
        print("Critical risk - trading disabled")

monitor = HealthMonitor(max_drawdown_percent=10.0, 
                       health_check_callback=on_health_change)
```

---

### 3. Live Health Monitor ✓
**File**: `src/health_monitor.py`

Lightweight watchdog for MT5 API and internet connectivity.

**Features**:
- MT5 API connectivity check
- Internet connectivity check (via Google DNS)
- Automatic disconnection/reconnection tracking
- Health status: HEALTHY, DEGRADED, UNHEALTHY
- Graceful pause/resume on disconnection
- Statistics tracking (failure counts, last check time)

**Usage**:
```python
from src.health_monitor import HealthMonitor, HealthStatus

monitor = HealthMonitor(
    max_drawdown_percent=10.0,
    check_interval=60  # Check every 60 seconds
)

# Run health check
is_healthy = monitor.check_health()

# Get current status
status = monitor.get_status()
monitor.print_status()

# Update balance (for drawdown tracking)
monitor.update_balance(current_balance=100000.0)
```

---

### 4. Parameter Loader from YAML ✓
**File**: `src/config_loader.py`

Dynamically loads all strategy parameters from `config/strategy.yml`.

**Features**:
- Load strategy parameters (MA periods, hold duration, timeframe)
- Load trading symbols list
- Load risk management config
- Load logging config
- Support for dot notation in config access
- Safe defaults fallback

**Usage**:
```python
from src.config_loader import ConfigLoader

config = ConfigLoader(config_path="config/strategy.yml")

# Get strategy parameters
params = config.get_strategy_params()
# Returns: {'fast_ma_period': 3, 'slow_ma_period': 10, 'hold_bars': 5, 'timeframe': 'M5'}

# Get trading symbols
symbols = config.get_trading_symbols()

# Get risk management settings
risk = config.get_risk_management()

# Get individual parameters
fast_ma = config.get_parameter('parameters.fast_ma_period', default=3)
```

**TradingBot Integration**:
The TradingBot now loads all parameters from config automatically:
```python
bot = TradingBot(symbols=[], config_path="config/strategy.yml")
# MA periods, hold bars, and risk settings all loaded from YAML
```

---

### 5. Version Tag in Logs ✓
**Files**: Updated `src/trading_bot.py`, `src/config_loader.py`

Auto-appends system version and strategy parameters to log header.

**Format**:
```
================================================================================
Eden v1.0.0 - MA(3,10) Strategy | M5 Timeframe
HOLD=5 bars | Symbols=10
Risk Cap: 10.0% | Max Positions: 10
================================================================================
```

**Implementation**:
- Reads version from `config/strategy.yml` 
- Automatically logs at bot startup
- Includes all critical parameters for reference

**Log File**:
Located at `logs/trading.log` with structured output.

---

### 6. Backtest-Mode Flag ✓
**File**: Updated `backtest.py`

Added `--backtest` flag for CLI-based backtesting with dynamic parameters.

**Usage**:
```bash
# Standard backtest
python backtest.py

# With backtest flag and custom date range
python backtest.py --backtest --start 2025-11-01 --end 2025-11-30

# Filter specific symbols
python backtest.py --backtest --symbols "Volatility 75 Index" "Volatility 50 Index"

# With custom config
python backtest.py --backtest --config config/strategy.yml
```

**CLI Features**:
- `--backtest`: Enable backtest mode
- `--start YYYY-MM-DD`: Start date (default: 2025-08-01)
- `--end YYYY-MM-DD`: End date (default: 2025-10-31)
- `--symbols SYMBOL ...`: Filter symbols
- `--config PATH`: Load custom strategy config
- `--verbose`: Enable verbose output

---

### 7. Volatility-Aware MA Crossover ✓
**File**: `src/volatility_adapter.py`

Adapts hold duration and MA parameters based on market volatility (ATR, std dev).

**Features**:
- ATR (Average True Range) calculation
- Standard deviation analysis
- Volatility percentile classification (LOW, MEDIUM, HIGH)
- Adaptive hold duration based on volatility:
  - HIGH volatility: longer holds (1.5x base)
  - LOW volatility: shorter holds (0.7x base)
  - MEDIUM: base duration
- Adaptive MA parameters:
  - HIGH: MA(5,13) slower, less whipsaws
  - LOW: MA(2,8) faster, catch moves quicker
  - MEDIUM: MA(3,10) base
- Adaptive stop loss based on ATR

**Usage**:
```python
from src.volatility_adapter import VolatilityAdapter

adapter = VolatilityAdapter(base_hold_bars=5, atr_period=14)

# Calculate volatility metrics
atr = adapter.calculate_atr(df)
std_dev = adapter.calculate_std_dev(df)
vol_metrics = adapter.classify_volatility(atr.iloc[-1], std_dev.iloc[-1], df)

# Get adaptive parameters
adaptive_hold = adapter.get_adaptive_hold_duration(vol_metrics)
adaptive_ma_fast, adaptive_ma_slow = adapter.get_adaptive_ma_params(vol_metrics)
adaptive_sl = adapter.get_adaptive_stop_loss(vol_metrics, entry_price=100.0)

print(f"Volatility Level: {vol_metrics.volatility_level}")
print(f"Adaptive Hold: {adaptive_hold} bars")
print(f"Adaptive MA: MA({adaptive_ma_fast},{adaptive_ma_slow})")
```

**Integration in TradingBot**:
Automatically used in `run_cycle()` for real-time volatility adaptation.

---

### 8. Daily Parameter Optimizer ✓
**File**: `src/volatility_adapter.py` - `ParameterOptimizer` class

Lightweight overnight optimizer for finding best MA periods and hold durations.

**Features**:
- Analyzes trade history to find optimal parameters
- Groups analysis by symbol and date ranges
- Evaluates hold duration performance
- Evaluates MA parameter combinations
- Aggregates results across all symbols
- Generates optimization reports

**Usage**:
```python
from src.volatility_adapter import ParameterOptimizer
from src.trade_journal import TradeJournal

# Load trade history
journal = TradeJournal()
trades = journal.trades

# Run optimization
optimizer = ParameterOptimizer(min_trades=10)

# Full optimization
best_params = optimizer.optimize_parameters(
    trade_history=trades,
    date_range=(datetime(2025, 11, 1), datetime(2025, 11, 30))
)

# Check results
print(best_params)
# {'optimized_hold_bars': 5, 'optimized_ma_params': 'MA(3,10)', ...}

# Get report
report = optimizer.get_optimization_report()
print(report)
```

**Overnight Optimization Strategy**:
- Run after market close
- Analyze last 24 hours of trades
- Update strategy.yml with best parameters
- Apply at next trading session

---

## File Structure

```
Eden/
├── src/
│   ├── __init__.py
│   ├── trading_bot.py              (UPDATED - integrated all features)
│   ├── backtest_engine.py
│   ├── config_loader.py            (NEW - YAML config loading)
│   ├── trade_journal.py            (NEW - CSV trade export)
│   ├── health_monitor.py           (NEW - health checks & risk management)
│   ├── volatility_adapter.py       (NEW - volatility-aware MA & optimizer)
│   └── backtest_engine.py
├── config/
│   └── strategy.yml               (UPDATED - added risk parameters)
├── logs/
│   ├── trading.log               (auto-created with version tag)
│   └── trade_history.csv         (auto-created trade journal)
├── backtest.py                   (UPDATED - added --backtest flag)
└── trade.py
```

---

## Configuration Reference

### strategy.yml Structure

```yaml
strategy:
  name: "MA Crossover Strategy"
  version: "1.0.0"
  status: "PRODUCTION"

parameters:
  fast_ma_period: 3
  slow_ma_period: 10
  timeframe: "M5"
  hold_bars: 5

trading_symbols:
  - "Volatility 75 Index"
  - "Volatility 100 Index"
  # ... etc

risk_management:
  position_size: 1.0
  max_concurrent_positions: 10
  max_drawdown_percent: 10.0        # NEW - Risk cap
  max_daily_loss_percent: 5.0       # NEW - Daily loss limit
  stop_loss_pips: null
  take_profit_pips: null

live_trading:
  enabled: true
  check_interval: 300               # 5 minutes

logging:
  level: "INFO"
  format: "%(asctime)s - %(levelname)s - %(message)s"
  file: "logs/trading.log"
```

---

## Running the System

### 1. Live Trading with All Features

```bash
python trade.py --interval 300
```

This will:
- ✓ Load config from `config/strategy.yml`
- ✓ Initialize trade journal (logs/trade_history.csv)
- ✓ Start health monitor (MT5 API + internet checks)
- ✓ Apply volatility adaptation
- ✓ Auto-disable if max drawdown is breached
- ✓ Export trades on shutdown

### 2. Backtest with Dynamic Parameters

```bash
python backtest.py --backtest --start 2025-11-01 --end 2025-11-30
```

### 3. Optimize Parameters Overnight

```python
from src.trade_journal import TradeJournal
from src.volatility_adapter import ParameterOptimizer

journal = TradeJournal()
optimizer = ParameterOptimizer()
best_params = optimizer.optimize_parameters(journal.trades)

# Update strategy.yml with best_params
```

---

## Monitoring Dashboard

### Trade Journal Metrics
- Total trades, closed/open positions
- Win rate, profit factor
- Average win/loss
- Max profit/loss

### Health Monitor Status
- Overall health: HEALTHY/DEGRADED/UNHEALTHY
- Risk level: SAFE/WARNING/CRITICAL
- Current balance vs peak balance
- Drawdown percentage
- MT5 API status
- Internet connectivity
- Disconnection duration

### Volatility Metrics
- ATR (Average True Range)
- Standard deviation
- Volatility percentile
- Volatility classification
- Adaptive parameters being used

---

## Troubleshooting

### Trading disabled due to high drawdown
- Check `logs/trading.log` for warning messages
- Review `logs/trade_history.csv` for losing trades
- Run optimizer to find better parameters
- Reset max_drawdown_percent in strategy.yml if acceptable

### MT5 connectivity issues
- Health monitor automatically detects and reports
- Trading pauses on UNHEALTHY status
- Resumes when connection restored
- Check network connectivity

### Trade journal not exporting
- Ensure `logs/` directory exists (auto-created)
- Check file permissions
- Verify trade.py exits gracefully (Ctrl+C)

---

## Next Steps

1. **Test Backtest**: `python backtest.py --backtest --start 2025-11-01 --end 2025-11-10`
2. **Run Optimizer**: Analyze trades to find optimal MA/hold parameters
3. **Monitor Live**: Run trading bot and check `logs/trade_history.csv` and health status
4. **Adjust Config**: Update `config/strategy.yml` with optimized parameters
5. **Review Metrics**: Use `trade_journal.get_statistics()` for daily reviews

---

## Dependencies

All core dependencies are already installed:
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `PyYAML` - YAML config loading
- `MetaTrader5` - MT5 API

Ensure `config/strategy.yml` exists before running.

---

Generated: 2025-11-03
Eden v1.0.0