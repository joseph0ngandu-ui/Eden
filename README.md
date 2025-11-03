# Eden Trading System

**Production-grade automated trading system using the MA(3,10) strategy**

Eden is a professional trading system implementing a proven MA(3,10) crossover strategy on M5 timeframe. Rigorously backtested on 10 trading symbols with $1.32M validated PnL.

## Quick Start

### Backtest
```bash
python backtest.py                          # Run default backtest
python backtest.py --symbols "VIX75" "VIX100"  # Specific symbols
```

### Live Trading
```bash
python trade.py                             # Start live trading
python trade.py --interval 300              # Custom check interval
```

## Performance

| Metric | Value |
|--------|-------|
| **Total Trades** | 13,820 |
| **Total PnL** | $1,323,131.69 |
| **Win Rate** | 49.8% |
| **Return** | 1,323.13% |
| **Profit Factor** | 1.09 |

## Strategy

- **Entry**: MA(3) crosses above MA(10) on M5
- **Exit**: Fixed 5-bar hold (25 minutes)
- **Timeframe**: M5 candles
- **Symbols**: 10 indices + XAUUSD

## Installation

```bash
pip install pandas numpy MetaTrader5
```

Ensure MetaTrader 5 terminal is running with your account logged in.

## Documentation

- **Strategy Details**: See `config/strategy.yml`
- **API Reference**: See `src/backtest_engine.py` and `src/trading_bot.py`
- **Configuration**: Edit `config/strategy.yml` to customize parameters

## Files

```
src/
  ├── backtest_engine.py    # Core backtest engine
  └── trading_bot.py        # Live trading bot
config/
  └── strategy.yml          # Strategy configuration
backtest.py                 # Backtest runner
trade.py                    # Live trading runner
```

⚠️ **Disclaimer**: Trading involves significant risk. Past performance does not guarantee future results. Always test on demo before live trading
