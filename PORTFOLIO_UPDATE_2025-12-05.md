# Portfolio Update - 2025-12-05

## New Index Strategies Added

Added 3 US index strategies for diversification:
- **US30m** (Dow Jones 30)
- **US500m** (S&P 500)  
- **USTECm** (Nasdaq 100)

### Index Momentum (Active)
| Metric | Value | Target |
|--------|-------|--------|
| Monthly Return | 12.11% | ≥13% |
| Max DD | 12.42% | <9.5% |
| Daily DD | 7.46% | <4.5% |
| Profit Factor | 1.19 | >1.0 |
| Win Rate | 33.9% | - |
| Trades (3mo) | 564 | - |

### Index NY Breakout (Disabled)
Lower returns (5.76%/mo) but better risk metrics.

### Index Mean Reversion (Disabled)
Excessive drawdown (24.5% max DD).

## Active Strategy Portfolio

| Strategy | Allocation | Symbols |
|----------|------------|---------|
| Asian Fade | 60% | USDJPYm, AUDJPYm |
| Index Momentum | 40% | US30m, US500m, USTECm |

## Risk Parameters
- Base Risk: 0.35% per trade
- Daily DD Limit: 4.0%
- Max DD Limit: 9.5%

## Files
- `trading/pro_strategies.py` - Strategy engine with index methods
- `scripts/index_backtest.py` - Dedicated index backtester
- `reports/index_strategy_results.json` - Test results
