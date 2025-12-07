# Portfolio Risk Tuning Configuration (2025-12-05)

## Targets Achieved
- Daily DD: < 4.5% (Achieved: 2.47%)
- Max DD: < 9.5% (Achieved: 9.03%)
- Monthly Return: 1.28% (positive)
- Profit Factor: 1.02

## Active Strategies (5)
| Strategy | Allocation | Monthly Return | PF |
|----------|------------|----------------|-----|
| Asian Fade | 30% | +87.99% | 1.31 |
| Mean Reversion | 30% | +40.32% | 1.12 |
| Overlap Scalper | 20% | +15.33% | 1.10 |
| Gold Breakout | 15% | +3.03% | 1.57 |
| Trend Follower | 5% | +3.0% | 1.01 |

## Disabled Strategies (2)
- RSI Momentum: -15.67% monthly, PF 0.97
- Volatility Expansion: -21.89% monthly, PF 0.88

## Risk Parameters
- Base Risk: 0.35% per trade
- Daily DD Hard Stop: 4.0%
- Max DD Circuit Breaker: Progressive scaling at 1%, 3%, 5%, 7%, 8%

## Files Modified
- trading/ml_portfolio_optimizer.py
- trading/pro_strategies.py  
- scripts/comprehensive_backtest.py
