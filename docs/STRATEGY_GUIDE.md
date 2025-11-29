# Eden Trading System - Strategy Guide & Operations Manual

## 🏆 System Overview
**Target**: 12% Monthly Return
**Max Drawdown**: <8% (Hard Limit)
**Risk Model**: "Prop Firm Certified" (0.15% Risk per Trade)

This system is a high-frequency, multi-strategy portfolio designed to pass prop firm challenges and manage funded accounts. It trades 4 uncorrelated strategies simultaneously to smooth out the equity curve.

---

## 🛡️ Risk Management (CRITICAL)

### 1. The "Golden Rule"
- **Risk Per Trade**: `0.15%` (Fixed)
- **Max Open Trades**: `5` (Hard Cap)
- **Max Instant Exposure**: `0.75%` (5 * 0.15%)

### 2. Daily Circuit Breaker
- **Daily Loss Limit**: `2.0%`
- **Action**: If equity drops 2% in a single day, the bot **STOPS TRADING** immediately until the next server day (00:00).
- **Purpose**: Prevents emotional tilt and catastrophic "black swan" days.

### 3. Drawdown Protection
- The 0.15% risk size is mathematically calculated to keep Max Drawdown around **7%** in worst-case scenarios (verified by backtest).
- **Do NOT increase risk** unless you have a buffer of +10% profit.

---

## 📊 Strategy Portfolio

### 1. London/NY Overlap Scalper
- **Pairs**: EURUSD, GBPUSD
- **Time**: 12:00 - 16:00 GMT (Peak Liquidity)
- **Logic**: Captures momentum bursts when London and New York sessions overlap.
- **Entry**: Price momentum > ATR threshold + Volume spike.
- **Exit**: Fixed Reward:Risk (2.5:1).

### 2. Asian Range Fade
- **Pairs**: USDJPY, AUDJPY
- **Time**: 22:00 - 06:00 GMT (Asian Session)
- **Logic**: Exploits mean reversion during low-volatility hours.
- **Entry**: Fades price when it deviates >80% or <20% of the session range.
- **Exit**: Reversion to mean (50% of range).

### 3. Gold London Breakout
- **Pairs**: XAUUSD
- **Time**: 07:00 - 09:00 GMT (London Open)
- **Logic**: Trades the explosive volatility of Gold when London opens.
- **Entry**: Breakout of the Asian Session High/Low.
- **Exit**: Trend following (3:1 Reward).

### 4. Volatility Expansion
- **Pairs**: All
- **Time**: 24/7
- **Logic**: Identifies "Squeezes" (low volatility) and enters on expansion.
- **Entry**: ATR expansion > 1.5x average.
- **Exit**: ATR-based trailing stop.

---

## ⚙️ Configuration Guide

### `config/config.yaml`
Key parameters to monitor:

```yaml
risk_per_trade_percent: 0.15  # DO NOT CHANGE
max_positions: 5              # DO NOT CHANGE
max_daily_loss_percent: 2.0   # Circuit Breaker
```

### `config/strategy_configs.json`
Contains granular settings for each strategy. Only advanced users should modify this.

---

## 🚀 Operations & Troubleshooting

### Starting the Bot
Run the startup script:
```powershell
./restart_all.ps1
```

### Monitoring
Check the logs in `logs/eden_vix100_bot.log` (name may vary).
- **Healthy**: "Polling every 60s..."
- **Trade**: "LIVE TRADE: LONG EURUSD..."
- **Skipped**: "SKIPPING TRADE: Max positions reached"

### Common Issues

**1. "SKIPPING TRADE: Daily loss limit reached"**
- **Cause**: You lost 2% today.
- **Fix**: Do nothing. The bot is protecting you. It will resume tomorrow.

**2. "Order failed: Invalid Volume"**
- **Cause**: Account balance too small for 0.01 lots at 0.15% risk.
- **Fix**: Add funds (Min ~$3,500) or switch to a Cent account.

**3. No Trades for Hours**
- **Cause**: Normal. Strategies wait for specific setups.
- **Fix**: Be patient. Over-trading kills accounts.

---

## ⚠️ Disclaimer
Trading involves risk. This bot is a tool, not a guarantee. Past performance (45% in backtest) does not guarantee future results. Always monitor the bot.
