# Eden Trading Bot - Research Log

> **Last Updated:** 2025-12-07 (Final Verified Config)
> **Status:** LIVE on FundedNext
> **Balance:** ~$10,000

---

## 🎯 CURRENT SAFE CONFIGURATION

**Verified via Accurate Backtest** (90 days, uses actual bot code):
- **Base Risk:** 0.6% per trade
- **Monthly Return:** ~6.5-7%
- **Max Drawdown:** ~7-8%
- **Phase 1 Time:** ~5 weeks

### Active Symbols & Strategies

| Symbol | Strategy | Timeframe | 90-Day R |
|:---|:---|:---:|---:|
| USTECm | Index Vol Expansion | M15 | **+27.05R** |
| US500m | Index Vol Expansion | M15 | **+17.75R** |
| EURUSDm | Vol Squeeze + Momentum | M5/D1 | **+18.55R** |
| USDJPYm | Vol Squeeze | M5 | **+7.70R** |
| USDCADm | Momentum | D1 | +1.75R |
| EURJPYm | Momentum | D1 | +0.50R |
| CADJPYm | Momentum | D1 | -0.15R |

**Total:** 73.15R over 90 days @ 1.0% risk = ~24% return

---

## ❌ FAILED/DISABLED STRATEGIES

| Strategy | Symbol | Result | Reason |
|:---|:---|---:|:---|
| Gold Spread Hunter | XAUUSDm | -6.5R | Spread friction |
| Index Vol Expansion | US30m | -7.6R | Divergence from USTEC |
| Vol Squeeze | AUDUSDm | -8.75R | No edge |
| London Breakout | Multiple | Mixed | High DD (10.8R) |

---

## 📊 OPTIMIZATION RESEARCH (2025-12-07)

### Attempted Optimizations

| Approach | Result | Verdict |
|:---|:---|:---|
| Fix Momentum D1 Data | +6R recovered | ✅ Success |
| Add AUDUSD | -8.75R | ❌ Failed |
| Weighted Allocation (1.4x Index) | 15%+ DD | ❌ Too Risky |
| Higher Base Risk (1.0%) | 30%+ DD | ❌ Would Fail |

### Mathematical Limit
At any risk level, DD/Return ratio is ~40%. To stay under 9.5% DD:
- Max safe base risk: ~0.35%
- Expected monthly return: ~7%

---

## 🚀 HOW TO START

Use the `/start-bot` workflow or run:
```powershell
cd c:\Users\opc\Desktop\Eden
powershell -File scripts/startup/restart_bot.ps1
```

---

## 📁 KEY FILES

| Purpose | Path |
|:---|:---|
| Configuration | `config/config.yaml` |
| Strategy Logic | `trading/pro_strategies.py` |
| Bot Runner | `watchdog.py` |
| Startup Script | `scripts/startup/restart_bot.ps1` |
| Accurate Backtest | `scripts/research/accurate_backtest.py` |
