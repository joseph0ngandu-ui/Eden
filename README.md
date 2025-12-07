# 🌌 Eden Trading Bot

**Status:** 🟢 LIVE on FundedNext
**Objective:** >13% Monthly Return | <4.5% Daily DD | <9.5% Max DD

---

## 🚀 Quick Start

```powershell
# Start the bot
.\scripts\startup\restart_bot.ps1
```

---

## 📊 Active Strategies

| Strategy | TF | Pairs | Risk | Edge |
|:---|:---:|:---|:---:|:---|
| **Index Vol Expansion** | M15 | US30/USTEC/US500 | 0.75% | Squeeze Breakout |
| **Gold Spread Hunter** | M15 | XAUUSD | 0.50% | Low-Spread Momentum |
| **Forex Vol Squeeze** | M5 | EUR/JPY pairs | 0.25% | Defensive |
| **Momentum Continuation** | D1 | USDCAD/EURUSD/EURJPY/CADJPY | 0.50% | Trend Follow |

---

## 📁 Folder Structure

```
Eden/
├── config/              # Configuration (config.yaml)
├── docs/                # Documentation
│   └── RESEARCH_LOG.md  # Complete research history
├── logs/                # Runtime logs
├── scripts/
│   ├── research/        # Strategy research scripts
│   ├── startup/         # Startup scripts (restart_bot.ps1)
│   └── utilities/       # One-off utilities
├── trading/             # Core trading logic
│   ├── pro_strategies.py    # Strategy engine
│   ├── trading_bot.py       # Main bot
│   └── ml_models/           # ML models
├── backend/             # API backend
├── infrastructure/      # Deployment scripts
└── tests/               # Test files
```

---

## 🔬 Research Summary

Complete research history in [`docs/RESEARCH_LOG.md`](docs/RESEARCH_LOG.md).

**Deployed:** 4 strategies (Gold, Indices, Forex, Momentum)
**Reserved:** London Breakout (GBPCADm) - DD too high
**Rejected:** Asian Fade, NY Close Reversion, Session Overlap

---

## 🛡️ Risk Management

- **Daily Loss Limit:** 4.5% hard stop
- **Max Drawdown:** 9.5%
- **Dynamic Allocation:** Index 1.5x | Gold 1.0x | Forex 0.5x

---

*Built with Autonomy by Antigravity.*
