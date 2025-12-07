# Eden Trading Bot

> **FundedNext Prop Firm Trading Bot**
> Verified Safe Configuration: ~7% Monthly Return, ~5 Weeks to Pass Phase 1

---

## Quick Start

```powershell
cd c:\Users\opc\Desktop\Eden
powershell -File scripts/startup/restart_bot.ps1
```

Or use the `/start-bot` workflow.

---

## Active Strategies

| Symbol | Strategy | TF | Edge |
|:---|:---|:---:|:---|
| **USTECm** | Index Vol Expansion | M15 | +27R/90d |
| **US500m** | Index Vol Expansion | M15 | +17R/90d |
| **EURUSDm** | Vol Squeeze + Momentum | M5/D1 | +18R/90d |
| **USDJPYm** | Vol Squeeze | M5 | +7R/90d |
| USDCADm | Momentum | D1 | +2R/90d |
| EURJPYm | Momentum | D1 | +0.5R/90d |
| CADJPYm | Momentum | D1 | -0.2R/90d |

---

## Risk Settings

- **Per Trade:** 0.6%
- **Daily Loss Limit:** 4.5%
- **Max Drawdown:** 9.5%

---

## Folder Structure

```
Eden/
├── config/config.yaml      # Trading configuration
├── trading/
│   ├── pro_strategies.py   # Strategy logic
│   ├── trading_bot.py      # Main bot
│   └── regime_detector.py  # Market regime
├── scripts/
│   ├── startup/            # Bot startup scripts
│   └── research/           # Backtest scripts
├── docs/
│   └── RESEARCH_LOG.md     # Strategy research
├── logs/                   # Trading logs
└── watchdog.py             # Bot monitor
```

---

## Documentation

- [Research Log](docs/RESEARCH_LOG.md) - Strategy validation history
- [Workflow: /start-bot](.agent/workflows/start-bot.md) - How to start
