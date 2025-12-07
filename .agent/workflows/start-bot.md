---
description: Start the Eden Trading Bot with verified safe configuration
---

# Start Bot Workflow

This workflow starts the Eden Trading Bot with the **verified safe configuration** (0.6% risk, ~7% monthly return, ~5 weeks to pass Phase 1).

## Prerequisites
- MetaTrader 5 must be installed and logged in
- Python environment must be set up

## Steps

// turbo-all

1. Navigate to the Eden directory:
```powershell
cd c:\Users\opc\Desktop\Eden
```

2. Kill any existing Python processes to ensure clean start:
```powershell
taskkill /IM python.exe /F 2>$null
```

3. Wait for processes to terminate:
```powershell
Start-Sleep -Seconds 2
```

4. Start the bot using the official startup script:
```powershell
powershell -File scripts/startup/restart_bot.ps1
```

## What Gets Started
- **Watchdog** (monitors and restarts bot if needed)
- **Trading Bot** with:
  - Symbols: USTECm, US500m, EURUSDm, USDJPYm, USDCADm, EURJPYm, CADJPYm
  - Risk: 0.6% per trade
  - Strategies: Index Vol Expansion, Forex Vol Squeeze, Momentum Continuation

## Quick Command
To start the bot in a single command:
```powershell
cd c:\Users\opc\Desktop\Eden; taskkill /IM python.exe /F 2>$null; Start-Sleep -Seconds 2; powershell -File scripts/startup/restart_bot.ps1
```

## Verification
Check logs at: `logs/eden_trading.log`
Check watchdog output in terminal for "Connected. Balance: XXXX"
