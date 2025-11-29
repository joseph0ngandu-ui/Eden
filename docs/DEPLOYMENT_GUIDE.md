# Eden Bot Deployment Guide - Hybrid Aggressive Configuration

## 🎯 Active Strategies

The bot now runs **2 strategies only**:

### 1. **Pro Multi-Strategy Engine** (Prop Firm Certified)
- **Risk**: 0.15% per trade
- **Max Positions**: 7 concurrent trades
- **Target**: 16% monthly returns
- **Max Drawdown**: ~12% (95% confidence)
- **Sub-Strategies**:
  - London/NY Overlap Scalper (EURUSD, GBPUSD)
  - Asian Range Fade (USDJPY, AUDJPY)
  - Gold London Breakout (XAUUSD)
  - Volatility Expansion (All pairs)

### 2. **Volatility Burst** (VIX-Specific)
- Legacy strategy for Volatility Index trading
- Risk-managed independently

## 🔧 Configuration

### `config/config.yaml`
```yaml
max_positions: 7
max_drawdown_percent: 10
risk_per_trade_percent: 0.15
```

### Safety Features (Active)
- ✅ Correlation Filter: Max 2 positions per currency
- ✅ Dynamic Risk Scaling: Reduces risk at 5%, 8%, 10% DD
- ✅ Emergency Stop: Hard stop at 10% drawdown
- ✅ Daily Loss Limit: 2%
- ✅ Weekly Loss Limit: 4%
- ✅ Monthly Loss Limit: 8%

## 🚀 Starting the Bot

### Option 1: Using Restart Script
```powershell
.\restart_all.ps1
```

### Option 2: Manual Start
```powershell
# Start backend
cd backend
python -m uvicorn main:app --host 127.0.0.1 --port 8443

# Start bot
cd ..
python infrastructure/bot_runner.py
```

## 📊 Expected Performance

| Metric | Target | 95% Confidence |
|--------|---------|----------------|
| Monthly Return | 16% | 12-20% |
| Max Drawdown | 7-8% | <12% |
| Win Rate | ~60% | - |
| Sharpe Ratio | >2.0 | - |

## ⚠️ Monitoring

Check logs hourly for:
- Drawdown level (if >8%, system auto-reduces risk)
- Daily P&L (if -2%, trading pauses until next day)
- Open positions (should never exceed 7)

## 🛑 Emergency Procedures

**If DD hits 10%**: Bot auto-stops trading
**Manual Stop**: `taskkill /F /IM python.exe` (use with caution)

## 📝 Removed Strategies

The following strategies have been **deprecated** and removed:
- ❌ ICT Strategies (Silver Bullet, Unicorn, Venom)
- ❌ All old momentum/MA strategies

Only **Pro Engine** and **VIX Burst** remain active.
