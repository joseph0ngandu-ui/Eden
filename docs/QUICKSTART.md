# 🚀 Eden Trading Bot - Quick Start Guide

## ✅ Status: Ready for Deployment

Your trading bot has been **upgraded to the profitable Volatility Burst v1.3 strategy** and all traces of the unprofitable MA Crossover strategy have been removed.

---

## 📊 Strategy Performance

| Metric | Value |
|--------|-------|
| **Total P&L** | +$1,864.15 |
| **Win Rate** | 46.39% |
| **Profit Factor** | 1.02 |
| **Total Trades** | 1,563 |
| **Test Period** | Jan-Oct 2025 |
| **Best Symbol** | Volatility 75 Index (+$2,228.72) |

---

## 🎯 Deploy to AWS in 3 Steps

### Step 1: Configure AWS Credentials
```powershell
# If not already configured
aws configure
```

### Step 2: Run Deployment Script
```powershell
# From project root
.\deploy_to_aws.ps1
```

### Step 3: Monitor Deployment
- The script will handle:
  - ✅ Building Docker image
  - ✅ Pushing to ECR
  - ✅ Updating ECS service
  - ✅ Waiting for stabilization

---

## 🔍 Verify Deployment

### Check Health
```bash
curl https://your-api-gateway-url/health
```

### Verify Strategy
```bash
curl -H "Authorization: Bearer <token>" \
  https://your-api-gateway-url/strategy/config
```

**Expected Response:**
```json
{
  "name": "VolatilityBurst_v1.3",
  "confidence_threshold": 0.7,
  "tp_atr_multiplier": 1.2,
  "sl_atr_multiplier": 1.2,
  ...
}
```

---

## 📱 Monitor via iOS App

1. **Bot Status**: Real-time balance, P&L, positions
2. **Trade History**: View recent trades and performance
3. **Performance Stats**: Win rate, profit factor, drawdown
4. **Controls**: Start, stop, pause trading

---

## 📈 What Changed

### ❌ REMOVED (Unprofitable)
- MA_Crossover_v1.2 strategy (-$385k loss)
- All related config files
- All backtest reports

### ✅ DEPLOYED (Profitable)
- Volatility Burst v1.3 (+$1.8k gain)
- Optimized parameters (0.7 confidence, 1.2x TP/SL)
- Synthetic Indices trading

---

## 🎛️ Strategy Configuration

```yaml
Strategy: VolatilityBurst_v1.3
Confidence Threshold: 0.7
TP ATR Multiplier: 1.2
SL ATR Multiplier: 1.2
Trail Trigger: 0.8R
Max Bars in Trade: 30
Risk Per Trade: 2.0%
Max Trades Per Day: 8
```

### Trading Symbols
- Volatility 75 Index ⭐ (Best performer)
- Volatility 100 Index
- Boom 500 Index
- Crash 500 Index
- Boom 1000 Index
- Step Index

---

## 🔔 Key Metrics to Monitor

| Metric | Target | Action if Outside Range |
|--------|--------|------------------------|
| Win Rate | ~46% | ±5%: Normal variance<br>>10%: Review confidence threshold |
| Profit Factor | >1.0 | <0.95: Consider pausing<br><0.85: Stop and review |
| Daily Drawdown | <10% | Pause trading, review trades |
| Max Positions | 1 per symbol | Should auto-limit |

---

## 🛠️ Troubleshooting

### Bot Not Trading
1. Check bot status: `GET /bot/status`
2. Verify `is_running: true`
3. Check CloudWatch logs for errors
4. Verify MT5 connection (if using live broker)

### Performance Below Expected
1. Check confidence scores of recent trades
2. Verify symbols being traded match backtest
3. Review entry/exit prices vs expected
4. Consider increasing confidence_threshold to 0.75

### High Drawdown
1. Stop bot immediately: `POST /bot/stop`
2. Review recent trades
3. Check if market conditions changed
4. Consider reducing position size or max trades per day

---

## 📚 Documentation

- **Full Deployment Guide**: `DEPLOYMENT.md`
- **Removed Files**: `REMOVED_FILES.md`
- **Strategy Implementation**: `src/volatility_burst.py`
- **Configuration**: `config/volatility_burst.yml`
- **Backtest Results**: `reports/vb_v1.3_backtest_results.json`

---

## 🎉 You're All Set!

Your bot is now configured with the **profitable Volatility Burst v1.3 strategy** and ready to trade.

**Next Step**: Run `.\deploy_to_aws.ps1` to deploy to production!

---

**Last Updated**: 2025-11-10  
**Version**: Volatility Burst v1.3  
**Status**: ✅ PROFITABLE - READY FOR DEPLOYMENT
