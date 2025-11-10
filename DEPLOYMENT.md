# Eden Trading Bot - Deployment Guide

## Current Strategy: Volatility Burst v1.3 (Profitable Configuration)

### Performance Metrics (Backtested Jan-Oct 2025)
- **Total P&L**: +$1,864.15
- **Win Rate**: 46.39%
- **Profit Factor**: 1.02
- **Total Trades**: 1,563 trades

### Strategy Configuration
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
- Volatility 75 Index
- Volatility 100 Index
- Boom 500 Index
- Crash 500 Index
- Boom 1000 Index
- Step Index

### Per-Symbol Performance
| Symbol | Total Trades | Win Rate | P&L | Profit Factor |
|--------|--------------|----------|-----|---------------|
| Volatility 75 Index | 506 | 47.04% | +$2,228.72 | 1.023 |
| Volatility 100 Index | 506 | 46.25% | -$189.33 | 0.910 |
| Boom 500 Index | 65 | 35.38% | -$117.90 | 0.530 |
| Crash 500 Index | 62 | 56.45% | +$15.46 | 1.130 |
| Boom 1000 Index | 3 | 66.67% | +$7.53 | 1.450 |
| Step Index | 421 | 45.84% | -$80.32 | 0.887 |

## Deployment Steps

### 1. Build Docker Image
```bash
cd backend
docker build -t eden-trading-bot:vb-v1.3 .
```

### 2. Deploy to AWS ECS
```bash
# Tag image for ECR
docker tag eden-trading-bot:vb-v1.3 <your-ecr-repo>/eden-trading-bot:latest

# Push to ECR
docker push <your-ecr-repo>/eden-trading-bot:latest

# Update ECS service
aws ecs update-service --cluster eden-cluster --service eden-bot --force-new-deployment
```

### 3. Verify Deployment
```bash
# Check health endpoint
curl https://your-api-gateway-url/health

# Check strategy configuration
curl -H "Authorization: Bearer <token>" https://your-api-gateway-url/strategy/config
```

## Key Changes from Previous Version
- ✅ **REMOVED**: MA_Crossover_v1 strategy (unprofitable, -$385k loss)
- ✅ **DEPLOYED**: Volatility Burst v1.3 (profitable, +$1.8k gain)
- ✅ **Updated**: All backend models to use VB v1.3 configuration
- ✅ **Optimized**: Parameters based on extensive backtesting
- ✅ **Cleaned**: Removed all traces of unprofitable strategy

## Monitoring
- Monitor via iOS app: Real-time bot status and performance
- API Gateway: Check CloudWatch logs for trade execution
- ECS Health Monitor: Lambda function monitors container health
- Key Metrics to Watch:
  - Win rate should stay around 46%
  - Profit factor should remain above 1.0
  - Drawdown should not exceed historical max

## Rollback Plan
If performance degrades:
1. Stop bot via API: `POST /bot/stop`
2. Review recent trades via: `GET /trades/recent?days=1`
3. Check confidence scores and entry quality
4. Adjust confidence_threshold if needed (increase to be more selective)

## Strategy Files
- Strategy Implementation: `src/volatility_burst.py`
- Configuration: `config/volatility_burst.yml`
- Backtest Results: `reports/vb_v1.3_backtest_results.json`
- Optimization Results: `reports/vb_v1.3_optimization_results.json`

## Contact & Support
For issues or questions:
- Check CloudWatch logs
- Review ECS task definitions
- Monitor API Gateway metrics
- Check database for trade records

---
**Last Updated**: 2025-11-10
**Version**: Volatility Burst v1.3
**Status**: ✅ PROFITABLE - DEPLOYED TO PRODUCTION
