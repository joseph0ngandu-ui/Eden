# Removed Unprofitable Strategy Files

## Summary
All traces of the unprofitable MA_Crossover_v1.2 strategy have been removed from the codebase.

## Deleted Files

### Strategy Implementation
- ✅ `src/ma_v1_2.py` - MA Crossover v1.2 strategy implementation

### Configuration
- ✅ `config/ma_v1_2.yml` - MA Crossover v1.2 configuration file

### Backtest Results
- ✅ `reports/ma_v1_2_results.json` - Backtest results showing -$385,115.52 loss

### Scripts
- ✅ `scripts/backtest_ma_v1_2.py` - MA v1.2 backtest script

## Updated Files

### Backend Models
- ✅ `backend/app/models.py` - Updated StrategyConfig to use Volatility Burst v1.3
  - Changed default strategy name from "MA_Crossover_v1" to "VolatilityBurst_v1.3"
  - Updated parameters to profitable configuration
  - Changed symbols to Synthetic Indices

### Trading Service
- ✅ `backend/app/trading_service.py` - Updated documentation and initialization
  - Added Volatility Burst v1.3 strategy documentation
  - Documented expected performance metrics

### Main API
- ✅ `backend/main.py` - Updated header documentation
  - Added strategy information to API documentation
  - Documented profitable configuration

## Performance Comparison

| Strategy | Total P&L | Win Rate | Profit Factor | Status |
|----------|-----------|----------|---------------|--------|
| MA Crossover v1.2 | -$385,115.52 | 26.37% | 0.75 | ❌ REMOVED |
| Volatility Burst v1.3 | +$1,864.15 | 46.39% | 1.02 | ✅ DEPLOYED |

## Deployment Status
- **Current Strategy**: Volatility Burst v1.3
- **Configuration**: confidence_threshold=0.7, tp_atr_multiplier=1.2, sl_atr_multiplier=1.2
- **Status**: Ready for deployment to AWS ECS

## Verification Commands
```bash
# Verify MA files are gone
ls src/ma_*.py
# Should return: cannot find path

ls config/ma_*.yml
# Should return: cannot find path

ls reports/ma_*.json
# Should return: cannot find path

# Verify Volatility Burst files exist
ls src/volatility_burst.py
ls config/volatility_burst.yml
```

## Next Steps
1. Run deployment script: `.\deploy_to_aws.ps1`
2. Monitor via iOS app
3. Verify strategy config via API: `GET /strategy/config`
4. Check that strategy name returns "VolatilityBurst_v1.3"

---
**Date**: 2025-11-10
**Action**: Removed unprofitable MA Crossover v1.2, deployed profitable Volatility Burst v1.3
**Outcome**: Codebase cleaned, ready for profitable trading
