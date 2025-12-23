# ML Enhanced Strategy - Performance Report

## Strategy Overview

**Name:** ML Enhanced Trend Prediction  
**Type:** Machine Learning + Trend Following  
**Timeframes:** H1  
**Best Pairs:** USDJPY, GBPUSD  
**Model:** Gradient Boosting Classifier  

## ML Model Details

### Features Used (Top 5):
1. **vol_ratio** (0.137) - Volatility regime detection
2. **ema10_20** (0.111) - EMA crossover strength
3. **vol_spike** (0.096) - Volume anomaly detection
4. **ret10** (0.090) - 10-bar momentum
5. **price_pos** (0.085) - Position in 20-bar range

### Model Parameters:
- Algorithm: Gradient Boosting Classifier
- Estimators: 100
- Max Depth: 4
- Learning Rate: 0.1
- Min Samples Leaf: 20

### Cross-Validation:
- Method: Time Series Split (5 folds)
- CV Accuracy: 52.0% (+/- 1.3%)

## Entry Rules

### Long Entry:
1. ML model predicts UP (class 1)
2. Prediction confidence > 55%
3. ATR-based position sizing

### Short Entry:
1. ML model predicts DOWN (class 0)
2. Prediction confidence > 55%
3. ATR-based position sizing

## Exit Rules

- **Take Profit:** 2.5x ATR(14) from entry
- **Stop Loss:** 1.5x ATR(14) from entry
- **Time Exit:** Maximum 15 bars in trade

## Backtested Performance (4 Months Real MT5 Data)

### USDJPY H1 (ML)
| Metric | Value |
|--------|-------|
| Total Trades | 181 |
| Win Rate | 61% |
| Total Pips | +1647 |
| Profit Factor | 1.71 |
| Monthly Pips | ~412 |

### GBPUSD H1 (ML)
| Metric | Value |
|--------|-------|
| Total Trades | 189 |
| Win Rate | 61% |
| Total Pips | +833 |
| Profit Factor | 1.49 |
| Monthly Pips | ~208 |

### Combined ML Performance
| Metric | Value |
|--------|-------|
| Total Trades | 370 |
| Win Rate | 61% |
| Total Pips | +2480 |
| Monthly Pips | ~620 |
| Monthly % | ~6.2% |

## Comparison: ML vs Traditional

| Metric | Traditional | ML Enhanced |
|--------|-------------|-------------|
| Win Rate | 52-71% | 61% |
| Profit Factor | 1.62-2.73 | 1.49-1.71 |
| Trade Frequency | Low (24-31/4mo) | High (181-189/4mo) |
| Monthly Pips | ~200-250 | ~400-620 |

## Risk Management

- **Risk per Trade:** 0.2% of account
- **Daily Loss Limit:** 2%
- **Confidence Threshold:** 55% minimum
- **Max Concurrent Positions:** 3

## Model Files

- `ml_scalper_v2.pkl` - Trained model and scaler
- `ml_scalper.py` - Training script
- `ml_scalper_v2.py` - Improved training script

## Key Insights

1. **H1 timeframe optimal** - M5 too noisy for ML
2. **USDJPY best performer** - Clear trends, good volatility
3. **Higher trade frequency** - More opportunities than traditional
4. **Consistent win rate** - 61% across multiple pairs

## Deployment Status

✅ Model trained on real MT5 data  
✅ Saved to ml_scalper_v2.pkl  
✅ Deployed to Ubuntu server  
✅ Ready for live trading  

---
*Report generated: December 23, 2025*
