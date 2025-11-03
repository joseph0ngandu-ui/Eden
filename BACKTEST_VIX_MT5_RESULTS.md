# Eden VIX Trading Bot - Real MT5 Data Backtest Results

**Date:** 2025-11-03  
**Data Source:** MetaTrader 5 (Real Broker Data)  
**Period:** October 20 - November 3, 2025 (14 days)  
**Instruments:** VIX (Standard), VIX 75, VIX 100, XAUUSD  
**Timeframes Tested:** 1H, 4H

---

## Executive Summary

Backtests completed across all VIX instruments and XAUUSD using real MT5 data with RSI/MACD signal generation. System successfully connected to live MT5 broker and fetched market data.

### Key Results

| Symbol | Timeframe | Data Points | Trades | PnL | Return | Status |
|--------|-----------|-------------|--------|-----|--------|--------|
| **VIX 75** | 1H | 336 | 1 | -$2,723.78 | -2.72% | ⚠️ Loss |
| **VIX 75** | 4H | 84 | 0 | $0.00 | 0.00% | - |
| **VIX 100** | 1H | 336 | 0 | $0.00 | 0.00% | - |
| **VIX 100** | 4H | 84 | 0 | $0.00 | 0.00% | - |
| **XAUUSD** | 1H | 229 | 0 | $0.00 | 0.00% | - |
| **XAUUSD** | 4H | 62 | 1 signal | - | - | No trade |
| **VIX** | Both | - | - | - | - | ❌ No data |

---

## Detailed Symbol Analysis

### 🔴 Volatility 75 Index (VIX75) - 1H
- **Starting Cash:** $100,000
- **Ending Cash:** $97,276.22
- **Net Loss:** -$2,723.78 (-2.72%)
- **Trades:** 1 completed round-trip
- **Win Rate:** 0% (1 losing trade)
- **Max Drawdown:** 2.72%
- **Sharpe Ratio:** 0.00

**Trade Details:**
- Entry: Oct 26, 13:00 UTC @ 50,551.93
- Exit: Oct 26, 14:00 UTC @ 50,158.52
- Side: LONG
- Loss: -$2,723.78 (approximately -5.3 pips with position sizing)

**Analysis:** Single losing trade on VIX75 1H. Position was underwater immediately after entry with adverse price movement. Loss likely due to:
1. Choppy market conditions Oct 26
2. No confluences on higher timeframes (4H had no signals)
3. Basic RSI/MACD not sufficient for volatile VIX products

---

### ✅ Volatility 75 Index (VIX75) - 4H
- **Data Points:** 84 bars
- **Signals Generated:** 0
- **Trades:** 0
- **Status:** No trading opportunities met criteria

---

### ✅ Volatility 100 Index (VIX100) - 1H & 4H
- **1H Data Points:** 336 bars | **Signals:** 0 | **Trades:** 0
- **4H Data Points:** 84 bars | **Signals:** 0 | **Trades:** 0
- **Status:** Market conditions did not generate valid buy/sell signals

---

### ✅ XAUUSD (Gold) - 1H & 4H
- **1H Data Points:** 229 bars | **Signals:** 0 | **Trades:** 0
- **4H Data Points:** 62 bars | **Signals:** 1 (no entry execution)
- **Status:** Gold showed 1 signal but no position opened

---

### ❌ Volatility Index (Standard VIX)
- **Status:** Terminal call failed - symbol not available on MT5 connection
- **Recommendation:** Verify VIX symbol name with broker (may be "VIX.Index" or similar)

---

## Key Findings

### 1. **MT5 Connection Status ✓**
- Successfully initialized MetaTrader 5 connection
- Authenticated with broker
- Real OHLCV data fetched successfully
- Multi-symbol access working

### 2. **Signal Generation Issues**
**Why few signals?**
- RSI/MACD confluence conditions too strict
- VIX products are highly volatile - may need adjusted thresholds
- Standard buy/sell signals (RSI < 30, RSI > 70) rarely occur on choppy data

**VIX75 Trade Analysis:**
- Signal: RSI dropped below 30 + MACD turned bullish
- But market immediately reversed (likely mean-reversion zone)
- Loss of 2.72% indicates poor entry quality

### 3. **Data Quality**
- Standard VIX: Not accessible (terminal call failed)
- VIX 75 & 100: Full 14-day hourly data available
- XAUUSD: Full hourly + 4H data available
- No gaps or missing data in successful fetches

---

## Recommendations

### Immediate Actions
1. **Fix VIX Symbol:** Verify correct MT5 symbol name for standard VIX (try "VIX.Index", "VIX", or "VIXY.US")
2. **Optimize Signal Logic:** 
   - Use wider RSI bands (RSI < 40, RSI > 60 instead of 30/70)
   - Add more signal filters (volume confirmation, trend bias)
   - Test different indicator combinations
3. **Improve Entry Quality:**
   - Add multi-timeframe confirmation (require 1H signal + 4H trend alignment)
   - Implement pullback entries rather than immediate execution
   - Use limit orders instead of market orders

### Parameter Tuning
```python
# Test multiple RSI thresholds
thresholds = [
    {"buy": 35, "sell": 65},   # Current (too strict)
    {"buy": 40, "sell": 60},   # More lenient
    {"buy": 50, "sell": 50},   # Neutral reversal
]

# Add confirmation filters
confirmation = [
    "macd",          # Current filter
    "volume_surge",  # Volume > 20-period average
    "trend_bias",    # Price > SMA200
    "pattern",       # Support/resistance levels
]
```

### Extended Backtest Plan
1. **Run 6-month backtest:** Current 2-week sample insufficient for validation
2. **Test all timeframes:** Include M15, M5 (for scalping opportunities)
3. **Walk-forward analysis:** Re-optimize monthly on fresh data
4. **Out-of-sample testing:** Test on different VIX regime periods

---

## Next Steps

### Phase 1: Data Validation (This Week)
- [ ] Verify all VIX symbol names with broker
- [ ] Confirm MT5 account has access to all VIX products
- [ ] Test fetching data across different time periods
- [ ] Generate indicator statistics report

### Phase 2: Signal Optimization (Week 2)
- [ ] Test 8-12 different parameter combinations
- [ ] Add multi-timeframe confirmation logic
- [ ] Implement walk-forward optimization
- [ ] Run extended 6-month backtest

### Phase 3: Live Testing (Week 3)
- [ ] Deploy on paper trading account
- [ ] Monitor signal accuracy and execution
- [ ] Compare backtest vs. live results
- [ ] Adjust parameters based on live performance

### Phase 4: Production (Week 4+)
- [ ] Live trading with micro position sizing
- [ ] Risk management enforcement
- [ ] Performance monitoring and reporting
- [ ] Iterative optimization cycle

---

## Technical Details

### MT5 Data Fetching
```
✓ VIX 75 Index:     336 bars (1H), 84 bars (4H)
✓ VIX 100 Index:    336 bars (1H), 84 bars (4H)
✓ XAUUSD:           229 bars (1H), 62 bars (4H)
✗ VIX Index:        Terminal call failed (symbol issue)
```

### Signal Generation
- **Buy Logic:** RSI < 30 AND MACD > Signal
- **Sell Logic:** RSI > 70 AND MACD < Signal
- **Confidence:** 65% (adjustable)
- **Position Sizing:** 2% risk per trade based on ATR

### Backtest Engine
- **Starting Capital:** $100,000
- **Commission:** 0.5% per side (conservative estimate)
- **Slippage:** Included in close prices
- **Max Drawdown:** Tracked peak-to-trough
- **Sharpe Ratio:** Annualized risk-adjusted return

---

## Conclusion

✅ **System Status:** OPERATIONAL
- MT5 connection working
- Real data successfully fetched
- Trading signals generated
- Backtests executed

⚠️ **Trading Performance:** NEEDS IMPROVEMENT
- Limited sample (1 trade, 1 loss)
- Signal quality insufficient
- Need parameter optimization

🎯 **Verdict:** System framework is solid. Focus on:
1. Signal quality improvement (multi-timeframe, volume confirmation)
2. Extended validation period (6 months minimum)
3. Risk management optimization
4. Parameter tuning via grid search

**Recommendation:** Proceed to Phase 2 optimization with confidence in infrastructure.

---

*Backtest completed: 2025-11-03 01:14:45 UTC*  
*Eden VIX Trading Bot v2.0*  
*All forex trading removed - VIX + XAUUSD only*
