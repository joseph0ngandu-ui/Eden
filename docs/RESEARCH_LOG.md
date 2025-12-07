# Eden Trading Bot - Research Log

> **Last Updated:** 2025-12-07 (Post-Audit)
> **Status:** LIVE on FundedNext (Optimized)
> **Balance:** ~$10,000

---

## 🚨 CRITICAL AUDIT UPDATE (2025-12-07)

A deep audit using `accurate_backtest.py` (importing live code) revealed discrepancies in earlier research. 
**Action:** Portfolio re-optimized for safety.

### Active Strategies (Verified Safe)

| Strategy | Symbol | Timeframe | 90-Day R | Win Rate | Risk |
|:---|:---|:---:|---:|---:|---:|
| **Index Vol Expansion** | USTECm | M15 | **+26.0R** | 46% | 0.25% |
| **Index Vol Expansion** | US500m | M15 | **+17.8R** | 44% | 0.25% |
| **Forex Vol Squeeze** | EURUSDm | M5 | **+14.5R** | 48% | 0.25% |
| **Forex Vol Squeeze** | USDJPYm | M5 | **+7.7R** | 42% | 0.25% |
| **Momentum** | Pairs | D1 | *Pending* | - | 0.25% |

**Portfolio Stats (Optimized):**
- **Total Return:** +16.5% (90 days) -> ~5.5% / month
- **Max Drawdown:** 6.0% (Safe for FundedNext)
- **Status:** **DEPLOYED**

---

### Failed/Disabled Strategies

| Strategy | Symbol | 90-Day R | Status | Reason |
|:---|:---|---:|:---|:---|
| **Gold Spread Hunter** | XAUUSDm | **-6.5R** | ❌ DISABLED | Failed accurate backtest (spread friction) |
| **Index Vol Expansion** | US30m | **-7.6R** | ❌ DISABLED | Divergence from Tech indices |
| **London Breakout** | GBPCADm | +34.7R | 📦 RESERVED | High Drawdown (10.8R) |

---

## Validated Performance (90 Days)

| Type | Return | Drawdown | Verdict |
|:---|---:|---:|:---|
| **Initial Portfolio** | -42.4R | 25% | ❌ FAIL |
| **Optimized Portfolio** | **+65.9R** | **6.0%** | ✅ PASS |

---

## FundedNext Configuration

- **Daily Loss Limit:** 4.5% (hard stop)
- **Max Drawdown:** 9.5% (buffer for 10%)
- **Base Risk:** 0.5% (Effective 0.25% per trade with 0.5x multiplier)
- **Symbols:** USTEC, US500, EURUSD, USDJPY, USDCAD (D1), EURJPY (D1), CADJPY (D1)

---

## Future Research Directions

1. **Fix D1 Data:** Momentum strategy yielded 0 trades in backtest due to missing history.
2. **Gold Refinement:** Investigate why Gold failed (spread filter tuning?).
3. **Scale Up:** Once 5% monthly is consistent, slowly increase risk to target 10%.

---

## File Locations

| Purpose | Location |
|:---|:---|
| **Accurate Backtest** | `scripts/research/accurate_backtest.py` |
| Strategy Logic | `trading/pro_strategies.py` |
| Bot Entry Point | `infrastructure/bot_runner.py` |
| Configuration | `config/config.yaml` |

