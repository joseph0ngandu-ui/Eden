# Eden Trading Bot - Research Log

> **Last Updated:** 2025-12-06
> **Status:** LIVE on FundedNext
> **Balance:** ~$10,000

---

## Active Strategies (Deployed)

| Strategy | Timeframe | Pairs | Risk | Result | MaxDD |
|:---|:---:|:---|:---:|---:|---:|
| **Index Volatility Expansion** | M15 | US30, USTEC, US500 | 0.75% (1.5x) | +19.8R | ~5R |
| **Gold Spread Hunter** | M15 | XAUUSD | 0.50% (1.0x) | +27.0R | ~4R |
| **Forex Volatility Squeeze** | M5 | EURUSD, USDJPY | 0.25% (0.5x) | +13.0R | ~3R |
| **Momentum Continuation** | D1 | USDCAD, EURUSD, EURJPY, CADJPY | 0.50% (1.0x) | +15.7R | 2.7R |

---

## Reserve Strategies (Back Pocket)

| Strategy | Pairs | Result | MaxDD | Why Reserved |
|:---|:---|---:|---:|:---|
| **London Breakout** | GBPCADm only | +34.7R | 10.8R | DD exceeds FundedNext daily limit risk |

---

## Research History

### Phase 1: Gold M15 Spread Hunter ✅ DEPLOYED
- **Logic:** Cost-exploiting momentum during low-spread periods
- **Result:** +27.0R (90 days), 44% WR
- **Key Finding:** M15 > M5 for Gold (less noise)

### Phase 2: Index Volatility Expansion ✅ DEPLOYED
- **Logic:** BB Squeeze breakout on indices during NY session
- **Result:** +19.8R combined (US30 + USTEC + US500)
- **Key Finding:** Indices ignore commissions (low cost impact)

### Phase 3: Asian Range Fade ❌ REJECTED
- **Logic:** Mean reversion in Asian session ranges
- **Result:** 0% Win Rate
- **Why Failed:** Breakout dominance + spread friction at session boundaries

### Phase 4: London Session Breakout 📦 RESERVED
- **Logic:** Asian range breakout at London open (07:00-10:00 GMT)
- **Pairs Tested:** 13 major pairs
- **Result:** Only GBPCADm profitable (+34.7R)
- **Why Reserved:** MaxDD 10.8R exceeds 5% daily limit safety threshold

### Phase 5: Multi-Frontier Research ✅ PARTIAL DEPLOY

#### Frontier 1: NY Close Reversion ❌ REJECTED
- **Logic:** Fade extended prices (>2 ATR from VWAP) at NY close
- **Result:** +66.3R BUT MaxDD 18.6R
- **Why Failed:** Drawdown too high for prop firm

#### Frontier 2: Session Overlap Scalping ❌ REJECTED
- **Logic:** EMA cross scalps during London/NY overlap
- **Result:** +3.0R, MaxDD 10.0R
- **Why Failed:** Low profit, high DD

#### Frontier 3: Momentum Continuation ✅ DEPLOYED
- **Logic:** Enter next day after strong D1 candle (>1.3x ADR)
- **Pairs:** USDCAD, EURUSD, EURJPY, CADJPY
- **Result:** +15.7R, 64.8% WR, MaxDD 2.7R
- **Why Approved:** Meets all criteria, low DD

---

## Failed/Disabled Strategies

| Strategy | Result | Why Disabled |
|:---|---:|:---|
| Quiet Before Storm | -7R | Negative expectancy |
| Triple Candle Breakout | -14R | Negative expectancy |
| VWAP Reversion M5 | +164R paper | 4000+ trades = commission risk |
| Asian Range Fade | 0% WR | Logic doesn't work |

---

## Validation Criteria

For any strategy to be deployed, it must pass:

1. **Total R:** > +15R (90 days)
2. **Max Drawdown:** < 5R (critical for FundedNext)
3. **Win Rate:** > 40% (psychological sustainability)
4. **Max Consecutive Losses:** < 8

---

## FundedNext Configuration

- **Daily Loss Limit:** 4.5% (buffer for 5% rule)
- **Max Drawdown:** 9.5% (buffer for 10% rule)
- **Base Risk:** 0.5% per trade
- **Dynamic Allocation:**
  - Indices: 1.5x (0.75%)
  - Gold/Momentum: 1.0x (0.50%)
  - Forex Squeeze: 0.5x (0.25%)

---

## Future Research Directions

1. **Overnight Gap Fade** - Weekend gaps that fill Monday
2. **News Event Fade** - Counter-trend after high-impact news
3. **VWAP Bands** - Intraday mean reversion with lower frequency
4. **Crypto Correlation** - BTC/Gold divergence plays

---

## File Locations

| Purpose | Location |
|:---|:---|
| Strategy Logic | `trading/pro_strategies.py` |
| Bot Entry Point | `infrastructure/bot_runner.py` |
| Configuration | `config/config.yaml` |
| Research Scripts | `scripts/research/` |
| Startup Scripts | `scripts/startup/` |
| ML Models | `trading/ml_models/` |
| Logs | `logs/` |
