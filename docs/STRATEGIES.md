# 🧠 Validated Trading Strategies

This document details the logic, edge source, and execution rules for the 4 active strategies in the Eden Bot.

## 1. Volatility Squeeze (Primary)
**Role:** Trend Capture (Anchor)
**Timeframe:** H1
**Pairs:** 28 Major/Minor Pairs (Wide & Slow)

### Logic
- **Concept:** Periods of low volatility (squeeze) are often followed by explosive trend moves.
- **Signal:**
  - `Vol(20) < Vol(50) * 0.6` (Volatility is compressed).
  - Price breaks the 20-period High/Low channel.
- **Execution:**
  - Enter on Breakout.
  - Stop Loss: Opposite channel band (or 2*ATR max).
  - Take Profit: 1.5R (Trailing logic handled by bot).

### Why It Works
Markets oscillate between expansion and contraction. By entering early in the expansion phase, we capture the "Meat of the Move".

---

## 2. Quiet Before The Storm
**Role:** Sniper (High Precision)
**Timeframe:** H1
**Pairs:** GBPUSD, XAUUSD

### Logic
- **Concept:** Institutional accumulation often happens in "Quiet" narrow ranges before the London/NY Open.
- **Signal:**
  - `Recent Volume < Average Volume * 0.6`.
  - `Body Size < Average Body * 1.2` (Doji/Spinning Tops).
  - Breakout of the 10-bar range.
- **Execution:**
  - Tight Stop Loss (0.2 ATR buffer).
  - Aggressive Target (2.5R).

---

## 3. VWAP Reversion M5 (High Frequency)
**Role:** Cash Flow Engine
**Timeframe:** M5
**Pairs:** EURUSD, GBPUSD

### Logic
- **Concept:** Price tends to revert to the Volume Weighted Average Price (VWAP) after extreme deviations intraday.
- **Signal:**
  - Price deviates `> 3.0 * ATR` from the Daily VWAP.
- **Execution:**
  - Fade the move (Mean Reversion).
  - Target: Return to VWAP.
  - Stop Loss: 0.5 ATR beyond deviation.

**⚠️ Risk Control:** This strategy creates volume. It relies heavily on **Kelly Sizing** and **Cooldown** to survive "Trending Days" where price does not revert.

---

## 4. Triple Candle Breakout
**Role:** Momentum (Secondary)
**Timeframe:** H1
**Pairs:** JPY Pairs

### Logic
- **Concept:** "Mother Bar" -> "Inside Bar" -> "Inside Bar 2" (Coiling Spring).
- **Execution:** Breakout of the Mother Bar High/Low.

---

## 🔗 Portfolio Synergy
- **H1 Strategies (1, 2, 4)** provide stability and catch big moves.
- **M5 Strategy (3)** provides daily activity and smooths the equity curve.
- **Correlation:** Low. Trend vs Mean Reversion ensures some strategies work when others fail.
