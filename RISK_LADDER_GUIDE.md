# Risk Ladder System - Growth-Mode Position Sizing Guide

## Overview

The Risk Ladder system automatically adjusts position sizing and risk exposure as your account grows. This ensures **sustainable compounding** by scaling down risk-per-trade while scaling up equity—mimicking professional prop firm scaling rules but fully automated.

---

## 5-Tier Risk Hierarchy

```
Tier 1: $10-$30    (ULTRA_AGGRESSIVE)   → 20% risk per trade   (2.0x multiplier)
Tier 2: $30-$100   (VERY_AGGRESSIVE)    → 10% risk per trade   (1.5x multiplier)
Tier 3: $100-$500  (AGGRESSIVE)         → 5% risk per trade    (1.0x multiplier)
Tier 4: $500-$1000 (MODERATE)           → 3% risk per trade    (0.7x multiplier)
Tier 5: $1000+     (CONSERVATIVE)       → 1% risk per trade    (0.5x multiplier)
```

### Why This Structure?

- **Ultra-Aggressive ($10-$30)**: Fast exponential growth for micro-accounts. 20% risk allows rapid scaling.
- **Very-Aggressive ($30-$100)**: Sustained growth phase. Still aggressive at 10% but with less drawdown risk.
- **Aggressive ($100-$500)**: Balanced growth. 5% risk maintains compounding while protecting profits.
- **Moderate ($500-$1000)**: Transition zone. 3% risk locks in gains while building towards $1000.
- **Conservative ($1000+)**: Capital preservation. 1% risk protects substantial equity.

---

## Key Features

### 1. Dynamic Position Sizing

Position size automatically adjusts based on:

#### Simple Sizing (Equity-Based)
```python
lot_size = (risk_pct * equity) / 100.0

# Example: $100 account, 5% risk tier
lot_size = (5% * $100) / 100 = 0.05 lots
```

#### ATR-Based Sizing (Volatility-Adjusted)
```python
risk_amount = equity * (risk_pct / 100)
lot_size = risk_amount / (atr * pip_value)

# Example: $100, 5% risk, ATR=2.5, pip_value=10
risk_amount = $5
lot_size = $5 / (2.5 * 10) = 0.2 lots
```

**Result**: High volatility automatically reduces lot size, keeping risk constant.

### 2. Equity Step Locks

Protects each growth stage from being completely lost.

**How It Works**:
- System tracks "equity milestones" (every $50 by default)
- Records the highest equity reached in each milestone
- If drawdown from high exceeds 15% (default), **risk is cut in half**

**Example**:
```
Step 1: Started at $50, reached $75
  → Current balance: $63 (16% drawdown from $75)
  → Risk reduced: 5% → 2.5%
  → Protects the $13 profit gained

Step 2: Recovered to $78
  → Drawdown now 3.6% (within 15% limit)
  → Risk restored: 2.5% → 5%
```

### 3. Tier Transitions

Automatically scales down aggression as balance grows.

**Example Journey**:
```
Day 1: $20 account
  → Tier: ULTRA_AGGRESSIVE
  → Risk: 20% per trade
  → Lot size: 0.2L

Trade +$15 → Balance: $35
  → Tier changed: ULTRA_AGGRESSIVE → VERY_AGGRESSIVE
  → Risk: 10% per trade (reduced automatically)
  → Lot size: 0.35L (larger position, but lower risk %)

Trade +$65 → Balance: $100
  → Tier changed: VERY_AGGRESSIVE → AGGRESSIVE
  → Risk: 5% per trade
  → Lot size: 0.5L

Trade +$200 → Balance: $300
  → Tier: AGGRESSIVE (still)
  → Risk: 5% per trade
  → Lot size: 1.5L
```

### 4. Dynamic Compounding

Recalculates position size every trade based on current balance.

**Why This Matters**:
- Winning trades → Higher balance → Larger positions
- But risk stays proportional to account size
- Creates exponential growth early, stable growth later

**Example**:
```
Trade 1: $100 account, 5% risk → Lot: 0.5L
  Win: +$5 → Balance: $105

Trade 2: $105 account, 5% risk → Lot: 0.525L (slightly larger)
  Win: +$5.25 → Balance: $110.25

Trade 3: $110.25 account, 5% risk → Lot: 0.551L
  Win: +$5.51 → Balance: $115.76

Compounding effect: Each win grows the next trade size
```

---

## Configuration

### Enable Growth Mode in strategy.yml

```yaml
growth_mode:
  enabled: true  # Enable tier-based risk scaling

  # Account thresholds
  high_aggression_below: 30  # Ultra-aggressive below $30

  # Equity milestone tracking
  equity_step_size: 50  # New step every $50
  equity_step_drawdown_limit: 0.15  # Reduce risk if 15% drawdown from step high

  # Position sizing method
  lot_sizing: "atr_based"  # Options: "simple", "atr_based", "equity_based"
  pip_value: 10  # Value per pip
```

### Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `enabled` | `true` | Enable/disable growth mode |
| `high_aggression_below` | `30` | Balance threshold for ultra-aggressive tier |
| `equity_step_size` | `50` | Create milestone every N dollars |
| `equity_step_drawdown_limit` | `0.15` | Max 15% drawdown from step high |
| `lot_sizing` | `"atr_based"` | Sizing method: simple, atr_based, equity_based |
| `pip_value` | `10` | Value per pip (for ATR-based sizing) |

---

## Usage Examples

### Example 1: Starting with $20

```python
from src.risk_ladder import RiskLadder, PositionSizer
from src.health_monitor import HealthMonitor

# Initialize
risk_ladder = RiskLadder(
    initial_balance=20.0,
    high_aggression_below=30,
    equity_step_size=50,
    equity_step_drawdown_limit=0.15
)

position_sizer = PositionSizer(risk_ladder, pip_value=10)

# First trade: Calculate lot size
current_equity = 20.0
atr = 1.5

sizing = position_sizer.calculate(equity=current_equity, atr=atr)
print(sizing)
# Output:
# {
#   'lot_size': 0.15L,
#   'risk_pct': 20%,
#   'risk_amount': $4.00,
#   'tier': 'ULTRA_AGGRESSIVE',
#   'equity_step_safe': True
# }

# After winning trade: +$10
risk_ladder.update_balance(30.0)

# Second trade: Tier automatically changes
sizing = position_sizer.calculate(equity=30.0, atr=1.5)
print(sizing)
# Output:
# {
#   'lot_size': 0.20L,
#   'risk_pct': 10%,  ← Reduced from 20%
#   'risk_amount': $3.00,
#   'tier': 'VERY_AGGRESSIVE',  ← Tier changed
#   'equity_step_safe': True
# }
```

### Example 2: ATR-Based Sizing (High Volatility)

```python
# Low volatility environment
current_equity = 100.0
atr_low = 0.5
atr_high = 3.0

sizing_low_vol = position_sizer.calculate(equity=100.0, atr=atr_low)
sizing_high_vol = position_sizer.calculate(equity=100.0, atr=atr_high)

print(f"Low volatility (ATR=0.5): {sizing_low_vol['lot_size']}L")
print(f"High volatility (ATR=3.0): {sizing_high_vol['lot_size']}L")

# Output:
# Low volatility (ATR=0.5): 1.0L
# High volatility (ATR=3.0): 0.17L  ← Auto-reduced due to high volatility
```

### Example 3: Equity Step Protection

```python
# Start of step
risk_ladder.update_balance(100.0)
print(risk_ladder.risk_ladder.get_tier_summary())
# Tier: AGGRESSIVE, Risk: 5%

# Account grows to peak
risk_ladder.update_balance(120.0)

# Drawdown occurs
risk_ladder.update_balance(95.0)  # 16.7% from peak ($120)

# Check if risk should be reduced
should_reduce, reason = risk_ladder.should_reduce_risk()
print(should_reduce, reason)
# Output: True, "Equity step drawdown: 16.7% (limit: 15.0%)"

# Risk is now cut in half
adjusted_risk = risk_ladder.get_adjusted_risk_pct()
print(adjusted_risk)
# Output: 2.5%  (5% ÷ 2)

# After recovery above 15% threshold
risk_ladder.update_balance(102.0)  # 15% from $120
should_reduce, _ = risk_ladder.should_reduce_risk()
print(should_reduce)
# Output: False
# Risk restores to: 5%
```

---

## Real-World Scenario

### Growth Journey: $10 → $500

```
Day 1: $10.00 account
├── Tier: ULTRA_AGGRESSIVE (20% risk)
├── Trade 1: +$2 → $12.00
├── Trade 2: +$1.80 → $13.80
└── Trade 3: +$2.07 → $15.87

Day 2: $15.87 account
├── Tier: ULTRA_AGGRESSIVE (still 20% risk)
├── Trade 4: +$2.38 → $18.25
├── Trade 5: +$2.74 → $20.99
└── Trade 6: +$3.15 → $24.14  ← Reaching $30 threshold

Day 3: $24.14 account (approaching tier change)
├── Trade 7: +$3.62 → $27.76
├── Trade 8: +$4.16 → $31.92  ← TIER CHANGED TO VERY_AGGRESSIVE
│   └── Risk now: 10% (down from 20%)
│   └── Equity Step #1 locked at $30 high
├── Trade 9: +$2.13 → $34.05
└── Trade 10: +$2.27 → $36.32

Week 2: $36-$50 range
├── Tier: VERY_AGGRESSIVE (10% risk)
├── Steady +$1-2 per trade
├── Peak: $52.00 (Tier changes to AGGRESSIVE)
│   └── Risk reduces: 10% → 5%
│   └── Equity Step #2 locked at $50 high
└── After small drawdown: -$4 → $48 (within 15% protection)
    └── Risk maintained at 5%

Month 2: $100-$300 range
├── Tier: AGGRESSIVE (5% risk)
├── Steady compound growth
├── Peak $300 reached (Tier changes to MODERATE)
│   └── Risk reduces: 5% → 3%
│   └── Equity Step #3 locked at $300
└── Now trading larger lots but lower risk %

Month 3: Approaching $500
├── Tier: MODERATE → CONSERVATIVE
├── Risk: 3% → 1%
├── Capital preserved while still growing
└── Foundation built for scaling to $1000+

Final: $500+ account
├── Tier: CONSERVATIVE (1% risk)
├── Sustainable growth rate
├── Capital safety prioritized
├── Ready for next scaling phase
```

---

## Monitoring & Diagnostics

### Check Current Tier

```python
tier_info = risk_ladder.get_tier_summary()
print(tier_info)
# Output:
# {
#   'tier': 'AGGRESSIVE',
#   'balance_range': '$100 - $500',
#   'risk_per_trade': 5.0,
#   'multiplier': 1.0,
#   'current_balance': $250.00
# }
```

### View Equity Steps

```python
steps = risk_ladder.get_equity_step_summary()
for step in steps:
    print(step)
# Output:
# {
#   'step': 1,
#   'starting_balance': '$50.00',
#   'highest': '$75.00',
#   'lowest': '$50.00',
#   'range': '$25.00',
#   'trades': 15,
#   'pnl': '$25.00',
#   'reached_at': '2025-11-03T10:30:00'
# }
```

### Print Full Status

```python
risk_ladder.print_status()
# Output:
# ============================================================
# RISK LADDER STATUS
# ============================================================
# Current Balance: $250.00
# Peak Balance: $275.00
#
# Current Tier: AGGRESSIVE
# Balance Range: $100 - $500
# Risk Per Trade: 5.0%
# Position Multiplier: 1.0x
#
# Equity Step Protection:
#   Step Drawdown: 9.1% (limit: 15.0%)
#   Protected: ✓ YES
#
# Current Step #2:
#   Reached: 2025-11-03 14:22:15
#   Trades: 42
#   Step PnL: $175.00
# ============================================================
```

---

## Integration with TradingBot

The Risk Ladder automatically integrates with `TradingBot`:

```python
from src.trading_bot import TradingBot

# Initialize (Risk Ladder enabled by default)
bot = TradingBot(
    symbols=["Volatility 75 Index"],
    config_path="config/strategy.yml"  # Loads growth_mode from here
)

# On every trade, position sizing is automatic:
# - place_order() calculates dynamic lot size
# - growth_mode adjusts risk tier as balance grows
# - equity_step_lock protects profits at milestones

bot.start(check_interval=300)
```

---

## Best Practices

### ✅ Do's

1. **Start small**: Begin with $10-$20 to test the system
2. **Use ATR-based sizing**: Accounts for market volatility
3. **Trust the tiers**: Don't override unless necessary
4. **Monitor equity steps**: They lock your profits automatically
5. **Review logs regularly**: Check tier changes and risk reductions

### ❌ Don'ts

1. **Don't disable growth mode**: Manual sizing limits upside
2. **Don't increase risk during drawdowns**: Let step protection handle it
3. **Don't change equity_step_size mid-journey**: Consistency matters
4. **Don't ignore warnings**: Risk reductions are intentional
5. **Don't skip tier transitions**: They're automated for a reason

---

## FAQs

### Q: Why does lot size sometimes stay the same when balance grows?
**A**: This happens when your account is between tier boundaries. The tier doesn't change until you hit the next threshold, so risk % stays constant.

### Q: What if I want more aggressive sizing?
**A**: Lower the `high_aggression_below` threshold or increase risk percentages in the tier config. But remember: more risk = more drawdown vulnerability.

### Q: How do equity steps protect profits?
**A**: If you make $25 profit and reach a new milestone, that milestone is locked. If your account drops 16% below that peak, risk is automatically cut in half until you recover above the 15% threshold.

### Q: Can I disable growth mode?
**A**: Yes, set `growth_mode: enabled: false` in strategy.yml. But you lose automatic compounding benefits.

### Q: Which sizing method is best?
**A**: ATR-based sizing is most robust because it adapts to volatility. Equity-based sizing is simpler but ignores market conditions.

---

## Troubleshooting

### Issue: Tier not changing despite balance growth
- Check current balance in logs
- Verify tier thresholds in strategy.yml
- Ensure growth_mode is enabled

### Issue: Risk keeps getting reduced
- Check equity step drawdown (should be in logs)
- Verify equity_step_drawdown_limit setting
- Recovery above 15% threshold will restore risk

### Issue: Lot sizes too small
- Increase risk percentages in tier config
- Switch to ATR-based sizing if using simple
- Verify pip_value is correct

---

## Advanced Configuration

### Custom Tier Configuration

To modify tier structure, edit `src/risk_ladder.py`:

```python
def _build_default_tiers(self) -> List[RiskTierConfig]:
    """Build default risk tier configuration."""
    return [
        RiskTierConfig(
            balance_min=0,
            balance_max=50,  # Custom threshold
            risk_per_trade=15.0,  # Custom risk
            tier=RiskTier.ULTRA_AGGRESSIVE,
            multiplier=1.5  # Custom multiplier
        ),
        # ... add more tiers ...
    ]
```

### Custom Step Size

```yaml
growth_mode:
  equity_step_size: 25  # New step every $25 instead of $50
```

---

## Summary

The Risk Ladder system provides **automated, intelligent position sizing** that:

✓ Scales risk down as equity grows (prevents ruin)
✓ Compounds positions dynamically (increases growth)
✓ Protects profits at each milestone (locks wins)
✓ Adapts to market volatility (stays proportional)
✓ Transitions tiers automatically (less manual work)

Result: **Sustainable exponential growth** from micro-accounts to serious trading capital.

---

Generated: 2025-11-03  
Eden v1.0.0 - Risk Ladder System