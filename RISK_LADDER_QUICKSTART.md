# Risk Ladder - Quick Start Guide

## TL;DR

The Risk Ladder automatically adjusts your position size and risk as your account grows. Start aggressive ($20% risk for $10-$30 accounts), then automatically scale back as you grow ($1% risk for $1000+).

## Enable It

In `config/strategy.yml`:

```yaml
growth_mode:
  enabled: true
  high_aggression_below: 30
  equity_step_size: 50
  equity_step_drawdown_limit: 0.15
  lot_sizing: "atr_based"
  pip_value: 10
```

That's it. No manual configuration needed.

## How It Works

### Your $10 → $500 Journey (Automatic)

```
$10   → Tier: ULTRA_AGGRESSIVE → 20% risk/trade → Fast compound
$30   → Tier: VERY_AGGRESSIVE → 10% risk/trade → Steady growth
$100  → Tier: AGGRESSIVE      → 5% risk/trade  → Balanced
$500  → Tier: MODERATE        → 3% risk/trade  → Protecting gains
$1000 → Tier: CONSERVATIVE    → 1% risk/trade  → Capital safe
```

### Position Size Adjusts Automatically

```python
# You don't do this manually anymore
# place_order(symbol, "BUY", volume=0.5)  # Old way - fixed size

# Now the bot does this:
# place_order(symbol, "BUY")  # New way - automatic sizing
# → Calculates: $20 × 20% / ATR ÷ pip_value
# → Adjusts for: tier, volatility, drawdown protection
```

## The 5 Tiers

| Balance | Tier | Risk/Trade | Multiplier | Goal |
|---------|------|-----------|-----------|------|
| $10-30 | 🟥 Ultra Aggressive | 20% | 2.0x | Fast scaling |
| $30-100 | 🟧 Very Aggressive | 10% | 1.5x | Growth phase |
| $100-500 | 🟨 Aggressive | 5% | 1.0x | Balanced |
| $500-1000 | 🟦 Moderate | 3% | 0.7x | Lock gains |
| $1000+ | 🟩 Conservative | 1% | 0.5x | Preserve capital |

## Equity Step Lock (Profit Protection)

```
Peak: $100
↓ (trading)
Current: $80 (20% down from peak)
```

**Risk Ladder Action:**
- 20% > 15% limit → Risk reduced from 5% → 2.5%
- Account recovers to $85 (15% from peak) → Risk restored

**Result:** Your profits are "locked in" at each milestone. Can't easily give them back.

## Two Sizing Methods

### Method 1: Simple (Equity-Based)
```
lot_size = (risk% × equity) / 100

$100 account, 5% risk = 0.05 lots
$500 account, 5% risk = 0.25 lots
```

### Method 2: ATR-Based (Smart - Recommended)
```
lot_size = (risk% × equity) / (ATR × pip_value)

High volatility (ATR=3.0) → Smaller lot
Low volatility (ATR=0.5)  → Larger lot
Risk stays CONSTANT
```

## Real Example: Starting with $20

```python
from src.risk_ladder import RiskLadder, PositionSizer

# Setup (done automatically in TradingBot)
ladder = RiskLadder(initial_balance=20.0)
sizer = PositionSizer(ladder, pip_value=10)

# Trade 1: Calculate position
sizing = sizer.calculate(equity=20.0, atr=1.5)
print(sizing['lot_size'])      # 0.13L
print(sizing['risk_pct'])      # 20%
print(sizing['tier'])          # ULTRA_AGGRESSIVE

# Win: +$10 → Balance now $30
ladder.update_balance(30.0)

# Trade 2: Tier automatically changes
sizing = sizer.calculate(equity=30.0, atr=1.5)
print(sizing['lot_size'])      # 0.20L (larger, but risk %)
print(sizing['risk_pct'])      # 10% (reduced automatically!)
print(sizing['tier'])          # VERY_AGGRESSIVE

# Win: +$20 → Balance now $50
ladder.update_balance(50.0)

# Peak reached at $75, then drawdown to $62 (17% down)
ladder.update_balance(75.0)
ladder.update_balance(62.0)

# Risk is now reduced
sizing = sizer.calculate(equity=62.0, atr=1.5)
print(sizing['risk_pct'])      # 2.5% (5% ÷ 2, step protection active)
```

## Monitoring

### Print Status
```python
risk_ladder.print_status()

# Output:
# ============================================================
# RISK LADDER STATUS
# ============================================================
# Current Balance: $75.00
# Peak Balance: $100.00
# 
# Current Tier: AGGRESSIVE
# Balance Range: $100 - $500
# Risk Per Trade: 5.0%
# 
# Equity Step Protection:
#   Step Drawdown: 25.0% (limit: 15.0%)
#   Protected: ✗ NO - Risk reduced
# ============================================================
```

### Check Tier
```python
tier = risk_ladder.get_tier_summary()
print(f"{tier['tier']}: {tier['risk_per_trade']}% risk")
# Output: AGGRESSIVE: 5.0% risk
```

## Common Scenarios

### "I started with $10 and want to know if I'm being sized correctly"

```python
# Enable debug logging in logs/trading.log
# Look for lines like:
# "Dynamic sizing: 0.13L (tier: ULTRA_AGGRESSIVE, risk: 20%)"
# "Risk tier changed: ULTRA_AGGRESSIVE → VERY_AGGRESSIVE"
```

### "My balance went up but lot size didn't increase"

This is normal between tier boundaries. Example:
- $45 balance, 5-tier system with $100 threshold
- Still in VERY_AGGRESSIVE tier (10% risk)
- Lot size stays constant until you hit $100

### "I want more aggressive sizing early on"

Edit `strategy.yml`:
```yaml
growth_mode:
  high_aggression_below: 50  # Changed from 30
```

This keeps 20% risk up to $50 instead of $30.

## Integration with TradingBot

It's automatic! Just run:

```bash
python trade.py --symbols "Volatility 75 Index"
```

The bot will:
- Detect growth_mode is enabled
- Initialize Risk Ladder on startup
- Auto-size every trade
- Reduce risk on tier changes
- Protect profits at each step
- Print full status on shutdown

## Disable It (If Needed)

In `strategy.yml`:
```yaml
growth_mode:
  enabled: false
```

Falls back to fixed lot sizing (from `risk_management.position_size`).

## Best Practices

✅ **Do:**
- Trust the automatic tier transitions
- Use ATR-based sizing (handles volatility)
- Monitor logs for tier changes
- Start with small accounts ($10-$50)

❌ **Don't:**
- Manually override lot sizes
- Disable growth mode (loses compounding)
- Change equity_step_size mid-way
- Ignore equity step drawdown warnings

## Troubleshooting

**Q: Lot sizes are too small**
- Increase risk percentages in tiers
- Switch to ATR-based sizing
- Check pip_value setting

**Q: Tier not changing despite growth**
- Check current balance in logs
- Verify growth_mode.enabled = true
- Restart bot with fresh config

**Q: Risk keeps getting cut in half**
- This is equity step protection
- Account drawdown > 15% from peak
- Risk restores when you recover above 15%

## Next Steps

1. Enable growth_mode in `config/strategy.yml`
2. Start bot: `python trade.py`
3. Check logs for tier changes and position sizing
4. Watch account grow with automatic risk scaling

---

**Result:** Exponential growth early, protected compounding later = sustainable trading.

Eden v1.0.0 - Risk Ladder