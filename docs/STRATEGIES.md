# Strategies

## Volatility Burst v1.3
- Squeeze detection: Bollinger Bands inside Keltner Channels
- Breakout: close outside BB
- Confidence: breakout strength, body size vs ATR, volume expansion, EMA momentum (non-linear boost)
- Exits: ATR TP/SL, trail to BE @ +0.8R, max bars
- Config: config/volatility_burst.yml

## MA v1.2
- Entry: MA(3) cross MA(10)
- Exits: ATR TP/SL, trail to BE, max bars
- Config: config/ma_v1_2.yml

## UltraSmall Mode (V75)
- 1 trade/day, conf ≥ 0.97, session gating, ATR top-quantile, strict $ risk caps
- Grid search: scripts/ultra_small_mode.py
