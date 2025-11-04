# Configuration

## VB v1.3 (config/volatility_burst.yml)
- risk: confidence_threshold, max_trades_per_day, max_bars_in_trade
- indicators: atr, bollinger_bands, keltner_channels, ema_momentum
- entry_exit: tp_atr_multiplier, sl_atr_multiplier, trail_trigger_r
- confidence weights and optional boost (confidence_boost_alpha)

## MA v1.2 (config/ma_v1_2.yml)
- indicators: atr period, MA fast/slow
- entry_exit: tp/sl multipliers, trail trigger

## Overrides
- Many scripts accept parameter overrides (see update_params in strategies)
- Optimizers write best overrides to reports/.