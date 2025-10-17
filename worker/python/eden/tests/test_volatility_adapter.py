from eden.risk.volatility_adapter import compute_volatility_factor, VolatilityConfig, adjust_stop_and_size

def test_volatility_factor_basic():
    row = {
        'atr_14': 10.0,
        '1H_atr_14': 5.0,
    }
    cfg = VolatilityConfig()
    vf = compute_volatility_factor(row, cfg)
    assert 1.9 < vf <= cfg.cap_max  # 10/5=2, within cap


def test_adjust_stop_and_size_conservative():
    cfg = VolatilityConfig(conservative_threshold=2.2)
    # vol factor above threshold triggers conservative adjustments
    stop, size, meta = adjust_stop_and_size(10.0, 1.0, 2.5, cfg)
    assert stop > 10.0
    assert size < (1.0 / 2.5)
    assert meta['conservative'] is True