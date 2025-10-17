from __future__ import annotations
from typing import Optional


def combine_stage_outputs(P_liq: float, P_cont: float, htf_bias: int, regime_tag: str = "normal", base_conf: float = 0.5) -> tuple[float, float]:
    """Combine stage outputs into final trade probability and risk multiplier.
    Very lightweight meta-model to keep runtime minimal.
    """
    regime_mult = {
        'low_vol': 0.7,
        'normal': 1.0,
        'high_vol': 0.8,
        'mania': 0.6,
    }.get(regime_tag or 'normal', 1.0)

    bias_boost = 1.15 if htf_bias != 0 else 0.9
    final_p = max(0.0, min(1.0, 0.4 * P_liq + 0.4 * P_cont + 0.2 * base_conf))
    final_p *= bias_boost
    final_p = max(0.0, min(1.0, final_p))

    # Risk scaling: map probability and regime to [0..1.5]
    if final_p >= 0.85:
        risk_mult = 1.2
    elif final_p >= 0.7:
        risk_mult = 0.8
    elif final_p >= 0.6:
        risk_mult = 0.5
    else:
        risk_mult = 0.0  # below threshold, prefer skip
    risk_mult *= regime_mult
    return float(final_p), float(risk_mult)