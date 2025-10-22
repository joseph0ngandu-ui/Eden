from __future__ import annotations

# Minimal PPO-like selector placeholder.
# If stable_baselines3 is available, this can be replaced with a trained PPO agent.


def select_strategy_weights(perf_weights: dict[str, float]) -> dict[str, float]:
    """Given preliminary performance-derived weights, slightly amplify
    top performers and dampen laggards to mimic a learned policy.
    """
    if not perf_weights:
        return perf_weights
    # Rank strategies
    ranked = sorted(perf_weights.items(), key=lambda kv: kv[1], reverse=True)
    out = {}
    for i, (name, w) in enumerate(ranked):
        boost = 1.15 if i == 0 else (1.05 if i == 1 else 1.0)
        out[name] = max(0.05, w * boost)
    return out
