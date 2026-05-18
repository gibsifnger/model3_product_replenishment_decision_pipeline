"""Reward model for RL environment smoke tests.

This module only converts existing candidate simulation outcomes into rewards.
It does not train an RL policy.
"""

from __future__ import annotations

REWARD_SCALE = 1000.0
GATE_FAIL_RAW_REWARD = -999999.0


def compute_reward_components(
    total_cost: float,
    gate_passed: bool,
    reward_scale: float = REWARD_SCALE,
) -> dict[str, float]:
    """Return raw and scaled reward components from precomputed simulation cost."""
    if not gate_passed:
        raw_reward = GATE_FAIL_RAW_REWARD
    else:
        raw_reward = -float(total_cost)

    scaled_reward = raw_reward / reward_scale
    return {
        "raw_reward": raw_reward,
        "scaled_reward": scaled_reward,
        "reward": scaled_reward,
        "reward_scale": reward_scale,
    }


def compute_candidate_reward(
    total_cost: float,
    gate_passed: bool,
    reward_scale: float = REWARD_SCALE,
) -> float:
    """Return the final scaled reward used by the smoke-test environment."""
    return compute_reward_components(
        total_cost=total_cost,
        gate_passed=gate_passed,
        reward_scale=reward_scale,
    )["reward"]
