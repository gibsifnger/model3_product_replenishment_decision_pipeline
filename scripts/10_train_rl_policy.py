#!/usr/bin/env python
"""Run RL policy entrypoint for smoke test or lightweight challenger training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGES_DIR = REPO_ROOT / "packages"
if str(PACKAGES_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGES_DIR))

from inventory_purchase_integrated.reinforcement_learning.rl_environment import (
    ReplenishmentDecisionEnv,
)
from inventory_purchase_integrated.reinforcement_learning.rl_policy import (
    LightweightRLConfig,
    write_lightweight_rl_outputs,
)
from inventory_purchase_integrated.reinforcement_learning.reward_model import REWARD_SCALE


def run_smoke_test() -> None:
    env = ReplenishmentDecisionEnv()
    observation, info = env.reset()
    print("env.reset() success")
    print(f"initial_observation_shape: {observation.shape}")
    print(f"sku_count: {info['sku_count']}")
    print(f"reward_formula: raw_reward = -total_cost; reward = raw_reward / {REWARD_SCALE:g}")

    action_sequence = [0, 1, 2]
    terminated = False
    for step_no, action_id in enumerate(action_sequence, start=1):
        observation, reward, terminated, truncated, step_info = env.step(action_id)
        print(
            f"step_no={step_no}: sku_id={step_info['sku_id']}, "
            f"action_id={step_info['action_id']}, "
            f"action={step_info['action_name']}, "
            f"total_cost={step_info['total_cost']:.6f}, "
            f"raw_reward={step_info['raw_reward']:.6f}, "
            f"reward={reward:.6f}, "
            f"simulation_status={step_info['simulation_status']}, "
            f"terminated={terminated}, truncated={truncated}"
        )

    print(f"terminated_after_3_skus: {terminated}")
    print(f"episode_total_reward: {env.episode_total_reward:.6f}")


def run_lightweight_training() -> None:
    config = LightweightRLConfig()
    training_log, decision_trace = write_lightweight_rl_outputs(config=config)

    final_steps = training_log.groupby("episode", as_index=False).tail(1)
    print("\n[rl_training_log] wrote rows to data/output/10_rl_training_log.csv")
    print("training log shape:")
    print(training_log.shape)
    print("\nepisode reward summary:")
    print(final_steps["episode_total_reward"].describe().to_string())
    print("\nRL decision trace:")
    print(decision_trace.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run environment reset and three step checks without training.",
    )
    parser.add_argument(
        "--train-lightweight",
        action="store_true",
        help="Train Q-table epsilon-greedy RL challenger.",
    )
    args = parser.parse_args()

    if args.smoke_test:
        run_smoke_test()
        return
    if args.train_lightweight:
        run_lightweight_training()
        return

    raise SystemExit("Use --smoke-test or --train-lightweight.")


if __name__ == "__main__":
    main()
