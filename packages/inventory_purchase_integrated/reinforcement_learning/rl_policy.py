"""Lightweight Q-table RL challenger for replenishment decisions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from inventory_purchase_integrated.decision.action_space import ACTION_NAMES
from inventory_purchase_integrated.package_spec import OUTPUT_DATA_DIR
from inventory_purchase_integrated.reinforcement_learning.rl_environment import (
    ReplenishmentDecisionEnv,
)

RISK_SCORE_FILENAME = "04_risk_score.csv"
CANDIDATE_ORDERS_FILENAME = "05_candidate_orders.csv"
CANDIDATE_SIMULATION_RESULT_FILENAME = "candidate_simulation_result.csv"
RL_TRAINING_LOG_FILENAME = "10_rl_training_log.csv"
RL_DECISION_TRACE_FILENAME = "11_rl_decision_trace.csv"

SIMULATION_DATA_DIR = OUTPUT_DATA_DIR.parent / "simulation"
DEFAULT_RISK_SCORE_PATH = OUTPUT_DATA_DIR / RISK_SCORE_FILENAME
DEFAULT_CANDIDATE_ORDERS_PATH = OUTPUT_DATA_DIR / CANDIDATE_ORDERS_FILENAME
DEFAULT_CANDIDATE_SIMULATION_RESULT_PATH = (
    SIMULATION_DATA_DIR / CANDIDATE_SIMULATION_RESULT_FILENAME
)
DEFAULT_RL_TRAINING_LOG_PATH = OUTPUT_DATA_DIR / RL_TRAINING_LOG_FILENAME
DEFAULT_RL_DECISION_TRACE_PATH = OUTPUT_DATA_DIR / RL_DECISION_TRACE_FILENAME

TRAINING_LOG_COLUMNS = (
    "episode",
    "step_no",
    "sku_id",
    "action_id",
    "action_name",
    "epsilon",
    "total_cost",
    "raw_reward",
    "reward",
    "simulation_status",
    "terminated",
    "episode_total_reward",
    "episode_total_cost",
)

RL_DECISION_TRACE_COLUMNS = (
    "snapshot_week",
    "sku_id",
    "warehouse_id",
    "policy_name",
    "selected_action_id",
    "selected_action_name",
    "recommended_order_qty",
    "selected_total_cost",
    "raw_reward",
    "reward",
    "primary_risk_type",
    "q_value",
    "action_rank",
    "rl_reason",
)

POLICY_NAME = "rl_lightweight"


@dataclass(frozen=True)
class LightweightRLConfig:
    episodes: int = 200
    learning_rate: float = 0.1
    gamma: float = 0.0
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    random_seed: int = 42


def _epsilon_for_episode(episode: int, config: LightweightRLConfig) -> float:
    if config.episodes <= 1:
        return config.epsilon_end
    progress = (episode - 1) / (config.episodes - 1)
    epsilon = config.epsilon_start + progress * (config.epsilon_end - config.epsilon_start)
    return max(config.epsilon_end, float(epsilon))


def _choose_action(
    q_values: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(0, len(q_values)))
    return int(np.argmax(q_values))


def train_lightweight_rl_challenger(
    config: LightweightRLConfig | None = None,
    risk_score_path: Path | None = None,
    simulation_result_path: Path | None = None,
    candidate_orders_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train a Q-table challenger and return training logs plus final RL choices."""
    cfg = config or LightweightRLConfig()
    env = ReplenishmentDecisionEnv(
        risk_score_path=risk_score_path,
        simulation_result_path=simulation_result_path,
    )
    rng = np.random.default_rng(cfg.random_seed)

    q_table = {
        row.sku_id: np.zeros(env.action_space.n, dtype=float)
        for row in env.items.itertuples(index=False)
    }
    training_rows: list[dict[str, object]] = []

    for episode in range(1, cfg.episodes + 1):
        env.reset()
        epsilon = _epsilon_for_episode(episode, cfg)
        episode_total_reward = 0.0
        episode_total_cost = 0.0
        episode_rows: list[dict[str, object]] = []
        terminated = False
        step_no = 0

        while not terminated:
            step_no += 1
            current_item = env.items.iloc[env.current_index]
            sku_id = current_item["sku_id"]
            action_id = _choose_action(q_table[sku_id], epsilon, rng)
            _, reward, terminated, _, info = env.step(action_id)

            old_value = q_table[sku_id][action_id]
            td_target = reward
            if cfg.gamma:
                td_target += cfg.gamma * float(np.max(q_table[sku_id]))
            q_table[sku_id][action_id] = old_value + cfg.learning_rate * (
                td_target - old_value
            )

            episode_total_reward += reward
            episode_total_cost += float(info["total_cost"])
            episode_rows.append(
                {
                    "episode": episode,
                    "step_no": step_no,
                    "sku_id": info["sku_id"],
                    "action_id": info["action_id"],
                    "action_name": info["action_name"],
                    "epsilon": epsilon,
                    "total_cost": info["total_cost"],
                    "raw_reward": info["raw_reward"],
                    "reward": reward,
                    "simulation_status": info["simulation_status"],
                    "terminated": terminated,
                    "episode_total_reward": 0.0,
                    "episode_total_cost": 0.0,
                }
            )

        for row in episode_rows:
            row["episode_total_reward"] = episode_total_reward
            row["episode_total_cost"] = episode_total_cost
        training_rows.extend(episode_rows)

    training_log = pd.DataFrame(training_rows, columns=TRAINING_LOG_COLUMNS)
    decision_trace = _build_rl_decision_trace(
        q_table=q_table,
        risk_score_path=risk_score_path or DEFAULT_RISK_SCORE_PATH,
        simulation_result_path=simulation_result_path
        or DEFAULT_CANDIDATE_SIMULATION_RESULT_PATH,
        candidate_orders_path=candidate_orders_path or DEFAULT_CANDIDATE_ORDERS_PATH,
    )
    return training_log, decision_trace


def _build_rl_decision_trace(
    q_table: dict[str, np.ndarray],
    risk_score_path: Path,
    simulation_result_path: Path,
    candidate_orders_path: Path,
) -> pd.DataFrame:
    risk = pd.read_csv(risk_score_path, encoding="utf-8-sig")
    simulation = pd.read_csv(simulation_result_path, encoding="utf-8-sig")
    candidates = pd.read_csv(candidate_orders_path, encoding="utf-8-sig")

    rows: list[dict[str, object]] = []
    for risk_row in risk.sort_values(["sku_id", "warehouse_id"]).itertuples(index=False):
        sku_id = risk_row.sku_id
        warehouse_id = risk_row.warehouse_id
        q_values = q_table[sku_id]
        eligible = simulation[
            (simulation["sku_id"] == sku_id)
            & (simulation["warehouse_id"] == warehouse_id)
            & (simulation["overall_gate_status"] == "PASS")
            & (simulation["simulation_status"] == "SIMULATED")
        ].copy()
        eligible["q_value"] = eligible["action_id"].map(lambda action_id: q_values[int(action_id)])
        eligible = eligible.sort_values(["q_value", "total_cost"], ascending=[False, True])
        selected = eligible.iloc[0]

        ranked_action_ids = list(np.argsort(-q_values))
        action_rank = ranked_action_ids.index(int(selected["action_id"])) + 1
        candidate = candidates[
            (candidates["sku_id"] == sku_id)
            & (candidates["warehouse_id"] == warehouse_id)
            & (candidates["action_id"] == selected["action_id"])
            & (candidates["action_name"] == selected["action_name"])
        ].iloc[0]

        rows.append(
            {
                "snapshot_week": risk_row.snapshot_week,
                "sku_id": sku_id,
                "warehouse_id": warehouse_id,
                "policy_name": POLICY_NAME,
                "selected_action_id": int(selected["action_id"]),
                "selected_action_name": selected["action_name"],
                "recommended_order_qty": candidate["candidate_order_qty"],
                "selected_total_cost": selected["total_cost"],
                "raw_reward": -float(selected["total_cost"]),
                "reward": -float(selected["total_cost"]) / 1000.0,
                "primary_risk_type": risk_row.primary_risk_type,
                "q_value": selected["q_value"],
                "action_rank": action_rank,
                "rl_reason": _rl_reason(
                    sku_id=sku_id,
                    primary_risk_type=risk_row.primary_risk_type,
                    selected_action_name=selected["action_name"],
                ),
            }
        )

    return pd.DataFrame(rows, columns=RL_DECISION_TRACE_COLUMNS)


def _rl_reason(sku_id: str, primary_risk_type: str, selected_action_name: str) -> str:
    if selected_action_name == "order_2w_cover":
        return (
            "RL challenger avoided gate-failed alternatives and selected order_2w_cover "
            "only where learned reward remained highest."
        )
    if primary_risk_type == "expiry" and selected_action_name in {"hold", "reduce_review"}:
        return (
            "RL challenger selected a conservative action for expiry-risk SKU due to "
            "cost/reward outcome."
        )
    return (
        f"RL challenger selected {selected_action_name} because it produced the highest "
        f"learned reward for {sku_id}."
    )


def write_lightweight_rl_outputs(
    training_log_path: Path | None = None,
    decision_trace_path: Path | None = None,
    config: LightweightRLConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train the lightweight challenger and write its log and decision trace."""
    log_path = training_log_path or DEFAULT_RL_TRAINING_LOG_PATH
    trace_path = decision_trace_path or DEFAULT_RL_DECISION_TRACE_PATH
    training_log, decision_trace = train_lightweight_rl_challenger(config=config)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    training_log.to_csv(log_path, index=False, encoding="utf-8-sig")
    decision_trace.to_csv(trace_path, index=False, encoding="utf-8-sig")
    return training_log, decision_trace
