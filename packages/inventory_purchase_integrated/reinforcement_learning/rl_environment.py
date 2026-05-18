"""Gymnasium environment for replenishment decision smoke tests."""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from inventory_purchase_integrated.decision.action_space import ACTION_NAMES
from inventory_purchase_integrated.package_spec import OUTPUT_DATA_DIR
from inventory_purchase_integrated.reinforcement_learning.reward_model import (
    REWARD_SCALE,
    compute_reward_components,
)

RISK_SCORE_FILENAME = "04_risk_score.csv"
CANDIDATE_SIMULATION_RESULT_FILENAME = "candidate_simulation_result.csv"

SIMULATION_DATA_DIR = OUTPUT_DATA_DIR.parent / "simulation"
DEFAULT_RISK_SCORE_PATH = OUTPUT_DATA_DIR / RISK_SCORE_FILENAME
DEFAULT_CANDIDATE_SIMULATION_RESULT_PATH = (
    SIMULATION_DATA_DIR / CANDIDATE_SIMULATION_RESULT_FILENAME
)

OBSERVATION_COLUMNS = (
    "forecast_4w_total_qty",
    "available_qty",
    "inventory_cover_weeks",
    "inbound_cover_weeks",
    "stockout_risk_score",
    "overstock_risk_score",
    "expiry_risk_score",
    "supplier_risk_score",
    "total_risk_score",
)


class ReplenishmentDecisionEnv(gym.Env):
    """One-episode SKU decision environment backed by existing simulation outputs."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        risk_score_path: Path | None = None,
        simulation_result_path: Path | None = None,
        reward_scale: float = REWARD_SCALE,
    ) -> None:
        super().__init__()
        self.risk_score_path = risk_score_path or DEFAULT_RISK_SCORE_PATH
        self.simulation_result_path = (
            simulation_result_path or DEFAULT_CANDIDATE_SIMULATION_RESULT_PATH
        )
        self.reward_scale = reward_scale
        self.risk_score = pd.read_csv(self.risk_score_path, encoding="utf-8-sig")
        self.simulation_result = pd.read_csv(
            self.simulation_result_path,
            encoding="utf-8-sig",
        )

        self._validate_inputs()
        self.items = self.risk_score.sort_values(["sku_id", "warehouse_id"]).reset_index(
            drop=True
        )
        self.action_names = ACTION_NAMES
        self.action_space = spaces.Discrete(len(self.action_names))
        self.observation_space = spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(len(OBSERVATION_COLUMNS),),
            dtype=np.float32,
        )
        self.current_index = 0
        self.episode_total_reward = 0.0

    def _validate_inputs(self) -> None:
        missing_risk = set(OBSERVATION_COLUMNS + ("sku_id", "warehouse_id")) - set(
            self.risk_score.columns
        )
        if missing_risk:
            missing_text = ", ".join(sorted(missing_risk))
            raise ValueError(f"risk_score is missing required columns: {missing_text}")

        required_simulation_columns = {
            "sku_id",
            "warehouse_id",
            "action_name",
            "overall_gate_status",
            "total_cost",
            "total_stockout_qty",
            "total_expired_qty",
        }
        missing_simulation = required_simulation_columns - set(self.simulation_result.columns)
        if missing_simulation:
            missing_text = ", ".join(sorted(missing_simulation))
            raise ValueError(
                f"candidate_simulation_result is missing required columns: {missing_text}"
            )

    def _observation(self) -> np.ndarray:
        if self.current_index >= len(self.items):
            return np.zeros(len(OBSERVATION_COLUMNS), dtype=np.float32)

        row = self.items.iloc[self.current_index]
        return row.loc[list(OBSERVATION_COLUMNS)].astype(float).to_numpy(dtype=np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.current_index = 0
        self.episode_total_reward = 0.0
        return self._observation(), {"sku_count": len(self.items)}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self.current_index >= len(self.items):
            return self._observation(), 0.0, True, False, {"already_terminated": True}

        action_id = int(action)
        action_name = self.action_names[action_id]
        item = self.items.iloc[self.current_index]
        simulation = self.simulation_result[
            (self.simulation_result["sku_id"] == item["sku_id"])
            & (self.simulation_result["warehouse_id"] == item["warehouse_id"])
            & (self.simulation_result["action_name"] == action_name)
        ]

        if simulation.empty:
            reward = -1000.0
            info = {
                "sku_id": item["sku_id"],
                "warehouse_id": item["warehouse_id"],
                "action_id": action_id,
                "action_name": action_name,
                "total_cost": 0.0,
                "raw_reward": -1000.0,
                "scaled_reward": reward,
                "reward": reward,
                "simulation_status": "MISSING_SIMULATION",
                "message": "simulation row missing",
            }
        else:
            simulation_row = simulation.iloc[0]
            gate_passed = simulation_row["overall_gate_status"] == "PASS"
            reward_components = compute_reward_components(
                total_cost=float(simulation_row["total_cost"]),
                gate_passed=gate_passed,
                reward_scale=self.reward_scale,
            )
            reward = reward_components["reward"]
            info = {
                "sku_id": item["sku_id"],
                "warehouse_id": item["warehouse_id"],
                "action_id": action_id,
                "action_name": action_name,
                "overall_gate_status": simulation_row["overall_gate_status"],
                "total_cost": float(simulation_row["total_cost"]),
                "raw_reward": reward_components["raw_reward"],
                "scaled_reward": reward_components["scaled_reward"],
                "reward": reward,
                "reward_scale": reward_components["reward_scale"],
                "simulation_status": simulation_row["simulation_status"],
            }

        self.episode_total_reward += reward
        self.current_index += 1
        terminated = self.current_index >= len(self.items)
        return self._observation(), reward, terminated, False, info
