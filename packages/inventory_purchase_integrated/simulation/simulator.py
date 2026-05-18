"""Digital twin simulator for candidate replenishment actions."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from inventory_purchase_integrated.package_spec import DATA_DIR, OUTPUT_DATA_DIR
from inventory_purchase_integrated.simulation.cost_model import (
    CONSTRAINT_VIOLATION_PENALTY,
    expedite_cost,
    overstock_penalty,
    purchase_cost,
    total_cost,
)
from inventory_purchase_integrated.simulation.transition_engine import simulate_weekly_transitions

FEATURE_SNAPSHOT_FILENAME = "02_feature_snapshot.csv"
DEMAND_FORECAST_FILENAME = "03_demand_forecast.csv"
CANDIDATE_ORDERS_FILENAME = "05_candidate_orders.csv"
GATE_RESULT_FILENAME = "06_gate_result.csv"
CANDIDATE_SIMULATION_RESULT_FILENAME = "candidate_simulation_result.csv"
WEEKLY_TRANSITION_LOG_FILENAME = "weekly_transition_log.csv"

SIMULATION_DATA_DIR = DATA_DIR / "simulation"
DEFAULT_FEATURE_SNAPSHOT_PATH = OUTPUT_DATA_DIR / FEATURE_SNAPSHOT_FILENAME
DEFAULT_DEMAND_FORECAST_PATH = OUTPUT_DATA_DIR / DEMAND_FORECAST_FILENAME
DEFAULT_CANDIDATE_ORDERS_PATH = OUTPUT_DATA_DIR / CANDIDATE_ORDERS_FILENAME
DEFAULT_GATE_RESULT_PATH = OUTPUT_DATA_DIR / GATE_RESULT_FILENAME
DEFAULT_CANDIDATE_SIMULATION_RESULT_PATH = (
    SIMULATION_DATA_DIR / CANDIDATE_SIMULATION_RESULT_FILENAME
)
DEFAULT_WEEKLY_TRANSITION_LOG_PATH = SIMULATION_DATA_DIR / WEEKLY_TRANSITION_LOG_FILENAME

CANDIDATE_SIMULATION_COLUMNS = (
    "snapshot_week",
    "sku_id",
    "warehouse_id",
    "action_id",
    "action_name",
    "candidate_order_qty",
    "overall_gate_status",
    "simulation_status",
    "total_forecast_qty",
    "total_sales_fulfilled_qty",
    "total_stockout_qty",
    "total_expired_qty",
    "ending_inventory_qty",
    "purchase_cost",
    "holding_cost",
    "stockout_penalty",
    "expiry_penalty",
    "expedite_cost",
    "overstock_penalty",
    "constraint_violation_penalty",
    "total_cost",
    "simulation_comment",
)

WEEKLY_TRANSITION_COLUMNS = (
    "snapshot_week",
    "sku_id",
    "warehouse_id",
    "action_id",
    "action_name",
    "simulation_week",
    "target_week",
    "beginning_inventory_qty",
    "inbound_arrival_qty",
    "forecast_qty",
    "sales_fulfilled_qty",
    "stockout_qty",
    "expired_qty",
    "ending_inventory_qty",
    "weekly_holding_cost",
    "weekly_stockout_penalty",
    "weekly_expiry_penalty",
)


def _required_columns(frame: pd.DataFrame, columns: set[str], dataset_name: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"{dataset_name} is missing required columns: {missing_text}")


def _read_inputs(
    feature_snapshot_path: Path,
    demand_forecast_path: Path,
    candidate_orders_path: Path,
    gate_result_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    snapshot = pd.read_csv(feature_snapshot_path)
    forecast = pd.read_csv(demand_forecast_path)
    candidates = pd.read_csv(candidate_orders_path, encoding="utf-8-sig")
    gate = pd.read_csv(gate_result_path, encoding="utf-8-sig")

    _required_columns(
        snapshot,
        {
            "snapshot_week",
            "sku_id",
            "warehouse_id",
            "on_hand_qty",
            "inbound_qty",
            "avg_sales_4w",
            "shelf_life_days",
            "unit_cost",
            "unit_price",
            "lead_time_days",
        },
        "feature_snapshot",
    )
    _required_columns(
        forecast,
        {
            "target_week",
            "sku_id",
            "warehouse_id",
            "forecast_horizon_week",
            "forecast_qty",
        },
        "demand_forecast",
    )
    _required_columns(
        candidates,
        {
            "snapshot_week",
            "sku_id",
            "warehouse_id",
            "action_id",
            "action_name",
            "candidate_order_qty",
            "forecast_4w_total_qty",
        },
        "candidate_orders",
    )
    _required_columns(
        gate,
        {
            "sku_id",
            "warehouse_id",
            "action_id",
            "action_name",
            "overall_gate_status",
        },
        "gate_result",
    )
    return snapshot, forecast, candidates, gate


def _simulation_inputs(candidates: pd.DataFrame, snapshot: pd.DataFrame, gate: pd.DataFrame) -> pd.DataFrame:
    snapshot_columns = [
        "sku_id",
        "warehouse_id",
        "avg_sales_4w",
        "shelf_life_days",
        "unit_cost",
        "unit_price",
        "lead_time_days",
    ]
    gate_columns = [
        "sku_id",
        "warehouse_id",
        "action_id",
        "action_name",
        "overall_gate_status",
    ]
    return (
        candidates.merge(snapshot.loc[:, snapshot_columns], on=["sku_id", "warehouse_id"], how="left")
        .merge(gate.loc[:, gate_columns], on=["sku_id", "warehouse_id", "action_id", "action_name"], how="left")
    )


def _failed_candidate_result(candidate: pd.Series) -> dict[str, object]:
    return {
        "snapshot_week": candidate["snapshot_week"],
        "sku_id": candidate["sku_id"],
        "warehouse_id": candidate["warehouse_id"],
        "action_id": candidate["action_id"],
        "action_name": candidate["action_name"],
        "candidate_order_qty": candidate["candidate_order_qty"],
        "overall_gate_status": candidate["overall_gate_status"],
        "simulation_status": "SKIPPED_GATE_FAIL",
        "total_forecast_qty": candidate["forecast_4w_total_qty"],
        "total_sales_fulfilled_qty": 0.0,
        "total_stockout_qty": 0.0,
        "total_expired_qty": 0.0,
        "ending_inventory_qty": candidate["on_hand_qty"],
        "purchase_cost": 0.0,
        "holding_cost": 0.0,
        "stockout_penalty": 0.0,
        "expiry_penalty": 0.0,
        "expedite_cost": 0.0,
        "overstock_penalty": 0.0,
        "constraint_violation_penalty": CONSTRAINT_VIOLATION_PENALTY,
        "total_cost": CONSTRAINT_VIOLATION_PENALTY,
        "simulation_comment": "Gate failed; transition simulation skipped",
    }


def _passed_candidate_result(candidate: pd.Series, transitions: pd.DataFrame) -> dict[str, object]:
    purchase = purchase_cost(float(candidate["candidate_order_qty"]), float(candidate["unit_cost"]))
    holding = float(transitions["weekly_holding_cost"].sum())
    stockout = float(transitions["weekly_stockout_penalty"].sum())
    expiry = float(transitions["weekly_expiry_penalty"].sum())
    expedite = expedite_cost(
        float(candidate["candidate_order_qty"]),
        float(candidate["unit_cost"]),
        str(candidate["action_name"]),
    )
    ending_inventory = float(transitions.iloc[-1]["ending_inventory_qty"])
    overstock = overstock_penalty(
        ending_inventory,
        float(candidate["forecast_4w_total_qty"]),
        float(candidate["unit_cost"]),
    )
    constraint = 0.0

    return {
        "snapshot_week": candidate["snapshot_week"],
        "sku_id": candidate["sku_id"],
        "warehouse_id": candidate["warehouse_id"],
        "action_id": candidate["action_id"],
        "action_name": candidate["action_name"],
        "candidate_order_qty": candidate["candidate_order_qty"],
        "overall_gate_status": candidate["overall_gate_status"],
        "simulation_status": "SIMULATED",
        "total_forecast_qty": float(transitions["forecast_qty"].sum()),
        "total_sales_fulfilled_qty": float(transitions["sales_fulfilled_qty"].sum()),
        "total_stockout_qty": float(transitions["stockout_qty"].sum()),
        "total_expired_qty": float(transitions["expired_qty"].sum()),
        "ending_inventory_qty": ending_inventory,
        "purchase_cost": purchase,
        "holding_cost": holding,
        "stockout_penalty": stockout,
        "expiry_penalty": expiry,
        "expedite_cost": expedite,
        "overstock_penalty": overstock,
        "constraint_violation_penalty": constraint,
        "total_cost": total_cost(purchase, holding, stockout, expiry, expedite, overstock, constraint),
        "simulation_comment": "Candidate simulated over 4-week horizon",
    }


def run_candidate_simulation(
    feature_snapshot_path: Path | None = None,
    demand_forecast_path: Path | None = None,
    candidate_orders_path: Path | None = None,
    gate_result_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate each candidate action without selecting a final action."""
    snapshot_path = feature_snapshot_path or DEFAULT_FEATURE_SNAPSHOT_PATH
    forecast_path = demand_forecast_path or DEFAULT_DEMAND_FORECAST_PATH
    candidate_path = candidate_orders_path or DEFAULT_CANDIDATE_ORDERS_PATH
    gate_path = gate_result_path or DEFAULT_GATE_RESULT_PATH

    snapshot, forecast, candidates, gate = _read_inputs(
        snapshot_path,
        forecast_path,
        candidate_path,
        gate_path,
    )
    simulation_inputs = _simulation_inputs(candidates, snapshot, gate)

    result_rows: list[dict[str, object]] = []
    transition_frames: list[pd.DataFrame] = []
    for candidate_tuple in simulation_inputs.sort_values(
        ["sku_id", "warehouse_id", "action_id"]
    ).itertuples(index=False):
        candidate = pd.Series(candidate_tuple._asdict())
        if candidate["overall_gate_status"] != "PASS":
            result_rows.append(_failed_candidate_result(candidate))
            continue

        forecast_rows = forecast[
            (forecast["sku_id"] == candidate["sku_id"])
            & (forecast["warehouse_id"] == candidate["warehouse_id"])
        ]
        transitions = simulate_weekly_transitions(candidate, forecast_rows)
        transition_frames.append(transitions)
        result_rows.append(_passed_candidate_result(candidate, transitions))

    candidate_result = pd.DataFrame(result_rows, columns=CANDIDATE_SIMULATION_COLUMNS)
    if transition_frames:
        weekly_log = pd.concat(transition_frames, ignore_index=True).loc[
            :,
            WEEKLY_TRANSITION_COLUMNS,
        ]
    else:
        weekly_log = pd.DataFrame(columns=WEEKLY_TRANSITION_COLUMNS)

    return candidate_result, weekly_log


def write_simulation_outputs(
    candidate_result_path: Path | None = None,
    weekly_transition_log_path: Path | None = None,
    feature_snapshot_path: Path | None = None,
    demand_forecast_path: Path | None = None,
    candidate_orders_path: Path | None = None,
    gate_result_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run simulation and write candidate-level and weekly transition outputs."""
    result_path = candidate_result_path or DEFAULT_CANDIDATE_SIMULATION_RESULT_PATH
    log_path = weekly_transition_log_path or DEFAULT_WEEKLY_TRANSITION_LOG_PATH

    candidate_result, weekly_log = run_candidate_simulation(
        feature_snapshot_path=feature_snapshot_path,
        demand_forecast_path=demand_forecast_path,
        candidate_orders_path=candidate_orders_path,
        gate_result_path=gate_result_path,
    )

    result_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_result.to_csv(result_path, index=False, encoding="utf-8-sig")
    weekly_log.to_csv(log_path, index=False, encoding="utf-8-sig")
    return candidate_result, weekly_log
