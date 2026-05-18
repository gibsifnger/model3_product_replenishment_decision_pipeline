"""HGB forecast plus simulation final action policy."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from inventory_purchase_integrated.decision.rule_based_policy import (
    FINAL_DECISION_COLUMNS,
    select_rule_based_decisions,
)
from inventory_purchase_integrated.package_spec import OUTPUT_DATA_DIR

RISK_SCORE_FILENAME = "04_risk_score.csv"
GATE_RESULT_FILENAME = "06_gate_result.csv"
CANDIDATE_SIMULATION_RESULT_FILENAME = "candidate_simulation_result.csv"
FINAL_DECISION_FILENAME = "07_final_decision.csv"

SIMULATION_DATA_DIR = OUTPUT_DATA_DIR.parent / "simulation"
DEFAULT_RISK_SCORE_PATH = OUTPUT_DATA_DIR / RISK_SCORE_FILENAME
DEFAULT_GATE_RESULT_PATH = OUTPUT_DATA_DIR / GATE_RESULT_FILENAME
DEFAULT_CANDIDATE_SIMULATION_RESULT_PATH = (
    SIMULATION_DATA_DIR / CANDIDATE_SIMULATION_RESULT_FILENAME
)
DEFAULT_FINAL_DECISION_PATH = OUTPUT_DATA_DIR / FINAL_DECISION_FILENAME

POLICY_NAME = "hgb_simulation"


def _required_columns(frame: pd.DataFrame, columns: set[str], dataset_name: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"{dataset_name} is missing required columns: {missing_text}")


def _read_inputs(
    risk_score_path: Path,
    gate_result_path: Path,
    simulation_result_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    risk = pd.read_csv(risk_score_path, encoding="utf-8-sig")
    gate = pd.read_csv(gate_result_path, encoding="utf-8-sig")
    simulation = pd.read_csv(simulation_result_path, encoding="utf-8-sig")

    _required_columns(
        risk,
        {"sku_id", "warehouse_id", "primary_risk_type", "expiry_risk_score"},
        "risk_score",
    )
    _required_columns(
        gate,
        {"sku_id", "warehouse_id", "action_id", "action_name", "overall_gate_status"},
        "gate_result",
    )
    _required_columns(
        simulation,
        {
            "snapshot_week",
            "sku_id",
            "warehouse_id",
            "action_id",
            "action_name",
            "candidate_order_qty",
            "overall_gate_status",
            "simulation_status",
            "total_forecast_qty",
            "total_stockout_qty",
            "total_expired_qty",
            "ending_inventory_qty",
            "total_cost",
        },
        "candidate_simulation_result",
    )
    return risk, gate, simulation


def _eligible_simulations(simulation: pd.DataFrame, gate: pd.DataFrame) -> pd.DataFrame:
    gate_pass_keys = gate[gate["overall_gate_status"] == "PASS"][
        ["sku_id", "warehouse_id", "action_id", "action_name"]
    ].drop_duplicates()
    eligible = simulation.merge(
        gate_pass_keys,
        on=["sku_id", "warehouse_id", "action_id", "action_name"],
        how="inner",
    )
    return eligible[
        (eligible["overall_gate_status"] == "PASS")
        & (eligible["simulation_status"] == "SIMULATED")
    ].copy()


def _conservative_expiry_selection(rows: pd.DataFrame, risk_row: pd.Series) -> tuple[pd.Series, str]:
    cost_winner = rows.sort_values(["total_cost", "action_id"]).iloc[0]
    if risk_row["primary_risk_type"] != "expiry" or risk_row["expiry_risk_score"] < 0.70:
        return cost_winner, "Selected lowest total_cost among gate-passed simulated candidates"

    hold_rows = rows[rows["action_name"] == "hold"]
    order_moq_rows = rows[rows["action_name"] == "order_moq"]
    conservative_rows = rows[rows["action_name"].isin(["hold", "reduce_review"])]
    if hold_rows.empty or order_moq_rows.empty or conservative_rows.empty:
        return cost_winner, "Selected lowest total_cost; conservative expiry tie-breaker unavailable"

    hold_cost = float(hold_rows.iloc[0]["total_cost"])
    order_moq_cost = float(order_moq_rows.iloc[0]["total_cost"])
    if hold_cost <= 0:
        savings_rate = 0.0
    else:
        savings_rate = (hold_cost - order_moq_cost) / hold_cost

    if savings_rate < 0.20:
        selected = conservative_rows.sort_values(["total_cost", "action_id"]).iloc[0]
        return (
            selected,
            "Expiry risk >= 0.7 and order_moq savings versus hold below 20%; selected conservative action",
        )
    return cost_winner, "Selected lowest total_cost; order_moq savings exceeded expiry tie-breaker threshold"


def _decision_row(selected: pd.Series, primary_risk_type: str, reason: str) -> dict[str, object]:
    return {
        "snapshot_week": selected["snapshot_week"],
        "sku_id": selected["sku_id"],
        "warehouse_id": selected["warehouse_id"],
        "policy_name": POLICY_NAME,
        "selected_action_id": selected["action_id"],
        "selected_action_name": selected["action_name"],
        "recommended_order_qty": selected["candidate_order_qty"],
        "selected_total_cost": selected["total_cost"],
        "total_forecast_qty": selected["total_forecast_qty"],
        "total_stockout_qty": selected["total_stockout_qty"],
        "total_expired_qty": selected["total_expired_qty"],
        "ending_inventory_qty": selected["ending_inventory_qty"],
        "primary_risk_type": primary_risk_type,
        "selection_reason": reason,
    }


def select_hgb_simulation_decisions(
    simulation_result: pd.DataFrame,
    risk_score: pd.DataFrame,
    gate_result: pd.DataFrame,
) -> pd.DataFrame:
    """Select one action per sku_id and warehouse_id from simulated gate-passed candidates."""
    eligible = _eligible_simulations(simulation_result, gate_result)
    risk_by_key = {
        (row.sku_id, row.warehouse_id): pd.Series(row._asdict())
        for row in risk_score.itertuples(index=False)
    }

    decisions: list[dict[str, object]] = []
    for key, rows in eligible.groupby(["sku_id", "warehouse_id"], sort=True):
        risk_row = risk_by_key.get(
            key,
            pd.Series({"primary_risk_type": "stable", "expiry_risk_score": 0.0}),
        )
        selected, reason = _conservative_expiry_selection(rows, risk_row)
        decisions.append(_decision_row(selected, risk_row["primary_risk_type"], reason))

    return pd.DataFrame(decisions, columns=FINAL_DECISION_COLUMNS)


def build_final_decisions(
    risk_score_path: Path | None = None,
    gate_result_path: Path | None = None,
    simulation_result_path: Path | None = None,
) -> pd.DataFrame:
    """Build rule-based and HGB simulation policy decisions."""
    risk_path = risk_score_path or DEFAULT_RISK_SCORE_PATH
    gate_path = gate_result_path or DEFAULT_GATE_RESULT_PATH
    simulation_path = simulation_result_path or DEFAULT_CANDIDATE_SIMULATION_RESULT_PATH

    risk, gate, simulation = _read_inputs(risk_path, gate_path, simulation_path)
    rule_based = select_rule_based_decisions(simulation, risk)
    hgb_simulation = select_hgb_simulation_decisions(simulation, risk, gate)
    return pd.concat([rule_based, hgb_simulation], ignore_index=True).loc[
        :,
        FINAL_DECISION_COLUMNS,
    ]


def write_final_decisions(
    output_path: Path | None = None,
    risk_score_path: Path | None = None,
    gate_result_path: Path | None = None,
    simulation_result_path: Path | None = None,
) -> pd.DataFrame:
    """Build and write stage-9 final policy decisions."""
    path = output_path or DEFAULT_FINAL_DECISION_PATH
    decisions = build_final_decisions(
        risk_score_path=risk_score_path,
        gate_result_path=gate_result_path,
        simulation_result_path=simulation_result_path,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    decisions.to_csv(path, index=False, encoding="utf-8-sig")
    return decisions
