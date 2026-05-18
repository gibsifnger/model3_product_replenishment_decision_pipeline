"""Rule-based final action selection from precomputed candidate simulations."""

from __future__ import annotations

import pandas as pd

POLICY_NAME = "rule_based"

FINAL_DECISION_COLUMNS = (
    "snapshot_week",
    "sku_id",
    "warehouse_id",
    "policy_name",
    "selected_action_id",
    "selected_action_name",
    "recommended_order_qty",
    "selected_total_cost",
    "total_forecast_qty",
    "total_stockout_qty",
    "total_expired_qty",
    "ending_inventory_qty",
    "primary_risk_type",
    "selection_reason",
)


def _candidate_pool(rows: pd.DataFrame, primary_risk_type: str) -> pd.DataFrame:
    if primary_risk_type == "expiry":
        preferred_actions = ("hold", "reduce_review")
    elif primary_risk_type == "stockout":
        preferred_actions = ("order_moq", "order_1w_cover", "expedite")
    else:
        preferred_actions = ("hold", "order_moq", "order_1w_cover", "reduce_review")

    pool = rows[rows["action_name"].isin(preferred_actions)]
    if pool.empty:
        pool = rows[rows["action_name"] == "hold"]
    if pool.empty:
        pool = rows
    return pool


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


def select_rule_based_decisions(
    simulation_result: pd.DataFrame,
    risk_score: pd.DataFrame,
) -> pd.DataFrame:
    """Select one action per sku_id and warehouse_id using rule-based preferences."""
    risk_lookup = risk_score.set_index(["sku_id", "warehouse_id"])["primary_risk_type"].to_dict()
    eligible = simulation_result[
        (simulation_result["overall_gate_status"] == "PASS")
        & (simulation_result["simulation_status"] == "SIMULATED")
    ].copy()

    decisions: list[dict[str, object]] = []
    for key, rows in eligible.groupby(["sku_id", "warehouse_id"], sort=True):
        primary_risk_type = risk_lookup.get(key, "stable")
        pool = _candidate_pool(rows, primary_risk_type)
        selected = pool.sort_values(["total_cost", "action_id"]).iloc[0]
        reason = (
            f"{primary_risk_type} risk rule considered preferred actions and selected lowest total_cost"
        )
        decisions.append(_decision_row(selected, primary_risk_type, reason))

    return pd.DataFrame(decisions, columns=FINAL_DECISION_COLUMNS)
