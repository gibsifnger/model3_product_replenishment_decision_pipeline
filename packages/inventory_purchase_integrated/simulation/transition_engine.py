"""Inventory transition engine for weekly candidate simulations."""

from __future__ import annotations

from math import ceil

import pandas as pd

from inventory_purchase_integrated.simulation.cost_model import (
    expiry_penalty,
    holding_cost,
    stockout_penalty,
)


def arrival_week_offset(lead_time_days: float, action_name: str) -> int:
    if action_name == "expedite":
        return 1
    return max(1, int(ceil(lead_time_days / 7)))


def estimate_expired_qty(
    shelf_life_days: float,
    ending_inventory_before_expiry: float,
    avg_sales_4w: float,
) -> float:
    if shelf_life_days <= 21 and ending_inventory_before_expiry > avg_sales_4w:
        return max(0.0, (ending_inventory_before_expiry - avg_sales_4w) * 0.25)
    return 0.0


def simulate_weekly_transitions(
    candidate: pd.Series,
    forecast_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Simulate four weekly transitions for one gate-passed candidate."""
    beginning_inventory = float(candidate["on_hand_qty"])
    existing_inbound_qty = float(candidate["inbound_qty"])
    candidate_order_qty = float(candidate["candidate_order_qty"])
    candidate_arrival_offset = arrival_week_offset(
        float(candidate["lead_time_days"]),
        str(candidate["action_name"]),
    )

    rows: list[dict[str, object]] = []
    for forecast in forecast_rows.sort_values("forecast_horizon_week").itertuples(index=False):
        simulation_week = int(forecast.forecast_horizon_week)
        inbound_arrival_qty = 0.0
        if simulation_week == 1:
            inbound_arrival_qty += existing_inbound_qty
        if simulation_week == candidate_arrival_offset:
            inbound_arrival_qty += candidate_order_qty

        forecast_qty = float(forecast.forecast_qty)
        available_before_demand = beginning_inventory + inbound_arrival_qty
        sales_fulfilled_qty = min(available_before_demand, forecast_qty)
        stockout_qty = max(0.0, forecast_qty - available_before_demand)
        ending_inventory_before_expiry = max(0.0, available_before_demand - forecast_qty)
        expired_qty = estimate_expired_qty(
            float(candidate["shelf_life_days"]),
            ending_inventory_before_expiry,
            float(candidate["avg_sales_4w"]),
        )
        ending_inventory_qty = max(0.0, ending_inventory_before_expiry - expired_qty)

        rows.append(
            {
                "snapshot_week": candidate["snapshot_week"],
                "sku_id": candidate["sku_id"],
                "warehouse_id": candidate["warehouse_id"],
                "action_id": candidate["action_id"],
                "action_name": candidate["action_name"],
                "simulation_week": simulation_week,
                "target_week": forecast.target_week,
                "beginning_inventory_qty": beginning_inventory,
                "inbound_arrival_qty": inbound_arrival_qty,
                "forecast_qty": forecast_qty,
                "sales_fulfilled_qty": sales_fulfilled_qty,
                "stockout_qty": stockout_qty,
                "expired_qty": expired_qty,
                "ending_inventory_qty": ending_inventory_qty,
                "weekly_holding_cost": holding_cost(ending_inventory_qty, float(candidate["unit_cost"])),
                "weekly_stockout_penalty": stockout_penalty(
                    stockout_qty,
                    float(candidate["unit_price"]),
                ),
                "weekly_expiry_penalty": expiry_penalty(expired_qty, float(candidate["unit_cost"])),
            }
        )
        beginning_inventory = ending_inventory_qty

    return pd.DataFrame(rows)
