"""Cost calculations for digital twin candidate simulations."""

from __future__ import annotations


CONSTRAINT_VIOLATION_PENALTY = 999999.0


def purchase_cost(candidate_order_qty: float, unit_cost: float) -> float:
    return candidate_order_qty * unit_cost


def holding_cost(ending_inventory_qty: float, unit_cost: float) -> float:
    return ending_inventory_qty * unit_cost * 0.02


def stockout_penalty(stockout_qty: float, unit_price: float) -> float:
    return stockout_qty * unit_price * 1.5


def expiry_penalty(expired_qty: float, unit_cost: float) -> float:
    return expired_qty * unit_cost


def expedite_cost(candidate_order_qty: float, unit_cost: float, action_name: str) -> float:
    if action_name != "expedite":
        return 0.0
    return candidate_order_qty * unit_cost * 0.15


def overstock_penalty(
    ending_inventory_qty: float,
    forecast_4w_total_qty: float,
    unit_cost: float,
) -> float:
    excess_qty = max(0.0, ending_inventory_qty - forecast_4w_total_qty)
    return excess_qty * unit_cost * 0.1


def total_cost(
    purchase: float,
    holding: float,
    stockout: float,
    expiry: float,
    expedite: float,
    overstock: float,
    constraint_violation: float,
) -> float:
    return purchase + holding + stockout + expiry + expedite + overstock + constraint_violation
