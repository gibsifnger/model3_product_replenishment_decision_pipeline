"""Generate replenishment order candidates without choosing a final action."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from inventory_purchase_integrated.decision.action_space import ACTIONS, Action
from inventory_purchase_integrated.package_spec import OUTPUT_DATA_DIR

FEATURE_SNAPSHOT_FILENAME = "02_feature_snapshot.csv"
DEMAND_FORECAST_FILENAME = "03_demand_forecast.csv"
RISK_SCORE_FILENAME = "04_risk_score.csv"
CANDIDATE_ORDERS_FILENAME = "05_candidate_orders.csv"

DEFAULT_FEATURE_SNAPSHOT_PATH = OUTPUT_DATA_DIR / FEATURE_SNAPSHOT_FILENAME
DEFAULT_DEMAND_FORECAST_PATH = OUTPUT_DATA_DIR / DEMAND_FORECAST_FILENAME
DEFAULT_RISK_SCORE_PATH = OUTPUT_DATA_DIR / RISK_SCORE_FILENAME
DEFAULT_CANDIDATE_ORDERS_PATH = OUTPUT_DATA_DIR / CANDIDATE_ORDERS_FILENAME

CANDIDATE_ORDER_COLUMNS = (
    "snapshot_week",
    "sku_id",
    "warehouse_id",
    "action_id",
    "action_name",
    "candidate_order_qty",
    "forecast_1w_qty",
    "forecast_2w_qty",
    "forecast_4w_total_qty",
    "available_qty",
    "on_hand_qty",
    "inbound_qty",
    "moq_qty",
    "box_multiple_qty",
    "stockout_risk_score",
    "overstock_risk_score",
    "expiry_risk_score",
    "primary_risk_type",
    "candidate_reason",
)


def _required_columns(frame: pd.DataFrame, columns: set[str], dataset_name: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"{dataset_name} is missing required columns: {missing_text}")


def _read_inputs(
    feature_snapshot_path: Path,
    demand_forecast_path: Path,
    risk_score_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    snapshot = pd.read_csv(feature_snapshot_path)
    forecast = pd.read_csv(demand_forecast_path)
    risk = pd.read_csv(risk_score_path, encoding="utf-8-sig")

    _required_columns(
        snapshot,
        {
            "snapshot_week",
            "sku_id",
            "warehouse_id",
            "available_qty",
            "on_hand_qty",
            "inbound_qty",
            "moq_qty",
            "box_multiple_qty",
        },
        "feature_snapshot",
    )
    _required_columns(
        forecast,
        {"sku_id", "warehouse_id", "forecast_horizon_week", "forecast_qty"},
        "demand_forecast",
    )
    _required_columns(
        risk,
        {
            "sku_id",
            "warehouse_id",
            "forecast_4w_total_qty",
            "stockout_risk_score",
            "overstock_risk_score",
            "expiry_risk_score",
            "primary_risk_type",
        },
        "risk_score",
    )
    return snapshot, forecast, risk


def _forecast_summary(forecast: pd.DataFrame) -> pd.DataFrame:
    forecast_1w = forecast[forecast["forecast_horizon_week"] == 1].groupby(
        ["sku_id", "warehouse_id"],
        as_index=False,
    ).agg(forecast_1w_qty=("forecast_qty", "sum"))
    forecast_2w = forecast[forecast["forecast_horizon_week"].isin([1, 2])].groupby(
        ["sku_id", "warehouse_id"],
        as_index=False,
    ).agg(forecast_2w_qty=("forecast_qty", "sum"))

    return forecast_1w.merge(forecast_2w, on=["sku_id", "warehouse_id"], how="outer")


def _candidate_qty(action: Action, row: pd.Series) -> int:
    if action.action_name == "hold":
        quantity = 0
    elif action.action_name == "order_moq":
        quantity = row["moq_qty"]
    elif action.action_name == "order_1w_cover":
        quantity = max(0, row["forecast_1w_qty"] - row["available_qty"])
    elif action.action_name == "order_2w_cover":
        quantity = max(0, row["forecast_2w_qty"] - row["available_qty"])
    elif action.action_name == "expedite":
        shortage = max(0, row["forecast_1w_qty"] - row["available_qty"])
        quantity = shortage if row["stockout_risk_score"] >= 0.50 else 0
    elif action.action_name == "reduce_review":
        quantity = 0
    else:
        raise ValueError(f"Unknown action: {action.action_name}")

    return max(0, int(round(quantity)))


def _base_candidate_frame(
    snapshot: pd.DataFrame,
    forecast: pd.DataFrame,
    risk: pd.DataFrame,
) -> pd.DataFrame:
    risk_columns = [
        "sku_id",
        "warehouse_id",
        "forecast_4w_total_qty",
        "stockout_risk_score",
        "overstock_risk_score",
        "expiry_risk_score",
        "primary_risk_type",
    ]
    snapshot_columns = [
        "snapshot_week",
        "sku_id",
        "warehouse_id",
        "available_qty",
        "on_hand_qty",
        "inbound_qty",
        "moq_qty",
        "box_multiple_qty",
    ]

    base = (
        snapshot.loc[:, snapshot_columns]
        .merge(_forecast_summary(forecast), on=["sku_id", "warehouse_id"], how="left")
        .merge(risk.loc[:, risk_columns], on=["sku_id", "warehouse_id"], how="left")
    )
    numeric_fill_columns = [
        "forecast_1w_qty",
        "forecast_2w_qty",
        "forecast_4w_total_qty",
        "stockout_risk_score",
        "overstock_risk_score",
        "expiry_risk_score",
    ]
    base[numeric_fill_columns] = base[numeric_fill_columns].fillna(0)
    base["primary_risk_type"] = base["primary_risk_type"].fillna("stable")
    return base


def build_candidate_orders(
    feature_snapshot_path: Path | None = None,
    demand_forecast_path: Path | None = None,
    risk_score_path: Path | None = None,
) -> pd.DataFrame:
    """Build one candidate row per sku_id, warehouse_id, and fixed action."""
    snapshot_path = (
        feature_snapshot_path
        if feature_snapshot_path is not None
        else DEFAULT_FEATURE_SNAPSHOT_PATH
    )
    forecast_path = (
        demand_forecast_path
        if demand_forecast_path is not None
        else DEFAULT_DEMAND_FORECAST_PATH
    )
    risk_path = risk_score_path if risk_score_path is not None else DEFAULT_RISK_SCORE_PATH

    snapshot, forecast, risk = _read_inputs(snapshot_path, forecast_path, risk_path)
    base = _base_candidate_frame(snapshot, forecast, risk)

    rows: list[dict[str, object]] = []
    for item in base.sort_values(["sku_id", "warehouse_id"]).itertuples(index=False):
        base_row = pd.Series(item._asdict())
        for action in ACTIONS:
            candidate = base_row.to_dict()
            candidate["action_id"] = action.action_id
            candidate["action_name"] = action.action_name
            candidate["candidate_order_qty"] = _candidate_qty(action, base_row)
            candidate["candidate_reason"] = action.reason
            rows.append(candidate)

    return pd.DataFrame(rows, columns=CANDIDATE_ORDER_COLUMNS)


def write_candidate_orders(
    output_path: Path | None = None,
    feature_snapshot_path: Path | None = None,
    demand_forecast_path: Path | None = None,
    risk_score_path: Path | None = None,
) -> pd.DataFrame:
    """Build and write stage-6 replenishment candidate order rows."""
    path = output_path if output_path is not None else DEFAULT_CANDIDATE_ORDERS_PATH
    candidates = build_candidate_orders(
        feature_snapshot_path=feature_snapshot_path,
        demand_forecast_path=demand_forecast_path,
        risk_score_path=risk_score_path,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(path, index=False, encoding="utf-8-sig")
    return candidates
