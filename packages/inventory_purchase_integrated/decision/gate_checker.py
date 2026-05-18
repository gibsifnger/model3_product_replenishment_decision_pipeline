"""Gate checks for replenishment candidate orders.

This module records operational pass/fail results and comments only. It does
not choose a final action or optimize costs.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from inventory_purchase_integrated.package_spec import OUTPUT_DATA_DIR

FEATURE_SNAPSHOT_FILENAME = "02_feature_snapshot.csv"
CANDIDATE_ORDERS_FILENAME = "05_candidate_orders.csv"
GATE_RESULT_FILENAME = "06_gate_result.csv"

DEFAULT_FEATURE_SNAPSHOT_PATH = OUTPUT_DATA_DIR / FEATURE_SNAPSHOT_FILENAME
DEFAULT_CANDIDATE_ORDERS_PATH = OUTPUT_DATA_DIR / CANDIDATE_ORDERS_FILENAME
DEFAULT_GATE_RESULT_PATH = OUTPUT_DATA_DIR / GATE_RESULT_FILENAME

GATE_RESULT_COLUMNS = (
    "snapshot_week",
    "sku_id",
    "warehouse_id",
    "action_id",
    "action_name",
    "candidate_order_qty",
    "moq_qty",
    "box_multiple_qty",
    "lead_time_days",
    "shelf_life_days",
    "forecast_2w_qty",
    "forecast_4w_total_qty",
    "available_qty",
    "primary_risk_type",
    "moq_gate_status",
    "box_multiple_gate_status",
    "lead_time_gate_status",
    "shelf_life_gate_status",
    "warehouse_capacity_gate_status",
    "overall_gate_status",
    "failed_gate_count",
    "gate_comment",
)

GATE_STATUS_COLUMNS = (
    "moq_gate_status",
    "box_multiple_gate_status",
    "lead_time_gate_status",
    "shelf_life_gate_status",
    "warehouse_capacity_gate_status",
)


def _required_columns(frame: pd.DataFrame, columns: set[str], dataset_name: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"{dataset_name} is missing required columns: {missing_text}")


def _read_inputs(
    candidate_orders_path: Path,
    feature_snapshot_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = pd.read_csv(candidate_orders_path, encoding="utf-8-sig")
    snapshot = pd.read_csv(feature_snapshot_path)

    _required_columns(
        candidates,
        {
            "snapshot_week",
            "sku_id",
            "warehouse_id",
            "action_id",
            "action_name",
            "candidate_order_qty",
            "moq_qty",
            "box_multiple_qty",
            "forecast_2w_qty",
            "forecast_4w_total_qty",
            "available_qty",
            "overstock_risk_score",
            "expiry_risk_score",
            "primary_risk_type",
        },
        "candidate_orders",
    )
    _required_columns(
        snapshot,
        {
            "sku_id",
            "warehouse_id",
            "lead_time_days",
            "shelf_life_days",
        },
        "feature_snapshot",
    )
    return candidates, snapshot


def _status_and_comment(row: pd.Series) -> tuple[dict[str, str], str]:
    comments: list[str] = []
    quantity = row["candidate_order_qty"]

    moq_status = "PASS"
    if quantity > 0 and quantity < row["moq_qty"]:
        moq_status = "FAIL"
        comments.append("candidate_order_qty is below moq_qty")

    box_multiple_status = "PASS"
    if quantity > 0 and row["box_multiple_qty"] > 0 and quantity % row["box_multiple_qty"] != 0:
        box_multiple_status = "FAIL"
        comments.append("candidate_order_qty is not a box_multiple_qty multiple")

    lead_time_status = "PASS"
    if pd.isna(row["lead_time_days"]):
        lead_time_status = "FAIL"
        comments.append("lead_time_days is missing")
    elif row["lead_time_days"] > 14:
        comments.append("lead_time_days exceeds 14 days; lead time warning")
    elif row["action_name"] == "expedite":
        comments.append("expedite candidate; lead time gate reviewed separately")

    shelf_life_status = "PASS"
    if row["shelf_life_days"] <= 21 and quantity > row["forecast_2w_qty"]:
        shelf_life_status = "FAIL"
        comments.append("short shelf-life item ordered above forecast_2w_qty")

    warehouse_capacity_status = "PASS"
    comments.append("max_order_qty not configured")
    if row["available_qty"] + quantity > row["forecast_4w_total_qty"] * 1.5:
        comments.append("available_qty plus candidate order may create overstock warning")

    if row["action_name"] == "hold" and quantity == 0:
        comments.append("hold action with zero order quantity")
    if row["action_name"] == "reduce_review" and quantity == 0:
        comments.append("reduce_review action with zero order quantity")
        if row["expiry_risk_score"] >= 0.50 or row["overstock_risk_score"] >= 0.50:
            comments.append("expiry or overstock risk supports reduction review")

    statuses = {
        "moq_gate_status": moq_status,
        "box_multiple_gate_status": box_multiple_status,
        "lead_time_gate_status": lead_time_status,
        "shelf_life_gate_status": shelf_life_status,
        "warehouse_capacity_gate_status": warehouse_capacity_status,
    }
    return statuses, "; ".join(comments) if comments else "All configured gates passed"


def _merge_gate_inputs(candidates: pd.DataFrame, snapshot: pd.DataFrame) -> pd.DataFrame:
    snapshot_gate_columns = [
        "sku_id",
        "warehouse_id",
        "lead_time_days",
        "shelf_life_days",
    ]
    return candidates.merge(
        snapshot.loc[:, snapshot_gate_columns],
        on=["sku_id", "warehouse_id"],
        how="left",
    )


def build_gate_result(
    candidate_orders_path: Path | None = None,
    feature_snapshot_path: Path | None = None,
) -> pd.DataFrame:
    """Build pass/fail gate results for candidate order rows."""
    candidate_path = (
        candidate_orders_path
        if candidate_orders_path is not None
        else DEFAULT_CANDIDATE_ORDERS_PATH
    )
    snapshot_path = (
        feature_snapshot_path
        if feature_snapshot_path is not None
        else DEFAULT_FEATURE_SNAPSHOT_PATH
    )

    candidates, snapshot = _read_inputs(candidate_path, snapshot_path)
    gate = _merge_gate_inputs(candidates, snapshot)

    result_rows: list[dict[str, object]] = []
    for row in gate.itertuples(index=False):
        row_series = pd.Series(row._asdict())
        statuses, comment = _status_and_comment(row_series)
        failed_gate_count = sum(1 for status in statuses.values() if status == "FAIL")
        result = row_series.to_dict()
        result.update(statuses)
        result["failed_gate_count"] = failed_gate_count
        result["overall_gate_status"] = "FAIL" if failed_gate_count else "PASS"
        result["gate_comment"] = comment
        result_rows.append(result)

    return pd.DataFrame(result_rows, columns=GATE_RESULT_COLUMNS)


def write_gate_result(
    output_path: Path | None = None,
    candidate_orders_path: Path | None = None,
    feature_snapshot_path: Path | None = None,
) -> pd.DataFrame:
    """Build and write stage-7 candidate gate check results."""
    path = output_path if output_path is not None else DEFAULT_GATE_RESULT_PATH
    gate_result = build_gate_result(
        candidate_orders_path=candidate_orders_path,
        feature_snapshot_path=feature_snapshot_path,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    gate_result.to_csv(path, index=False, encoding="utf-8-sig")
    return gate_result
