"""Risk scoring for current replenishment state snapshots."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from inventory_purchase_integrated.package_spec import OUTPUT_DATA_DIR
from inventory_purchase_integrated.state.feature_builder import DEFAULT_FEATURE_SNAPSHOT_PATH

DEMAND_FORECAST_FILENAME = "03_demand_forecast.csv"
RISK_SCORE_FILENAME = "04_risk_score.csv"
DEFAULT_DEMAND_FORECAST_PATH = OUTPUT_DATA_DIR / DEMAND_FORECAST_FILENAME
DEFAULT_RISK_SCORE_PATH = OUTPUT_DATA_DIR / RISK_SCORE_FILENAME

RISK_SCORE_COLUMNS = (
    "snapshot_week",
    "sku_id",
    "warehouse_id",
    "forecast_4w_total_qty",
    "available_qty",
    "inventory_cover_weeks",
    "inbound_cover_weeks",
    "stockout_risk_score",
    "overstock_risk_score",
    "expiry_risk_score",
    "supplier_risk_score",
    "total_risk_score",
    "primary_risk_type",
    "risk_comment",
)

RISK_COMPONENT_COLUMNS = (
    "stockout_risk_score",
    "overstock_risk_score",
    "expiry_risk_score",
    "supplier_risk_score",
)


def _clip_0_1(values: pd.Series) -> pd.Series:
    return values.clip(lower=0, upper=1)


def _required_columns(frame: pd.DataFrame, columns: set[str], dataset_name: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"{dataset_name} is missing required columns: {missing_text}")


def _read_inputs(
    feature_snapshot_path: Path,
    demand_forecast_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    snapshot = pd.read_csv(feature_snapshot_path)
    forecast = pd.read_csv(demand_forecast_path)

    _required_columns(
        snapshot,
        {
            "snapshot_week",
            "sku_id",
            "warehouse_id",
            "available_qty",
            "inventory_cover_weeks",
            "inbound_cover_weeks",
            "shelf_life_days",
            "storage_type",
            "recent_expired_qty",
            "supplier_risk_score",
        },
        "feature_snapshot",
    )
    _required_columns(
        forecast,
        {"sku_id", "warehouse_id", "forecast_qty"},
        "demand_forecast",
    )
    return snapshot, forecast


def _forecast_totals(forecast: pd.DataFrame) -> pd.DataFrame:
    return forecast.groupby(["sku_id", "warehouse_id"], as_index=False).agg(
        forecast_4w_total_qty=("forecast_qty", "sum")
    )


def _stockout_risk_score(features: pd.DataFrame) -> pd.Series:
    shortage_ratio = (
        features["forecast_4w_total_qty"] - features["available_qty"]
    ) / features["forecast_4w_total_qty"].replace(0, np.nan)
    return _clip_0_1(shortage_ratio.fillna(0))


def _overstock_risk_score(features: pd.DataFrame) -> pd.Series:
    cover_risk = (features["inventory_cover_weeks"] - 2.0) / 6.0
    sensitivity = np.where(
        (features["sku_id"] == "SKU_C")
        | (features["shelf_life_days"] <= 30)
        | (features["storage_type"].str.lower() == "chilled"),
        1.35,
        1.0,
    )
    return _clip_0_1(cover_risk * sensitivity)


def _expiry_risk_score(features: pd.DataFrame) -> pd.Series:
    shelf_life_risk = _clip_0_1((90 - features["shelf_life_days"]) / 90)
    expired_qty_risk = _clip_0_1(features["recent_expired_qty"] / 8)
    perishable_boost = np.where(
        (features["shelf_life_days"] <= 30)
        | (features["storage_type"].str.lower() == "chilled"),
        0.20,
        0.0,
    )
    return _clip_0_1((0.55 * shelf_life_risk) + (0.35 * expired_qty_risk) + perishable_boost)


def _primary_risk_type(row: pd.Series) -> str:
    scores = {
        "stockout": row["stockout_risk_score"],
        "overstock": row["overstock_risk_score"],
        "expiry": row["expiry_risk_score"],
        "supplier": row["supplier_risk_score"],
    }
    if max(scores.values()) < 0.30:
        return "stable"
    return max(scores, key=scores.get)


def _risk_comment(row: pd.Series) -> str:
    sku_id = row["sku_id"]
    primary = row["primary_risk_type"]

    if primary == "stockout":
        return (
            f"{sku_id}: 향후 4주 예측수요 대비 가용재고가 부족해 품절위험이 높음"
        )
    if primary == "overstock":
        return (
            f"{sku_id}: 재고 커버 주수가 높아 과잉재고 위험이 높음"
        )
    if primary == "expiry":
        return (
            f"{sku_id}: 짧은 유통기한과 최근 폐기 이력으로 폐기위험이 높음"
        )
    if primary == "supplier":
        return (
            f"{sku_id}: 공급사 리스크 점수가 상대적으로 높아 공급위험 관리가 필요함"
        )
    return f"{sku_id}: 주요 위험 점수가 낮아 현재 상태는 안정적임"


def build_risk_score(
    feature_snapshot_path: Path | None = None,
    demand_forecast_path: Path | None = None,
) -> pd.DataFrame:
    """Build sku_id and warehouse_id level risk scores without purchase decisions."""
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

    snapshot, forecast = _read_inputs(snapshot_path, forecast_path)
    risk = snapshot.merge(_forecast_totals(forecast), on=["sku_id", "warehouse_id"], how="left")
    risk["forecast_4w_total_qty"] = risk["forecast_4w_total_qty"].fillna(0)

    risk["stockout_risk_score"] = _stockout_risk_score(risk)
    risk["overstock_risk_score"] = _overstock_risk_score(risk)
    risk["expiry_risk_score"] = _expiry_risk_score(risk)
    risk["supplier_risk_score"] = _clip_0_1(risk["supplier_risk_score"])
    risk["total_risk_score"] = _clip_0_1(
        (risk["stockout_risk_score"] * 0.35)
        + (risk["overstock_risk_score"] * 0.25)
        + (risk["expiry_risk_score"] * 0.25)
        + (risk["supplier_risk_score"] * 0.15)
    )
    risk["primary_risk_type"] = risk.apply(_primary_risk_type, axis=1)
    risk["risk_comment"] = risk.apply(_risk_comment, axis=1)

    return risk.loc[:, RISK_SCORE_COLUMNS].sort_values(["sku_id", "warehouse_id"])


def write_risk_score(
    output_path: Path | None = None,
    feature_snapshot_path: Path | None = None,
    demand_forecast_path: Path | None = None,
) -> pd.DataFrame:
    """Build and write the stage-5 risk score CSV."""
    path = output_path if output_path is not None else DEFAULT_RISK_SCORE_PATH
    risk = build_risk_score(
        feature_snapshot_path=feature_snapshot_path,
        demand_forecast_path=demand_forecast_path,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    risk.to_csv(path, index=False, encoding="utf-8-sig")
    return risk
