"""Build current-state feature snapshots for stage-3 replenishment inputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from inventory_purchase_integrated.package_spec import INPUT_CSV_PATHS, OUTPUT_DATA_DIR
from inventory_purchase_integrated.schema import validate_required_columns

FEATURE_SNAPSHOT_FILENAME = "02_feature_snapshot.csv"
DEFAULT_FEATURE_SNAPSHOT_PATH = OUTPUT_DATA_DIR / FEATURE_SNAPSHOT_FILENAME

FEATURE_SNAPSHOT_COLUMNS = (
    "snapshot_week",
    "sku_id",
    "warehouse_id",
    "category",
    "abc_class",
    "storage_type",
    "shelf_life_days",
    "unit_cost",
    "unit_price",
    "on_hand_qty",
    "inbound_qty",
    "available_qty",
    "avg_sales_4w",
    "avg_sales_8w",
    "sales_std_8w",
    "inventory_cover_weeks",
    "inbound_cover_weeks",
    "promo_next_4w_flag",
    "recent_stockout_count",
    "recent_expired_qty",
    "lead_time_days",
    "moq_qty",
    "box_multiple_qty",
    "otif_rate",
    "supplier_risk_score",
)


def _read_inputs(input_paths: dict[str, Path]) -> dict[str, pd.DataFrame]:
    frames = {
        dataset_name: pd.read_csv(path)
        for dataset_name, path in input_paths.items()
    }

    for dataset_name, frame in frames.items():
        validate_required_columns(dataset_name, frame.columns)

    frames["weekly_sales_inventory"]["week"] = pd.to_datetime(
        frames["weekly_sales_inventory"]["week"]
    )
    frames["promotion_calendar"]["week"] = pd.to_datetime(
        frames["promotion_calendar"]["week"]
    )
    return frames


def _sales_window_features(history: pd.DataFrame, window: int, prefix: str) -> pd.DataFrame:
    recent = (
        history.sort_values(["sku_id", "warehouse_id", "week"])
        .groupby(["sku_id", "warehouse_id"], as_index=False)
        .tail(window)
    )
    return (
        recent.groupby(["sku_id", "warehouse_id"], as_index=False)
        .agg(**{prefix: ("sales_qty", "mean")})
    )


def _sales_std_8w(history: pd.DataFrame) -> pd.DataFrame:
    recent = (
        history.sort_values(["sku_id", "warehouse_id", "week"])
        .groupby(["sku_id", "warehouse_id"], as_index=False)
        .tail(8)
    )
    return (
        recent.groupby(["sku_id", "warehouse_id"], as_index=False)
        .agg(sales_std_8w=("sales_qty", "std"))
        .fillna({"sales_std_8w": 0})
    )


def _recent_event_features(history: pd.DataFrame) -> pd.DataFrame:
    recent = (
        history.sort_values(["sku_id", "warehouse_id", "week"])
        .groupby(["sku_id", "warehouse_id"], as_index=False)
        .tail(4)
    )
    return recent.groupby(["sku_id", "warehouse_id"], as_index=False).agg(
        recent_stockout_count=("stockout_flag", "sum"),
        recent_expired_qty=("expired_qty", "sum"),
    )


def _promo_next_4w(promotion_calendar: pd.DataFrame, snapshot_week: pd.Timestamp) -> pd.DataFrame:
    future_end = snapshot_week + pd.Timedelta(weeks=4)
    future_promos = promotion_calendar[
        (promotion_calendar["week"] > snapshot_week)
        & (promotion_calendar["week"] <= future_end)
    ]

    if future_promos.empty:
        return promotion_calendar[["sku_id"]].drop_duplicates().assign(promo_next_4w_flag=0)

    return (
        future_promos.groupby("sku_id", as_index=False)
        .agg(promo_next_4w_flag=("promo_flag", "max"))
    )


def build_feature_snapshot(input_paths: dict[str, Path] | None = None) -> pd.DataFrame:
    """Build one current-state feature row per sku_id and warehouse_id."""
    paths = input_paths if input_paths is not None else INPUT_CSV_PATHS
    frames = _read_inputs(paths)

    sku_master = frames["sku_master"]
    weekly = frames["weekly_sales_inventory"]
    promotions = frames["promotion_calendar"]
    supplier_constraints = frames["supplier_constraints"]

    snapshot_week = weekly["week"].max()
    history = weekly[weekly["week"] <= snapshot_week].copy()

    latest_inventory = history[history["week"] == snapshot_week].copy()
    latest_inventory = latest_inventory[
        ["week", "sku_id", "warehouse_id", "on_hand_qty", "inbound_qty"]
    ].rename(columns={"week": "snapshot_week"})

    features = (
        latest_inventory.merge(
            _sales_window_features(history, 4, "avg_sales_4w"),
            on=["sku_id", "warehouse_id"],
            how="left",
        )
        .merge(
            _sales_window_features(history, 8, "avg_sales_8w"),
            on=["sku_id", "warehouse_id"],
            how="left",
        )
        .merge(_sales_std_8w(history), on=["sku_id", "warehouse_id"], how="left")
        .merge(_recent_event_features(history), on=["sku_id", "warehouse_id"], how="left")
        .merge(_promo_next_4w(promotions, snapshot_week), on="sku_id", how="left")
        .merge(sku_master, on="sku_id", how="left")
        .merge(
            supplier_constraints[
                [
                    "sku_id",
                    "lead_time_days",
                    "moq_qty",
                    "box_multiple_qty",
                    "otif_rate",
                    "supplier_risk_score",
                ]
            ],
            on="sku_id",
            how="left",
        )
    )

    features["available_qty"] = features["on_hand_qty"] + features["inbound_qty"]
    features["inventory_cover_weeks"] = np.where(
        features["avg_sales_4w"] == 0,
        999,
        features["on_hand_qty"] / features["avg_sales_4w"],
    )
    features["inbound_cover_weeks"] = np.where(
        features["avg_sales_4w"] == 0,
        999,
        features["available_qty"] / features["avg_sales_4w"],
    )
    features["promo_next_4w_flag"] = features["promo_next_4w_flag"].fillna(0).astype(int)

    features["snapshot_week"] = features["snapshot_week"].dt.strftime("%Y-%m-%d")
    return features.loc[:, FEATURE_SNAPSHOT_COLUMNS].sort_values(
        ["sku_id", "warehouse_id"]
    )


def write_feature_snapshot(
    output_path: Path | None = None,
    input_paths: dict[str, Path] | None = None,
) -> pd.DataFrame:
    """Build and write the stage-3 feature snapshot CSV."""
    path = output_path if output_path is not None else DEFAULT_FEATURE_SNAPSHOT_PATH
    features = build_feature_snapshot(input_paths=input_paths)

    path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(path, index=False)
    return features
