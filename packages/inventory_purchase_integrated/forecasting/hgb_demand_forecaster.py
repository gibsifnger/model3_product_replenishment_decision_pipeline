"""HGB-based demand forecasting for stage-4 replenishment inputs."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from inventory_purchase_integrated.package_spec import INPUT_CSV_PATHS, OUTPUT_DATA_DIR
from inventory_purchase_integrated.schema import validate_required_columns

FEATURE_SNAPSHOT_FILENAME = "02_feature_snapshot.csv"
DEMAND_FORECAST_FILENAME = "03_demand_forecast.csv"
DEFAULT_FEATURE_SNAPSHOT_PATH = OUTPUT_DATA_DIR / FEATURE_SNAPSHOT_FILENAME
DEFAULT_DEMAND_FORECAST_PATH = OUTPUT_DATA_DIR / DEMAND_FORECAST_FILENAME

MODEL_NAME = "HistGradientBoostingRegressor"
FORECAST_HORIZON_WEEKS = 4

DEMAND_FORECAST_COLUMNS = (
    "forecast_run_week",
    "target_week",
    "sku_id",
    "warehouse_id",
    "forecast_horizon_week",
    "forecast_qty",
    "model_name",
    "training_rows",
)

MODEL_FEATURE_COLUMNS = (
    "sku_code",
    "warehouse_code",
    "lag_1_sales_qty",
    "lag_2_sales_qty",
    "rolling_4w_avg_sales",
    "rolling_8w_avg_sales",
    "week_of_year",
    "promo_flag",
    "expected_uplift_rate",
)


def _read_inputs(
    weekly_sales_path: Path,
    promotion_calendar_path: Path,
    feature_snapshot_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    weekly = pd.read_csv(weekly_sales_path)
    promotions = pd.read_csv(promotion_calendar_path)
    snapshot = pd.read_csv(feature_snapshot_path)

    validate_required_columns("weekly_sales_inventory", weekly.columns)
    validate_required_columns("promotion_calendar", promotions.columns)

    required_snapshot_columns = {"snapshot_week", "sku_id", "warehouse_id"}
    missing_snapshot_columns = required_snapshot_columns - set(snapshot.columns)
    if missing_snapshot_columns:
        missing_text = ", ".join(sorted(missing_snapshot_columns))
        raise ValueError(f"Feature snapshot is missing required columns: {missing_text}")

    weekly["week"] = pd.to_datetime(weekly["week"])
    promotions["week"] = pd.to_datetime(promotions["week"])
    snapshot["snapshot_week"] = pd.to_datetime(snapshot["snapshot_week"])
    return weekly, promotions, snapshot


def _with_promotion_features(
    weekly: pd.DataFrame,
    promotions: pd.DataFrame,
) -> pd.DataFrame:
    promo_features = promotions[
        ["week", "sku_id", "promo_flag", "expected_uplift_rate"]
    ].drop_duplicates(["week", "sku_id"])

    training_base = weekly.merge(promo_features, on=["week", "sku_id"], how="left")
    training_base["promo_flag"] = training_base["promo_flag"].fillna(0).astype(int)
    training_base["expected_uplift_rate"] = training_base["expected_uplift_rate"].fillna(0.0)
    return training_base


def _add_lag_features(training_base: pd.DataFrame) -> pd.DataFrame:
    grouped = training_base.sort_values(["sku_id", "warehouse_id", "week"]).groupby(
        ["sku_id", "warehouse_id"],
        group_keys=False,
    )
    featured = training_base.sort_values(["sku_id", "warehouse_id", "week"]).copy()
    featured["lag_1_sales_qty"] = grouped["sales_qty"].shift(1)
    featured["lag_2_sales_qty"] = grouped["sales_qty"].shift(2)
    featured["rolling_4w_avg_sales"] = grouped["sales_qty"].transform(
        lambda series: series.shift(1).rolling(4, min_periods=1).mean()
    )
    featured["rolling_8w_avg_sales"] = grouped["sales_qty"].transform(
        lambda series: series.shift(1).rolling(8, min_periods=1).mean()
    )
    featured["week_of_year"] = featured["week"].dt.isocalendar().week.astype(int)
    return featured.dropna(subset=["lag_1_sales_qty", "lag_2_sales_qty"])


def _code_maps(snapshot: pd.DataFrame) -> tuple[dict[str, int], dict[str, int]]:
    sku_ids = sorted(snapshot["sku_id"].unique())
    warehouse_ids = sorted(snapshot["warehouse_id"].unique())
    return (
        {sku_id: code for code, sku_id in enumerate(sku_ids)},
        {warehouse_id: code for code, warehouse_id in enumerate(warehouse_ids)},
    )


def _prepare_training_frame(
    weekly: pd.DataFrame,
    promotions: pd.DataFrame,
    sku_codes: dict[str, int],
    warehouse_codes: dict[str, int],
) -> pd.DataFrame:
    training = _add_lag_features(_with_promotion_features(weekly, promotions))
    training["sku_code"] = training["sku_id"].map(sku_codes)
    training["warehouse_code"] = training["warehouse_id"].map(warehouse_codes)
    return training.dropna(subset=MODEL_FEATURE_COLUMNS)


def _promotion_lookup(promotions: pd.DataFrame) -> dict[tuple[pd.Timestamp, str], tuple[int, float]]:
    rows = promotions[["week", "sku_id", "promo_flag", "expected_uplift_rate"]].drop_duplicates(
        ["week", "sku_id"]
    )
    return {
        (row.week, row.sku_id): (int(row.promo_flag), float(row.expected_uplift_rate))
        for row in rows.itertuples(index=False)
    }


def _predict_one_step(
    model: HistGradientBoostingRegressor,
    sku_id: str,
    warehouse_id: str,
    target_week: pd.Timestamp,
    sales_history: list[float],
    promo_lookup: dict[tuple[pd.Timestamp, str], tuple[int, float]],
    sku_codes: dict[str, int],
    warehouse_codes: dict[str, int],
) -> float:
    promo_flag, expected_uplift_rate = promo_lookup.get((target_week, sku_id), (0, 0.0))
    row = pd.DataFrame(
        [
            {
                "sku_code": sku_codes[sku_id],
                "warehouse_code": warehouse_codes[warehouse_id],
                "lag_1_sales_qty": sales_history[-1],
                "lag_2_sales_qty": sales_history[-2],
                "rolling_4w_avg_sales": float(np.mean(sales_history[-4:])),
                "rolling_8w_avg_sales": float(np.mean(sales_history[-8:])),
                "week_of_year": int(target_week.isocalendar().week),
                "promo_flag": promo_flag,
                "expected_uplift_rate": expected_uplift_rate,
            }
        ],
        columns=MODEL_FEATURE_COLUMNS,
    )
    return max(0.0, float(model.predict(row)[0]))


def build_demand_forecast(
    weekly_sales_path: Path | None = None,
    promotion_calendar_path: Path | None = None,
    feature_snapshot_path: Path | None = None,
) -> pd.DataFrame:
    """Train one HGB demand model and forecast four weekly rows per sku-warehouse."""
    weekly_path = weekly_sales_path if weekly_sales_path is not None else INPUT_CSV_PATHS[
        "weekly_sales_inventory"
    ]
    promo_path = (
        promotion_calendar_path
        if promotion_calendar_path is not None
        else INPUT_CSV_PATHS["promotion_calendar"]
    )
    snapshot_path = (
        feature_snapshot_path
        if feature_snapshot_path is not None
        else DEFAULT_FEATURE_SNAPSHOT_PATH
    )

    weekly, promotions, snapshot = _read_inputs(weekly_path, promo_path, snapshot_path)
    forecast_run_week = weekly["week"].max()
    sku_codes, warehouse_codes = _code_maps(snapshot)
    training = _prepare_training_frame(weekly, promotions, sku_codes, warehouse_codes)
    training_rows = len(training)

    model = HistGradientBoostingRegressor(random_state=42)
    model.fit(training.loc[:, MODEL_FEATURE_COLUMNS], training["sales_qty"])

    promo_by_week_sku = _promotion_lookup(promotions)
    forecast_rows: list[dict[str, object]] = []
    history = weekly.sort_values(["sku_id", "warehouse_id", "week"])

    for item in snapshot.sort_values(["sku_id", "warehouse_id"]).itertuples(index=False):
        sku_id = item.sku_id
        warehouse_id = item.warehouse_id
        sales_history = history[
            (history["sku_id"] == sku_id) & (history["warehouse_id"] == warehouse_id)
        ]["sales_qty"].astype(float).tolist()

        for horizon_week in range(1, FORECAST_HORIZON_WEEKS + 1):
            target_week = forecast_run_week + pd.Timedelta(weeks=horizon_week)
            forecast_qty = _predict_one_step(
                model=model,
                sku_id=sku_id,
                warehouse_id=warehouse_id,
                target_week=target_week,
                sales_history=sales_history,
                promo_lookup=promo_by_week_sku,
                sku_codes=sku_codes,
                warehouse_codes=warehouse_codes,
            )
            sales_history.append(forecast_qty)
            forecast_rows.append(
                {
                    "forecast_run_week": forecast_run_week.strftime("%Y-%m-%d"),
                    "target_week": target_week.strftime("%Y-%m-%d"),
                    "sku_id": sku_id,
                    "warehouse_id": warehouse_id,
                    "forecast_horizon_week": horizon_week,
                    "forecast_qty": forecast_qty,
                    "model_name": MODEL_NAME,
                    "training_rows": training_rows,
                }
            )

    return pd.DataFrame(forecast_rows, columns=DEMAND_FORECAST_COLUMNS)


def write_demand_forecast(
    output_path: Path | None = None,
    weekly_sales_path: Path | None = None,
    promotion_calendar_path: Path | None = None,
    feature_snapshot_path: Path | None = None,
) -> pd.DataFrame:
    """Build and write the stage-4 demand forecast CSV."""
    path = output_path if output_path is not None else DEFAULT_DEMAND_FORECAST_PATH
    forecast = build_demand_forecast(
        weekly_sales_path=weekly_sales_path,
        promotion_calendar_path=promotion_calendar_path,
        feature_snapshot_path=feature_snapshot_path,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    forecast.to_csv(path, index=False)
    return forecast
