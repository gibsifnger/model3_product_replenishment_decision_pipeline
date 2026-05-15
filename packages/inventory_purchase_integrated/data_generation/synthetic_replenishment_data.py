"""Synthetic input data generation for food/CPG replenishment experiments.

The generator creates only input datasets for downstream validation, feature,
forecasting, and simulation stages. It deliberately does not make purchase-order
recommendations and does not include simulator, policy, or gate logic.
"""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from inventory_purchase_integrated.package_spec import (
    HISTORICAL_WEEKS,
    INPUT_CSV_PATHS,
    SKU_IDS,
    WAREHOUSE_IDS,
)
from inventory_purchase_integrated.schema import REQUIRED_COLUMNS_BY_DATASET, validate_required_columns

DatasetRows = list[dict[str, Any]]


@dataclass(frozen=True)
class SyntheticDataBundle:
    """In-memory representation of the four stage-1 input datasets."""

    sku_master: DatasetRows
    weekly_sales_inventory: DatasetRows
    promotion_calendar: DatasetRows
    supplier_constraints: DatasetRows

    def as_dict(self) -> dict[str, DatasetRows]:
        """Return datasets keyed by their canonical schema names."""
        return {
            "sku_master": self.sku_master,
            "weekly_sales_inventory": self.weekly_sales_inventory,
            "promotion_calendar": self.promotion_calendar,
            "supplier_constraints": self.supplier_constraints,
        }


SKU_ATTRIBUTES: dict[str, dict[str, Any]] = {
    "SKU_A": {
        "sku_name": "Stable Pantry Staple",
        "category": "ambient_food",
        "demand_pattern": "stable_low_variance",
        "shelf_life_weeks": 26,
        "case_pack_size": 12,
        "unit_cost": 2.10,
        "list_price": 3.49,
        "storage_type": "ambient",
        "is_perishable": False,
    },
    "SKU_B": {
        "sku_name": "Promotion Sensitive Beverage",
        "category": "beverage",
        "demand_pattern": "promotion_sensitive",
        "shelf_life_weeks": 18,
        "case_pack_size": 24,
        "unit_cost": 1.25,
        "list_price": 2.29,
        "storage_type": "ambient",
        "is_perishable": False,
    },
    "SKU_C": {
        "sku_name": "Short Shelf Life Fresh Item",
        "category": "fresh_food",
        "demand_pattern": "slow_moving_perishable",
        "shelf_life_weeks": 3,
        "case_pack_size": 6,
        "unit_cost": 3.40,
        "list_price": 5.99,
        "storage_type": "chilled",
        "is_perishable": True,
    },
}

SUPPLIER_ATTRIBUTES: dict[str, dict[str, Any]] = {
    "SKU_A": {
        "supplier_id": "SUP_AMBIENT_01",
        "lead_time_weeks": 1,
        "min_order_qty": 48,
        "order_multiple": 12,
        "max_order_qty": 480,
        "service_level_target": 0.95,
    },
    "SKU_B": {
        "supplier_id": "SUP_BEVERAGE_01",
        "lead_time_weeks": 2,
        "min_order_qty": 96,
        "order_multiple": 24,
        "max_order_qty": 960,
        "service_level_target": 0.93,
    },
    "SKU_C": {
        "supplier_id": "SUP_FRESH_01",
        "lead_time_weeks": 1,
        "min_order_qty": 18,
        "order_multiple": 6,
        "max_order_qty": 180,
        "service_level_target": 0.90,
    },
}


def build_week_start_dates(weeks: int = HISTORICAL_WEEKS) -> list[date]:
    """Build a deterministic 52-week Monday calendar for reproducible data."""
    start_date = date(2025, 1, 6)
    return [start_date + timedelta(weeks=week_number) for week_number in range(weeks)]


def _validate_rows(dataset_name: str, rows: DatasetRows) -> None:
    columns = rows[0].keys() if rows else REQUIRED_COLUMNS_BY_DATASET[dataset_name]
    validate_required_columns(dataset_name, columns)


def generate_sku_master() -> DatasetRows:
    """Create SKU-level descriptive attributes for the fixed stage-1 scope."""
    rows = [{"sku_id": sku_id, **SKU_ATTRIBUTES[sku_id]} for sku_id in SKU_IDS]
    _validate_rows("sku_master", rows)
    return rows


def _promotion_for_sku_week(sku_id: str, week_index: int) -> tuple[int, str, float]:
    if sku_id == "SKU_B" and week_index in {7, 15, 23, 31, 39, 47}:
        return 1, "temporary_price_reduction", 0.25
    if sku_id == "SKU_A" and week_index in {12, 36}:
        return 1, "feature_display", 0.10
    return 0, "none", 0.00


def generate_promotion_calendar(week_start_dates: list[date] | None = None) -> DatasetRows:
    """Create SKU × warehouse × week promotion indicators used by later stages."""
    dates = week_start_dates if week_start_dates is not None else build_week_start_dates()
    rows: DatasetRows = []
    for week_index, week_start_date in enumerate(dates):
        for sku_id in SKU_IDS:
            for warehouse_id in WAREHOUSE_IDS:
                promotion_flag, promotion_type, discount_pct = _promotion_for_sku_week(
                    sku_id, week_index
                )
                rows.append(
                    {
                        "week_start_date": week_start_date.isoformat(),
                        "sku_id": sku_id,
                        "warehouse_id": warehouse_id,
                        "promotion_flag": promotion_flag,
                        "promotion_type": promotion_type,
                        "discount_pct": discount_pct,
                    }
                )
    _validate_rows("promotion_calendar", rows)
    return rows


def _sales_units(sku_id: str, week_index: int, promotion_flag: int, rng: random.Random) -> int:
    seasonal_multiplier = 1.0 + 0.08 * math.sin(2 * math.pi * week_index / 52)
    if sku_id == "SKU_A":
        return max(0, round(rng.gauss(82 * seasonal_multiplier, 5)))
    if sku_id == "SKU_B":
        baseline = rng.gauss(55 * seasonal_multiplier, 9)
        lift = rng.gauss(2.8, 0.25) if promotion_flag else 1.0
        return max(0, round(baseline * lift))
    if sku_id == "SKU_C":
        return max(0, round(rng.gauss(11 + 2 * seasonal_multiplier, 3)))
    raise KeyError(f"Unknown SKU '{sku_id}'")


def _inventory_units(sku_id: str, sales_units: int, week_index: int, rng: random.Random) -> int:
    if sku_id == "SKU_A":
        return max(24, round(rng.gauss(130 - 0.10 * sales_units, 10)))
    if sku_id == "SKU_B":
        return max(18, round(rng.gauss(110 - 0.08 * sales_units, 18)))
    if sku_id == "SKU_C":
        return max(0, round(rng.gauss(20 - 0.20 * sales_units + (week_index % 4), 5)))
    raise KeyError(f"Unknown SKU '{sku_id}'")


def generate_weekly_sales_inventory(
    promotion_calendar: DatasetRows,
    seed: int = 42,
) -> DatasetRows:
    """Create historical sales and inventory observations for downstream stages."""
    rng = random.Random(seed)
    rows: DatasetRows = []

    for row_number, promo_row in enumerate(promotion_calendar):
        sku_id = str(promo_row["sku_id"])
        week_index = row_number // (len(SKU_IDS) * len(WAREHOUSE_IDS))
        sales_units = _sales_units(
            sku_id=sku_id,
            week_index=week_index,
            promotion_flag=int(promo_row["promotion_flag"]),
            rng=rng,
        )
        ending_inventory_units = _inventory_units(sku_id, sales_units, week_index, rng)
        waste_units = 0
        if sku_id == "SKU_C":
            waste_base = 1.6 if ending_inventory_units > 18 else 0.3
            waste_units = max(0, round(rng.gauss(waste_base, 0.7)))
        stockout_units = max(0, round(rng.gauss(3, 2))) if ending_inventory_units < 10 else 0
        list_price = float(SKU_ATTRIBUTES[sku_id]["list_price"])
        unit_price = round(list_price * (1 - float(promo_row["discount_pct"])), 2)

        rows.append(
            {
                "week_start_date": promo_row["week_start_date"],
                "sku_id": sku_id,
                "warehouse_id": promo_row["warehouse_id"],
                "sales_units": sales_units,
                "ending_inventory_units": ending_inventory_units,
                "waste_units": waste_units,
                "stockout_units": stockout_units,
                "unit_price": unit_price,
            }
        )

    _validate_rows("weekly_sales_inventory", rows)
    return rows


def generate_supplier_constraints() -> DatasetRows:
    """Create supplier and order-constraint inputs for each SKU and warehouse."""
    rows: DatasetRows = []
    for sku_id in SKU_IDS:
        for warehouse_id in WAREHOUSE_IDS:
            rows.append(
                {
                    "sku_id": sku_id,
                    "warehouse_id": warehouse_id,
                    **SUPPLIER_ATTRIBUTES[sku_id],
                }
            )
    _validate_rows("supplier_constraints", rows)
    return rows


def generate_synthetic_replenishment_data(seed: int = 42) -> SyntheticDataBundle:
    """Generate all four input datasets as row dictionaries."""
    week_start_dates = build_week_start_dates()
    sku_master = generate_sku_master()
    promotion_calendar = generate_promotion_calendar(week_start_dates)
    weekly_sales_inventory = generate_weekly_sales_inventory(promotion_calendar, seed=seed)
    supplier_constraints = generate_supplier_constraints()
    return SyntheticDataBundle(
        sku_master=sku_master,
        weekly_sales_inventory=weekly_sales_inventory,
        promotion_calendar=promotion_calendar,
        supplier_constraints=supplier_constraints,
    )


def _write_csv(dataset_name: str, rows: DatasetRows, output_path: Path) -> None:
    validate_required_columns(dataset_name, rows[0].keys() if rows else ())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=REQUIRED_COLUMNS_BY_DATASET[dataset_name])
        writer.writeheader()
        writer.writerows(rows)


def write_synthetic_replenishment_data(
    output_paths: dict[str, Path] | None = None,
    seed: int = 42,
) -> SyntheticDataBundle:
    """Generate and write the four canonical input CSV files."""
    paths = output_paths if output_paths is not None else INPUT_CSV_PATHS
    bundle = generate_synthetic_replenishment_data(seed=seed)

    for dataset_name, rows in bundle.as_dict().items():
        _write_csv(dataset_name, rows, paths[dataset_name])

    return bundle
