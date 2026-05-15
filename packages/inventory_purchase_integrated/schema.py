"""Column schema definitions for stage-1 replenishment input datasets."""

from __future__ import annotations

from collections.abc import Iterable

SKU_MASTER_COLUMNS = (
    "sku_id",
    "sku_name",
    "category",
    "demand_pattern",
    "shelf_life_weeks",
    "case_pack_size",
    "unit_cost",
    "list_price",
    "storage_type",
    "is_perishable",
)

WEEKLY_SALES_INVENTORY_COLUMNS = (
    "week_start_date",
    "sku_id",
    "warehouse_id",
    "sales_units",
    "ending_inventory_units",
    "waste_units",
    "stockout_units",
    "unit_price",
)

PROMOTION_CALENDAR_COLUMNS = (
    "week_start_date",
    "sku_id",
    "warehouse_id",
    "promotion_flag",
    "promotion_type",
    "discount_pct",
)

SUPPLIER_CONSTRAINTS_COLUMNS = (
    "sku_id",
    "warehouse_id",
    "supplier_id",
    "lead_time_weeks",
    "min_order_qty",
    "order_multiple",
    "max_order_qty",
    "service_level_target",
)

REQUIRED_COLUMNS_BY_DATASET = {
    "sku_master": SKU_MASTER_COLUMNS,
    "weekly_sales_inventory": WEEKLY_SALES_INVENTORY_COLUMNS,
    "promotion_calendar": PROMOTION_CALENDAR_COLUMNS,
    "supplier_constraints": SUPPLIER_CONSTRAINTS_COLUMNS,
}


def missing_required_columns(dataset_name: str, actual_columns: Iterable[str]) -> tuple[str, ...]:
    """Return required columns that are absent from a named dataset."""
    if dataset_name not in REQUIRED_COLUMNS_BY_DATASET:
        known = ", ".join(sorted(REQUIRED_COLUMNS_BY_DATASET))
        raise KeyError(f"Unknown dataset '{dataset_name}'. Known datasets: {known}")

    actual_column_set = set(actual_columns)
    return tuple(
        column
        for column in REQUIRED_COLUMNS_BY_DATASET[dataset_name]
        if column not in actual_column_set
    )


def validate_required_columns(dataset_name: str, actual_columns: Iterable[str]) -> None:
    """Raise ValueError if a named dataset lacks any required stage-1 columns."""
    missing_columns = missing_required_columns(dataset_name, actual_columns)
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Dataset '{dataset_name}' is missing required columns: {missing_text}")
