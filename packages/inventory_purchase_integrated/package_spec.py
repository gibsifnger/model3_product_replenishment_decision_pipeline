"""Project-wide constants for the inventory replenishment decision pipeline.

This module intentionally contains only stable scope and path specifications for
stage 1. It does not include simulator, policy, forecasting, or gate logic.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_NAME = "model3_product_replenishment_decision_pipeline"
PACKAGE_NAME = "inventory_purchase_integrated"

DOMAIN = "food_and_consumer_goods_replenishment"
ROW_UNIT = "sku_warehouse_week"
WAREHOUSE_IDS = ("DC_01",)
SKU_IDS = ("SKU_A", "SKU_B", "SKU_C")
HISTORICAL_WEEKS = 52
DECISION_CADENCE = "weekly"
FORECAST_HORIZON_WEEKS = 4

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
INPUT_DATA_DIR = DATA_DIR / "input"

SKU_MASTER_FILENAME = "sku_master.csv"
WEEKLY_SALES_INVENTORY_FILENAME = "weekly_sales_inventory.csv"
PROMOTION_CALENDAR_FILENAME = "promotion_calendar.csv"
SUPPLIER_CONSTRAINTS_FILENAME = "supplier_constraints.csv"

INPUT_CSV_PATHS = {
    "sku_master": INPUT_DATA_DIR / SKU_MASTER_FILENAME,
    "weekly_sales_inventory": INPUT_DATA_DIR / WEEKLY_SALES_INVENTORY_FILENAME,
    "promotion_calendar": INPUT_DATA_DIR / PROMOTION_CALENDAR_FILENAME,
    "supplier_constraints": INPUT_DATA_DIR / SUPPLIER_CONSTRAINTS_FILENAME,
}
