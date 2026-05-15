"""Input quality validation for stage-2 replenishment data checks.

The validator records quality issues without modifying input data.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from inventory_purchase_integrated.package_spec import INPUT_CSV_PATHS, OUTPUT_DATA_DIR, WAREHOUSE_IDS
from inventory_purchase_integrated.schema import REQUIRED_COLUMNS_BY_DATASET, missing_required_columns

QUALITY_CHECK_COLUMNS = (
    "check_name",
    "table_name",
    "status",
    "issue_count",
    "message",
)

QUALITY_CHECK_FILENAME = "01_input_quality_check.csv"
DEFAULT_QUALITY_CHECK_PATH = OUTPUT_DATA_DIR / QUALITY_CHECK_FILENAME

DatasetRows = list[dict[str, str]]
QualityRows = list[dict[str, Any]]


def _check_row(check_name: str, table_name: str, issue_count: int, message: str) -> dict[str, Any]:
    return {
        "check_name": check_name,
        "table_name": table_name,
        "status": "PASS" if issue_count == 0 else "FAIL",
        "issue_count": issue_count,
        "message": message,
    }


def _read_csv(path: Path) -> DatasetRows:
    with path.open("r", newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def _read_existing_inputs(input_paths: dict[str, Path]) -> dict[str, DatasetRows]:
    return {
        dataset_name: _read_csv(path)
        for dataset_name, path in input_paths.items()
        if path.exists()
    }


def _headers_for(path: Path) -> tuple[str, ...]:
    if not path.exists():
        return ()
    with path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file)
        return tuple(next(reader, ()))


def _is_number(value: str) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _is_negative(value: str) -> bool:
    return (not _is_number(value)) or float(value) < 0


def _is_outside_range(value: str, lower: float, upper: float) -> bool:
    return (not _is_number(value)) or not lower <= float(value) <= upper


def _is_less_than(value: str, lower: float) -> bool:
    return (not _is_number(value)) or float(value) < lower


def _count_missing_files(input_paths: dict[str, Path]) -> int:
    return sum(1 for path in input_paths.values() if not path.exists())


def _schema_checks(input_paths: dict[str, Path]) -> QualityRows:
    checks: QualityRows = []
    for dataset_name, required_columns in REQUIRED_COLUMNS_BY_DATASET.items():
        path = input_paths[dataset_name]
        if not path.exists():
            checks.append(
                _check_row(
                    "required_columns_exist",
                    dataset_name,
                    len(required_columns),
                    f"File missing; expected columns: {', '.join(required_columns)}",
                )
            )
            continue

        missing_columns = missing_required_columns(dataset_name, _headers_for(path))
        checks.append(
            _check_row(
                "required_columns_exist",
                dataset_name,
                len(missing_columns),
                "All required columns exist."
                if not missing_columns
                else f"Missing columns: {', '.join(missing_columns)}",
            )
        )
    return checks


def _duplicate_key_check(rows_by_table: dict[str, DatasetRows]) -> dict[str, Any]:
    table_name = "weekly_sales_inventory"
    rows = rows_by_table.get(table_name, [])
    seen: set[tuple[str, str, str]] = set()
    duplicate_count = 0

    for row in rows:
        key = (row.get("week", ""), row.get("sku_id", ""), row.get("warehouse_id", ""))
        if key in seen:
            duplicate_count += 1
        else:
            seen.add(key)

    return _check_row(
        "weekly_duplicate_key",
        table_name,
        duplicate_count,
        "No duplicate week + sku_id + warehouse_id keys."
        if duplicate_count == 0
        else "Duplicate week + sku_id + warehouse_id rows found.",
    )


def _negative_value_checks(rows_by_table: dict[str, DatasetRows]) -> QualityRows:
    table_name = "weekly_sales_inventory"
    rows = rows_by_table.get(table_name, [])
    checks: QualityRows = []

    for column in ("sales_qty", "on_hand_qty", "inbound_qty", "expired_qty"):
        issue_count = sum(1 for row in rows if _is_negative(row.get(column, "")))
        checks.append(
            _check_row(
                f"{column}_non_negative",
                table_name,
                issue_count,
                f"All {column} values are non-negative."
                if issue_count == 0
                else f"{column} has negative or non-numeric values.",
            )
        )
    return checks


def _binary_flag_check(
    rows_by_table: dict[str, DatasetRows],
    table_name: str,
    column: str,
) -> dict[str, Any]:
    rows = rows_by_table.get(table_name, [])
    issue_count = sum(1 for row in rows if row.get(column) not in {"0", "1"})
    return _check_row(
        f"{column}_binary",
        table_name,
        issue_count,
        f"All {column} values are 0 or 1."
        if issue_count == 0
        else f"{column} contains values other than 0 or 1.",
    )


def _non_negative_check(
    rows_by_table: dict[str, DatasetRows],
    table_name: str,
    column: str,
) -> dict[str, Any]:
    rows = rows_by_table.get(table_name, [])
    issue_count = sum(1 for row in rows if _is_less_than(row.get(column, ""), 0))
    return _check_row(
        f"{column}_non_negative",
        table_name,
        issue_count,
        f"All {column} values are greater than or equal to 0."
        if issue_count == 0
        else f"{column} has negative or non-numeric values.",
    )


def _range_check(
    rows_by_table: dict[str, DatasetRows],
    table_name: str,
    column: str,
    lower: float,
    upper: float,
) -> dict[str, Any]:
    rows = rows_by_table.get(table_name, [])
    issue_count = sum(1 for row in rows if _is_outside_range(row.get(column, ""), lower, upper))
    return _check_row(
        f"{column}_range",
        table_name,
        issue_count,
        f"All {column} values are between {lower:g} and {upper:g}."
        if issue_count == 0
        else f"{column} has values outside {lower:g} to {upper:g}, or non-numeric values.",
    )


def _sku_reference_checks(rows_by_table: dict[str, DatasetRows]) -> QualityRows:
    sku_master_ids = {row.get("sku_id", "") for row in rows_by_table.get("sku_master", [])}
    checks: QualityRows = []

    for table_name in ("weekly_sales_inventory", "promotion_calendar", "supplier_constraints"):
        rows = rows_by_table.get(table_name, [])
        issue_count = sum(1 for row in rows if row.get("sku_id", "") not in sku_master_ids)
        checks.append(
            _check_row(
                "sku_id_exists_in_sku_master",
                table_name,
                issue_count,
                "All sku_id values exist in sku_master."
                if issue_count == 0
                else "Rows contain sku_id values that do not exist in sku_master.",
            )
        )
    return checks


def _warehouse_checks(rows_by_table: dict[str, DatasetRows]) -> QualityRows:
    allowed_warehouses = set(WAREHOUSE_IDS)
    checks: QualityRows = []

    for table_name, rows in rows_by_table.items():
        if not rows or "warehouse_id" not in rows[0]:
            continue
        issue_count = sum(1 for row in rows if row.get("warehouse_id", "") not in allowed_warehouses)
        checks.append(
            _check_row(
                "warehouse_id_allowed",
                table_name,
                issue_count,
                f"All warehouse_id values are in {', '.join(allowed_warehouses)}."
                if issue_count == 0
                else f"warehouse_id contains values outside {', '.join(allowed_warehouses)}.",
            )
        )
    return checks


def run_input_quality_validation(input_paths: dict[str, Path] | None = None) -> QualityRows:
    """Run stage-2 validation checks and return quality-check result rows."""
    paths = input_paths if input_paths is not None else INPUT_CSV_PATHS
    rows_by_table = _read_existing_inputs(paths)

    missing_file_count = _count_missing_files(paths)
    checks: QualityRows = [
        _check_row(
            "input_files_exist",
            "all_input_tables",
            missing_file_count,
            "All input CSV files exist."
            if missing_file_count == 0
            else "One or more required input CSV files are missing.",
        )
    ]

    checks.extend(_schema_checks(paths))
    checks.append(_duplicate_key_check(rows_by_table))
    checks.extend(_negative_value_checks(rows_by_table))
    checks.append(_binary_flag_check(rows_by_table, "weekly_sales_inventory", "stockout_flag"))
    checks.append(_binary_flag_check(rows_by_table, "promotion_calendar", "promo_flag"))
    checks.append(
        _non_negative_check(rows_by_table, "promotion_calendar", "expected_uplift_rate")
    )
    checks.append(_range_check(rows_by_table, "supplier_constraints", "otif_rate", 0, 1))
    checks.append(
        _range_check(rows_by_table, "supplier_constraints", "supplier_risk_score", 0, 1)
    )
    checks.extend(_sku_reference_checks(rows_by_table))
    checks.extend(_warehouse_checks(rows_by_table))

    return checks


def write_input_quality_check(
    output_path: Path | None = None,
    input_paths: dict[str, Path] | None = None,
) -> QualityRows:
    """Run input quality validation and write the quality-check CSV."""
    path = output_path if output_path is not None else DEFAULT_QUALITY_CHECK_PATH
    rows = run_input_quality_validation(input_paths=input_paths)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=QUALITY_CHECK_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    return rows
