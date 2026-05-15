#!/usr/bin/env python
"""Generate stage-1 synthetic CSV inputs and print dataset heads."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGES_DIR = REPO_ROOT / "packages"
if str(PACKAGES_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGES_DIR))

from inventory_purchase_integrated.package_spec import INPUT_CSV_PATHS
from inventory_purchase_integrated.schema import REQUIRED_COLUMNS_BY_DATASET, validate_required_columns
from inventory_purchase_integrated.data_generation.synthetic_replenishment_data import (
    write_synthetic_replenishment_data,
)


def _format_head(rows: list[dict[str, Any]], dataset_name: str, limit: int = 5) -> str:
    columns = list(REQUIRED_COLUMNS_BY_DATASET[dataset_name])
    preview_rows = rows[:limit]
    widths = {
        column: max(
            len(column),
            *(len(str(row[column])) for row in preview_rows),
        )
        for column in columns
    }
    header = "  ".join(column.ljust(widths[column]) for column in columns)
    separator = "  ".join("-" * widths[column] for column in columns)
    body = [
        "  ".join(str(row[column]).ljust(widths[column]) for column in columns)
        for row in preview_rows
    ]
    return "\n".join([header, separator, *body])


def main() -> None:
    bundle = write_synthetic_replenishment_data()

    for dataset_name, rows in bundle.as_dict().items():
        validate_required_columns(dataset_name, rows[0].keys() if rows else ())
        output_path = INPUT_CSV_PATHS[dataset_name]
        print(f"\n[{dataset_name}] wrote {len(rows):,} rows to {output_path.relative_to(REPO_ROOT)}")
        print(_format_head(rows, dataset_name))


if __name__ == "__main__":
    main()
