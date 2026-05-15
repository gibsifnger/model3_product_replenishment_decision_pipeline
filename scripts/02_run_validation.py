#!/usr/bin/env python
"""Run stage-2 input quality validation and print validation results."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGES_DIR = REPO_ROOT / "packages"
if str(PACKAGES_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGES_DIR))

from inventory_purchase_integrated.validation.input_quality_validator import (
    DEFAULT_QUALITY_CHECK_PATH,
    QUALITY_CHECK_COLUMNS,
    write_input_quality_check,
)


def _format_head(rows: list[dict[str, Any]], limit: int = 10) -> str:
    preview_rows = rows[:limit]
    widths = {
        column: max(
            len(column),
            *(len(str(row[column])) for row in preview_rows),
        )
        for column in QUALITY_CHECK_COLUMNS
    }
    header = "  ".join(column.ljust(widths[column]) for column in QUALITY_CHECK_COLUMNS)
    separator = "  ".join("-" * widths[column] for column in QUALITY_CHECK_COLUMNS)
    body = [
        "  ".join(str(row[column]).ljust(widths[column]) for column in QUALITY_CHECK_COLUMNS)
        for row in preview_rows
    ]
    return "\n".join([header, separator, *body])


def _status_summary(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "PASS": sum(1 for row in rows if row["status"] == "PASS"),
        "FAIL": sum(1 for row in rows if row["status"] == "FAIL"),
    }


def main() -> None:
    rows = write_input_quality_check()
    output_path = DEFAULT_QUALITY_CHECK_PATH
    summary = _status_summary(rows)

    print(
        f"\n[input_quality_check] wrote {len(rows):,} rows to "
        f"{output_path.relative_to(REPO_ROOT)}"
    )
    print("\nhead:")
    print(_format_head(rows))
    print("\nPASS/FAIL summary:")
    print(f"PASS: {summary['PASS']}")
    print(f"FAIL: {summary['FAIL']}")


if __name__ == "__main__":
    main()
