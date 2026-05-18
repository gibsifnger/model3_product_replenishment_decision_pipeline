#!/usr/bin/env python
"""Run stage-7 gate checks for candidate order rows."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGES_DIR = REPO_ROOT / "packages"
if str(PACKAGES_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGES_DIR))

from inventory_purchase_integrated.decision.gate_checker import (
    DEFAULT_GATE_RESULT_PATH,
    write_gate_result,
)


def main() -> None:
    gate_result = write_gate_result()
    output_path = DEFAULT_GATE_RESULT_PATH

    print(
        f"\n[gate_result] wrote {len(gate_result):,} rows to "
        f"{output_path.relative_to(REPO_ROOT)}"
    )
    print("\nhead:")
    print(gate_result.head().to_string(index=False))
    print("\nshape:")
    print(gate_result.shape)
    print("\noverall_gate_status distribution:")
    print(gate_result["overall_gate_status"].value_counts().sort_index().to_string())
    print("\naction PASS/FAIL summary:")
    print(
        gate_result.pivot_table(
            index="action_name",
            columns="overall_gate_status",
            values="sku_id",
            aggfunc="count",
            fill_value=0,
        ).to_string()
    )


if __name__ == "__main__":
    main()
