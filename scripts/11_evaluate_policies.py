#!/usr/bin/env python
"""Evaluate rule-based, HGB simulation, and lightweight RL policies."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGES_DIR = REPO_ROOT / "packages"
if str(PACKAGES_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGES_DIR))

from inventory_purchase_integrated.evaluation.policy_evaluator import (
    DEFAULT_POLICY_COMPARISON_PATH,
    write_policy_comparison,
)


def main() -> None:
    comparison = write_policy_comparison()

    print(
        f"\n[policy_comparison] wrote {len(comparison):,} rows to "
        f"{DEFAULT_POLICY_COMPARISON_PATH.relative_to(REPO_ROOT)}"
    )
    print("\ncomparison shape:")
    print(comparison.shape)
    print("\npolicy average selected_total_cost:")
    print(
        comparison.groupby("policy_name")["selected_total_cost"]
        .mean()
        .sort_index()
        .to_string()
    )
    print("\nSKU action comparison:")
    print(
        comparison.pivot_table(
            index=["sku_id", "warehouse_id"],
            columns="policy_name",
            values="selected_action_name",
            aggfunc="first",
        ).to_string()
    )


if __name__ == "__main__":
    main()
