#!/usr/bin/env python
"""Run stage-8 digital twin simulation for candidate order rows."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGES_DIR = REPO_ROOT / "packages"
if str(PACKAGES_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGES_DIR))

from inventory_purchase_integrated.simulation.simulator import (
    DEFAULT_CANDIDATE_SIMULATION_RESULT_PATH,
    DEFAULT_WEEKLY_TRANSITION_LOG_PATH,
    write_simulation_outputs,
)


def main() -> None:
    candidate_result, weekly_log = write_simulation_outputs()

    print(
        f"\n[candidate_simulation_result] wrote {len(candidate_result):,} rows to "
        f"{DEFAULT_CANDIDATE_SIMULATION_RESULT_PATH.relative_to(REPO_ROOT)}"
    )
    print(
        f"[weekly_transition_log] wrote {len(weekly_log):,} rows to "
        f"{DEFAULT_WEEKLY_TRANSITION_LOG_PATH.relative_to(REPO_ROOT)}"
    )
    print("\ncandidate result head:")
    print(candidate_result.head().to_string(index=False))
    print("\ncandidate result shape:")
    print(candidate_result.shape)
    print("\nsimulation_status distribution:")
    print(candidate_result["simulation_status"].value_counts().sort_index().to_string())
    print("\naction total_cost summary:")
    print(
        candidate_result.groupby("action_name")["total_cost"]
        .agg(["min", "mean", "max"])
        .to_string()
    )


if __name__ == "__main__":
    main()
