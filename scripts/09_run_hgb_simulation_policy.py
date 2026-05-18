#!/usr/bin/env python
"""Run stage-9 rule-based and HGB simulation policies."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGES_DIR = REPO_ROOT / "packages"
if str(PACKAGES_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGES_DIR))

from inventory_purchase_integrated.decision.hgb_simulation_policy import (
    DEFAULT_FINAL_DECISION_PATH,
    write_final_decisions,
)


def main() -> None:
    decisions = write_final_decisions()

    print(
        f"\n[final_decision] wrote {len(decisions):,} rows to "
        f"{DEFAULT_FINAL_DECISION_PATH.relative_to(REPO_ROOT)}"
    )
    print("\nhead:")
    print(decisions.head().to_string(index=False))
    print("\nshape:")
    print(decisions.shape)
    print("\npolicy action distribution:")
    print(
        decisions.groupby(["policy_name", "selected_action_name"])
        .size()
        .to_string()
    )


if __name__ == "__main__":
    main()
