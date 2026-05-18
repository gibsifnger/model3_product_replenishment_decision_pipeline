#!/usr/bin/env python
"""Generate stage-6 replenishment candidate order rows."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGES_DIR = REPO_ROOT / "packages"
if str(PACKAGES_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGES_DIR))

from inventory_purchase_integrated.decision.candidate_generator import (
    DEFAULT_CANDIDATE_ORDERS_PATH,
    write_candidate_orders,
)


def main() -> None:
    candidates = write_candidate_orders()
    output_path = DEFAULT_CANDIDATE_ORDERS_PATH

    print(
        f"\n[candidate_orders] wrote {len(candidates):,} rows to "
        f"{output_path.relative_to(REPO_ROOT)}"
    )
    print("\nhead:")
    print(candidates.head().to_string(index=False))
    print("\nshape:")
    print(candidates.shape)
    print("\naction row count:")
    print(candidates["action_name"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
