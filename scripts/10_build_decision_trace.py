#!/usr/bin/env python
"""Build stage-10 human-readable final decision traces."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGES_DIR = REPO_ROOT / "packages"
if str(PACKAGES_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGES_DIR))

from inventory_purchase_integrated.trace.decision_trace_builder import (
    DEFAULT_DECISION_TRACE_PATH,
    write_decision_trace,
)


def main() -> None:
    trace = write_decision_trace()

    print(
        f"\n[decision_trace] wrote {len(trace):,} rows to "
        f"{DEFAULT_DECISION_TRACE_PATH.relative_to(REPO_ROOT)}"
    )
    print("\nhead:")
    print(trace.head().to_string(index=False))
    print("\nshape:")
    print(trace.shape)
    print("\nSKU trace:")
    for row in trace.sort_values(["sku_id", "policy_name"]).itertuples(index=False):
        print(f"{row.sku_id} / {row.policy_name}: {row.final_trace_comment}")


if __name__ == "__main__":
    main()
