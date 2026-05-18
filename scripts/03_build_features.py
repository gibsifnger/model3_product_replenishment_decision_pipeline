#!/usr/bin/env python
"""Build stage-3 current-state feature snapshot and print a preview."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGES_DIR = REPO_ROOT / "packages"
if str(PACKAGES_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGES_DIR))

from inventory_purchase_integrated.state.feature_builder import (
    DEFAULT_FEATURE_SNAPSHOT_PATH,
    write_feature_snapshot,
)


def main() -> None:
    features = write_feature_snapshot()
    output_path = DEFAULT_FEATURE_SNAPSHOT_PATH

    print(
        f"\n[feature_snapshot] wrote {len(features):,} rows to "
        f"{output_path.relative_to(REPO_ROOT)}"
    )
    print("\nhead:")
    print(features.head().to_string(index=False))
    print("\nshape:")
    print(features.shape)


if __name__ == "__main__":
    main()
