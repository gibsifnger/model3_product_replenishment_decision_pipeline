#!/usr/bin/env python
"""Run stage-5 replenishment risk scoring and print risk results."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGES_DIR = REPO_ROOT / "packages"
if str(PACKAGES_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGES_DIR))

from inventory_purchase_integrated.state.risk_model import (
    DEFAULT_RISK_SCORE_PATH,
    write_risk_score,
)


def main() -> None:
    risk = write_risk_score()
    output_path = DEFAULT_RISK_SCORE_PATH

    print(
        f"\n[risk_score] wrote {len(risk):,} rows to "
        f"{output_path.relative_to(REPO_ROOT)}"
    )
    print("\nhead:")
    print(risk.head().to_string(index=False))
    print("\nshape:")
    print(risk.shape)
    print("\nprimary_risk_type distribution:")
    print(risk["primary_risk_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
