#!/usr/bin/env python
"""Run stage-4 HGB demand forecast and print forecast results."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGES_DIR = REPO_ROOT / "packages"
if str(PACKAGES_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGES_DIR))

from inventory_purchase_integrated.forecasting.hgb_demand_forecaster import (
    DEFAULT_DEMAND_FORECAST_PATH,
    write_demand_forecast,
)


def main() -> None:
    forecast = write_demand_forecast()
    output_path = DEFAULT_DEMAND_FORECAST_PATH

    print(
        f"\n[demand_forecast] wrote {len(forecast):,} rows to "
        f"{output_path.relative_to(REPO_ROOT)}"
    )
    print("\nhead:")
    print(forecast.head().to_string(index=False))
    print("\nshape:")
    print(forecast.shape)
    print("\nforecast_qty by SKU:")
    print(
        forecast.groupby("sku_id")["forecast_qty"]
        .agg(["min", "mean", "max", "sum"])
        .to_string()
    )


if __name__ == "__main__":
    main()
