# model3_product_replenishment_decision_pipeline

Stage 1 of this repository builds the input-data foundation for a digital-twin-style SCM simulator for food and consumer-goods replenishment decisions.

## Fixed scope

- **Domain:** food/consumer-goods distribution replenishment
- **Row unit:** SKU × warehouse × week
- **SKUs:**
  - `SKU_A`: stable demand with low sales variance
  - `SKU_B`: promotion-sensitive demand with large promotion-week lift
  - `SKU_C`: slow-moving, short-shelf-life item with waste / excess-inventory risk
- **Warehouse:** `DC_01`
- **Historical window:** 52 weeks
- **Decision cadence:** weekly
- **Future forecast horizon:** 4 weeks

## Stage-1 files

```text
packages/inventory_purchase_integrated/package_spec.py
packages/inventory_purchase_integrated/schema.py
packages/inventory_purchase_integrated/data_generation/synthetic_replenishment_data.py
scripts/01_generate_synthetic_data.py
requirements.txt
README.md
```

The synthetic data generator only creates input data for later validation, feature, forecasting, and simulation stages. It does **not** perform purchase-order decisions, and it does **not** contain simulator, policy, or gate logic.

## Generated input CSVs

Running the stage-1 script writes these files under `data/input/`:

1. `sku_master.csv`
2. `weekly_sales_inventory.csv`
3. `promotion_calendar.csv`
4. `supplier_constraints.csv`

`weekly_sales_inventory.csv` contains `3 SKUs × 1 warehouse × 52 weeks = 156 rows`.

## Quick start

```bash
python -m pip install -r requirements.txt
python scripts/01_generate_synthetic_data.py
```

The script validates required columns from `schema.py`, writes all four CSVs, and prints each generated dataset head.

## Stage 2 validation

```bash
python scripts/02_run_validation.py
```

The validation script checks the four input CSVs against the schema contract and writes `data/output/01_input_quality_check.csv`.

## Stage 3 feature snapshot

```bash
python scripts/03_build_features.py
```

The feature builder reads the four input CSVs and writes the current-state `sku_id × warehouse_id` snapshot to `data/output/02_feature_snapshot.csv`.

## Stage 4 demand forecast

```bash
python scripts/04_run_demand_forecast.py
```

The HGB forecaster reads sales, promotion, and feature snapshot inputs, then writes the 4-week demand forecast to `data/output/03_demand_forecast.csv`.

## Stage 5 risk score

```bash
python scripts/05_run_risk_score.py
```

The risk model reads the feature snapshot and demand forecast, then writes SKU-warehouse risk scores to `data/output/04_risk_score.csv`.

## Stage 6 candidate orders

```bash
python scripts/06_generate_candidates.py
```

The candidate generator expands each SKU-warehouse row across fixed actions and writes order candidates to `data/output/05_candidate_orders.csv`.

## Stage 7 gate check

```bash
python scripts/07_run_gate_check.py
```

The gate checker validates candidate orders against operational constraints and writes pass/fail results to `data/output/06_gate_result.csv`.

## Stage 8 digital twin simulation

```bash
python scripts/08_run_simulation.py
```

The simulator evaluates gate-passed candidate actions over four future weeks and writes results to `data/simulation/`.
