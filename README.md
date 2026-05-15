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
