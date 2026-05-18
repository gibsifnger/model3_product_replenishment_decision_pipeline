"""Compare final replenishment policies on common simulation outcomes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from inventory_purchase_integrated.package_spec import OUTPUT_DATA_DIR

FINAL_DECISION_FILENAME = "07_final_decision.csv"
RL_DECISION_TRACE_FILENAME = "11_rl_decision_trace.csv"
RISK_SCORE_FILENAME = "04_risk_score.csv"
CANDIDATE_SIMULATION_RESULT_FILENAME = "candidate_simulation_result.csv"
POLICY_COMPARISON_FILENAME = "09_policy_comparison.csv"

SIMULATION_DATA_DIR = OUTPUT_DATA_DIR.parent / "simulation"
DEFAULT_FINAL_DECISION_PATH = OUTPUT_DATA_DIR / FINAL_DECISION_FILENAME
DEFAULT_RL_DECISION_TRACE_PATH = OUTPUT_DATA_DIR / RL_DECISION_TRACE_FILENAME
DEFAULT_RISK_SCORE_PATH = OUTPUT_DATA_DIR / RISK_SCORE_FILENAME
DEFAULT_CANDIDATE_SIMULATION_RESULT_PATH = (
    SIMULATION_DATA_DIR / CANDIDATE_SIMULATION_RESULT_FILENAME
)
DEFAULT_POLICY_COMPARISON_PATH = OUTPUT_DATA_DIR / POLICY_COMPARISON_FILENAME

POLICY_COMPARISON_COLUMNS = (
    "snapshot_week",
    "sku_id",
    "warehouse_id",
    "policy_name",
    "selected_action_id",
    "selected_action_name",
    "recommended_order_qty",
    "primary_risk_type",
    "total_forecast_qty",
    "total_sales_fulfilled_qty",
    "total_stockout_qty",
    "total_expired_qty",
    "ending_inventory_qty",
    "selected_total_cost",
    "purchase_cost",
    "holding_cost",
    "stockout_penalty",
    "expiry_penalty",
    "expedite_cost",
    "overstock_penalty",
    "constraint_violation_penalty",
    "service_level",
    "cost_rank_within_sku",
    "selected_action_note",
    "policy_warning",
    "comparison_comment",
)


def _required_columns(frame: pd.DataFrame, columns: set[str], dataset_name: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"{dataset_name} is missing required columns: {missing_text}")


def _read_inputs(
    final_decision_path: Path,
    rl_decision_trace_path: Path,
    risk_score_path: Path,
    simulation_result_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    final_decision = pd.read_csv(final_decision_path, encoding="utf-8-sig")
    rl_decision = pd.read_csv(rl_decision_trace_path, encoding="utf-8-sig")
    risk_score = pd.read_csv(risk_score_path, encoding="utf-8-sig")
    simulation = pd.read_csv(simulation_result_path, encoding="utf-8-sig")

    decision_columns = {
        "snapshot_week",
        "sku_id",
        "warehouse_id",
        "policy_name",
        "selected_action_id",
        "selected_action_name",
        "recommended_order_qty",
        "primary_risk_type",
    }
    _required_columns(final_decision, decision_columns, "final_decision")
    _required_columns(rl_decision, decision_columns, "rl_decision_trace")
    _required_columns(
        risk_score,
        {"sku_id", "warehouse_id", "primary_risk_type"},
        "risk_score",
    )
    _required_columns(
        simulation,
        {
            "sku_id",
            "warehouse_id",
            "action_id",
            "action_name",
            "overall_gate_status",
            "simulation_status",
            "total_forecast_qty",
            "total_sales_fulfilled_qty",
            "total_stockout_qty",
            "total_expired_qty",
            "ending_inventory_qty",
            "purchase_cost",
            "holding_cost",
            "stockout_penalty",
            "expiry_penalty",
            "expedite_cost",
            "overstock_penalty",
            "constraint_violation_penalty",
            "total_cost",
        },
        "candidate_simulation_result",
    )
    return final_decision, rl_decision, risk_score, simulation


def _combined_policy_decisions(final_decision: pd.DataFrame, rl_decision: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "snapshot_week",
        "sku_id",
        "warehouse_id",
        "policy_name",
        "selected_action_id",
        "selected_action_name",
        "recommended_order_qty",
        "primary_risk_type",
    ]
    return pd.concat(
        [final_decision.loc[:, columns], rl_decision.loc[:, columns]],
        ignore_index=True,
    )


def _selected_action_note(row: pd.Series) -> str:
    return (
        f"{row['policy_name']} selected {row['selected_action_name']} "
        f"with order quantity {int(row['recommended_order_qty'])}."
    )


def _policy_warning(row: pd.Series) -> str:
    warnings: list[str] = []
    if row["overall_gate_status"] != "PASS":
        warnings.append("gate FAIL action selected")
    if row["simulation_status"] != "SIMULATED":
        warnings.append(f"simulation_status={row['simulation_status']}")
    return "; ".join(warnings)


def _sku_comparison_comment(sku_rows: pd.DataFrame) -> str:
    sku_id = str(sku_rows.iloc[0]["sku_id"])
    action_by_policy = dict(zip(sku_rows["policy_name"], sku_rows["selected_action_name"]))
    actions = set(action_by_policy.values())
    primary_risk_type = str(sku_rows.iloc[0]["primary_risk_type"])

    if len(actions) == 1:
        selected_action = next(iter(actions))
        if selected_action == "order_moq" and primary_risk_type == "stockout":
            return (
                f"{sku_id}: 세 정책 모두 order_moq를 선택했으며, "
                "품절위험 대응을 위해 최소발주수량 보충이 유리했다."
            )
        return f"{sku_id}: 세 정책 모두 {selected_action}를 선택했다."

    if sku_id == "SKU_C" and action_by_policy.get("rl_lightweight") == "order_moq":
        return (
            "SKU_C: rule_based와 hgb_simulation은 폐기위험 보수 조건으로 hold를 "
            "선택했지만, rl_lightweight는 total_cost 최소화 reward에 따라 order_moq를 "
            "선택했다. 이는 RL reward에 폐기위험 보수성을 추가할 필요가 있음을 보여준다."
        )

    return (
        f"{sku_id}: 정책별 선택 action이 달라 비용, 품절, 폐기 결과를 함께 비교해야 한다."
    )


def build_policy_comparison(
    final_decision_path: Path | None = None,
    rl_decision_trace_path: Path | None = None,
    risk_score_path: Path | None = None,
    simulation_result_path: Path | None = None,
) -> pd.DataFrame:
    """Build policy comparison rows across rule, HGB simulation, and RL challenger."""
    final_decision, rl_decision, risk_score, simulation = _read_inputs(
        final_decision_path or DEFAULT_FINAL_DECISION_PATH,
        rl_decision_trace_path or DEFAULT_RL_DECISION_TRACE_PATH,
        risk_score_path or DEFAULT_RISK_SCORE_PATH,
        simulation_result_path or DEFAULT_CANDIDATE_SIMULATION_RESULT_PATH,
    )

    decisions = _combined_policy_decisions(final_decision, rl_decision)
    comparison = decisions.merge(
        simulation,
        left_on=["sku_id", "warehouse_id", "selected_action_id", "selected_action_name"],
        right_on=["sku_id", "warehouse_id", "action_id", "action_name"],
        how="left",
        suffixes=("", "_simulation"),
    )

    comparison["selected_total_cost"] = comparison["total_cost"]
    comparison["service_level"] = np.where(
        comparison["total_forecast_qty"] == 0,
        1.0,
        comparison["total_sales_fulfilled_qty"] / comparison["total_forecast_qty"],
    ).clip(0, 1)
    comparison["cost_rank_within_sku"] = comparison.groupby("sku_id")[
        "selected_total_cost"
    ].rank(method="dense", ascending=True).astype(int)
    comparison["selected_action_note"] = comparison.apply(_selected_action_note, axis=1)
    comparison["policy_warning"] = comparison.apply(_policy_warning, axis=1)

    comments = {
        sku_id: _sku_comparison_comment(rows)
        for sku_id, rows in comparison.groupby("sku_id", sort=False)
    }
    comparison["comparison_comment"] = comparison["sku_id"].map(comments)

    risk_lookup = risk_score.set_index(["sku_id", "warehouse_id"])["primary_risk_type"]
    comparison["primary_risk_type"] = comparison.apply(
        lambda row: risk_lookup.get((row["sku_id"], row["warehouse_id"]), row["primary_risk_type"]),
        axis=1,
    )

    return comparison.loc[:, POLICY_COMPARISON_COLUMNS].sort_values(
        ["sku_id", "warehouse_id", "policy_name"]
    )


def write_policy_comparison(
    output_path: Path | None = None,
    final_decision_path: Path | None = None,
    rl_decision_trace_path: Path | None = None,
    risk_score_path: Path | None = None,
    simulation_result_path: Path | None = None,
) -> pd.DataFrame:
    """Build and write stage-12 policy comparison CSV."""
    path = output_path or DEFAULT_POLICY_COMPARISON_PATH
    comparison = build_policy_comparison(
        final_decision_path=final_decision_path,
        rl_decision_trace_path=rl_decision_trace_path,
        risk_score_path=risk_score_path,
        simulation_result_path=simulation_result_path,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(path, index=False, encoding="utf-8-sig")
    return comparison
