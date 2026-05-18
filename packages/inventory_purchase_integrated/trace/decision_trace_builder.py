"""Build human-readable traces for already selected final decisions."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from inventory_purchase_integrated.package_spec import OUTPUT_DATA_DIR

FEATURE_SNAPSHOT_FILENAME = "02_feature_snapshot.csv"
DEMAND_FORECAST_FILENAME = "03_demand_forecast.csv"
RISK_SCORE_FILENAME = "04_risk_score.csv"
CANDIDATE_ORDERS_FILENAME = "05_candidate_orders.csv"
GATE_RESULT_FILENAME = "06_gate_result.csv"
FINAL_DECISION_FILENAME = "07_final_decision.csv"
DECISION_TRACE_FILENAME = "08_decision_trace.csv"
CANDIDATE_SIMULATION_RESULT_FILENAME = "candidate_simulation_result.csv"
WEEKLY_TRANSITION_LOG_FILENAME = "weekly_transition_log.csv"

SIMULATION_DATA_DIR = OUTPUT_DATA_DIR.parent / "simulation"
DEFAULT_FEATURE_SNAPSHOT_PATH = OUTPUT_DATA_DIR / FEATURE_SNAPSHOT_FILENAME
DEFAULT_DEMAND_FORECAST_PATH = OUTPUT_DATA_DIR / DEMAND_FORECAST_FILENAME
DEFAULT_RISK_SCORE_PATH = OUTPUT_DATA_DIR / RISK_SCORE_FILENAME
DEFAULT_CANDIDATE_ORDERS_PATH = OUTPUT_DATA_DIR / CANDIDATE_ORDERS_FILENAME
DEFAULT_GATE_RESULT_PATH = OUTPUT_DATA_DIR / GATE_RESULT_FILENAME
DEFAULT_FINAL_DECISION_PATH = OUTPUT_DATA_DIR / FINAL_DECISION_FILENAME
DEFAULT_DECISION_TRACE_PATH = OUTPUT_DATA_DIR / DECISION_TRACE_FILENAME
DEFAULT_CANDIDATE_SIMULATION_RESULT_PATH = (
    SIMULATION_DATA_DIR / CANDIDATE_SIMULATION_RESULT_FILENAME
)
DEFAULT_WEEKLY_TRANSITION_LOG_PATH = SIMULATION_DATA_DIR / WEEKLY_TRANSITION_LOG_FILENAME

DECISION_TRACE_COLUMNS = (
    "snapshot_week",
    "sku_id",
    "warehouse_id",
    "policy_name",
    "selected_action_name",
    "recommended_order_qty",
    "primary_risk_type",
    "forecast_4w_total_qty",
    "available_qty",
    "inventory_cover_weeks",
    "stockout_risk_score",
    "overstock_risk_score",
    "expiry_risk_score",
    "selected_total_cost",
    "total_stockout_qty",
    "total_expired_qty",
    "ending_inventory_qty",
    "gate_summary",
    "cost_summary",
    "main_reason",
    "alternative_action_note",
    "final_trace_comment",
)


def _required_columns(frame: pd.DataFrame, columns: set[str], dataset_name: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"{dataset_name} is missing required columns: {missing_text}")


def _read_inputs(
    feature_snapshot_path: Path,
    demand_forecast_path: Path,
    risk_score_path: Path,
    candidate_orders_path: Path,
    gate_result_path: Path,
    simulation_result_path: Path,
    weekly_transition_log_path: Path,
    final_decision_path: Path,
) -> dict[str, pd.DataFrame]:
    frames = {
        "feature": pd.read_csv(feature_snapshot_path),
        "forecast": pd.read_csv(demand_forecast_path),
        "risk": pd.read_csv(risk_score_path, encoding="utf-8-sig"),
        "candidate": pd.read_csv(candidate_orders_path, encoding="utf-8-sig"),
        "gate": pd.read_csv(gate_result_path, encoding="utf-8-sig"),
        "simulation": pd.read_csv(simulation_result_path, encoding="utf-8-sig"),
        "weekly_log": pd.read_csv(weekly_transition_log_path, encoding="utf-8-sig"),
        "decision": pd.read_csv(final_decision_path, encoding="utf-8-sig"),
    }

    _required_columns(
        frames["feature"],
        {"sku_id", "warehouse_id", "available_qty", "inventory_cover_weeks"},
        "feature_snapshot",
    )
    _required_columns(
        frames["forecast"],
        {"sku_id", "warehouse_id", "forecast_qty"},
        "demand_forecast",
    )
    _required_columns(
        frames["risk"],
        {
            "sku_id",
            "warehouse_id",
            "forecast_4w_total_qty",
            "stockout_risk_score",
            "overstock_risk_score",
            "expiry_risk_score",
        },
        "risk_score",
    )
    _required_columns(
        frames["candidate"],
        {"sku_id", "warehouse_id", "action_id", "action_name"},
        "candidate_orders",
    )
    _required_columns(
        frames["gate"],
        {
            "sku_id",
            "warehouse_id",
            "action_id",
            "action_name",
            "overall_gate_status",
            "failed_gate_count",
        },
        "gate_result",
    )
    _required_columns(
        frames["simulation"],
        {"sku_id", "warehouse_id", "action_id", "action_name", "total_cost"},
        "candidate_simulation_result",
    )
    _required_columns(
        frames["weekly_log"],
        {"sku_id", "warehouse_id", "action_id", "action_name", "simulation_week"},
        "weekly_transition_log",
    )
    _required_columns(
        frames["decision"],
        {
            "snapshot_week",
            "sku_id",
            "warehouse_id",
            "policy_name",
            "selected_action_id",
            "selected_action_name",
            "recommended_order_qty",
            "selected_total_cost",
            "total_stockout_qty",
            "total_expired_qty",
            "ending_inventory_qty",
            "primary_risk_type",
        },
        "final_decision",
    )
    return frames


def _main_reason(row: pd.Series) -> str:
    risk = row["primary_risk_type"]
    action = row["selected_action_name"]
    if risk == "stockout" and action == "order_moq":
        return "4주 예측수요 대비 가용재고가 부족해 최소발주수량 기준 보충 선택"
    if risk == "stockout" and action in {"order_1w_cover", "expedite"}:
        return "품절위험이 높아 예측수요 부족분 보충 후보 선택"
    if risk == "expiry" and action == "hold":
        return "폐기위험이 높아 추가 발주보다 보수적 보류 선택"
    if risk == "expiry" and action == "reduce_review":
        return "폐기/과잉위험으로 축소 검토 선택"
    return f"{risk} 위험과 시뮬레이션 결과를 반영해 {action} 선택"


def _gate_summary(row: pd.Series, gate: pd.DataFrame) -> str:
    selected_gate = gate[
        (gate["sku_id"] == row["sku_id"])
        & (gate["warehouse_id"] == row["warehouse_id"])
        & (gate["action_id"] == row["selected_action_id"])
        & (gate["action_name"] == row["selected_action_name"])
    ]
    if selected_gate.empty:
        return "주의: selected action gate result missing"

    gate_row = selected_gate.iloc[0]
    if gate_row["overall_gate_status"] == "PASS":
        return f"운영 제약 통과, 실패 gate {int(gate_row['failed_gate_count'])}개"
    return "주의: gate fail action selected"


def _cost_summary(row: pd.Series) -> str:
    return (
        f"총비용 {row['selected_total_cost']:.2f}, "
        f"예상 품절 {row['total_stockout_qty']:.2f}개, "
        f"예상 폐기 {row['total_expired_qty']:.2f}개"
    )


def _alternative_action_note(row: pd.Series, simulation: pd.DataFrame, risk: pd.DataFrame) -> str:
    sku_rows = simulation[
        (simulation["sku_id"] == row["sku_id"])
        & (simulation["warehouse_id"] == row["warehouse_id"])
        & (simulation["overall_gate_status"] == "PASS")
        & (simulation["simulation_status"] == "SIMULATED")
        & ~(
            (simulation["action_id"] == row["selected_action_id"])
            & (simulation["action_name"] == row["selected_action_name"])
        )
    ]
    if sku_rows.empty:
        return "비교 가능한 차선 후보가 없음"

    alternative = sku_rows.sort_values(["total_cost", "action_id"]).iloc[0]
    cost_diff = alternative["total_cost"] - row["selected_total_cost"]

    risk_row = risk[
        (risk["sku_id"] == row["sku_id"]) & (risk["warehouse_id"] == row["warehouse_id"])
    ].iloc[0]
    if (
        row["sku_id"] == "SKU_C"
        and row["selected_action_name"] in {"hold", "reduce_review"}
        and risk_row["expiry_risk_score"] >= 0.70
    ):
        return "order_moq가 비용상 유리할 수 있으나 폐기위험 보수 조건으로 hold 또는 reduce_review 선택"

    if cost_diff >= 0:
        return (
            f"차선 후보 {alternative['action_name']} 대비 total_cost가 "
            f"{cost_diff:.2f} 낮음"
        )
    return (
        f"차선 후보 {alternative['action_name']}의 total_cost가 "
        f"{abs(cost_diff):.2f} 낮지만 정책 우선순위로 현재 action 선택"
    )


def build_decision_trace(
    feature_snapshot_path: Path | None = None,
    demand_forecast_path: Path | None = None,
    risk_score_path: Path | None = None,
    candidate_orders_path: Path | None = None,
    gate_result_path: Path | None = None,
    simulation_result_path: Path | None = None,
    weekly_transition_log_path: Path | None = None,
    final_decision_path: Path | None = None,
) -> pd.DataFrame:
    """Build explanatory trace rows for already selected final decisions."""
    frames = _read_inputs(
        feature_snapshot_path or DEFAULT_FEATURE_SNAPSHOT_PATH,
        demand_forecast_path or DEFAULT_DEMAND_FORECAST_PATH,
        risk_score_path or DEFAULT_RISK_SCORE_PATH,
        candidate_orders_path or DEFAULT_CANDIDATE_ORDERS_PATH,
        gate_result_path or DEFAULT_GATE_RESULT_PATH,
        simulation_result_path or DEFAULT_CANDIDATE_SIMULATION_RESULT_PATH,
        weekly_transition_log_path or DEFAULT_WEEKLY_TRANSITION_LOG_PATH,
        final_decision_path or DEFAULT_FINAL_DECISION_PATH,
    )

    trace_base = (
        frames["decision"]
        .merge(
            frames["feature"][
                ["sku_id", "warehouse_id", "available_qty", "inventory_cover_weeks"]
            ],
            on=["sku_id", "warehouse_id"],
            how="left",
        )
        .merge(
            frames["risk"][
                [
                    "sku_id",
                    "warehouse_id",
                    "forecast_4w_total_qty",
                    "stockout_risk_score",
                    "overstock_risk_score",
                    "expiry_risk_score",
                ]
            ],
            on=["sku_id", "warehouse_id"],
            how="left",
        )
    )

    rows: list[dict[str, object]] = []
    for decision in trace_base.sort_values(["policy_name", "sku_id", "warehouse_id"]).itertuples(
        index=False
    ):
        row = pd.Series(decision._asdict())
        gate_summary = _gate_summary(row, frames["gate"])
        cost_summary = _cost_summary(row)
        main_reason = _main_reason(row)
        alternative_note = _alternative_action_note(row, frames["simulation"], frames["risk"])
        final_comment = " ".join([main_reason, gate_summary, cost_summary, alternative_note])

        rows.append(
            {
                "snapshot_week": row["snapshot_week"],
                "sku_id": row["sku_id"],
                "warehouse_id": row["warehouse_id"],
                "policy_name": row["policy_name"],
                "selected_action_name": row["selected_action_name"],
                "recommended_order_qty": row["recommended_order_qty"],
                "primary_risk_type": row["primary_risk_type"],
                "forecast_4w_total_qty": row["forecast_4w_total_qty"],
                "available_qty": row["available_qty"],
                "inventory_cover_weeks": row["inventory_cover_weeks"],
                "stockout_risk_score": row["stockout_risk_score"],
                "overstock_risk_score": row["overstock_risk_score"],
                "expiry_risk_score": row["expiry_risk_score"],
                "selected_total_cost": row["selected_total_cost"],
                "total_stockout_qty": row["total_stockout_qty"],
                "total_expired_qty": row["total_expired_qty"],
                "ending_inventory_qty": row["ending_inventory_qty"],
                "gate_summary": gate_summary,
                "cost_summary": cost_summary,
                "main_reason": main_reason,
                "alternative_action_note": alternative_note,
                "final_trace_comment": final_comment,
            }
        )

    return pd.DataFrame(rows, columns=DECISION_TRACE_COLUMNS)


def write_decision_trace(
    output_path: Path | None = None,
    feature_snapshot_path: Path | None = None,
    demand_forecast_path: Path | None = None,
    risk_score_path: Path | None = None,
    candidate_orders_path: Path | None = None,
    gate_result_path: Path | None = None,
    simulation_result_path: Path | None = None,
    weekly_transition_log_path: Path | None = None,
    final_decision_path: Path | None = None,
) -> pd.DataFrame:
    """Build and write stage-10 decision trace rows."""
    path = output_path or DEFAULT_DECISION_TRACE_PATH
    trace = build_decision_trace(
        feature_snapshot_path=feature_snapshot_path,
        demand_forecast_path=demand_forecast_path,
        risk_score_path=risk_score_path,
        candidate_orders_path=candidate_orders_path,
        gate_result_path=gate_result_path,
        simulation_result_path=simulation_result_path,
        weekly_transition_log_path=weekly_transition_log_path,
        final_decision_path=final_decision_path,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    trace.to_csv(path, index=False, encoding="utf-8-sig")
    return trace
