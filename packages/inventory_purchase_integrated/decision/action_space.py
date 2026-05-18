"""Fixed action space for stage-6 replenishment candidate generation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Action:
    """A candidate action label, not a final replenishment decision."""

    action_id: int
    action_name: str
    reason: str


ACTIONS = (
    Action(0, "hold", "현재 후보에서는 발주하지 않는 기준안"),
    Action(1, "order_moq", "최소발주수량 기준 후보"),
    Action(2, "order_1w_cover", "1주 예측수요 부족분 보충 후보"),
    Action(3, "order_2w_cover", "2주 예측수요 부족분 보충 후보"),
    Action(4, "expedite", "품절위험 대응 긴급발주 후보"),
    Action(5, "reduce_review", "폐기/과잉위험으로 발주 축소 검토 후보"),
)

ACTION_NAMES = tuple(action.action_name for action in ACTIONS)
