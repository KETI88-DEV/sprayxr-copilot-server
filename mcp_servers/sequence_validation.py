from __future__ import annotations

from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

from mcp_servers.common import MCPResponse, success_response


router = APIRouter(prefix="/sequence_validation", tags=["sequence_validation"])


class StrokePoint(BaseModel):
    x: float
    y: float
    z: float
    timestamp: float


class SequenceValidationRequest(BaseModel):
    trajectory: list[StrokePoint] = Field(min_length=1)
    zone_id: str
    zone_type: Literal["수직 패널", "수평 패널", "곡면", "모서리"]
    reference_pattern: Literal[
        "top_to_bottom",
        "bottom_to_top",
        "left_to_right",
        "right_to_left",
    ]


def _compute_direction(trajectory: list[StrokePoint], zone_type: str) -> str:
    if zone_type == "수직 패널":
        decreasing_count = 0
        increasing_count = 0
        for idx in range(1, len(trajectory)):
            delta = trajectory[idx].y - trajectory[idx - 1].y
            if delta < 0:
                decreasing_count += 1
            elif delta > 0:
                increasing_count += 1
        return "top_to_bottom" if decreasing_count >= increasing_count else "bottom_to_top"
    return "left_to_right"


@router.post("/validate", response_model=MCPResponse)
def validate_sequence(payload: SequenceValidationRequest) -> MCPResponse:
    if len(payload.trajectory) < 2:
        result = {
            "zone_id": payload.zone_id,
            "zone_type": payload.zone_type,
            "computed_direction": None,
            "reference_pattern": payload.reference_pattern,
            "is_valid": None,
            "insufficient_data": True,
            "message": "최소 2개 이상의 trajectory 포인트가 필요합니다.",
            "violations": [],
        }
        return success_response(result=result)

    computed_direction = _compute_direction(payload.trajectory, payload.zone_type)
    violation = computed_direction != payload.reference_pattern
    result = {
        "zone_id": payload.zone_id,
        "zone_type": payload.zone_type,
        "computed_direction": computed_direction,
        "reference_pattern": payload.reference_pattern,
        "is_valid": not violation,
        "insufficient_data": False,
        "violations": (
            [
                {
                    "reason": "direction_mismatch",
                    "expected": payload.reference_pattern,
                    "actual": computed_direction,
                }
            ]
            if violation
            else []
        ),
    }
    return success_response(result=result)
