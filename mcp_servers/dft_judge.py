from __future__ import annotations

from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

from mcp_servers.common import MCPException, MCPResponse, success_response
from trainer.demo_api import coating_assessment


router = APIRouter(prefix="/dft_judge", tags=["dft_judge"])


class DFTSpec(BaseModel):
    nominal: float | None = None
    # 학습된 호출 모양을 받기만 한다. 판정에는 쓰지 않는다(PR #8 기준).
    tolerance_rule: Literal["90/10"] | None = None


class DFTEvaluateRequest(BaseModel):
    measurements: list[float] = Field(min_length=1)
    spec: DFTSpec | None = None


class Violation(BaseModel):
    point_index: int
    value: float
    threshold: float
    reason: str


@router.post("/evaluate", response_model=MCPResponse)
def evaluate_dft(payload: DFTEvaluateRequest) -> MCPResponse:
    nominal = payload.spec.nominal if payload.spec else None
    if nominal is not None and nominal <= 0:
        raise MCPException(
            code="INVALID_PARAMS",
            message="nominal must be greater than zero",
            details={"nominal": nominal},
            status_code=400,
        )

    assessed = coating_assessment.assess_readings(payload.measurements, nominal)
    grade = assessed["grade"]
    target = assessed["target_thickness_um"]
    mean_um = assessed["painted_mean_um"]
    violations: list[Violation] = []
    if grade == "below":
        violations.append(Violation(point_index=-1, value=mean_um, threshold=target, reason="mean_below_target"))
    elif grade == "above":
        violations.append(Violation(
            point_index=-1, value=mean_um, threshold=assessed["normal_band_um"][1], reason="mean_at_or_above_over_limit"))

    result = {
        "pass_": grade == "normal",
        "verdict": "unavailable" if grade is None else ("pass" if grade == "normal" else "fail"),
        "grade": grade,
        "status": assessed["status"],
        "violations": [item.model_dump() for item in violations],
        "mean": mean_um,
        "nominal": target,
        "target_basis": assessed["target_basis"],
        "normal_band_um": assessed["normal_band_um"],
        "six_band_counts": assessed["six_band_counts"],
        "coverage_pct": assessed["coverage_percent"],
        "out_of_spec_count": sum(1 for g in assessed["point_grades"] if g != "normal"),
        "measurements": payload.measurements,
        "rule": coating_assessment.ASSESSMENT_VERSION,
    }
    return success_response(result=result)
