from __future__ import annotations

from statistics import mean
from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

from mcp_servers.common import MCPException, MCPResponse, success_response


router = APIRouter(prefix="/dft_judge", tags=["dft_judge"])


class DFTSpec(BaseModel):
    nominal: float
    tolerance_rule: Literal["90/10"]


class DFTEvaluateRequest(BaseModel):
    measurements: list[float] = Field(min_length=1)
    spec: DFTSpec


class Violation(BaseModel):
    point_index: int
    value: float
    threshold: float
    reason: str


@router.post("/evaluate", response_model=MCPResponse)
def evaluate_dft(payload: DFTEvaluateRequest) -> MCPResponse:
    if payload.spec.nominal <= 0:
        raise MCPException(
            code="INVALID_PARAMS",
            message="nominal must be greater than zero",
            details={"nominal": payload.spec.nominal},
            status_code=400,
        )

    threshold = payload.spec.nominal * 0.9
    measurement_mean = mean(payload.measurements)
    violations: list[Violation] = []

    for idx, value in enumerate(payload.measurements):
        if value < threshold:
            violations.append(
                Violation(
                    point_index=idx,
                    value=value,
                    threshold=threshold,
                    reason="below_90_percent_of_nominal",
                )
            )

    if measurement_mean < payload.spec.nominal:
        violations.append(
            Violation(
                point_index=-1,
                value=measurement_mean,
                threshold=payload.spec.nominal,
                reason="mean_below_nominal",
            )
        )

    in_threshold_count = sum(1 for value in payload.measurements if value >= threshold)
    coverage_pct = (in_threshold_count / len(payload.measurements)) * 100.0

    result = {
        "pass_": len(violations) == 0,
        "violations": [item.model_dump() for item in violations],
        "mean": measurement_mean,
        "coverage_pct": coverage_pct,
    }
    return success_response(result=result)
