from __future__ import annotations

from statistics import mean

from fastapi import APIRouter
from pydantic import BaseModel, Field

from mcp_servers.common import MCPException, MCPResponse, success_response


router = APIRouter(prefix="/anomaly_detection", tags=["anomaly_detection"])


class Environment(BaseModel):
    temperature: float
    humidity: float


class AnomalyRequest(BaseModel):
    image_context: str
    dft_measurements: list[float] = Field(min_length=1)
    env: Environment


@router.post("/detect", response_model=MCPResponse)
def detect_anomaly(payload: AnomalyRequest) -> MCPResponse:
    if not payload.image_context.strip():
        raise MCPException(
            code="INVALID_PARAMS",
            message="image_context must not be empty",
            status_code=400,
        )

    avg = mean(payload.dft_measurements)
    under_threshold = avg * 0.8
    over_threshold = avg * 1.2

    anomaly_type = "none"
    if any(value < under_threshold for value in payload.dft_measurements):
        anomaly_type = "under_thickness"
    elif any(value > over_threshold for value in payload.dft_measurements):
        anomaly_type = "over_thickness"

    result = {
        "anomaly_detected": anomaly_type != "none",
        "anomaly_type": anomaly_type,
        "mean": avg,
        "image_context": payload.image_context,
        "env": payload.env.model_dump(),
        "thresholds": {
            "under": under_threshold,
            "over": over_threshold,
        },
    }
    return success_response(result=result)
