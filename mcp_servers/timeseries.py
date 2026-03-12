from __future__ import annotations

from statistics import mean

from fastapi import APIRouter
from pydantic import BaseModel

from mcp_servers.common import MCPException, MCPResponse, success_response


router = APIRouter(prefix="/timeseries", tags=["timeseries"])
_sessions: dict[str, list["DataPoint"]] = {}


class DataPoint(BaseModel):
    timestamp: float
    speed: float
    distance: float
    angle: float


class TimeSeriesUpdateRequest(BaseModel):
    session_id: str
    data: DataPoint


def _compute_trend(points: list[DataPoint]) -> str:
    if len(points) < 2:
        return "stable"

    window = min(3, len(points))
    first_values = [point.speed for point in points[:window]]
    last_values = [point.speed for point in points[-window:]]
    start_mean = mean(first_values)
    end_mean = mean(last_values)

    if end_mean > start_mean:
        return "increasing"
    if end_mean < start_mean:
        return "decreasing"
    return "stable"


def reset_sessions() -> None:
    _sessions.clear()


@router.post("/update", response_model=MCPResponse)
def update_timeseries(payload: TimeSeriesUpdateRequest) -> MCPResponse:
    series = _sessions.setdefault(payload.session_id, [])
    series.append(payload.data)
    result = {
        "session_id": payload.session_id,
        "count": len(series),
        "trend": _compute_trend(series),
    }
    return success_response(result=result)


@router.get("/query", response_model=MCPResponse)
def query_timeseries(session_id: str) -> MCPResponse:
    series = _sessions.get(session_id)
    if series is None:
        raise MCPException(
            code="SESSION_NOT_FOUND",
            message="session_id not found",
            details={"session_id": session_id},
            status_code=404,
        )

    speeds = [point.speed for point in series]
    distances = [point.distance for point in series]
    angles = [point.angle for point in series]
    result = {
        "session_id": session_id,
        "count": len(series),
        "trend": _compute_trend(series),
        "latest": series[-1].model_dump(),
        "means": {
            "speed": mean(speeds),
            "distance": mean(distances),
            "angle": mean(angles),
        },
    }
    return success_response(result=result)
