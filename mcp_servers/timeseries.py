from __future__ import annotations

from collections import OrderedDict
from statistics import mean

from fastapi import APIRouter
from pydantic import BaseModel

from mcp_servers.common import MCPException, MCPResponse, success_response


router = APIRouter(prefix="/timeseries", tags=["timeseries"])
# 세션을 지우는 경로가 없어 한도를 둔다. 가장 오래 갱신이 없던 세션, 가장 오래된 점부터 버린다.
_MAX_SESSIONS = 64
_MAX_POINTS = 2000
_sessions: OrderedDict[str, list["DataPoint"]] = OrderedDict()


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
    _sessions.move_to_end(payload.session_id)
    series.append(payload.data)
    del series[:-_MAX_POINTS]
    while len(_sessions) > _MAX_SESSIONS:
        _sessions.popitem(last=False)
    result = {
        "session_id": payload.session_id,
        "count": len(series),
        "trend": _compute_trend(series),
    }
    return success_response(result=result)


class TimeSeriesQueryRequest(BaseModel):
    session_id: str


def _query_session(session_id: str) -> MCPResponse:
    series = _sessions.get(session_id)
    if series is None:
        result = {
            "session_id": session_id,
            "count": 0,
            "trend": "no_data",
            "latest": None,
            "means": None,
            "message": "세션 데이터가 없습니다. update를 먼저 호출하세요.",
        }
        return success_response(result=result)

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


@router.get("/query", response_model=MCPResponse)
def query_timeseries_get(session_id: str) -> MCPResponse:
    return _query_session(session_id)


@router.post("/query", response_model=MCPResponse)
def query_timeseries_post(payload: TimeSeriesQueryRequest) -> MCPResponse:
    return _query_session(payload.session_id)
