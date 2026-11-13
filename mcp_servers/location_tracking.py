from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from mcp_servers.common import MCPResponse, success_response


router = APIRouter(prefix="/location_tracking", tags=["location_tracking"])


class Coordinate(BaseModel):
    x: float
    y: float
    z: float
    timestamp: float | None = None


class LocationTrackingRequest(BaseModel):
    coordinates: list[Coordinate] = Field(min_length=1)
    ship_model_id: str | None = None


@router.post("/identify", response_model=MCPResponse)
def identify_location(payload: LocationTrackingRequest) -> MCPResponse:
    result = {
        "zone_id": "B-3",
        "zone_type": "수직 패널",
        "coverage_pct": 0.62,
        "uncovered_zones": ["B-4", "C-1"],
        "points_processed": len(payload.coordinates),
        "ship_model_id": payload.ship_model_id,
    }
    return success_response(result=result)
