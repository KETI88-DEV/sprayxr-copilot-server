from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from mcp_servers.common import MCPException, MCPResponse, success_response


router = APIRouter(prefix="/vlm", tags=["vlm"])


class VLMAnalyzeRequest(BaseModel):
    heatmap_path: str
    screenshot_path: str | None = None
    report_json: dict[str, object] | None = None


@router.post("/analyze", response_model=MCPResponse)
def analyze_vlm(payload: VLMAnalyzeRequest) -> MCPResponse:
    if not payload.heatmap_path.strip():
        raise MCPException(
            code="INVALID_PARAMS",
            message="heatmap_path must not be empty",
            status_code=400,
        )

    result: dict[str, object] = {
        "regions_of_interest": [],
        "detected_issues": [],
        "paint_distribution": {},
        "estimated_thickness": [],
    }
    return success_response(result=result)
