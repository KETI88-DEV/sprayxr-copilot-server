from __future__ import annotations

from collections import defaultdict
from typing import Literal, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from mcp_servers.common import MCPResponse, success_response


router = APIRouter(prefix="/sequence_validation", tags=["sequence_validation"])


# ---------- 구역 위치별 우선순위 (중력/오염 방지 기반) ----------
_POSITION_PRIORITY: dict[str, int] = {
    "upper": 1,
    "center": 2,
    "middle": 2,
    "left": 3,
    "right": 3,
    "lower": 4,
}

# 구역 유형별 보정: 곡면/모서리는 후순위 (건조 간섭 최소화)
_TYPE_PENALTY: dict[str, int] = {
    "수직 패널": 0,
    "수평 패널": 0,
    "곡면": 1,
    "모서리": 2,
}


class ZoneInfo(BaseModel):
    zone_id: str
    block_id: str
    zone_type: Literal["수직 패널", "수평 패널", "곡면", "모서리"]
    position: Literal["upper", "middle", "lower", "left", "right", "center"]
    is_painted: bool = False
    area_m2: float = 0.0


class SequenceValidationRequest(BaseModel):
    zone_map: list[ZoneInfo] = Field(min_length=1)
    current_zone_id: Optional[str] = None
    structure_type: Literal["block", "hull_section", "deck", "superstructure"] = "block"


def _zone_sort_key(zone: ZoneInfo) -> tuple[int, int, str]:
    """구역 우선순위 키: (위치 순위 + 유형 보정, 면적 역순 보정, zone_id)."""
    pos_priority = _POSITION_PRIORITY.get(zone.position, 5)
    type_penalty = _TYPE_PENALTY.get(zone.zone_type, 0)
    return (pos_priority + type_penalty, -int(zone.area_m2 * 100), zone.zone_id)


def _block_sort_key(block_id: str, zones: list[ZoneInfo]) -> tuple[int, str]:
    """블록 우선순위 키: (미도장 구역 중 최상위 위치 점수, block_id)."""
    unpainted = [z for z in zones if not z.is_painted]
    if not unpainted:
        return (99, block_id)
    best_score = min(_POSITION_PRIORITY.get(z.position, 5) for z in unpainted)
    return (best_score, block_id)


def _group_by_block(zones: list[ZoneInfo]) -> dict[str, list[ZoneInfo]]:
    groups: dict[str, list[ZoneInfo]] = defaultdict(list)
    for z in zones:
        groups[z.block_id].append(z)
    return dict(groups)


def _get_zone_reason(zone: ZoneInfo) -> str:
    """다음 구역 선택 이유 생성."""
    pos_label = {
        "upper": "상부", "center": "중앙부", "middle": "중간부",
        "lower": "하부", "left": "좌측", "right": "우측",
    }.get(zone.position, zone.position)
    reasons = [f"{pos_label} 위치 (중력 방향 오염 방지)"]
    if zone.zone_type in ("수직 패널", "수평 패널"):
        reasons.append("평면 구역 우선")
    return ", ".join(reasons)


@router.post("/validate", response_model=MCPResponse)
def validate_sequence(payload: SequenceValidationRequest) -> MCPResponse:
    zone_dict = {z.zone_id: z for z in payload.zone_map}
    block_groups = _group_by_block(payload.zone_map)

    all_unpainted = [z for z in payload.zone_map if not z.is_painted]
    if payload.current_zone_id:
        all_unpainted = [z for z in all_unpainted if z.zone_id != payload.current_zone_id]

    if not all_unpainted:
        return success_response(result={
            "structure_type": payload.structure_type,
            "current_zone_id": payload.current_zone_id,
            "all_painted": True,
            "block_sequence": [],
            "next_block": None,
            "next_zone": None,
            "current_block_remaining": [],
            "message": "모든 구역의 도장이 완료되었습니다.",
        })

    current_block_id = None
    if payload.current_zone_id and payload.current_zone_id in zone_dict:
        current_block_id = zone_dict[payload.current_zone_id].block_id

    block_order = sorted(
        block_groups.keys(),
        key=lambda bid: _block_sort_key(bid, block_groups[bid]),
    )
    blocks_with_remaining = [
        bid for bid in block_order
        if any(not z.is_painted and z.zone_id != payload.current_zone_id
               for z in block_groups[bid])
    ]

    current_block_zones: list[str] = []
    next_zone_id: str | None = None
    next_block_id: str | None = None

    if current_block_id:
        remaining_in_block = [
            z for z in block_groups.get(current_block_id, [])
            if not z.is_painted and z.zone_id != payload.current_zone_id
        ]
        remaining_in_block.sort(key=_zone_sort_key)
        current_block_zones = [z.zone_id for z in remaining_in_block]

        if current_block_zones:
            next_zone_id = current_block_zones[0]
            next_block_id = current_block_id
        elif blocks_with_remaining:
            next_block_id = blocks_with_remaining[0]
            next_block_zones = [
                z for z in block_groups[next_block_id] if not z.is_painted
            ]
            next_block_zones.sort(key=_zone_sort_key)
            next_zone_id = next_block_zones[0].zone_id if next_block_zones else None
    elif blocks_with_remaining:
        next_block_id = blocks_with_remaining[0]
        next_block_zones = [z for z in block_groups[next_block_id] if not z.is_painted]
        next_block_zones.sort(key=_zone_sort_key)
        next_zone_id = next_block_zones[0].zone_id if next_block_zones else None

    next_zone = zone_dict.get(next_zone_id) if next_zone_id else None
    zone_reason = _get_zone_reason(next_zone) if next_zone else ""

    total = len(payload.zone_map)
    painted = sum(1 for z in payload.zone_map if z.is_painted)
    if payload.current_zone_id:
        painted += 1

    same_block = current_block_id and next_block_id == current_block_id
    if same_block:
        msg = (
            f"현재 블록({current_block_id}) 내 다음 구역: {next_zone_id} ({zone_reason}). "
            f"블록 내 잔여 {len(current_block_zones)}개."
        )
    elif next_block_id:
        msg = (
            f"현재 블록 완료. 다음 블록: {next_block_id}, "
            f"시작 구역: {next_zone_id} ({zone_reason}). "
            f"잔여 블록 {len(blocks_with_remaining)}개."
        )
    else:
        msg = "모든 구역의 도장이 완료되었습니다."

    result = {
        "structure_type": payload.structure_type,
        "current_zone_id": payload.current_zone_id,
        "current_block_id": current_block_id,
        "all_painted": False,
        "block_sequence": blocks_with_remaining,
        "next_block": next_block_id,
        "next_zone": next_zone_id,
        "zone_reason": zone_reason,
        "current_block_remaining": current_block_zones,
        "progress": f"{painted}/{total} 구역 완료",
        "message": msg,
    }
    return success_response(result=result)
