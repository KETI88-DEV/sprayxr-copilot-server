from __future__ import annotations

import re
from typing import TYPE_CHECKING, Final, TypedDict

from trainer.demo_api.contracts import (
    SequenceSnapshot as _SequenceSnapshot,
    SequenceUpdate as _SequenceUpdate,
    SequenceZone as _SequenceZone,
)

if TYPE_CHECKING:
    from trainer.demo_api.sim_session import SimSessionState

__all__ = [
    "default_snapshot",
    "default_tool_environment",
    "sequence_update_from_payload",
    "session_snapshot",
    "session_tool_environment",
]

_ZONE_ID_PATTERN = re.compile(r"^[A-Z]-\d+$")


class SequenceToolZone(TypedDict):
    zone_id: str
    block_id: str
    zone_type: str
    position: str
    is_painted: bool


class SequenceToolEnvironment(TypedDict):
    zone_map: list[SequenceToolZone]
    current_zone_id: str
    structure_type: str


_TOOL_ZONE_METADATA: Final[dict[str, tuple[str, str, str]]] = {
    "A-1": ("BLK-A", "수직 패널", "upper"),
    "A-2": ("BLK-A", "수직 패널", "middle"),
    "A-3": ("BLK-A", "곡면", "lower"),
}
_UNKNOWN_ZONE_METADATA: Final[tuple[str, str, str]] = ("BLK-A", "미상", "unknown")


def default_snapshot(level_id: str) -> _SequenceSnapshot:
    return _SequenceSnapshot(
        level_id=level_id,
        label="Training Level 01",
        zones=[
            _SequenceZone(zone_id="A-1", label="상부 작업면", status="completed"),
            _SequenceZone(zone_id="A-2", label="중간부 작업면", status="next"),
            _SequenceZone(zone_id="A-3", label="하부 곡면", status="pending"),
        ],
        current_zone_id="A-1",
        next_zone_ids=["A-2", "A-3"],
        completed_zone_ids=["A-1"],
    )


def default_tool_environment(level_id: str) -> SequenceToolEnvironment:
    snapshot = default_snapshot(level_id)
    zone_map: list[SequenceToolZone] = []
    for zone in snapshot.zones:
        block_id, zone_type, position = _TOOL_ZONE_METADATA[zone.zone_id]
        zone_map.append(
            {
                "zone_id": zone.zone_id,
                "block_id": block_id,
                "zone_type": zone_type,
                "position": position,
                "is_painted": zone.status == "completed",
            }
        )

    return {
        "zone_map": zone_map,
        "current_zone_id": snapshot.current_zone_id,
        "structure_type": "block",
    }


def session_snapshot(state: SimSessionState) -> _SequenceSnapshot:
    base = default_snapshot(state.level_id)
    labels = {zone.zone_id: zone.label for zone in base.zones}

    zones: list[_SequenceZone] = []
    for zone_id in state.recommended_flow:
        if zone_id in state.completed_zone_ids:
            status = "completed"
        elif zone_id == state.current_zone_id:
            status = "current"
        else:
            status = "pending"
        zones.append(
            _SequenceZone(zone_id=zone_id, label=labels.get(zone_id, zone_id), status=status)
        )

    if state.current_zone_id in state.recommended_flow:
        index = state.recommended_flow.index(state.current_zone_id)
        next_zone_ids = list(state.recommended_flow[index + 1 :])
    else:
        next_zone_ids = []

    return _SequenceSnapshot(
        level_id=state.level_id,
        label=base.label,
        zones=zones,
        current_zone_id=state.current_zone_id,
        next_zone_ids=next_zone_ids,
        completed_zone_ids=list(state.completed_zone_ids),
    )


def session_tool_environment(state: SimSessionState) -> SequenceToolEnvironment:
    zone_map: list[SequenceToolZone] = []
    for zone_id in state.recommended_flow:
        block_id, zone_type, position = _TOOL_ZONE_METADATA.get(zone_id, _UNKNOWN_ZONE_METADATA)
        zone_map.append(
            {
                "zone_id": zone_id,
                "block_id": block_id,
                "zone_type": zone_type,
                "position": position,
                "is_painted": zone_id in state.completed_zone_ids,
            }
        )

    return {
        "zone_map": zone_map,
        "current_zone_id": state.current_zone_id,
        "structure_type": "block",
    }


def sequence_update_from_payload(
    level_id: str,
    next_path: dict | None,
    steps: list[dict] | None,
) -> _SequenceUpdate | None:
    if next_path is None:
        return None

    current_zone = next_path.get("current_zone")
    if not current_zone:
        return None

    current_zone_id = current_zone
    next_zone_ids: list[str] = []

    next_zone = next_path.get("next_zone")
    if isinstance(next_zone, str) and next_zone:
        next_zone_ids.append(next_zone)

    for step in steps or []:
        action = step.get("action")
        if not isinstance(action, str) or not action:
            continue
        token = action.split(maxsplit=1)[0]
        if not _ZONE_ID_PATTERN.fullmatch(token):
            continue
        if token == current_zone_id or token in next_zone_ids:
            continue
        next_zone_ids.append(token)

    return _SequenceUpdate(
        level_id=level_id,
        current_zone_id=current_zone_id,
        next_zone_ids=next_zone_ids,
        completed_zone_ids=[current_zone_id],
    )
