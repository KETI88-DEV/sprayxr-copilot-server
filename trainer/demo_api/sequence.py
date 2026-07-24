from __future__ import annotations

import re

from trainer.demo_api.contracts import (
    SequenceSnapshot as _SequenceSnapshot,
    SequenceUpdate as _SequenceUpdate,
    SequenceZone as _SequenceZone,
)

__all__ = ["default_snapshot", "sequence_update_from_payload"]

_ZONE_ID_PATTERN = re.compile(r"^[A-Z]-\d+$")


def default_snapshot(level_id: str) -> _SequenceSnapshot:
    return _SequenceSnapshot(
        level_id=level_id,
        label="SP0035 Level 01",
        zones=[
            _SequenceZone(zone_id="A-1", label="상부 작업면", status="completed"),
            _SequenceZone(zone_id="A-2", label="중간부 작업면", status="next"),
            _SequenceZone(zone_id="A-3", label="하부 곡면", status="pending"),
        ],
        current_zone_id="A-1",
        next_zone_ids=["A-2", "A-3"],
        completed_zone_ids=["A-1"],
    )


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
