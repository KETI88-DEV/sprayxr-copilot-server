# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false
from pydantic import ValidationError
import pytest

from trainer.demo_api.contracts import SequenceSnapshot, SequenceUpdate, SequenceZone


def test_sequence_update_rejects_duplicate_next_zones() -> None:
    with pytest.raises(ValidationError):
        SequenceUpdate(
            type="update_sequence",
            level_id="SP0035-L01",
            current_zone_id="A-1",
            next_zone_ids=["A-2", "A-2"],
            completed_zone_ids=["A-1"],
        )


def test_snapshot_requires_current_zone_to_exist() -> None:
    with pytest.raises(ValidationError):
        SequenceSnapshot(
            level_id="SP0035-L01",
            label="SP0035 Level 01",
            zones=[SequenceZone(zone_id="A-1", label="상부", status="pending")],
            current_zone_id="A-2",
            next_zone_ids=[],
            completed_zone_ids=[],
        )


def test_zone_status_enumerates_allowed_values() -> None:
    zone = SequenceZone(zone_id="A-1", label="상부", status="completed")
    assert zone.status == "completed"
