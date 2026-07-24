from trainer.demo_api.sequence import default_snapshot, sequence_update_from_payload


def test_default_snapshot_has_stable_structure() -> None:
    snapshot = default_snapshot("SP0035-L01")
    assert snapshot.level_id == "SP0035-L01"
    ids = [zone.zone_id for zone in snapshot.zones]
    assert ids == ["A-1", "A-2", "A-3"]
    assert snapshot.current_zone_id == "A-1"


def test_next_path_becomes_sequence_update() -> None:
    update = sequence_update_from_payload(
        "SP0035-L01",
        {"current_zone": "A-1", "next_zone": "A-2"},
        [{"step": 1, "action": "A-1 완료"}, {"step": 2, "action": "A-2 이동"}],
    )
    assert update.current_zone_id == "A-1"
    assert update.next_zone_ids == ["A-2"]
    assert update.completed_zone_ids == ["A-1"]


def test_missing_next_zone_returns_none() -> None:
    result = sequence_update_from_payload("SP0035-L01", None, [])
    assert result is None
