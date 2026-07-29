from trainer.demo_api.sequence import default_snapshot, sequence_update_from_payload


def test_default_snapshot_has_stable_structure() -> None:
    snapshot = default_snapshot("TR-L01")
    assert snapshot.level_id == "TR-L01"
    ids = [zone.zone_id for zone in snapshot.zones]
    assert ids == ["A-1", "A-2", "A-3"]
    assert snapshot.current_zone_id == "A-1"


def test_next_path_becomes_sequence_update() -> None:
    update = sequence_update_from_payload(
        "TR-L01",
        {"current_zone": "A-1", "next_zone": "A-2"},
        [{"step": 1, "action": "A-1 완료"}, {"step": 2, "action": "A-2 이동"}],
    )
    assert update.current_zone_id == "A-1"
    assert update.next_zone_ids == ["A-2"]
    assert update.completed_zone_ids == ["A-1"]


def test_missing_next_zone_returns_none() -> None:
    result = sequence_update_from_payload("TR-L01", None, [])
    assert result is None


from trainer.demo_api.sequence import session_snapshot, session_tool_environment
from trainer.demo_api.sim_session import SimSessionState, SimSessionStore


def _state_at(zone_id: str) -> SimSessionState:
    store = SimSessionStore()
    _ = store.start(
        session_id="s-1", level_id="TR-L01", recommended_flow=["A-1", "A-2", "A-3"]
    )
    _ = store.runtime(session_id="s-1", event_id="ev-0", target_zone_id=zone_id, anomaly=False)
    state = store.current()
    assert state is not None
    return state


def test_session_snapshot_marks_current_zone_and_completed_prefix() -> None:
    snapshot = session_snapshot(_state_at("A-2"))
    statuses = {zone.zone_id: zone.status for zone in snapshot.zones}
    assert statuses == {"A-1": "completed", "A-2": "current", "A-3": "pending"}
    assert snapshot.current_zone_id == "A-2"
    assert snapshot.next_zone_ids == ["A-3"]
    assert snapshot.completed_zone_ids == ["A-1"]
    assert snapshot.label == "Training Level 01"
    assert snapshot.level_id == "TR-L01"


def test_session_snapshot_at_first_zone_completes_nothing() -> None:
    snapshot = session_snapshot(_state_at("A-1"))
    statuses = {zone.zone_id: zone.status for zone in snapshot.zones}
    assert statuses == {"A-1": "current", "A-2": "pending", "A-3": "pending"}
    assert snapshot.completed_zone_ids == []
    assert snapshot.next_zone_ids == ["A-2", "A-3"]


def test_session_snapshot_keeps_zone_labels_from_default_fixture() -> None:
    snapshot = session_snapshot(_state_at("A-3"))
    labels = {zone.zone_id: zone.label for zone in snapshot.zones}
    assert labels == {"A-1": "상부 작업면", "A-2": "중간부 작업면", "A-3": "하부 곡면"}


def test_session_tool_environment_marks_painted_zones() -> None:
    environment = session_tool_environment(_state_at("A-3"))
    painted = {zone["zone_id"]: zone["is_painted"] for zone in environment["zone_map"]}
    assert painted == {"A-1": True, "A-2": True, "A-3": False}
    assert environment["current_zone_id"] == "A-3"
    assert environment["structure_type"] == "block"


def test_session_tool_environment_keeps_zone_metadata() -> None:
    environment = session_tool_environment(_state_at("A-2"))
    by_id = {zone["zone_id"]: zone for zone in environment["zone_map"]}
    assert by_id["A-3"]["block_id"] == "BLK-A"
    assert by_id["A-3"]["zone_type"] == "곡면"
    assert by_id["A-3"]["position"] == "lower"


def test_session_derivations_tolerate_zone_outside_default_metadata() -> None:
    store = SimSessionStore()
    _ = store.start(session_id="s-1", level_id="TR-L01", recommended_flow=["A-1", "Z-9"])
    state = store.current()
    assert state is not None

    snapshot = session_snapshot(state)
    assert [zone.label for zone in snapshot.zones] == ["상부 작업면", "Z-9"]

    environment = session_tool_environment(state)
    by_id = {zone["zone_id"]: zone for zone in environment["zone_map"]}
    assert by_id["Z-9"]["block_id"] == "BLK-A"
