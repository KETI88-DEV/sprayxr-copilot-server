from __future__ import annotations

import pytest

from trainer.demo_api.sim_session import SimSessionStore

FLOW = ["A-1", "A-2", "A-3"]


def _started() -> SimSessionStore:
    store = SimSessionStore()
    _ = store.start(session_id="s-1", level_id="SP0035-L01", recommended_flow=FLOW)
    return store


def test_start_begins_with_nothing_painted() -> None:
    store = _started()
    state = store.current()
    assert state is not None
    assert state.phase == "start"
    assert state.recommended_flow == ("A-1", "A-2", "A-3")
    assert state.current_zone_id == "A-1"
    assert state.completed_zone_ids == ()
    assert state.anomaly_count == 0
    assert state.unresolved_zone_id is None
    assert state.last_event_id is None


def test_runtime_zone_in_flow_moves_position_and_completes_earlier_zones() -> None:
    store = _started()
    state = store.runtime(session_id="s-1", event_id="ev-0", target_zone_id="A-3", anomaly=False)
    assert state is not None
    assert state.phase == "runtime"
    assert state.current_zone_id == "A-3"
    assert state.completed_zone_ids == ("A-1", "A-2")
    assert state.unresolved_zone_id is None
    assert state.last_event_id == "ev-0"


def test_runtime_zone_outside_flow_records_unresolved_without_moving() -> None:
    store = _started()
    state = store.runtime(session_id="s-1", event_id="ev-0", target_zone_id="B-3", anomaly=False)
    assert state is not None
    assert state.current_zone_id == "A-1"
    assert state.completed_zone_ids == ()
    assert state.unresolved_zone_id == "B-3"


def test_runtime_without_target_keeps_position() -> None:
    store = _started()
    state = store.runtime(session_id="s-1", event_id="ev-0", target_zone_id=None, anomaly=True)
    assert state is not None
    assert state.current_zone_id == "A-1"
    assert state.anomaly_count == 1


def test_runtime_anomaly_count_accumulates() -> None:
    store = _started()
    _ = store.runtime(session_id="s-1", event_id="ev-0", target_zone_id=None, anomaly=True)
    state = store.runtime(session_id="s-1", event_id="ev-1", target_zone_id=None, anomaly=True)
    assert state is not None
    assert state.anomaly_count == 2


def test_runtime_for_unknown_session_is_ignored() -> None:
    store = _started()
    assert (
        store.runtime(session_id="other", event_id="ev-0", target_zone_id="A-2", anomaly=False)
        is None
    )
    current = store.current()
    assert current is not None
    assert current.current_zone_id == "A-1"


def test_end_marks_phase_without_completing_remaining_zones() -> None:
    store = _started()
    _ = store.runtime(session_id="s-1", event_id="ev-0", target_zone_id="A-2", anomaly=False)
    state = store.end(session_id="s-1")
    assert state is not None
    assert state.phase == "session_end"
    assert state.current_zone_id == "A-2"
    assert state.completed_zone_ids == ("A-1",)


def test_end_for_unknown_session_is_ignored() -> None:
    store = _started()
    assert store.end(session_id="other") is None
    current = store.current()
    assert current is not None
    assert current.phase == "start"


def test_new_start_replaces_previous_session() -> None:
    store = _started()
    _ = store.runtime(session_id="s-1", event_id="ev-0", target_zone_id="A-3", anomaly=False)
    _ = store.start(session_id="s-2", level_id="SP0035-L01", recommended_flow=FLOW)
    state = store.current()
    assert state is not None
    assert state.session_id == "s-2"
    assert state.current_zone_id == "A-1"
    assert state.completed_zone_ids == ()
    assert state.anomaly_count == 0


def test_current_is_none_before_any_start() -> None:
    assert SimSessionStore().current() is None


def test_start_rejects_empty_flow() -> None:
    """A zoneless session cannot produce a valid SequenceSnapshot, so refuse it at the source."""
    store = SimSessionStore()
    with pytest.raises(ValueError):
        _ = store.start(session_id="s-1", level_id="SP0035-L01", recommended_flow=[])
    assert store.current() is None
