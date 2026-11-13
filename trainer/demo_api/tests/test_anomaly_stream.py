"""이상 알림 방송기의 세션 수 한도."""

from __future__ import annotations

from trainer.demo_api import anomaly_stream
from trainer.demo_api.anomaly_stream import AnomalyBroadcaster, AnomalyEvent


def _publish(bus: AnomalyBroadcaster, session_id: str, sequence_id: str = "SEQ-001") -> AnomalyEvent | None:
    return bus.publish(session_id=session_id, sequence_id=sequence_id, anomaly_type="distance_far", payload={})


def test_oldest_session_is_dropped_past_the_limit() -> None:
    bus = AnomalyBroadcaster()
    for index in range(anomaly_stream._MAX_SESSIONS + 1):
        _publish(bus, f"s{index}")
    assert bus.since("s0", last_event_id=0) == []
    assert len(bus.since(f"s{anomaly_stream._MAX_SESSIONS}", last_event_id=0)) == 1


def test_recently_published_session_survives() -> None:
    bus = AnomalyBroadcaster()
    for index in range(anomaly_stream._MAX_SESSIONS):
        _publish(bus, f"s{index}")
    _publish(bus, "s0", sequence_id="SEQ-002")
    _publish(bus, "new")
    assert len(bus.since("s0", last_event_id=0)) == 2
    assert bus.since("s1", last_event_id=0) == []
    # 살아남은 세션은 발행 이력도 그대로라 같은 알림을 다시 내지 않는다.
    assert _publish(bus, "s0", sequence_id="SEQ-002") is None
