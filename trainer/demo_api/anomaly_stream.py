"""이상 알림 SSE 방송기 (명세서 p.10, §4).

`GET /api/v1/stream/anomalies/{session_id}`는 세션 시작과 함께 상시 연결을 유지하고
(`text/event-stream`), 이상이 있을 때만 이벤트를 보낸다. 명세서·계약이 요구하는 성질:

* 이상이 있을 때만 보낸다 — 정상에는 아무것도 보내지 않는다
* 연결 유지용 comment heartbeat 를 최대 15초 간격으로 보낸다
* 모든 이벤트에 증가하는 `id`를 담아 `Last-Event-ID` 재연결을 받는다
* 같은 `sequence_id`의 동일 이상을 중복 발행하지 않는다

마지막 규칙 때문에 발행 이력을 세션별로 들고 있어야 한다. `MonitorRecorder`와 달리
구독자가 세션으로 갈리므로 큐를 세션별로 둔다.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Final

logger = logging.getLogger(__name__)

HEARTBEAT_SEC: Final[float] = 15.0
# 새 이벤트를 확인하는 주기. heartbeat 와 분리한다 — 묶으면 이상이 최대
# HEARTBEAT_SEC 만큼 밀려 명세서 p.4의 "실시간 Push"가 되지 않는다.
POLL_SEC: Final[float] = 0.25
_BUFFER_PER_SESSION: Final[int] = 200


@dataclass(frozen=True)
class AnomalyEvent:
    id: int
    session_id: str
    sequence_id: str
    anomaly_type: str
    payload: dict[str, Any]


@dataclass
class _SessionState:
    events: deque[AnomalyEvent] = field(default_factory=lambda: deque(maxlen=_BUFFER_PER_SESSION))
    # (sequence_id, anomaly_type) — 같은 시퀀스의 동일 이상을 두 번 발행하지 않는다.
    published: set[tuple[str, str]] = field(default_factory=set)


class AnomalyBroadcaster:
    """세션별 이상 이벤트 버퍼. 구독자는 `id` 커서로 읽는다."""

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._sessions: dict[str, _SessionState] = {}
        self._seq = 0

    def publish(
        self, *, session_id: str, sequence_id: str, anomaly_type: str, payload: dict[str, Any]
    ) -> AnomalyEvent | None:
        """``None``이면 같은 시퀀스의 같은 이상이 이미 발행됐다는 뜻이다."""
        with self._guard:
            state = self._sessions.setdefault(session_id, _SessionState())
            key = (sequence_id, anomaly_type)
            if key in state.published:
                logger.info(
                    "anomaly already published session=%s sequence=%s type=%s",
                    session_id, sequence_id, anomaly_type,
                )
                return None
            self._seq += 1
            event = AnomalyEvent(self._seq, session_id, sequence_id, anomaly_type, payload)
            state.published.add(key)
            state.events.append(event)
            return event

    def since(self, session_id: str, *, last_event_id: int) -> list[AnomalyEvent]:
        with self._guard:
            state = self._sessions.get(session_id)
            if state is None:
                return []
            return [event for event in state.events if event.id > last_event_id]

    def reset(self, session_id: str) -> None:
        """세션 시작 시 이전 발행 이력을 지운다 — 같은 세션 id 재사용을 막지 않는다."""
        with self._guard:
            self._sessions.pop(session_id, None)
