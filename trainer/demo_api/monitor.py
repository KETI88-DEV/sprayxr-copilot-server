from __future__ import annotations

import threading
from collections import deque
from datetime import datetime, timezone
from typing import Final

import anyio
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream

from trainer.demo_api.sim_contracts import Direction, MonitorEvent, MonitorSnapshot, Phase

_MAX_EVENTS: Final[int] = 500
_SUBSCRIBER_BUFFER: Final[int] = 256


class MonitorRecorder:
    """Bounded in-memory recorder for simulator interface I/O with live fan-out.

    ``record`` is called from async request handlers, so pushing to subscriber
    streams stays on the event loop and never blocks. The lock only guards the
    buffer and the subscriber list, which the recorder also mutates from the
    module-level app singleton shared across requests.
    """

    def __init__(self, maxlen: int = _MAX_EVENTS) -> None:
        self._events: deque[MonitorEvent] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._seq = 0
        self._subscribers: list[MemoryObjectSendStream[MonitorEvent]] = []

    def record(
        self, session_id: str, phase: Phase, direction: Direction, payload: dict[str, object]
    ) -> MonitorEvent:
        with self._lock:
            self._seq += 1
            event = MonitorEvent(
                seq=self._seq,
                session_id=session_id,
                phase=phase,
                direction=direction,
                timestamp=datetime.now(timezone.utc).isoformat(),
                payload=payload,
            )
            self._events.append(event)
            subscribers = list(self._subscribers)

        for stream in subscribers:
            try:
                stream.send_nowait(event)
            except (anyio.WouldBlock, anyio.BrokenResourceError, anyio.ClosedResourceError):
                # Slow or gone subscriber: drop this event for that client only.
                # The buffer stays authoritative, so a reconnect replays by cursor.
                pass

        return event

    def snapshot(self, cursor: int, limit: int) -> MonitorSnapshot:
        with self._lock:
            events = list(self._events)

        dropped_before: int | None = None
        if events and cursor < events[0].seq - 1:
            dropped_before = events[0].seq

        selected = [event for event in events if event.seq > cursor][:limit]
        next_cursor = selected[-1].seq if selected else cursor
        return MonitorSnapshot(
            events=selected, next_cursor=next_cursor, dropped_before=dropped_before
        )

    def subscribe(
        self,
    ) -> tuple[MemoryObjectSendStream[MonitorEvent], MemoryObjectReceiveStream[MonitorEvent]]:
        send, receive = anyio.create_memory_object_stream[MonitorEvent](
            max_buffer_size=_SUBSCRIBER_BUFFER
        )
        with self._lock:
            self._subscribers.append(send)
        return send, receive

    def unsubscribe(self, send: MemoryObjectSendStream[MonitorEvent]) -> None:
        with self._lock:
            if send in self._subscribers:
                self._subscribers.remove(send)
