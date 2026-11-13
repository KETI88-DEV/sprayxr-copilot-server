from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone

from trainer.demo_api.sim_contracts import Phase


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True, slots=True)
class SimSessionState:
    session_id: str
    level_id: str
    phase: Phase
    recommended_flow: tuple[str, ...]
    current_zone_id: str
    completed_zone_ids: tuple[str, ...]
    last_event_id: str | None
    anomaly_count: int
    unresolved_zone_id: str | None
    updated_at: str


class SimSessionStore:
    """Single-slot in-memory owner of the current simulator session state.

    One slot on purpose: the chat ``environment`` must be unambiguous about which
    session it describes, and the demo has one operator. A new ``start`` replaces
    the previous session; ``session_end`` keeps the final state readable so the
    panel does not blank out.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: SimSessionState | None = None

    def start(
        self, *, session_id: str, level_id: str, recommended_flow: Sequence[str]
    ) -> SimSessionState:
        flow = tuple(recommended_flow)
        if not flow:
            raise ValueError("recommended_flow must not be empty")
        state = SimSessionState(
            session_id=session_id,
            level_id=level_id,
            phase="start",
            recommended_flow=flow,
            current_zone_id=flow[0],
            completed_zone_ids=(),
            last_event_id=None,
            anomaly_count=0,
            unresolved_zone_id=None,
            updated_at=_now(),
        )
        with self._lock:
            self._state = state
        return state

    def runtime(
        self, *, session_id: str, event_id: str, target_zone_id: str | None, anomaly: bool
    ) -> SimSessionState | None:
        with self._lock:
            state = self._state
            if state is None or state.session_id != session_id:
                return None

            current_zone_id = state.current_zone_id
            completed = state.completed_zone_ids
            unresolved = state.unresolved_zone_id

            if target_zone_id is not None:
                if target_zone_id in state.recommended_flow:
                    index = state.recommended_flow.index(target_zone_id)
                    current_zone_id = target_zone_id
                    completed = state.recommended_flow[:index]
                    unresolved = None
                else:
                    unresolved = target_zone_id

            updated = replace(
                state,
                phase="runtime",
                current_zone_id=current_zone_id,
                completed_zone_ids=completed,
                last_event_id=event_id,
                anomaly_count=state.anomaly_count + (1 if anomaly else 0),
                unresolved_zone_id=unresolved,
                updated_at=_now(),
            )
            self._state = updated
            return updated

    def end(self, *, session_id: str) -> SimSessionState | None:
        with self._lock:
            state = self._state
            if state is None or state.session_id != session_id:
                return None
            updated = replace(state, phase="session_end", updated_at=_now())
            self._state = updated
            return updated

    def current(self) -> SimSessionState | None:
        with self._lock:
            return self._state
