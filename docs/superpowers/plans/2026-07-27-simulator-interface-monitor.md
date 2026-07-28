# 시뮬레이터 인터페이스 I/O 모니터 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 더미로 송수신을 검증하고 세션 시작/런타임/끝을 구분해 시뮬레이터 인터페이스 I/O를 실시간 관측하는 read-only 모니터 콘솔을 구현한다.

**Architecture:** GPU 없는 결정론 스텁 3단계 인터페이스(`/api/sim/*`)가 각 호출의 input·output을 bounded ring buffer에 기록하고, read-only 모니터(`/api/monitor/*`: 스냅샷 + SSE)로 노출한다. React 패널이 SSE를 구독해 세션·단계별 I/O 타임라인을 렌더하며, CLI/UI 더미 드라이버가 송수신을 발생시킨다.

**Tech Stack:** Python 3.11 · FastAPI · Pydantic v2 (strict) · anyio · httpx(CLI) / React 19 · TypeScript · Zod · Vitest.

**Spec:** `docs/superpowers/specs/2026-07-27-simulator-interface-monitor-design.md`

## Global Constraints

- GPU/LLM 미사용. 모든 출력은 결정론(고정 임계·템플릿). 임계 상수: `standoff_mm > 320.0` 또는 `angle_deg < 60.0` → anomaly.
- 링버퍼 `maxlen=500`. 호출 1건 → input 1건 + output 1건 기록.
- Pydantic 모델은 `extra="forbid"`(StrictModel). payload 필드는 기존 `contracts.py`의 `StreamEvent.data`와 동일하게 `dict[str, object]` 사용.
- 기존 `/api/health`·`/api/sequence`·`/api/chat`와 `AgentPipeline.run()`·`run_streaming()`은 불변. 신규는 `/api/sim/*`·`/api/monitor/*`로 격리.
- 신규 파이썬 의존성 도입 금지. CLI HTTP는 기존 bench와 동일 `httpx` 사용.
- `session_coaching.py`는 import만(수정 없음): `SessionSummary`, `SessionMetrics`, `compute_risk_flags`.
- 파일당 250 pure LOC 이하. Zod는 Pydantic과 동형 미러.
- 인터페이스 핸들러는 `async def`(이벤트 루프) — `recorder.record()`의 broadcast는 루프에서 실행.

### 실행 런타임 (검증됨)

호스트 `python3`(3.12)에는 fastapi·anyio·pytest가 **없다**. 모든 파이썬 명령은 실행 중인 트레이너 컨테이너에서 돌린다. 저장소 `/home/mjkim/ship-paint-ai`는 컨테이너의 `/workspace`에 마운트되어 있다.

```bash
# 파이썬 (컨테이너: python 3.11.9 / fastapi 0.135.1 / starlette 0.52.1 / pydantic 2.12.5 / anyio 4.12.1 / httpx 0.28.1 / pytest 9.1.1)
docker exec -w /workspace ship-paint-trainer python3 -m pytest <path> -q

# 프론트 (호스트: node 24 / npm 11, demo-ui/node_modules 설치됨)
cd demo-ui && npm test -- --run
```

기준선: `docker exec -w /workspace ship-paint-trainer python3 -m pytest trainer/demo_api/tests -q` → 29 passed.

**이하 모든 Task의 `python3 ...` Run 명령에는 위 `docker exec -w /workspace ship-paint-trainer` 접두사를 붙인다.** `basedpyright`/`ruff`는 이 런타임에 설치되어 있지 않으므로 타입 게이트는 `tsc -b`(프론트)와 pytest(백엔드)로 대체한다.

### TestClient 제약 (실측 확인)

Starlette `TestClient`는 `portal.call(self.app, scope, receive, send)`(`starlette/testclient.py:345`)로 **ASGI 호출이 끝난 뒤에** 응답을 만든다. 따라서 **끝나지 않는 SSE 엔드포인트는 TestClient로 테스트할 수 없다** — `client.stream(...)` 진입 자체가 영구 블로킹된다(60초 스파이크에서 출력 0줄, exit 124). 그래서 Task 5는 SSE 본문을 순수 async generator로 분리해 `anyio`로 직접 테스트하고, 실 HTTP SSE 경로는 Task 12에서 실제 uvicorn + curl로 검증한다.

### 동시 작업 중인 워킹트리

`demo-ui/src/App.tsx`, `demo-ui/tests/App.test.tsx`, `demo-ui/src/styles.css`, `trainer/demo_api/app.py`에는 이 작업과 무관한 **미커밋 변경**이 이미 있다. 커밋은 반드시 **본 작업이 생성한 경로만 명시적으로 스테이징**한다(`git add <경로>`, `git add -A` 금지). 공유 파일 수정은 델타를 최소화하고, 남의 hunk를 함께 커밋하지 않는다. 그래서 신규 스타일은 `styles.css`가 아니라 전용 CSS 파일에 넣고, App 스트리밍 로직은 전용 훅 파일로 뺀다.

---

## 파일 구조

**백엔드**
- Create `trainer/demo_api/sim_contracts.py` — 입력/출력/MonitorEvent Pydantic 계약.
- Create `trainer/demo_api/monitor.py` — `MonitorRecorder`(ring buffer + snapshot + anyio broadcast).
- Create `trainer/demo_api/simulator_stub.py` — 결정론 출력 빌더(default_snapshot/compute_risk_flags 재사용).
- Create `trainer/demo_api/simulator.py` — `/api/sim/*` 라우터 + `_run_phase` 헬퍼.
- Create `trainer/demo_api/monitor_api.py` — `monitor_sse_body` 순수 generator + `/api/monitor/*` 라우터.
- Modify `trainer/demo_api/app.py` — recorder 싱글턴 + 두 라우터 include (4줄).
- Create `trainer/scripts/sim_dummy_driver.py` — PEP 723 CLI 더미 드라이버.
- Test `trainer/demo_api/tests/test_sim_contracts.py`, `test_monitor_recorder.py`, `test_simulator_stub.py`, `test_simulator_interface.py`, `test_monitor_api.py`, `test_app_sim_wiring.py`; `trainer/scripts/tests/test_sim_dummy_driver.py`.

**프론트**
- Create `demo-ui/src/simContracts.ts` — Zod 미러.
- Create `demo-ui/src/simMonitorClient.ts` — 모니터 SSE 구독 + 더미 세션 송신.
- Create `demo-ui/src/useMonitorStream.ts` — App이 쓰는 구독 훅(공유 파일 델타 최소화용).
- Create `demo-ui/src/components/SimulatorMonitorPanel.tsx` + `demo-ui/src/simMonitor.css` — 타임라인 패널과 전용 스타일(`styles.css` 비수정).
- Modify `demo-ui/src/api.ts` — `API_BASE_URL`에 `export` 추가(1단어, DRY).
- Modify `demo-ui/src/App.tsx` — 훅 1줄 + 패널 1개(약 5줄).
- Modify `demo-ui/tests/App.test.tsx` — fetch mock에 `/api/monitor/stream` 분기 추가(기존 mock은 미등록 URL에 throw).
- Test `demo-ui/tests/simContracts.test.ts`, `simMonitorClient.test.ts`, `simulatorMonitorPanel.test.tsx`, `appSimPanel.test.tsx`.

---

## Task 1: 인터페이스·모니터 계약 (Pydantic)

**Files:**
- Create: `trainer/demo_api/sim_contracts.py`
- Test: `trainer/demo_api/tests/test_sim_contracts.py`

**Interfaces:**
- Produces: `Phase = Literal["start","runtime","session_end"]`, `Direction = Literal["input","output"]`; models `WorkerProfile`, `PaintSpec`, `SessionStartInput`, `GunMetrics`, `EventTelemetry`, `SessionEventInput`, `SessionStartOutput`, `SessionEventOutput`, `SessionReviewOutput`, `MonitorEvent`, `MonitorSnapshot`.

- [ ] **Step 1: Write the failing test**

```python
# trainer/demo_api/tests/test_sim_contracts.py
from __future__ import annotations

import pytest
from pydantic import ValidationError

from trainer.demo_api.sim_contracts import (
    EventTelemetry,
    GunMetrics,
    MonitorEvent,
    MonitorSnapshot,
    SessionEventInput,
    SessionEventOutput,
    SessionReviewOutput,
    SessionStartInput,
    SessionStartOutput,
)


def test_session_start_input_parses_nested_models() -> None:
    parsed = SessionStartInput.model_validate(
        {
            "session_id": "s1",
            "level_id": "SP0035-L01",
            "worker": {"trainee_id": "t1", "skill_level": "beginner", "handedness": "right"},
            "spec": {"nominal_um": 100.0},
        }
    )
    assert parsed.worker.skill_level == "beginner"
    assert parsed.spec.nominal_um == 100.0


def test_start_input_rejects_extra_field() -> None:
    with pytest.raises(ValidationError):
        SessionStartInput.model_validate(
            {
                "session_id": "s1",
                "level_id": "L",
                "worker": {"trainee_id": "t", "skill_level": "beginner", "handedness": "right"},
                "spec": {"nominal_um": 100.0},
                "unexpected": 1,
            }
        )


def test_event_input_allows_optional_telemetry_and_request() -> None:
    only_request = SessionEventInput.model_validate({"event_id": "e1", "user_request": "블라스팅?"})
    assert only_request.telemetry is None
    with_tele = SessionEventInput.model_validate(
        {"event_id": "e2", "telemetry": {"gun_metrics": {"speed_m_per_s": 0.8, "standoff_mm": 350.0, "angle_deg": 55.0}}}
    )
    assert isinstance(with_tele.telemetry, EventTelemetry)
    assert isinstance(with_tele.telemetry.gun_metrics, GunMetrics)


def test_output_and_monitor_models_round_trip() -> None:
    start_out = SessionStartOutput(session_id="s1", recommended_flow=["A-1"], flow_source="rule", guidance="g")
    assert start_out.flow_source == "rule"
    event_out = SessionEventOutput(session_id="s1", event_id="e1", kind="ack", message="ok", anomaly=None)
    assert event_out.kind == "ack"
    review = SessionReviewOutput(session_id="s1", summary="ok", risk_flags=[], action_items=[], citations=[])
    assert review.risk_flags == []
    event = MonitorEvent(seq=1, session_id="s1", phase="start", direction="input", timestamp="t", payload={"a": 1})
    snap = MonitorSnapshot(events=[event], next_cursor=1, dropped_before=None)
    assert snap.events[0].direction == "input"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest trainer/demo_api/tests/test_sim_contracts.py -q`
Expected: FAIL (`ModuleNotFoundError: trainer.demo_api.sim_contracts`).

- [ ] **Step 3: Write minimal implementation**

```python
# trainer/demo_api/sim_contracts.py
from __future__ import annotations

from typing import Annotated, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

Phase = Literal["start", "runtime", "session_end"]
Direction = Literal["input", "output"]


class StrictModel(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")


class WorkerProfile(StrictModel):
    trainee_id: str
    skill_level: Literal["beginner", "intermediate", "advanced"]
    handedness: Literal["left", "right"]


class PaintSpec(StrictModel):
    nominal_um: float


class SessionStartInput(StrictModel):
    session_id: Annotated[str, Field(min_length=1)]
    level_id: Annotated[str, Field(min_length=1)]
    worker: WorkerProfile
    spec: PaintSpec


class GunMetrics(StrictModel):
    speed_m_per_s: float
    standoff_mm: float
    angle_deg: float


class EventTelemetry(StrictModel):
    gun_metrics: GunMetrics | None = None
    pose_joints: dict[str, float] | None = None
    coordinates: list[float] | None = None


class SessionEventInput(StrictModel):
    event_id: Annotated[str, Field(min_length=1)]
    user_request: str | None = None
    telemetry: EventTelemetry | None = None


class SessionStartOutput(StrictModel):
    session_id: str
    recommended_flow: list[str]
    flow_source: Literal["rule"]
    guidance: str


class SessionEventOutput(StrictModel):
    session_id: str
    event_id: str
    kind: Literal["rag", "anomaly", "ack"]
    message: str
    anomaly: dict[str, object] | None = None


class SessionReviewOutput(StrictModel):
    session_id: str
    summary: str
    risk_flags: list[dict[str, object]]
    action_items: list[dict[str, str]]
    citations: list[str]


class MonitorEvent(StrictModel):
    seq: int
    session_id: str
    phase: Phase
    direction: Direction
    timestamp: str
    payload: dict[str, object]


class MonitorSnapshot(StrictModel):
    events: list[MonitorEvent]
    next_cursor: int
    dropped_before: int | None = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest trainer/demo_api/tests/test_sim_contracts.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add trainer/demo_api/sim_contracts.py trainer/demo_api/tests/test_sim_contracts.py
git commit -m "feat(demo-api): 시뮬레이터 인터페이스·모니터 Pydantic 계약"
```

---

## Task 2: 모니터 레코더 (ring buffer + broadcast)

**Files:**
- Create: `trainer/demo_api/monitor.py`
- Test: `trainer/demo_api/tests/test_monitor_recorder.py`

**Interfaces:**
- Consumes: `MonitorEvent`, `MonitorSnapshot`, `Phase`, `Direction` (Task 1).
- Produces: `MonitorRecorder` with `record(session_id, phase, direction, payload) -> MonitorEvent`, `snapshot(cursor, limit) -> MonitorSnapshot`, `subscribe() -> tuple[MemoryObjectSendStream[MonitorEvent], MemoryObjectReceiveStream[MonitorEvent]]`, `unsubscribe(send) -> None`.

- [ ] **Step 1: Write the failing test**

```python
# trainer/demo_api/tests/test_monitor_recorder.py
from __future__ import annotations

import anyio

from trainer.demo_api.monitor import MonitorRecorder


def test_record_assigns_monotonic_seq_and_snapshot_filters_by_cursor() -> None:
    rec = MonitorRecorder(maxlen=10)
    a = rec.record("s1", "start", "input", {"k": 1})
    b = rec.record("s1", "start", "output", {"k": 2})
    assert (a.seq, b.seq) == (1, 2)
    snap = rec.snapshot(cursor=1, limit=10)
    assert [e.seq for e in snap.events] == [2]
    assert snap.next_cursor == 2
    assert snap.dropped_before is None


def test_snapshot_reports_gap_when_cursor_predates_buffer() -> None:
    rec = MonitorRecorder(maxlen=3)
    for i in range(5):
        rec.record("s1", "runtime", "input", {"i": i})
    snap = rec.snapshot(cursor=0, limit=10)
    assert snap.dropped_before == snap.events[0].seq
    assert [e.seq for e in snap.events] == [3, 4, 5]


def test_subscribe_receives_live_events() -> None:
    async def _run() -> list[int]:
        rec = MonitorRecorder(maxlen=10)
        send, receive = rec.subscribe()
        rec.record("s1", "start", "input", {"x": 1})
        rec.record("s1", "start", "output", {"x": 2})
        seqs: list[int] = []
        with anyio.fail_after(1.0):
            async with receive:
                for _ in range(2):
                    ev = await receive.receive()
                    seqs.append(ev.seq)
        rec.unsubscribe(send)
        return seqs

    assert anyio.run(_run) == [1, 2]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest trainer/demo_api/tests/test_monitor_recorder.py -q`
Expected: FAIL (`ModuleNotFoundError: trainer.demo_api.monitor`).

- [ ] **Step 3: Write minimal implementation**

```python
# trainer/demo_api/monitor.py
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
    """Bounded in-memory recorder for simulator interface I/O with live broadcast.

    record() is called from async request handlers (event loop), so send_nowait
    to subscriber streams stays on the loop. Thread lock guards buffer mutation.
    """

    def __init__(self, maxlen: int = _MAX_EVENTS) -> None:
        self._events: deque[MonitorEvent] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._seq = 0
        self._subscribers: list[MemoryObjectSendStream[MonitorEvent]] = []

    def record(self, session_id: str, phase: Phase, direction: Direction, payload: dict[str, object]) -> MonitorEvent:
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
                # The buffer stays authoritative, so a reconnect replays from cursor.
                pass
        return event

    def snapshot(self, cursor: int, limit: int) -> MonitorSnapshot:
        with self._lock:
            events = list(self._events)
        dropped_before: int | None = None
        if events and cursor < events[0].seq - 1:
            dropped_before = events[0].seq
        selected = [e for e in events if e.seq > cursor][:limit]
        next_cursor = selected[-1].seq if selected else cursor
        return MonitorSnapshot(events=selected, next_cursor=next_cursor, dropped_before=dropped_before)

    def subscribe(self) -> tuple[MemoryObjectSendStream[MonitorEvent], MemoryObjectReceiveStream[MonitorEvent]]:
        send, receive = anyio.create_memory_object_stream[MonitorEvent](max_buffer_size=_SUBSCRIBER_BUFFER)
        with self._lock:
            self._subscribers.append(send)
        return send, receive

    def unsubscribe(self, send: MemoryObjectSendStream[MonitorEvent]) -> None:
        with self._lock:
            if send in self._subscribers:
                self._subscribers.remove(send)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest trainer/demo_api/tests/test_monitor_recorder.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add trainer/demo_api/monitor.py trainer/demo_api/tests/test_monitor_recorder.py
git commit -m "feat(demo-api): 모니터 ring buffer 레코더 + live broadcast"
```

---

## Task 3: 결정론 출력 빌더

**Files:**
- Create: `trainer/demo_api/simulator_stub.py`
- Test: `trainer/demo_api/tests/test_simulator_stub.py`

**Interfaces:**
- Consumes: Task 1 models; `default_snapshot` (`trainer/demo_api/sequence.py`); `SessionSummary`, `compute_risk_flags` (`trainer/scripts/session_coaching.py`).
- Produces: `build_start_output(payload: SessionStartInput) -> SessionStartOutput`; `build_event_output(session_id: str, payload: SessionEventInput) -> SessionEventOutput`; `build_review_output(payload: SessionSummary) -> SessionReviewOutput`; constants `STANDOFF_LIMIT_MM=320.0`, `ANGLE_MIN_DEG=60.0`.

- [ ] **Step 1: Write the failing test**

```python
# trainer/demo_api/tests/test_simulator_stub.py
from __future__ import annotations

from trainer.demo_api import simulator_stub as stub
from trainer.demo_api.sim_contracts import (
    EventTelemetry,
    GunMetrics,
    SessionEventInput,
    SessionStartInput,
    WorkerProfile,
)
from trainer.scripts.session_coaching import SessionMetrics, SessionSummary


def _start_input() -> SessionStartInput:
    return SessionStartInput(
        session_id="s1",
        level_id="SP0035-L01",
        worker=WorkerProfile(trainee_id="t", skill_level="beginner", handedness="right"),
        spec={"nominal_um": 100.0},
    )


def test_start_output_uses_default_snapshot_zone_order() -> None:
    out = stub.build_start_output(_start_input())
    assert out.recommended_flow == ["A-1", "A-2", "A-3"]
    assert out.flow_source == "rule"
    assert "SP0035-L01" in out.guidance


def test_event_output_rag_when_user_request_present() -> None:
    out = stub.build_event_output("s1", SessionEventInput(event_id="e1", user_request="블라스팅이 뭐야"))
    assert out.kind == "rag"
    assert out.anomaly is None


def test_event_output_anomaly_when_gun_metrics_out_of_band() -> None:
    tele = EventTelemetry(gun_metrics=GunMetrics(speed_m_per_s=0.8, standoff_mm=350.0, angle_deg=55.0))
    out = stub.build_event_output("s1", SessionEventInput(event_id="e2", telemetry=tele))
    assert out.kind == "anomaly"
    assert out.anomaly is not None
    assert out.anomaly["severity"] == "high"


def test_event_output_ack_when_in_band_telemetry() -> None:
    tele = EventTelemetry(gun_metrics=GunMetrics(speed_m_per_s=0.8, standoff_mm=300.0, angle_deg=72.0))
    out = stub.build_event_output("s1", SessionEventInput(event_id="e3", telemetry=tele))
    assert out.kind == "ack"


def test_review_output_reuses_compute_risk_flags() -> None:
    metrics = SessionMetrics(
        avg_dft=100.0, dft_values=[100.0], avg_speed=0.8, speed_variance=0.05,
        avg_distance=0.3, distance_variance=0.05, avg_angle=70.0, angle_consistency=0.9,
        sequence_correct=False, under_thickness_count=0, over_thickness_count=0,
    )
    summary = SessionSummary(session_id="s1", scenario_id="sc", mode="live", zone_id="A-2", zone_type="panel", metrics=metrics)
    out = stub.build_review_output(summary)
    assert any(f["flag"] == "sequence_violation" for f in out.risk_flags)
    assert out.action_items
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest trainer/demo_api/tests/test_simulator_stub.py -q`
Expected: FAIL (`ModuleNotFoundError: trainer.demo_api.simulator_stub`).

- [ ] **Step 3: Write minimal implementation**

```python
# trainer/demo_api/simulator_stub.py
from __future__ import annotations

from typing import Final

from trainer.demo_api.sequence import default_snapshot
from trainer.demo_api.sim_contracts import (
    SessionEventInput,
    SessionEventOutput,
    SessionReviewOutput,
    SessionStartInput,
    SessionStartOutput,
)
from trainer.scripts.session_coaching import SessionSummary, compute_risk_flags

STANDOFF_LIMIT_MM: Final[float] = 320.0
ANGLE_MIN_DEG: Final[float] = 60.0


def build_start_output(payload: SessionStartInput) -> SessionStartOutput:
    zones = [zone.zone_id for zone in default_snapshot(payload.level_id).zones]
    guidance = f"레벨 {payload.level_id} 권장 도장 순서: {' → '.join(zones)}. 상부에서 하부 순으로 진행하십시오."
    return SessionStartOutput(
        session_id=payload.session_id,
        recommended_flow=zones,
        flow_source="rule",
        guidance=guidance,
    )


def build_event_output(session_id: str, payload: SessionEventInput) -> SessionEventOutput:
    if payload.user_request is not None and payload.user_request.strip():
        return SessionEventOutput(
            session_id=session_id,
            event_id=payload.event_id,
            kind="rag",
            message=f"질의 응답(스텁): '{payload.user_request.strip()}' 관련 문서 근거를 확인하세요.",
            anomaly=None,
        )
    gun = payload.telemetry.gun_metrics if payload.telemetry is not None else None
    if gun is not None and (gun.standoff_mm > STANDOFF_LIMIT_MM or gun.angle_deg < ANGLE_MIN_DEG):
        both = gun.standoff_mm > STANDOFF_LIMIT_MM and gun.angle_deg < ANGLE_MIN_DEG
        anomaly: dict[str, object] = {
            "type": "spray_param_out_of_band",
            "standoff_mm": gun.standoff_mm,
            "angle_deg": gun.angle_deg,
            "severity": "high" if both else "medium",
        }
        return SessionEventOutput(
            session_id=session_id,
            event_id=payload.event_id,
            kind="anomaly",
            message="스프레이 파라미터 이탈 감지(스텁).",
            anomaly=anomaly,
        )
    return SessionEventOutput(
        session_id=session_id,
        event_id=payload.event_id,
        kind="ack",
        message="이벤트 누적됨.",
        anomaly=None,
    )


def build_review_output(payload: SessionSummary) -> SessionReviewOutput:
    flags = compute_risk_flags(payload.metrics)
    if flags:
        summary = f"세션 {payload.session_id} 리뷰: 위험 플래그 {len(flags)}건 감지."
        action_items = [
            {"action": f"{flag['flag']} 개선 필요", "priority": str(flag["severity"])}
            for flag in flags[:3]
        ]
    else:
        summary = f"세션 {payload.session_id} 리뷰: 위험 플래그 없음. 기준 충족."
        action_items = [{"action": "현재 작업 품질 유지.", "priority": "low"}]
    return SessionReviewOutput(
        session_id=payload.session_id,
        summary=summary,
        risk_flags=flags,
        action_items=action_items,
        citations=[],
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest trainer/demo_api/tests/test_simulator_stub.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add trainer/demo_api/simulator_stub.py trainer/demo_api/tests/test_simulator_stub.py
git commit -m "feat(demo-api): 결정론 시뮬레이터 출력 빌더 (default_snapshot/compute_risk_flags 재사용)"
```

---

## Task 4: 시뮬레이터 인터페이스 라우터

**Files:**
- Create: `trainer/demo_api/simulator.py`
- Test: `trainer/demo_api/tests/test_simulator_interface.py`

**Interfaces:**
- Consumes: `MonitorRecorder` (Task 2); Task 1 input models; `simulator_stub` builders (Task 3); `SessionSummary`.
- Produces: `create_simulator_router(recorder: MonitorRecorder) -> APIRouter` mounting `POST /api/sim/sessions/start`, `POST /api/sim/sessions/{session_id}/events`, `POST /api/sim/sessions/{session_id}/complete`.

- [ ] **Step 1: Write the failing test**

```python
# trainer/demo_api/tests/test_simulator_interface.py
from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from trainer.demo_api.monitor import MonitorRecorder
from trainer.demo_api.simulator import create_simulator_router


def _client() -> tuple[TestClient, MonitorRecorder]:
    recorder = MonitorRecorder()
    app = FastAPI()
    app.include_router(create_simulator_router(recorder))
    return TestClient(app), recorder


def _start_body() -> dict[str, object]:
    return {
        "session_id": "s1",
        "level_id": "SP0035-L01",
        "worker": {"trainee_id": "t", "skill_level": "beginner", "handedness": "right"},
        "spec": {"nominal_um": 100.0},
    }


def test_start_returns_recommended_flow_and_records_input_output() -> None:
    client, recorder = _client()
    resp = client.post("/api/sim/sessions/start", json=_start_body())
    assert resp.status_code == 200
    assert resp.json()["recommended_flow"] == ["A-1", "A-2", "A-3"]
    snap = recorder.snapshot(cursor=0, limit=10)
    assert [(e.phase, e.direction) for e in snap.events] == [("start", "input"), ("start", "output")]


def test_event_anomaly_round_trip_records_pair() -> None:
    client, recorder = _client()
    body = {"event_id": "e1", "telemetry": {"gun_metrics": {"speed_m_per_s": 0.8, "standoff_mm": 350.0, "angle_deg": 55.0}}}
    resp = client.post("/api/sim/sessions/s1/events", json=body)
    assert resp.status_code == 200
    assert resp.json()["kind"] == "anomaly"
    snap = recorder.snapshot(cursor=0, limit=10)
    assert [(e.phase, e.direction) for e in snap.events] == [("runtime", "input"), ("runtime", "output")]


def test_invalid_input_returns_422_and_records_error_output() -> None:
    client, recorder = _client()
    resp = client.post("/api/sim/sessions/start", json={"session_id": "s1"})
    assert resp.status_code == 422
    snap = recorder.snapshot(cursor=0, limit=10)
    assert snap.events[0].direction == "input"
    assert snap.events[1].payload["error"] == "validation_error"


def test_path_body_session_id_mismatch_rejected() -> None:
    client, _recorder = _client()
    body = {"session_id": "OTHER", "event_id": "e1"}
    resp = client.post("/api/sim/sessions/s1/events", json=body)
    assert resp.status_code == 422
    assert resp.json()["error"] == "session_id_mismatch"


def test_complete_review_uses_risk_flags() -> None:
    client, _recorder = _client()
    body = {
        "session_id": "s1", "scenario_id": "sc", "mode": "live", "zone_id": "A-2", "zone_type": "panel",
        "metrics": {
            "avg_dft": 100.0, "dft_values": [100.0], "avg_speed": 0.8, "speed_variance": 0.05,
            "avg_distance": 0.3, "distance_variance": 0.05, "avg_angle": 70.0, "angle_consistency": 0.9,
            "sequence_correct": False,
        },
    }
    resp = client.post("/api/sim/sessions/s1/complete", json=body)
    assert resp.status_code == 200
    assert any(f["flag"] == "sequence_violation" for f in resp.json()["risk_flags"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest trainer/demo_api/tests/test_simulator_interface.py -q`
Expected: FAIL (`ModuleNotFoundError: trainer.demo_api.simulator`).

- [ ] **Step 3: Write minimal implementation**

```python
# trainer/demo_api/simulator.py
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TypeVar

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError

from trainer.demo_api import simulator_stub as stub
from trainer.demo_api.monitor import MonitorRecorder
from trainer.demo_api.sim_contracts import Phase, SessionEventInput, SessionStartInput
from trainer.scripts.session_coaching import SessionSummary

logger = logging.getLogger(__name__)

TInput = TypeVar("TInput", bound=BaseModel)


def create_simulator_router(recorder: MonitorRecorder) -> APIRouter:
    router = APIRouter(prefix="/api/sim", tags=["simulator"])

    async def _run_phase(
        request: Request,
        phase: Phase,
        path_session_id: str | None,
        model_cls: type[TInput],
        build_output: Callable[[TInput], BaseModel],
    ) -> JSONResponse:
        error: dict[str, object]
        try:
            raw: object = await request.json()
        except ValueError:
            sid = path_session_id or "unknown"
            _ = recorder.record(sid, phase, "input", {"error": "invalid_json"})
            error = {"error": "invalid_json"}
            _ = recorder.record(sid, phase, "output", error)
            return JSONResponse(status_code=422, content=error)

        body: dict[str, object] = raw if isinstance(raw, dict) else {}
        session_id = path_session_id or str(body.get("session_id", "")) or "unknown"
        _ = recorder.record(session_id, phase, "input", body)

        if path_session_id is not None and "session_id" in body and body["session_id"] != path_session_id:
            error = {"error": "session_id_mismatch", "path": path_session_id, "body": body["session_id"]}
            _ = recorder.record(session_id, phase, "output", error)
            return JSONResponse(status_code=422, content=error)

        try:
            payload = model_cls.model_validate(body)
        except ValidationError as exc:
            # include_url/include_context off: pydantic puts raw exception objects
            # in ctx, and JSONResponse would then fail to serialize the body.
            error = {"error": "validation_error", "detail": exc.errors(include_url=False, include_context=False)}
            _ = recorder.record(session_id, phase, "output", error)
            return JSONResponse(status_code=422, content=error)

        output = build_output(payload)
        out_payload: dict[str, object] = output.model_dump(mode="json")
        _ = recorder.record(session_id, phase, "output", out_payload)
        return JSONResponse(status_code=200, content=out_payload)

    @router.post("/sessions/start")
    async def start(request: Request) -> JSONResponse:
        return await _run_phase(request, "start", None, SessionStartInput, stub.build_start_output)

    @router.post("/sessions/{session_id}/events")
    async def events(session_id: str, request: Request) -> JSONResponse:
        return await _run_phase(
            request, "runtime", session_id, SessionEventInput, lambda p: stub.build_event_output(session_id, p)
        )

    @router.post("/sessions/{session_id}/complete")
    async def complete(session_id: str, request: Request) -> JSONResponse:
        return await _run_phase(request, "session_end", session_id, SessionSummary, stub.build_review_output)

    return router
```

> Note: `start`/`events`/`complete` are referenced by FastAPI via decorators; if the linter flags them unused, add `# pyright: reportUnusedFunction=false` at file top (matches `app.py`).

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest trainer/demo_api/tests/test_simulator_interface.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add trainer/demo_api/simulator.py trainer/demo_api/tests/test_simulator_interface.py
git commit -m "feat(demo-api): /api/sim 3단계 인터페이스 라우터 + I/O 기록"
```

---

## Task 5: 모니터 라우터 (스냅샷 + SSE)

**Files:**
- Create: `trainer/demo_api/monitor_api.py`
- Test: `trainer/demo_api/tests/test_monitor_api.py`

**Interfaces:**
- Consumes: `MonitorRecorder` (Task 2).
- Produces: `monitor_sse_body(recorder: MonitorRecorder, cursor: int) -> AsyncGenerator[bytes, None]` (순수, 직접 테스트 대상); `create_monitor_router(recorder: MonitorRecorder) -> APIRouter` mounting `GET /api/monitor/events?cursor&limit` (JSON `MonitorSnapshot`) and `GET /api/monitor/stream?cursor` (SSE `event: monitor` / `event: gap`).

> **왜 generator를 분리하나:** Global Constraints의 "TestClient 제약" 참고. 끝나지 않는 SSE 라우트는 `TestClient`로 못 부른다(영구 블로킹, 실측 확인). 본문을 순수 generator로 빼면 `anyio`로 프레임을 정확히 검증할 수 있고, 라우트는 얇은 wrapper라 Task 6(경로 등록)과 Task 12(실 uvicorn + curl)로 덮인다.

- [ ] **Step 1: Write the failing test**

```python
# trainer/demo_api/tests/test_monitor_api.py
from __future__ import annotations

import json

import anyio
from fastapi import FastAPI
from fastapi.testclient import TestClient

from trainer.demo_api.monitor import MonitorRecorder
from trainer.demo_api.monitor_api import create_monitor_router, monitor_sse_body


def _parse_frame(raw: bytes) -> tuple[str, dict[str, object]]:
    lines = raw.decode("utf-8").split("\n")
    payload: dict[str, object] = json.loads(lines[1].removeprefix("data: "))
    return lines[0].removeprefix("event: "), payload


def test_events_snapshot_returns_ordered_events() -> None:
    recorder = MonitorRecorder(maxlen=3)
    app = FastAPI()
    app.include_router(create_monitor_router(recorder))
    client = TestClient(app)

    _ = recorder.record("s1", "start", "input", {"a": 1})
    _ = recorder.record("s1", "start", "output", {"a": 2})

    response = client.get("/api/monitor/events", params={"cursor": 0, "limit": 10})
    assert response.status_code == 200
    body = response.json()
    assert [event["seq"] for event in body["events"]] == [1, 2]
    assert body["next_cursor"] == 2
    assert body["dropped_before"] is None


def test_sse_body_emits_gap_then_replay_then_live() -> None:
    async def _run() -> list[tuple[str, dict[str, object]]]:
        recorder = MonitorRecorder(maxlen=3)
        for index in range(5):
            _ = recorder.record("s1", "runtime", "input", {"i": index})

        frames: list[tuple[str, dict[str, object]]] = []
        body = monitor_sse_body(recorder, cursor=0)
        with anyio.fail_after(2.0):
            # gap marker + the 3 events still in the bounded buffer
            for _ in range(4):
                frames.append(_parse_frame(await anext(body)))
            # recorded after the generator subscribed -> forwarded live
            _ = recorder.record("s1", "session_end", "output", {"done": True})
            frames.append(_parse_frame(await anext(body)))
        await body.aclose()
        return frames

    frames = anyio.run(_run)
    assert [name for name, _ in frames] == ["gap", "monitor", "monitor", "monitor", "monitor"]
    assert frames[0][1]["dropped_before"] == 3
    assert [payload["seq"] for _, payload in frames[1:]] == [3, 4, 5, 6]
    assert frames[4][1]["phase"] == "session_end"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest trainer/demo_api/tests/test_monitor_api.py -q`
Expected: FAIL (`ModuleNotFoundError: trainer.demo_api.monitor_api`).

- [ ] **Step 3: Write minimal implementation**

```python
# trainer/demo_api/monitor_api.py
# pyright: reportUnusedFunction=false
from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Final

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse

from trainer.demo_api.monitor import MonitorRecorder

_REPLAY_LIMIT: Final[int] = 10_000


def _frame(event: str, data: dict[str, object]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")


async def monitor_sse_body(recorder: MonitorRecorder, cursor: int) -> AsyncGenerator[bytes, None]:
    """Replay buffered events after `cursor`, then forward live ones.

    Subscribing before taking the snapshot is what makes the handoff lossless:
    an event recorded between the two lands in the subscriber queue and is
    de-duplicated by seq, instead of falling into the gap between them.
    """
    send, receive = recorder.subscribe()
    try:
        snapshot = recorder.snapshot(cursor=cursor, limit=_REPLAY_LIMIT)
        if snapshot.dropped_before is not None:
            yield _frame("gap", {"dropped_before": snapshot.dropped_before})
        last = cursor
        for event in snapshot.events:
            yield _frame("monitor", event.model_dump(mode="json"))
            last = event.seq
        async with receive:
            async for event in receive:
                if event.seq <= last:
                    continue
                yield _frame("monitor", event.model_dump(mode="json"))
                last = event.seq
    finally:
        recorder.unsubscribe(send)
        send.close()


def create_monitor_router(recorder: MonitorRecorder) -> APIRouter:
    router = APIRouter(prefix="/api/monitor", tags=["monitor"])

    @router.get("/events")
    def events(cursor: int = 0, limit: int = 200) -> JSONResponse:
        snapshot = recorder.snapshot(cursor=cursor, limit=limit)
        return JSONResponse(content=snapshot.model_dump(mode="json"))

    @router.get("/stream")
    async def stream(cursor: int = 0) -> StreamingResponse:
        return StreamingResponse(
            monitor_sse_body(recorder, cursor),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return router
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest trainer/demo_api/tests/test_monitor_api.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add trainer/demo_api/monitor_api.py trainer/demo_api/tests/test_monitor_api.py
git commit -m "feat(demo-api): /api/monitor 스냅샷 + read-only SSE 스트림"
```

---

## Task 6: app.py 배선

**Files:**
- Modify: `trainer/demo_api/app.py` (inside `create_app()`, before `return app`)
- Test: `trainer/demo_api/tests/test_app_sim_wiring.py`

**Interfaces:**
- Consumes: `MonitorRecorder`, `create_simulator_router`, `create_monitor_router`.
- Produces: mounted routes on the app; `app.state.monitor_recorder`.

- [ ] **Step 1: Write the failing test**

```python
# trainer/demo_api/tests/test_app_sim_wiring.py
from __future__ import annotations

from trainer.demo_api.app import create_app


def test_app_mounts_simulator_and_monitor_routes_without_touching_existing() -> None:
    app = create_app()
    paths = {getattr(route, "path", "") for route in app.routes}
    assert "/api/sim/sessions/start" in paths
    assert "/api/sim/sessions/{session_id}/events" in paths
    assert "/api/sim/sessions/{session_id}/complete" in paths
    assert "/api/monitor/events" in paths
    assert "/api/monitor/stream" in paths
    # existing routes intact
    assert "/api/health" in paths
    assert "/api/chat" in paths
    assert hasattr(app.state, "monitor_recorder")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest trainer/demo_api/tests/test_app_sim_wiring.py -q`
Expected: FAIL (sim/monitor paths absent).

- [ ] **Step 3: Write minimal implementation**

In `trainer/demo_api/app.py`, add imports near the existing demo_api imports:

```python
from trainer.demo_api.monitor import MonitorRecorder
from trainer.demo_api.monitor_api import create_monitor_router
from trainer.demo_api.simulator import create_simulator_router
```

Inside `create_app()`, immediately before `return app`, add:

```python
    recorder = MonitorRecorder()
    app.state.monitor_recorder = recorder
    app.include_router(create_simulator_router(recorder))
    app.include_router(create_monitor_router(recorder))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest trainer/demo_api/tests/test_app_sim_wiring.py trainer/demo_api/tests -q`
Expected: PASS (new test + existing demo_api tests still green).

- [ ] **Step 5: Commit**

```bash
git add trainer/demo_api/app.py trainer/demo_api/tests/test_app_sim_wiring.py
git commit -m "feat(demo-api): app에 시뮬레이터·모니터 라우터 배선"
```

---

## Task 7: CLI 더미 드라이버

**Files:**
- Create: `trainer/scripts/sim_dummy_driver.py`
- Test: `trainer/scripts/tests/test_sim_dummy_driver.py` (스크립트 테스트는 `trainer/scripts/tests/`에 두는 기존 관례)

**Interfaces:**
- Produces: `build_dummy_session(session_id: str, events: int, anomaly: bool) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object]]` (pure, testable) and `run(api_url, session_id, events, anomaly) -> None` (httpx POSTs). CLI via `--api-url --session-id --events --anomaly`.

> `trainer/scripts/`에는 `__init__.py`가 없지만 `trainer`가 정규 패키지라 `trainer.scripts.*`가 네임스페이스 패키지로 임포트된다(기존 `trainer.scripts.session_coaching`과 동일, 실측 확인). importlib 수동 로딩 불필요.

- [ ] **Step 1: Write the failing test**

```python
# trainer/scripts/tests/test_sim_dummy_driver.py
from __future__ import annotations

from trainer.scripts.sim_dummy_driver import build_dummy_session


def test_build_dummy_session_shapes_three_phases() -> None:
    start, events, complete = build_dummy_session("s1", events=2, anomaly=True)
    assert start["session_id"] == "s1"
    assert "worker" in start and "spec" in start
    assert len(events) == 2
    assert events[0]["event_id"] == "ev-0"
    assert any("telemetry" in event for event in events)
    assert complete["session_id"] == "s1"
    assert "metrics" in complete


def test_anomaly_flag_injects_out_of_band_gun_metrics() -> None:
    _start, events, _complete = build_dummy_session("s1", events=1, anomaly=True)
    telemetry = events[0]["telemetry"]
    assert isinstance(telemetry, dict)
    gun_metrics = telemetry["gun_metrics"]
    assert isinstance(gun_metrics, dict)
    assert gun_metrics["standoff_mm"] > 320.0 or gun_metrics["angle_deg"] < 60.0


def test_clean_run_keeps_every_event_in_band() -> None:
    _start, events, complete = build_dummy_session("s1", events=4, anomaly=False)
    for event in events:
        telemetry = event.get("telemetry")
        if not isinstance(telemetry, dict):
            continue
        gun_metrics = telemetry["gun_metrics"]
        assert isinstance(gun_metrics, dict)
        assert gun_metrics["standoff_mm"] <= 320.0
        assert gun_metrics["angle_deg"] >= 60.0
    metrics = complete["metrics"]
    assert isinstance(metrics, dict)
    assert metrics["sequence_correct"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest trainer/scripts/tests/test_sim_dummy_driver.py -q`
Expected: FAIL (`ModuleNotFoundError: trainer.scripts.sim_dummy_driver`).

- [ ] **Step 3: Write minimal implementation**

```python
# trainer/scripts/sim_dummy_driver.py
# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx"]
# ///
# ─── How to run ───
#   MCP/데모 API 기동 후:
#   python3 trainer/scripts/sim_dummy_driver.py --api-url http://127.0.0.1:8120 --events 3 --anomaly
from __future__ import annotations

import argparse
import json

import httpx


def build_dummy_session(
    session_id: str, events: int, anomaly: bool
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object]]:
    start: dict[str, object] = {
        "session_id": session_id,
        "level_id": "SP0035-L01",
        "worker": {"trainee_id": "t-dummy", "skill_level": "beginner", "handedness": "right"},
        "spec": {"nominal_um": 100.0},
    }
    event_payloads: list[dict[str, object]] = []
    for index in range(events):
        if anomaly and index == 0:
            event_payloads.append(
                {
                    "event_id": f"ev-{index}",
                    "telemetry": {"gun_metrics": {"speed_m_per_s": 0.8, "standoff_mm": 350.0, "angle_deg": 55.0}},
                }
            )
        elif index % 2 == 0:
            event_payloads.append(
                {
                    "event_id": f"ev-{index}",
                    "telemetry": {"gun_metrics": {"speed_m_per_s": 0.8, "standoff_mm": 300.0, "angle_deg": 72.0}},
                }
            )
        else:
            event_payloads.append({"event_id": f"ev-{index}", "user_request": "에폭시 도료란?"})
    complete: dict[str, object] = {
        "session_id": session_id,
        "scenario_id": "dummy",
        "mode": "live",
        "zone_id": "A-2",
        "zone_type": "panel",
        "metrics": {
            "avg_dft": 100.0, "dft_values": [95.0, 102.0, 88.0], "avg_speed": 0.8, "speed_variance": 0.05,
            "avg_distance": 0.3, "distance_variance": 0.05, "avg_angle": 70.0, "angle_consistency": 0.9,
            "sequence_correct": not anomaly, "under_thickness_count": 1 if anomaly else 0,
        },
    }
    return start, event_payloads, complete


def run(api_url: str, session_id: str, events: int, anomaly: bool) -> None:
    start, event_payloads, complete = build_dummy_session(session_id, events, anomaly)
    base = api_url.rstrip("/")
    with httpx.Client(timeout=10.0) as client:
        r0 = client.post(f"{base}/api/sim/sessions/start", json=start)
        print("start ->", r0.status_code, json.dumps(r0.json(), ensure_ascii=False))
        for payload in event_payloads:
            r = client.post(f"{base}/api/sim/sessions/{session_id}/events", json=payload)
            print("event ->", r.status_code, json.dumps(r.json(), ensure_ascii=False))
        rc = client.post(f"{base}/api/sim/sessions/{session_id}/complete", json=complete)
        print("complete ->", rc.status_code, json.dumps(rc.json(), ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Dummy simulator interface driver.")
    _ = parser.add_argument("--api-url", default="http://127.0.0.1:8120")
    _ = parser.add_argument("--session-id", default="dummy-session")
    _ = parser.add_argument("--events", type=int, default=3)
    _ = parser.add_argument("--anomaly", action="store_true")
    args = parser.parse_args()
    run(args.api_url, args.session_id, args.events, args.anomaly)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest trainer/scripts/tests/test_sim_dummy_driver.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add trainer/scripts/sim_dummy_driver.py trainer/scripts/tests/test_sim_dummy_driver.py
git commit -m "feat(scripts): 시뮬레이터 인터페이스 더미 드라이버 (CLI)"
```

---

## Task 8: 프론트 Zod 계약

**Files:**
- Create: `demo-ui/src/simContracts.ts`
- Test: `demo-ui/tests/simContracts.test.ts`

**Interfaces:**
- Produces: `phaseSchema`, `directionSchema`, `monitorEventSchema`, `monitorSnapshotSchema`; types `Phase`, `MonitorEvent`, `MonitorSnapshot`.

- [ ] **Step 1: Write the failing test**

```ts
// demo-ui/tests/simContracts.test.ts
import { describe, expect, it } from 'vitest'
import { monitorEventSchema, monitorSnapshotSchema } from '../src/simContracts'

describe('simContracts', () => {
  it('parses a valid monitor event', () => {
    const event = monitorEventSchema.parse({
      seq: 1, session_id: 's1', phase: 'start', direction: 'input', timestamp: 't', payload: { a: 1 },
    })
    expect(event.phase).toBe('start')
  })

  it('rejects an unknown phase', () => {
    expect(() =>
      monitorEventSchema.parse({ seq: 1, session_id: 's1', phase: 'bogus', direction: 'input', timestamp: 't', payload: {} }),
    ).toThrow()
  })

  it('parses a snapshot with nullable dropped_before', () => {
    const snap = monitorSnapshotSchema.parse({ events: [], next_cursor: 0, dropped_before: null })
    expect(snap.next_cursor).toBe(0)
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd demo-ui && npm test -- --run simContracts`
Expected: FAIL (cannot resolve `../src/simContracts`).

- [ ] **Step 3: Write minimal implementation**

```ts
// demo-ui/src/simContracts.ts
import { z } from 'zod'

export const phaseSchema = z.enum(['start', 'runtime', 'session_end'])
export type Phase = z.infer<typeof phaseSchema>

export const directionSchema = z.enum(['input', 'output'])
export type Direction = z.infer<typeof directionSchema>

export const monitorEventSchema = z
  .object({
    seq: z.number().int().nonnegative(),
    session_id: z.string(),
    phase: phaseSchema,
    direction: directionSchema,
    timestamp: z.string(),
    payload: z.record(z.string(), z.unknown()),
  })
  .strict()
export type MonitorEvent = z.infer<typeof monitorEventSchema>

export const monitorSnapshotSchema = z
  .object({
    events: z.array(monitorEventSchema),
    next_cursor: z.number().int().nonnegative(),
    dropped_before: z.number().int().nonnegative().nullable().default(null),
  })
  .strict()
export type MonitorSnapshot = z.infer<typeof monitorSnapshotSchema>
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd demo-ui && npm test -- --run simContracts`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add demo-ui/src/simContracts.ts demo-ui/tests/simContracts.test.ts
git commit -m "feat(demo-ui): 모니터 Zod 계약"
```

---

## Task 9: 프론트 모니터 클라이언트

**Files:**
- Create: `demo-ui/src/simMonitorClient.ts`
- Modify: `demo-ui/src/api.ts` — `const API_BASE_URL` 앞에 `export` 추가(1단어). 새 파일에서 base URL 규칙을 복제하지 않기 위함.
- Test: `demo-ui/tests/simMonitorClient.test.ts`

**Interfaces:**
- Consumes: `API_BASE_URL`, `createSseParser` (`./api`), `monitorEventSchema` (Task 8).
- Produces: `streamMonitor(cursor?, signal?) -> AsyncGenerator<MonitorEvent>`; `sendDummySession(sessionId: string) -> Promise<void>` (3 POSTs to `/api/sim/*`).

> `/api/monitor/events` 스냅샷은 프론트에서 쓰지 않는다 — `/stream`이 접속 시 cursor 이후를 replay하므로 중복이다(스냅샷은 백엔드 테스트와 Task 12 curl 검증용). `gap` 프레임은 `monitor`가 아니므로 자연히 건너뛴다: 스펙 §10의 "SSE 파서 gap 처리"를 두 번째 테스트로 고정한다.

- [ ] **Step 1: Write the failing test**

```ts
// demo-ui/tests/simMonitorClient.test.ts
import { afterEach, describe, expect, it, vi } from 'vitest'
import { sendDummySession, streamMonitor } from '../src/simMonitorClient'
import type { MonitorEvent } from '../src/simContracts'

afterEach(() => {
  vi.unstubAllGlobals()
})

function sseResponse(frames: readonly string[]): Response {
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      const encoder = new TextEncoder()
      for (const frame of frames) {
        controller.enqueue(encoder.encode(frame))
      }
      controller.close()
    },
  })
  return new Response(body, { status: 200, headers: { 'Content-Type': 'text/event-stream' } })
}

describe('sendDummySession', () => {
  it('issues start, one event, and complete POSTs', async () => {
    const calls: string[] = []
    vi.stubGlobal(
      'fetch',
      vi.fn(async (url: string) => {
        calls.push(url)
        return new Response(JSON.stringify({ ok: true }), { status: 200 })
      }),
    )

    await sendDummySession('s1')

    expect(calls.some((url) => url.endsWith('/api/sim/sessions/start'))).toBe(true)
    expect(calls.some((url) => url.includes('/api/sim/sessions/s1/events'))).toBe(true)
    expect(calls.some((url) => url.endsWith('/api/sim/sessions/s1/complete'))).toBe(true)
  })
})

describe('streamMonitor', () => {
  it('skips the gap marker and yields parsed monitor events', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        sseResponse([
          'event: gap\ndata: {"dropped_before":3}\n\n',
          'event: monitor\ndata: {"seq":3,"session_id":"s1","phase":"start","direction":"input","timestamp":"t","payload":{"level_id":"SP0035-L01"}}\n\n',
          'event: monitor\ndata: {"seq":4,"session_id":"s1","phase":"start","direction":"output","timestamp":"t","payload":{"flow_source":"rule"}}\n\n',
        ]),
      ),
    )

    const received: MonitorEvent[] = []
    for await (const event of streamMonitor(0)) {
      received.push(event)
    }

    expect(received.map((event) => event.seq)).toEqual([3, 4])
    expect(received[1]?.direction).toBe('output')
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd demo-ui && npm test -- --run simMonitorClient`
Expected: FAIL (cannot resolve `../src/simMonitorClient`).

- [ ] **Step 3: Write minimal implementation**

In `demo-ui/src/api.ts`, export the existing base URL constant (line 5) so it is defined once:

```ts
export const API_BASE_URL =
```

Then create the client:

```ts
// demo-ui/src/simMonitorClient.ts
import { API_BASE_URL, createSseParser } from './api'
import { monitorEventSchema, type MonitorEvent } from './simContracts'

async function postJson(path: string, body: unknown): Promise<void> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!response.ok) {
    throw new Error(`sim request failed: ${response.status}`)
  }
}

export async function* streamMonitor(cursor = 0, signal?: AbortSignal): AsyncGenerator<MonitorEvent> {
  const init: RequestInit = signal === undefined ? {} : { signal }
  const response = await fetch(`${API_BASE_URL}/api/monitor/stream?cursor=${cursor}`, init)
  if (response.body === null) {
    throw new Error('monitor stream missing body')
  }
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  const parser = createSseParser()
  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) {
        break
      }
      for (const frame of parser.push(decoder.decode(value, { stream: true }))) {
        if (frame.event === 'monitor') {
          yield monitorEventSchema.parse(frame.data)
        }
      }
    }
  } finally {
    reader.releaseLock()
  }
}

export async function sendDummySession(sessionId: string): Promise<void> {
  await postJson('/api/sim/sessions/start', {
    session_id: sessionId,
    level_id: 'SP0035-L01',
    worker: { trainee_id: 't-dummy', skill_level: 'beginner', handedness: 'right' },
    spec: { nominal_um: 100.0 },
  })
  await postJson(`/api/sim/sessions/${encodeURIComponent(sessionId)}/events`, {
    event_id: 'ev-0',
    telemetry: { gun_metrics: { speed_m_per_s: 0.8, standoff_mm: 350.0, angle_deg: 55.0 } },
  })
  await postJson(`/api/sim/sessions/${encodeURIComponent(sessionId)}/complete`, {
    session_id: sessionId,
    scenario_id: 'dummy',
    mode: 'live',
    zone_id: 'A-2',
    zone_type: 'panel',
    metrics: {
      avg_dft: 100.0, dft_values: [95.0, 102.0, 88.0], avg_speed: 0.8, speed_variance: 0.05,
      avg_distance: 0.3, distance_variance: 0.05, avg_angle: 70.0, angle_consistency: 0.9, sequence_correct: true,
    },
  })
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd demo-ui && npm test -- --run simMonitorClient`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add demo-ui/src/simMonitorClient.ts demo-ui/tests/simMonitorClient.test.ts demo-ui/src/api.ts
git commit -m "feat(demo-ui): 모니터 SSE 클라이언트 + 더미 세션 송신"
```

---

## Task 10: React 모니터 패널

**Files:**
- Create: `demo-ui/src/components/SimulatorMonitorPanel.tsx`
- Create: `demo-ui/src/simMonitor.css` (전용 스타일 — 동시 수정 중인 `styles.css`는 건드리지 않는다)
- Test: `demo-ui/tests/simulatorMonitorPanel.test.tsx`

**Interfaces:**
- Consumes: `MonitorEvent`, `Phase` (Task 8); `sendDummySession` (Task 9).
- Produces: `SimulatorMonitorPanel({ events, onSendDummy }: { events: readonly MonitorEvent[]; onSendDummy: () => void })` — a pure presentational component grouping events by session then phase, with input/output badges and a "더미 세션 송신" button. (Streaming wiring lives in App; the component takes events as props for deterministic testing.)

- [ ] **Step 1: Write the failing test**

```tsx
// demo-ui/tests/simulatorMonitorPanel.test.tsx
import { render, screen, fireEvent } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { SimulatorMonitorPanel } from '../src/components/SimulatorMonitorPanel'
import type { MonitorEvent } from '../src/simContracts'

const EVENTS: MonitorEvent[] = [
  { seq: 1, session_id: 's1', phase: 'start', direction: 'input', timestamp: 't1', payload: { level_id: 'SP0035-L01' } },
  { seq: 2, session_id: 's1', phase: 'start', direction: 'output', timestamp: 't2', payload: { recommended_flow: ['A-1'] } },
  { seq: 3, session_id: 's1', phase: 'runtime', direction: 'input', timestamp: 't3', payload: { event_id: 'e1' } },
]

describe('SimulatorMonitorPanel', () => {
  it('renders phase-grouped input/output rows', () => {
    render(<SimulatorMonitorPanel events={EVENTS} onSendDummy={() => {}} />)
    expect(screen.getByText('세션 시작')).toBeInTheDocument()
    expect(screen.getByText('런타임')).toBeInTheDocument()
    expect(screen.getAllByText('input').length).toBeGreaterThanOrEqual(2)
    expect(screen.getAllByText('output').length).toBeGreaterThanOrEqual(1)
  })

  it('fires onSendDummy when the button is clicked', () => {
    const onSend = vi.fn()
    render(<SimulatorMonitorPanel events={[]} onSendDummy={onSend} />)
    fireEvent.click(screen.getByRole('button', { name: '더미 세션 송신' }))
    expect(onSend).toHaveBeenCalledOnce()
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd demo-ui && npm test -- --run simulatorMonitorPanel`
Expected: FAIL (cannot resolve component).

- [ ] **Step 3: Write minimal implementation**

```tsx
// demo-ui/src/components/SimulatorMonitorPanel.tsx
import { useMemo, type ReactElement } from 'react'
import type { MonitorEvent, Phase } from '../simContracts'
import '../simMonitor.css'

const PHASE_LABEL: Record<Phase, string> = {
  start: '세션 시작',
  runtime: '런타임',
  session_end: '세션 끝',
}
const PHASE_ORDER: readonly Phase[] = ['start', 'runtime', 'session_end']

type Props = {
  readonly events: readonly MonitorEvent[]
  readonly onSendDummy: () => void
}

export function SimulatorMonitorPanel({ events, onSendDummy }: Props): ReactElement {
  const bySession = useMemo(() => {
    const map = new Map<string, MonitorEvent[]>()
    for (const event of events) {
      const list = map.get(event.session_id) ?? []
      list.push(event)
      map.set(event.session_id, list)
    }
    return [...map.entries()]
  }, [events])

  return (
    <section className="sim-monitor" aria-label="시뮬레이터 인터페이스 모니터">
      <div className="sim-monitor__bar">
        <h3 className="sim-monitor__title">시뮬레이터 인터페이스 모니터</h3>
        <button type="button" className="sim-monitor__send" onClick={onSendDummy}>
          더미 세션 송신
        </button>
      </div>

      {bySession.length === 0 && <p className="sim-monitor__empty">아직 수신된 I/O가 없습니다.</p>}

      <div className="sim-monitor__scroll">
        {bySession.map(([sessionId, sessionEvents]) => (
          <div className="sim-monitor__session" key={sessionId}>
            <p className="sim-monitor__session-id">세션 {sessionId}</p>
            {PHASE_ORDER.map((phase) => {
              const phaseEvents = sessionEvents.filter((event) => event.phase === phase)
              if (phaseEvents.length === 0) {
                return null
              }
              return (
                <div className="sim-monitor__phase" data-phase={phase} key={phase}>
                  <p className="sim-monitor__phase-label">{PHASE_LABEL[phase]}</p>
                  <ol className="sim-monitor__rows">
                    {phaseEvents.map((event) => (
                      <li className="sim-monitor__row" data-direction={event.direction} key={event.seq}>
                        <span className="sim-monitor__dir">{event.direction}</span>
                        <code className="sim-monitor__payload">{JSON.stringify(event.payload)}</code>
                      </li>
                    ))}
                  </ol>
                </div>
              )
            })}
          </div>
        ))}
      </div>
    </section>
  )
}
```

Add `demo-ui/src/simMonitor.css` with phase-scoped styling. Keep it self-contained (no dependency on `styles.css` tokens beyond the CSS variables already defined there being optional):

```css
/* demo-ui/src/simMonitor.css */
.sim-monitor {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  padding: 0.75rem 1rem;
  border-top: 1px solid rgb(255 255 255 / 12%);
  font-size: 0.8125rem;
}
.sim-monitor__bar { display: flex; align-items: center; justify-content: space-between; gap: 1rem; }
.sim-monitor__title { margin: 0; font-size: 0.875rem; font-weight: 600; }
.sim-monitor__send { padding: 0.25rem 0.75rem; border-radius: 0.375rem; cursor: pointer; }
.sim-monitor__empty { margin: 0; opacity: 0.6; }
.sim-monitor__scroll { max-height: 14rem; overflow-y: auto; display: flex; flex-direction: column; gap: 0.5rem; }
.sim-monitor__session-id { margin: 0 0 0.25rem; font-weight: 600; opacity: 0.75; }
.sim-monitor__phase { border-left: 3px solid var(--sim-phase-color, #64748b); padding-left: 0.5rem; margin-bottom: 0.375rem; }
.sim-monitor__phase[data-phase='start'] { --sim-phase-color: #38bdf8; }
.sim-monitor__phase[data-phase='runtime'] { --sim-phase-color: #a3e635; }
.sim-monitor__phase[data-phase='session_end'] { --sim-phase-color: #fbbf24; }
.sim-monitor__phase-label { margin: 0 0 0.25rem; font-weight: 600; color: var(--sim-phase-color); }
.sim-monitor__rows { margin: 0; padding: 0; list-style: none; display: flex; flex-direction: column; gap: 0.125rem; }
.sim-monitor__row { display: flex; gap: 0.5rem; align-items: baseline; }
.sim-monitor__dir {
  flex: 0 0 3.25rem;
  text-transform: uppercase;
  font-size: 0.6875rem;
  letter-spacing: 0.04em;
  opacity: 0.8;
}
.sim-monitor__row[data-direction='input'] .sim-monitor__dir { color: #7dd3fc; }
.sim-monitor__row[data-direction='output'] .sim-monitor__dir { color: #fca5a5; }
.sim-monitor__payload {
  flex: 1 1 auto;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-family: ui-monospace, monospace;
  opacity: 0.85;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd demo-ui && npm test -- --run simulatorMonitorPanel`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add demo-ui/src/components/SimulatorMonitorPanel.tsx demo-ui/src/simMonitor.css demo-ui/tests/simulatorMonitorPanel.test.tsx
git commit -m "feat(demo-ui): 시뮬레이터 인터페이스 모니터 패널"
```

---

## Task 11: 구독 훅 + App.tsx 배선

**Files:**
- Create: `demo-ui/src/useMonitorStream.ts` (구독 로직 전량 — 동시 수정 중인 `App.tsx` 델타를 최소화)
- Modify: `demo-ui/src/App.tsx` (import 3줄 + 훅 1줄 + `useCallback` 3줄 + JSX 6줄)
- Modify: `demo-ui/tests/App.test.tsx` (fetch mock에 `/api/monitor/stream` 분기 1개 추가)
- Test: `demo-ui/tests/appSimPanel.test.tsx`

**Interfaces:**
- Consumes: `streamMonitor`, `sendDummySession` (Task 9); `SimulatorMonitorPanel` (Task 10); `MonitorEvent` (Task 8).
- Produces: `useMonitorStream() -> readonly MonitorEvent[]`; App에 렌더된 `<SimulatorMonitorPanel>`(기존 chat/sequence UI 비파괴).

> 기존 `App.test.tsx`의 fetch mock은 등록되지 않은 URL에 `throw`한다. App이 마운트 시 `/api/monitor/stream`을 열므로 분기를 추가하지 않으면 조용히 삼켜진 예외로 테스트 의도가 흐려진다. 빈 스트림을 돌려주는 분기 하나를 명시적으로 추가한다.

- [ ] **Step 1: Write the failing test**

```tsx
// demo-ui/tests/appSimPanel.test.tsx
import { render, screen, within } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { App } from '../src/App'

// The factory is hoisted above imports, so the fixtures live inside it.
vi.mock('../src/simMonitorClient', () => ({
  streamMonitor: async function* () {
    yield {
      seq: 1, session_id: 'qa-1', phase: 'start', direction: 'input',
      timestamp: 't1', payload: { level_id: 'SP0035-L01' },
    }
    yield {
      seq: 2, session_id: 'qa-1', phase: 'start', direction: 'output',
      timestamp: 't2', payload: { flow_source: 'rule' },
    }
  },
  sendDummySession: vi.fn(),
}))

afterEach(() => {
  vi.restoreAllMocks()
})

describe('App simulator monitor panel', () => {
  it('renders streamed simulator I/O next to the existing console', async () => {
    vi.spyOn(globalThis, 'fetch').mockImplementation(
      async () => new Response('{}', { status: 500 }),
    )

    render(<App />)

    const panel = await screen.findByLabelText('시뮬레이터 인터페이스 모니터')
    expect(await within(panel).findByText('세션 시작')).toBeInTheDocument()
    expect(await within(panel).findByText('input')).toBeInTheDocument()
    expect(await within(panel).findByText('output')).toBeInTheDocument()
    // The existing console is untouched.
    expect(screen.getByPlaceholderText('도장 관련 질문을 입력하세요')).toBeInTheDocument()
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd demo-ui && npm test -- --run appSimPanel`
Expected: FAIL (`Unable to find a label with the text of: 시뮬레이터 인터페이스 모니터`).

- [ ] **Step 3: Write minimal implementation**

Create the hook:

```ts
// demo-ui/src/useMonitorStream.ts
import { useEffect, useState } from 'react'
import type { MonitorEvent } from './simContracts'
import { streamMonitor } from './simMonitorClient'

/** Subscribes to the read-only monitor stream for the caller's lifetime. */
export function useMonitorStream(): readonly MonitorEvent[] {
  const [events, setEvents] = useState<readonly MonitorEvent[]>([])

  useEffect(() => {
    const controller = new AbortController()
    void (async () => {
      try {
        for await (const event of streamMonitor(0, controller.signal)) {
          setEvents((previous) => [...previous, event])
        }
      } catch {
        // Observability only: a dropped monitor stream must never break the
        // console, and the panel keeps whatever it already received.
      }
    })()
    return () => controller.abort()
  }, [])

  return events
}
```

In `demo-ui/src/App.tsx`, add three imports (keeping the existing alphabetical grouping — `SimulatorMonitorPanel` after `MonitorPanel`, the other two after `./stageLabels` / `./telemetry`):

```tsx
import { SimulatorMonitorPanel } from './components/SimulatorMonitorPanel'
import { sendDummySession } from './simMonitorClient'
import { useMonitorStream } from './useMonitorStream'
```

Inside the `App` body, right after `const loggedStagesRef = useRef<Set<string>>(new Set())`:

```tsx
  const monitorEvents = useMonitorStream()
```

Next to the existing `useCallback` handlers (after `onToggleLog`):

```tsx
  const onSendDummySession = useCallback(() => {
    void sendDummySession(crypto.randomUUID())
  }, [])
```

In the returned JSX, between the `</div>` that closes `.workspace` and `<ToolLogPanel ... />`:

```tsx
      <SimulatorMonitorPanel events={monitorEvents} onSendDummy={onSendDummySession} />
```

In `demo-ui/tests/App.test.tsx`, add this branch to the `fetch` mock immediately before the final `throw new Error(\`unexpected fetch: ${url}\`)`:

```ts
      if (url.includes('/api/monitor/stream')) {
        return new Response(
          new ReadableStream<Uint8Array>({
            start(controller) {
              controller.close()
            },
          }),
          { status: 200, headers: { 'Content-Type': 'text/event-stream' } },
        )
      }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd demo-ui && npm test -- --run`
Expected: PASS (new `appSimPanel` + every existing suite still green).

- [ ] **Step 5: Commit**

```bash
git add demo-ui/src/useMonitorStream.ts demo-ui/tests/appSimPanel.test.tsx
git commit -m "feat(demo-ui): 모니터 스트림 구독 훅"
```

> `App.tsx`와 `App.test.tsx`에는 이 작업과 무관한 미커밋 변경이 섞여 있다. 두 파일은 hunk 단위로 본 작업 델타만 스테이징할 수 있을 때만 커밋하고, 불가능하면 커밋하지 말고 미커밋으로 남긴 뒤 최종 보고에 명시한다. 남의 hunk를 절대 함께 커밋하지 않는다.

---

## Task 12: 최종 통합 검증 (E2E 더미 왕복)

**Files:** (검증 전용, 신규 파일 없음)

- [ ] **Step 1: 전체 백엔드 회귀**

Run: `docker exec -w /workspace ship-paint-trainer python3 -m pytest trainer/demo_api/tests trainer/scripts/tests -q`
Expected: PASS (기존 29건 + 신규 전부 green).

- [ ] **Step 2: 전체 프론트 회귀 + 타입 빌드**

Run: `cd demo-ui && npm test -- --run && npm run build:web`
Expected: PASS + `tsc -b`(테스트 파일 포함, `strict`/`noUncheckedIndexedAccess`/`exactOptionalPropertyTypes`) 통과 후 빌드 성공.

- [ ] **Step 3: 실 왕복 (실제 uvicorn + 더미 드라이버 + 모니터 SSE/스냅샷)**

```bash
# 데모 API 기동 (demo 모드, GPU 불필요)
docker exec -d -w /workspace ship-paint-trainer python3 -m trainer.demo_api.app --host 127.0.0.1 --port 8121
sleep 4

# 3단계 더미 세션 송신
docker exec -w /workspace ship-paint-trainer python3 trainer/scripts/sim_dummy_driver.py \
  --api-url http://127.0.0.1:8121 --session-id qa-sim --events 3 --anomaly

# 모니터 스냅샷: (phase, direction) 순서 확인
docker exec -w /workspace ship-paint-trainer python3 -c "
import json, urllib.request
snap = json.load(urllib.request.urlopen('http://127.0.0.1:8121/api/monitor/events?cursor=0&limit=50'))
print(len(snap['events']), [(e['phase'], e['direction']) for e in snap['events']])
"

# 실 HTTP SSE 경로 (TestClient로 못 덮는 부분): replay 프레임 확인 후 종료
docker exec -w /workspace ship-paint-trainer timeout 5 curl -sN \
  'http://127.0.0.1:8121/api/monitor/stream?cursor=0' | head -8
```

Expected:
- start/events/complete 응답 각 200.
- 스냅샷 이벤트 **정확히 10건**, 순서는
  `(start,input)(start,output)(runtime,input)(runtime,output)×3(session_end,input)(session_end,output)`.
- 첫 runtime output의 `kind`가 `anomaly`, 마지막 session_end output의 `risk_flags`에 `under_thickness` 포함.
- `curl -sN`이 `event: monitor` 프레임을 replay로 흘려보낸다(gap 없음 — 10 < maxlen 500).

정리: `docker exec ship-paint-trainer pkill -f "trainer.demo_api.app --host 127.0.0.1 --port 8121"`

- [ ] **Step 4: Commit (없음 — 검증만)**

검증 통과 시 별도 커밋 없음. 실패 시 해당 Task로 복귀.

---

## Self-Review

**1. Spec coverage:**
- §4 계약 → Task 1. §5 결정론 출력 → Task 3. §6 모니터(레코더+SSE) → Task 2, 5. §7 더미 드라이버 → Task 7(CLI), Task 9(UI 송신). §8 패널 → Task 10, 11. §9 에러 처리 → Task 4(`_run_phase` invalid_json / session_id_mismatch / validation_error 모두 422 + output 기록). §10 테스트 → 각 Task의 test + Task 12. §11 파일 계획 → Task 1~11 파일 일치(추가: `useMonitorStream.ts`, `simMonitor.css` — 공유 파일 델타 최소화 목적). §12 경계 → Task 6 test가 기존 라우트 존재를 고정. §13 확장점 → 미구현. 빌더 seam은 `_run_phase(build_output=...)` 콜러블로 이미 분리되어 있어 후속 `?live=1` 라우팅을 붙일 자리가 남는다.
- 스펙 §6.3의 `/api/monitor/events`는 백엔드 테스트와 Task 12 curl 검증으로 덮인다(프론트는 `/stream` replay로 충분해 호출하지 않음).

**2. Placeholder scan:** TBD/TODO 없음. 모든 step에 실제 코드·명령·기대 결과 포함.

**3. Type consistency:** `Phase`/`Direction` 리터럴, `MonitorEvent`(seq/session_id/phase/direction/timestamp/payload), 레코더 `record/snapshot/subscribe/unsubscribe`, 빌더 `build_start_output/build_event_output/build_review_output`, `monitor_sse_body`/`create_simulator_router`/`create_monitor_router` 시그니처가 Task 전반에 일관. 임계 상수 `STANDOFF_LIMIT_MM=320.0`/`ANGLE_MIN_DEG=60.0`는 stub·driver·frontend 더미에서 동일 값 사용. SSE 프레임(`event: monitor`/`gap`, `data: <json>`)이 `monitor_api._frame`과 `simMonitorClient.streamMonitor`(`frame.event === 'monitor'`)에서 정합. `api.ts`의 `parseFrame`은 top-level에 `data` 키가 있을 때만 언랩하는데 `MonitorEvent`에는 `data` 키가 없으므로 봉투 전체가 그대로 파싱된다.

**4. 실측으로 고친 항목 (초안 대비):**
1. Task 5 SSE 테스트가 `TestClient`에서 영구 블로킹 → `monitor_sse_body` 순수 generator 분리 + `anyio` 직접 테스트.
2. Task 7 드라이버 경로 `parents[2]`가 `/trainer/trainer/scripts`를 가리킴 → importlib 제거하고 `trainer.scripts.sim_dummy_driver` 직접 임포트(네임스페이스 패키지 동작 확인), 테스트 위치도 `trainer/scripts/tests/`로 이동.
3. 호스트 `python3`에 의존성 없음 → 모든 파이썬 명령에 `docker exec -w /workspace ship-paint-trainer` 접두사 명시.
4. `exc.errors()`가 직렬화 불가 객체를 담을 수 있음 → `include_url=False, include_context=False`.
5. `send_nowait`의 `ClosedResourceError` 미처리 → 예외 목록에 추가.
6. `fetchMonitorSnapshot`이 어디서도 안 쓰임(YAGNI) → 삭제하고 `API_BASE_URL`은 `api.ts`에서 export해 재사용.
7. `vi.restoreAllMocks()`는 `stubGlobal`을 되돌리지 않음 → `vi.unstubAllGlobals()`.
8. 동시 수정 중인 `styles.css`/`App.tsx` → 전용 `simMonitor.css`와 `useMonitorStream.ts`로 공유 파일 델타 최소화.
9. 기존 `App.test.tsx` fetch mock이 미등록 URL에 throw → `/api/monitor/stream` 분기 추가.
10. Task 12 기대 이벤트 수 "8건 이상" → `--events 3` 기준 **정확히 10건**으로 확정.
