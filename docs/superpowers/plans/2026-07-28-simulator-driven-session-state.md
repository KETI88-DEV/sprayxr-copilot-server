# 시뮬레이터 주도 세션 상태 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 시뮬레이터가 보낸 값이 채팅 `environment`와 좌측 도장 순서 패널에 실제로 반영되게 하고, 더미 입력을 세션 시작/런타임/세션 끝으로 나눠 보낼 수 있게 한다.

**Architecture:** 서버에 단일 프로세스 in-memory `SimSessionStore`를 두고 `/api/sim/*`가 이를 갱신한다. 세 소비자(`GET /api/sequence/{id}`, 신규 `GET /api/sim/session`, live runner의 `environment`)가 모두 store에서 읽는다. 활성 세션이 없으면 기존 fixture로 폴백해 현재 동작을 보존한다. 프런트는 전이 규칙을 복제하지 않고 모니터 이벤트를 신호 삼아 두 엔드포인트를 다시 조회한다.

**Tech Stack:** Python 3.11 / FastAPI 0.135 / Pydantic v2(`extra="forbid"`) / anyio / httpx · React 19 / Zod 4 / Vite 8 / Vitest 4 / Playwright

**Spec:** `docs/superpowers/specs/2026-07-28-simulator-driven-session-state-design.md`

## Global Constraints

- 모든 Python 명령은 컨테이너에서 저장소 루트 기준으로 실행한다: `docker exec -w /workspace ship-paint-trainer python3 ...`. 호스트 `python3`(3.12)에는 의존성이 없다.
- 프런트 명령은 호스트의 `demo-ui/`에서 실행한다.
- 경계의 JSON/SSE/tool 봉투를 보존한다. `AgentPipeline.run()` 동작과 시그니처는 건드리지 않는다.
- Pydantic 모델은 `StrictModel`(`extra="forbid"`)을 상속하고, Zod는 `.strict()`를 쓴다. `contracts.py` ↔ `contracts.ts`, `sim_contracts.py` ↔ `simContracts.ts`는 같은 커밋에서 함께 바꾼다.
- `demo-ui/tsconfig.json`은 `strict` + `noUncheckedIndexedAccess` + `exactOptionalPropertyTypes`이고 `src`와 `tests`를 모두 타입 검사한다. 배열 인덱싱 결과는 `undefined` 가능성을 좁혀야 한다.
- 비종료 SSE 라우트를 Starlette `TestClient`로 구동하지 않는다.
- 커밋은 Conventional Commits + 한국어 요약: `<type>(<scope>): <요약>`. 워크트리는 공유되므로 **명시 경로만 스테이징**하고 `git add -A`를 쓰지 않는다.
- 기준선: Python 176 passed, Vitest 79 passed / 15 files. 작업 종료 시 이 수치 이상이어야 하고 기존 테스트는 수정 없이 통과해야 한다.

## File Structure

| 파일 | 책임 |
|---|---|
| `trainer/demo_api/sim_session.py` (신규) | `SimSessionState` + `SimSessionStore`. 순수 동기 전이 규칙, 외부 의존 없음 |
| `trainer/demo_api/zone_resolver.py` (신규) | `ZoneResolver` Protocol, `McpZoneResolver`(httpx), `NullZoneResolver` |
| `trainer/demo_api/sim_contracts.py` | `zone_id` 필드 추가, `SimSessionView`/`SimSessionResponse` 추가 |
| `trainer/demo_api/sequence.py` | `session_snapshot`, `session_tool_environment` 추가 |
| `trainer/demo_api/simulator.py` | store·resolver 주입, `on_success` 훅, `GET /session` |
| `trainer/demo_api/app.py` | store 생성·배선, `/api/sequence`가 store 우선 |
| `trainer/demo_api/chat_service.py` | live runner `environment`를 store에서 생성 |
| `demo-ui/src/simContracts.ts` | `simSessionSchema`, `simSessionResponseSchema` |
| `demo-ui/src/simMonitorClient.ts` | 3단계 송신기 + `fetchSimSession` |
| `demo-ui/src/components/SimulatorMonitorPanel.tsx` | 3버튼 + 비활성 규칙 |
| `demo-ui/src/components/SequencePanel.tsx` | 세션 단계·ID·미매핑 구역 배지 |
| `demo-ui/src/App.tsx` | 세션 생명주기 상태, 모니터 이벤트 시 재조회 |
| `demo-ui/src/styles.css`, `src/simMonitor.css` | 폭 붕괴 수정, 버튼 스타일 |

---

## Task 1: 셸 레이아웃 폭 붕괴 수정

다른 작업과 독립이다. 현재 뷰포트 1280px에서 `document.scrollWidth`가 3020px이라 사용자 말풍선(`justify-self: end`)이 화면 밖 right=2988에 놓이고 컴포저 `초기화` 버튼도 잘린다.

**Files:**
- Modify: `demo-ui/src/styles.css:193-198`
- Modify: `demo-ui/src/simMonitor.css:1-8`, `:42-48`, `:56-60`, `:80-93`
- Test: `demo-ui/e2e/layout-width.spec.ts` (신규)

**Interfaces:**
- Consumes: 없음
- Produces: 없음 (CSS 전용)

- [ ] **Step 1: 실패하는 e2e 회귀 테스트 작성**

`demo-ui/e2e/layout-width.spec.ts`:

```ts
import { expect, test } from '@playwright/test'

async function horizontalOverflow(page: import('@playwright/test').Page): Promise<number> {
  return page.evaluate(() => {
    const doc = document.documentElement
    return doc.scrollWidth - doc.clientWidth
  })
}

test('셸은 긴 모니터 페이로드에도 가로로 넘치지 않는다', async ({ page }) => {
  await page.goto('/', { waitUntil: 'domcontentloaded' })

  // 모니터에 긴 JSON 한 줄을 만들어 넘침 조건을 재현한다.
  await page.getByRole('button', { name: '세션 시작' }).click()
  await expect(page.locator('.sim-monitor__row')).not.toHaveCount(0)

  expect(await horizontalOverflow(page)).toBe(0)
})
```

- [ ] **Step 2: 실패 확인**

Run: `cd demo-ui && npx playwright test e2e/layout-width.spec.ts`
Expected: FAIL. 버튼 이름 `세션 시작`이 아직 없으면 timeout, 있으면 overflow가 0이 아님.

> Task 11 이전에는 `세션 시작` 버튼이 없다. 이 단계에서는 아래 임시 본문으로 넘침만 먼저 고정하고, Task 11 완료 후 위 최종 본문으로 되돌린다.
>
> ```ts
> test('셸은 긴 모니터 페이로드에도 가로로 넘치지 않는다', async ({ page }) => {
>   await page.goto('/', { waitUntil: 'domcontentloaded' })
>   await page.evaluate(() => {
>     const scroll = document.querySelector('.sim-monitor__scroll') ?? document.querySelector('.sim-monitor')
>     if (scroll === null) throw new Error('sim-monitor not rendered')
>     const row = document.createElement('div')
>     row.className = 'sim-monitor__row'
>     row.innerHTML = `<span class="sim-monitor__dir">INPUT</span><span class="sim-monitor__payload">${'{"k":"'.padEnd(4000, 'x')}"}</span>`
>     scroll.appendChild(row)
>   })
>   expect(await horizontalOverflow(page)).toBe(0)
> })
> ```

- [ ] **Step 3: CSS 수정**

`demo-ui/src/styles.css`의 `.app-shell`에 열 정의를 추가한다.

```css
.app-shell {
  height: 100%;
  display: grid;
  grid-template-rows: auto minmax(0, 1fr) auto;
  grid-template-columns: minmax(0, 1fr);
  background: var(--hull-900);
}
```

`demo-ui/src/simMonitor.css`의 flex 체인에 `min-width: 0`을 넣는다.

```css
.sim-monitor {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  min-width: 0;
  padding: 0.75rem 1rem;
  border-top: 1px solid rgb(255 255 255 / 12%);
  font-size: 0.8125rem;
}

.sim-monitor__scroll {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  min-width: 0;
  max-height: 14rem;
  overflow-y: auto;
}

.sim-monitor__phase {
  min-width: 0;
  margin-bottom: 0.375rem;
  padding-left: 0.5rem;
  border-left: 3px solid var(--sim-phase-color, #64748b);
}

.sim-monitor__rows {
  display: flex;
  flex-direction: column;
  gap: 0.125rem;
  min-width: 0;
  margin: 0;
  padding: 0;
  list-style: none;
}

.sim-monitor__row {
  display: flex;
  gap: 0.5rem;
  min-width: 0;
  align-items: baseline;
}
```

- [ ] **Step 4: 통과 확인**

Run: `cd demo-ui && npx playwright test e2e/layout-width.spec.ts`
Expected: PASS, overflow가 0.

- [ ] **Step 5: 회귀 확인 후 커밋**

```bash
cd demo-ui && npm test && npm run build:web
git add demo-ui/src/styles.css demo-ui/src/simMonitor.css demo-ui/e2e/layout-width.spec.ts
git commit -m "fix(demo-ui): 모니터 페이로드가 셸 폭을 밀어내던 레이아웃 붕괴 수정"
```

---

## Task 2: `zone_id` 계약 확장

**Files:**
- Modify: `trainer/demo_api/sim_contracts.py:44-47`
- Modify: `demo-ui/src/simContracts.ts`
- Test: `trainer/demo_api/tests/test_sim_contracts.py`, `demo-ui/tests/simContracts.test.ts`

**Interfaces:**
- Produces: `SessionEventInput.zone_id: str | None`(기본 `None`), `SimSessionView`, `SimSessionResponse`, Zod `simSessionSchema` / `simSessionResponseSchema`

- [ ] **Step 1: 실패하는 Python 테스트 작성**

`trainer/demo_api/tests/test_sim_contracts.py` 끝에 추가:

```python
def test_session_event_accepts_optional_zone_id() -> None:
    payload = SessionEventInput.model_validate({"event_id": "ev-0", "zone_id": "A-2"})
    assert payload.zone_id == "A-2"


def test_session_event_zone_id_defaults_to_none() -> None:
    payload = SessionEventInput.model_validate({"event_id": "ev-0"})
    assert payload.zone_id is None


def test_sim_session_response_allows_null_session() -> None:
    response = SimSessionResponse.model_validate({"session": None})
    assert response.session is None
```

파일 상단 import에 `SimSessionResponse`를 추가한다.

- [ ] **Step 2: 실패 확인**

Run: `docker exec -w /workspace ship-paint-trainer python3 -m pytest trainer/demo_api/tests/test_sim_contracts.py -q`
Expected: FAIL — `ImportError: cannot import name 'SimSessionResponse'`

- [ ] **Step 3: Pydantic 계약 구현**

`trainer/demo_api/sim_contracts.py`의 `SessionEventInput`에 필드를 더하고 뷰 모델을 추가한다.

```python
class SessionEventInput(StrictModel):
    event_id: Annotated[str, Field(min_length=1)]
    user_request: str | None = None
    telemetry: EventTelemetry | None = None
    zone_id: str | None = None


class SimSessionView(StrictModel):
    session_id: str
    level_id: str
    phase: Phase
    recommended_flow: list[str]
    current_zone_id: str
    completed_zone_ids: list[str]
    last_event_id: str | None
    anomaly_count: int
    unresolved_zone_id: str | None
    updated_at: str


class SimSessionResponse(StrictModel):
    session: SimSessionView | None
```

- [ ] **Step 4: 통과 확인**

Run: `docker exec -w /workspace ship-paint-trainer python3 -m pytest trainer/demo_api/tests/test_sim_contracts.py -q`
Expected: PASS

- [ ] **Step 5: 실패하는 Zod 테스트 작성**

`demo-ui/tests/simContracts.test.ts` 끝에 추가:

```ts
import { simSessionResponseSchema, simSessionSchema } from '../src/simContracts'

describe('simSessionSchema', () => {
  const valid = {
    session_id: 's-1',
    level_id: 'SP0035-L01',
    phase: 'runtime',
    recommended_flow: ['A-1', 'A-2', 'A-3'],
    current_zone_id: 'A-2',
    completed_zone_ids: ['A-1'],
    last_event_id: 'ev-1',
    anomaly_count: 1,
    unresolved_zone_id: null,
    updated_at: '2026-07-28T00:00:00+00:00',
  }

  it('백엔드 SimSessionView 봉투를 그대로 받는다', () => {
    expect(simSessionSchema.parse(valid).current_zone_id).toBe('A-2')
  })

  it('알 수 없는 키를 거부한다', () => {
    expect(() => simSessionSchema.parse({ ...valid, extra: 1 })).toThrow()
  })

  it('세션 없음을 null로 표현한다', () => {
    expect(simSessionResponseSchema.parse({ session: null }).session).toBeNull()
  })
})
```

- [ ] **Step 6: 실패 확인**

Run: `cd demo-ui && npx vitest run tests/simContracts.test.ts`
Expected: FAIL — `simSessionSchema` export 없음

- [ ] **Step 7: Zod 미러 구현**

`demo-ui/src/simContracts.ts` 끝에 추가:

```ts
export const simSessionSchema = z
  .object({
    session_id: z.string(),
    level_id: z.string(),
    phase: phaseSchema,
    recommended_flow: z.array(z.string()),
    current_zone_id: z.string(),
    completed_zone_ids: z.array(z.string()),
    last_event_id: z.string().nullable(),
    anomaly_count: z.number().int().nonnegative(),
    unresolved_zone_id: z.string().nullable(),
    updated_at: z.string(),
  })
  .strict()
export type SimSession = z.infer<typeof simSessionSchema>

export const simSessionResponseSchema = z
  .object({ session: simSessionSchema.nullable() })
  .strict()
export type SimSessionResponse = z.infer<typeof simSessionResponseSchema>
```

- [ ] **Step 8: 통과 확인 후 커밋**

```bash
cd demo-ui && npx vitest run tests/simContracts.test.ts && npm run build:web
docker exec -w /workspace ship-paint-trainer python3 -m pytest trainer/demo_api/tests/test_sim_contracts.py -q
git add trainer/demo_api/sim_contracts.py trainer/demo_api/tests/test_sim_contracts.py demo-ui/src/simContracts.ts demo-ui/tests/simContracts.test.ts
git commit -m "feat(demo-api): 시뮬레이터 런타임 zone_id와 세션 조회 계약 추가"
```

---

## Task 3: `SimSessionStore` 전이 규칙

**Files:**
- Create: `trainer/demo_api/sim_session.py`
- Test: `trainer/demo_api/tests/test_sim_session.py` (신규)

**Interfaces:**
- Consumes: `trainer.demo_api.sim_contracts.Phase`
- Produces:
  - `SimSessionState` — frozen dataclass. 필드: `session_id: str`, `level_id: str`, `phase: Phase`, `recommended_flow: tuple[str, ...]`, `current_zone_id: str`, `completed_zone_ids: tuple[str, ...]`, `last_event_id: str | None`, `anomaly_count: int`, `unresolved_zone_id: str | None`, `updated_at: str`
  - `SimSessionStore.start(*, session_id: str, level_id: str, recommended_flow: Sequence[str]) -> SimSessionState`
  - `SimSessionStore.runtime(*, session_id: str, event_id: str, target_zone_id: str | None, anomaly: bool) -> SimSessionState | None`
  - `SimSessionStore.end(*, session_id: str) -> SimSessionState | None`
  - `SimSessionStore.current() -> SimSessionState | None`

- [ ] **Step 1: 실패하는 테스트 작성**

`trainer/demo_api/tests/test_sim_session.py`:

```python
from __future__ import annotations

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


def test_runtime_for_unknown_session_is_ignored() -> None:
    store = _started()
    assert store.runtime(session_id="other", event_id="ev-0", target_zone_id="A-2", anomaly=False) is None
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


def test_new_start_replaces_previous_session() -> None:
    store = _started()
    _ = store.runtime(session_id="s-1", event_id="ev-0", target_zone_id="A-3", anomaly=False)
    _ = store.start(session_id="s-2", level_id="SP0035-L01", recommended_flow=FLOW)
    state = store.current()
    assert state is not None
    assert state.session_id == "s-2"
    assert state.current_zone_id == "A-1"


def test_current_is_none_before_any_start() -> None:
    assert SimSessionStore().current() is None
```

- [ ] **Step 2: 실패 확인**

Run: `docker exec -w /workspace ship-paint-trainer python3 -m pytest trainer/demo_api/tests/test_sim_session.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'trainer.demo_api.sim_session'`

- [ ] **Step 3: 구현**

`trainer/demo_api/sim_session.py`:

```python
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

    One slot on purpose: the chat ``environment`` must be unambiguous about
    which session it describes, and the demo has one operator. A new ``start``
    replaces the previous session; ``session_end`` keeps the final state
    readable so the panel does not blank out.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: SimSessionState | None = None

    def start(
        self, *, session_id: str, level_id: str, recommended_flow: Sequence[str]
    ) -> SimSessionState:
        flow = tuple(recommended_flow)
        state = SimSessionState(
            session_id=session_id,
            level_id=level_id,
            phase="start",
            recommended_flow=flow,
            current_zone_id=flow[0] if flow else "",
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
```

- [ ] **Step 4: 통과 확인**

Run: `docker exec -w /workspace ship-paint-trainer python3 -m pytest trainer/demo_api/tests/test_sim_session.py -q`
Expected: PASS (9 passed)

- [ ] **Step 5: 커밋**

```bash
git add trainer/demo_api/sim_session.py trainer/demo_api/tests/test_sim_session.py
git commit -m "feat(demo-api): 시뮬레이터 세션 상태 store와 전이 규칙"
```

---

## Task 4: `ZoneResolver` (location_tracking 폴백)

**Files:**
- Create: `trainer/demo_api/zone_resolver.py`
- Test: `trainer/demo_api/tests/test_zone_resolver.py` (신규)

**Interfaces:**
- Produces:
  - `ZoneResolver` Protocol — `async def resolve(self, coordinates: list[float]) -> str | None`
  - `NullZoneResolver` — 항상 `None`
  - `McpZoneResolver(base_url: str, timeout_sec: float = 2.0)`

- [ ] **Step 1: 실패하는 테스트 작성**

`trainer/demo_api/tests/test_zone_resolver.py`:

```python
from __future__ import annotations

from typing import Any

import anyio
import httpx

from trainer.demo_api.zone_resolver import McpZoneResolver, NullZoneResolver


def _run(coro: Any) -> Any:
    return anyio.run(lambda: coro)


def test_null_resolver_returns_none() -> None:
    async def scenario() -> str | None:
        return await NullZoneResolver().resolve([1.0, 2.0, 3.0])

    assert _run(scenario()) is None


def test_mcp_resolver_returns_zone_id_from_success_envelope() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/location_tracking/identify"
        return httpx.Response(200, json={"status": "success", "result": {"zone_id": "B-3"}})

    async def scenario() -> str | None:
        transport = httpx.MockTransport(handler)
        resolver = McpZoneResolver("http://mcp.test", transport=transport)
        return await resolver.resolve([1.0, 2.0, 3.0])

    assert _run(scenario()) == "B-3"


def test_mcp_resolver_returns_none_on_error_status() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"status": "error"})

    async def scenario() -> str | None:
        transport = httpx.MockTransport(handler)
        resolver = McpZoneResolver("http://mcp.test", transport=transport)
        return await resolver.resolve([1.0, 2.0, 3.0])

    assert _run(scenario()) is None


def test_mcp_resolver_returns_none_when_transport_raises() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("unreachable")

    async def scenario() -> str | None:
        transport = httpx.MockTransport(handler)
        resolver = McpZoneResolver("http://mcp.test", transport=transport)
        return await resolver.resolve([1.0, 2.0, 3.0])

    assert _run(scenario()) is None


def test_mcp_resolver_ignores_coordinates_that_are_not_triplets() -> None:
    async def scenario() -> str | None:
        resolver = McpZoneResolver("http://mcp.test")
        return await resolver.resolve([1.0, 2.0])

    assert _run(scenario()) is None
```

- [ ] **Step 2: 실패 확인**

Run: `docker exec -w /workspace ship-paint-trainer python3 -m pytest trainer/demo_api/tests/test_zone_resolver.py -q`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 구현**

`trainer/demo_api/zone_resolver.py`:

```python
from __future__ import annotations

import logging
from typing import Protocol

import httpx

logger = logging.getLogger(__name__)


class ZoneResolver(Protocol):
    async def resolve(self, coordinates: list[float]) -> str | None: ...


class NullZoneResolver:
    """Used when no MCP endpoint is configured, and in tests."""

    async def resolve(self, coordinates: list[float]) -> str | None:
        _ = coordinates
        return None


class McpZoneResolver:
    """Resolves a zone through the MCP location_tracking tool.

    Every failure returns None instead of raising: this is an observability
    fallback, and a missing or slow MCP must never break the simulator
    interface contract.
    """

    def __init__(
        self,
        base_url: str,
        timeout_sec: float = 2.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_sec = timeout_sec
        self._transport = transport

    async def resolve(self, coordinates: list[float]) -> str | None:
        if len(coordinates) < 3:
            return None

        x, y, z = coordinates[0], coordinates[1], coordinates[2]
        payload = {"coordinates": [{"x": x, "y": y, "z": z}]}

        try:
            async with httpx.AsyncClient(
                timeout=self._timeout_sec, transport=self._transport
            ) as client:
                response = await client.post(
                    f"{self._base_url}/location_tracking/identify", json=payload
                )
                if response.status_code != 200:
                    return None
                body: object = response.json()
        except Exception:  # noqa: BROAD_EXCEPT_OK
            logger.debug("zone resolver: location_tracking unavailable", exc_info=True)
            return None

        if not isinstance(body, dict):
            return None
        result = body.get("result")
        if not isinstance(result, dict):
            return None
        zone_id = result.get("zone_id")
        return zone_id if isinstance(zone_id, str) and zone_id else None
```

- [ ] **Step 4: 통과 확인**

Run: `docker exec -w /workspace ship-paint-trainer python3 -m pytest trainer/demo_api/tests/test_zone_resolver.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: 커밋**

```bash
git add trainer/demo_api/zone_resolver.py trainer/demo_api/tests/test_zone_resolver.py
git commit -m "feat(demo-api): 좌표 기반 구역 해석기와 MCP 폴백"
```

---

## Task 5: 세션 파생 스냅샷과 tool environment

**Files:**
- Modify: `trainer/demo_api/sequence.py`
- Test: `trainer/demo_api/tests/test_sequence.py`

**Interfaces:**
- Consumes: `SimSessionState`(Task 3)
- Produces:
  - `session_snapshot(state: SimSessionState) -> SequenceSnapshot`
  - `session_tool_environment(state: SimSessionState) -> SequenceToolEnvironment`

- [ ] **Step 1: 실패하는 테스트 작성**

`trainer/demo_api/tests/test_sequence.py` 끝에 추가:

```python
from trainer.demo_api.sequence import session_snapshot, session_tool_environment
from trainer.demo_api.sim_session import SimSessionStore


def _state_at(zone_id: str):
    store = SimSessionStore()
    _ = store.start(
        session_id="s-1", level_id="SP0035-L01", recommended_flow=["A-1", "A-2", "A-3"]
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
    assert snapshot.label == "SP0035 Level 01"


def test_session_tool_environment_marks_painted_zones() -> None:
    environment = session_tool_environment(_state_at("A-3"))
    painted = {zone["zone_id"]: zone["is_painted"] for zone in environment["zone_map"]}
    assert painted == {"A-1": True, "A-2": True, "A-3": False}
    assert environment["current_zone_id"] == "A-3"
    assert environment["structure_type"] == "block"
```

- [ ] **Step 2: 실패 확인**

Run: `docker exec -w /workspace ship-paint-trainer python3 -m pytest trainer/demo_api/tests/test_sequence.py -q`
Expected: FAIL — `ImportError: cannot import name 'session_snapshot'`

- [ ] **Step 3: 구현**

`trainer/demo_api/sequence.py`의 `__all__`을 갱신하고 함수를 추가한다. 순환 import를 피하려고 `SimSessionState`는 `TYPE_CHECKING`에서만 가져온다.

```python
from typing import TYPE_CHECKING, Final, TypedDict

if TYPE_CHECKING:
    from trainer.demo_api.sim_session import SimSessionState

__all__ = [
    "default_snapshot",
    "default_tool_environment",
    "sequence_update_from_payload",
    "session_snapshot",
    "session_tool_environment",
]

_UNKNOWN_ZONE_METADATA: Final[tuple[str, str, str]] = ("BLK-A", "미상", "unknown")


def session_snapshot(state: SimSessionState) -> _SequenceSnapshot:
    base = default_snapshot(state.level_id)
    labels = {zone.zone_id: zone.label for zone in base.zones}

    zones: list[_SequenceZone] = []
    for zone_id in state.recommended_flow:
        if zone_id in state.completed_zone_ids:
            status = "completed"
        elif zone_id == state.current_zone_id:
            status = "current"
        else:
            status = "pending"
        zones.append(
            _SequenceZone(zone_id=zone_id, label=labels.get(zone_id, zone_id), status=status)
        )

    if state.current_zone_id in state.recommended_flow:
        index = state.recommended_flow.index(state.current_zone_id)
        next_zone_ids = list(state.recommended_flow[index + 1 :])
    else:
        next_zone_ids = []

    return _SequenceSnapshot(
        level_id=state.level_id,
        label=base.label,
        zones=zones,
        current_zone_id=state.current_zone_id,
        next_zone_ids=next_zone_ids,
        completed_zone_ids=list(state.completed_zone_ids),
    )


def session_tool_environment(state: SimSessionState) -> SequenceToolEnvironment:
    zone_map: list[SequenceToolZone] = []
    for zone_id in state.recommended_flow:
        block_id, zone_type, position = _TOOL_ZONE_METADATA.get(zone_id, _UNKNOWN_ZONE_METADATA)
        zone_map.append(
            {
                "zone_id": zone_id,
                "block_id": block_id,
                "zone_type": zone_type,
                "position": position,
                "is_painted": zone_id in state.completed_zone_ids,
            }
        )

    return {
        "zone_map": zone_map,
        "current_zone_id": state.current_zone_id,
        "structure_type": "block",
    }
```

- [ ] **Step 4: 통과 확인**

Run: `docker exec -w /workspace ship-paint-trainer python3 -m pytest trainer/demo_api/tests/test_sequence.py -q`
Expected: PASS. 기존 `default_snapshot` 테스트도 그대로 통과해야 한다.

- [ ] **Step 5: 커밋**

```bash
git add trainer/demo_api/sequence.py trainer/demo_api/tests/test_sequence.py
git commit -m "feat(demo-api): 세션 상태에서 도장 순서 스냅샷·tool environment 파생"
```

---

## Task 6: 시뮬레이터 라우터에 store·resolver 배선

**Files:**
- Modify: `trainer/demo_api/simulator.py`
- Test: `trainer/demo_api/tests/test_simulator_interface.py`

**Interfaces:**
- Consumes: `SimSessionStore`(Task 3), `ZoneResolver`(Task 4), `SessionEventInput.zone_id`(Task 2)
- Produces: `create_simulator_router(recorder: MonitorRecorder, store: SimSessionStore, resolver: ZoneResolver) -> APIRouter`

- [ ] **Step 1: 실패하는 테스트 작성**

`trainer/demo_api/tests/test_simulator_interface.py` 끝에 추가. 기존 테스트의 `create_simulator_router(recorder)` 호출부도 새 시그니처로 함께 갱신한다.

```python
from trainer.demo_api.sim_session import SimSessionStore
from trainer.demo_api.zone_resolver import NullZoneResolver

START_BODY = {
    "session_id": "s-1",
    "level_id": "SP0035-L01",
    "worker": {"trainee_id": "t-1", "skill_level": "beginner", "handedness": "right"},
    "spec": {"nominal_um": 100.0},
}


class _FixedResolver:
    def __init__(self, zone_id: str | None) -> None:
        self._zone_id = zone_id

    async def resolve(self, coordinates: list[float]) -> str | None:
        _ = coordinates
        return self._zone_id


def _client(store: SimSessionStore, resolver: object = NullZoneResolver()) -> TestClient:
    app = FastAPI()
    app.include_router(create_simulator_router(MonitorRecorder(), store, resolver))
    return TestClient(app)


def test_start_populates_store_with_flow() -> None:
    store = SimSessionStore()
    response = _client(store).post("/api/sim/sessions/start", json=START_BODY)
    assert response.status_code == 200
    state = store.current()
    assert state is not None
    assert state.recommended_flow == ("A-1", "A-2", "A-3")
    assert state.current_zone_id == "A-1"


def test_runtime_zone_id_moves_current_position() -> None:
    store = SimSessionStore()
    client = _client(store)
    _ = client.post("/api/sim/sessions/start", json=START_BODY)
    response = client.post(
        "/api/sim/sessions/s-1/events", json={"event_id": "ev-0", "zone_id": "A-2"}
    )
    assert response.status_code == 200
    state = store.current()
    assert state is not None
    assert state.current_zone_id == "A-2"
    assert state.completed_zone_ids == ("A-1",)


def test_runtime_falls_back_to_resolver_when_zone_id_absent() -> None:
    store = SimSessionStore()
    client = _client(store, _FixedResolver("A-3"))
    _ = client.post("/api/sim/sessions/start", json=START_BODY)
    _ = client.post(
        "/api/sim/sessions/s-1/events",
        json={"event_id": "ev-0", "telemetry": {"coordinates": [1.0, 2.0, 3.0]}},
    )
    state = store.current()
    assert state is not None
    assert state.current_zone_id == "A-3"


def test_rejected_payload_leaves_store_untouched() -> None:
    store = SimSessionStore()
    client = _client(store)
    _ = client.post("/api/sim/sessions/start", json=START_BODY)
    response = client.post("/api/sim/sessions/s-1/events", json={"zone_id": "A-2"})
    assert response.status_code == 422
    state = store.current()
    assert state is not None
    assert state.phase == "start"
    assert state.current_zone_id == "A-1"


def test_session_endpoint_returns_null_before_any_start() -> None:
    response = _client(SimSessionStore()).get("/api/sim/session")
    assert response.status_code == 200
    assert response.json() == {"session": None}


def test_session_endpoint_exposes_current_state() -> None:
    store = SimSessionStore()
    client = _client(store)
    _ = client.post("/api/sim/sessions/start", json=START_BODY)
    body = client.get("/api/sim/session").json()
    assert body["session"]["session_id"] == "s-1"
    assert body["session"]["phase"] == "start"
    assert body["session"]["recommended_flow"] == ["A-1", "A-2", "A-3"]
```

- [ ] **Step 2: 실패 확인**

Run: `docker exec -w /workspace ship-paint-trainer python3 -m pytest trainer/demo_api/tests/test_simulator_interface.py -q`
Expected: FAIL — `create_simulator_router()` 인자 개수 불일치

- [ ] **Step 3: 라우터 구현**

`trainer/demo_api/simulator.py`를 다음과 같이 바꾼다. `_run_phase`는 성공 시에만 호출되는 비동기 훅을 받는다.

```python
from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TypeVar

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError

from trainer.demo_api import simulator_stub as stub
from trainer.demo_api.monitor import MonitorRecorder
from trainer.demo_api.sim_contracts import (
    Phase,
    SessionEventInput,
    SessionStartInput,
    SimSessionResponse,
    SimSessionView,
)
from trainer.demo_api.sim_session import SimSessionState, SimSessionStore
from trainer.demo_api.zone_resolver import ZoneResolver
from trainer.scripts.session_coaching import SessionSummary

TInput = TypeVar("TInput", bound=BaseModel)


def _to_view(state: SimSessionState) -> SimSessionView:
    return SimSessionView(
        session_id=state.session_id,
        level_id=state.level_id,
        phase=state.phase,
        recommended_flow=list(state.recommended_flow),
        current_zone_id=state.current_zone_id,
        completed_zone_ids=list(state.completed_zone_ids),
        last_event_id=state.last_event_id,
        anomaly_count=state.anomaly_count,
        unresolved_zone_id=state.unresolved_zone_id,
        updated_at=state.updated_at,
    )


def create_simulator_router(
    recorder: MonitorRecorder, store: SimSessionStore, resolver: ZoneResolver
) -> APIRouter:
    """Deterministic three-phase simulator interface that records every exchange.

    Every call writes exactly two monitor events - the raw request body as
    ``input`` and the produced response as ``output`` - so rejected payloads stay
    observable instead of disappearing behind a 422. Accepted calls additionally
    advance ``store``, which is what the sequence snapshot and the chat
    ``environment`` read from.
    """
    router = APIRouter(prefix="/api/sim", tags=["simulator"])

    async def _run_phase(
        request: Request,
        phase: Phase,
        path_session_id: str | None,
        model_cls: type[TInput],
        build_output: Callable[[TInput], BaseModel],
        on_success: Callable[[TInput, BaseModel], Awaitable[None]] | None = None,
    ) -> JSONResponse:
        error: dict[str, object]

        try:
            raw: object = await request.json()
        except ValueError:
            session_id = path_session_id or "unknown"
            error = {"error": "invalid_json"}
            _ = recorder.record(session_id, phase, "input", dict(error))
            _ = recorder.record(session_id, phase, "output", error)
            return JSONResponse(status_code=422, content=error)

        body: dict[str, object] = raw if isinstance(raw, dict) else {}
        session_id = path_session_id or str(body.get("session_id", "")) or "unknown"
        _ = recorder.record(session_id, phase, "input", body)

        if path_session_id is not None and "session_id" in body and body["session_id"] != path_session_id:
            error = {
                "error": "session_id_mismatch",
                "path": path_session_id,
                "body": body["session_id"],
            }
            _ = recorder.record(session_id, phase, "output", error)
            return JSONResponse(status_code=422, content=error)

        try:
            payload = model_cls.model_validate(body)
        except ValidationError as exc:
            # url/context off: pydantic puts raw exception objects in ctx, which
            # JSONResponse could not serialize.
            error = {
                "error": "validation_error",
                "detail": exc.errors(include_url=False, include_context=False),
            }
            _ = recorder.record(session_id, phase, "output", error)
            return JSONResponse(status_code=422, content=error)

        output = build_output(payload)
        output_payload: dict[str, object] = output.model_dump(mode="json")
        _ = recorder.record(session_id, phase, "output", output_payload)
        if on_success is not None:
            await on_success(payload, output)
        return JSONResponse(status_code=200, content=output_payload)

    @router.post("/sessions/start")
    async def start(request: Request) -> JSONResponse:
        async def _apply(payload: SessionStartInput, output: BaseModel) -> None:
            flow = getattr(output, "recommended_flow", [])
            _ = store.start(
                session_id=payload.session_id,
                level_id=payload.level_id,
                recommended_flow=list(flow),
            )

        return await _run_phase(
            request, "start", None, SessionStartInput, stub.build_start_output, _apply
        )

    @router.post("/sessions/{session_id}/events")
    async def events(session_id: str, request: Request) -> JSONResponse:
        async def _apply(payload: SessionEventInput, output: BaseModel) -> None:
            target = payload.zone_id
            if target is None:
                coordinates = (
                    payload.telemetry.coordinates if payload.telemetry is not None else None
                )
                if coordinates is not None:
                    target = await resolver.resolve(list(coordinates))
            _ = store.runtime(
                session_id=session_id,
                event_id=payload.event_id,
                target_zone_id=target,
                anomaly=getattr(output, "kind", "") == "anomaly",
            )

        return await _run_phase(
            request,
            "runtime",
            session_id,
            SessionEventInput,
            lambda payload: stub.build_event_output(session_id, payload),
            _apply,
        )

    @router.post("/sessions/{session_id}/complete")
    async def complete(session_id: str, request: Request) -> JSONResponse:
        async def _apply(payload: SessionSummary, output: BaseModel) -> None:
            _ = payload
            _ = output
            _ = store.end(session_id=session_id)

        return await _run_phase(
            request, "session_end", session_id, SessionSummary, stub.build_review_output, _apply
        )

    @router.get("/session")
    def session_state() -> JSONResponse:
        state = store.current()
        response = SimSessionResponse(session=None if state is None else _to_view(state))
        return JSONResponse(content=response.model_dump(mode="json"))

    return router
```

- [ ] **Step 4: 통과 확인**

Run: `docker exec -w /workspace ship-paint-trainer python3 -m pytest trainer/demo_api/tests/test_simulator_interface.py -q`
Expected: PASS. 기존 모니터 기록 테스트(422 포함)도 그대로 통과해야 한다.

- [ ] **Step 5: 커밋**

```bash
git add trainer/demo_api/simulator.py trainer/demo_api/tests/test_simulator_interface.py
git commit -m "feat(demo-api): 시뮬레이터 라우터가 세션 store를 갱신하고 세션 조회를 노출"
```

---

## Task 7: 앱 배선 — `/api/sequence`와 live `environment`

**Files:**
- Modify: `trainer/demo_api/app.py:28-93`
- Modify: `trainer/demo_api/chat_service.py:127-139`
- Test: `trainer/demo_api/tests/test_app_sim_wiring.py`, `trainer/demo_api/tests/test_chat_service_live_runner.py`

**Interfaces:**
- Consumes: Task 3·4·5·6 산출물
- Produces: `build_default_live_runner(store: SimSessionStore | None = None) -> LiveRunner`

- [ ] **Step 1: 실패하는 테스트 작성**

`trainer/demo_api/tests/test_app_sim_wiring.py` 끝에 추가:

```python
START_BODY = {
    "session_id": "s-1",
    "level_id": "SP0035-L01",
    "worker": {"trainee_id": "t-1", "skill_level": "beginner", "handedness": "right"},
    "spec": {"nominal_um": 100.0},
}


def test_sequence_returns_fixture_before_any_session() -> None:
    client = TestClient(create_app(ChatService()))
    body = client.get("/api/sequence/SP0035-L01").json()
    assert body["current_zone_id"] == "A-1"
    assert body["completed_zone_ids"] == ["A-1"]


def test_sequence_follows_simulator_session() -> None:
    client = TestClient(create_app(ChatService()))
    _ = client.post("/api/sim/sessions/start", json=START_BODY)
    _ = client.post("/api/sim/sessions/s-1/events", json={"event_id": "ev-0", "zone_id": "A-2"})

    body = client.get("/api/sequence/SP0035-L01").json()
    assert body["current_zone_id"] == "A-2"
    assert body["completed_zone_ids"] == ["A-1"]
    statuses = {zone["zone_id"]: zone["status"] for zone in body["zones"]}
    assert statuses == {"A-1": "completed", "A-2": "current", "A-3": "pending"}


def test_sequence_ignores_session_for_a_different_level() -> None:
    client = TestClient(create_app(ChatService()))
    _ = client.post("/api/sim/sessions/start", json=START_BODY)
    body = client.get("/api/sequence/OTHER-L02").json()
    assert body["level_id"] == "OTHER-L02"
    assert body["completed_zone_ids"] == ["A-1"]
```

`trainer/demo_api/tests/test_chat_service_live_runner.py` 끝에 추가:

```python
from trainer.demo_api.chat_service import build_default_live_runner
from trainer.demo_api.sim_session import SimSessionStore


def test_live_runner_environment_falls_back_to_fixture_without_session(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _FakePipeline:
        async def run_streaming(self, **kwargs: object):
            captured.update(kwargs)
            yield "complete", {}

    monkeypatch.setattr(
        "trainer.demo_api.chat_service._get_or_build_pipeline", lambda: _FakePipeline()
    )
    runner = build_default_live_runner(SimSessionStore())

    async def scenario() -> None:
        async for _event in runner(
            ChatRequest(message="hi", session_id="s-1", mode="live", level_id="SP0035-L01")
        ):
            pass

    anyio.run(scenario)
    environment = captured["environment"]
    assert isinstance(environment, dict)
    assert environment["current_zone_id"] == "A-1"
    painted = {zone["zone_id"]: zone["is_painted"] for zone in environment["zone_map"]}
    assert painted == {"A-1": True, "A-2": False, "A-3": False}


def test_live_runner_environment_uses_active_session(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _FakePipeline:
        async def run_streaming(self, **kwargs: object):
            captured.update(kwargs)
            yield "complete", {}

    monkeypatch.setattr(
        "trainer.demo_api.chat_service._get_or_build_pipeline", lambda: _FakePipeline()
    )
    store = SimSessionStore()
    _ = store.start(
        session_id="s-1", level_id="SP0035-L01", recommended_flow=["A-1", "A-2", "A-3"]
    )
    _ = store.runtime(session_id="s-1", event_id="ev-0", target_zone_id="A-3", anomaly=False)
    runner = build_default_live_runner(store)

    async def scenario() -> None:
        async for _event in runner(
            ChatRequest(message="hi", session_id="s-1", mode="live", level_id="SP0035-L01")
        ):
            pass

    anyio.run(scenario)
    environment = captured["environment"]
    assert isinstance(environment, dict)
    assert environment["current_zone_id"] == "A-3"
    painted = {zone["zone_id"]: zone["is_painted"] for zone in environment["zone_map"]}
    assert painted == {"A-1": True, "A-2": True, "A-3": False}
```

- [ ] **Step 2: 실패 확인**

Run: `docker exec -w /workspace ship-paint-trainer python3 -m pytest trainer/demo_api/tests/test_app_sim_wiring.py trainer/demo_api/tests/test_chat_service_live_runner.py -q`
Expected: FAIL — `/api/sequence`가 세션을 무시하고, `build_default_live_runner()`가 인자를 받지 않음

- [ ] **Step 3: `chat_service.py` 수정**

```python
def build_default_live_runner(store: SimSessionStore | None = None) -> LiveRunner:
    async def _runner(request: ChatRequest) -> AsyncIterator[tuple[str, dict[str, object]]]:
        yield "status", {"stage": "initializing", "chunks": 0}
        pipeline = await to_thread.run_sync(_get_or_build_pipeline)

        state = store.current() if store is not None else None
        if state is not None and state.level_id == request.level_id:
            environment = dict(session_tool_environment(state))
        else:
            environment = dict(default_tool_environment(request.level_id))

        async for event in pipeline.run_streaming(
            instruction=request.message,
            session_id=request.session_id,
            enable_thinking=False,
            environment=environment,
        ):
            yield event

    return _runner
```

import에 `from trainer.demo_api.sequence import default_tool_environment, session_tool_environment`와 `from trainer.demo_api.sim_session import SimSessionStore`를 더한다.

- [ ] **Step 4: `app.py` 수정**

store와 resolver를 ChatService보다 먼저 만들고, `/api/sequence`가 store를 우선 읽게 한다.

```python
def create_app(chat_service: ChatService | None = None) -> FastAPI:
    recorder = MonitorRecorder()
    store = SimSessionStore()
    mcp_base_url = os.environ.get("MCP_BASE_URL", "http://localhost:8100")
    resolver = McpZoneResolver(mcp_base_url)

    if chat_service is not None:
        service = chat_service
    else:
        live_mode = os.environ.get("DEMO_LIVE_MODE", "off")
        if live_mode not in {"off", "on"}:
            raise RuntimeError("DEMO_LIVE_MODE must be 'off' or 'on'")

        logger.info("DEMO_LIVE_MODE=%s", live_mode)

        if live_mode == "on":
            service = ChatService(live_runner=build_default_live_runner(store))
        else:
            service = ChatService()

    app = FastAPI(title="Ship Painting Demo API")
    app.state.chat_service = service
    app.state.monitor_recorder = recorder
    app.state.sim_session_store = store
    ...

    @app.get("/api/sequence/{level_id}")
    def sequence(level_id: str) -> dict[str, object]:
        if _LEVEL_ID_PATTERN.fullmatch(level_id) is None:
            raise HTTPException(status_code=400, detail="invalid level_id")
        state = store.current()
        if state is not None and state.level_id == level_id:
            return session_snapshot(state).model_dump(mode="json")
        return default_snapshot(level_id).model_dump(mode="json")

    ...
    app.include_router(create_simulator_router(recorder, store, resolver))
    app.include_router(create_monitor_router(recorder))
    return app
```

기존 파일 하단의 `recorder = MonitorRecorder()` 블록은 제거한다(위로 옮겼다). import에 `session_snapshot`, `SimSessionStore`, `McpZoneResolver`를 더한다.

- [ ] **Step 5: 통과 확인**

Run: `docker exec -w /workspace ship-paint-trainer python3 -m pytest trainer mcp_servers -q`
Expected: PASS. 기준선 176개 + 이번에 추가한 테스트 전부.

- [ ] **Step 6: 커밋**

```bash
git add trainer/demo_api/app.py trainer/demo_api/chat_service.py trainer/demo_api/tests/test_app_sim_wiring.py trainer/demo_api/tests/test_chat_service_live_runner.py
git commit -m "feat(demo-api): 순서 스냅샷과 live environment를 시뮬레이터 세션에서 읽도록 배선"
```

---

## Task 8: 프런트 API 클라이언트 — 3단계 송신기와 세션 조회

**Files:**
- Modify: `demo-ui/src/simMonitorClient.ts`
- Test: `demo-ui/tests/simMonitorClient.test.ts`

**Interfaces:**
- Consumes: `simSessionResponseSchema`(Task 2), `/api/sim/*`(Task 6)
- Produces:
  - `sendSessionStart(sessionId: string, levelId: string): Promise<void>`
  - `sendRuntimeEvent(sessionId: string, eventId: string, zoneId: string, anomaly: boolean): Promise<void>`
  - `sendSessionEnd(sessionId: string): Promise<void>`
  - `fetchSimSession(signal?: AbortSignal): Promise<SimSession | null>`
- `sendDummySession`은 제거한다(3버튼이 대체).

- [ ] **Step 1: 실패하는 테스트 작성**

`demo-ui/tests/simMonitorClient.test.ts` 끝에 추가:

```ts
import { fetchSimSession, sendRuntimeEvent, sendSessionEnd, sendSessionStart } from '../src/simMonitorClient'

describe('3단계 송신기', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
  })

  it('세션 시작은 start 엔드포인트에 레벨과 세션 ID를 보낸다', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response('{}', { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)

    await sendSessionStart('s-1', 'SP0035-L01')

    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit]
    expect(url).toContain('/api/sim/sessions/start')
    expect(JSON.parse(String(init.body))).toMatchObject({
      session_id: 's-1',
      level_id: 'SP0035-L01',
    })
  })

  it('런타임은 zone_id를 싣고 이상 여부에 따라 파라미터를 바꾼다', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response('{}', { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)

    await sendRuntimeEvent('s-1', 'ev-1', 'A-2', true)

    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit]
    expect(url).toContain('/api/sim/sessions/s-1/events')
    const body = JSON.parse(String(init.body)) as Record<string, unknown>
    expect(body).toMatchObject({ event_id: 'ev-1', zone_id: 'A-2' })
    expect(body.telemetry).toMatchObject({ gun_metrics: { standoff_mm: 350 } })
  })

  it('세션 끝은 complete 엔드포인트를 호출한다', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response('{}', { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)

    await sendSessionEnd('s-1')

    const [url] = fetchMock.mock.calls[0] as [string]
    expect(url).toContain('/api/sim/sessions/s-1/complete')
  })

  it('세션 조회는 session이 null이면 null을 돌려준다', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(new Response(JSON.stringify({ session: null }), { status: 200 })),
    )
    await expect(fetchSimSession()).resolves.toBeNull()
  })
})
```

- [ ] **Step 2: 실패 확인**

Run: `cd demo-ui && npx vitest run tests/simMonitorClient.test.ts`
Expected: FAIL — export 없음

- [ ] **Step 3: 구현**

`demo-ui/src/simMonitorClient.ts`에서 `sendDummySession`을 아래로 대체한다.

```ts
import { API_BASE_URL, createSseParser } from './api'
import {
  monitorEventSchema,
  simSessionResponseSchema,
  type MonitorEvent,
  type SimSession,
} from './simContracts'

const OUT_OF_BAND = { speed_m_per_s: 0.8, standoff_mm: 350.0, angle_deg: 55.0 }
const IN_BAND = { speed_m_per_s: 0.8, standoff_mm: 300.0, angle_deg: 72.0 }

export async function sendSessionStart(sessionId: string, levelId: string): Promise<void> {
  await postJson('/api/sim/sessions/start', {
    session_id: sessionId,
    level_id: levelId,
    worker: { trainee_id: 't-dummy', skill_level: 'beginner', handedness: 'right' },
    spec: { nominal_um: 100.0 },
  })
}

export async function sendRuntimeEvent(
  sessionId: string,
  eventId: string,
  zoneId: string,
  anomaly: boolean,
): Promise<void> {
  await postJson(`/api/sim/sessions/${encodeURIComponent(sessionId)}/events`, {
    event_id: eventId,
    zone_id: zoneId,
    telemetry: { gun_metrics: anomaly ? OUT_OF_BAND : IN_BAND },
  })
}

export async function sendSessionEnd(sessionId: string): Promise<void> {
  await postJson(`/api/sim/sessions/${encodeURIComponent(sessionId)}/complete`, {
    session_id: sessionId,
    scenario_id: 'dummy',
    mode: 'live',
    zone_id: 'A-2',
    zone_type: 'panel',
    metrics: {
      avg_dft: 100.0,
      dft_values: [95.0, 102.0, 88.0],
      avg_speed: 0.8,
      speed_variance: 0.05,
      avg_distance: 0.3,
      distance_variance: 0.05,
      avg_angle: 70.0,
      angle_consistency: 0.9,
      sequence_correct: true,
      under_thickness_count: 1,
    },
  })
}

export async function fetchSimSession(signal?: AbortSignal): Promise<SimSession | null> {
  const init: RequestInit = signal === undefined ? {} : { signal }
  const response = await fetch(`${API_BASE_URL}/api/sim/session`, init)
  if (!response.ok) {
    throw new Error(`sim session request failed: ${response.status}`)
  }
  const body: unknown = await response.json()
  return simSessionResponseSchema.parse(body).session
}
```

- [ ] **Step 4: 통과 확인 후 커밋**

Run: `cd demo-ui && npx vitest run tests/simMonitorClient.test.ts && npm run build:web`
Expected: PASS. `App.tsx`가 아직 `sendDummySession`을 참조하면 `build:web`이 실패하므로 Task 10에서 함께 정리한다. 이 단계에서는 `App.tsx`의 import와 호출만 임시로 `sendSessionStart`로 바꿔 컴파일을 통과시킨 뒤 커밋한다.

```bash
git add demo-ui/src/simMonitorClient.ts demo-ui/tests/simMonitorClient.test.ts demo-ui/src/App.tsx
git commit -m "feat(demo-ui): 시뮬레이터 3단계 송신기와 세션 조회 클라이언트"
```

---

## Task 9: 세션 배지가 있는 `SequencePanel`

**Files:**
- Modify: `demo-ui/src/components/SequencePanel.tsx`
- Modify: `demo-ui/src/styles.css` (`.sequence__session` 규칙 추가)
- Test: `demo-ui/tests/SequencePanel.test.tsx`

**Interfaces:**
- Consumes: `SimSession`(Task 2)
- Produces: `SequencePanel({ snapshot, session }: { snapshot: SequenceSnapshot; session: SimSession | null })`

- [ ] **Step 1: 실패하는 테스트 작성**

`demo-ui/tests/SequencePanel.test.tsx` 끝에 추가:

```tsx
const SNAPSHOT = {
  level_id: 'SP0035-L01',
  label: 'SP0035 Level 01',
  zones: [
    { zone_id: 'A-1', label: '상부', status: 'completed' as const },
    { zone_id: 'A-2', label: '중간', status: 'current' as const },
    { zone_id: 'A-3', label: '하부', status: 'pending' as const },
  ],
  current_zone_id: 'A-2',
  next_zone_ids: ['A-3'],
  completed_zone_ids: ['A-1'],
}

const SESSION = {
  session_id: 'abcdef12-0000-0000-0000-000000000000',
  level_id: 'SP0035-L01',
  phase: 'runtime' as const,
  recommended_flow: ['A-1', 'A-2', 'A-3'],
  current_zone_id: 'A-2',
  completed_zone_ids: ['A-1'],
  last_event_id: 'ev-1',
  anomaly_count: 1,
  unresolved_zone_id: null,
  updated_at: '2026-07-28T00:00:00+00:00',
}

it('세션이 없으면 대기로 표시한다', () => {
  render(<SequencePanel snapshot={SNAPSHOT} session={null} />)
  expect(screen.getByTestId('sim-phase')).toHaveTextContent('대기')
})

it('세션 단계와 짧은 세션 ID를 표시한다', () => {
  render(<SequencePanel snapshot={SNAPSHOT} session={SESSION} />)
  expect(screen.getByTestId('sim-phase')).toHaveTextContent('런타임')
  expect(screen.getByTestId('sim-session-id')).toHaveTextContent('abcdef12')
})

it('미매핑 구역을 알린다', () => {
  render(<SequencePanel snapshot={SNAPSHOT} session={{ ...SESSION, unresolved_zone_id: 'B-3' }} />)
  expect(screen.getByTestId('sim-unresolved')).toHaveTextContent('B-3')
})
```

- [ ] **Step 2: 실패 확인**

Run: `cd demo-ui && npx vitest run tests/SequencePanel.test.tsx`
Expected: FAIL — `session` prop 없음, `sim-phase` testid 없음

- [ ] **Step 3: 구현**

`demo-ui/src/components/SequencePanel.tsx`:

```tsx
import type { ReactElement } from 'react'
import type { SequenceSnapshot, ZoneStatus } from '../contracts'
import type { Phase, SimSession } from '../simContracts'

const STATUS_LABEL: Record<ZoneStatus, string> = {
  current: '현재',
  next: '다음',
  completed: '완료',
  pending: '대기',
}

const PHASE_LABEL: Record<Phase, string> = {
  start: '세션 시작',
  runtime: '런타임',
  session_end: '세션 끝',
}

type Props = {
  readonly snapshot: SequenceSnapshot
  readonly session: SimSession | null
}

export function SequencePanel({ snapshot, session }: Props): ReactElement {
  const completed = snapshot.zones.filter((zone) => zone.status === 'completed').length

  return (
    <section aria-label="painting-sequence" className="sequence">
      <header className="sequence__header">
        <p className="panel__eyebrow">도장 순서</p>
        <h3 className="sequence__title">{snapshot.label}</h3>
        <dl className="sequence__id">
          <dt className="sequence__id-label">레벨 ID</dt>
          <dd className="sequence__id-value" title="도장 순서 조회에 사용하는 식별자">
            {snapshot.level_id}
          </dd>
        </dl>
        <p className="sequence__session">
          <span className="sequence__session-phase" data-testid="sim-phase">
            {session === null ? '대기' : PHASE_LABEL[session.phase]}
          </span>
          {session !== null && (
            <span className="sequence__session-id" data-testid="sim-session-id">
              {session.session_id.slice(0, 8)}
            </span>
          )}
        </p>
        {session?.unresolved_zone_id != null && (
          <p className="sequence__unresolved" data-testid="sim-unresolved">
            미매핑 구역 {session.unresolved_zone_id}
          </p>
        )}
        <p className="sequence__progress">
          구역 {completed} / {snapshot.zones.length} 완료
        </p>
      </header>

      <ol className="sequence__list">
        {snapshot.zones.map((zone, index) => (
          <li
            key={zone.zone_id}
            data-testid={`zone-${zone.zone_id}`}
            data-status={zone.status}
            className="sequence__zone"
          >
            <span className="sequence__zone-index">{String(index + 1).padStart(2, '0')}</span>
            <span className="sequence__zone-id">{zone.zone_id}</span>
            <span className="sequence__zone-label">{zone.label}</span>
            <span className="sequence__zone-badge">{STATUS_LABEL[zone.status]}</span>
          </li>
        ))}
      </ol>
    </section>
  )
}
```

`demo-ui/src/styles.css`의 `.sequence__progress` 규칙 바로 앞에 추가:

```css
.sequence__session {
  display: flex;
  gap: var(--s-2);
  align-items: baseline;
  font-size: var(--t-micro);
}

.sequence__session-phase {
  font-weight: 650;
  color: var(--signal-ink);
}

.sequence__session-id {
  font-family: var(--font-mono);
  color: var(--ink-300);
}

.sequence__unresolved {
  font-size: var(--t-micro);
  color: var(--ink-300);
}
```

`MonitorPanel`이 `SequencePanel`을 렌더하므로 `MonitorPanel` props에 `simSession: SimSession | null`을 더하고 그대로 넘긴다.

- [ ] **Step 4: 통과 확인 후 커밋**

Run: `cd demo-ui && npx vitest run tests/SequencePanel.test.tsx && npm run build:web`

```bash
git add demo-ui/src/components/SequencePanel.tsx demo-ui/src/components/MonitorPanel.tsx demo-ui/src/styles.css demo-ui/tests/SequencePanel.test.tsx
git commit -m "feat(demo-ui): 도장 순서 패널에 시뮬레이터 세션 단계 표시"
```

---

## Task 10: 3버튼 패널과 App 배선

**Files:**
- Modify: `demo-ui/src/components/SimulatorMonitorPanel.tsx`
- Modify: `demo-ui/src/App.tsx`
- Test: `demo-ui/tests/simulatorMonitorPanel.test.tsx`, `demo-ui/tests/appSimPanel.test.tsx`

**Interfaces:**
- Consumes: Task 8 송신기, Task 9 패널 props
- Produces: `SimulatorMonitorPanel({ events, lifecycle, onStart, onRuntime, onEnd })`, `lifecycle: 'idle' | 'started'`

- [ ] **Step 1: 실패하는 패널 테스트 작성**

`demo-ui/tests/simulatorMonitorPanel.test.tsx` 끝에 추가:

```tsx
it('대기 상태에서는 런타임과 세션 끝이 비활성이다', () => {
  render(
    <SimulatorMonitorPanel
      events={[]}
      lifecycle="idle"
      onStart={() => {}}
      onRuntime={() => {}}
      onEnd={() => {}}
    />,
  )
  expect(screen.getByRole('button', { name: '세션 시작' })).toBeEnabled()
  expect(screen.getByRole('button', { name: '런타임' })).toBeDisabled()
  expect(screen.getByRole('button', { name: '세션 끝' })).toBeDisabled()
})

it('시작 후에는 세 버튼의 활성 상태가 뒤바뀐다', () => {
  render(
    <SimulatorMonitorPanel
      events={[]}
      lifecycle="started"
      onStart={() => {}}
      onRuntime={() => {}}
      onEnd={() => {}}
    />,
  )
  expect(screen.getByRole('button', { name: '세션 시작' })).toBeDisabled()
  expect(screen.getByRole('button', { name: '런타임' })).toBeEnabled()
  expect(screen.getByRole('button', { name: '세션 끝' })).toBeEnabled()
})

it('각 버튼이 대응하는 핸들러를 부른다', async () => {
  const onStart = vi.fn()
  const onRuntime = vi.fn()
  const onEnd = vi.fn()
  render(
    <SimulatorMonitorPanel
      events={[]}
      lifecycle="started"
      onStart={onStart}
      onRuntime={onRuntime}
      onEnd={onEnd}
    />,
  )
  await userEvent.click(screen.getByRole('button', { name: '런타임' }))
  await userEvent.click(screen.getByRole('button', { name: '세션 끝' }))
  expect(onRuntime).toHaveBeenCalledTimes(1)
  expect(onEnd).toHaveBeenCalledTimes(1)
  expect(onStart).not.toHaveBeenCalled()
})
```

- [ ] **Step 2: 실패 확인**

Run: `cd demo-ui && npx vitest run tests/simulatorMonitorPanel.test.tsx`
Expected: FAIL — `더미 세션 송신` 버튼만 존재

- [ ] **Step 3: 패널 구현**

`demo-ui/src/components/SimulatorMonitorPanel.tsx`의 props와 바를 교체한다. 타임라인 렌더링 부분은 그대로 둔다.

```tsx
export type SimLifecycle = 'idle' | 'started'

type Props = {
  readonly events: readonly MonitorEvent[]
  readonly lifecycle: SimLifecycle
  readonly onStart: () => void
  readonly onRuntime: () => void
  readonly onEnd: () => void
}
```

```tsx
      <div className="sim-monitor__bar">
        <h3 className="sim-monitor__title">시뮬레이터 인터페이스 모니터</h3>
        <div className="sim-monitor__actions">
          <button
            type="button"
            className="sim-monitor__send"
            onClick={onStart}
            disabled={lifecycle !== 'idle'}
          >
            세션 시작
          </button>
          <button
            type="button"
            className="sim-monitor__send"
            onClick={onRuntime}
            disabled={lifecycle !== 'started'}
          >
            런타임
          </button>
          <button
            type="button"
            className="sim-monitor__send"
            onClick={onEnd}
            disabled={lifecycle !== 'started'}
          >
            세션 끝
          </button>
        </div>
      </div>
```

`demo-ui/src/simMonitor.css`에 추가:

```css
.sim-monitor__actions {
  display: flex;
  gap: 0.375rem;
}

.sim-monitor__send:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}
```

- [ ] **Step 4: 실패하는 App 테스트 작성**

`demo-ui/tests/appSimPanel.test.tsx` 끝에 추가:

```tsx
it('세션 시작 후 런타임 클릭이 다음 구역을 보낸다', async () => {
  const posts: Array<{ url: string; body: Record<string, unknown> }> = []
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)
      if (url.includes('/api/monitor/stream')) {
        return new Response(new ReadableStream(), { status: 200 })
      }
      if (url.includes('/api/sim/session')) {
        return new Response(JSON.stringify({ session: null }), { status: 200 })
      }
      if (url.includes('/api/sequence/')) {
        return new Response(JSON.stringify(SEQUENCE_SNAPSHOT), { status: 200 })
      }
      if (url.includes('/api/sim/sessions/')) {
        posts.push({ url, body: JSON.parse(String(init?.body ?? '{}')) })
        return new Response('{}', { status: 200 })
      }
      return new Response(JSON.stringify({ status: 'ok', servers: 1, timestamp: 'x' }), {
        status: 200,
      })
    }),
  )

  render(<App />)
  await userEvent.click(await screen.findByRole('button', { name: '세션 시작' }))
  await userEvent.click(await screen.findByRole('button', { name: '런타임' }))

  const runtimePost = posts.find((post) => post.url.includes('/events'))
  expect(runtimePost).toBeDefined()
  expect(runtimePost?.body.zone_id).toBe('A-2')
})
```

`SEQUENCE_SNAPSHOT`은 `current_zone_id: 'A-1'`, zones `A-1/A-2/A-3`인 기존 픽스처를 재사용한다.

- [ ] **Step 5: App 배선 구현**

`demo-ui/src/App.tsx`에 상태와 핸들러를 더한다.

```tsx
const [simLifecycle, setSimLifecycle] = useState<SimLifecycle>('idle')
const [simSessionId, setSimSessionId] = useState<string | null>(null)
const [simSession, setSimSession] = useState<SimSession | null>(null)
const runtimeCountRef = useRef<number>(0)

const onSimStart = useCallback(() => {
  const id = crypto.randomUUID()
  setSimSessionId(id)
  setSimLifecycle('started')
  runtimeCountRef.current = 0
  void sendSessionStart(id, snapshot.level_id)
}, [snapshot.level_id])

const onSimRuntime = useCallback(() => {
  if (simSessionId === null) {
    return
  }
  const index = snapshot.zones.findIndex((zone) => zone.zone_id === snapshot.current_zone_id)
  const nextZone = snapshot.zones[index + 1] ?? snapshot.zones[index]
  if (nextZone === undefined) {
    return
  }
  const count = runtimeCountRef.current
  runtimeCountRef.current = count + 1
  void sendRuntimeEvent(simSessionId, `ev-${count}`, nextZone.zone_id, count === 0)
}, [simSessionId, snapshot])

const onSimEnd = useCallback(() => {
  if (simSessionId === null) {
    return
  }
  setSimLifecycle('idle')
  void sendSessionEnd(simSessionId)
}, [simSessionId])
```

모니터 이벤트가 늘 때 두 조회를 함께 갱신한다.

```tsx
useEffect(() => {
  if (monitorEvents.length === 0) {
    return
  }
  let cancelled = false
  void (async () => {
    const [snapshotResult, sessionResult] = await Promise.allSettled([
      fetchSequenceSnapshot(LEVEL_ID),
      fetchSimSession(),
    ])
    if (cancelled) {
      return
    }
    if (snapshotResult.status === 'fulfilled') {
      setSnapshot(snapshotResult.value)
    }
    if (sessionResult.status === 'fulfilled') {
      setSimSession(sessionResult.value)
    }
    if (snapshotResult.status === 'rejected' || sessionResult.status === 'rejected') {
      appendLog({
        kind: 'error',
        at: Date.now(),
        message: '시뮬레이터 상태 갱신 실패',
        retryable: true,
      })
    }
  })()
  return () => {
    cancelled = true
  }
}, [monitorEvents.length, appendLog])
```

렌더 부분에서 `MonitorPanel`에 `simSession={simSession}`을, `SimulatorMonitorPanel`에 `lifecycle`/`onStart`/`onRuntime`/`onEnd`를 넘긴다.

- [ ] **Step 6: 통과 확인 후 커밋**

Run: `cd demo-ui && npm test && npm run build:web`
Expected: PASS, 기준선 79개 + 신규 테스트

```bash
git add demo-ui/src/App.tsx demo-ui/src/components/SimulatorMonitorPanel.tsx demo-ui/src/simMonitor.css demo-ui/tests/simulatorMonitorPanel.test.tsx demo-ui/tests/appSimPanel.test.tsx
git commit -m "feat(demo-ui): 더미 입력 3단계 분리와 좌측 상태 동기화"
```

---

## Task 11: 전체 검증과 실사용 QA

**Files:**
- Modify: `demo-ui/e2e/layout-width.spec.ts` (Task 1의 임시 본문을 최종 본문으로 복구)

- [ ] **Step 1: e2e 최종 본문 복구**

Task 1 Step 1의 최종 본문(`세션 시작` 버튼 클릭 버전)으로 되돌린다.

- [ ] **Step 2: 전체 게이트 실행**

```bash
docker exec -w /workspace ship-paint-trainer python3 -m pytest trainer mcp_servers -q
cd demo-ui && npm run build:web && npm test && npm run test:e2e
```

Expected: Python 176+ passed, Vitest 79+ passed, build 성공, Playwright 3 passed.

- [ ] **Step 3: 실제 사용 QA**

컨테이너의 데모 API를 재기동한 뒤(신규 라우트 반영) 브라우저에서 직접 조작한다.

```bash
docker exec ship-paint-trainer pkill -f "trainer.demo_api.app"
docker exec -d -w /workspace ship-paint-trainer python3 -m trainer.demo_api.app --host 0.0.0.0 --port 8120 --live
curl -s http://127.0.0.1:8120/api/sim/session
```

확인 항목:
1. `세션 시작` 클릭 → 좌측 도장 순서가 A-1 `현재`, 나머지 `대기`로 바뀐다. 단계 배지가 `세션 시작`.
2. `런타임` 클릭 → A-1 `완료`, A-2 `현재`. 단계 배지가 `런타임`. 모니터에 anomaly 출력.
3. `런타임` 재클릭 → A-3 `현재`, A-1·A-2 `완료`.
4. `세션 끝` 클릭 → 단계 배지가 `세션 끝`, 위치 유지, 버튼 활성 상태 복귀.
5. 가로 스크롤이 생기지 않고 사용자 말풍선이 화면 안에 보인다.
6. `live` 모드에서 "다음 순서가 어떻게 돼?"를 물어 응답이 시뮬레이터가 알려준 현재 구역을 반영하는지 확인한다.

- [ ] **Step 4: 산출물 정리와 최종 커밋**

임시 파일과 기동한 서버를 정리한 뒤:

```bash
git add demo-ui/e2e/layout-width.spec.ts
git commit -m "test(demo-ui): 레이아웃 폭 회귀를 3버튼 흐름으로 검증"
```

---

## Self-Review

**1. Spec coverage**

| 스펙 절 | 담당 Task |
|---|---|
| §4 상태 모델 | Task 3 |
| §5.1~5.3 전이 규칙 | Task 3 |
| §5.4 갱신 시점·422 무변경 | Task 6 |
| §6 ZoneResolver + B-3 한계 | Task 4, Task 6 |
| §7 `zone_id` 계약 | Task 2 |
| §8.1 `/api/sequence` 폴백 | Task 7 |
| §8.2 `GET /api/sim/session` | Task 6 |
| §8.3 live `environment` | Task 7 |
| §9.1 3버튼·생명주기 | Task 8, Task 10 |
| §9.2 좌측 패널 갱신·실패 시 유지 | Task 9, Task 10 |
| §9.3 레이아웃 폭 수정 | Task 1 |
| §10 테스트 전략 | 각 Task의 RED 단계 + Task 11 |

누락 없음.

**2. Placeholder scan**

"적절한 오류 처리", "TBD", "위와 유사하게" 없음. 모든 코드 단계에 실제 코드가 있다.

**3. Type consistency**

- `SimSessionStore.start/runtime/end/current` 시그니처가 Task 3 정의와 Task 6·7 사용처에서 동일하다.
- `session_snapshot` / `session_tool_environment`가 Task 5에서 정의되고 Task 7에서 같은 이름으로 쓰인다.
- `SimSessionView` 필드 목록(Task 2)과 Zod `simSessionSchema` 필드 목록(Task 2), `_to_view`(Task 6)가 일치한다.
- `SimLifecycle`이 Task 10에서 정의되고 같은 Task 안에서만 쓰인다.
- Task 8에서 `sendDummySession`을 제거하므로 Task 10까지 `App.tsx`가 컴파일되도록 Task 8 Step 4에 임시 조치를 명시했다.
