# Electron·React Chat Demo Implementation Plan (1차: 3D 없이)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve the existing Gradio demo while adding a React web UI and secure Electron desktop shell that show AI chat and a structured painting-sequence panel. No 3D viewer in this iteration.

**Architecture:** A focused FastAPI demo API owns chat streaming and AgentPipeline-to-sequence translation. Gradio continues to run through `demo_app.py`; React and Electron share one renderer build and consume the FastAPI HTTP/SSE contract.

**Tech Stack:** Python 3.11, FastAPI, Pydantic, pytest, React, TypeScript, Vite, Zod, Vitest, Playwright, Electron.

**Design:** `docs/superpowers/specs/2026-07-24-electron-glb-chat-demo-design.md`

**Source references:** Electron security and preload patterns from `/electron/electron`; React setup from Vite/React docs.

**Git note:** Commit steps are intentionally omitted because repository policy requires explicit user authorization before creating commits.

**Out of scope for this plan:** Three.js/GLB/FBX viewer, Blender conversion CLI, zone-to-mesh mapping. These are deferred to a follow-up spec once a zone↔mesh or zone↔coordinate mapping is provided.

---

## Locked File Structure

### Python backend
- `trainer/demo_api/__init__.py`: package marker only.
- `trainer/demo_api/contracts.py`: Pydantic API and sequence contracts.
- `trainer/demo_api/scenario_adapter.py`: shared access to existing Gradio scenarios.
- `trainer/demo_api/sequence.py`: AgentPipeline → sequence-update translation.
- `trainer/demo_api/chat_service.py`: demo/live execution as typed event streams.
- `trainer/demo_api/app.py`: FastAPI routes, CORS, CLI entry point.
- `trainer/demo_api/tests/`: backend unit and API tests.
- `trainer/scripts/demo_app.py`: retained Gradio app; only imports shared scenario logic.

### Shared web/Electron frontend
- `demo-ui/package.json`: isolated frontend/Electron dependencies and scripts.
- `demo-ui/src/contracts.ts`: Zod schemas and inferred TypeScript types.
- `demo-ui/src/api.ts`: health, sequence, and POST-SSE client.
- `demo-ui/src/sequenceState.ts`: deterministic sequence reducer.
- `demo-ui/src/components/`: small UI components.
- `demo-ui/src/App.tsx`, `demo-ui/src/styles.css`: desktop composition and visual system.
- `demo-ui/electron/main.ts`, `demo-ui/electron/preload.ts`: secure Electron shell.
- `demo-ui/tests/`: Vitest unit/component tests.
- `demo-ui/e2e/`: Playwright browser tests.

---

### Task 1: Lock Existing Gradio Behavior and Extract Shared Scenario Access

**Files:**
- Create: `trainer/demo_api/__init__.py`
- Create: `trainer/demo_api/scenario_adapter.py`
- Create: `trainer/demo_api/tests/test_scenario_adapter.py`
- Modify: `trainer/scripts/demo_app.py:584-639`

- [ ] **Step 1: Write characterization tests for the sequence scenario**

```python
# trainer/demo_api/tests/test_scenario_adapter.py
from trainer.demo_api.scenario_adapter import classify_question, get_demo_scenario


def test_sequence_question_returns_scene_source_data() -> None:
    assert classify_question("다음에 어떤 구역을 칠해야 해?") == "sequence"
    scenario = get_demo_scenario("다음에 어떤 구역을 칠해야 해?")
    assert scenario["tool_calls"][0]["name"] == "sequence_validation"
    assert scenario["xr_commands"]["next_path"]["current_zone"] == "A-1"
    assert scenario["xr_commands"]["next_path"]["next_zone"] == "A-2"


def test_unknown_question_uses_general_scenario() -> None:
    scenario = get_demo_scenario("지원하지 않는 자유 질문")
    assert scenario["tool_calls"] == []
    assert "구체적인 정보" in scenario["response"]
```

- [ ] **Step 2: Run the tests and verify the missing module failure**

Run: `python3 -m pytest trainer/demo_api/tests/test_scenario_adapter.py -q`

Expected: collection fails with `ModuleNotFoundError: No module named 'trainer.demo_api.scenario_adapter'`.

- [ ] **Step 3: Add a narrow adapter over the existing scenario table**

```python
# trainer/demo_api/scenario_adapter.py
from __future__ import annotations

from typing import Any


def classify_question(text: str) -> str:
    from trainer.scripts.demo_app import _classify_question
    return _classify_question(text)


def get_demo_scenario(question: str) -> dict[str, Any]:
    from trainer.scripts.demo_app import DEMO_SCENARIOS
    scenario_type = classify_question(question)
    return DEMO_SCENARIOS.get(scenario_type, DEMO_SCENARIOS["general"])
```

Keep `trainer/demo_api/__init__.py` empty. The adapter deliberately avoids moving the large scenario fixture and proves Gradio behavior remains the source of truth.

- [ ] **Step 4: Make `_demo_stream` consume the adapter without changing output**

Replace direct classification/table lookup at `demo_app.py:588-590` with:

```python
from trainer.demo_api.scenario_adapter import get_demo_scenario

scenario = get_demo_scenario(question)
xr_json = _build_xr_json(scenario)
```

Place the import with the other module imports, not inside `_demo_stream`.

- [ ] **Step 5: Run characterization and existing contract tests**

Run: `python3 -m pytest trainer/demo_api/tests/test_scenario_adapter.py trainer/scripts/test_tool_calling.py trainer/scripts/test_response_contract.py -q`

Expected: all tests pass; existing suite still reports only its prior two deprecation warnings.

---

### Task 2: Define Typed Sequence and Stream Contracts

**Files:**
- Create: `trainer/demo_api/contracts.py`
- Create: `trainer/demo_api/tests/test_contracts.py`

- [ ] **Step 1: Write contract parsing tests**

```python
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
```

- [ ] **Step 2: Verify tests fail because contracts do not exist**

Run: `python3 -m pytest trainer/demo_api/tests/test_contracts.py -q`

Expected: import failure for `trainer.demo_api.contracts`.

- [ ] **Step 3: Implement strict Pydantic contracts**

```python
# trainer/demo_api/contracts.py
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


ZoneStatus = Literal["current", "next", "completed", "pending"]


class SequenceZone(StrictModel):
    zone_id: Annotated[str, Field(min_length=1)]
    label: Annotated[str, Field(min_length=1)]
    status: ZoneStatus = "pending"


class SequenceSnapshot(StrictModel):
    level_id: Annotated[str, Field(min_length=1)]
    label: Annotated[str, Field(min_length=1)]
    zones: list[SequenceZone]
    current_zone_id: str
    next_zone_ids: list[str]
    completed_zone_ids: list[str]

    @model_validator(mode="after")
    def zones_reference_each_other(self) -> "SequenceSnapshot":
        ids = {zone.zone_id for zone in self.zones}
        if self.current_zone_id not in ids:
            raise ValueError("current_zone_id must exist in zones")
        for group in (self.next_zone_ids, self.completed_zone_ids):
            for zone_id in group:
                if zone_id not in ids:
                    raise ValueError(f"unknown zone_id: {zone_id}")
        return self


class SequenceUpdate(StrictModel):
    type: Literal["update_sequence"] = "update_sequence"
    level_id: str
    current_zone_id: str
    next_zone_ids: list[str]
    completed_zone_ids: list[str]

    @model_validator(mode="after")
    def zones_are_unique(self) -> "SequenceUpdate":
        if len(self.next_zone_ids) != len(set(self.next_zone_ids)):
            raise ValueError("next_zone_ids must be unique")
        return self


class ChatRequest(StrictModel):
    message: Annotated[str, Field(min_length=1, max_length=4000)]
    session_id: Annotated[str, Field(min_length=1)]
    mode: Literal["demo", "live"] = "demo"
    level_id: Annotated[str, Field(min_length=1)] = "SP0035-L01"


class StreamEvent(StrictModel):
    event: Literal[
        "session", "status", "message_delta", "tool_result",
        "sequence_update", "complete", "error",
    ]
    data: dict[str, object]
```

- [ ] **Step 4: Run contract tests**

Run: `python3 -m pytest trainer/demo_api/tests/test_contracts.py -q`

Expected: 3 tests pass.

---

### Task 3: Translate AgentPipeline Output into Sequence State

**Files:**
- Create: `trainer/demo_api/sequence.py`
- Create: `trainer/demo_api/tests/test_sequence.py`

- [ ] **Step 1: Write translation tests**

```python
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
```

- [ ] **Step 2: Run and verify failures**

Run: `python3 -m pytest trainer/demo_api/tests/test_sequence.py -q`

Expected: import failure for `trainer.demo_api.sequence`.

- [ ] **Step 3: Implement deterministic translation without parsing prose**

`sequence_update_from_payload()` uses only:
- structured `current_zone` / `next_zone` fields from the AgentPipeline `next_path` payload
- exact zone IDs at the start of each `steps[i]["action"]` string, matched by prefix `<zone_id> `

It must never fabricate zone IDs or infer coordinates. Return `None` when `current_zone` is missing.

`default_snapshot(level_id)` returns the deterministic three-zone snapshot used by the initial GET response (A-1 completed, A-2 next, A-3 pending).

- [ ] **Step 4: Run sequence tests**

Run: `python3 -m pytest trainer/demo_api/tests/test_sequence.py trainer/demo_api/tests/test_contracts.py -q`

Expected: all tests pass.

---

### Task 4: Build Typed Demo/Live Chat Event Streaming

**Files:**
- Create: `trainer/demo_api/chat_service.py`
- Create: `trainer/demo_api/tests/test_chat_service.py`

- [ ] **Step 1: Write an async demo stream test**

```python
import pytest

from trainer.demo_api.chat_service import ChatService
from trainer.demo_api.contracts import ChatRequest


@pytest.mark.asyncio
async def test_sequence_demo_emits_update_before_complete() -> None:
    request = ChatRequest(message="다음에 어떤 구역을 칠해야 해?", session_id="s-1")
    events = [event async for event in ChatService().stream(request)]
    names = [event.event for event in events]
    assert names[0] == "session"
    assert "sequence_update" in names
    assert names[-1] == "complete"
    update = next(event for event in events if event.event == "sequence_update")
    # Actual "sequence" scenario in demo_app.py has step 3 "A-3 ... 대기",
    # so next_zone_ids preserves both zones. Do NOT trim.
    assert update.data["next_zone_ids"] == ["A-2", "A-3"]
    assert update.data["current_zone_id"] == "A-1"
    assert update.data["completed_zone_ids"] == ["A-1"]


@pytest.mark.asyncio
async def test_live_without_runner_emits_error() -> None:
    request = ChatRequest(message="hi", session_id="s-1", mode="live")
    events = [event async for event in ChatService().stream(request)]
    assert events[-1].event == "error"
    assert events[-1].data["retryable"] is False
```

- [ ] **Step 2: Run and verify the missing service failure**

Run: `python3 -m pytest trainer/demo_api/tests/test_chat_service.py -q`

Expected: import failure for `trainer.demo_api.chat_service`.

- [ ] **Step 3: Implement demo events from shared scenarios**

`ChatService.stream()` must emit typed `StreamEvent` values in this order:

1. `session`
2. one `status` for RAG
3. zero or more `tool_result`
4. optional `sequence_update`
5. one or more `message_delta`
6. `complete`

Use `get_demo_scenario()` and `sequence_update_from_payload()`. Split response text into fixed 24-character chunks; do not sleep in service code. Keep live execution behind a constructor-injected callable so tests do not load a model:

```python
LiveRunner = Callable[[ChatRequest], dict[str, object]]

class ChatService:
    def __init__(self, live_runner: LiveRunner | None = None) -> None:
        self.live_runner = live_runner

    async def stream(self, request: ChatRequest) -> AsyncIterator[StreamEvent]: ...
```

If live mode is requested without a runner, emit one `error` event with `retryable: false` and stop.

- [ ] **Step 4: Add a live-runner adapter that imports AgentPipeline lazily**

Cache the pipeline in a module-level singleton, matching current Gradio behavior. Use environment variables `CONFIG_PATH`, `ADAPTER_PATH`, and `MCP_BASE_URL`; do not duplicate model defaults in the frontend.

- [ ] **Step 5: Run service tests**

Run: `python3 -m pytest trainer/demo_api/tests/test_chat_service.py -q`

Expected: all tests pass without GPU or MCP.

---

### Task 5: Expose FastAPI Health, Sequence, and POST-SSE Routes

**Files:**
- Create: `trainer/demo_api/app.py`
- Create: `trainer/demo_api/tests/test_app.py`

- [ ] **Step 1: Write API tests**

```python
from fastapi.testclient import TestClient

from trainer.demo_api.app import create_app


def test_health_and_sequence_are_available() -> None:
    client = TestClient(create_app())
    assert client.get("/api/health").json()["status"] == "ok"
    body = client.get("/api/sequence/SP0035-L01").json()
    assert body["level_id"] == "SP0035-L01"
    assert body["current_zone_id"] == "A-1"


def test_chat_returns_sse_sequence_update() -> None:
    client = TestClient(create_app())
    with client.stream("POST", "/api/chat", json={
        "message": "다음에 어떤 구역을 칠해야 해?",
        "session_id": "s-1",
        "mode": "demo",
        "level_id": "SP0035-L01",
    }) as response:
        body = "".join(response.iter_text())
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: sequence_update" in body
    assert '"next_zone_ids":["A-2"]' in body
```

- [ ] **Step 2: Run and verify route failures**

Run: `python3 -m pytest trainer/demo_api/tests/test_app.py -q`

Expected: import failure for `trainer.demo_api.app`.

- [ ] **Step 3: Implement an app factory and SSE serialization**

Required routes:

```python
GET  /api/health
GET  /api/sequence/{level_id}
POST /api/chat
```

Serialize each event exactly as:

```python
yield f"event: {event.event}\ndata: {event.model_dump_json()}\n\n"
```

Use `StreamingResponse(..., media_type="text/event-stream")`. Set `Cache-Control: no-cache` and `X-Accel-Buffering: no`. Allow CORS only from `DEMO_UI_ORIGINS`, defaulting to `http://127.0.0.1:5173,http://localhost:5173`.

- [ ] **Step 4: Add a typed CLI entry point**

`python3 -m trainer.demo_api.app --host 0.0.0.0 --port 8120` must call `uvicorn.run("trainer.demo_api.app:app", ...)`. Include argparse, logging, and invalid-port handling.

- [ ] **Step 5: Run backend gates and live HTTP manual check**

Run: `python3 -m pytest trainer/demo_api/tests trainer/scripts/test_tool_calling.py trainer/scripts/test_response_contract.py -q`

Then start the API and verify:

```bash
python3 -m trainer.demo_api.app --host 127.0.0.1 --port 8120
curl -s http://127.0.0.1:8120/api/health
```

Expected health body includes `{"status":"ok"}`.

---

### Task 6: Scaffold the Typed React/Vite Application and Contracts

**Files:**
- Create: `demo-ui/package.json`
- Create: `demo-ui/tsconfig.json`
- Create: `demo-ui/tsconfig.app.json`
- Create: `demo-ui/vite.config.ts`
- Create: `demo-ui/index.html`
- Create: `demo-ui/src/main.tsx`
- Create: `demo-ui/src/contracts.ts`
- Create: `demo-ui/tests/contracts.test.ts`

- [ ] **Step 1: Initialize isolated dependencies**

From `demo-ui/`:

```bash
npm install react react-dom zod
npm install -D typescript vite @vitejs/plugin-react @types/react @types/react-dom vitest jsdom @testing-library/react @testing-library/jest-dom @testing-library/user-event
```

Do not modify the root `package.json`; it currently serves an unrelated `codex` dependency.

- [ ] **Step 2: Add scripts and strict compiler settings**

```json
{
  "scripts": {
    "dev:web": "vite --host 127.0.0.1",
    "build:web": "tsc -b && vite build",
    "test": "vitest run",
    "test:watch": "vitest"
  }
}
```

Enable `strict`, `noUncheckedIndexedAccess`, `exactOptionalPropertyTypes`, and `noEmit` for app TypeScript. Set Vite `base: "./"` so packaged Electron can load the build.

- [ ] **Step 3: Write Zod contract tests**

```typescript
import { describe, expect, it } from 'vitest'
import { sequenceSnapshotSchema, sequenceUpdateSchema } from '../src/contracts'

describe('sequence contracts', () => {
  it('rejects duplicate next zones in update', () => {
    const parsed = sequenceUpdateSchema.safeParse({
      type: 'update_sequence', level_id: 'SP0035-L01', current_zone_id: 'A-1',
      next_zone_ids: ['A-2', 'A-2'], completed_zone_ids: ['A-1'],
    })
    expect(parsed.success).toBe(false)
  })

  it('parses a snapshot', () => {
    const parsed = sequenceSnapshotSchema.parse({
      level_id: 'SP0035-L01', label: 'Level 01',
      zones: [{zone_id:'A-1', label:'상부', status:'completed'}],
      current_zone_id: 'A-1',
      next_zone_ids: [], completed_zone_ids: ['A-1'],
    })
    expect(parsed.current_zone_id).toBe('A-1')
  })
})
```

- [ ] **Step 4: Implement matching schemas and inferred types**

Use `z.object(...).strict()` for every boundary object. Add `.superRefine()` uniqueness checks for `next_zone_ids`. Export `SequenceSnapshot`, `SequenceZone`, `SequenceUpdate`, and a discriminated `StreamEvent` type with `z.infer`.

- [ ] **Step 5: Run frontend type and contract tests**

Run: `cd demo-ui && npm test && npm run build:web`

Expected: tests pass and `dist/index.html` is created without TypeScript errors.

---

### Task 7: Implement Sequence State and Panel Component

**Files:**
- Create: `demo-ui/src/sequenceState.ts`
- Create: `demo-ui/src/components/SequencePanel.tsx`
- Create: `demo-ui/tests/sequenceState.test.ts`
- Create: `demo-ui/tests/SequencePanel.test.tsx`

- [ ] **Step 1: Test deterministic reducer**

```typescript
import { describe, expect, it } from 'vitest'
import { applySequenceUpdate } from '../src/sequenceState'

const zones = [
  {zone_id:'A-1', label:'상부', status:'pending' as const},
  {zone_id:'A-2', label:'중간', status:'pending' as const},
  {zone_id:'A-3', label:'하부', status:'pending' as const},
]

it('maps current, next, completed, and pending statuses', () => {
  const result = applySequenceUpdate(zones, {
    type:'update_sequence', level_id:'SP0035-L01',
    current_zone_id:'A-1', next_zone_ids:['A-2'], completed_zone_ids:['A-1'],
  })
  expect(result.map(zone => zone.status)).toEqual(['current', 'next', 'pending'])
})

it('leaves unknown zones untouched', () => {
  const result = applySequenceUpdate(zones, {
    type:'update_sequence', level_id:'SP0035-L01',
    current_zone_id:'Z-9', next_zone_ids:['Z-8'], completed_zone_ids:[],
  })
  expect(result.map(zone => zone.status)).toEqual(['pending', 'pending', 'pending'])
})
```

- [ ] **Step 2: Implement the pure reducer**

`applySequenceUpdate` returns a new `SequenceZone[]`. Priority when a zone appears in more than one group: `current` > `next` > `completed` > `pending`. Never mutate the input.

- [ ] **Step 3: Test the panel rendering**

```tsx
import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { SequencePanel } from '../src/components/SequencePanel'

it('shows badges for each zone status', () => {
  render(<SequencePanel snapshot={{
    level_id:'SP0035-L01', label:'SP0035 Level 01',
    zones:[
      {zone_id:'A-1', label:'상부', status:'completed'},
      {zone_id:'A-2', label:'중간', status:'current'},
      {zone_id:'A-3', label:'하부', status:'next'},
    ],
    current_zone_id:'A-2', next_zone_ids:['A-3'], completed_zone_ids:['A-1'],
  }} />)
  expect(screen.getByTestId('zone-A-1')).toHaveAttribute('data-status', 'completed')
  expect(screen.getByTestId('zone-A-2')).toHaveAttribute('data-status', 'current')
  expect(screen.getByTestId('zone-A-3')).toHaveAttribute('data-status', 'next')
})
```

- [ ] **Step 4: Implement the panel**

Render a labeled list. Each row shows `zone_id`, `label`, and a status badge with `data-status={status}`. Do not add any 3D or canvas code.

- [ ] **Step 5: Run unit and build gates**

Run: `cd demo-ui && npm test && npm run build:web`

Expected: reducer and panel tests pass; build succeeds.

---

### Task 8: Implement the POST-SSE Client, Chat, and Desktop Layout

**Files:**
- Create: `demo-ui/src/api.ts`
- Create: `demo-ui/src/components/ChatPanel.tsx`
- Create: `demo-ui/src/components/StatusBar.tsx`
- Create: `demo-ui/src/components/ToolLogPanel.tsx`
- Create: `demo-ui/src/App.tsx`
- Create: `demo-ui/src/styles.css`
- Create: `demo-ui/tests/api.test.ts`
- Create: `demo-ui/tests/App.test.tsx`

- [ ] **Step 1: Test SSE parsing across split network chunks**

```typescript
import { describe, expect, it } from 'vitest'
import { createSseParser } from '../src/api'

it('parses an event split across chunks', () => {
  const parser = createSseParser()
  expect(parser.push('event: message_delta\ndata: {"event":"message_')).toEqual([])
  const events = parser.push('delta","data":{"text":"안녕"}}\n\n')
  expect(events[0]?.event).toBe('message_delta')
})
```

- [ ] **Step 2: Implement fetch-based POST-SSE parsing**

`streamChat(request, signal)` POSTs JSON to `/api/chat`, validates every decoded event with Zod, and yields an async iterable. It must throw a typed `ApiError` for non-2xx responses, invalid event frames, and missing response bodies. Use `AbortSignal` for the stop button.

- [ ] **Step 3: Test the user flow with a mocked API**

Render `App`, return one sequence snapshot, submit the sequence example, then assert that the assistant text appears, that the sequence panel updates A-2 to status `next`, and that clicking stop aborts the active request.

- [ ] **Step 4: Implement the desktop composition**

```text
TopStatusBar
├─ SequencePanel (left)
├─ ChatPanel (center)
└─ ToolLogPanel (collapsible bottom)
```

Use a dark graphite palette with high-contrast neutral panels and orange sequence accents. Avoid gradients, oversized hero typography, pill-heavy controls, and decorative cards. Preserve visible keyboard focus and a minimum 44px pointer target for primary controls.

- [ ] **Step 5: Wire API events to UI state**

- `status` updates the top bar and tool log.
- `message_delta` appends only the new text.
- `tool_result` appends a collapsible structured record.
- `sequence_update` passes through `applySequenceUpdate`.
- `complete` returns the composer to idle.
- `error` preserves partial output and exposes retry.

- [ ] **Step 6: Run unit, component, and build gates**

Run: `cd demo-ui && npm test && npm run build:web`

Expected: all tests pass and production assets build.

---

### Task 9: Add a Secure Electron Shell

**Files:**
- Modify: `demo-ui/package.json`
- Create: `demo-ui/tsconfig.electron.json`
- Create: `demo-ui/electron/main.ts`
- Create: `demo-ui/electron/preload.ts`
- Create: `demo-ui/electron/preload.d.ts`
- Create: `demo-ui/tests/electronMain.test.ts`

- [ ] **Step 1: Install Electron development tooling**

Run from `demo-ui/`:

```bash
npm install -D electron concurrently cross-env wait-on
```

- [ ] **Step 2: Write a test around exported BrowserWindow options**

```typescript
import { describe, expect, it } from 'vitest'
import { browserWindowOptions } from '../electron/main'

it('isolates and sandboxes the renderer', () => {
  const options = browserWindowOptions('/tmp/preload.js')
  expect(options.webPreferences).toMatchObject({
    contextIsolation: true,
    nodeIntegration: false,
    sandbox: true,
  })
})
```

- [ ] **Step 3: Implement main process with secure official defaults**

Export `browserWindowOptions(preloadPath)` as a pure function for testing. `createWindow()` loads `VITE_DEV_SERVER_URL` only when set; otherwise `loadFile(path.join(__dirname, '../dist/index.html'))`. Deny unexpected navigation and window creation. Use `app.whenReady()`, recreate on macOS `activate`, and quit on `window-all-closed` outside macOS.

- [ ] **Step 4: Keep preload capability minimal**

```typescript
import { contextBridge } from 'electron'

contextBridge.exposeInMainWorld('desktop', Object.freeze({
  platform: process.platform,
  electronVersion: process.versions.electron,
}))
```

Do not expose filesystem, shell, process execution, or generic IPC methods.

- [ ] **Step 5: Add Electron scripts**

```json
{
  "scripts": {
    "build:electron": "tsc -p tsconfig.electron.json",
    "dev:electron": "concurrently -k \"vite --host 127.0.0.1\" \"wait-on http://127.0.0.1:5173 && npm run build:electron && cross-env VITE_DEV_SERVER_URL=http://127.0.0.1:5173 electron dist-electron/main.js\"",
    "start:electron": "npm run build:web && npm run build:electron && electron dist-electron/main.js"
  }
}
```

- [ ] **Step 6: Run tests, builds, and Electron smoke**

Run: `cd demo-ui && npm test && npm run build:web && npm run build:electron`

Then start the FastAPI demo API and run `npm run start:electron`. Expected: one window opens, DevTools console has no security warnings caused by app configuration, and the API health indicator becomes online.

---

### Task 10: Add Browser E2E Tests and Update Documentation

**Files:**
- Modify: `demo-ui/package.json`
- Create: `demo-ui/playwright.config.ts`
- Create: `demo-ui/e2e/chat-sequence.spec.ts`
- Create: `demo-ui/e2e/api-offline.spec.ts`
- Modify: `README.md`

- [ ] **Step 1: Install Playwright and configure servers**

Run: `cd demo-ui && npm install -D @playwright/test && npx playwright install chromium`

Configure Playwright to start both:

```text
python3 -m trainer.demo_api.app --host 127.0.0.1 --port 8120
npm run dev:web -- --port 5173
```

Set frontend `VITE_API_BASE_URL=http://127.0.0.1:8120` and reuse servers outside CI.

- [ ] **Step 2: Write the end-to-end sequence test**

```typescript
import { test, expect } from '@playwright/test'

test('sequence question updates chat and order panel', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByText('SP0035 Level 01')).toBeVisible()
  await page.getByPlaceholder('도장 관련 질문을 입력하세요').fill('다음에 어떤 구역을 칠해야 해?')
  await page.getByRole('button', { name: '전송' }).click()
  await expect(page.getByText('다음 도장 구역: A-2')).toBeVisible()
  await expect(page.getByTestId('zone-A-2')).toHaveAttribute('data-status', 'next')
})
```

- [ ] **Step 3: Test the API-offline path**

Stop the API before loading the page and assert the status bar shows offline plus a retry action. Do not assert on chat streaming when offline.

- [ ] **Step 4: Run all automated gates**

```bash
python3 -m pytest trainer/demo_api/tests trainer/scripts/test_tool_calling.py trainer/scripts/test_response_contract.py -q
cd demo-ui && npm test && npm run build:web && npm run build:electron && npx playwright test
```

Expected: all Python, TypeScript, build, and browser E2E gates pass.

- [ ] **Step 5: Manually verify all three surfaces**

1. `DEMO_MODE=true python3 trainer/scripts/demo_app.py` → submit the sequence example, confirm Gradio still streams the same answer and XR JSON.
2. Start the FastAPI API and React web app → submit the sequence question, confirm the chat answer and sequence panel update (A-1 completed → A-2 next).
3. Run Electron → repeat the same flow, inspect the renderer console for security warnings.
4. Stop the API and confirm React/Electron show offline status and retry without crashing.

- [ ] **Step 6: Document exact operator commands**

Update `README.md` with separate sections for Gradio legacy demo, FastAPI backend, React web demo, and Electron desktop demo. Add an explicit note that 3D visualization is deferred until a `zone_id ↔ mesh/coordinate` mapping is provided.

---

## Final Self-Review Checklist

- [x] Every included design requirement (1차) maps to a task above (Gradio Task 1, contracts Task 2, sequence translation Task 3, chat streaming Task 4, FastAPI Task 5, React scaffold Task 6, sequence UI Task 7, chat/desktop layout Task 8, Electron Task 9, E2E and docs Task 10).
- [x] Gradio is preserved and manually exercised (Task 1 keeps `demo_app.py`; Task 10 Step 5 re-runs it).
- [x] React web and Electron share one renderer build (Task 9 Electron loads the same `dist/index.html` produced by Task 6/8).
- [x] No 3D, GLB, FBX, or Blender code is introduced in this plan.
- [x] Sequence updates use structured zone IDs and never infer coordinates from prose (Task 3 Step 3).
- [x] Python and TypeScript boundary data are strictly validated (Task 2 Pydantic `extra="forbid"`, Task 6 Zod `.strict()`).
- [x] Electron renderer has no Node integration or generic IPC bridge (Task 9 Steps 3–4).
- [x] Browser and Electron surfaces are manually used, not only compiled (Task 10 Step 5).
