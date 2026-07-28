# 시뮬레이터 인터페이스 I/O 모니터 설계

**Date**: 2026-07-27
**Status**: Approved (design) — 구현 전
**Scope**: Unity XR 시뮬레이터와 코파일럿 백엔드 사이 인터페이스의 input/output을 더미로 송수신 검증하고, 세션 시작/런타임/끝 3단계로 구분해 실시간 관측하는 read-only 모니터 콘솔.

---

## 1. 목표 / 비목표

### 목표
- 시뮬레이터 인터페이스의 **input(시뮬레이터→코파일럿)과 output(코파일럿→시뮬레이터)** 을 관측한다.
- **더미 페이로드로 송수신을 검증**한다(전송·수신·봉투·lifecycle 계약이 동작함을 증명).
- 모든 왕복을 **세션 시작 / 런타임 / 세션 끝** 3단계로 구분한다.
- GPU/LLM 없이 **결정론적**으로 동작해 pytest로 계약을 완전히 고정한다.

### 비목표 (이번 범위 아님)
- 실제 Unity 클라이언트 연동, 실제 코칭 LLM 추론.
- DGX 플랜 T13의 프로덕션 세션 상태머신·영속화(DB)·인증.
- 실시간 순서 이탈 지적, 자세 DTW 등 실제 분석 로직.
- 인터페이스를 실제 `AgentPipeline`에 연결(확장점만 남기고 미구현 — §13).

---

## 2. 배경 · 재사용 지점

- 전용 시뮬레이터 엔드포인트는 아직 없다. 현재 demo API는 `/api/health`, `/api/sequence/{level_id}`, `/api/chat`(SSE)뿐.
- 3단계 매핑은 `.omo/plans/dgx-spark-qwen-coaching-agent.md` §0.9에서 확정: 시작=순서 안내, 런타임=RAG+이상탐지, 끝=회고 리뷰.
- 기존 결정론 자산을 재사용한다:
  - `trainer/scripts/session_coaching.py` — `SessionSummary`, `SessionMetrics`, `compute_risk_flags()`(순수 결정론).
  - `trainer/demo_api/sequence.py` — `default_snapshot(level_id)`(zone 순서 → recommended_flow).
  - `trainer/demo_api/app.py` SSE 패턴, `demo-ui/src/api.ts` SSE 파서.

---

## 3. 아키텍처

```
[더미 드라이버: CLI 스크립트 | React "송신" 버튼]
        │  POST 더미 3단계 입력
        ▼
  FastAPI 시뮬레이터 인터페이스  /api/sim/*        ← GPU/LLM 없음, 결정론 스텁
    start → events(xN) → complete
        │  각 호출마다 input 1건 + output 1건 기록
        ▼
  MonitorRecorder  (bounded ring buffer, thread-safe, 단조 seq, anyio broadcast)
        │
        ▼
  read-only 모니터  /api/monitor/*
    GET /events?cursor  (스냅샷)   ·   GET /stream?cursor  (SSE: replay → live)
        │
        ▼
[운영자] React SimulatorMonitorPanel — 세션·단계(시작/런타임/끝)별 input↔output 타임라인
```

- 단일 프로세스 in-memory 링버퍼(`collections.deque(maxlen=500)` + `threading.Lock` + 단조 `seq`). 외부 의존성·DB 없음.
- 기존 3개 라우트는 불변. 신규는 `/api/sim/*`, `/api/monitor/*`로 네임스페이스 분리.

---

## 4. 인터페이스 계약 (3단계)

`Phase = "start" | "runtime" | "session_end"` (UI 라벨: 세션 시작 / 런타임 / 세션 끝).
백엔드는 Pydantic v2 strict(`extra="forbid"`), 프론트는 동형 Zod 스키마로 미러.

### 4.1 입력 계약

| 단계 | 엔드포인트 | 입력 모델 | 필드 |
|---|---|---|---|
| start | `POST /api/sim/sessions/start` | `SessionStartInput` | `session_id: str`, `level_id: str`, `worker: WorkerProfile`, `spec: PaintSpec` |
| runtime | `POST /api/sim/sessions/{session_id}/events` | `SessionEventInput` | `event_id: str`, `user_request: str \| None`, `telemetry: EventTelemetry \| None` |
| session_end | `POST /api/sim/sessions/{session_id}/complete` | `SessionSummary` (재사용) | `session_id, scenario_id, mode, zone_id, zone_type, metrics: SessionMetrics, env, timestamp` |

보조 모델:
- `WorkerProfile`: `{ trainee_id: str, skill_level: Literal["beginner","intermediate","advanced"], handedness: Literal["left","right"] }`
- `PaintSpec`: `{ nominal_um: float }`
- `GunMetrics`: `{ speed_m_per_s: float, standoff_mm: float, angle_deg: float }`
- `EventTelemetry`: `{ gun_metrics: GunMetrics \| None = None, pose_joints: dict[str, float] \| None = None, coordinates: list[float] \| None = None }`

경로의 `{session_id}`와 본문 `session_id`가 불일치하면 422로 거부(경로가 정본).

### 4.2 출력 계약 (결정론 스텁)

| 단계 | 출력 모델 | 필드 |
|---|---|---|
| start | `SessionStartOutput` | `session_id`, `recommended_flow: list[str]`, `flow_source: Literal["rule"]`, `guidance: str` |
| runtime | `SessionEventOutput` | `session_id`, `event_id`, `kind: Literal["rag","anomaly","ack"]`, `message: str`, `anomaly: dict \| None` |
| session_end | `SessionReviewOutput` | `session_id`, `summary: str`, `risk_flags: list[dict]`, `action_items: list[dict]`, `citations: list[str]` |

---

## 5. 결정론 출력 규칙 (LLM 없음)

- **start**: `recommended_flow = [z.zone_id for z in default_snapshot(level_id).zones]`, `flow_source="rule"`, `guidance`는 레벨·흐름을 채운 한국어 템플릿 문자열.
- **runtime** (입력 형태로 분기, 우선순위 순):
  1. `user_request`가 있으면 → `kind="rag"`, `message`는 요청을 반영한 스텁 답변 템플릿, `anomaly=None`.
  2. `telemetry.gun_metrics`가 있고 임계 이탈(`standoff_mm > 320` 또는 `angle_deg < 60`)이면 → `kind="anomaly"`, `anomaly={ "type": "spray_param_out_of_band", "standoff_mm", "angle_deg", "severity" }`.
  3. 그 외 → `kind="ack"`(telemetry 누적 확인), `message="이벤트 누적됨"`.
- **session_end**: `risk_flags = compute_risk_flags(summary.metrics)`(실제 결정론 재사용), `action_items`·`summary`는 플래그로부터 템플릿 생성, `citations=[]`(스텁).

임계값·템플릿 문구는 모듈 상수(`Final`)로 고정 → 테스트가 값 그대로 검증.

---

## 6. 모니터 (레코더 + SSE)

### 6.1 이벤트 봉투
`MonitorEvent = { seq: int, session_id: str, phase: Phase, direction: Literal["input","output"], timestamp: str(ISO-8601), payload: dict }`

- 인터페이스 호출 1건 → **input 1건 + output 1건** 기록(입출력 쌍이 타임라인에 명확).
- `seq`는 프로세스 단조 증가. `timestamp`는 UTC ISO.

### 6.2 레코더
- `collections.deque(maxlen=500)` + `threading.Lock` + 단조 `seq` 카운터.
- `record(session_id, phase, direction, payload) -> MonitorEvent`.
- `snapshot(cursor, limit) -> (events, next_cursor, dropped_before)` — `cursor` 이후 이벤트. 버퍼 최소 seq보다 cursor가 작으면 `dropped_before=<oldest_seq>`.
- **live 구독**: `anyio` memory object stream 기반 broadcast. 인터페이스 핸들러는 `async def`(이벤트 루프에서 실행)이므로 `record()`의 구독자 send-stream push는 루프 컨텍스트에서 안전하다(폴링·sleep 없음). 구독 해제 시 안전 정리.

### 6.3 엔드포인트
- `GET /api/monitor/events?cursor=0&limit=200` → `MonitorSnapshot { events, next_cursor, dropped_before }`. 초기 로드·테스트용.
- `GET /api/monitor/stream?cursor=0` → read-only SSE.
  - 프레임: `event: monitor\ndata: {MonitorEvent json}\n\n`.
  - 접속 시 `cursor` 이후 버퍼 이벤트를 replay → 이후 live 이벤트 forward.
  - cursor가 버퍼보다 오래됨 → 먼저 `event: gap\ndata: {"dropped_before": <seq>}\n\n` 1건 후 replay.

---

## 7. 더미 드라이버 (송수신 확인)

- **CLI**: `trainer/scripts/sim_dummy_driver.py` (PEP 723 인라인 메타 + 실행 주석). 한 세션 전체(start → events×N → complete)를 인터페이스에 순차 전송, 각 응답을 stdout 요약.
  - HTTP 클라이언트는 기존 bench 스크립트와 동일하게 `httpx` 사용(신규 의존성 미도입 — AGENTS.md "루트 의존성 워크플로 신설 금지" 준수; programming 스킬의 httpx2 기본값은 이 프로젝트 매니페스트가 httpx로 픽스한 것으로 간주).
  - 인자: `--api-url`, `--session-id`, `--events N`, `--anomaly`(이탈 telemetry 주입 토글).
- **UI 버튼**: 패널의 "더미 세션 송신" — 동일 3엔드포인트를 브라우저에서 호출(대화형 확인).

---

## 8. React 모니터 패널

- `demo-ui/src/components/SimulatorMonitorPanel.tsx` — `/api/monitor/stream` 구독, 세션별·단계별 input↔output 타임라인 렌더. 단계 색/라벨(세션 시작·런타임·세션 끝) 구분, input/output 방향 배지.
- `demo-ui/src/simContracts.ts` — `MonitorEvent`/`Phase`/스냅샷 Zod 스키마(§4·§6 미러).
- `demo-ui/src/simMonitorClient.ts` — 모니터 SSE 클라이언트(기존 `api.ts` SSE 파서 재사용) + 더미 세션 송신 함수(3콜).
- `demo-ui/src/App.tsx`에 **비파괴적으로** 패널/탭 추가(기존 채팅·순서 UI 유지).

---

## 9. 에러 처리

- 각 핸들러는 `Request` 원문 JSON을 읽어 **input 이벤트 먼저 기록** → Pydantic 파싱.
  - 파싱 성공 → 결정론 output 생성 → output 이벤트 기록 → 200 반환.
  - `ValidationError` → `{ "error": ... }` output 이벤트 기록 → 422 반환(실패도 관측 대상).
- 공통 헬퍼 `_run_phase(request, phase, model_cls, build_output)`로 3핸들러가 동일 규율 공유.
- 링버퍼 오버플로우 → 최오래건 드롭(bounded), 재접속 cursor 유실은 `gap`으로 통지.
- 모니터 경로는 상태 불변(read-only), 동시성 안전.
- GPU/LLM 부재 → 타임아웃·모델 실패 경로 없음.

---

## 10. 테스트 전략 (TDD · GPU-free · 결정론)

- **Unit(백엔드)**: 레코더(append/snapshot/cursor replay/overflow gap/동시성), 3 결정론 빌더(입력→고정 출력, 임계 분기), `_run_phase` 422 기록.
- **Integration(FastAPI TestClient)**: 전체 더미 세션(start→events×N→complete) → 기대 `MonitorEvent` 순서/개수(2×호출), `/monitor/events` 스냅샷 순서, `/monitor/stream` replay+live, 잘못된 입력 422+기록, 경로/본문 session_id 불일치 422.
- **Frontend(Vitest)**: Zod 스키마 경계, 픽스처 이벤트→단계별 타임라인 렌더, "송신" 버튼 3콜, SSE 파서 gap 처리.
- **E2E**: `sim_dummy_driver.py` 실행 + `curl /api/monitor/events`로 순서 검증. (선택) Playwright로 패널 렌더 확인.

---

## 11. 파일 계획 (250 pure LOC 상한 · 모듈 경계)

**백엔드**
- `trainer/demo_api/sim_contracts.py` — 입력/출력/MonitorEvent Pydantic (~130 LOC)
- `trainer/demo_api/monitor.py` — 레코더 + broadcast + snapshot (~120 LOC)
- `trainer/demo_api/simulator.py` — 3 핸들러 + `_run_phase` + 라우터 (~120 LOC)
- `trainer/demo_api/simulator_stub.py` — 결정론 출력 빌더(§5). `simulator.py`가 250 초과 위험 시 분리 (~100 LOC)
- `trainer/demo_api/app.py` — `include_router` 2줄 추가(sim, monitor). 기존 불변.
- `trainer/scripts/sim_dummy_driver.py` — PEP 723 CLI (~90 LOC)
- `trainer/demo_api/tests/test_monitor_recorder.py`, `test_simulator_interface.py`, `test_monitor_stream.py`

**프론트**
- `demo-ui/src/simContracts.ts`, `demo-ui/src/simMonitorClient.ts`, `demo-ui/src/components/SimulatorMonitorPanel.tsx`, `demo-ui/src/App.tsx` 배선
- `demo-ui/tests/simContracts.test.ts`, `simulatorMonitorPanel.test.tsx`

---

## 12. 경계 규율 (불변 항목)

- `/api/health`, `/api/sequence`, `/api/chat`와 `AgentPipeline.run()`/`run_streaming()`은 **손대지 않는다**.
- 신규 라우트는 `/api/sim/*`, `/api/monitor/*` 네임스페이스로 격리.
- `session_coaching.py`의 재사용은 **import만**(수정 없음). `compute_risk_flags`·`SessionSummary`는 그대로.
- 정확한 JSON/SSE 봉투를 경계에서 보존(AGENTS.md).

---

## 13. 확장점 (유예 — YAGNI)

- 인터페이스 핸들러의 output 빌더를 교체 가능한 seam으로 두어, 후속에 `?live=1`로 실제 `AgentPipeline` 라우팅을 붙일 자리만 남긴다. **이번엔 구현하지 않는다.**
- DB 영속화·인증(bearer)·프로덕션 세션 상태머신은 DGX T13 트랙으로 이관.

---

## 14. 확정된 결정 요약

1. 접근법 **A(결정론 스텁 인터페이스)** — GPU 불필요.
2. 서페이스 = 백엔드 인터페이스 + read-only 모니터 SSE + React 패널.
3. 모니터 기록 = **호출당 input+output 2건**.
4. 세션 끝 출력은 `compute_risk_flags` 재사용으로 의미 있는 결정론 리뷰.
5. 더미 드라이버 = CLI(테스트/자동화) + UI 버튼(대화형).
6. HTTP 클라이언트는 기존 코드와 동일 `httpx`(신규 의존성 없음).
