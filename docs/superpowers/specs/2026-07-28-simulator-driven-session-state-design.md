# 시뮬레이터 주도 세션 상태 설계

**Date**: 2026-07-28
**Status**: Approved (design) — 구현 전
**Scope**: 시뮬레이터 인터페이스가 보낸 값을 현재 상태의 유일한 소유자로 만들고, 좌측 도장 순서 패널과 채팅 `environment`가 둘 다 그 상태를 읽게 한다. 더미 입력을 세션 시작/런타임/세션 끝으로 분리 전송한다. 부수로 셸 레이아웃 폭 붕괴를 고친다.

---

## 1. 목표 / 비목표

### 목표
- 시뮬레이터가 보낸 값이 **모델이 받는 `environment`** 에 반영된다.
- 시뮬레이터가 보낸 값이 **좌측 도장 순서 패널**에 반영된다.
- 더미 입력을 **세션 시작 / 런타임 / 세션 끝** 버튼으로 분리해 보낼 수 있고, 각 단계마다 좌측 상태가 변한다.
- 상태 소유자는 서버다. CLI 드라이버·실제 Unity 클라이언트가 보내도 동일하게 반영된다.
- 채팅 말풍선·컴포저가 화면 밖으로 밀려나는 레이아웃 결함을 제거한다.

### 비목표
- 실제 Unity 클라이언트 연동, 인증, DB 영속화, 다중 동시 세션.
- `location_tracking` MCP 도구를 실제 구현으로 대체하는 일.
- `AgentPipeline.run()` 동작 변경. 평가 경로는 그대로 둔다.
- 3D 뷰어, 구역 메시 매핑.

---

## 2. 배경 — 확인된 단절

현재 코드에는 세 개의 경로가 있고 서로 연결되어 있지 않다.

```
/api/sim/*                →  MonitorRecorder (관측 전용)      ← 여기서 끝난다
/api/chat (live)          →  environment = default_tool_environment(level_id)
GET /api/sequence/{id}    →  default_snapshot(level_id)
```

- `trainer/demo_api/chat_service.py:135` — live runner가 매 턴 `default_tool_environment(request.level_id)`를 넘긴다.
- `trainer/demo_api/sequence.py:38` — `default_snapshot`은 A-1 완료 / A-2 다음 / A-3 대기, `current_zone_id="A-1"`로 **하드코딩**돼 있다.
- `trainer/demo_api/simulator.py` — `/api/sim/*`는 `MonitorRecorder`에만 기록한다. 다른 소비자가 없다.

결과: 시뮬레이터가 무엇을 보내도 모델은 매 턴 동일한 가짜 현재 상태를 받고, 좌측 패널도 변하지 않는다. 좌측 패널을 움직이는 유일한 경로는 채팅 응답의 `sequence_update`(모델의 도구 출력에서 파생)뿐이다.

---

## 3. 아키텍처

```
[더미 3버튼 | CLI 드라이버 | Unity(장래)]
        │  POST /api/sim/*
        ▼
  시뮬레이터 라우터  ──┬──▶ MonitorRecorder      (기존, 불변: I/O 관측)
                      └──▶ SimSessionStore      (신규: 현재 상태 소유)
                                  │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
GET /api/sequence/{id}  GET /api/sim/session   build_default_live_runner
→ 세션 파생 스냅샷       → 단계·이상·미매핑      → environment = 세션 파생
        │                     │                     │
        └──────────┬──────────┘                     ▼
                   ▼                        AgentPipeline (모델 컨텍스트)
          좌측 도장 순서 패널
```

`SimSessionStore`는 `MonitorRecorder`와 같은 단일 프로세스 in-memory 패턴이다(`threading.Lock`, 외부 의존 없음). `app.state`에 recorder와 나란히 보관하고 라우터에 주입한다.

---

## 4. 상태 모델

```python
@dataclass(frozen=True, slots=True)
class SimSessionState:
    session_id: str
    level_id: str
    phase: Phase                        # "start" | "runtime" | "session_end"
    recommended_flow: tuple[str, ...]   # ("A-1", "A-2", "A-3")
    current_zone_id: str
    completed_zone_ids: tuple[str, ...]
    last_event_id: str | None
    anomaly_count: int
    unresolved_zone_id: str | None      # flow에 없는 구역이 관측된 경우
    updated_at: str                     # ISO8601 UTC
```

**세션 슬롯은 하나다.** 새 `start`가 이전 세션을 대체한다. `session_end` 후에도 상태는 남아 있어 패널이 최종 상태를 계속 보여준다. 다중 동시 세션은 비목표이며, 슬롯이 하나여야 "채팅 컨텍스트가 어느 세션을 말하는가"가 모호해지지 않는다.

---

## 5. 전이 규칙

### 5.1 `start`
- `recommended_flow` = `default_snapshot(level_id).zones`의 `zone_id` 순서.
- `current_zone_id` = flow 첫 구역. `completed_zone_ids` = 빈 값.
- 새 훈련 세션은 아무것도 칠하지 않은 상태에서 시작하므로, `default_snapshot`이 A-1을 완료로 표시하는 것과 달리 **전 구역 미완료**로 출발한다.
- `phase = "start"`, `anomaly_count = 0`, `unresolved_zone_id = None`.

### 5.2 `runtime`
대상 구역(`target`) 결정은 다음 순서로 한 번만 시도한다.

1. 입력에 `zone_id`가 있으면 그 값.
2. 없고 `telemetry.coordinates`가 있으면 `ZoneResolver`로 해석(§6).
3. 둘 다 없으면 `target = None`.

`target` 적용:

| 조건 | 결과 |
|---|---|
| `target`이 `recommended_flow`에 있음 | `target` 이전 구역들을 `completed`로, `current_zone_id = target`, `unresolved_zone_id = None` |
| `target`이 flow에 없음 | 위치 **이동 없음**, `unresolved_zone_id = target` 기록 |
| `target is None` | 위치·완료 상태 변화 없음 |

- 스텁 출력이 `kind == "anomaly"`이면 `anomaly_count += 1`.
- `last_event_id = event_id`, `phase = "runtime"`.

### 5.3 `session_end`
- `phase = "session_end"`. 위치와 완료 목록은 그대로 둔다. 실제로 마지막 구역까지 칠했는지는 알 수 없으므로 임의로 완료 처리하지 않는다.

### 5.4 갱신 시점과 실패 입력
`_run_phase`는 검증을 통과하고 출력을 만든 뒤, 200을 반환하기 직전에 `store.apply(phase, payload, output)`을 호출한다. 입력과 출력을 모두 넘기는 이유는 위치가 입력(`zone_id`/`coordinates`)에서, `anomaly_count`가 출력(`kind == "anomaly"`)에서 오기 때문이다.

422로 거부된 요청(잘못된 JSON, `session_id` 불일치, 검증 실패)은 상태를 전혀 바꾸지 않는다. 모니터 기록(입력 1건 + 출력 1건)은 기존대로 남으므로, 거부된 요청도 관측은 가능하되 상태에는 영향이 없다.

---

## 6. ZoneResolver (`location_tracking` 폴백)

```python
class ZoneResolver(Protocol):
    async def resolve(self, coordinates: list[float]) -> str | None: ...
```

- 기본 구현은 `MCP_BASE_URL`의 `POST /location_tracking/identify`를 호출하고 `result.zone_id`를 반환한다. 타임아웃 2초.
- 실패·타임아웃·비정상 응답은 **조용히 `None`** 을 반환한다. 관측용 폴백이 세션 인터페이스를 깨뜨려서는 안 된다.
- 라우터에 주입되므로 테스트는 fake resolver를 쓰고 MCP를 필요로 하지 않는다.

**알려진 한계**: 현재 `mcp_servers/location_tracking.py`는 입력과 무관하게 항상 `zone_id: "B-3"`을 반환하는 stub이고, `B-3`는 SP0035 구역 맵(A-1/A-2/A-3)에 없다. 따라서 폴백 경로는 지금은 항상 `unresolved_zone_id = "B-3"`로 귀결되며 위치를 옮기지 않는다. 배선과 규칙은 완성해 두고, 도구가 실제 구현으로 대체되면 코드 변경 없이 동작한다. UI는 이 값을 "미매핑 구역"으로 표시해 사용자가 원인을 알 수 있게 한다.

---

## 7. 계약 변경

`SessionEventInput`에 선택 필드 하나를 더한다. 기존 페이로드와 호환되며 다른 봉투는 그대로다.

```python
class SessionEventInput(StrictModel):
    event_id: Annotated[str, Field(min_length=1)]
    user_request: str | None = None
    telemetry: EventTelemetry | None = None
    zone_id: str | None = None          # 신규
```

`demo-ui/src/simContracts.ts`의 Zod 스키마를 같은 커밋에서 함께 바꾼다. `SessionEventOutput`을 비롯한 출력 계약은 변경하지 않는다.

---

## 8. 소비자 배선

### 8.1 `GET /api/sequence/{level_id}`
- 활성 세션이 있고 `session.level_id == level_id`이면 세션 파생 스냅샷을 반환한다.
- 그 외에는 기존 `default_snapshot(level_id)`을 그대로 반환한다. 기존 테스트와 초기 화면이 유지된다.

세션 파생 스냅샷:
- `zones`: `recommended_flow` 순서. 상태는 `completed` → 완료 목록에 있는 구역, `current` → `current_zone_id`, 나머지 `pending`. 기존 fixture는 작업 대상 구역을 `next`로 표시하지만, 세션에서는 실제로 작업 중인 구역이므로 `current`(UI 배지 "현재")가 맞다.
- `label`은 기존 `default_snapshot`의 값을 재사용한다.

### 8.2 `GET /api/sim/session` (신규, read-only)
세션 메타데이터를 그대로 노출한다. 항상 200이며, 활성 세션이 없으면 `{"session": null}`이다.

```json
{"session": {"session_id": "...", "level_id": "SP0035-L01", "phase": "runtime",
             "current_zone_id": "A-2", "completed_zone_ids": ["A-1"],
             "last_event_id": "ev-1", "anomaly_count": 1,
             "unresolved_zone_id": null, "updated_at": "..."}}
```

`SequenceSnapshot`은 채팅의 `sequence_update`와 미러 관계인 공유 계약이므로 여기에 시뮬레이터 전용 필드를 섞지 않는다. 단계·이상 횟수·미매핑 구역은 이 엔드포인트가 전담한다.

### 8.3 채팅 `environment`
- `build_default_live_runner`가 활성 세션이 있으면 세션 파생 `SequenceToolEnvironment`를, 없으면 기존 `default_tool_environment(level_id)`를 넘긴다.
- 파생 규칙은 기존과 동일한 shape(`zone_map` / `current_zone_id` / `structure_type`)이며 `_TOOL_ZONE_METADATA`를 재사용한다. `is_painted`는 완료 목록 기준이다.
- `run()` 시그니처와 평가 경로는 건드리지 않는다. 경계 어댑터에서만 값이 달라진다.

---

## 9. UI 변경

### 9.1 더미 송신 3버튼
`더미 세션 송신` 하나를 `세션 시작` / `런타임` / `세션 끝`로 교체한다.

- 클라이언트가 세션 생명주기를 관리한다: `idle → started → ended`.
- `idle`에서는 `런타임`·`세션 끝` 비활성. `세션 시작`이 새 UUID를 만든다.
- `런타임`은 반복 가능하다. 클릭할 때마다 `event_id`가 증가하고 **`zone_id`로 다음 구역**을 보낸다. 다음 구역은 UI가 이미 들고 있는 도장 순서 스냅샷(`snapshot.zones`)에서 현재 구역의 바로 뒤를 고르고, 마지막 구역이면 그대로 유지한다. 그래야 클릭마다 위치가 실제로 움직인다.
- 첫 번째 런타임 클릭만 이탈 파라미터(`standoff_mm: 350`, `angle_deg: 55`)를 포함해 이상 감지 경로를 보여준다. 이후 클릭은 정상 파라미터다. 기존 `sendDummySession`의 동작을 보존한다.
- `세션 끝`이 `complete`를 보내고 `idle`로 되돌린다.

### 9.2 좌측 패널 갱신
- 모니터 스트림에서 새 이벤트를 받으면 `/api/sequence/{level_id}`와 `/api/sim/session`을 함께 다시 조회한다. 서버가 단일 소유자이므로 클라이언트가 전이 규칙을 복제하지 않는다.
- 클릭 1회당 이벤트 2건 규모이므로 디바운스는 두지 않는다.
- `SequencePanel`에 세션 단계(대기/세션 시작/런타임/세션 끝)와 짧은 세션 ID를 표시한다. `unresolved_zone_id`가 있으면 "미매핑 구역 B-3"을 함께 보여준다.
- 조회가 실패해도 패널은 마지막으로 성공한 값을 유지하고 도구 로그에 오류를 남긴다. 모니터와 마찬가지로 관측 경로가 콘솔을 깨뜨려서는 안 된다.

### 9.3 레이아웃 폭 붕괴 수정 (독립 선행 작업)
측정 결과: 뷰포트 1280px에서 `document.scrollWidth`가 3020px, `.sim-monitor`를 숨기면 1280px로 복귀한다.

원인은 `.sim-monitor__payload`가 `white-space: nowrap`인데 조상 flex 체인에 `min-width: 0`이 없고, `.app-shell` 그리드 열이 `minmax(0, 1fr)`이 아니라는 것이다. 긴 JSON 한 줄이 셸 전체 폭을 밀어내고, `justify-self: end`인 사용자 말풍선이 화면 밖(right=2988)에 놓인다.

수정:
- `.app-shell { grid-template-columns: minmax(0, 1fr); }`
- `.sim-monitor`, `.sim-monitor__scroll`, `.sim-monitor__phase`, `.sim-monitor__rows`, `.sim-monitor__row`에 `min-width: 0`.

이 수정만으로 말풍선 잘림과 컴포저 `초기화` 버튼 잘림이 함께 해결된다. 다른 두 작업과 독립이므로 먼저 커밋한다.

---

## 10. 테스트 전략

| 대상 | 방식 |
|---|---|
| `SimSessionStore` 전이 규칙 | 순수 단위 테스트. start/runtime/session_end, flow 밖 구역, `target is None`, 422 무변경 |
| `ZoneResolver` 폴백 | fake resolver 주입. 성공·`None` 반환·예외 각각. MCP 불필요 |
| 세션 파생 스냅샷 / environment | 순수 함수 테스트. 활성 세션 없음 → 기존 fixture와 동일함을 고정 |
| 라우터 배선 | `TestClient`로 `/api/sim/*` → `/api/sequence/{id}`, `/api/sim/session` 왕복. 세션 없음 → `{"session": null}`. SSE 경로는 건드리지 않음 |
| live runner environment | fake pipeline으로 넘어온 `environment` 값 검증 |
| 프런트 3버튼·생명주기 | Vitest. fetch mock으로 호출 URL·본문·비활성 상태 검증 |
| 좌측 패널 갱신 | Vitest. 모니터 이벤트 도착 시 스냅샷 재조회 발생 검증 |
| 레이아웃 | Playwright에서 `document.scrollWidth === clientWidth` 단언 |

기존 176 Python 테스트와 79 Vitest 테스트는 전부 통과 상태를 유지한다. 특히 활성 세션이 없을 때의 동작이 기존과 동일해야 한다.

---

## 11. 남는 것 (이번 범위 아님)

- 다중 동시 세션과 세션 영속화.
- `location_tracking` 실제 구현. 그전까지 좌표 폴백은 `unresolved_zone_id`로만 관측된다.
- 시뮬레이터 이벤트를 실제 `AgentPipeline`에 흘려 코칭 응답을 생성하는 일. 확장점은 `_run_phase(build_output=...)`에 그대로 남는다.
- 세션 상태의 SSE 푸시. 지금은 모니터 이벤트를 신호 삼아 스냅샷을 다시 조회한다.
