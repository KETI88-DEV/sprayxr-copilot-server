# Query·이상탐지 응답 계약 교정 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `docs/ai_copilot_response_contract.md`에 Query 4종, 이상탐지 SSE 3종, 방향 5종과 내부 trajectory HTTP·시뮬레이터 HTTP/SSE 경계 예시를 일관된 필드·enum·문체로 반영한다.

**Architecture:** 내부 trajectory coaching HTTP는 원시 지표와 숙련자 비교를 반환한다. 시뮬레이터 HTTP는 전체 결과를 `trajectory_feedback` 아래 namespaced하고, SSE는 이상이 있을 때만 정규화된 `anomaly_detection` 요약을 전달한다. Query는 기존 응답 구조를 유지하되 네 `task_type`별 요청·응답 예시를 분리한다.

**Tech Stack:** Markdown, JSON, SSE text frames, Python 3.11 표준 `json` 모듈(컨테이너 검증)

## Global Constraints

- 설계 정본은 `docs/superpowers/specs/2026-08-07-query-anomaly-response-contract-design.md`다.
- 시뮬레이터 외부 응답에 `feedback_bundle`, `sequence_feedback_bundle`, `source_mcp`, `region_id`, `combined`, DFT 용어를 사용하지 않는다.
- 위치는 `zone_id`와 `A-1`, `A-2`, `A-3`으로 통일한다.
- 도막 상태 enum은 `under_app | normal | over_app`이다.
- 경로 이상 enum은 `direction_error | trajectory_deviation | distance_speed_error`다.
- 방향 enum은 `horizontal | vertical | diagonal | curved_irregular | stationary`다.
- `anomaly_type`은 `coating`, `trajectory` 배열이며 복합 시 순서는 `coating`, `trajectory`다.
- `action`은 단일 코드 문자열이다.
- 안내 문구는 존댓말 기반의 중립적 시스템 안내체를 사용한다.
- `docs/Interface.pdf`는 검증 자료일 뿐 커밋하지 않는다.

---

### Task 1: 시퀀스 HTTP와 이상 SSE 경계 교정

**Files:**
- Modify: `docs/ai_copilot_response_contract.md`

**Interfaces:**
- Consumes: 승인 설계의 경계 원칙과 anomaly 상위 필드
- Produces: 시퀀스 HTTP `data`와 조건부 `anomaly_detection` SSE 계약

- [ ] **Step 1: 내부·외부 경계 문구 교정**

`feedback_bundle`은 내부 조합용이며 외부 HTTP/SSE에 직렬화하지 않는다고 명시한다. 시퀀스 HTTP `data`에는 `event_id`, `sequence_id`, `text_guide`, `anomaly`, `sequence_data_ack`와 선택 결과를 직접 둔다.

- [ ] **Step 2: anomaly 필드와 정렬 규칙 교정**

`anomaly.detected`, `anomaly_type`, `items`를 정의하고 `anomaly_type=["coating","trajectory"]`일 때 `items`도 도막 항목 다음 경로 항목으로 정렬한다고 명시한다.

- [ ] **Step 3: 시뮬레이터용 SSE 예시 3개 작성**

다음 예시를 각각 독립된 SSE frame으로 작성한다.

```json
{"anomaly_type":["coating"],"items":[{"axis":"thickness","type":"under_app","zone_id":"A-2","action":"increase_dwell"}]}
```

```json
{"anomaly_type":["trajectory"],"items":[{"axis":"trajectory","type":"trajectory_deviation","zone_id":"A-2","deviation_mm":45.8,"action":"stabilize_wrist"}]}
```

```json
{"anomaly_type":["coating","trajectory"],"items":[{"axis":"thickness","type":"under_app","zone_id":"A-2","action":"increase_dwell"},{"axis":"trajectory","type":"trajectory_deviation","zone_id":"A-2","deviation_mm":45.8,"action":"stabilize_wrist"}]}
```

- [ ] **Step 4: 경로 이상 3종 정의**

`direction_error`에는 `current_direction`, `expected_direction`; `trajectory_deviation`에는 `deviation_mm`; `distance_speed_error`에는 `distance_mm`, `target_distance_mm` 예시를 제공한다.

- [ ] **Step 5: 문체 확인**

세 경로 안내 문구를 설계 문서의 중립적 문장과 정확히 일치시킨다.

### Task 2: Query `task_type`별 예시 분리

**Files:**
- Modify: `docs/ai_copilot_response_contract.md`

**Interfaces:**
- Consumes: 기존 Query 응답 필드 구조
- Produces: 네 `task_type`별 요청 JSON과 응답 JSON

- [ ] **Step 1: Query enum과 공통 규칙 명시**

허용 예시를 `trajectory_recomm`, `pose_recomm`, `error_analysis`, `knowledge_search` 네 값으로 정리하고 `route`를 응답에 추가하지 않는다.

- [ ] **Step 2: `trajectory_recomm` 요청·응답 작성**

프롬프트는 `다음 도장 구역과 권장 이동 경로를 알려줘.`를 사용한다. 응답은 현재 `A-2`, 다음 `A-3`, 가로 일직선 패스 안내와 `sequence_context`를 포함한다.

- [ ] **Step 3: `pose_recomm` 요청·응답 작성**

프롬프트는 `현재 자세에서 교정할 부분을 알려줘.`를 사용한다. 응답은 현재 MCP 출력인 `expert_pose_id`, `similarity_score`, 관절별 `distance`, `angle_corrections`, `correction_text`, `summary`를 포함한다. 손떨림·몸통 기준 방향·누적 자세 평가를 생성하지 않는다.

- [ ] **Step 4: `error_analysis` 요청·응답 작성**

프롬프트는 `A-2 구역의 저도막 원인과 조치 방법을 알려줘.`를 사용한다. 응답은 `under_app`, `zone_id="A-2"`, `action="increase_dwell"` 근거를 포함한다.

- [ ] **Step 5: `knowledge_search` 요청·응답 작성**

프롬프트는 `저도막이 무엇인지 설명해줘.`를 사용한다. `session_id=null`, 세 context `null`, `retrieval_used=true`, 비어 있지 않은 `citations`를 사용한다.

### Task 3: trajectory 응답 경계와 방향 enum

**Files:**
- Modify: `docs/ai_copilot_response_contract.md`

**Interfaces:**
- Consumes: 승인 설계의 trajectory 방향·경계 정의
- Produces: 내부 분석 HTTP, 시뮬레이터 전체 HTTP, SSE 요약의 분리된 예시

- [ ] **Step 1: 방향 enum 5종 정의**

`trajectory_metrics.direction`의 허용값을 `horizontal`, `vertical`, `diagonal`, `curved_irregular`, `stationary`로 정의한다. `direction`은 기계 판정용이고 `stroke_pattern`은 사용자 표시용이라고 명시한다.

- [ ] **Step 2: 내부 trajectory coaching HTTP 작성**

내부 응답에는 `trajectory_metrics`, `expert_comparison`, `coaching_message` 전체를 둔다. 교정 예시는 `direction="horizontal"`, `straightness_score=72`, `expert_deviation_mm=45.8`, 목표·사용자 속도 `2800/2200`, 목표·사용자 거리 `400/450`을 사용한다.

- [ ] **Step 3: 시뮬레이터 HTTP 작성**

전체 trajectory 결과를 `data.trajectory_feedback` 아래에 두고, 최상위에는 `text_guide`, `sequence_data_ack`, 정규화 `anomaly`를 둔다. anomaly item은 `trajectory_deviation` 다음 `distance_speed_error` 순서로 정렬한다.

- [ ] **Step 4: 시뮬레이터 SSE 요약 작성**

SSE에는 `event_id`, `sequence_id`, `text_guide`, `anomaly`, `sequence_data_ack`만 두고 `trajectory_metrics`, `expert_comparison` 전체를 포함하지 않는다.

- [ ] **Step 5: 계산과 안내 문구 정합성 확인**

`speed_gap_mms = 2200 - 2800 = -600`, `distance_gap_mm = 450 - 400 = 50`을 유지하고 안내 문구가 속도 증가와 거리 감소를 지시하는지 확인한다.

### Task 4: 문서 계약 검증

**Files:**
- Verify: `docs/ai_copilot_response_contract.md`

**Interfaces:**
- Consumes: Task 1·2·3의 최종 문서
- Produces: 파싱·enum·정렬·금지어 검증 증거

- [ ] **Step 1: JSON과 SSE payload 파싱**

Run:

```bash
docker exec -w /workspace ship-paint-trainer python3 -c 'import json,re,pathlib; text=pathlib.Path("docs/ai_copilot_response_contract.md").read_text(); [json.loads(x) for x in re.findall(r"```json\n(.*?)\n```", text, re.S)]; [json.loads(line.removeprefix("data: ")) for line in text.splitlines() if line.startswith("data: {")]; print("json_sse_parse=ok")'
```

Expected: `json_sse_parse=ok`

- [ ] **Step 2: Query 4종과 anomaly 3종 검증**

Python 검증에서 네 `task_type` 요청 예시가 각각 존재하고 SSE payload의 `anomaly_type`이 `['coating']`, `['trajectory']`, `['coating','trajectory']` 순서로 존재하는지 확인한다.

- [ ] **Step 3: trajectory 경계와 방향 enum 검증**

내부 HTTP에 `trajectory_metrics`, `expert_comparison`이 존재하고 시뮬레이터 HTTP에서는 두 객체가 `trajectory_feedback` 아래에 있는지 확인한다. `SEQ-005` SSE payload에는 두 객체가 없는지 확인하고 방향 다섯 값과 gap 계산을 검증한다.

- [ ] **Step 4: 금지 용어와 공백 검사**

`source_mcp`, `region_id`, `combined`, 외부 bundle event, DFT 용어가 예시 payload에 없는지 검사하고 다음을 실행한다.

```bash
GIT_MASTER=1 git diff --check -- docs/ai_copilot_response_contract.md
```

Expected: exit code 0, no output.

- [ ] **Step 5: 문서만 원자적 커밋**

```bash
GIT_MASTER=1 git add docs/ai_copilot_response_contract.md
GIT_MASTER=1 git commit -m "docs(api): trajectory 응답 경계와 방향 enum 추가"
```

`docs/Interface.pdf`와 다른 작업 파일은 stage하지 않는다.
