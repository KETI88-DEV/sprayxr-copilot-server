# AI Co-Pilot 응답 계약

**문서 버전:** 1.0-draft  
**작성일:** 2026-08-06  
**적용 범위:** 시퀀스 종료 판정, 사용자 Query 응답, 세션 종료 리포트

---

## 1. 목적

이 문서는 시뮬레이터가 받아야 하는 AI Co-Pilot 응답 3종을 정의한다.

1. 스프레이 Trigger ON → OFF 시퀀스마다 실행되는 도막 판정
2. 사용자가 Query를 전송했을 때 생성되는 답변
3. 세션 완료 후 생성되는 최종 도막 리포트

### 1.1 책임 분리

- 도막 상태, 두께, 6영역 분류, 처방 코드, 회복 가능성 및 통계값은 **도막 이상탐지·콘텐츠센터 모듈**이 결정한다.
- LLM은 `message`, `answer`, `guideline`, `llm_briefing`과 같은 **자연어 설명만 생성**한다.
- LLM은 콘텐츠센터가 산출한 상태, 수치, 분류 또는 처방 코드를 변경하거나 재계산하지 않는다.
- Query는 현재 세션 상태, 누적 시퀀스, 최근 도막 판정, RAG 문서 및 MCP 결과를 조합해 답변할 수 있다.

### 1.2 메시지 흐름

```text
POST /sessions/start
GET  /stream/anomalies/{session_id}

POST /telemetry/sequence
  → 시퀀스마다 도막 판정 실행
  → HTTP coating_feedback 반환
  → SSE coating_feedback 전달

POST /copilot/query
  → 순서·도막·자세·지식 질문에 답변

POST /sessions/complete
  → 최종 히트맵 및 세션 통계 확정
  → report_id 반환

GET  /sessions/{session_id}/reports/{report_id}
  → 최종 리포트 조회

POST /sessions/end
  → 세션 종료 및 SSE 연결 해제
```

---

## 2. 공통 HTTP 응답

### 2.1 APIResponse 래퍼

| 필드 | 타입 | 필수 | 설명 |
|---|---|:---:|---|
| `status` | String | Y | `SUCCESS` 또는 `ERROR` |
| `code` | Integer | Y | HTTP 상태 코드 |
| `message` | String | Y | 처리 결과 메시지 |
| `data` | Object 또는 Null | Y | 응답 본문. 일반 오류는 `null`, 비동기 작업 상태는 Object 가능 |

```json
{
  "status": "SUCCESS",
  "code": 200,
  "message": "Operation completed successfully",
  "data": {}
}
```

### 2.2 공통 응답 메타데이터

성공 응답의 `data`에는 다음 필드를 둔다.

| 필드 | 타입 | 필수 | 설명 |
|---|---|:---:|---|
| `request_id` | String | Y | HTTP 요청 추적 ID |
| `session_id` | String 또는 Null | Y | 세션 ID. 일반 지식 Query는 `null` 가능 |
| `user_id` | String | Y | 작업자 ID |
| `scenario_id` | String 또는 Null | Y | 시나리오 ID. 일반 지식 Query는 `null` 가능 |
| `generated_at` | String | Y | 응답 생성 시각, ISO 8601 UTC |
| `latency_ms` | Integer | Y | 요청 처리 시간(ms) |

### 2.3 공통 오류 샘플

```json
{
  "status": "ERROR",
  "code": 409,
  "message": "Session state conflict",
  "data": null
}
```

일반 요청 오류는 안정적인 HTTP 상태 코드와 메시지를 사용한다. 비동기 리포트 작업 자체의 실패는 상태 조회 성공과 구분하기 위해 `200` 응답의 `report_status="failed"`로 표현한다. 스택 트레이스, SQL 문장, 파일 경로, 모델 프롬프트 및 내부 서비스 URL은 반환하지 않는다.

---

## 3. 시퀀스 종료 도막 판정

### 3.1 실행 규칙

`POST /telemetry/sequence`를 수신할 때마다 다음 절차를 반드시 실행한다.

```text
시퀀스 수신
  → 중복 여부와 프레임 저장 확인
  → 누적 히트맵 기반 도막 판정
  → HTTP coating_feedback 반환
  → 동일한 coating_feedback을 SSE로 전달
```

- `status`가 `normal`인 시퀀스도 반드시 판정하고 결과를 반환한다.
- UI는 `status != normal`일 때만 경고를 표시한다.
- HTTP와 SSE는 동일한 `feedback_id`를 사용한다.
- 클라이언트는 `feedback_id`를 기준으로 HTTP/SSE 중복 표시를 방지한다.
- `sequence_id`는 세션 내에서 1부터 증가하는 버스트 순번이다.

### 3.2 `sequence_data_ack`

| 필드 | 타입 | 필수 | 설명 |
|---|---|:---:|---|
| `stored` | Boolean | Y | 시퀀스 저장 성공 여부 |
| `duplicate` | Boolean | Y | 이미 처리한 시퀀스의 재전송 여부 |
| `frames_received` | Integer | Y | 요청에서 수신한 프레임 수 |
| `frames_accepted` | Integer | Y | 검증 후 저장한 프레임 수 |

### 3.3 `coating_feedback`

| 필드 | 타입 | 필수 | 설명 |
|---|---|:---:|---|
| `feedback_id` | String | Y | HTTP/SSE 중복 제거용 고유 ID |
| `user_id` | String | Y | 작업자 ID |
| `scenario_id` | String | Y | 시나리오 ID |
| `session_id` | String | Y | 세션 ID |
| `sequence_id` | Integer | Y | 세션 내 버스트 순번 |
| `zone_id` | String | Y | 대표 판정 구역 ID |
| `map_num` | Integer | Y | Height map 번호 |
| `uv_bin` | Array[Integer] | Y | `[x_bin, y_bin]` 형식의 세그먼트 좌표 |
| `measured_at` | String | Y | 측정 시각, ISO 8601 UTC |
| `thickness_um` | Number | Y | 해당 세그먼트의 현재 누적 두께(µm) |
| `target_nDFT_um` | Number | Y | 목표 명목 건조도막 두께(µm) |
| `normal_band_um` | Array[Number] | Y | `[최솟값, 최댓값]` 정상 두께 범위(µm) |
| `status` | String | Y | `under_app`, `normal`, `over_app` |
| `zone_class` | String | Y | `under2`, `under1`, `normal`, `over1`, `over2`, `over3` |
| `under_frac` | Number | Y | 이번 버스트 경로 중 규격 미달 비율, 0~1 |
| `action` | String 또는 Null | Y | 권장 처방 코드. 정상은 `null` |
| `action_reachable` | Boolean 또는 Null | Y | 행동 조정만으로 회복 가능한지 여부. 정상은 `null` |
| `message` | String | Y | LLM이 생성한 설명 문장 |
| `provisional` | Boolean | Y | 추정값 여부. 누적 히트맵 실측은 `false` |
| `algorithm_version` | String | Y | 도막 판정 알고리즘 버전 |

### 3.4 상태 불변 규칙

| `zone_class` | `status` |
|---|---|
| `under2`, `under1` | `under_app` |
| `normal` | `normal` |
| `over1`, `over2`, `over3` | `over_app` |

`status`와 `zone_class`의 조합이 위 표와 다르면 서버는 계약 오류로 처리한다.

### 3.5 처방 코드

| 코드 | 설명 |
|---|---|
| `increase_dwell` | 해당 구역의 체류 시간 또는 반복 분사를 증가 |
| `adjust_distance` | 작업면과 스프레이건 사이의 이격거리 조정 |
| `adjust_spraytime` | 분사 지속 시간 조정 |

`action_reachable=false`이면 작업자 동작만으로 회복하기 어렵고 도료·압력·노즐 등 공정 설정 조정이 필요하다는 의미다.

### 3.6 HTTP 응답 샘플

```json
{
  "status": "SUCCESS",
  "code": 200,
  "message": "Sequence coating analysis completed",
  "data": {
    "request_id": "req-seq-0012",
    "session_id": "SESSION-001",
    "user_id": "W-017",
    "scenario_id": "1-A",
    "generated_at": "2026-08-06T14:31:04Z",
    "latency_ms": 186,
    "sequence_data_ack": {
      "stored": true,
      "duplicate": false,
      "frames_received": 527,
      "frames_accepted": 527
    },
    "coating_feedback": {
      "feedback_id": "feedback-SESSION-001-12",
      "user_id": "W-017",
      "scenario_id": "1-A",
      "session_id": "SESSION-001",
      "sequence_id": 12,
      "zone_id": "A-2",
      "map_num": 2,
      "uv_bin": [3, 1],
      "measured_at": "2026-08-06T14:31:03Z",
      "thickness_um": 118,
      "target_nDFT_um": 160,
      "normal_band_um": [160, 240],
      "status": "under_app",
      "zone_class": "under1",
      "under_frac": 0.8,
      "action": "increase_dwell",
      "action_reachable": true,
      "message": "하단 좌측 구역이 저도막입니다. 이 구역을 조금 더 오래 또는 반복 분사하세요.",
      "provisional": false,
      "algorithm_version": "coating-feedback-1.0"
    }
  }
}
```

### 3.7 정상 판정 샘플

```json
{
  "feedback_id": "feedback-SESSION-001-13",
  "user_id": "W-017",
  "scenario_id": "1-A",
  "session_id": "SESSION-001",
  "sequence_id": 13,
  "zone_id": "A-3",
  "map_num": 2,
  "uv_bin": [4, 2],
  "measured_at": "2026-08-06T14:32:11Z",
  "thickness_um": 182,
  "target_nDFT_um": 160,
  "normal_band_um": [160, 240],
  "status": "normal",
  "zone_class": "normal",
  "under_frac": 0.0,
  "action": null,
  "action_reachable": null,
  "message": "현재 구역의 누적 도막 두께는 정상 범위입니다.",
  "provisional": false,
  "algorithm_version": "coating-feedback-1.0"
}
```

### 3.8 SSE 샘플

SSE는 HTTP 응답의 `coating_feedback` 객체를 그대로 전달한다. `id:`와 본문의 `feedback_id`는 동일해야 한다.

```text
id: feedback-SESSION-001-12
event: coating_feedback
data: {"feedback_id":"feedback-SESSION-001-12","user_id":"W-017","scenario_id":"1-A","session_id":"SESSION-001","sequence_id":12,"zone_id":"A-2","map_num":2,"uv_bin":[3,1],"measured_at":"2026-08-06T14:31:03Z","thickness_um":118,"target_nDFT_um":160,"normal_band_um":[160,240],"status":"under_app","zone_class":"under1","under_frac":0.8,"action":"increase_dwell","action_reachable":true,"message":"하단 좌측 구역이 저도막입니다. 이 구역을 조금 더 오래 또는 반복 분사하세요.","provisional":false,"algorithm_version":"coating-feedback-1.0"}
```

---

## 4. Query 응답

### 4.1 처리 범위

Query는 다음 질문을 처리할 수 있다.

- 현재 작업 순서와 다음 작업 구역
- 권장 순서 이탈 여부
- 현재 또는 최근 도막 상태
- 도막 이상 원인과 권장 조치
- 현재 작업 자세
- 일반 도장 지식과 시방서 기준

순서·도막·자세 등 현재 훈련 상태를 묻는 Query에는 `session_id`가 필수다. 일반 지식 Query에는 `session_id`를 생략할 수 있다.

`use_rag=false` 또는 `retrieval_mode=disabled`는 문서 검색만 비활성화한다. 현재 세션 상태, 누적 시퀀스, 최근 도막 판정 및 MCP 결과는 계속 사용할 수 있다.

### 4.2 Query 요청 샘플

```json
{
  "session_id": "SESSION-001",
  "user_id": "W-017",
  "prompt": "현재 순서와 도막 상태 괜찮아?",
  "task_type": "error_analysis",
  "use_rag": true
}
```

`task_type`은 라우팅 힌트다. 하나의 Query가 순서와 도막처럼 여러 의도를 포함하면 서버는 복수 의도를 해석하고 여러 MCP 도구를 호출할 수 있다.

### 4.3 Query 응답 필드

| 필드 | 타입 | 필수 | 설명 |
|---|---|:---:|---|
| `answer` | String | Y | 질문에 대한 최종 답변 |
| `resolved_intents` | Array[String] | Y | 서버가 해석한 질문 의도 목록 |
| `context_status` | String | Y | `available`, `not_available` |
| `sequence_context` | Object 또는 Null | Y | 현재·예상·다음 구역과 순서 상태 |
| `coating_context` | Object 또는 Null | Y | 최근 도막 판정 근거 |
| `retrieval_used` | Boolean | Y | 실제 RAG 검색 실행 여부 |
| `citations` | Array[String] | Y | RAG 문서 ID. 미사용 시 빈 배열 |

`route`와 `tool_calls_made`는 공개 응답에 포함하지 않고 서버 운영 로그에만 기록한다.

### 4.4 세션 기반 Query 응답 샘플

```json
{
  "status": "SUCCESS",
  "code": 200,
  "message": "Copilot query completed",
  "data": {
    "request_id": "req-query-0042",
    "session_id": "SESSION-001",
    "user_id": "W-017",
    "scenario_id": "1-A",
    "generated_at": "2026-08-06T14:33:10Z",
    "latency_ms": 31240,
    "answer": "현재 작업 순서는 권장 흐름에 맞습니다. 다만 최근 시퀀스에서 중간 구역의 누적 두께가 118µm로 정상 범위 160~240µm보다 낮습니다. 해당 구역을 조금 더 오래 또는 반복 분사하세요.",
    "resolved_intents": [
      "sequence_status",
      "coating_status"
    ],
    "context_status": "available",
    "sequence_context": {
      "current_zone_id": "A-2",
      "expected_zone_id": "A-2",
      "next_zone_id": "A-3",
      "adherence_status": "on_track",
      "deviations": []
    },
    "coating_context": {
      "feedback_id": "feedback-SESSION-001-12",
      "sequence_id": 12,
      "status": "under_app",
      "zone_class": "under1",
      "thickness_um": 118,
      "normal_band_um": [160, 240],
      "action": "increase_dwell",
      "action_reachable": true
    },
    "retrieval_used": true,
    "citations": [
      "DOC-0421-C0007",
      "SPEC-4.2-COATING"
    ]
  }
}
```

### 4.5 일반 지식 Query 응답 샘플

```json
{
  "status": "SUCCESS",
  "code": 200,
  "message": "Copilot query completed",
  "data": {
    "request_id": "req-query-0043",
    "session_id": null,
    "user_id": "W-017",
    "scenario_id": null,
    "generated_at": "2026-08-06T14:34:05Z",
    "latency_ms": 18400,
    "answer": "저도막은 요구된 최소 건조도막 두께보다 실제 도막 두께가 낮은 상태입니다.",
    "resolved_intents": ["knowledge_search"],
    "context_status": "not_available",
    "sequence_context": null,
    "coating_context": null,
    "retrieval_used": true,
    "citations": ["SPEC-4.2-COATING"]
  }
}
```

---

## 5. 세션 종료 리포트

### 5.1 생성 규칙

- `POST /sessions/complete`는 최종 히트맵과 세션 통계를 확정하고 리포트 생성을 시작한다.
- 리포트 생성은 장시간 실행될 수 있으므로 `202 Accepted`와 `report_id`를 먼저 반환한다.
- 클라이언트는 `GET /sessions/{session_id}/reports/{report_id}`로 결과를 조회한다.
- 리포트가 완료된 후 `/sessions/end`를 호출해 세션 상태와 SSE 연결을 종료한다.
- 도막 수치와 분류는 콘텐츠센터 결과를 그대로 사용한다.
- LLM은 `guideline`과 `llm_briefing`의 자연어 표현만 생성한다.

### 5.2 Complete 접수 응답

| 필드 | 타입 | 필수 | 설명 |
|---|---|:---:|---|
| `report_id` | String | Y | 리포트 ID |
| `report_status` | String | Y | 접수 시 `pending` |

```json
{
  "status": "SUCCESS",
  "code": 202,
  "message": "Session report generation started",
  "data": {
    "request_id": "req-complete-0001",
    "session_id": "SESSION-001",
    "user_id": "W-017",
    "scenario_id": "1-A",
    "generated_at": "2026-08-06T14:40:00Z",
    "latency_ms": 48,
    "report_id": "report-SESSION-001",
    "report_status": "pending"
  }
}
```

### 5.3 완료 리포트 필드

| 필드 | 타입 | 필수 | 설명 |
|---|---|:---:|---|
| `report_id` | String | Y | 리포트 ID |
| `report_status` | String | Y | `pending`, `complete`, `failed` |
| `algorithm_version` | String | Y | 콘텐츠센터 리포트 알고리즘 버전 |
| `source_artifact_id` | String | Y | 분석에 사용한 최종 히트맵 ID |
| `coating_distribution_index` | Number | Y | 표준 도막 분포 지수 |
| `zone_summary` | Object | Y | 표준 6영역의 면적 비율 |
| `distribution` | Object | Y | 세그먼트 개수 분포 |
| `reachability` | Object | Y | 이상 세그먼트의 회복 가능성 개수 |
| `worst_zones` | Array[Object] | Y | 우선 개선 대상 세그먼트 |
| `root_cause` | Object | Y | 지배 원인과 설명력 |
| `guideline` | String | Y | 핵심 개선 가이드 |
| `llm_briefing` | String | Y | LLM 자연어 최종 브리핑 |

### 5.4 최종 리포트 수치 규칙

- `zone_summary`는 면적 비율이며 여섯 값의 합은 허용 오차 내에서 1.0이어야 한다.
- `distribution`은 세그먼트 개수이며 `below + in_band + above`가 전체 세그먼트 수다.
- `reachability`는 비정상 세그먼트만 집계한다.
- 다음 관계가 성립해야 한다.

```text
reachability.reachable + reachability.needs_process
= distribution.below + distribution.above
```

제공된 샘플은 `81 + 75 = 151 + 5 = 156`으로 위 규칙을 만족한다.

- `behavior_r2`, `process_r2`의 범위는 0~1이다.
- `coating_distribution_index`의 공식과 범위는 콘텐츠센터 알고리즘 정의서에서 별도로 버전 관리한다.
- `guideline`과 `llm_briefing`은 구조화된 수치를 변경하거나 새로운 판정값을 만들어서는 안 된다.

### 5.5 완료 리포트 샘플

```json
{
  "status": "SUCCESS",
  "code": 200,
  "message": "Session report completed",
  "data": {
    "request_id": "req-report-0001",
    "session_id": "SESSION-001",
    "user_id": "W-017",
    "scenario_id": "1-A",
    "generated_at": "2026-08-06T14:40:18Z",
    "latency_ms": 18120,
    "report": {
      "report_id": "report-SESSION-001",
      "report_status": "complete",
      "algorithm_version": "coating-session-report-1.0",
      "source_artifact_id": "heightmap-SESSION-001-final",
      "coating_distribution_index": 72.4,
      "zone_summary": {
        "under2": 0.31,
        "under1": 0.22,
        "normal": 0.24,
        "over1": 0.12,
        "over2": 0.07,
        "over3": 0.04
      },
      "distribution": {
        "below": 151,
        "in_band": 17,
        "above": 5
      },
      "reachability": {
        "reachable": 81,
        "needs_process": 75
      },
      "worst_zones": [
        {
          "map_num": 2,
          "uv_bin": [2, 3],
          "thickness_um": 56,
          "status": "under_app",
          "zone_class": "under2",
          "cause": "이동속도 과다·이격 과다",
          "action": "increase_dwell",
          "action_reachable": true
        }
      ],
      "root_cause": {
        "dominant": "operator_behavior",
        "behavior_r2": 0.44,
        "process_r2": 0.12,
        "note": "저도막 주 원인은 작업자 동작인 속도·거리·체류이며 일부 구역은 도료 세팅 병행이 필요합니다."
      },
      "guideline": "시나리오 1-A에서 저도막 세그먼트가 다수 확인되었습니다. 주요 개선 방법은 부족 구역을 더 오래 또는 반복 분사하는 것입니다. 행동 조정으로 회복 가능한 구역과 도료 세팅 조정이 필요한 구역을 구분해 작업하세요.",
      "llm_briefing": "이번 세션은 저도막 영역이 넓게 분포했습니다. 특히 두께가 낮은 구역은 이동 속도와 이격거리 조정이 우선적으로 필요합니다. 행동 조정만으로 회복하기 어려운 구역은 도료 설정과 분사 조건을 함께 점검하세요."
    }
  }
}
```

### 5.6 리포트 실패 샘플

리포트 생성 실패는 같은 `report_id`에 대해 안정적인 실패 상태를 반환한다.

```json
{
  "status": "SUCCESS",
  "code": 200,
  "message": "Session report status retrieved",
  "data": {
    "request_id": "req-report-0002",
    "session_id": "SESSION-001",
    "user_id": "W-017",
    "scenario_id": "1-A",
    "generated_at": "2026-08-06T14:40:20Z",
    "latency_ms": 18234,
    "report_id": "report-SESSION-001",
    "report_status": "failed",
    "error_code": "REPORT_GENERATION_FAILED",
    "retryable": true
  }
}
```

---

## 6. 중복·재전송 규칙

- 모든 POST 요청은 `Idempotency-Key` 헤더를 사용한다.
- 같은 `Idempotency-Key`와 같은 요청 본문을 재전송하면 최초 응답을 그대로 반환한다.
- 같은 키에 다른 본문을 보내면 `409 Conflict`를 반환한다.
- `(session_id, sequence_id)`는 세션 내에서 유일해야 한다.
- 중복 시퀀스는 다시 저장하거나 다시 판정하지 않고 기존 `feedback_id`를 반환한다.
- HTTP 응답과 SSE 이벤트가 모두 도착하더라도 UI는 `feedback_id`당 한 번만 표시한다.
- Query 재전송은 동일한 `request_id` 또는 idempotency 기록으로 중복 생성을 방지한다.
- 세션 완료 요청을 재전송하면 새 리포트를 만들지 않고 기존 `report_id`를 반환한다.

---

## 7. SSE 연결 규칙

- SSE event name은 `coating_feedback`을 사용한다.
- 이벤트 `id:`는 `feedback_id`와 동일하다.
- 정상과 이상 판정 모두 전달한다. UI 경고 여부는 `status`로 판단한다.
- 재연결 시 `Last-Event-ID`를 사용해 누락 이벤트를 재전송할 수 있어야 한다.
- 서버는 heartbeat를 주기적으로 전송한다.
- 세션이 종료되면 스트림을 닫는다.
- SSE 이벤트에는 원시 스켈레톤, 전체 프레임, CSV 또는 이미지 데이터를 포함하지 않는다.

---

## 8. 확정이 필요한 외부 규칙

다음 항목은 API 구현 전에 담당 기관과 값 또는 공식을 확정해야 한다.

1. `coating_distribution_index`의 정확한 산출 공식과 유효 범위
2. `zone_class` 6영역별 두께 경계값
3. `action` 선택 규칙과 우선순위
4. `action_reachable` 판정 기준
5. `under_frac`의 공간 집계 기준과 분모
6. `uv_bin` 해상도와 좌표 원점
7. `map_num`과 최종 Height map artifact의 매핑 규칙
8. Query 및 최종 리포트의 목표 처리시간과 timeout
9. 리포트 보존 기간과 재생성 정책
