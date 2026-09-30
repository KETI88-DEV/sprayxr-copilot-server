# AI Co-Pilot SIM 연동 및 Spark 배포 계약 v2.5-Spark-1

- 문서 상태: 구현 및 인수시험 기준안
- 작성일: 2026-09-16
- 적용 대상: 도장 훈련 시뮬레이터, AI Co-Pilot API, NVIDIA DGX Spark 배포본
- 공개 API 기준 경로: `/api/v1`

## 1. 목적과 판정 기준

이 문서는 도장 SIM과 AI Co-Pilot 사이의 입력, 출력, 상태, 오류, 성능 및 배포 인수 기준을 하나로 정의한다. 구현 코드, 데모 UI, 예제 또는 이전 문서가 이 문서와 다르면 이 문서를 우선한다.

다음 조건을 모두 만족해야 **v2.5 제품 배포 완료**로 판정한다.

1. §5의 공개 엔드포인트가 모두 OpenAPI에 존재한다.
2. §6~§12의 요청과 응답 구조를 그대로 준수한다.
3. 세션 상태와 시퀀스 데이터가 Query 및 최종 리포트에 실제로 반영된다.
4. 도막, 순서, 자세 판정은 결정론적 엔진이 수행하고 LLM이 수치를 재계산하지 않는다.
5. 이상 알림 SSE와 비동기 리포트 흐름이 실제로 동작한다.
6. §16의 계약, 성능, 재기동 및 수동 인수시험을 통과한다.

모델이 로드되고 자유 대화가 가능하다는 사실만으로는 v2.5 배포 완료로 판정하지 않는다.

## 2. 계약의 적용 범위

### 2.1 포함

- 훈련 세션 시작과 종료
- Trigger ON→OFF 단위 90 FPS 시퀀스 업로드
- 도막, 작업 순서, 자세의 3축 판정
- 사용자 질의와 RAG 근거 제공
- 이상 발생 시 SSE 알림
- 훈련 종료 후 비동기 종합 리포트 생성과 조회
- DGX Spark에서 실행하는 vLLM, MCP, Agent API 및 웹 클라이언트

### 2.2 제외

- `/api/chat`: 내부 개발용 자유 대화 SSE
- `/api/sim/*`: 내부 시뮬레이터 데모 및 테스트 인터페이스
- `/api/monitor/*`: 개발·운영 관찰용 인터페이스
- 모델 학습, LoRA 생성 및 양자화 과정
- 도장 SIM 내부 DB와 렌더링 구현

제외 경로는 제품 UI가 호출하거나 외부 SIM 계약으로 노출해서는 안 된다.

## 3. 목표 아키텍처와 책임

```text
도장 SIM
  └─ HTTP/SSE /api/v1
       └─ Agent API (FastAPI, port 8120/container, 58000/host)
            ├─ Session Store
            ├─ Sequence Ingestor
            ├─ 필수 분석 인터페이스 5종
            │    ├─ coating_analysis
            │    ├─ anomaly_detection
            │    ├─ pose_analysis
            │    ├─ sequence_analysis
            │    └─ knowledge_search
            ├─ RAG: BGE-M3 + reranker + FAISS
            ├─ Report Job Store
            └─ OpenAI-compatible HTTP
                 └─ vLLM Qwen3.6-27B 병합 모델
```

Ollama, LangGraph 또는 특정 UI 프레임워크는 공개 계약의 일부가 아니다. 내부 구현은 교체할 수 있지만 `/api/v1`의 wire format과 동작은 유지해야 한다.

### 3.1 결정론적 엔진 책임

- 입력 데이터 검증과 단위 변환
- 도막 상태 및 이상 유형 판정
- 구역 방문 순서와 방향 판정
- 자세 유사도와 관절 교정량 산출
- 수치, 상태, 처방 코드 및 통계 생성
- 중복 시퀀스 검출과 세션 상태 전이

### 3.2 LLM 책임

- 판정 결과를 사용자 친화적인 자연어로 설명
- RAG 문서 근거를 이용한 지식 답변
- 이미 산출된 상태와 처방을 문맥에 맞게 요약
- 최종 리포트의 자연어 브리핑 생성

LLM은 도막 두께, 유사도, 이상 상태, 순서 판정 또는 처방 코드를 변경하거나 새로 계산하지 않는다.

### 3.3 제품 UI 책임

- §6~§12의 `/api/v1`만 호출
- 요청 ID와 세션 ID를 생성·보존
- 구조화된 context와 citations를 텍스트 답변과 별도로 표시
- SSE 재연결과 마지막 이벤트 ID 관리
- 리포트 상태를 2~5초 간격으로 조회
- 내부 단계명이나 모델 로딩 상태를 사용자에게 오해를 일으키는 문구로 표시하지 않음

## 4. 공통 규약

### 4.1 전송

- 문자 인코딩: UTF-8
- JSON Content-Type: `application/json`
- SSE Content-Type: `text/event-stream`
- 리포트 접수: `multipart/form-data`
- 시각: ISO 8601 UTC, 예: `2026-09-16T08:45:39Z`
- 좌표계: 모든 위치, 자세, 충돌점 및 히트맵은 동일한 월드 좌표계
- 공개 식별자: 레벨은 `TR-L01`, 구역은 `A-1`, `A-2`, `A-3`

### 4.2 공통 응답 래퍼

모든 일반 HTTP 응답은 다음 구조를 사용한다.

```json
{
  "status": "SUCCESS",
  "code": 200,
  "message": "Operation completed successfully",
  "data": {}
}
```

- `status`: `SUCCESS` 또는 `ERROR`
- `code`: 실제 HTTP 상태 코드와 동일한 정수
- `message`: 안정적인 사용자용 메시지
- `data`: 성공 데이터 또는 `null`

성공 `data`는 가능한 경우 다음 메타데이터를 포함한다.

```json
{
  "request_id": "req-0001",
  "session_id": "SESSION-001",
  "user_id": "USER-017",
  "scenario_id": "TR-L01",
  "generated_at": "2026-09-16T08:45:39Z",
  "latency_ms": 28450
}
```

### 4.3 오류

```json
{
  "status": "ERROR",
  "code": 422,
  "message": "Invalid request",
  "data": null
}
```

| HTTP | 의미 |
|---:|---|
| 400 | JSON 또는 multipart 형식 오류 |
| 404 | 세션, 리포트 또는 자원 없음 |
| 409 | 세션 상태 충돌 또는 식별자 충돌 |
| 422 | 필드 타입, 값, 단위 또는 계약 위반 |
| 503 | 모델 또는 필수 내부 서비스 사용 불가 |

스택 트레이스, 파일 경로, 프롬프트, 모델 경로, 내부 URL 및 인증 정보는 반환하지 않는다.

### 4.4 중복과 재전송

- `request_id`는 질의와 리포트 요청의 멱등 키다.
- `(session_id, sequence_id)`는 시퀀스 업로드의 멱등 키다.
- 동일 키와 동일 payload의 재전송은 기존 결과를 반환한다.
- 동일 키와 다른 payload는 HTTP 409를 반환한다.
- 서버는 이미 확정된 상태를 중복 요청으로 다시 전이하지 않는다.

## 5. 공개 엔드포인트

| 기능 | Method | Path | 응답 방식 |
|---|---|---|---|
| 세션 시작 | POST | `/api/v1/sessions/start` | JSON |
| 시퀀스 전송 | POST | `/api/v1/telemetry/sequence` | JSON |
| 질의 | POST | `/api/v1/copilot/query` | JSON |
| 이상 알림 | GET | `/api/v1/stream/anomalies/{session_id}` | SSE |
| 세션 종료 | POST | `/api/v1/sessions/end` | JSON |
| 분석 요청 | POST | `/api/v1/sessions/complete` | multipart/form-data |
| 분석 상태 | GET | `/api/v1/sessions/{session_id}/reports/{report_id}` | JSON |
| 분석 결과 | GET | `/api/v1/reports/{report_id}` | JSON |

## 6. 세션 시작

### 6.1 요청

`POST /api/v1/sessions/start`

```json
{
  "user_id": "USER-017",
  "scenario_id": "TR-L01",
  "session_id": "SESSION-001",
  "started_at": "2026-09-16T08:00:00Z",
  "worker": {
    "skill_level": "novice",
    "handedness": "right"
  }
}
```

| 필드 | 타입 | 필수 | 규칙 |
|---|---|:---:|---|
| `user_id` | String | Y | 비어 있지 않음 |
| `scenario_id` | String | Y | 현재 레벨은 `TR-L01` |
| `session_id` | String | Y | 세션 동안 불변 |
| `started_at` | String | Y | ISO 8601 UTC |
| `worker.skill_level` | String | Y | `expert` 또는 `novice` |
| `worker.handedness` | String | Y | `right` 또는 `left` |

### 6.2 응답

```json
{
  "status": "SUCCESS",
  "code": 200,
  "message": "Session started",
  "data": {
    "request_id": "session-start-SESSION-001",
    "session_id": "SESSION-001",
    "user_id": "USER-017",
    "scenario_id": "TR-L01",
    "generated_at": "2026-09-16T08:00:00Z",
    "latency_ms": 12,
    "recommended_flow": ["A-1", "A-2", "A-3"],
    "current_zone_id": "A-1"
  }
}
```

성공 이후에만 시퀀스, 질의, SSE 및 종료 요청을 허용한다.

## 7. 시퀀스 전송과 3축 판정

### 7.1 요청

`POST /api/v1/telemetry/sequence`

Trigger ON→OFF 한 번을 하나의 시퀀스로 정의한다. 프레임은 수집 순서대로 전송한다.

```json
{
  "user_id": "USER-017",
  "scenario_id": "TR-L01",
  "session_id": "SESSION-001",
  "zone_id": "A-1",
  "sequence_id": "SEQ-0001",
  "sequence_setting": {
    "tool_type": "spray_gun",
    "pressure_bar": 4.0,
    "nozzle_size_inch": 0.017,
    "thinning_ratio": 0.1,
    "viscosity_sec": 22.0,
    "flow_rate_ml_min": 320.0
  },
  "sequence_info": [
    {
      "index": 0,
      "time": "2026-09-16T08:01:00.000Z",
      "tool_pos_rot": {
        "pos": [0.1, 1.2, 0.5],
        "rot": [0.0, 90.0, 0.0]
      },
      "skeleton_joints": {
        "pelvis": {"pos": [0.0, 1.0, 0.0], "rot": [0.0, 0.0, 0.0]},
        "right_elbow": {"pos": [0.3, 1.4, 0.2], "rot": [4.0, 8.0, 2.0]}
      },
      "gun_metrics": {
        "speed_mms": 2200.0,
        "distance_mm": 450.0,
        "raypoint": [1.1, 1.3, 0.0],
        "raytexmapnum": 1,
        "raytexuv": [221, 144],
        "spraytime_ms": 11
      }
    }
  ],
  "sequence_result": {
    "avg_speed_mms": 2200.0,
    "avg_distance_mm": 450.0,
    "avg_thickness_um": 82.0,
    "cur_sequence_heatmap": [],
    "acc_sequence_heatmap": []
  }
}
```

### 7.2 처리 규칙

서버는 한 요청에서 다음 순서로 처리한다.

1. 세션, 구역, 단위, 프레임 순서 및 중복 여부 검증
2. 원본 또는 정규화된 시퀀스 저장
3. `coating_analysis`, `pose_analysis`, `sequence_analysis` 실행
4. 세 분석 결과를 `anomaly_detection`으로 조합
5. 세션 누적 상태 반영
6. HTTP 응답 생성
7. 이상이 있으면 동일 판정 결과로 SSE 발행

상태 전이는 HTTP/SSE 발행 전에 확정한다. HTTP와 SSE가 서로 다른 판정을 만들어서는 안 된다.

### 7.3 응답

```json
{
  "status": "SUCCESS",
  "code": 200,
  "message": "Sequence evaluated",
  "data": {
    "request_id": "sequence-SEQ-0001",
    "session_id": "SESSION-001",
    "user_id": "USER-017",
    "scenario_id": "TR-L01",
    "generated_at": "2026-09-16T08:01:01Z",
    "latency_ms": 185,
    "sequence_id": "SEQ-0001",
    "sequence_data_ack": {
      "stored": true,
      "duplicate": false,
      "frames_received": 1,
      "frames_accepted": 1
    },
    "coating_feedback": {
      "status": "under_app",
      "zone_id": "A-1",
      "thickness_um": 82.0,
      "nominal_um": 100.0,
      "action_codes": ["increase_dwell", "adjust_distance"]
    },
    "sequence_feedback": {
      "status": "normal",
      "current_zone_id": "A-1",
      "next_zone_id": "A-2",
      "current_direction": "horizontal",
      "expected_direction": "horizontal",
      "deviation_mm": 12.0
    },
    "pose_feedback": {
      "analysis_status": "available",
      "similarity_score": 0.82,
      "correction_mode": "joint_only",
      "corrections": [
        {
          "joint": "right_elbow",
          "label": "오른쪽 팔꿈치가 숙련자 통상 범위를 벗어났습니다.",
          "target_validated": false,
          "angle_delta_deg": null,
          "direction": null
        }
      ]
    },
    "text_guide": "이격거리를 줄이고 A-1 구역의 체류 시간을 늘리세요."
  }
}
```

판정할 데이터가 부족하면 임의 값을 만들지 않고 `analysis_status="not_available"`과 안정적인 `reason_code`를 반환한다.

### 7.4 구현 메모 (2026-09-30)

`trainer/demo_api/telemetry_router.py`. **응답 `data`는 §7.3의 12키와 정확히 같다** — 필드를 더하지 않는다. 시험이 키 집합을 단언하므로 하나라도 늘면 실패한다.

**`analysis_status`·`reason_code`를 세 feedback에 같은 방법으로 넣는다.** §7.3 말미가 "판정할 데이터가 부족하면 임의 값을 만들지 않고 `analysis_status="not_available"`과 안정적인 `reason_code`를 반환한다"고 규정하므로, `pose_feedback`만이 아니라 `coating_feedback`·`sequence_feedback`도 같은 두 필드로 상태를 말한다. 엔진 미구현은 `reason_code: "engine_not_implemented"`다.

**엔진 상태.** `pose_analysis`만 실제로 돈다. `coating_analysis`·`sequence_analysis`는 미구현이고, §7.2의 4·7단계(`anomaly_detection` 조합과 SSE 발행)도 아직 없다. 자세는 MCP `/pose_estimation/sequence`로 HTTP 위임한다.

**엔진 장애는 수신을 실패로 만들지 않는다.** MCP가 HTTP 500이거나 `status != "success"`면 200 + `pose_feedback.reason_code: "engine_unavailable"`이고 시퀀스는 저장된다.

**`sequence_data_ack`는 관측된 사실이다.** 같은 `(session_id, sequence_id)`에 **같은 본문**이 다시 오면 `duplicate=true`, `frames_accepted=0`이고 다시 채점하지 않는다. **다른 본문**이 오면 `stored=false`, `duplicate=false`다 — 두 조합이 서로 구별되므로 별도 필드가 필요 없다. 조용히 덮어쓰지 않는다.

**요청 쪽 관용 (응답 모양은 바꾸지 않는다).**

- 최상위는 `forbid`다. 오타가 조용히 무시되면 엉뚱한 세션에 쓴다.
- **프레임 하위는 `allow`**이고 미지 키는 **요청당 WARNING 로그 1건**으로 보고한다(응답에는 싣지 않는다). 현행 정책을 프레임까지 적용하면 실물 샘플의 정상 시퀀스 100%가 422다.
- `skeleton_joints`는 실물의 `{관절: [x,y,z]}`와 §7.1 예시의 `{관절: {pos, rot}}`를 모두 받는다. 검증은 통과시키고 해석은 엔진이 한다.
- `session_id` 부재는 422가 아니다. 저장은 하고 자세 채점은 건너뛴다(`session_id_missing`). 없는 세션을 지어내면 자세 누적이 엉뚱한 곳에 쌓이고, 422로 버리면 텔레메트리가 사라진다.

**`POST /api/v1/copilot/analyze-sequence`.** 계약에 없는 경로인데 벤더 클라이언트가 **같은 본문을 두 경로로** 보낸다(`docs/sprayxr_v25_sample_delta_2026-09-28.md` B18, 벤더 질의 15). 404를 주면 현장에서 실패하므로 멱등 별칭으로 받고 두 번째는 `duplicate`로 식별한다. **벤더 결함에 대한 수용책이며 계약에 추가하는 것이 아니다.** 회신이 오면 제거한다.

**상태 저장소.** SQLite(계획 E7), 경로는 `SEQUENCE_DB_PATH`. 기본값은 agent-api가 쓸 수 있는 유일한 마운트 아래이며 전용 `STATE_DIR` 볼륨이 제 자리다(후속). 경로를 쓸 수 없으면 메모리로 내려가고 수신은 계속된다.

## 8. 이상 알림 SSE

### 8.1 연결

`GET /api/v1/stream/anomalies/{session_id}`

- 세션 시작 후 연결한다.
- 이상이 있을 때만 `anomaly_detection` 이벤트를 전송한다.
- 연결 유지용 comment heartbeat를 최대 15초 간격으로 보낸다.
- 모든 이벤트에 증가하는 `id`를 포함한다.
- 클라이언트는 `Last-Event-ID`로 재연결할 수 있다.
- 같은 `sequence_id`의 동일 이상을 중복 발행하지 않는다.

### 8.2 이벤트

```text
id: 42
event: anomaly_detection
data: {"event_id":"anomaly-SESSION-001-SEQ-0001","session_id":"SESSION-001","user_id":"USER-017","scenario_id":"TR-L01","sequence_id":"SEQ-0001","generated_at":"2026-09-16T08:01:01Z","text_guide":"이격거리를 줄이고 체류 시간을 늘리세요.","anomaly":{"detected":true,"anomaly_type":["coating"],"items":[{"axis":"thickness","type":"under_app","zone_id":"A-1","action":"increase_dwell"}]},"sequence_data_ack":{"stored":true,"duplicate":false,"frames_received":1,"frames_accepted":1}}

```

SSE의 판정 값은 같은 시퀀스 HTTP 응답의 값과 일치해야 한다.

## 9. Co-Pilot 질의

### 9.1 요청

`POST /api/v1/copilot/query`

```json
{
  "user_id": "USER-017",
  "scenario_id": "TR-L01",
  "session_id": "SESSION-001",
  "request_id": "REQ-0041",
  "prompt": "현재 자세와 도막 상태를 반영해서 다음 작업을 알려줘.",
  "task_type": "pose_recomm",
  "use_rag": "Y",
  "context": {
    "sequence_id": "SEQ-0001"
  }
}
```

| `task_type` | 필수 동작 |
|---|---|
| `trajectory_recomm` | 현재·다음 구역과 궤적 판정 설명 |
| `pose_recomm` | 자세 유사도와 교정 항목 설명 |
| `error_analysis` | 도막·순서·자세 판정을 조합한 원인 설명 |
| `knowledge_search` | 도장 문서 검색과 근거 인용 |

모든 `task_type`은 실제 구현되어야 한다. 고정된 “준비 중” 응답은 제품 계약 위반이다.

### 9.2 응답

```json
{
  "status": "SUCCESS",
  "code": 200,
  "message": "Copilot query completed",
  "data": {
    "request_id": "REQ-0041",
    "session_id": "SESSION-001",
    "user_id": "USER-017",
    "scenario_id": "TR-L01",
    "generated_at": "2026-09-16T08:02:00Z",
    "latency_ms": 28450,
    "degraded": false,
    "answer": "숙련자 자세 대비 유사도는 82%입니다. 오른쪽 팔꿈치 자세가 통상 범위를 벗어났으므로 자세를 다시 확인하고 이격거리를 줄이세요.",
    "resolved_intents": ["pose_recommendation", "coating_error_analysis"],
    "context_status": "available",
    "sequence_context": {
      "current_zone_id": "A-1",
      "next_zone_id": "A-2",
      "expected_direction": "horizontal"
    },
    "coating_context": {
      "sequence_id": "SEQ-0001",
      "zone_id": "A-1",
      "status": "under_app",
      "thickness_um": 82.0
    },
    "pose_context": {
      "similarity_score": 0.82,
      "correction_mode": "joint_only",
      "corrections": [
        {
          "joint": "right_elbow",
          "target_validated": false,
          "angle_delta_deg": null,
          "direction": null
        }
      ]
    },
    "retrieval_used": true,
    "citations": ["DOC-doc_0022-C0214"]
  }
}
```

### 9.3 Query 불변 규칙

- `use_rag="Y"`이면 실제 검색을 수행하고 사용한 문서만 `citations`에 넣는다. `citations`는 **이번 요청이 검색한 청크의 부분집합**이며 검색 집합 자체가 아니다. 답변이 기대지 않은 문서는 들어가지 않으므로 보통 1~3개다.
- **`answer` 본문에 나타나는 모든 문서 id는 `citations`에 있다.** 답변이 인용했으나 이번 요청이 검색하지 않은 id는 본문과 배열에서 함께 제거되고 서버 경고로 남는다. 본문과 배열이 어긋나는 응답은 존재하지 않는다.
- `use_rag="N"`이면 `retrieval_used=false`, `citations=[]`이고, **`answer` 본문에도 문서 id가 나타나지 않는다.** 문서를 주지 않은 요청이 출처를 제시할 수는 없다.
- `answer`는 **마커 없는 산문**이다(§9.2 예시와 같다). 절 표기(`[답변]`·`[근거]`·`[판정]`·`[원인]`·`[조치]`)는 어느 것도 포함되지 않고, **문서 id도 `answer`에 나타나지 않는다** — 근거는 `citations`로만 전달된다. `knowledge_search`에서 판정·원인·조치 절이 빠지는 이유는 측정값 없이 판정하지 않기 때문이다.
- 답변을 생성하지 못하면 `answer`는 `답변을 생성하지 못했습니다.`이고 `degraded=true`, `citations=[]`다.
  - 설계·근거: `docs/superpowers/specs/2026-09-29-knowledge-answer-structured-output-design.md`
- 세션이 있으면 최신 누적 상태를 사용한다.
- 세션이 없는 일반 지식 질의는 context를 `not_available`로 반환할 수 있다.
- 내부 도구 호출, 프롬프트 또는 사고 과정은 반환하지 않는다.
- 정상 완료는 `degraded=false`다.
- 제한 시간 내 생성이 끝나지 않으면 가능한 판정 및 RAG 근거를 보존한 축약 응답과 `degraded=true`를 반환한다.

## 10. 세션 종료

`POST /api/v1/sessions/end`

```json
{
  "user_id": "USER-017",
  "scenario_id": "TR-L01",
  "session_id": "SESSION-001",
  "ended_at": "2026-09-16T09:00:00Z",
  "worker": {
    "skill_level": "novice",
    "handedness": "right"
  }
}
```

성공 시 세션을 종료하고 SSE 연결을 정상 종료한다. 종료된 세션에는 새 시퀀스를 받을 수 없다. 리포트가 진행 중이라면 리포트 job은 계속 실행한다.

## 11. 분석 요청과 상태 조회

### 11.1 분석 요청

`POST /api/v1/sessions/complete`

Content-Type은 `multipart/form-data`이며 다음 파트를 사용한다.

| 파트 | 타입 | 필수 | 설명 |
|---|---|:---:|---|
| `user_id` | String | Y | 작업자 ID |
| `scenario_id` | String | Y | `TR-L01` |
| `session_id` | String | Y | 세션 ID |
| `request_id` | String | Y | 멱등 키 |
| `final_score` | Integer | Y | 0~100 |
| `is_pass` | Boolean | Y | 합격 여부 |
| `avg_thickness_um` | Float | Y | 최종 평균 도막 |
| `paint_used` | Float | Y | 도료 사용량 |
| `paint_utilization` | Float | Y | 도료 사용률 |
| `paint_area` | Float | Y | 작업 면적 |
| `paint_time_ms` | Integer | Y | 작업 시간 |
| `paint_speed_mms` | Float | Y | 작업 속도 |
| `height_map_image` | PNG | Y | 최종 도막 Height Map |
| `telemetry_csv` | CSV | Y | 전체 원시 텔레메트리 |

접수 응답은 장기 생성을 기다리지 않는다.

```json
{
  "status": "SUCCESS",
  "code": 202,
  "message": "Session report accepted",
  "data": {
    "request_id": "REQ-REPORT-0034",
    "session_id": "SESSION-001",
    "user_id": "USER-017",
    "scenario_id": "TR-L01",
    "generated_at": "2026-09-16T09:00:01Z",
    "latency_ms": 450,
    "report_id": "REPORT-SESSION-001",
    "report_status": "pending"
  }
}
```

### 11.2 상태 조회

`GET /api/v1/sessions/{session_id}/reports/{report_id}`

`report_status`는 `pending`, `complete`, `failed` 중 하나다. job 실패는 HTTP 요청 자체가 성공했으면 HTTP 200과 `report_status="failed"`로 표현한다.

```json
{
  "status": "SUCCESS",
  "code": 200,
  "message": "Session report status",
  "data": {
    "request_id": "REQ-REPORT-0034",
    "session_id": "SESSION-001",
    "user_id": "USER-017",
    "scenario_id": "TR-L01",
    "generated_at": "2026-09-16T09:00:10Z",
    "latency_ms": 8,
    "report_id": "REPORT-SESSION-001",
    "report_status": "pending",
    "reason_code": null
  }
}
```

## 12. 분석 결과

`GET /api/v1/reports/{report_id}`

완료 전에는 HTTP 409를 반환한다. 완료된 리포트는 네 객체를 제공한다.

```json
{
  "status": "SUCCESS",
  "code": 200,
  "message": "Session report completed",
  "data": {
    "request_id": "REQ-REPORT-0034",
    "session_id": "SESSION-001",
    "user_id": "USER-017",
    "scenario_id": "TR-L01",
    "generated_at": "2026-09-16T09:00:30Z",
    "latency_ms": 12,
    "report": {
      "report_id": "REPORT-SESSION-001",
      "report_status": "complete",
      "coating_report": {
        "analysis_status": "available",
        "algorithm_version": "coating-session-report-1.0",
        "average_thickness_um": 96.2,
        "guideline": "A-1 저도막 구간을 우선 보수하세요."
      },
      "sequence_report": {
        "analysis_status": "available",
        "algorithm_version": "sequence-session-report-1.0",
        "recommended_flow": ["A-1", "A-2", "A-3"],
        "visited_flow": ["A-1", "A-2", "A-3"],
        "guideline": "권장 순서를 준수했습니다."
      },
      "pose_report": {
        "analysis_status": "available",
        "algorithm_version": "pose-session-report-1.0",
        "average_similarity_score": 0.81,
        "priority_corrections": [
          {
            "priority": 1,
            "zone_id": "A-1",
            "joint": "right_elbow",
            "occurrence_rate": 0.67,
            "correction_guidance": "오른쪽 팔꿈치 자세가 숙련자 통상 범위를 반복적으로 벗어났습니다."
          }
        ]
      },
      "llm_briefing": {
        "analysis_status": "available",
        "summary": "작업 순서는 안정적이지만 A-1 저도막과 팔꿈치 자세 교정이 필요합니다.",
        "citations": ["DOC-doc_0022-C0214"]
      }
    }
  }
}
```

수치와 상태는 결정론적 보고서에서 가져오며 `llm_briefing`은 이를 설명만 한다.

## 13. 제품 UI 계약

제품 UI는 내부 데모 UI와 구분한다.

### 13.1 입력

- 세션 화면은 `user_id`, `scenario_id`, `session_id`, 작업자 숙련도와 주손을 보존한다.
- 질의 화면은 `prompt`, `task_type`, `use_rag`, 선택적 `context.sequence_id`를 전송한다.
- 운영 UI는 `message`, `mode`, `level_id`만 받는 내부 `/api/chat` 구조를 사용하지 않는다.

### 13.2 출력

- `answer`와 구조화 context를 분리해 표시한다.
- RAG 사용 시 citation ID를 표시한다.
- 도막, 순서, 자세 수치와 판정 코드를 임의로 자연어에서 재추출하지 않는다.
- 진행 상태는 `요청 검증`, `문서 검색`, `응답 생성`처럼 실제 단계를 사용한다.
- 이미 로드된 모델에 대해 매 요청마다 `모델 초기화`라고 표시하지 않는다.
- 리포트는 `coating_report`, `sequence_report`, `pose_report`, `llm_briefing`으로 구분한다.

## 14. 성능 및 가용성 계약

| 경로 | 기준 |
|---|---|
| 세션 시작·종료 | p95 500ms 이하 |
| 시퀀스 HTTP 판정 | p95 500ms 이하, LLM 호출 금지 |
| 이상 SSE | 시퀀스 판정 확정 후 1초 이내 |
| Query | 전체 p95 60초 이하, 초과 시 축약 `degraded` 응답 |
| 분석 요청 접수 | 2초 이내 `report_id` 반환 |
| 분석 상태 조회 | p95 200ms 이하 |
| 리포트 생성 | 비동기, 연결 종료와 무관하게 완료 |

LLM 응답 시간은 모델 정밀도와 서빙 백엔드에 따라 달라질 수 있다. 제품 UI는 Query를 동기 HTTP로 호출하되 명확한 진행 상태를 표시한다. 실제 토큰 스트리밍을 추가할 경우 별도 버전으로 계약을 확장하며 기존 JSON 응답을 깨지 않는다.

## 15. Spark 배포 계약

- 제품 추론은 외부 vLLM OpenAI-compatible 서버를 사용한다.
- 모델은 LoRA 병합이 끝난 완성 체크포인트만 배포한다.
- Agent API는 요청마다 모델 또는 RAG 인덱스를 다시 로드하지 않는다.
- 모델, RAG 인덱스, 임베더 및 리랭커는 readiness 전에 준비되어야 한다.
- `/api/ready`는 파이프라인, 모델, RAG 인덱스 및 필수 서비스 상태를 검증한다.
- 외부 공개 전 인증 또는 신뢰 네트워크 정책을 적용한다.
- 이미지 digest, 코드 커밋 및 번들 manifest를 동일 릴리스 기록에 보존한다.
- 양자화 후보가 품질 게이트를 실패하면 승인된 정밀도로 롤백한다.

모델 정밀도 변경은 API 계약 변경이 아니다. FP8, BF16 또는 향후 승인된 형식 모두 같은 `/api/v1` 계약을 제공해야 한다.

## 16. 인수시험

### 16.1 계약 시험

- OpenAPI에 §5의 8개 엔드포인트가 모두 존재한다.
- 내부 분석 인터페이스가 §18의 5종으로만 구성되고 스키마를 준수한다.
- 모든 요청 모델은 알 수 없는 필드를 거부한다.
- Python Pydantic 모델과 UI Zod 모델이 동일한 필드와 제약을 가진다.
- 성공·오류 응답이 공통 래퍼를 준수한다.
- PDF 예제와 이 문서 예제를 fixture로 실행한다.

### 16.2 상태 시험

1. 세션 시작 전 시퀀스는 409다.
2. 세션 시작 후 첫 시퀀스가 누적 상태를 갱신한다.
3. 같은 시퀀스 재전송은 중복 저장하지 않는다.
4. Query가 최신 도막·순서·자세 context를 사용한다.
5. 세션 종료 후 새 시퀀스는 409다.
6. 리포트 job은 세션 종료 또는 클라이언트 연결 종료 뒤에도 완료된다.

### 16.3 실제 표면 시험

- 실제 HTTP로 세션 시작부터 종료까지 전체 흐름을 수행한다.
- 90 FPS 샘플 시퀀스를 업로드하고 HTTP 판정과 SSE가 일치하는지 확인한다.
- 네 가지 `task_type`을 각각 호출해 고정 응답이 아닌 실제 context를 확인한다.
- `use_rag` Y/N에 따른 citations 차이를 확인한다.
- multipart 리포트를 접수하고 `pending → complete` 전이를 확인한다.
- 실제 브라우저에서 입력 필드, context, citations 및 리포트 화면을 확인한다.
- 컨테이너 재기동 후 모델과 인덱스가 한 번만 초기화되는지 확인한다.

### 16.4 배포 승인 실패 조건

다음 중 하나라도 해당하면 제품 배포를 승인하지 않는다.

- 제품 UI가 `/api/chat` 또는 `/api/sim/*`을 호출함
- 공개 API 일부가 OpenAPI에 없음
- 시퀀스 판정이 stub 또는 고정 fixture임
- 네 가지 Query task 중 하나라도 “준비 중” 고정 응답임
- HTTP 판정과 SSE 판정이 다름
- 리포트가 동기 장기 요청이거나 프로세스 메모리에만 존재해 재기동 시 소실됨
- 모델 초기화가 요청마다 반복됨
- 계약과 다른 필드명, 단위 또는 응답 래퍼를 사용함

## 17. 현재 구현과의 차이

2026-09-16 Spark 배포본은 모델 서빙과 내부 데모 실행에는 성공했지만 이 계약을 충족하지 않는다.

| 항목 | 현재 상태 | 목표 상태 |
|---|---|---|
| vLLM Qwen3.6-27B | 동작 | 유지 |
| `/api/v1/copilot/query` | `knowledge_search`만 실제 실행 | 네 task 실제 실행 |
| `/api/v1/sessions/start` | 없음 | 구현 |
| `/api/v1/telemetry/sequence` | 없음 | 3축 실제 판정 구현 |
| `/api/v1/stream/anomalies/{session_id}` | 없음 | SSE 구현 |
| `/api/v1/sessions/end` | 없음 | 구현 |
| `/api/v1/sessions/complete` | 없음 | 비동기 job 구현 |
| 리포트 상태·결과 | 없음 | 구현 |
| 제품 UI | `/api/chat`, `/api/sim/*` 사용 | `/api/v1` 전환 |
| 시퀀스 결과 | 내부 deterministic stub | 실제 판정 엔진 연결 |
| 내부 분석 인터페이스 | 중복 책임을 가진 7개 MCP | §18의 필수 5종으로 통합 |
| 진행 문구 | 매 요청 `모델 초기화` 표시 | 실제 단계 표시 |

이 표의 모든 목표 상태가 구현되고 §16을 통과한 뒤 새 Spark 릴리스를 생성한다.

## 18. 필수 분석 인터페이스 계약

### 18.1 구성 원칙

필수 내부 분석 인터페이스는 다음 5개뿐이다.

| 인터페이스 | 단일 책임 |
|---|---|
| `coating_analysis` | 도막 분포와 규격 준수 판정 |
| `pose_analysis` | 숙련자 자세 유사도와 검증된 교정 항목 산출 |
| `sequence_analysis` | 구역 순서, 궤적, 방향, 거리와 속도 판정 |
| `anomaly_detection` | 앞선 세 분석 결과를 정규화된 이상 목록으로 조합 |
| `knowledge_search` | RAG 문서 검색과 citation 제공 |

다음 기능은 별도 MCP로 두지 않는다.

- `location_tracking`: SIM 입력이 `zone_id`, `raypoint`, `raytexmapnum`, `raytexuv`를 제공하므로 제거한다. `zone_id`가 없으면 요청을 422로 거부하며 좌표로 추정하지 않는다.
- `timeseries`: 90 FPS `sequence_info`를 `pose_analysis`와 `sequence_analysis` 입력으로 직접 정규화한다.
- `dft_judge`: 90/10 판정과 도막 분포 판정을 `coating_analysis`가 함께 소유한다.
- `vlm_analysis`: v2.5 필수 실시간 경로에서 제외한다. 향후 선택 기능으로 추가하더라도 이 5개 인터페이스의 판정을 덮어쓰지 않는다.

각 인터페이스는 별도 프로세스일 필요가 없다. 하나의 MCP 서버에 다섯 라우터로 탑재할 수 있지만 입력·출력과 책임 경계는 분리한다.

### 18.2 공통 입력 context

모든 분석 요청은 `context`를 가진다.

```json
{
  "context": {
    "request_id": "REQ-SEQUENCE-0001",
    "session_id": "SESSION-001",
    "scenario_id": "TR-L01",
    "sequence_id": "SEQ-0001",
    "zone_id": "A-1"
  }
}
```

| 필드 | 타입 | 필수 | 설명 |
|---|---|:---:|---|
| `request_id` | String | Y | 분석 호출 추적 ID |
| `session_id` | String 또는 Null | 조건부 | 시퀀스 분석은 필수, 일반 지식 검색은 null 가능 |
| `scenario_id` | String | Y | 현재 `TR-L01` |
| `sequence_id` | String 또는 Null | 조건부 | 시퀀스 기반 분석은 필수 |
| `zone_id` | String 또는 Null | 조건부 | 시퀀스 기반 분석은 `A-1`, `A-2`, `A-3` 중 하나 |

`user_id`는 분석에 필요하지 않으므로 MCP로 전달하지 않는다. 작업자 특성 중 분석에 필요한 `skill_level`, `handedness`만 해당 입력에 명시한다.

### 18.3 공통 출력 envelope

```json
{
  "status": "success",
  "result": {
    "analysis_status": "available",
    "reason_code": null,
    "algorithm_version": "coating-2.5.0"
  },
  "error": null,
  "meta": {
    "request_id": "REQ-SEQUENCE-0001",
    "session_id": "SESSION-001",
    "sequence_id": "SEQ-0001",
    "generated_at": "2026-09-16T08:01:01Z",
    "latency_ms": 18
  }
}
```

- `status`: 호출 성공 여부인 `success` 또는 `error`다. 분석 결과의 정상·이상 여부가 아니다.
- `analysis_status`: `available` 또는 `not_available`이다.
- `reason_code`: 분석 가능 시 `null`, 불가능 시 안정적인 코드다.
- `algorithm_version`: 결과를 만든 알고리즘 버전이다.
- `error`: 호출 실패 시에만 `{code, message, details}`를 가진다.
- 분석 불가능 상태를 임의의 정상값으로 대체하지 않는다.
- 모든 입력 모델은 `extra="forbid"`와 동등하게 알 수 없는 필드를 거부한다.

### 18.4 `coating_analysis`

내부 경로는 `POST /coating_analysis/analyze`다. 이 인터페이스가 도막 분포와 90/10 규격 판정의 유일한 소유자다.

#### 입력

```json
{
  "context": {
    "request_id": "REQ-SEQUENCE-0001",
    "session_id": "SESSION-001",
    "scenario_id": "TR-L01",
    "sequence_id": "SEQ-0001",
    "zone_id": "A-1"
  },
  "spec": {
    "target_thickness_um": 100,
    "normal_band_um": [90, 120],
    "tolerance_rule": "90/10"
  },
  "sequence_result": {
    "avg_thickness_um": 82,
    "cur_sequence_heatmap": [
      {"map_num": 1, "uv_bin": [3, 1], "thickness_um": 18}
    ],
    "acc_sequence_heatmap": [
      {"map_num": 1, "uv_bin": [3, 1], "thickness_um": 82}
    ]
  },
  "dft_measurements": [
    {
      "value_um": 82,
      "map_num": 1,
      "uv_bin": [3, 1],
      "position": "lower"
    }
  ]
}
```

| 필드 | 필수 | 규칙 |
|---|:---:|---|
| `spec.target_thickness_um` | Y | 0보다 큰 µm 값 |
| `spec.normal_band_um` | Y | `[min, max]`, `0 < min <= max` |
| `spec.tolerance_rule` | Y | 현재 `90/10`만 허용 |
| `sequence_result.avg_thickness_um` | Y | SIM이 계산한 시퀀스 평균 |
| `cur_sequence_heatmap` | Y | 현재 시퀀스가 추가한 도막의 sparse cell 배열 |
| `acc_sequence_heatmap` | Y | 현재까지 누적된 도막의 sparse cell 배열 |
| `dft_measurements` | N | 실제 DFT 측정값이 있을 때만 제공 |

히트맵 cell은 `map_num`, `[u, v]` 정수 bin, `thickness_um`를 가진다. 이미지 파일 경로는 받지 않는다.

#### 출력

```json
{
  "status": "success",
  "result": {
    "analysis_status": "available",
    "reason_code": null,
    "algorithm_version": "coating-2.5.0",
    "zone_id": "A-1",
    "status": "under_app",
    "pass_90_10": false,
    "statistics": {
      "mean_um": 82,
      "min_um": 82,
      "max_um": 82,
      "normal_coverage_ratio": 0,
      "under_ratio": 1,
      "over_ratio": 0
    },
    "distribution": {
      "under2": 0,
      "under1": 1,
      "normal": 0,
      "over1": 0,
      "over2": 0,
      "over3": 0
    },
    "worst_regions": [
      {
        "map_num": 1,
        "uv_bin": [3, 1],
        "thickness_um": 82,
        "zone_class": "under1",
        "action": "increase_dwell",
        "action_reachable": true
      }
    ]
  },
  "error": null,
  "meta": {
    "request_id": "REQ-SEQUENCE-0001",
    "session_id": "SESSION-001",
    "sequence_id": "SEQ-0001",
    "generated_at": "2026-09-16T08:01:01Z",
    "latency_ms": 12
  }
}
```

- `status`는 `under_app`, `normal`, `over_app` 중 하나다.
- `zone_class`와 `status` 조합은 §3.7의 불변 규칙을 따른다.
- DFT 측정이 없거나 표본이 부족하면 `pass_90_10=null`과 별도 `reason_code`를 사용한다. 히트맵 분포 결과를 DFT 실측 합불로 위장하지 않는다.
- LLM 설명문은 생성하지 않는다.

### 18.5 `pose_analysis`

내부 경로는 `POST /pose_analysis/compare`다. 비교 단위는 프레임 하나가 아니라 Trigger ON→OFF 패스다.

#### 입력

```json
{
  "context": {
    "request_id": "REQ-SEQUENCE-0001",
    "session_id": "SESSION-001",
    "scenario_id": "TR-L01",
    "sequence_id": "SEQ-0001",
    "zone_id": "A-1"
  },
  "worker": {
    "skill_level": "novice",
    "handedness": "right"
  },
  "reference": {
    "level_id": "TR-L01",
    "bucket": "horizontal-mid-+y"
  },
  "pose_frames": [
    {
      "index": 0,
      "time": "2026-09-16T08:01:00.000Z",
      "joints": {
        "pelvis": {"pos": [0, 1, 0], "rot": [0, 0, 0]},
        "right_elbow": {"pos": [0.3, 1.4, 0.2], "rot": [4, 8, 2]}
      }
    }
  ]
}
```

- Agent API는 `sequence_info[].skeleton_joints`만 `pose_frames`로 정규화한다.
- 프레임 순서, 관절 이름, 좌표계와 단위는 요청 전에 검증한다.
- `handedness=left`이면 좌우 관절 매핑 규칙을 적용하고 그 버전을 결과에 기록한다.
- 레퍼런스가 없거나 품질 기준에 미달하면 값을 추정하지 않는다.

#### 출력

```json
{
  "status": "success",
  "result": {
    "analysis_status": "available",
    "reason_code": null,
    "algorithm_version": "pose-2.5.0",
    "reference_id": "horizontal-mid-+y",
    "reference_quality": "single_worker",
    "support": 56,
    "similarity_score": 0.82,
    "expert_percentile": 68,
    "correction_mode": "joint_only",
    "corrections": [
      {
        "joint": "right_elbow",
        "label": "오른쪽 팔꿈치가 숙련자 통상 범위를 벗어났습니다.",
        "severity": "warning",
        "target_validated": false,
        "angle_delta_deg": null,
        "direction": null
      }
    ]
  },
  "error": null,
  "meta": {
    "request_id": "REQ-SEQUENCE-0001",
    "session_id": "SESSION-001",
    "sequence_id": "SEQ-0001",
    "generated_at": "2026-09-16T08:01:01Z",
    "latency_ms": 24
  }
}
```

- `similarity_score`는 같은 버킷 숙련자 분포와의 유사도이며 자세 합불, 안전성 또는 숙련도 점수가 아니다.
- 교정 목표 검증 게이트를 통과하기 전에는 `correction_mode="joint_only"`로 관절만 지목한다.
- 검증되지 않은 각도와 방향을 사용자에게 지시하지 않는다. 이때 `angle_delta_deg`와 `direction`은 `null`이다.
- `expert_percentile`은 교차적합 보정이 유효할 때만 제공하며 아니면 `null`이다.

### 18.6 `sequence_analysis`

내부 경로는 `POST /sequence_analysis/analyze`다. 입력에 `zone_id`가 있으므로 좌표 기반 위치 추론을 수행하지 않는다.

#### 입력

```json
{
  "context": {
    "request_id": "REQ-SEQUENCE-0001",
    "session_id": "SESSION-001",
    "scenario_id": "TR-L01",
    "sequence_id": "SEQ-0001",
    "zone_id": "A-1"
  },
  "flow": {
    "recommended_flow": ["A-1", "A-2", "A-3"],
    "visited_flow": [],
    "expected_zone_id": "A-1",
    "expected_direction": "horizontal"
  },
  "trajectory_frames": [
    {
      "index": 0,
      "time": "2026-09-16T08:01:00.000Z",
      "tool_pos": [0.1, 1.2, 0.5],
      "raypoint": [1.1, 1.3, 0],
      "raytexmapnum": 1,
      "raytexuv": [221, 144],
      "speed_mms": 2200,
      "distance_mm": 450,
      "spray_angle_deg": 72
    }
  ],
  "targets": {
    "speed_mms": 2800,
    "distance_mm": 400
  }
}
```

- Agent API는 `sequence_info[].tool_pos_rot`와 `gun_metrics`를 `trajectory_frames`로 정규화한다.
- 속도는 `mm/s`, 거리는 `mm`, 각도는 degree로 고정한다.
- `zone_id`는 public request의 값을 그대로 사용하며 분석기가 변경하지 않는다.

#### 출력

```json
{
  "status": "success",
  "result": {
    "analysis_status": "available",
    "reason_code": null,
    "algorithm_version": "sequence-2.5.0",
    "current_zone_id": "A-1",
    "expected_zone_id": "A-1",
    "next_zone_id": "A-2",
    "adherence_status": "on_track",
    "deviations": [],
    "trajectory_metrics": {
      "direction": "horizontal",
      "expected_direction": "horizontal",
      "straightness_score": 72,
      "deviation_mm": 45.8,
      "mean_speed_mms": 2200,
      "target_speed_mms": 2800,
      "mean_distance_mm": 450,
      "target_distance_mm": 400
    },
    "issues": [
      {
        "type": "trajectory_deviation",
        "severity": "warning",
        "action": "stabilize_wrist"
      },
      {
        "type": "distance_speed_error",
        "severity": "warning",
        "action": "adjust_distance_speed"
      }
    ]
  },
  "error": null,
  "meta": {
    "request_id": "REQ-SEQUENCE-0001",
    "session_id": "SESSION-001",
    "sequence_id": "SEQ-0001",
    "generated_at": "2026-09-16T08:01:01Z",
    "latency_ms": 20
  }
}
```

방향 enum은 `horizontal`, `vertical`, `diagonal`, `curved_irregular`, `stationary`만 사용한다. 순서 이탈과 궤적 문제를 하나의 문자열로 합치지 않고 `deviations`와 `issues`로 분리한다.

### 18.7 `anomaly_detection`

내부 경로는 `POST /anomaly_detection/detect`다. 이 인터페이스는 원시 센서 값을 다시 판정하지 않고 세 분석 결과를 조합한다.

#### 입력

```json
{
  "context": {
    "request_id": "REQ-SEQUENCE-0001",
    "session_id": "SESSION-001",
    "scenario_id": "TR-L01",
    "sequence_id": "SEQ-0001",
    "zone_id": "A-1"
  },
  "coating_result": {
    "analysis_status": "available",
    "status": "under_app",
    "worst_regions": [
      {"map_num": 1, "uv_bin": [3, 1], "action": "increase_dwell"}
    ]
  },
  "pose_result": {
    "analysis_status": "available",
    "similarity_score": 0.82,
    "corrections": []
  },
  "sequence_result": {
    "analysis_status": "available",
    "adherence_status": "on_track",
    "issues": [
      {"type": "distance_speed_error", "action": "adjust_distance_speed"}
    ]
  },
  "sequence_data_ack": {
    "stored": true,
    "duplicate": false,
    "frames_received": 90,
    "frames_accepted": 90
  }
}
```

#### 출력

```json
{
  "status": "success",
  "result": {
    "analysis_status": "available",
    "reason_code": null,
    "algorithm_version": "anomaly-2.5.0",
    "detected": true,
    "anomaly_type": ["coating", "trajectory"],
    "items": [
      {
        "axis": "thickness",
        "type": "under_app",
        "zone_id": "A-1",
        "map_num": 1,
        "uv_bin": [3, 1],
        "action": "increase_dwell"
      },
      {
        "axis": "trajectory",
        "type": "distance_speed_error",
        "zone_id": "A-1",
        "action": "adjust_distance_speed"
      }
    ],
    "guidance_codes": ["INCREASE_DWELL", "ADJUST_DISTANCE_SPEED"],
    "text_guide": "A-1 구역의 체류 시간을 늘리고 이격거리와 이동 속도를 기준에 맞추세요."
  },
  "error": null,
  "meta": {
    "request_id": "REQ-SEQUENCE-0001",
    "session_id": "SESSION-001",
    "sequence_id": "SEQ-0001",
    "generated_at": "2026-09-16T08:01:01Z",
    "latency_ms": 4
  }
}
```

- 도막 상태는 `coating_result`, 자세 상태는 `pose_result`, 순서·궤적 상태는 `sequence_result`를 정본으로 사용한다.
- 입력 결과와 다른 수치나 분류를 새로 만들지 않는다.
- `text_guide`는 SSE 지연 목표를 지킬 수 있도록 `guidance_codes` 기반 결정론적 템플릿으로 생성한다.
- `detected=false`이면 SSE를 발행하지 않는다.

### 18.8 `knowledge_search`

내부 경로는 `POST /knowledge_search/search`다. 이 인터페이스는 검색만 담당하며 최종 자연어 답변은 Agent API의 LLM 단계가 생성한다.

#### 입력

```json
{
  "context": {
    "request_id": "REQ-0044",
    "session_id": null,
    "scenario_id": "TR-L01",
    "sequence_id": null,
    "zone_id": null
  },
  "query": "저도막의 원인과 보수 방법은?",
  "top_k": 5,
  "filters": {
    "domain_categories": ["coating", "safety"],
    "source_types": ["pdf"]
  }
}
```

#### 출력

```json
{
  "status": "success",
  "result": {
    "analysis_status": "available",
    "reason_code": null,
    "algorithm_version": "rag-2.5.0",
    "retrieval_used": true,
    "query": "저도막의 원인과 보수 방법은?",
    "hits": [
      {
        "rank": 1,
        "citation_id": "DOC-doc_0022-C0214",
        "doc_id": "doc_0022",
        "chunk_id": "DOC-doc_0022-C0214",
        "text": "규정 도막 두께와 후속도장 표면상태를 확인한다.",
        "domain_category": "coating",
        "source_type": "pdf",
        "rerank_score": 0.975
      }
    ],
    "citations": ["DOC-doc_0022-C0214"]
  },
  "error": null,
  "meta": {
    "request_id": "REQ-0044",
    "session_id": null,
    "sequence_id": null,
    "generated_at": "2026-09-16T08:02:00Z",
    "latency_ms": 210
  }
}
```

- `hits[].text`는 실제 인덱스 chunk 원문이며 LLM이 만든 문장이 아니다.
- `citation_id`는 최종 Query 응답의 `citations`와 **같은 식별자 공간**이며, 포함 관계는 `Query.citations ⊆ result.citations`다. 이 인터페이스의 `citations`는 검색한 청크 전체(≤ `top_k`)이고, Query 응답의 `citations`는 그중 답변이 실제로 인용한 것이다. 두 값이 같다고 가정하는 클라이언트는 답변이 기대지 않은 문서까지 출처로 표시한다.
- 검색 결과가 없으면 `analysis_status="available"`, `retrieval_used=true`, `hits=[]`, `citations=[]`를 반환한다. 검색 서비스 장애만 `status="error"`다.
- `use_rag="N"`인 Query는 이 인터페이스를 호출하지 않는다.

### 18.9 호출 순서와 데이터 소유권

```text
POST /api/v1/telemetry/sequence
  → 공개 payload 검증·정규화·저장
  → coating_analysis ─┐
  → pose_analysis ────┼→ anomaly_detection
  → sequence_analysis ┘
  → 세션 상태 확정
  → HTTP 응답
  → detected=true일 때 동일 결과로 SSE 발행

POST /api/v1/copilot/query
  → task_type에 필요한 최신 세션 분석 결과 조회
  → use_rag=Y일 때 knowledge_search
  → LLM은 구조화 결과와 검색 근거를 설명
  → v2.5 Query 응답 조립
```

| 데이터 | 정본 소유자 |
|---|---|
| `zone_id`, ray 충돌점, UV | SIM 입력 |
| 도막 상태·분포·90/10 결과 | `coating_analysis` |
| 자세 유사도·교정 가능 여부 | `pose_analysis` |
| 순서·방향·궤적·거리·속도 결과 | `sequence_analysis` |
| 통합 이상 목록·즉시 안내 | `anomaly_detection` |
| 검색 chunk·citation | `knowledge_search` |
| 자연어 Query 답변·최종 브리핑 | Agent API의 LLM 단계 |

하위 인터페이스는 다른 인터페이스가 소유한 값을 재계산하거나 덮어쓰지 않는다.

### 18.10 기존 MCP 정리 기준

| 기존 MCP | 처리 |
|---|---|
| `location_tracking` | 제거 |
| `timeseries` | `pose_analysis`, `sequence_analysis` 입력 정규화로 흡수 |
| `dft_judge` | `coating_analysis`로 통합 |
| `anomaly_detection` | 원시 재판정 제거 후 결과 조합기로 변경 |
| `pose_estimation` | `pose_analysis`로 명칭과 계약 변경 |
| `sequence_validation` | `sequence_analysis`로 확장 |
| `vlm_analysis` | 필수 배포에서 제외 |
| 기존 Agent RAG | `knowledge_search` 계약으로 노출 |

제품 배포 전에 Agent API, 도구 스키마 JSON, MCP Pydantic 모델 및 테스트 fixture를 이 표에 맞춰 동시에 변경해야 한다.
