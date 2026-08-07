# AI Co-Pilot 응답 계약

**문서 버전:** 1.0-draft  
**작성일:** 2026-08-06  
**적용 범위:** 시퀀스 종료 판정, 사용자 Query 응답, 세션 종료 리포트

---

## 1. 목적

이 문서는 시뮬레이터가 받아야 하는 AI Co-Pilot 응답 3종을 정의한다.

1. 스프레이 Trigger ON → OFF 시퀀스마다 실행되는 도막·순서·자세 판정
2. 사용자가 Query를 전송했을 때 생성되는 답변
3. 세션 완료 후 생성되는 도막·순서·자세 누적 리포트

### 1.1 책임 분리

- 도막 상태·두께·처방은 **도막 이상탐지·콘텐츠센터 모듈**, 작업 순서 판정은 **순서 검증 MCP**, 자세 판정은 **자세 추정 MCP**가 결정한다.
- LLM은 `message`, `answer`, `guideline`, `llm_briefing`과 같은 **자연어 설명만 생성**한다.
- LLM은 콘텐츠센터가 산출한 상태, 수치, 분류 또는 처방 코드를 변경하거나 재계산하지 않는다.
- Query는 현재 세션 상태, 누적 시퀀스, 최근 도막·순서·자세 판정, RAG 문서 및 MCP 결과를 조합해 답변할 수 있다.

### 1.2 메시지 흐름

```text
POST /sessions/start
GET  /stream/anomalies/{session_id}

POST /telemetry/sequence
  → 시퀀스마다 도막·순서·자세 판정 실행
  → HTTP data에 판정 결과를 직접 반환
  → 도막 또는 경로 이상 검출 시에만 SSE anomaly_detection 추가 전달

POST /copilot/query
  → 순서·도막·자세·지식 질문에 답변

POST /sessions/complete
  → 최종 히트맵 및 도막·순서·자세 통계 확정
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

## 3. 시퀀스 종료 3축 판정

### 3.1 실행 규칙

`POST /telemetry/sequence`를 수신할 때마다 다음 절차를 반드시 실행한다.

```text
시퀀스 수신
  → 중복 여부와 프레임 저장 확인
  → 누적 히트맵 기반 도막 판정
  → 권장 흐름 기반 순서 판정
  → 기준 자세 기반 자세 판정
  → HTTP data에 판정 결과를 직접 반환
  → anomaly.detected=true이면 anomaly_detection SSE를 추가 전달
```

- 정상인 축도 반드시 판정하고 결과를 반환한다.
- 원천 데이터나 기준 데이터가 없어 분석할 수 없는 축은 값을 추정하지 않고 `analysis_status="not_available"`과 `reason_code`를 반환한다.
- UI는 각 축의 `analysis_status="available"`인 결과만 해석하고, 해당 축의 상태가 정상 범위를 벗어날 때만 경고를 표시한다.
- `anomaly_detection` 이벤트는 도막 또는 경로 판정 결과를 집계한 `anomaly.detected=true`인 경우에만 Push한다.
- `anomaly.items[].axis`는 `thickness`, `trajectory`만 사용한다. 자세는 이상탐지하지 않고 `pose_feedback`으로만 판정·안내한다.
- `feedback_bundle`은 서버 내부에서 여러 판정 결과를 조합할 때 사용할 수 있지만 HTTP 또는 SSE 경계에는 노출하지 않는다.
- 시뮬레이터는 HTTP `data`의 공통 결과와 namespaced 모듈 결과, 조건부 `anomaly_detection` SSE만 수신한다.
- `sequence_id`는 세션 내에서 1부터 증가하는 버스트 순번이다.

### 3.2 `sequence_data_ack`

| 필드 | 타입 | 필수 | 설명 |
|---|---|:---:|---|
| `stored` | Boolean | Y | 시퀀스 저장 성공 여부 |
| `duplicate` | Boolean | Y | 이미 처리한 시퀀스의 재전송 여부 |
| `frames_received` | Integer | Y | 요청에서 수신한 프레임 수 |
| `frames_accepted` | Integer | Y | 검증 후 저장한 프레임 수 |

### 3.3 시퀀스 HTTP `data` 필드

| 필드 | 타입 | 필수 | 설명 |
|---|---|:---:|---|
| `event_id` | String | Y | 시퀀스 판정 이벤트 ID |
| `sequence_id` | Integer | Y | 세션 내 버스트 순번 |
| `text_guide` | String | Y | 작업자에게 표시할 종합 안내 |
| `anomaly` | Object | Y | `detected`, `anomaly_type`, `items`를 포함한 도막·경로 이상 결과 |
| `sequence_data_ack` | Object | Y | 시퀀스 저장·프레임 수신 결과 |
| `coating_feedback` | Object | Y | 도막 판정 또는 분석 불가 상태 |
| `sequence_feedback` | Object | Y | 순서 판정 또는 분석 불가 상태 |
| `pose_feedback` | Object | Y | 자세 판정 또는 분석 불가 상태 |
| `trajectory_feedback` | Object | N | 전체 경로 지표·숙련자 비교·코칭 결과 |
| `coating_result` | Object | N | 도막 측정 상세 결과 |
| `citations` | Array[String] | N | 안내 생성에 사용한 문서 ID |

`feedback_bundle`이라는 내부 조합 객체를 사용하더라도 직렬화하지 않는다. 제공되는 세 축 객체는 `analysis_status`를 가진다. `available`이면 해당 축의 판정 필드를 제공하고 `reason_code=null`이다. `not_available`이면 판정값을 만들지 않고 `reason_code`에 안정적인 사유 코드를 제공한다.

### 3.4 `coating_feedback`

| 필드 | 타입 | 필수 | 설명 |
|---|---|:---:|---|
| `analysis_status` | String | Y | `available`, `not_available` |
| `reason_code` | String 또는 Null | Y | 분석 불가 사유. 분석 가능 시 `null` |
| `zone_id` | String | 조건부 | 대표 판정 구역 ID |
| `map_num` | Integer | 조건부 | Height map 번호 |
| `uv_bin` | Array[Integer] | 조건부 | `[x_bin, y_bin]` 형식의 세그먼트 좌표 |
| `thickness_um` | Number | 조건부 | 해당 세그먼트의 현재 누적 두께(µm) |
| `target_thickness_um` | Number | 조건부 | 목표 도막 두께(µm) |
| `normal_band_um` | Array[Number] | 조건부 | `[최솟값, 최댓값]` 정상 두께 범위(µm) |
| `status` | String | 조건부 | `under_app`, `normal`, `over_app` |
| `zone_class` | String | 조건부 | `under2`, `under1`, `normal`, `over1`, `over2`, `over3` |
| `under_frac` | Number | 조건부 | 이번 버스트 경로 중 규격 미달 비율, 0~1 |
| `action` | String 또는 Null | 조건부 | 권장 처방 코드. 정상은 `null` |
| `action_reachable` | Boolean 또는 Null | 조건부 | 행동 조정만으로 회복 가능한지 여부. 정상은 `null` |
| `message` | String | Y | LLM이 생성한 설명 문장 |
| `provisional` | Boolean | 조건부 | 추정값 여부. 누적 히트맵 실측은 `false` |
| `algorithm_version` | String | Y | 도막 판정 알고리즘 버전 |

### 3.5 `sequence_feedback`

| 필드 | 타입 | 필수 | 설명 |
|---|---|:---:|---|
| `analysis_status` | String | Y | `available`, `not_available` |
| `reason_code` | String 또는 Null | Y | 분석 불가 사유. 분석 가능 시 `null` |
| `current_zone_id` | String | 조건부 | 실제 작업 구역 |
| `expected_zone_id` | String | 조건부 | 권장 흐름상 현재 작업 구역 |
| `next_zone_id` | String 또는 Null | 조건부 | 다음 권장 구역. 마지막 구역은 `null` |
| `adherence_status` | String | 조건부 | `on_track`, `out_of_order`, `unknown` |
| `deviations` | Array[Object] | 조건부 | 누락·역순·재작업 등 이탈 목록 |
| `message` | String | Y | 판정 설명 또는 분석 불가 안내 |
| `algorithm_version` | String | Y | 순서 판정 알고리즘 버전 |

### 3.6 `pose_feedback`

| 필드 | 타입 | 필수 | 설명 |
|---|---|:---:|---|
| `analysis_status` | String | Y | `available`, `not_available` |
| `reason_code` | String 또는 Null | Y | 분석 불가 사유. 분석 가능 시 `null` |
| `mode` | String | 조건부 | 현재 구현의 비교 모드 `compare` |
| `expert_pose_id` | String | 조건부 | 비교에 사용한 현재 fixture 자세 ID |
| `similarity_score` | Number | 조건부 | 기준 자세와의 위치 유사도, 0~1 |
| `corrections` | Array[Object] | 조건부 | 관절별 거리·각도 차이와 생성된 교정 문구 |
| `summary` | String | Y | MCP가 생성한 비교 요약 또는 분석 불가 안내 |

조건부 필드는 `analysis_status="available"`일 때 필수다. `not_available`일 때는 생략한다. 허용 `reason_code`의 초기 집합은 `source_data_missing`, `reference_data_missing`, `quality_insufficient`, `tool_unavailable`이다.

현재 `expert_pose_id`는 실제 숙련자 데이터가 아니라 코드에 고정된 `EP-001` 계열 fixture다. 단일 자세 비교는 관절 위치 거리와 `roll`, `pitch`, `yaw` 차이만 계산한다. 손떨림, 몸통 기준 이동 방향, 시간에 따른 자세 안정성은 판정하지 않는다.

### 3.7 도막 상태 불변 규칙

| `zone_class` | `status` |
|---|---|
| `under2`, `under1` | `under_app` |
| `normal` | `normal` |
| `over1`, `over2`, `over3` | `over_app` |

`status`와 `zone_class`의 조합이 위 표와 다르면 서버는 계약 오류로 처리한다.

### 3.8 도막 처방 코드

| 코드 | 설명 |
|---|---|
| `increase_dwell` | 해당 구역의 체류 시간 또는 반복 분사를 증가 |
| `adjust_distance` | 작업면과 스프레이건 사이의 이격거리 조정 |
| `adjust_spraytime` | 분사 지속 시간 조정 |

`action_reachable=false`이면 작업자 동작만으로 회복하기 어렵고 도료·압력·노즐 등 공정 설정 조정이 필요하다는 의미다.

### 3.9 HTTP 응답 샘플

```json
{
  "status": "SUCCESS",
  "code": 200,
  "message": "Sequence feedback analysis completed",
  "data": {
    "request_id": "req-seq-0012",
    "session_id": "SESSION-001",
    "user_id": "W-017",
    "scenario_id": "1-A",
    "generated_at": "2026-08-06T14:31:04Z",
    "latency_ms": 186,
    "event_id": "anomaly-SESSION-001-12",
    "sequence_id": 12,
    "text_guide": "A-2 구역은 저도막입니다. 해당 구역의 체류 시간을 늘리세요.",
    "anomaly": {
      "detected": true,
      "anomaly_type": ["coating"],
      "items": [
        {
          "axis": "thickness",
          "type": "under_app",
          "zone_id": "A-2",
          "action": "increase_dwell"
        }
      ]
    },
    "sequence_data_ack": {
      "stored": true,
      "duplicate": false,
      "frames_received": 527,
      "frames_accepted": 527
    },
    "coating_feedback": {
        "analysis_status": "available",
        "reason_code": null,
        "zone_id": "A-2",
        "map_num": 2,
        "uv_bin": [3, 1],
        "thickness_um": 118,
        "target_thickness_um": 160,
        "normal_band_um": [160, 240],
        "status": "under_app",
        "zone_class": "under1",
        "under_frac": 0.8,
        "action": "increase_dwell",
        "action_reachable": true,
        "message": "하단 좌측 구역이 저도막입니다. 이 구역을 조금 더 오래 또는 반복 분사하세요.",
        "provisional": false,
        "algorithm_version": "coating-feedback-1.0"
    },
    "sequence_feedback": {
        "analysis_status": "available",
        "reason_code": null,
        "current_zone_id": "A-2",
        "expected_zone_id": "A-2",
        "next_zone_id": "A-3",
        "adherence_status": "on_track",
        "deviations": [],
        "message": "권장 순서에 맞게 A-2 구역을 작업하고 있습니다. 다음 구역은 A-3입니다.",
        "algorithm_version": "sequence-feedback-1.0"
    },
    "pose_feedback": {
        "analysis_status": "not_available",
        "reason_code": "reference_data_missing",
        "message": "이 시퀀스에 적용할 기준 자세가 없어 자세 판정을 제공할 수 없습니다.",
        "algorithm_version": "pose-feedback-1.0"
    }
  }
}
```

### 3.10 자세 판정 가능 샘플

```json
{
  "analysis_status": "available",
  "reason_code": null,
  "mode": "compare",
  "expert_pose_id": "EP-001",
  "similarity_score": 0.82,
  "corrections": [
    {
      "joint": "elbow_r",
      "label": "오른쪽 팔꿈치",
      "distance": 0.08,
      "angle_corrections": {
        "pitch": 5.0
      },
      "correction_text": "오른쪽 팔꿈치: pitch 5° 과다"
    }
  ],
  "summary": "숙련자 자세(EP-001) 대비 유사도 82%. 교정 필요 관절 1개: 오른쪽 팔꿈치: pitch 5° 과다"
}
```

### 3.11 `anomaly_detection` SSE

시퀀스 HTTP 응답과 별개로 도막 또는 경로 이상을 집계한 `anomaly.detected`가 `true`일 때만 Push한다. SSE `id:`는 본문의 `event_id`와 동일하다.

| 필드 | 타입 | 필수 | 설명 |
|---|---|:---:|---|
| `event_id` | String | Y | 이상 알림 중복 제거용 고유 ID |
| `sequence_id` | Integer | Y | 이상이 검출된 시퀀스 순번 |
| `text_guide` | String | Y | 작업자에게 표시할 이상 대응 안내 |
| `anomaly` | Object | Y | `detected`, `anomaly_type`, `items`를 포함한 이상 결과 |
| `sequence_data_ack` | Object | Y | 해당 시퀀스의 저장 결과 |

`anomaly_type`은 `coating`, `trajectory` 값의 배열이다. 두 이상이 함께 발생하면 `["coating", "trajectory"]` 순서로 두고 `items`도 도막 항목 다음 경로 항목으로 정렬한다. `combined` 같은 합성 값은 사용하지 않는다.

공통 item은 `axis`, `type`, `zone_id`, 단일 문자열 `action`을 가진다. 자세는 이상탐지 대상이 아니므로 `axis="pose"` 항목을 생성하지 않는다.

#### 3.11.1 도막 이상만 발생한 경우

```text
id: anomaly-SESSION-001-12
event: anomaly_detection
data: {"event_id":"anomaly-SESSION-001-12","sequence_id":12,"text_guide":"A-2 구역의 도막 두께가 부족합니다. 해당 구역의 체류 시간을 늘리세요.","anomaly":{"detected":true,"anomaly_type":["coating"],"items":[{"axis":"thickness","type":"under_app","zone_id":"A-2","action":"increase_dwell"}]},"sequence_data_ack":{"stored":true,"duplicate":false,"frames_received":527,"frames_accepted":527}}
```

도막 상태 enum은 `under_app`, `normal`, `over_app`이다. 정상 시에는 SSE를 보내지 않으므로 도막 anomaly item에는 `under_app` 또는 `over_app`만 나타난다.

#### 3.11.2 경로 이상만 발생한 경우

```text
id: anomaly-SESSION-001-13
event: anomaly_detection
data: {"event_id":"anomaly-SESSION-001-13","sequence_id":13,"text_guide":"손떨림으로 경로가 약 45.8mm 이탈했습니다. 손목을 고정하고 좌우 일직선 경로를 유지하세요.","anomaly":{"detected":true,"anomaly_type":["trajectory"],"items":[{"axis":"trajectory","type":"trajectory_deviation","zone_id":"A-2","deviation_mm":45.8,"action":"stabilize_wrist"}]},"sequence_data_ack":{"stored":true,"duplicate":false,"frames_received":486,"frames_accepted":486}}
```

경로 이상 `type`은 다음 세 값만 사용한다.

`trajectory_metrics.direction`은 다음 다섯 값을 사용한다.

| `direction` | 표시 의미 |
|---|---|
| `horizontal` | 가로 |
| `vertical` | 세로 |
| `diagonal` | 사선 |
| `curved_irregular` | 곡선·불규칙 |
| `stationary` | 제자리·정지 |

`direction`은 기계 판정용 enum이고 `stroke_pattern`은 사용자 표시용 문자열이다. 현재 방향과 목표 방향이 다를 때만 `direction_error`를 생성하며 세부 분류 임계값은 궤적 분석 알고리즘이 소유한다.

| `type` | 의미 | 추가 시각화 필드 | `action` 예시 |
|---|---|---|---|
| `direction_error` | 방향성 오류 | `current_direction`, `expected_direction` | `correct_pass_direction` |
| `trajectory_deviation` | 손떨림·궤적 이탈 | `deviation_mm` | `stabilize_wrist` |
| `distance_speed_error` | 거리·속도 오차 | `distance_mm`, `target_distance_mm`, `speed_mms`, `target_speed_mms` | `adjust_distance_speed` |

```json
{
  "axis": "trajectory",
  "type": "direction_error",
  "zone_id": "A-1",
  "current_direction": "vertical",
  "expected_direction": "horizontal",
  "action": "correct_pass_direction"
}
```

안내 문구: `현재 가로 50% 겹침 패스 모드이지만 세로 방향으로 도장하고 있습니다. 노즐 이동 방향을 가로 일직선으로 조정하세요.`

```json
{
  "axis": "trajectory",
  "type": "trajectory_deviation",
  "zone_id": "A-2",
  "deviation_mm": 45.8,
  "action": "stabilize_wrist"
}
```

안내 문구: `손떨림으로 경로가 약 45.8mm 이탈했습니다. 손목을 고정하고 좌우 일직선 경로를 유지하세요.`

```json
{
  "axis": "trajectory",
  "type": "distance_speed_error",
  "zone_id": "A-3",
  "distance_mm": 450,
  "target_distance_mm": 400,
  "speed_mms": 2200,
  "target_speed_mms": 2800,
  "action": "adjust_distance_speed"
}
```

안내 문구: `이격거리가 450mm로 기준보다 멀고 이동 속도가 느립니다. 이격거리를 약 400mm로 줄이고 이동 속도를 2800mm/s 수준으로 높이세요.`

#### 3.11.3 도막·경로 이상이 함께 발생한 경우

```text
id: anomaly-SESSION-001-14
event: anomaly_detection
data: {"event_id":"anomaly-SESSION-001-14","sequence_id":14,"text_guide":"A-2 구역의 도막 두께가 부족하고 경로가 약 45.8mm 이탈했습니다. 해당 구역의 체류 시간을 늘리고 손목을 고정해 좌우 일직선 경로를 유지하세요.","anomaly":{"detected":true,"anomaly_type":["coating","trajectory"],"items":[{"axis":"thickness","type":"under_app","zone_id":"A-2","action":"increase_dwell"},{"axis":"trajectory","type":"trajectory_deviation","zone_id":"A-2","deviation_mm":45.8,"action":"stabilize_wrist"}]},"sequence_data_ack":{"stored":true,"duplicate":false,"frames_received":512,"frames_accepted":512}}
```

`anomaly.detected=false`이면 `anomaly_detection` 이벤트를 Push하지 않는다.

### 3.12 Trajectory 응답 경계

내부 trajectory coaching HTTP, 시뮬레이터 HTTP와 시뮬레이터 SSE를 서로 다른 경계로 취급한다.

| 경계 | 목적 | 전달 범위 |
|---|---|---|
| 내부 trajectory coaching HTTP | 분석 모듈 간 원시 결과 전달 | `trajectory_metrics`, `expert_comparison`, `coaching_message` 전체 |
| 시뮬레이터 HTTP | 시퀀스 완료 후 전체 결과 조회 | 공통 응답과 namespaced `trajectory_feedback`, 정규화 `anomaly` |
| 시뮬레이터 SSE | 이상 즉시 시각화·행동 안내 | `text_guide`, 정규화 `anomaly`, `sequence_data_ack` 요약 |

내부 분석 필드를 시뮬레이터 응답 최상위에 평탄화하지 않는다. 시뮬레이터 HTTP는 전체 trajectory 결과를 `trajectory_feedback` 아래에 두고 SSE에는 전체 지표를 반복하지 않는다.

#### 3.12.1 내부 trajectory coaching HTTP

```json
{
  "status": "SUCCESS",
  "code": 200,
  "message": "Sequence trajectory coaching analysis completed",
  "data": {
    "session_id": "1_1000053_1",
    "sequence_id": "SEQ-005",
    "user_id": "W-017",
    "trajectory_metrics": {
      "direction": "horizontal",
      "stroke_pattern": "Horizontal Stroke (가로 패스)",
      "straightness_score": 72,
      "spatial_span_x_m": 0.4,
      "spatial_span_y_m": 0,
      "expert_deviation_mm": 45.8,
      "expert_3d_shape_dev_mm": 45.8
    },
    "expert_comparison": {
      "mode_key": "MODE_1",
      "target_pattern": "가로(Horizontal) 50% 겹침 패스",
      "user_pattern": "Horizontal Stroke (가로 패스)",
      "target_speed_mms": 2800,
      "user_speed_mms": 2200,
      "speed_gap_mms": -600,
      "target_distance_mm": 400,
      "user_distance_mm": 450,
      "distance_gap_mm": 50,
      "straightness_score": 72,
      "expert_3d_shape_dev_mm": 45.8
    },
    "coaching_message": "가로 패스 방향은 적절합니다. 손목을 고정하고 이격거리를 약 400mm로 줄이며 이동 속도를 2800mm/s 수준으로 높이세요."
  }
}
```

#### 3.12.2 시뮬레이터 HTTP

내부 `sequence_id="SEQ-005"`는 시뮬레이터 계약의 세션 내 순번 `sequence_id=5`에 대응한다. 내부 식별자와 외부 순번을 같은 필드로 혼용하지 않는다.

```json
{
  "status": "SUCCESS",
  "code": 200,
  "message": "Sequence analysis completed",
  "data": {
    "event_id": "sequence-SESSION-001-5",
    "request_id": "req-sequence-0005",
    "session_id": "SESSION-001",
    "sequence_id": 5,
    "user_id": "W-017",
    "scenario_id": "1-A",
    "generated_at": "2026-08-07T10:15:00Z",
    "latency_ms": 185,
    "text_guide": "가로 패스 방향은 적절합니다. 손목을 고정하고 이격거리를 약 400mm로 줄이며 이동 속도를 2800mm/s 수준으로 높이세요.",
    "sequence_data_ack": {
      "stored": true,
      "duplicate": false,
      "frames_received": 486,
      "frames_accepted": 486
    },
    "trajectory_feedback": {
      "trajectory_metrics": {
        "direction": "horizontal",
        "stroke_pattern": "Horizontal Stroke (가로 패스)",
        "straightness_score": 72,
        "spatial_span_x_m": 0.4,
        "spatial_span_y_m": 0,
        "expert_deviation_mm": 45.8,
        "expert_3d_shape_dev_mm": 45.8
      },
      "expert_comparison": {
        "mode_key": "MODE_1",
        "target_pattern": "가로(Horizontal) 50% 겹침 패스",
        "user_pattern": "Horizontal Stroke (가로 패스)",
        "target_speed_mms": 2800,
        "user_speed_mms": 2200,
        "speed_gap_mms": -600,
        "target_distance_mm": 400,
        "user_distance_mm": 450,
        "distance_gap_mm": 50,
        "straightness_score": 72,
        "expert_3d_shape_dev_mm": 45.8
      },
      "coaching_message": "가로 패스 방향은 적절합니다. 손목을 고정하고 이격거리를 약 400mm로 줄이며 이동 속도를 2800mm/s 수준으로 높이세요."
    },
    "anomaly": {
      "detected": true,
      "anomaly_type": ["trajectory"],
      "items": [
        {
          "axis": "trajectory",
          "type": "trajectory_deviation",
          "zone_id": "A-2",
          "deviation_mm": 45.8,
          "action": "stabilize_wrist"
        },
        {
          "axis": "trajectory",
          "type": "distance_speed_error",
          "zone_id": "A-2",
          "distance_mm": 450,
          "target_distance_mm": 400,
          "speed_mms": 2200,
          "target_speed_mms": 2800,
          "action": "adjust_distance_speed"
        }
      ]
    }
  }
}
```

#### 3.12.3 시뮬레이터 SSE

```text
id: anomaly-SESSION-001-5
event: anomaly_detection
data: {"event_id":"anomaly-SESSION-001-5","sequence_id":5,"text_guide":"손목을 고정하고 이격거리를 약 400mm로 줄이며 이동 속도를 2800mm/s 수준으로 높이세요.","anomaly":{"detected":true,"anomaly_type":["trajectory"],"items":[{"axis":"trajectory","type":"trajectory_deviation","zone_id":"A-2","deviation_mm":45.8,"action":"stabilize_wrist"},{"axis":"trajectory","type":"distance_speed_error","zone_id":"A-2","distance_mm":450,"target_distance_mm":400,"speed_mms":2200,"target_speed_mms":2800,"action":"adjust_distance_speed"}]},"sequence_data_ack":{"stored":true,"duplicate":false,"frames_received":486,"frames_accepted":486}}
```

SSE에는 `trajectory_metrics`, `expert_comparison` 전체를 포함하지 않는다.

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

`use_rag=false` 또는 `retrieval_mode=disabled`는 문서 검색만 비활성화한다. 현재 세션 상태, 누적 시퀀스, 최근 도막·순서·자세 판정 및 MCP 결과는 계속 사용할 수 있다.

### 4.2 `task_type`

Query 예시는 다음 네 종류로 구분한다.

| `task_type` | 설명 |
|---|---|
| `trajectory_recomm` | 다음 구역과 권장 이동 경로 안내 |
| `pose_recomm` | 현재 자세 비교와 교정 안내 |
| `error_analysis` | 도막·경로 이상 원인과 조치 안내 |
| `knowledge_search` | 일반 도장 지식과 시방서 검색 |

### 4.3 Query 응답 필드

| 필드 | 타입 | 필수 | 설명 |
|---|---|:---:|---|
| `answer` | String | Y | 질문에 대한 최종 답변 |
| `resolved_intents` | Array[String] | Y | 서버가 해석한 질문 의도 목록 |
| `context_status` | String | Y | `available`, `not_available` |
| `sequence_context` | Object 또는 Null | Y | 현재·예상·다음 구역과 순서 상태 |
| `coating_context` | Object 또는 Null | Y | 최근 도막 판정 근거 |
| `pose_context` | Object 또는 Null | Y | 최근 자세 판정 근거 |
| `retrieval_used` | Boolean | Y | 실제 RAG 검색 실행 여부 |
| `citations` | Array[String] | Y | RAG 문서 ID. 미사용 시 빈 배열 |

`route`와 `tool_calls_made`는 공개 응답에 포함하지 않고 서버 운영 로그에만 기록한다.

### 4.4 `trajectory_recomm` 예시

#### 요청

```json
{
  "session_id": "SESSION-001",
  "user_id": "W-017",
  "prompt": "다음 도장 구역과 권장 이동 경로를 알려줘.",
  "task_type": "trajectory_recomm",
  "use_rag": false
}
```

#### 응답

```json
{
  "status": "SUCCESS",
  "code": 200,
  "message": "Copilot query completed",
  "data": {
    "request_id": "req-query-0041",
    "session_id": "SESSION-001",
    "user_id": "W-017",
    "scenario_id": "1-A",
    "generated_at": "2026-08-06T14:32:40Z",
    "latency_ms": 180,
    "answer": "현재 A-2 구역을 작업하고 있으며 다음 권장 구역은 A-3입니다. 가로 일직선 패스를 유지하며 A-3 구역으로 이동하세요.",
    "resolved_intents": ["trajectory_recommendation"],
    "context_status": "available",
    "sequence_context": {
      "current_zone_id": "A-2",
      "next_zone_id": "A-3",
      "expected_direction": "horizontal",
      "action": "continue_to_next_zone"
    },
    "coating_context": null,
    "pose_context": null,
    "retrieval_used": false,
    "citations": []
  }
}
```

### 4.5 `pose_recomm` 예시

#### 요청

```json
{
  "session_id": "SESSION-001",
  "user_id": "W-017",
  "prompt": "현재 자세에서 교정할 부분을 알려줘.",
  "task_type": "pose_recomm",
  "use_rag": false
}
```

#### 응답

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
    "latency_ms": 210,
    "answer": "숙련자 자세(EP-001) 대비 유사도는 82%입니다. 오른쪽 팔꿈치 pitch가 기준보다 5° 큽니다.",
    "resolved_intents": ["pose_recommendation"],
    "context_status": "available",
    "sequence_context": null,
    "coating_context": null,
    "pose_context": {
      "mode": "compare",
      "expert_pose_id": "EP-001",
      "similarity_score": 0.82,
      "corrections": [
        {
          "joint": "elbow_r",
          "label": "오른쪽 팔꿈치",
          "distance": 0.08,
          "angle_corrections": {
            "pitch": 5.0
          },
          "correction_text": "오른쪽 팔꿈치: pitch 5° 과다"
        }
      ],
      "summary": "숙련자 자세(EP-001) 대비 유사도 82%. 교정 필요 관절 1개: 오른쪽 팔꿈치: pitch 5° 과다"
    },
    "retrieval_used": false,
    "citations": []
  }
}
```

### 4.6 `error_analysis` 예시

#### 요청

```json
{
  "session_id": "SESSION-001",
  "user_id": "W-017",
  "prompt": "A-2 구역의 저도막 원인과 조치 방법을 알려줘.",
  "task_type": "error_analysis",
  "use_rag": false
}
```

#### 응답

```json
{
  "status": "SUCCESS",
  "code": 200,
  "message": "Copilot query completed",
  "data": {
    "request_id": "req-query-0043",
    "session_id": "SESSION-001",
    "user_id": "W-017",
    "scenario_id": "1-A",
    "generated_at": "2026-08-06T14:33:40Z",
    "latency_ms": 195,
    "answer": "A-2 구역은 이동 속도가 빠르고 체류 시간이 부족해 저도막이 발생했습니다. 해당 구역의 체류 시간을 늘리세요.",
    "resolved_intents": ["coating_error_analysis"],
    "context_status": "available",
    "sequence_context": null,
    "coating_context": {
      "sequence_id": 12,
      "zone_id": "A-2",
      "status": "under_app",
      "thickness_um": 118,
      "target_thickness_um": 160,
      "action": "increase_dwell"
    },
    "pose_context": null,
    "retrieval_used": false,
    "citations": []
  }
}
```

### 4.7 `knowledge_search` 예시

#### 요청

```json
{
  "session_id": null,
  "user_id": "W-017",
  "prompt": "저도막이 무엇인지 설명해줘.",
  "task_type": "knowledge_search",
  "use_rag": true
}
```

#### 응답

```json
{
  "status": "SUCCESS",
  "code": 200,
  "message": "Copilot query completed",
  "data": {
    "request_id": "req-query-0044",
    "session_id": null,
    "user_id": "W-017",
    "scenario_id": null,
    "generated_at": "2026-08-06T14:34:05Z",
    "latency_ms": 184,
    "answer": "저도막은 실제 도막 두께가 요구된 최소 도막 두께보다 낮은 상태입니다.",
    "resolved_intents": ["knowledge_search"],
    "context_status": "not_available",
    "sequence_context": null,
    "coating_context": null,
    "pose_context": null,
    "retrieval_used": true,
    "citations": ["SPEC-4.2-COATING"]
  }
}
```

---

## 5. 세션 종료 리포트

### 5.1 생성 규칙

- `POST /sessions/complete`는 최종 히트맵과 도막·순서·자세 세션 통계를 확정하고 리포트 생성을 시작한다.
- 리포트 생성은 장시간 실행될 수 있으므로 `202 Accepted`와 `report_id`를 먼저 반환한다.
- 클라이언트는 `GET /sessions/{session_id}/reports/{report_id}`로 결과를 조회한다.
- 리포트가 완료된 후 `/sessions/end`를 호출해 세션 상태와 SSE 연결을 종료한다.
- 도막 수치와 분류는 콘텐츠센터, 순서 판정은 순서 검증 MCP, 자세 판정은 자세 추정 MCP의 결과를 그대로 사용한다.
- 분석 근거가 부족한 축은 누적값을 추정하지 않고 해당 하위 리포트에 `analysis_status="not_available"`과 `reason_code`를 반환한다.
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
| `coating_report` | Object | Y | 도막 누적 판정 또는 분석 불가 상태 |
| `sequence_report` | Object | Y | 순서 준수 누적 판정 또는 분석 불가 상태 |
| `pose_report` | Object | Y | 자세 누적 판정 또는 분석 불가 상태 |
| `llm_briefing` | String | Y | LLM 자연어 최종 브리핑 |

각 하위 리포트는 `analysis_status`와 `reason_code`를 필수로 가진다. `analysis_status="available"`인 도막·순서 리포트는 `algorithm_version`과 구조화된 누적 판정 필드를 제공한다.

- `coating_report`: `source_artifact_id`, `coating_distribution_index`, `zone_summary`, `distribution`, `reachability`, `worst_zones`, `root_cause`, `guideline`
- `sequence_report`: `recommended_flow`, `visited_flow`, `adherence_rate`, `deviation_count`, `deviations`, `guideline`
- `pose_report`: 현재 구현은 누적 자세 평가를 지원하지 않으므로 `analysis_status="not_available"`과 `reason_code`만 제공한다.

### 5.4 최종 리포트 수치 규칙

- `coating_report.zone_summary`는 면적 비율이며 여섯 값의 합은 허용 오차 내에서 1.0이어야 한다.
- `coating_report.distribution`은 세그먼트 개수이며 `below + in_band + above`가 전체 세그먼트 수다.
- `coating_report.reachability`는 비정상 세그먼트만 집계한다.
- 다음 관계가 성립해야 한다.

```text
coating_report.reachability.reachable + coating_report.reachability.needs_process
= coating_report.distribution.below + coating_report.distribution.above
```

제공된 샘플은 `81 + 75 = 151 + 5 = 156`으로 위 규칙을 만족한다.

- `behavior_r2`, `process_r2`의 범위는 0~1이다.
- `coating_distribution_index`의 공식과 범위는 콘텐츠센터 알고리즘 정의서에서 별도로 버전 관리한다.
- `sequence_report.adherence_rate`의 범위는 0~1이다.
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
      "coating_report": {
        "analysis_status": "available",
        "reason_code": null,
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
        "guideline": "부족 구역을 더 오래 또는 반복 분사하고 공정 설정 조정이 필요한 구역을 구분하세요."
      },
      "sequence_report": {
        "analysis_status": "available",
        "reason_code": null,
        "algorithm_version": "sequence-session-report-1.0",
        "recommended_flow": ["A-1", "A-2", "A-3"],
        "visited_flow": ["A-1", "A-2", "A-3"],
        "adherence_rate": 1.0,
        "deviation_count": 0,
        "deviations": [],
        "guideline": "권장 순서를 유지했습니다. 다음 훈련에서도 같은 구역 흐름을 유지하세요."
      },
      "pose_report": {
        "analysis_status": "not_available",
        "reason_code": "reference_data_missing"
      },
      "llm_briefing": "권장 작업 순서는 유지했지만 저도막 영역이 넓게 분포했습니다. 부족 구역의 체류 시간을 늘리세요. 현재 구현에서는 자세 누적 평가를 제공하지 않습니다."
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
- 중복 시퀀스는 다시 저장하거나 다시 판정하지 않고 기존 `event_id`를 반환한다.
- `anomaly_detection` SSE는 별도 알림이므로 `event_id`로 중복 제거한다.
- Query 재전송은 동일한 `request_id` 또는 idempotency 기록으로 중복 생성을 방지한다.
- 세션 완료 요청을 재전송하면 새 리포트를 만들지 않고 기존 `report_id`를 반환한다.

---

## 7. SSE 연결 규칙

- 도막 또는 경로 이상 결과가 `anomaly.detected=true`인 경우에만 `anomaly_detection` 이벤트를 Push하고 SSE `id:`는 본문의 `event_id`와 동일하게 둔다.
- `anomaly_detection` 이벤트의 `anomaly.items[].axis`는 `thickness`, `trajectory`만 허용한다. 자세 교정 필요 상태는 이상 항목이나 이벤트 발생 조건으로 사용하지 않는다.
- `feedback_bundle` 또는 `sequence_feedback_bundle`은 시뮬레이터에 전송하지 않는다.
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
8. 순서 이탈 유형과 `adherence_rate` 산출 기준
9. 자세 기준 데이터 ID 체계, 관절별 허용 편차 및 점수 산출 기준
10. Query 및 최종 리포트의 목표 처리시간과 timeout
11. 리포트 보존 기간과 재생성 정책
