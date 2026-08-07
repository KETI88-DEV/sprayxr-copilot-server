# Query·이상탐지 응답 계약 교정 설계

**상태:** 승인됨
**작성일:** 2026-08-07
**적용 문서:** `docs/ai_copilot_response_contract.md`

## 1. 목표

내부 trajectory coaching HTTP와 시뮬레이터 HTTP/SSE 경계를 분리하고, 시뮬레이터가 시각화와 행동 안내에 바로 사용할 수 있는 이상탐지 응답을 정의한다. Query의 네 `task_type`마다 독립된 요청·응답 예시를 제공하며 필드명, enum, 정렬 순서와 안내 문체를 일관되게 교정한다.

## 2. 경계 원칙

- `feedback_bundle`은 서버 내부 결과 조합에만 사용할 수 있으며 시뮬레이터 HTTP/SSE 응답에 포함하지 않는다.
- 시퀀스 종료 HTTP 응답은 판정 결과를 `data`에 직접 담는다.
- SSE는 `anomaly.detected=true`인 경우에만 `event: anomaly_detection`으로 Push한다.
- 정상 시퀀스는 HTTP에서 `anomaly.detected=false`를 반환하고 이상 SSE는 Push하지 않는다.
- 자세 판정은 이상탐지 대상이 아니다. `axis="pose"` 항목을 만들지 않는다.
- `source_mcp`, `region_id`, `combined`, `route`, DFT 관련 명칭은 새 예시에 사용하지 않는다.
- 위치 식별자는 `zone_id`로 통일하고 현재 값은 `A-1`, `A-2`, `A-3`만 사용한다.

| 경계 | 목적 | 전달 범위 |
|---|---|---|
| 내부 trajectory coaching HTTP | 분석 모듈 간 원시 결과 전달 | `trajectory_metrics`, `expert_comparison`, `coaching_message` 전체 |
| 시뮬레이터 HTTP | 시퀀스 완료 후 전체 결과 조회 | 공통 응답과 namespaced `trajectory_feedback`, 정규화 `anomaly` |
| 시뮬레이터 SSE | 이상 즉시 시각화·행동 안내 | `text_guide`, 정규화 `anomaly`, `sequence_data_ack` 요약 |

- 내부 분석 필드를 시뮬레이터 응답 최상위에 평탄화하지 않는다.
- 시뮬레이터 HTTP는 전체 trajectory 결과를 `trajectory_feedback` 아래에 둔다.
- SSE에는 `trajectory_metrics`, `expert_comparison` 전체를 반복하지 않는다.

## 3. `anomaly_detection` SSE 구조

### 3.1 상위 필드

| 필드 | 타입 | 설명 |
|---|---|---|
| `event_id` | String | SSE 이벤트 식별자 |
| `sequence_id` | Integer | 이상이 검출된 시퀀스 순번 |
| `text_guide` | String | 사용자에게 표시하는 중립적 안내 문구 |
| `anomaly` | Object | 이상 종류와 항목 목록 |
| `sequence_data_ack` | Object | 시퀀스 저장 결과 |
| `coating_result` | Object, optional | 도막 측정 상세 결과 |
| `citations` | Array[String], optional | 안내 생성에 사용한 문서 ID |

### 3.2 `anomaly` 필드

```json
{
  "detected": true,
  "anomaly_type": ["coating", "trajectory"],
  "items": []
}
```

- `anomaly_type`은 배열이며 허용값은 `coating`, `trajectory`다.
- 배열의 고정 순서는 `coating` 다음 `trajectory`다.
- `combined` 같은 합성 enum은 사용하지 않는다.
- `items`도 `anomaly_type` 순서와 동일하게 도막 항목 다음 경로 항목으로 정렬한다.

### 3.3 공통 item 필드

| 필드 | 타입 | 설명 |
|---|---|---|
| `axis` | String | `thickness` 또는 `trajectory` |
| `type` | String | 축별 이상 유형 |
| `zone_id` | String | 시뮬레이터가 강조 표시할 구역 |
| `action` | String | 시뮬레이터가 처리할 단일 행동 코드 |

`action`은 객체나 배열이 아닌 코드 문자열 하나다. 사용자에게 보여 줄 설명은 `text_guide`가 담당한다.

## 4. 도막 이상

도막 상태 enum은 `under_app`, `normal`, `over_app` 세 값이다. SSE는 이상 시에만 발생하므로 도막 anomaly item의 `type`에는 `under_app` 또는 `over_app`만 나타난다.

### 도막 이상만 있는 경우

```text
id: anomaly-SESSION-001-12
event: anomaly_detection
data: {"event_id":"anomaly-SESSION-001-12","sequence_id":12,"text_guide":"A-2 구역의 도막 두께가 부족합니다. 해당 구역의 체류 시간을 늘리세요.","anomaly":{"detected":true,"anomaly_type":["coating"],"items":[{"axis":"thickness","type":"under_app","zone_id":"A-2","action":"increase_dwell"}]},"sequence_data_ack":{"stored":true,"duplicate":false,"frames_received":527,"frames_accepted":527}}
```

## 5. 경로 이상

### 5.1 방향 enum

`trajectory_metrics.direction`은 다음 다섯 값을 사용한다.

| `direction` | 표시 의미 |
|---|---|
| `horizontal` | 가로 |
| `vertical` | 세로 |
| `diagonal` | 사선 |
| `curved_irregular` | 곡선·불규칙 |
| `stationary` | 제자리·정지 |

`direction`은 기계 판정용 enum이고 `stroke_pattern`은 사용자 표시용 문자열이다. 방향 분류와 방향 오류를 구분하며, 현재 방향이 목표 방향과 다를 때만 `direction_error`를 생성한다. 세부 분류 임계값은 궤적 분석 알고리즘이 소유한다.

경로 이상 item의 `axis`는 `trajectory`이며 `type`은 다음 세 값만 사용한다.

| `type` | 의미 | 추가 시각화 값 | `action` 예시 |
|---|---|---|---|
| `direction_error` | 방향성 오류 | `current_direction`, `expected_direction` | `correct_pass_direction` |
| `trajectory_deviation` | 손떨림·궤적 이탈 | `deviation_mm` | `stabilize_wrist` |
| `distance_speed_error` | 거리·속도 오차 | `distance_mm`, `target_distance_mm` | `adjust_distance_speed` |

### 5.2 방향성 오류

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

### 5.3 손떨림·궤적 이탈

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

### 5.4 거리·속도 오차

```json
{
  "axis": "trajectory",
  "type": "distance_speed_error",
  "zone_id": "A-3",
  "distance_mm": 620,
  "target_distance_mm": 400,
  "action": "adjust_distance_speed"
}
```

안내 문구: `이격거리가 620mm로 기준보다 멀고 이동 속도가 느립니다. 벽면과 약 400mm 거리를 유지하고 이동 속도를 높이세요.`

### 경로 이상만 있는 경우

```text
id: anomaly-SESSION-001-13
event: anomaly_detection
data: {"event_id":"anomaly-SESSION-001-13","sequence_id":13,"text_guide":"손떨림으로 경로가 약 45.8mm 이탈했습니다. 손목을 고정하고 좌우 일직선 경로를 유지하세요.","anomaly":{"detected":true,"anomaly_type":["trajectory"],"items":[{"axis":"trajectory","type":"trajectory_deviation","zone_id":"A-2","deviation_mm":45.8,"action":"stabilize_wrist"}]},"sequence_data_ack":{"stored":true,"duplicate":false,"frames_received":486,"frames_accepted":486}}
```

## 6. 도막·경로 복합 이상

```text
id: anomaly-SESSION-001-14
event: anomaly_detection
data: {"event_id":"anomaly-SESSION-001-14","sequence_id":14,"text_guide":"A-2 구역의 도막 두께가 부족하고 경로가 약 45.8mm 이탈했습니다. 해당 구역의 체류 시간을 늘리고 손목을 고정해 좌우 일직선 경로를 유지하세요.","anomaly":{"detected":true,"anomaly_type":["coating","trajectory"],"items":[{"axis":"thickness","type":"under_app","zone_id":"A-2","action":"increase_dwell"},{"axis":"trajectory","type":"trajectory_deviation","zone_id":"A-2","deviation_mm":45.8,"action":"stabilize_wrist"}]},"sequence_data_ack":{"stored":true,"duplicate":false,"frames_received":512,"frames_accepted":512}}
```

## 7. Trajectory 응답 경계 예시

### 7.1 내부 trajectory coaching HTTP

내부 분석 응답은 정규화 anomaly가 아니라 원시 궤적 지표, 숙련자 비교와 코칭 문구를 반환한다.

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
    "coaching_message": "가로 패스 방향은 적절합니다. 경로 이탈을 줄이기 위해 손목을 고정하고, 이격거리와 이동 속도를 목표값에 맞게 조정하세요."
  }
}
```

### 7.2 시뮬레이터 HTTP

시뮬레이터 HTTP에는 전체 분석 결과를 `trajectory_feedback` 아래에 두고 시각화용 `anomaly`를 함께 제공한다.

```json
{
  "status": "SUCCESS",
  "code": 200,
  "message": "Sequence analysis completed",
  "data": {
    "event_id": "sequence-SESSION-001-5",
    "session_id": "SESSION-001",
    "sequence_id": "SEQ-005",
    "user_id": "W-017",
    "text_guide": "가로 패스 방향은 적절합니다. 손목을 고정하고 이격거리와 이동 속도를 조정하세요.",
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
      "coaching_message": "가로 패스 방향은 적절합니다. 경로 이탈을 줄이기 위해 손목을 고정하고, 이격거리와 이동 속도를 목표값에 맞게 조정하세요."
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

### 7.3 시뮬레이터 SSE

SSE에는 전체 분석 지표를 반복하지 않고 이상 요약만 전달한다.

```text
id: anomaly-SESSION-001-5
event: anomaly_detection
data: {"event_id":"anomaly-SESSION-001-5","sequence_id":"SEQ-005","text_guide":"손목을 고정하고 이격거리와 이동 속도를 조정하세요.","anomaly":{"detected":true,"anomaly_type":["trajectory"],"items":[{"axis":"trajectory","type":"trajectory_deviation","zone_id":"A-2","deviation_mm":45.8,"action":"stabilize_wrist"},{"axis":"trajectory","type":"distance_speed_error","zone_id":"A-2","distance_mm":450,"target_distance_mm":400,"speed_mms":2200,"target_speed_mms":2800,"action":"adjust_distance_speed"}]},"sequence_data_ack":{"stored":true,"duplicate":false,"frames_received":486,"frames_accepted":486}}
```

## 8. Query `task_type` 4종

다음 네 값만 예시화한다.

1. `trajectory_recomm`
2. `pose_recomm`
3. `error_analysis`
4. `knowledge_search`

각 `task_type`에는 독립된 요청 JSON과 응답 JSON을 하나씩 제공한다. 응답은 현재 문서 구조를 유지한다.

| 필드 | 설명 |
|---|---|
| `answer` | 최종 답변 |
| `resolved_intents` | 해석된 의도 목록 |
| `context_status` | 세션 근거 사용 가능 여부 |
| `sequence_context` | 순서·경로 근거 또는 `null` |
| `coating_context` | 도막 근거 또는 `null` |
| `pose_context` | 자세 근거 또는 `null` |
| `retrieval_used` | 문서 검색 실행 여부 |
| `citations` | 문서 ID 목록 |

- `route`는 Query 응답에 추가하지 않는다.
- `trajectory_recomm`은 순서·경로 추천 질문과 `sequence_context`를 사용한다.
- `pose_recomm`은 자세 교정 질문과 `pose_context`를 사용한다.
- `error_analysis`는 도막 또는 경로 이상 원인·조치 질문을 사용한다.
- `knowledge_search`는 일반 도장 지식 질문을 사용하며 세션 context는 `null`일 수 있다.

네 예시는 다음 내용을 사용한다.

| `task_type` | 요청 `prompt` | 핵심 응답 | 주요 context |
|---|---|---|---|
| `trajectory_recomm` | `다음 도장 구역과 권장 이동 경로를 알려줘.` | 현재 `A-2`, 다음 `A-3`, 가로 일직선 패스 유지 안내 | `sequence_context` 사용 |
| `pose_recomm` | `현재 자세에서 교정할 부분을 알려줘.` | 손목 고정과 팔꿈치 위치 조정 안내 | `pose_context` 사용 |
| `error_analysis` | `A-2 구역의 저도막 원인과 조치 방법을 알려줘.` | `under_app` 근거와 `increase_dwell` 조치 안내 | `coating_context` 사용 |
| `knowledge_search` | `저도막이 무엇인지 설명해줘.` | 요구 최소 도막 두께보다 낮은 상태라는 지식 답변 | 세 context 모두 `null` |

- `trajectory_recomm`, `pose_recomm`, `error_analysis` 요청은 `session_id="SESSION-001"`을 사용한다.
- `knowledge_search` 요청은 `session_id=null`을 사용한다.
- `knowledge_search`만 `retrieval_used=true`와 비어 있지 않은 `citations`를 사용한다.
- 나머지 세 예시는 `retrieval_used=false`, `citations=[]`로 두고 현재 세션과 모듈 결과를 근거로 답한다.

## 9. 문체와 용어

- `text_guide`, `answer`, `guideline`은 존댓말 기반의 중립적 시스템 안내체로 작성한다.
- 반말, 훈계조, 역할극 어미인 `~하네`, `~하게나`, `~보게`를 사용하지 않는다.
- 위치는 `zone`, 경로는 `trajectory`, 도막은 `coating` 또는 `thickness`로 통일한다.
- 도막 상태에는 `under_app`, `normal`, `over_app` 외의 표현을 사용하지 않는다.

## 10. 검증 기준

- 문서의 모든 `json` 코드 블록과 SSE `data:` JSON이 파싱되어야 한다.
- Query 요청·응답 예시는 네 `task_type`을 정확히 한 번씩 포함해야 한다.
- 이상 SSE 예시는 `coating`, `trajectory`, 두 종류 동시 발생을 각각 포함해야 한다.
- `anomaly_type`과 `items`는 `coating` 다음 `trajectory` 순서를 지켜야 한다.
- `source_mcp`, `region_id`, `combined`, 외부 `feedback_bundle`, `sequence_feedback_bundle`, DFT 관련 용어가 없어야 한다.
- 문서 예시의 위치 ID는 `A-1`, `A-2`, `A-3`만 사용한다.
- `direction`은 `horizontal`, `vertical`, `diagonal`, `curved_irregular`, `stationary` 다섯 값만 사용한다.
- 내부 trajectory coaching HTTP에는 `trajectory_metrics`, `expert_comparison`이 최상위 분석 결과로 존재해야 한다.
- 시뮬레이터 HTTP의 전체 trajectory 결과는 `trajectory_feedback` 아래에 있어야 한다.
- 시뮬레이터 SSE에는 `trajectory_metrics`, `expert_comparison`을 포함하지 않는다.
- 교정 예시는 `expert_deviation_mm=45.8`, `target_distance_mm=400`, `user_distance_mm=450`, `distance_gap_mm=50`의 관계를 유지해야 한다.
