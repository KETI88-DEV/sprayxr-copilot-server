# Rule Engine 인터페이스 계약서

## 목적
이 문서는 RealCOAT가 이미 계산하는 점수 지표를 에이전트 파이프라인과 연동하기 위한 입력/출력 계약(Integration Contract)을 정의한다.
실제 점수 계산 로직은 Rule Engine 구현체 또는 RealCOAT 내부 모듈이 담당하며, 본 문서는 데이터 경계만 명시한다.

## 입력 데이터
- `motion_data`: 분사기 이동 궤적과 시간 기반 동작 정보
- `thickness_points`: 위치별 도막 두께 측정 포인트
- `sequence_log`: 작업 순서 이벤트 로그

## 출력 데이터
- `speed_score`: 속도 준수 점수 (0~100)
- `distance_score`: 거리 준수 점수 (0~100)
- `angle_score`: 각도 준수 점수 (0~100)
- `sequence_compliance`: 시퀀스 준수율 (0~1)
- `efficiency`: 작업 효율 지표 (0~1)

## 입력 JSON Schema 예시
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "RuleEngineInput",
  "type": "object",
  "properties": {
    "session_id": {"type": "string"},
    "motion_data": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "timestamp": {"type": "number", "description": "epoch seconds"},
          "speed": {"type": "number", "description": "m/s"},
          "distance": {"type": "number", "description": "mm"},
          "angle": {"type": "number", "description": "degree"}
        },
        "required": ["timestamp", "speed", "distance", "angle"]
      }
    },
    "thickness_points": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "zone_id": {"type": "string"},
          "x": {"type": "number"},
          "y": {"type": "number"},
          "dft": {"type": "number", "description": "um"}
        },
        "required": ["zone_id", "x", "y", "dft"]
      }
    },
    "sequence_log": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "timestamp": {"type": "number"},
          "zone_id": {"type": "string"},
          "action": {"type": "string"},
          "order_index": {"type": "integer", "minimum": 0}
        },
        "required": ["timestamp", "zone_id", "action", "order_index"]
      }
    }
  },
  "required": ["session_id", "motion_data", "thickness_points", "sequence_log"]
}
```

## 출력 JSON Schema 예시
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "RuleEngineOutput",
  "type": "object",
  "properties": {
    "session_id": {"type": "string"},
    "speed_score": {"type": "number", "minimum": 0, "maximum": 100},
    "distance_score": {"type": "number", "minimum": 0, "maximum": 100},
    "angle_score": {"type": "number", "minimum": 0, "maximum": 100},
    "sequence_compliance": {"type": "number", "minimum": 0, "maximum": 1},
    "efficiency": {"type": "number", "minimum": 0, "maximum": 1}
  },
  "required": [
    "session_id",
    "speed_score",
    "distance_score",
    "angle_score",
    "sequence_compliance",
    "efficiency"
  ]
}
```

## 통합 메모
- RealCOAT는 이미 속도/거리/각도/순서 관련 핵심 지표를 계산한다.
- 본 계약은 에이전트 도구 호출 계층에서 동일 형식으로 점수 수신/소비하기 위한 표준 인터페이스다.
