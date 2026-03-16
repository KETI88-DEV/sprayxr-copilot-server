# Export 자동화 프로토콜

## 목적
이 문서는 RealCOAT 결과를 트레이너 파이프라인으로 전달하기 위한 API/파일 기반 Export 인터페이스를 정의한다.
현재 단계는 수동 업로드를 기본으로 하며, 자동 감지는 향후 확장으로 분리한다.

## 현재 운영 방식
- 사용자 또는 운영자가 RealCOAT 산출물을 수동 업로드한다.
- 업로드 대상: `session_summary.json` 및 연관 미디어(히트맵, 스크린샷, 로그)
- 트레이너 측은 파일 수신 후 스키마 검증을 수행하고, tool-calling 입력으로 매핑한다.

## 향후 자동화 방향
- 지정 디렉터리 또는 API 엔드포인트에서 신규 세션 파일 자동 감지
- 감지 즉시 검증/인덱싱 후 에이전트 분석 큐에 등록
- 실패 시 재시도 정책 및 운영 알림(Webhook/Slack 등) 연동

## 인터페이스 옵션

### 1) 파일 기반 인터페이스
- 입력 루트: `/exports/realcoat/`
- 세션 단위 디렉터리: `/exports/realcoat/{session_id}/`
- 필수 파일: `session_summary.json`
- 선택 파일: `heatmap.png`, `screenshot.png`, `sequence_log.json`, `motion_data.json`

### 2) API 기반 인터페이스
- Endpoint 예시: `POST /integration/realcoat/session`
- Content-Type: `application/json`
- Payload: `session_summary.json` 스키마와 동일

## session_summary.json 스키마 (필드 명세)
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "SessionSummary",
  "type": "object",
  "properties": {
    "session_id": {"type": "string", "description": "세션 고유 ID"},
    "project_id": {"type": "string", "description": "프로젝트/선박 블록 식별자"},
    "operator_id": {"type": "string", "description": "작업자 식별자"},
    "started_at": {"type": "string", "format": "date-time", "description": "작업 시작 시각(ISO 8601)"},
    "ended_at": {"type": "string", "format": "date-time", "description": "작업 종료 시각(ISO 8601)"},
    "duration_sec": {"type": "number", "minimum": 0, "description": "총 작업 시간(초)"},
    "zone_ids": {
      "type": "array",
      "items": {"type": "string"},
      "description": "작업 대상 구역 목록"
    },
    "heatmap_path": {"type": "string", "description": "히트맵 파일 경로"},
    "screenshot_path": {"type": ["string", "null"], "description": "대표 스크린샷 경로(선택)"},
    "motion_data_path": {"type": ["string", "null"], "description": "모션 데이터 파일 경로(선택)"},
    "sequence_log_path": {"type": ["string", "null"], "description": "시퀀스 로그 파일 경로(선택)"},
    "thickness_points": {
      "type": "array",
      "description": "위치별 DFT 측정값",
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
    "metrics": {
      "type": "object",
      "properties": {
        "speed_score": {"type": "number", "minimum": 0, "maximum": 100},
        "distance_score": {"type": "number", "minimum": 0, "maximum": 100},
        "angle_score": {"type": "number", "minimum": 0, "maximum": 100},
        "sequence_compliance": {"type": "number", "minimum": 0, "maximum": 1},
        "efficiency": {"type": "number", "minimum": 0, "maximum": 1}
      },
      "required": ["speed_score", "distance_score", "angle_score", "sequence_compliance", "efficiency"]
    },
    "quality_flags": {
      "type": "object",
      "description": "품질 경고/이슈 플래그",
      "additionalProperties": {"type": ["boolean", "string", "number"]}
    },
    "export_version": {"type": "string", "description": "Export 포맷 버전"}
  },
  "required": [
    "session_id",
    "project_id",
    "operator_id",
    "started_at",
    "ended_at",
    "duration_sec",
    "zone_ids",
    "heatmap_path",
    "thickness_points",
    "metrics",
    "export_version"
  ]
}
```

## 연동 체크리스트
- `session_summary.json` 필수 필드 누락 여부 검증
- 파일 경로 존재성 검증(히트맵/선택 파일)
- `export_version` 기반 역호환 처리
- 수동 업로드 실패 시 운영 로그와 재처리 큐 기록
