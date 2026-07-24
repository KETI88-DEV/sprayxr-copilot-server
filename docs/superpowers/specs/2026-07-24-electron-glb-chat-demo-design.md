# Electron·React 채팅 데모 설계 (1차: 3D 없이)

## 1. 목표

기존 선박 도장 AI Co-Pilot의 Gradio 데모를 유지하면서 두 인터페이스를 추가한다.

1. React 기반 웹 데모
2. 동일 React 빌드를 사용하는 Electron 데스크톱 앱

1차 구현에서는 3D 뷰어를 넣지 않는다. `zone_id`를 3D 좌표에 매핑할 수 없기 때문이다. 대신 채팅과 “다음 도장 순서” 목록을 구조화된 데이터로 제공한다. 3D 시각화는 향후 단계에서 추가한다.

## 2. 확정된 범위 (1차)

### 포함
- 기존 `trainer/scripts/demo_app.py` Gradio 실행 경로 유지
- 공통 Python 서비스 계층과 FastAPI 데모 API
- React + TypeScript + Vite 웹 UI
- Electron 데스크톱 셸이 동일 React 빌드를 로드
- AI 채팅과 스트리밍 응답
- 구조화된 “도장 순서” 패널: 현재 구역, 다음 구역, 완료 구역, 대기 구역
- RAG·도구 호출·XR 로그 접이식 패널
- Demo/Live 모드와 기존 AgentPipeline 응답 재사용
- 자동 검증(Python, React, API, Playwright, Electron smoke)과 수동 QA

### 제외 (1차)
- Three.js/GLB/FBX 뷰어와 3D 좌표 마커
- Blender 변환 스크립트 (자산 매핑이 확보되기 전에는 사용하지 않음)
- `zone_id`↔메시 노드 자동 매핑
- Unity/XR 런타임 변경
- 실시간 센서 데이터 연동
- Electron 배포 서명·설치 프로그램

### 3D 재도입 조건 (별도 후속 설계)
아래 조건 중 하나가 충족되면 별도 spec에서 3D 계층을 추가한다.
- 레벨별 GLB/FBX와 각 `zone_id`가 대응되는 노드 이름 또는 표면 ID가 제공됨
- 또는 좌표 마커 JSON(`zone_id → [x, y, z]`)이 실제 자산과 함께 승인됨

## 3. 사용자 경험

### 화면 구성 (1차)
- 상단 상태바: 앱 이름, DEMO/LIVE 상태, API·MCP 연결 상태
- 좌측 패널: 작업 순서 목록(현재 · 다음 · 완료 · 대기 상태 배지)
- 중앙·우측 패널: AI 채팅과 스트리밍 답변, 예시 질문, 전송·중단·초기화
- 하단 접이식 패널: RAG 청크, MCP 도구 호출, XR/scene action 원본 로그

### 순서 표현 규칙
- 현재 구역: 청록색 배지 + `현재`
- 다음 구역: 주황색 번호 배지
- 완료 구역: 흐린 녹색 배지 + `완료`
- 대기 구역: 중성 회색 배지

## 4. 아키텍처

```text
                   ┌─────────────────────┐
Browser ──────────▶│ React/Vite Web UI   │
                   │ Chat + Sequence     │
                   └─────────┬───────────┘
                             │ HTTP + SSE
Electron BrowserWindow ──────┘
                             ▼
                   ┌─────────────────────┐
                   │ FastAPI Demo API    │
                   │ chat/scene/health   │
                   └──────┬───────┬──────┘
                          │       │
                   demo service  AgentPipeline
                          │       │
                          └── MCP/RAG/QLoRA

Gradio demo ─────────────▶ 공유 시나리오 어댑터 / AgentPipeline
```

### 경계
- React와 Electron은 Python 구현 세부사항을 알지 않고 HTTP/SSE 계약만 사용한다.
- Electron은 UI 로딩과 데스크톱 수명주기만 담당한다. Node API는 renderer에 직접 노출하지 않는다.
- Gradio는 제거하지 않는다. 기존 실행 명령을 보존하고 시나리오 접근을 공유한다.
- AgentPipeline, RAG, MCP 내부 로직은 수정하지 않는다.

## 5. 컴포넌트

### Python
- `demo_service`: demo/live 질문 실행, 세션 관리, 시나리오와 AgentPipeline 호출
- `sequence_service`: AgentPipeline 응답에서 현재/다음/완료 구역을 구조화
- `demo_api`: health, scene(순서), chat SSE API 제공

### React
- `AppShell`: 데스크톱 레이아웃과 연결 상태
- `SequencePanel`: 순서 목록 표시(1차의 “scene” 표현)
- `ChatPanel`: 메시지, 스트리밍 상태, 중단·재시도
- `ToolLogPanel`: RAG, MCP, XR 이벤트
- `apiClient`: health, sequence, SSE chat 계약

### Electron
- main process: 단일 BrowserWindow 생성, 개발 URL 또는 패키지된 React 자산 로딩
- preload: 앱 버전·플랫폼 정보만 context bridge로 노출
- renderer: 웹과 동일한 React 앱
- 보안 기본값: `contextIsolation: true`, `nodeIntegration: false`, `sandbox: true`

## 6. 데이터 계약

### Sequence snapshot (초기 로딩용)
```json
{
  "level_id": "SP0035-L01",
  "label": "SP0035 Level 01",
  "zones": [
    {"zone_id": "A-1", "label": "상부 작업면", "status": "completed"},
    {"zone_id": "A-2", "label": "중간부 작업면", "status": "next"},
    {"zone_id": "A-3", "label": "하부 곡면", "status": "pending"}
  ],
  "current_zone_id": "A-1",
  "next_zone_ids": ["A-2", "A-3"],
  "completed_zone_ids": ["A-1"]
}
```

### Sequence update (채팅 응답에서 유도)
```json
{
  "type": "update_sequence",
  "level_id": "SP0035-L01",
  "current_zone_id": "A-1",
  "next_zone_ids": ["A-2", "A-3"],
  "completed_zone_ids": ["A-1"]
}
```

기존 AgentPipeline의 `next_path`, `steps`, `current_zone_id`를 서비스 계층에서 이 계약으로 변환한다. UI는 원본 파이프라인 응답 형태에 의존하지 않는다.

### Chat SSE 이벤트
- `session`: 세션 ID와 모드
- `status`: RAG, MCP, 추론 진행 상태
- `message_delta`: 답변 텍스트 조각
- `tool_result`: 접이식 로그에 표시할 도구 결과
- `sequence_update`: 순서 상태 갱신
- `complete`: 최종 응답 메타데이터
- `error`: 사용자 표시 메시지와 재시도 가능 여부

## 7. 오류 처리
- API 미연결: 상태바를 offline으로 전환하고 재연결 버튼 제공
- SSE 중단: 수신한 메시지 보존, 재시도 가능 상태로 종료
- live pipeline 오류: demo mode로 자동 전환하지 않고 명확히 실패 표시
- 응답에 순서 정보가 없으면 기존 순서를 유지하고 안내 문구를 표시
- Gradio는 영향받지 않고 그대로 동작해야 한다

## 8. 검증 기준 (1차)

### 자동 검증
- Python: 시퀀스 변환 함수, SSE 이벤트 순서, 계약 스키마, 라우트 테스트
- React: 시퀀스 리듀서, 채팅 스트림 중단·재시도, 상태 렌더링
- API 통합: demo 질문이 message와 sequence_update를 함께 반환
- Playwright: 질문 전송, 스트리밍 답변, 순서 목록 갱신
- Electron smoke: BrowserWindow 시작, preload 경계, React 앱과 API health 표시

### 수동 QA
1. 기존 Gradio 데모(`DEMO_MODE=true python3 trainer/scripts/demo_app.py`) 실행 후 “다음에 어떤 구역을 칠해야 해?” 정상 응답 확인
2. FastAPI + React 웹 데모에서 동일 질문 후 채팅 답변과 순서 목록이 A-1→A-2로 갱신되는지 확인
3. Electron 앱에서 동일 흐름 반복
4. API를 잠시 중단해 offline 상태와 재연결 확인

## 9. 완료 조건 (1차)
- 기존 Gradio 데모 실행 명령이 계속 동작한다
- React 웹 데모와 Electron 앱이 동일 FastAPI 계약을 사용한다
- 질문 결과가 채팅 답변과 구조화된 순서 목록을 함께 갱신한다
- 3D 자산 없이도 데모 흐름 전체를 실행할 수 있다
- 자동 검증과 웹·Electron 수동 QA가 통과한다

## 10. 결정 기록
- 1차는 3D 뷰어를 포함하지 않는다. 근거: `zone_id`↔좌표 매핑 부재
- React UI를 신규 기본 데모로 만들되 Gradio는 유지한다
- 웹과 Electron은 동일 React 빌드를 사용한다
- FBX/GLB 변환은 자산 매핑이 확보된 후속 단계로 이관
- 순서 상태는 서비스 계층에서 구조화 계약으로 변환한다
