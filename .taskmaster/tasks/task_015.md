# Task ID: 15

**Title:** Tool Calling 프레임워크 설계 및 인터페이스 정의

**Status:** pending

**Dependencies:** 9 ✓, 19 ⧖

**Priority:** high

**Description:** JSON Schema 기반 도구 정의(pose_estimation, anomaly_detection, timeseries, location_tracking, sequence_validation, dft_judge), Qwen3 native function calling + Thinking Mode 포맷 적용, 호출 규약 설계

**Details:**

1. 6개 도구의 JSON Schema 정의 (PRD §16.3 + Architecture §3.2.6 기준)
2. Architecture §3.2.4에서 </ToolPos>, </ToolSeq> 등 custom tag 방식을 제안했으나, Qwen3는 native function calling + thinking mode를 지원하므로 표준 FC 포맷으로 구현 (Architecture의 tag 개념을 FC tool_calls로 매핑). Qwen3의 <think> 블록 내에서 도구 필요성을 추론한 뒤 tool_calls를 생성하는 방식.
3. Qwen3(4B/8B)의 tool/function calling 포맷에 맞춘 tool description 작성 (Qwen-Agent 호환)
4. 도구 호출 규약: request/response 포맷, 에러 코드, 타임아웃
5. 각 도구별 입출력 예시 + 테스트 케이스
6. Thinking Mode 연동: 복잡한 요청 시 /think → 도구 선택 추론 → tool_calls, 단순 조회 시 /no_think → 직접 호출
7. 산출물: /workspace/trainer/configs/tool_schemas/*.json, /workspace/trainer/scripts/tool_calling_adapter.py, /workspace/trainer/scripts/test_tool_calling.py

**Test Strategy:**

JSON Schema 유효성 검증, Qwen3 function calling 포맷 테스트 (tool choice + tool response 파싱 + thinking mode 연동), 6개 도구 모두 정의 완료 확인

## Subtasks

### 15.1. 6개 도구 JSON Schema 정의

**Status:** pending  
**Dependencies:** None  

PRD §16.3 + Architecture §3.2.6 기준으로 pose_estimation, anomaly_detection, timeseries, location_tracking, sequence_validation, dft_judge 6개 도구의 JSON Schema를 정의

**Details:**

각 도구별 입력/출력 JSON Schema를 /workspace/trainer/configs/tool_schemas/ 디렉터리에 개별 파일로 작성. Architecture §3.2.6의 MCP 모듈 정의와 PRD §16.3의 입출력 명세를 통합 반영. 스키마에는 name, description, parameters(type, properties, required), response_schema를 포함. Qwen2.5 function calling 호환 포맷 적용.

### 15.2. Qwen3 Function Calling 포맷 어댑터 구현

**Status:** pending  
**Dependencies:** 15.1  

Qwen3(4B/8B)의 tool/function calling 메시지 포맷에 맞춘 프롬프트 템플릿 및 파서 구현. Architecture §3.2.4의 Tool Tag 개념을 Qwen3 native FC + Thinking Mode로 매핑

**Details:**

Architecture에서 </ToolPos>, </ToolSeq> 등 custom tag 방식을 제안했으나, Qwen3는 native function calling + thinking mode를 지원하므로 표준 FC 포맷으로 구현. tools 파라미터에 JSON Schema 전달 → 모델이 <think> 블록에서 추론 후 tool_calls로 응답 → 파싱하여 MCP 호출로 라우팅하는 어댑터. enable_thinking 토글 지원. Qwen-Agent 호환 tool-call template 사용. 산출물: /workspace/trainer/scripts/tool_calling_adapter.py

### 15.3. 도구 호출 규약 설계 (request/response/error)

**Status:** pending  
**Dependencies:** 15.1  

Tool Calling → MCP Server 간 호출 규약 정의. Architecture §3.2.5의 양방향 통신(파란 양방향 화살표) 구현 스펙

**Details:**

request 포맷: {tool_name, parameters, call_id, timeout_ms}. response 포맷: {call_id, status, result, error}. 에러 코드 체계: TOOL_NOT_FOUND, INVALID_PARAMS, TIMEOUT, SERVER_ERROR. 타임아웃 기본값 30초. 재시도 정책: 최대 1회. 산출물: /workspace/trainer/configs/tool_calling_protocol.json

### 15.4. 도구별 입출력 예시 및 통합 테스트

**Status:** pending  
**Dependencies:** 15.2, 15.3  

6개 도구 각각에 대해 입출력 예시 작성 + tool calling adapter end-to-end 테스트

**Details:**

각 도구별 최소 2개 예시 (정상 케이스 + 엣지 케이스). PRD §16.4 시나리오 D/E/F에서 사용되는 실제 호출 패턴을 예시에 포함. test_tool_calling.py에서 mock MCP 응답 기반 어댑터 테스트. 산출물: /workspace/trainer/configs/tool_schemas/examples/, /workspace/trainer/scripts/test_tool_calling.py
