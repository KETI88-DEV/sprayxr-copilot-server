# Task ID: 18

**Title:** Agent Orchestration 및 CoT 추론 파이프라인

**Status:** pending

**Dependencies:** 15, 16, 19 ⧖

**Priority:** high

**Description:** RAG Core + Tool Calling + Chain of Thought 통합 Agent 파이프라인. instruction → 도구 선택 → MCP 호출 → context 수집 → CoT 추론 → XR 콘텐츠용 구조화 응답 생성

**Details:**

1. Agent 파이프라인 설계: Architecture §4 [1]~[11] 단계 구현
   instruction parsing → tool selection → MCP call → context aggregation → CoT reasoning → structured response
2. Qwen3의 native function calling + Thinking Mode 활용한 도구 선택 로직 (Architecture §3.2.4 Thinking 역할)
   - enable_thinking=True: 복잡한 요청에서 <think> 블록으로 추론 → tool_calls 생성
   - enable_thinking=False: 단순 조회에서 직접 tool_calls 생성 (저지연)
   - /think, /no_think 소프트 스위치로 턴별 모드 전환
3. MCP Server 호출 및 응답 파싱 (Architecture §3.2.5 Tool Calling ↔ MCP Server)
4. RAG 검색 + MCP 결과 + CoT 추론 통합 (Architecture §3.2.7 전체 맥락)
5. 구조화된 응답 생성: XR 콘텐츠용 JSON 포함 (Architecture §3.2.8 Decoder + §3.2.9 Conversion Output Format)
   - <think> 블록은 사용자 응답에서 strip하고, 디버그 로그에만 기록
6. 세션 컨텍스트 순환 갱신: Architecture §6.3 구현 - 응답 후 Context 업데이트 → 다음 질문에 활용
7. Qwen-Agent 프레임워크 활용 검토 (tool-calling template + MCP 통합 내장)
8. PRD §16.4 시나리오 D/E/F 기준 end-to-end 시연
9. 산출물: /workspace/trainer/scripts/agent_pipeline.py, /workspace/trainer/scripts/demo_agent.py

**Test Strategy:**

시나리오 D (실시간 코칭) end-to-end 테스트, 시나리오 E (이상 탐지) end-to-end 테스트, 시나리오 F (XR 콘텐츠) end-to-end 테스트, 응답 JSON 스키마 유효성 확인

## Subtasks

### 18.1. Agent 파이프라인 코어 설계 (instruction → tool selection → response)

**Status:** pending  
**Dependencies:** None  

Architecture §4 End-to-End Pipeline의 [1]~[10] 단계를 구현하는 Agent 파이프라인 코어 클래스 설계

**Details:**

AgentPipeline 클래스: (1) instruction parsing (Conversion Request Format 역할), (2) context 수집 (환경 정보+사용자 입력), (3) LLM에 tools 파라미터 전달 → function calling 응답 수신, (4) tool_calls 파싱 → MCP 서버 호출, (5) MCP 결과 + RAG 검색 결과 통합 → context 구성, (6) CoT 추론 → 최종 응답 생성. 산출물: /workspace/trainer/scripts/agent_pipeline.py

### 18.2. Tool Selection 로직 (Qwen3 FC + Thinking Mode 기반 도구 선택)

**Status:** pending  
**Dependencies:** 18.1  

Qwen3의 native function calling + Thinking Mode로 도구 자동 선택하는 로직 구현. Architecture §3.2.4 Thinking의 '필요 도구 식별' 역할. <think> 블록에서 도구 필요성 추론 → tool_calls 생성.

**Details:**

T15의 6개 도구 JSON Schema를 Qwen3 tools 파라미터로 전달. Thinking Mode 활성 시 <think> 블록에서 도구 필요성 추론 후 tool_calls 생성. Non-thinking 모드에서는 직접 tool_calls 생성. 모델 응답에서 tool_calls 추출 → T15의 tool_calling_adapter로 라우팅. 다중 도구 호출(multi-tool) 지원: 단일 턴에서 여러 도구 동시 호출 가능. 도구 미사용 판단(RAG만으로 충분한 경우) 처리.

### 18.3. MCP 호출 및 Context Aggregation

**Status:** pending  
**Dependencies:** 18.1, 18.2  

Tool Calling 결과로 MCP Server 실제 호출 + RAG 검색 결과와 통합하여 CoT용 context 구성. Architecture §3.2.7 전체 맥락 역할

**Details:**

MCP 서버 HTTP 호출 (T16의 FastAPI 서버). 응답 파싱 + 에러 핸들링 (타임아웃, 서버 에러). RAG 검색: 기존 search_index.py 활용. context 통합 포맷: {mcp_results: {tool_name: result}, rag_context: [chunks], environment: {}, instruction: ''}. Architecture §6.3 Context 순환 갱신: 이전 턴의 context를 다음 턴에 전달하는 세션 관리.

### 18.4. CoT 추론 및 구조화 응답 생성

**Status:** pending  
**Dependencies:** 18.3  

통합 context 기반 Chain of Thought 추론 → 최종 구조화 응답 생성. Architecture §3.2.3 CoT + §3.2.8 Decoder 역할

**Details:**

Qwen3 Thinking Mode 활용: enable_thinking=True로 MCP 결과 + RAG 문서 + 환경 정보를 input으로, <think> 블록에서 단계적 추론 수행 → 최종 답변 도출. <think> 블록은 사용자 응답에서 strip, 디버그/평가 로그에만 기록. 응답 포맷: 일반 텍스트 답변 + 선택적 XR 콘텐츠 JSON (PRD §16.4 시나리오 F). Conversion Output Format: 응답 유형(text, xr_instruction, alert)에 따라 구조 분기. 추론 예산 제어(token budget) 파라미터 노출.

### 18.5. 세션 컨텍스트 관리 및 순환 갱신

**Status:** pending  
**Dependencies:** 18.3, 18.4  

Architecture §6.3 컨텍스트 순환 갱신 구현. 다중 턴 대화에서 이전 MCP 결과/CoT 추론을 다음 턴에 활용

**Details:**

SessionContext 클래스: session_id별로 이전 턴의 mcp_results, rag_chunks, cot_reasoning을 저장. 다음 턴에서 관련 컨텍스트를 자동 주입. 컨텍스트 윈도우 관리: 최근 3턴까지 유지, 오래된 컨텍스트는 요약 후 압축. Vector DB에 세션 컨텍스트 벡터 저장 (Architecture §3.2.7).

### 18.6. End-to-End 시연 스크립트 (시나리오 D/E/F)

**Status:** pending  
**Dependencies:** 18.4, 18.5  

PRD §16.4 시나리오 D(실시간 코칭), E(이상 탐지+의사결정), F(XR 콘텐츠 생성) 3개 시나리오 end-to-end 시연

**Details:**

demo_agent.py: 3개 시나리오를 순차 실행하여 Agent 파이프라인 전체 동작을 시연. 각 시나리오별 입력 데이터 사전 정의, MCP 서버 기동 → Agent 호출 → 응답 출력. 실행 로그에 tool_calls, mcp_results, cot_reasoning, final_response 각 단계를 기록. 산출물: /workspace/trainer/scripts/demo_agent.py
