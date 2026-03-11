# Task ID: 17

**Title:** Agentic 평가셋 설계 및 CoT 평가 루브릭

**Status:** pending

**Dependencies:** 5 ✓, 15, 19 ⧖

**Priority:** high

**Description:** Co-Pilot형 30문항 설계(Tool Calling 6, 이상탐지 해석 6, 순서/의사결정 6, CoT 추론 6, XR 콘텐츠 생성 6). tool_calls_required, reasoning_steps, expected_tool_sequence 필드 포함

**Details:**

1. 기존 50문항(문서 QA)과 별도로 Agentic 평가셋 30문항 작성
2. 각 문항 구조: q_id, question, category, difficulty, tool_calls_required[], reasoning_steps, expected_tool_sequence[], expected_answer, evaluation_criteria
3. Tool Calling 문항: 올바른 도구 선택 + 파라미터 구성 테스트
4. 이상 탐지: 센서 데이터 → 원인 분석 → 조치 제안
5. 순서/의사결정: 조건부 분기, 우선순위 판단
6. CoT 추론: 다단계 추론 + 근거 연결
7. XR 콘텐츠: 구조화된 지시문 생성
8. CoT 평가 루브릭 작성 (추론 과정 품질 평가 기준)
9. 산출물: /data/eval/eval_set_agentic.jsonl, /workspace/trainer/configs/eval_rubric_agentic.md

**Test Strategy:**

30문항 전체 필드 완성도 확인, JSON Schema 유효성 검증, 카테고리별 6문항 균형 확인, 기존 50문항과 중복 없음 확인

## Subtasks

### 17.1. Agentic 평가셋 스키마 및 메타데이터 설계

**Status:** pending  
**Dependencies:** None  

기존 eval_set.jsonl과 호환되면서 tool_calls_required, reasoning_steps, expected_tool_sequence 등 agentic 필드를 추가한 평가셋 스키마 정의

**Details:**

스키마: {q_id, question, category, difficulty(easy/medium/hard), tool_calls_required[], reasoning_steps(int), expected_tool_sequence[], expected_answer, evaluation_criteria{}, context_data{}}. category: tool_calling, anomaly_interpretation, sequence_decision, cot_reasoning, xr_content. 산출물: 스키마 문서 + 빈 템플릿 jsonl.

### 17.2. Tool Calling + 이상 탐지 해석 문항 작성 (12문항)

**Status:** pending  
**Dependencies:** 17.1  

Tool Calling 6문항(도구 선택+파라미터 구성) + 이상 탐지 해석 6문항(센서 데이터→원인→조치). PRD §16.6 기준

**Details:**

Tool Calling: dft_judge, anomaly_detection, sequence_validation 등 다양한 도구 조합 테스트. PRD §16.4 시나리오 D/E 기반 실제 데이터 활용. 이상 탐지: DFT 편차, 온습도 이상, 작업 속도 변화 등 센서 데이터 해석 문항. 각 문항에 expected_tool_sequence와 reasoning_steps 명시.

### 17.3. 순서/의사결정 + CoT 추론 + XR 콘텐츠 문항 작성 (18문항)

**Status:** pending  
**Dependencies:** 17.1  

순서/의사결정 6문항 + CoT 추론 6문항 + XR 콘텐츠 생성 6문항. Architecture §6.1 자율 도구 호출 패턴 반영

**Details:**

순서/의사결정: 환경 조건(습도, 이슬점)에 따른 조건부 분기, 공정 우선순위 판단. CoT: 다단계 추론(블라스팅→프라이머→상도 과정), 문서 근거 연결 필수. XR 콘텐츠: PRD §16.4 시나리오 F 기반, ar_instruction JSON 생성 문항. xr_type, steps, highlight_zone, tool_angle 등 구조화 출력 요구.

### 17.4. CoT 평가 루브릭 작성

**Status:** pending  
**Dependencies:** 17.1  

Agentic 평가셋 전용 CoT 평가 루브릭. 추론 과정 품질, 도구 사용 정확도, 응답 구조화 수준 평가 기준

**Details:**

평가 차원: (1) tool_selection_accuracy: 올바른 도구 선택 0-2, (2) parameter_correctness: 파라미터 정확도 0-2, (3) reasoning_depth: CoT 추론 깊이/논리성 0-2, (4) context_integration: MCP 결과+RAG 문서 통합 수준 0-2, (5) response_structure: 응답 구조화 품질 0-2. 산출물: /workspace/trainer/configs/eval_rubric_agentic.md

### 17.5. Agentic 평가 실행 스크립트 및 검증

**Status:** pending  
**Dependencies:** 17.2, 17.3, 17.4  

30문항 평가셋 + CoT 루브릭을 사용하는 자동 평가 스크립트 구현

**Details:**

run_agentic_eval.py: 문항별로 Agent 파이프라인(T18) 호출 → 응답 수집 → CoT 루브릭 기반 자동 채점. 도구 호출 정확도(expected vs actual tool_calls), 파라미터 일치율, 응답 구조 검증을 자동화. 산출물: /workspace/trainer/scripts/run_agentic_eval.py
