# Task ID: 10

**Title:** 세션 코칭 응답 시스템 구현

**Status:** pending

**Dependencies:** 4 ✓, 9 ✓, 15, 16

**Priority:** high

**Description:** Co-Pilot 방향 세션 코칭: session_summary.json 입력 → Tool Calling(dft_judge, anomaly_detection) → Chain of Thought 추론 → 코칭 피드백 JSON 생성. 기존 시뮬레이터 지표(이격거리, DFT, 속도, 각도, under/over-thickness)를 MCP 서버로 전달하여 분석 결과를 받고, RAG 문서 근거와 결합하여 행동항목/위험플래그/근거문서를 포함한 구조화 코칭 응답 생성

**Details:**

1. session_summary.json 파싱 및 지표 추출
2. dft_judge MCP 호출: DFT 측정값 → 합부 판정
3. anomaly_detection MCP 호출: 패턴 이상 탐지
4. RAG 검색: 관련 문서 근거 수집
5. CoT 추론: MCP 결과 + RAG 문서 → 코칭 피드백 생성
6. 출력: action_items, risk_flags, citations, xr_highlights 포함 JSON
7. PRD §3.2 시나리오 B + §16.4 시나리오 D/E 통합
8. 산출물: /workspace/trainer/scripts/session_coach.py

**Test Strategy:**

세션 요약 JSON 파싱 성공, 코칭 응답 JSON 스키마 유효성 확인, action_items 최소 3개 이상, citations chunk_id 존재 확인, 도메인 검토자 5-10건 코칭 품질 평가(4점/5점 이상)

## Subtasks

### 10.1. 입력 스키마 정의 및 검증 로직 구현

**Status:** pending  
**Dependencies:** None  

session_summary.json 입력 스키마(session_id, scenario_id, mode, 성능 지표들)를 정의하고 Pydantic/JSON 스키마로 구현하여 입력 데이터 검증 로직을 완성

**Details:**

session_id, scenario_id, mode(practice/test), avg_speed, avg_distance_mm, avg_angle_deg, dft_mean_um, dft_min_um, dft_max_um, under_thickness_ratio, over_thickness_ratio, paint_usage_ratio, notes[] 필드를 정의하고, /workspace/trainer/scripts/session_scorer.py에서 Pydantic 모델로 구현. 입력 데이터 유효성 확인(required 필드, 범위 체크, 타입 검증) 기능 추가. 잘못된 입력에 대한 명확한 에러 메시지 제공.

### 10.2. 위험 플래그 규칙 엔진 구현

**Status:** pending  
**Dependencies:** 10.1  

under_thickness, over_thickness, distance_variance, speed_variance, angle_consistency_low, sequence_violation 6가지 위험 플래그를 계산하는 규칙 엔진을 구현

**Details:**

입력 스키마의 수치 지표들을 기반으로 각 위험 플래그의 판정 기준(threshold)을 정의. under_thickness_ratio > 15%, over_thickness_ratio > 10% 등으로 설정. 각 플래그별로 심각도(severity: high/medium/low) 및 판정 근거(reason)을 함께 반환하도록 구현. /workspace/trainer/scripts/session_scorer.py에 함수로 추가. 도메인 전문가 협의하여 threshold 값 최종 결정.

### 10.3. RAG API 통합 및 프롬프트 엔지니어링

**Status:** pending  
**Dependencies:** 10.1, 10.2  

/workspace/rag_api/와 통합하여 선박 도장 규정 문서 검색 기능을 구현하고, 코칭 피드백 생성을 위한 LLM 프롬프트를 설계

**Details:**

RAG API의 검색 엔드포인트를 호출하는 함수 구현(/rag_api/search 또는 동등 엔드포인트 사용). 입력: 위험 플래그, 성능 지표 요약. 출력: 관련 문서 chunks(chunk_id, source, text, score 포함). 프롬프트 템플릿 설계: (1) 지표 요약 섹션 (2) RAG 검색 결과(citations 포함) (3) 코칭 생성 요청(3개 action_items, 근거 문서 인용 필수). 두 가지 RAG 구성 모두 지원: RAG-only, RAG+QLoRA.

### 10.4. 코칭 응답 JSON 스키마 구현 및 LLM 통합

**Status:** pending  
**Dependencies:** 10.3  

응답 스키마(session_id, summary, action_items[3개], risk_flags[], citations[], confidence, model_name, timestamp)를 정의하고 LLM 호출 후 응답을 파싱하여 스키마에 맞게 변환

**Details:**

응답 스키마를 Pydantic 모델로 구현: session_id(str), summary(str, <500 chars), action_items(list[{id, description, steps[], execution_order, re_measurement_time}]), risk_flags(list[{flag_name, severity, evidence}]), citations(list[{chunk_id, source, relevance_score}]), confidence(float 0-1), model_name(str), timestamp(ISO8601). LLM 호출 함수 구현: prompt 전달 → 모델 응답 → JSON 파싱 → Pydantic 검증 → 응답 반환. action_items은 정확한 수치, 실행 순서, 재측정 시점 포함 필수. LLM 라이브러리(transformers, vllm 등) 연동.

### 10.5. 도메인 평가 및 최종 검증

**Status:** pending  
**Dependencies:** 10.4  

구현된 코칭 시스템을 5-10건의 실제 세션 또는 합성 세션으로 테스트하고, 도메인 전문가(선박 도장 기술자) 평가를 거쳐 자연성, 안전성, 실행 가능성 검증

**Details:**

평가 데이터셋 준비: 5-10개 session_summary.json 샘플 (다양한 성과, 위험도 포함). 각 세션마다 코칭 응답 생성. 평가 기준: (1) 자연성(Natural): 코칭 텍스트가 현실적이고 이해하기 쉬운가 (2) 안전성(Safety): 권고사항이 안전규정을 준수하는가 (3) 실행 가능성(Feasibility): 제시된 action_items이 현장에서 실제로 수행 가능한가. 도메인 전문가가 각 항목을 4/5점 이상으로 평가하도록 설계. 평가 결과 정리 및 개선 사항 반영.
