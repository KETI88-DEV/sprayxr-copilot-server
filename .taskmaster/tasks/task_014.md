# Task ID: 14

**Title:** 향후 확장 인터페이스 정의 및 VLM Stub 설계

**Status:** pending

**Dependencies:** 4, 10

**Priority:** low

**Description:** 향후 멀티모달 단계를 위해 VLM stub(heatmap/screenshot→JSON), 규칙 엔진 stub(자세/시퀀스/두께 점수), 세션 export 자동화 인터페이스 정의 및 프로토타입 스켈레톤 작성

**Details:**

1. VLM Stub 설계:
   - 입력: heatmap.png, screenshot.png, report.json(RealCOAT export)
   - 처리: 향후 Qwen2-VL-2B-Instruct 또는 동등 모델
   - 출력: {regions_of_interest[], detected_issues[], paint_distribution, estimated_thickness[]}
   - 프로토타입: 입력 수용 및 출력 스키마만 정의, 실제 VLM은 Week 2+
   - 파일: /workspace/trainer/scripts/vlm_stub.py
2. 규칙 엔진 Stub:
   - 입력: motion data, thickness points, sequence log
   - 처리: 외부 규칙/시뮬레이터(RealCOAT)가 이미 계산
   - 출력: {speed_score, distance_score, angle_score, sequence_compliance, efficiency}
   - 이번 주: JSON 스키마만 정의, 통합은 Day 5-6에 시도
   - 파일: /workspace/trainer/scripts/rule_engine_interface.md
3. 세션 Export 자동화:
   - RealCOAT API/파일 export 인터페이스 정의
   - session_summary.json 스키마 확정
   - 자동 감지 및 sidecar 서버로 전달 로직
   - 프로토타입: 수동 파일 업로드 지원, 자동화는 향후
   - 파일: /workspace/trainer/scripts/export_automation.md
4. 산출물:
   - /workspace/trainer/scripts/vlm_stub.py: 입출력 스키마, mock 응답
   - /workspace/trainer/scripts/rule_engine_interface.md: 규칙 엔진과 LLM 통합점
   - /workspace/trainer/scripts/export_automation.md: RealCOAT 연동 프로토콜
5. 향후 확장 로드맵: VLM 추가(Week 2) → 규칙 엔진 통합(Week 3) → 음성/UI(Week 4+)
6. Qdrant 벡터 DB 확장: /qdrant_storage/에 벡터 인덱스 저장(FAISS→Qdrant 마이그레이션 고려)

**Test Strategy:**

스키마 JSON 유효성 확인, mock 응답 생성 가능, 프로토타입이 차기 작업 프레임워크 제시 가능 여부 확인

## Subtasks

### 14.1. VLM Stub 입출력 스키마 정의 및 프로토타입 작성

**Status:** pending  
**Dependencies:** None  

VLM stub의 입출력 인터페이스를 완전히 정의하고 mock 응답을 생성하는 프로토타입 스켈레톤 작성

**Details:**

/workspace/trainer/scripts/vlm_stub.py 파일을 생성하여 입력(heatmap.png, screenshot.png, report.json)을 수용하고 출력 스키마(regions_of_interest[], detected_issues[], paint_distribution, estimated_thickness[])를 정의합니다. 실제 VLM 모델(Qwen2-VL-2B-Instruct) 호출 없이 입출력 형식만 구현하며, mock 응답을 반환하는 stub 함수를 작성합니다. JSON 스키마 유효성을 보장하도록 Pydantic 모델 또는 JSONSchema를 사용합니다.

### 14.2. 규칙 엔진 Stub 인터페이스 정의 및 JSON 스키마 작성

**Status:** pending  
**Dependencies:** None  

규칙 엔진의 입출력 인터페이스를 마크다운 문서로 정의하고 통합 지점을 명시

**Details:**

/workspace/trainer/scripts/rule_engine_interface.md 파일을 작성하여 규칙 엔진의 입력(motion data, thickness points, sequence log), 처리 로직(RealCOAT 외부 규칙/시뮬레이터에서 이미 계산됨), 출력(speed_score, distance_score, angle_score, sequence_compliance, efficiency)을 명시합니다. 각 점수의 범위(0-100), 계산 방식, LLM과의 통합점을 문서화하고, JSON 출력 스키마 예시를 제공합니다. 실제 점수 계산은 Week 3에서 수행함을 명시합니다.

### 14.3. 세션 Export 자동화 인터페이스 프로토콜 정의

**Status:** pending  
**Dependencies:** None  

RealCOAT과의 연동 프로토콜을 정의하고 session_summary.json 스키마를 확정

**Details:**

/workspace/trainer/scripts/export_automation.md 파일을 작성하여 RealCOAT API 또는 파일 export 인터페이스를 정의합니다. session_summary.json의 완전한 스키마(필드명, 데이터 타입, 범위, 필수 여부)를 확정하고, 자동 감지 로직(파일 감시, polling)과 sidecar 서버로의 전달 방식을 프로토콜로 명시합니다. 현재는 수동 파일 업로드를 지원하는 인터페이스만 정의하고, 자동화는 향후 구현임을 명시합니다.

### 14.4. VLM Stub, 규칙 엔진, Export 인터페이스 통합 테스트 및 문서화

**Status:** pending  
**Dependencies:** 14.1, 14.2, 14.3  

세 개의 stub/인터페이스가 일관성 있게 작동하는지 확인하고 차기 작업의 입력으로 활용 가능한지 검증

**Details:**

생성된 세 개의 파일(/workspace/trainer/scripts/vlm_stub.py, rule_engine_interface.md, export_automation.md)을 통합하여 전체 워크플로우가 일관성 있게 연결되는지 확인합니다. VLM stub의 출력이 규칙 엔진의 입력 요구사항을 만족하는지, session_summary.json이 export 프로토콜과 일치하는지 검증합니다. 통합 테스트용 end-to-end 스크립트 또는 체크리스트를 작성하여 차기 작업(Week 2+ VLM 추가, Day 5-6 규칙 엔진 통합)이 이 인터페이스 정의를 기반으로 진행될 수 있음을 확인합니다.

### 14.5. 향후 확장 로드맵 문서화 및 최종 검증

**Status:** pending  
**Dependencies:** 14.1, 14.2, 14.3, 14.4  

VLM→규칙 엔진→음성/UI 확장 단계를 명시하고 현재 스텁 설계가 로드맵과 부합하는지 최종 검증

**Details:**

/workspace/trainer/scripts/ 또는 /output/roadmap.md에 향후 확장 로드맵을 문서화합니다: Week 2에서 VLM 추가(실제 Qwen2-VL-2B-Instruct 호출) → Week 3에서 규칙 엔진 통합 → Week 4+에서 음성/UI 확장. 현재 작업(Task 14)에서 정의한 스텁과 인터페이스가 이러한 확장 단계에 충분히 대응 가능하도록 설계되었음을 명시합니다. 스텁 설계의 확장성(extensibility) 검증, 리소스 요구사항(메모리, 계산량), 외부 의존성(RealCOAT API, Qwen2-VL-2B-Instruct 모델)을 정리합니다.
