# Task ID: 16

**Title:** MCP Server Stub 구현

**Status:** pending

**Dependencies:** 15

**Priority:** high

**Description:** 6개 MCP Server Stub 구현 (FastAPI 기반). 각 서버의 역할: (1) pose_estimation — 숙련자 BVH reference vs 현재 작업자 6DOF 자세 비교, 관절별 편차 벡터 및 교정 방향 출력. (2) anomaly_detection — 히트맵 이미지 + DFT 정량 데이터 멀티모달 입력 (LLaSA 계열 예정), Stub은 fixture 반환. (3) timeseries — ReAct Agent 루프 내 session state 순환 갱신용, 매 턴 누적 업데이트. (4) location_tracking — 선박 3D 공간 내 현재 구역 식별 (PaintNet trajectory 기반), pose/sequence에 공간 context 공급, Phase 2는 fixture stub. (5) sequence_validation — 구역 내 stroke 단위 도장 순서 검증 (위→아래 등, 공정 단계 순서 아님), 숙련자 stroke 패턴이 reference. (6) dft_judge — 규제/리포트용 명시적 pass/fail, 90/10 룰 실제 로직 구현 (유일하게 mock 아님).

**Details:**

1. /workspace/mcp_servers/ 디렉터리 구조 생성
2. 각 서버별 FastAPI 앱 구현 (Pydantic 입력 검증, 에러 핸들링)
3. pose_estimation: PoseInput(current_pose: List[float] 6DOF BVH, reference_pose: List[float]) → PoseOutput(joint_deviations, correction_direction)
4. anomaly_detection: AnomalyInput(heatmap_path: str, dft_values: List[float], env: dict) → AnomalyOutput(anomaly_type, location, severity, image_evidence, likely_cause) — 멀티모달 fixture
5. timeseries: TimeseriesInput(session_id, speed_history: List[float], distance_history: List[float]) → TimeseriesOutput(trend, deviation, fatigue_signal) — 세션 state 누적
6. location_tracking: LocationInput(trajectory: List[dict] x/y/z) → LocationOutput(zone_id, zone_type, coverage_pct, unpainted_zones) — fixture stub, Phase 4에서 실제 3D 연동
7. sequence_validation: SequenceInput(trajectory, zone_id, reference_strokes: List) → SequenceOutput(violation: bool, current_direction, recommended_direction, reason)
8. dft_judge: DFTInput(measurements: List[float], nominal: float, tolerance: str) → DFTOutput(pass: bool, violations: List[dict]) — 90/10 룰 실제 구현
9. 구현 우선순위: dft_judge(실제 로직) → anomaly/sequence/pose stub → timeseries stub → location stub
10. 통합 테스트 스크립트 및 각 서버 /health 엔드포인트
11. 산출물: /workspace/mcp_servers/{pose,anomaly,timeseries,location,sequence,dft}/, 테스트 스크립트

**Test Strategy:**

각 서버 /health 엔드포인트 응답 확인, 정상 입력에 대한 fixture 응답 포맷 검증, 잘못된 입력에 대한 Pydantic 검증 에러 핸들링, dft_judge 90/10 룰 정확성 테스트(시나리오 E 기준), anomaly_detection 멀티모달 입력 스키마 검증(이미지 경로 + DFT 배열), sequence_validation stroke 순서 위반 탐지, pose_estimation BVH 입력 파싱 및 편차 벡터 출력, timeseries 세션 누적 업데이트 상태 확인, location_tracking 구역 ID 반환

## Subtasks

### 16.1. MCP 서버 공통 프레임워크 및 디렉터리 구조

**Status:** pending  
**Dependencies:** None  

/workspace/mcp_servers/ 디렉터리 생성 + FastAPI 공통 베이스(에러 핸들링, 로깅, Pydantic 모델 베이스) 구현

**Details:**

디렉터리 구조: mcp_servers/__init__.py, mcp_servers/common.py (공통 모델/에러), mcp_servers/pose_estimation.py, anomaly_detection.py, timeseries.py, location_tracking.py, sequence_validation.py, dft_judge.py, main.py (통합 라우터). 공통 응답 포맷: {status, result, error, timestamp}. Health check 엔드포인트 포함.

### 16.2. dft_judge 서버 구현 (90/10 룰 실제 로직)

**Status:** pending  
**Dependencies:** 16.1  

DFT 합부 판정 서버 구현. PRD §3.2.6 dft_judge 스펙 기반. 90/10 룰 실제 판정 로직 (규칙 기반, 즉시 구현 최우선)

**Details:**

입력: measurements(DFT 측정값 배열), spec(nominal, tolerance_rule='90/10'). 90/10 룰: 모든 측정값이 nominal의 90% 이상 + 전체 평균이 nominal 이상. 출력: {pass: bool, violations: [{index, value, threshold}], mean: float, coverage_pct: float}. PRD §16.4 시나리오 E 예시([150,85,92,110,78], nominal=100) 검증. anomaly_detection과 분리하여 규제/리포트용 명시적 합부 판정.

### 16.3. anomaly_detection stub 구현 (멀티모달 준비)

**Status:** pending  
**Dependencies:** 16.1  

히트맵 이미지 + DFT 정량 데이터 멀티모달 이상 탐지 stub. LLaSA 계열 모델 연동 준비, Phase 2는 fixture 응답

**Details:**

입력: heatmap_image_path(str), measurements(DFT 배열), env{temp, humidity}. Phase 2 stub 구현: 입력 스키마 검증 후 fixture 응답 반환. 출력: {anomaly: bool, anomaly_type: str, location: str, severity: str, likely_cause: str, image_evidence: str, anomaly_points: []}. likely_cause 매핑 예시: under_thickness→excessive_speed|low_viscosity. PRD §3.2.6: 단순 수치 임계값이 아닌 이미지+정량 데이터 복합 분석 방향. dataset.md `결과(heatmap)`, `장비상태` 참조.

### 16.4. sequence_validation stub 구현 (stroke 순서 검증)

**Status:** pending  
**Dependencies:** 16.1  

구역 내 stroke 단위 도장 순서 검증 stub. 공정 단계 순서가 아닌 위→아래, 아래→위 등 붓질 패턴 검증

**Details:**

입력: current_trajectory(stroke 궤적 배열), location(구역 ID), reference_sequence(숙련자 stroke 패턴). Phase 2 stub: fixture 응답. 출력: {stroke_violations: [{stroke_id, expected_direction, actual_direction}], next_stroke_recommendation: str, progress_pct: float}. PRD §3.2.6: 공정 단계(블라스팅→프라이머)가 아닌 구역 내 붓질 순서. dataset.md `작업시퀀스(stroke 단위)`, `언어명령` 참조. 숙련자 stroke 패턴이 기준.

### 16.5. pose_estimation stub 구현 (BVH 자세 비교)

**Status:** pending  
**Dependencies:** 16.1  

숙련자 BVH 데이터 대비 현재 작업자 자세 비교 stub. 관절별 편차 벡터와 교정 방향 출력

**Details:**

입력: reference_pose(숙련자 BVH 데이터), current_pose(현재 작업자 BVH, 6DOF). Phase 2 stub: BVH 입력 파싱 검증 후 fixture 응답. 출력: {joint_deviations: [{joint_name, deviation_vector: [x,y,z], severity}], correction_direction: str, similarity_score: float}. PRD §3.2.6: 단순 관절 좌표가 아닌 BVH 형식. dataset.md `작업행동(worker_pose)` 숙련자 데이터가 기준값.

### 16.6. timeseries stub 구현 (세션 state 누적)

**Status:** pending  
**Dependencies:** 16.1  

ReAct Agent 루프 내 세션 state용 stub. 매 턴 누적 업데이트되는 속도/거리/각도 트렌드 추적

**Details:**

입력: session_id, speed_series(속도 시계열), distance_series(거리 시계열), angle_series(각도 시계열), update_mode(append|reset). Phase 2 stub: 세션별 누적 상태 mock 저장 + fixture 응답. 출력: {trend: str, stroke_deviation: float, fatigue_pattern: str, session_summary: {avg_speed, total_distance}}. PRD §3.2.6: 단발 호출이 아닌 매 턴 업데이트. dataset.md `작업시간`, `장비궤적` 시계열 참조.

### 16.7. location_tracking stub 구현 (3D 공간 구역 식별)

**Status:** pending  
**Dependencies:** 16.1  

선박 3D 공간 내 현재 도장 구역 식별 stub. PaintNet trajectory 기반, Phase 4 연동 준비

**Details:**

입력: coordinates(x, y, z 궤적 배열), ship_model_metadata(선박 3D 모델 구역 정보). Phase 2 stub: fixture 응답. 출력: {current_zone_id: str, zone_characteristics: str, uncovered_zones: [], coverage_pct: float}. PRD §3.2.6: pose_estimation, sequence_validation에 공간 context 공급. dataset.md `장비궤적`, PaintNet 기반 trajectory simulation 참조. Phase 4~5 실제 연동.

### 16.8. MCP 서버 통합 테스트 및 라우터

**Status:** pending  
**Dependencies:** 16.2, 16.3, 16.4, 16.5, 16.6, 16.7  

6개 서버 통합 라우터(main.py) + end-to-end 테스트 스크립트. 우선순위별 검증

**Details:**

main.py에서 FastAPI include_router로 6개 서버 통합. 단일 프로세스에서 모든 엔드포인트 서빙. 통합 테스트: 각 서버 health check + 우선순위별 검증 (dft_judge 실제 로직 > anomaly stub > sequence stub > pose stub > timeseries stub > location stub). 시나리오 D/E/F 기반 순차 호출 테스트. 산출물: /workspace/mcp_servers/main.py, /workspace/mcp_servers/test_integration.py
