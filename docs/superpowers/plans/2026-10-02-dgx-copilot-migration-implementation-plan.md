# DGX Spark Co-Pilot 통합 이관 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 기존 Co-Pilot의 코드·DB·전문가 기준을 DGX Spark에 복구 가능하게 이관하고, 기존 텍스트 Query를 유지하면서 음성 Query와 시퀀스 속도·이격거리 guide를 운영 API에 추가한다.

**Architecture:** 기존 x86 프로젝트는 체크섬이 있는 원본 스냅샷으로 보존하고, 런타임 기능은 DGX의 ARM64 컨테이너 코드에 통합한다. 전문가 판정은 결정론적 모듈, 음성 입력은 STT 어댑터, 텍스트 답변은 기존 27B Query 서비스를 공유한다. 새 이미지는 별도 태그와 내부 포트로 검증한 후에만 운영 전환 후보가 된다.

**Tech Stack:** Python 3.12, FastAPI, Pydantic v2, SQLite, pytest, Docker Compose, vLLM/OpenAI-compatible API, multilingual Whisper-compatible STT

**Spec:** `docs/superpowers/specs/2026-10-02-dgx-copilot-migration-design.md`

## Global Constraints

- 기존 `POST /api/v1/copilot/query` JSON 계약을 변경하지 않는다.
- TSB 시퀀스 입력에 누적 히트맵을 다시 요구하지 않는다.
- 속도·거리·Zone 판정은 LLM을 호출하지 않는다.
- 음성은 WAV 16 kHz mono, 최대 15초, 언어 `ko|vi|id|th`만 허용한다.
- 실전모드 A-1~A-3은 Zone ID가 보완될 때까지 기준 부재로 처리한다.
- x86 가상환경은 보존만 하고 ARM64 런타임에서 사용하지 않는다.
- 기존 외부 서비스, 포트포워딩과 원본 서버는 전체 검증 전까지 변경하거나 폐기하지 않는다.
- `.env`의 내용은 읽거나 출력하지 않고 현재 배포 값은 이름만 참조한다.
- 코드 변경은 테스트 우선, 명시적 파일 staging, 한국어 Conventional Commit으로 커밋한다.

## Review Focus

- 한 요청에 거리 단위가 mm와 m로 혼재하면 조용히 오판하지 않고 `invalid_units` 또는 기준 부재를 반환하는지 Task 3에서 고정한다.
- 경계값 부근에서 slow/ok 또는 near/ok가 매 요청마다 진동하지 않도록 히스테리시스를 Task 3에서 검증한다.
- 15초를 넘거나 헤더와 실제 PCM 형식이 다른 WAV가 STT로 전달되지 않는지 Task 5에서 검증한다.
- STT 성공 뒤 27B가 실패해도 transcript를 로그에 원문으로 남기거나 내부 오류를 노출하지 않는지 Task 5에서 검증한다.
- 기존 DB와 DGX DB에 같은 외부 ID가 있을 때 기존 행을 덮어쓰지 않고 출처가 보존되는지 Task 2에서 검증한다.

---

### Task 1: 복구 가능한 원본 스냅샷과 manifest

**Files:**
- Create: `deploy/spark/scripts/legacy_snapshot_manifest.py`
- Create: `deploy/spark/scripts/test_legacy_snapshot_manifest.py`
- Create on DGX: `/home/keti/shipcoat/migration/legacy-paint-ai-server/`
- Create on DGX: `/home/keti/shipcoat/migration/manifests/`

**Interfaces:**
- Consumes: 기존 서버 `/home/ketivrar/paint_ai_server`
- Produces: `build_manifest(root: Path) -> dict[str, ManifestEntry]`, 원본·복사본 SHA-256 manifest와 검증 결과

- [ ] **Step 1: manifest 정렬, 파일 크기, SHA-256 비교 테스트를 작성한다.**
- [ ] **Step 2: 컨테이너에서 해당 테스트를 실행해 구현 부재로 실패하는지 확인한다.**
- [ ] **Step 3: `build_manifest`, `write_manifest`, `verify_manifest`를 구현한다.**
- [ ] **Step 4: 테스트를 통과시킨다.**
- [ ] **Step 5: 기존 서비스 설정·DB 통계·프로젝트 manifest를 읽기 전용으로 채집한다.**
- [ ] **Step 6: `rsync --archive --partial`로 DGX staging에 복사하고 복사본 manifest를 검증한다.**
- [ ] **Step 7: manifest와 스크립트만 명시적으로 커밋한다.**

### Task 2: 전문가 자료와 DB의 영속 데이터 이관

**Files:**
- Create: `deploy/spark/scripts/import_legacy_copilot.py`
- Create: `deploy/spark/scripts/test_import_legacy_copilot.py`
- Modify: `deploy/spark/compose.yaml`
- Create on DGX: `/home/keti/shipcoat/data/expert_reference_data/`
- Create on DGX: `/home/keti/shipcoat/backups/`

**Interfaces:**
- Consumes: Task 1에서 검증된 snapshot, 기존 `shipcoat.db`, 전문가 CSV·PNG·프로필
- Produces: `plan_import(source_db: Path, target_db: Path) -> ImportPlan`, `apply_import(plan: ImportPlan) -> ImportReport`, 읽기 전용 전문가 데이터 mount

- [ ] **Step 1: 사용자·세션·시퀀스 충돌, 재실행 멱등성, 출처 보존 테스트를 작성한다.**
- [ ] **Step 2: 테스트 실패를 확인한다.**
- [ ] **Step 3: 대상 DB 자동 백업, 스키마별 변환, 충돌 시 원본 보존 import를 구현한다.**
- [ ] **Step 4: 테스트를 통과시킨다.**
- [ ] **Step 5: 전문가 raw와 생성 프로필을 source hash와 함께 영속 경로로 복사하고 modes 1~10을 검증한다.**
- [ ] **Step 6: Compose에 전문가 데이터 read-only mount와 명시적 경로 변수를 추가한다.**
- [ ] **Step 7: 대상 DB 백업 후 dry-run과 실제 import를 수행하고 행 수·참조 무결성을 비교한다.**
- [ ] **Step 8: 코드와 Compose 변경을 커밋한다.**

### Task 3: 결정론적 속도·이격거리 평가기

**Files:**
- Create: `trainer/demo_api/expert_guidance.py`
- Create: `trainer/demo_api/tests/test_expert_guidance.py`

**Interfaces:**
- Consumes: `mode_id: int`, `zone_id: str`, `sequence_info`, `sequence_result`, 전문가 프로필 경로
- Produces: `ExpertGuidanceEngine.evaluate(request: AnalyzeSequenceRequest) -> SequenceGuide`; `GuideItem(status, message, available, reason)`

- [ ] **Step 1: slow/ok/fast, near/ok/far, 기준 부재, invalid_units, 경계 히스테리시스와 최근 2~3개 시퀀스 안정화 테스트를 작성한다.**
- [ ] **Step 2: 테스트 실패를 확인한다.**
- [ ] **Step 3: 프로필 loader와 단위 정규화기를 구현한다.**
- [ ] **Step 4: 세션·mode·Zone별 상태를 가진 평가기와 히스테리시스를 구현한다.**
- [ ] **Step 5: 단위 및 평가기 테스트를 통과시킨다.**
- [ ] **Step 6: 프로필 파일 변경 시 안전한 재적재와 손상 파일 fallback 테스트를 추가하고 통과시킨다.**
- [ ] **Step 7: 평가기와 테스트를 커밋한다.**

### Task 4: analyze-sequence guide 계약과 연동

**Files:**
- Modify: `trainer/demo_api/telemetry_contracts.py`
- Modify: `trainer/demo_api/telemetry_router.py`
- Modify: `trainer/demo_api/app.py`
- Modify: `trainer/demo_api/tests/test_telemetry_router.py`
- Modify: `trainer/demo_api/tests/test_app_live_wiring.py`

**Interfaces:**
- Consumes: Task 3 `ExpertGuidanceEngine.evaluate`
- Produces: `TelemetrySequenceData.guide: SequenceGuide | None`; analyze 경로에서는 항상 객체, telemetry 저장 경로에서는 기존 계약 유지

- [ ] **Step 1: analyze 응답의 guide 전체 상태, 정상 `ok` 복귀, 기준 부재와 telemetry 기존 응답 호환 테스트를 작성한다.**
- [ ] **Step 2: 기존 12키 assertion이 의도한 경로별 계약 차이를 드러내며 실패하는지 확인한다.**
- [ ] **Step 3: guide Pydantic 계약과 analyze 전용 평가 호출을 구현한다.**
- [ ] **Step 4: duplicate 시퀀스도 현재 guide를 반환하되 누적 상태를 중복 갱신하지 않게 구현한다.**
- [ ] **Step 5: telemetry 및 wiring 테스트를 통과시킨다.**
- [ ] **Step 6: 변경을 커밋한다.**

### Task 5: STT 어댑터와 별도 음성 Query API

**Files:**
- Create: `trainer/demo_api/speech_to_text.py`
- Create: `trainer/demo_api/voice_contracts.py`
- Create: `trainer/demo_api/voice_router.py`
- Create: `trainer/demo_api/tests/test_speech_to_text.py`
- Create: `trainer/demo_api/tests/test_voice_router.py`
- Modify: `trainer/demo_api/copilot_router.py`
- Modify: `trainer/demo_api/app.py`

**Interfaces:**
- Consumes: multipart WAV와 기존 Query pipeline provider
- Produces: `SpeechToText.transcribe(audio: bytes, language: VoiceLanguage) -> Transcript`; `POST /api/v1/copilot/query-voice`; 공통 `execute_copilot_query(query, pipeline_provider, ...) -> CopilotResponse`

- [ ] **Step 1: 정상 WAV, stereo, 비-16kHz, 손상 헤더, 15초 초과, 지원하지 않는 언어 테스트를 작성한다.**
- [ ] **Step 2: STT 성공, 빈 transcript, STT 503, 모델 503, transcript 포함 성공 응답과 기존 텍스트 계약 불변 테스트를 작성한다.**
- [ ] **Step 3: 테스트 실패를 확인한다.**
- [ ] **Step 4: WAV validator와 `SpeechToText` protocol 및 lazy provider를 구현한다.**
- [ ] **Step 5: 기존 Query 실행부를 공통 서비스로 추출하되 기존 JSON 응답 필드를 변경하지 않는다.**
- [ ] **Step 6: 음성 router와 전용 응답 계약을 구현한다.**
- [ ] **Step 7: 민감한 transcript와 내부 예외를 로그에 남기지 않는지 테스트하고 전체 Query 테스트를 통과시킨다.**
- [ ] **Step 8: 변경을 커밋한다.**

### Task 6: DGX용 STT 런타임 선택과 컨테이너 통합

**Files:**
- Modify: `deploy/spark/requirements/agent-api.lock.txt`
- Modify: `deploy/spark/images/agent.Dockerfile`
- Modify: `deploy/spark/compose.yaml`
- Modify: `deploy/spark/.env.example`
- Create: `deploy/spark/scripts/bench_voice_query.py`
- Create: `deploy/spark/scripts/test_bench_voice_query.py`

**Interfaces:**
- Consumes: Task 5 `SpeechToText` provider, 로컬 STT 모델 경로
- Produces: ARM64 빌드 가능한 agent image, 모델 lazy preload/readiness, 언어별 p50/p95 및 메모리 보고서

- [ ] **Step 1: 현재 CUDA·PyTorch 환경에서 후보 STT를 별도 throwaway 컨테이너로 실행해 ko/vi/id/th와 메모리를 측정한다.**
- [ ] **Step 2: 가장 작은 적합 후보 하나를 선택하고 버전과 hash를 lock한다.**
- [ ] **Step 3: dependency/import/readiness 실패 테스트를 작성하고 실패를 확인한다.**
- [ ] **Step 4: Dockerfile, lock, Compose 모델 mount와 health/readiness를 구현한다.**
- [ ] **Step 5: 언어별 샘플과 27B 동시 요청 benchmark 스크립트를 구현하고 테스트한다.**
- [ ] **Step 6: 새 이미지 빌드 후 STT-only와 27B 동시 부하에서 OOM 없이 동작하는지 검증한다.**
- [ ] **Step 7: 런타임 변경을 커밋한다.**

### Task 7: Swagger 예시와 TSB 모의 연동

**Files:**
- Modify: `trainer/demo_api/copilot_contracts.py`
- Modify: `trainer/demo_api/telemetry_contracts.py`
- Modify: `trainer/demo_api/voice_contracts.py`
- Modify: `trainer/demo_api/app.py`
- Modify: `deploy/spark/scripts/smoke_v2_5_api.py`
- Modify: `trainer/demo_api/tests/test_contracts.py`

**Interfaces:**
- Consumes: Tasks 4~6의 공개 API
- Produces: 순서와 설명이 명확한 OpenAPI 예시, 정상·오류 예시, 음성/텍스트/시퀀스 smoke 결과

- [ ] **Step 1: OpenAPI에 기존 Query, 음성 Query, analyze guide 예시와 정확한 multipart 필드가 노출되는 테스트를 작성한다.**
- [ ] **Step 2: 테스트 실패를 확인한다.**
- [ ] **Step 3: route summary, description, tag 순서, request/response examples를 추가한다.**
- [ ] **Step 4: DB viewer 링크와 누적 히트맵 비요구 사항을 상단 설명에 명시한다.**
- [ ] **Step 5: smoke 스크립트에 정상·잘못된 음성 및 guide 검증을 추가하고 통과시킨다.**
- [ ] **Step 6: 변경을 커밋한다.**

### Task 8: staged 배포와 완료 감사

**Files:**
- Modify if required: `deploy/spark/compose.yaml`
- Create runtime evidence: `/home/keti/shipcoat/migration/reports/`

**Interfaces:**
- Consumes: 모든 이전 Task의 커밋, 새 이미지, 이전/현재 DB 백업
- Produces: 내부 접근 가능한 후보 서비스, 회귀·부하·재부팅·복구 보고서, 외부 전환 체크리스트

- [ ] **Step 1: 컨테이너 전체 pytest와 MCP 테스트를 실행해 통과 수와 로그를 저장한다.**
- [ ] **Step 2: 새 태그로 이미지를 빌드하고 기존 운영 stack과 충돌하지 않는 내부 포트에서 후보 stack을 기동한다.**
- [ ] **Step 3: 텍스트 Query, 음성 Query, analyze guide, pose_recomm, knowledge_search, report, DB viewer와 OpenAPI smoke를 실행한다.**
- [ ] **Step 4: VR fast 요청과 상세 생성 동시 부하에서 p50/p95, 메모리 최고점, OOM 여부를 기록한다.**
- [ ] **Step 5: 후보 stack 재시작과 DGX 재부팅 후 영속 DB·전문가 profile·health 복구를 검증한다.**
- [ ] **Step 6: 원본/복사본 manifest, DB 행 수, modes 1~10, API 요구사항별 증거를 완료 감사표로 대조한다.**
- [ ] **Step 7: rollback 명령과 공유기 포트포워딩 대상을 기록하되 외부 설정은 변경하지 않는다.**
- [ ] **Step 8: 최종 branch 상태와 diff를 검토하고 전체 변경을 한 번 더 회귀시험한다.**
