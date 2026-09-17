# v2.5 계약 전면 전환 계획

- 작성: 2026-09-17
- 기준 계약: `docs/ai_copilot_v2_5_deployment_contract.md`(v2.5-Spark-1, f6a15cc). 이 계획과 계약이 다르면 계약을 따른다.
- 선행 계획: `docs/superpowers/plans/2026-09-15-spark-rag-first-release.md`(병합·FP8·copilot/query 1단계). Spark 배포와 실행 확인은 끝났다(사용자 확인, 2026-09-17).
- 근거: 계약 전체와 현재 코드를 대조한 조사(2026-09-17). 모든 LOC·시간은 추정이다.

## 1. 결정

| # | 결정 | 출처 |
|---|---|---|
| E1 | **기존 MCP 7종은 전면 교체한다.** 운영에는 계약 §18의 5종(`coating_analysis`, `pose_analysis`, `sequence_analysis`, `anomaly_detection`, `knowledge_search`)만 남긴다. `location_tracking`, `timeseries`, `dft_judge`, `vlm_analysis`, 옛 `pose_estimation`, `sequence_validation`, 옛 `anomaly_detection`의 라우터, 스키마 JSON, 프로토콜, 테스트 fixture를 제거하거나 교체한다. | 사용자 결정 |
| E2 | **LLM 도구 호출을 제거한다(기본값).** 모든 경로에서 결정론 엔진이 먼저 분석하고, LLM은 구조화 결과와 RAG 근거를 설명만 한다(계약 §3.2, §18.9). `/api/chat` 내부 경로도 같은 방식으로 바꾼다. 그러므로 v7 SFT는 필요 없다. | E1에 따른 기본값. 되돌릴 수 있다 |
| E3 | **품질 평가를 새로 정의한다(기본값).** agentic 30문항의 tool_selection/parameter 채점은 폐기한다. 새 평가는 task_type 4종 × 세션 결과 fixture로 만든 문항에 대해 (a) 구조화 수치·상태를 바꾸거나 지어내지 않는지, (b) citations가 실제 프롬프트 chunk와 일치하는지, (c) 답변 구조와 근거성을 결정론 규칙으로 채점한다. NF4 기준선 6.704와 옛 FP8 게이트 수치는 무효가 되고, 새 평가로 NF4(A40)와 FP8(Spark)을 다시 측정한다. 단계적 게이트 규칙(8회 → 95% Welch CI → 18회)은 그대로 쓴다. | E1·E2에 따른 기본값 |
| E4 | 분석 엔진은 agent-api 프로세스 안에서 직접 호출한다. MCP 서버의 5개 라우터는 같은 엔진 모듈을 감싼 얇은 HTTP 래퍼다. | 사용자 결정. 계약 §18.1 746행이 허용 |
| E5 | 계약에 없는 판정 임계값은 잠정값으로 진행한다. 대상: 도막 6단계 경계, 도막 spec, 방향·궤적 이탈 임계, 속도·거리 목표와 허용오차, 구역별 기대 방향. 값은 `trainer/configs/scenarios/TR-L01.yaml`에 두고 출처와 `provisional: true`를 기록한다. `algorithm_version`에도 반영한다. | 사용자 결정 |
| E6 | 자세 레퍼런스(`expert_pose_reference.json`)가 없으므로 이번 릴리스에서 `pose_analysis`는 `not_available`(`reason_code: reference_data_missing`)을 정상 결과로 인정한다. §16.3 인수 기록에 예외로 적는다. 엔진과 계약은 완전히 구현한다. | 사용자 결정 |
| E7 | 영속 저장소는 SQLite(표준 라이브러리, WAL, 단일 워커)로 한다. 세션, 시퀀스, 이상 이벤트, 질의, 리포트를 저장한다. 재기동하면 `pending` 리포트를 다시 큐에 넣는다. | 계약 §14·§16.4 700행 |
| E8 | 제품 UI는 `demo-ui` 안에 새 진입점(`src/product/`)으로 만들고 web 이미지 기본 화면으로 둔다. 외부에서 `/api/chat`, `/api/sim/*`, `/api/monitor/*`를 호출하지 못하게 막는다(nginx `location`, 앱 profile 환경변수). | 기본값. 계약 §2.2 |
| E9 | 옛 계약 문서 `docs/ai_copilot_response_contract.md`는 v2.5가 암묵적으로 참조하는 부분(§3.7 불변 규칙 등)에 한해 보조 규범으로 쓴다. 충돌하면 v2.5를 따른다. | 기본값 |

## 2. 규모와 리소스 (추정)

| 항목 | 추정 |
|---|---|
| 코드 | 운영·스크립트 약 7.5k줄 신규·수정, 약 2~3k줄 삭제, 테스트 약 7k줄 |
| 에이전트 작업 시간 | 순차 약 75시간, 웨이브 병렬 약 24~30시간(리뷰·통합 포함) |
| GPU 측정 | 새 평가 NF4 기준선 8 seed(A40, 라이브 API 중단 창, 약 6~7시간), 필요하면 18 seed |
| Spark 실기 | 1~2일: 이미지 빌드, §16.3 실측, 성능, 소크, FP8 게이트 |
| 컨텍스트 | 전체 약 5~7M 토큰. 태스크당 L 60~100k, M 120~200k, H 220~350k |

컨텍스트 운영 규칙:
- 계약 전체는 약 25k 토큰이다. 담당자에게는 §5의 줄 범위만 넘긴다(태스크당 5~12k).
- 오케스트레이터는 이 문서와 태스크 결과 요약만 유지한다. 코드 본문은 읽지 않는다.
- 웨이브마다 결과를 이 문서 §6 진행 기록에 적는다. 세션이 바뀌어도 이 문서만으로 이어서 진행할 수 있어야 한다.
- 같은 파일을 두 태스크가 동시에 수정하지 않는다(§4 소유 규칙). 병렬 태스크는 서로 다른 파일만 만진다.

## 3. 태스크

| ID | 제목 | 산출물 | 의존 | 웨이브 | LOC(운영/테스트) | 난이도 | 소요 |
|---|---|---|---|---|---|---|---|
| T0 | 계약·PDF 예제를 fixture로 추출하고, §7의 계약 빈칸에 잠정 규칙을 확정 | `trainer/demo_api/tests/fixtures/v25/*.json`, 이 문서 §7 | — | W0 | JSON 약 40개 | L | 1h |
| T1 | 공개 v1 Pydantic 계약(요청·응답·래퍼·오류) | `trainer/demo_api/v1/contracts.py`(`copilot_contracts.py` 흡수) | T0 | W1 | 450/350 | M | 2.5h |
| T2 | 분석 5종 I/O 계약과 공통 envelope(§18.2·§18.3) | `trainer/analysis/contracts.py`, `trainer/analysis/__init__.py` | T0 | W1 | 400/300 | M | 2h |
| T3 | SQLite 저장소, 상태 전이, 멱등·409 | `trainer/demo_api/v1/store.py` | T1 | W1b | 450/400 | M/H | 3h |
| T4 | 시나리오 설정 로더와 잠정 임계값(E5) | `trainer/configs/scenarios/TR-L01.yaml`, `trainer/analysis/scenario.py` | T0 | W1 | 150/80 | L | 1h |
| T5 | `coating_analysis` 엔진 | `trainer/analysis/coating.py` | T2,T4 | W2 | 250/250 | M | 2.5h |
| T6 | `sequence_analysis` 엔진(방향 5종, 순서 이탈, 궤적 지표) | `trainer/analysis/sequence.py` | T2,T4 | W2 | 350/300 | H | 4h |
| T7 | `pose_analysis` 엔진(관절 지목만, 레퍼런스 없으면 not_available) | `trainer/analysis/pose.py` | T2,T4 | W2 | 250/250 | M | 3h |
| T8 | `anomaly_detection` 조합기(재판정 금지) | `trainer/analysis/anomaly.py` | T2 | W2 | 150/150 | L | 1h |
| T9 | `knowledge_search` 인터페이스(필터, 실제 프롬프트 chunk만 citation) | `trainer/analysis/knowledge.py` | T2 | W2 | 150/150 | L/M | 1.5h |
| T10 | MCP 재구성: 5종 라우터, 옛 7종 삭제, 테스트 교체 | `mcp_servers/*` | T5–T9 | W3 | 300/250, 삭제 약 1.2k | M | 2.5h |
| T11 | 시퀀스 수집기·오케스트레이터(§18.9 순서, HTTP·SSE 동일 결과) | `trainer/demo_api/v1/sequence_ingest.py` | T1,T3,T5–T8 | W3 | 350/400 | H | 4h |
| T12 | v1 라우터 골격, start/end, 오류 핸들러(400/404/409/422/503) | `trainer/demo_api/v1/router.py`, `errors.py` | T1,T3 | W3 | 250/250 | M | 2h |
| T13 | 이상 알림 SSE(`Last-Event-ID`, heartbeat, 종료) | `trainer/demo_api/v1/anomaly_stream.py` | T3 | W3 | 200/250 | M | 2.5h |
| T14 | Query 재작성: task_type별 세션 context 조회, `explain()` 설명 단계, degraded 60초, `grounded_system` 프롬프트 | `copilot_router.py`, `agent_pipeline.py`, `rag_config_*.yaml` | T1,T3,T9 | W3b | 400/400 | H | 4.5h |
| T15 | 리포트 job: multipart 접수, 집계, 브리핑, 재기동 재개 | `trainer/demo_api/v1/reports.py`, `report_aggregate.py` | T3,T11,T14 | W4 | 550/450 | H | 5h |
| T16 | app 배선, readiness(저장소 쓰기 가능 여부), 내부 경로 profile 차단 | `app.py`, `readiness.py` | T12–T15 | W4b | 120/120 | M | 1.5h |
| T17 | LLM 도구 호출 제거(E2): `tool_calling_adapter`, 도구 루프, `live_sequence_fallback`, `zone_resolver`, 옛 스키마·프로토콜 삭제. `/api/chat`은 결정론 결과 설명 경로로 전환. 옛 도구 목록이 들어 있는 system prompt 정리 | `trainer/scripts/*`, `trainer/configs/*`, `trainer/demo_api/chat_service.py`, 관련 테스트 | T10,T14 | W4 | 300/300, 삭제 약 1.5k | H | 4h |
| T18a | UI v1 클라이언트, Zod, SSE 파서의 `id:`·heartbeat 처리 | `demo-ui/src/v1/*`, `demo-ui/src/api.ts`(파서) | T1 | W3 | 300/250 | M | 2h |
| T18b | 제품 세션·질의 화면(context·citation 분리 표시, 실제 단계 문구) | `demo-ui/src/product/*`(진입점 소유) | T18a | W4 | 300/250 | M | 2.5h |
| T18c | 제품 SSE·리포트 화면(multipart, 2~5초 폴링) | `demo-ui/src/product/*`(T18b와 다른 파일) | T18a | W4 | 300/250 | M/H | 2.5h |
| T19 | Pydantic↔Zod 스키마 동등성 시험 | `trainer/demo_api/tests/test_v1_schema_export.py`, `demo-ui/tests/v1Parity.test.ts` | T1,T18a | W4 | 50/150 | M | 1.5h |
| T20 | 계약·상태 시험(§16.1·§16.2), 운영 MCP 라우터가 정확히 5개인지 | `trainer/demo_api/tests/test_v1_contract.py`, `test_v1_state.py` | T16 | W5 | —/500 | M | 2.5h |
| T21 | 실표면·성능 스크립트(§14·§16.3) | `deploy/spark/scripts/acceptance_v25.py`, `trainer/scripts/bench_v25.py` | T16 | W5 | 600/150 | M | 3h |
| T22 | 제품 UI Playwright | `demo-ui/e2e/product-*.spec.ts`, 옛 `/api/chat` 의존 spec 정리 | T18b,T18c,T16 | W5 | —/200 | M | 1.5h |
| T23 | Spark 배포: compose(상태 볼륨, MCP 5종), lock(`python-multipart`), nginx(업로드 한도, 내부 경로 차단), README, 배포 테스트 | `deploy/spark/*`, `trainer/scripts/tests/test_spark_deploy_contract.py` | T10,T16,T17 | W5 | 250 + 문서/120 | M | 2.5h |
| T24 | 새 품질 평가(E3): 문항·fixture, 결정론 채점기, 생성 러너(openai/HF 백엔드), `quality_gate.py` 입력 호환 | `trainer/eval/v25/*`, `deploy/spark/scripts/run_fp8_gate.sh` | T14,T17 | W5 | 800/300 | H | 6h |
| T25 | '모델 초기화' 문구 수정(§17 721행) | `chat_service.py`, `demo-ui/src/stageLabels.ts` | — | W1 | 20/30 | L | 0.5h |
| T26 | 새 평가로 NF4 기준선 측정(A40, 8 seed, 라이브 API 중단 창) | `/output/eval_results/v25_nf4_baseline_<commit>/` | T24 | W6 | — | L | GPU 6~7h |
| T27 | 문서 정리: AGENTS.md, 이 문서의 §7 해결 내역을 계약 문서 보완안으로 정리 | `AGENTS.md`, `docs/*` | 전체 | W6 | 문서 | L | 1h |
| T28 | Spark 실기: 이미지 빌드, §16.3·§14 실측, 소크, FP8 새 게이트 | Spark | T23,T26 | W7 | — | — | 1~2일(사용자) |

## 4. 웨이브와 파일 소유

| 웨이브 | 동시 실행 | 파일 소유 규칙 |
|---|---|---|
| W0 | T0 | — |
| W1 | T1, T2, T4, T25 → T3 | T1~T4는 모두 신규 파일. T25만 `chat_service.py`, `stageLabels.ts`를 수정 |
| W2 | T5, T6, T7, T8, T9 | 엔진 파일이 서로 분리됨. `trainer/analysis/__init__.py`는 T2만 수정 |
| W3 | T10, T11, T12, T13, T18a | T12가 v1 라우터 조립 파일을 소유하고 T13은 `APIRouter`만 제공. `demo-ui/src/api.ts`는 T18a 단독 |
| W3b | T14 | `copilot_router.py`, `agent_pipeline.py`, `rag_config_*.yaml`을 T14가 단독 소유 |
| W4 | T15, T17, T18b, T18c, T19 | T15는 `agent_pipeline.py`를 수정하지 않고 `explain()`만 호출. T17은 T14 커밋 뒤에 `agent_pipeline.py`의 도구 루프 부분을 수정(T15와 파일이 겹치지 않음을 확인). T18b가 제품 진입점 소유 |
| W4b | T16 | `app.py`는 T16 단독 |
| W5 | T20, T21, T22, T23, T24 | `compose.yaml`, `test_spark_deploy_contract.py`는 T23 단독. `run_fp8_gate.sh`는 T24 단독 |
| W6 | T26, T27 | GPU 측정 중에는 다른 CPU 무거운 작업을 돌리지 않는다 |
| W7 | T28 | 사용자 Spark 작업 |

모든 태스크 공통 규칙:
- AGENTS.md를 따른다. Python은 컨테이너에서만 실행하고, 파일은 명시 경로로 stage하며, Conventional Commits 한국어로 커밋한다.
- TDD로 진행한다. 실패할 수 있는 단언만 쓴다. 출처와 다른 라벨을 붙인 숫자를 쓰지 않는다.
- 게이트: `pytest trainer mcp_servers -q`. UI 태스크는 `npm run build:web`과 `npm test`를 추가로 통과해야 한다.
- 라이브 서비스(:8100/:8120/:8130)를 재시작하지 않는다. 필요하면 오케스트레이터가 한다. 컨테이너에서 프로세스를 종료할 때는 `pgrep -f "^..."`처럼 앵커를 건 패턴을 쓴다.

## 5. 태스크별 필독 범위

계약 파일은 `docs/ai_copilot_v2_5_deployment_contract.md`, "옛 문서"는 `docs/ai_copilot_response_contract.md`다. 모든 태스크는 계약 1-165행(§1~§4)과 AGENTS.md를 공통으로 읽는다.

| 태스크 | 계약 줄 | 코드·문서 |
|---|---|---|
| T0 | 166-614, 725-1286 | PDF v2.5(`docs/260813 …v2.5).pdf`), `trainer/demo_api/tests/test_copilot_router.py` |
| T1 | 166-614 | `copilot_contracts.py`, `contracts.py`(StrictModel), `copilot_router.py:1-60` |
| T2 | 725-1241 | `mcp_servers/common.py` |
| T3 | 94-165, 179-228, 466-614, 634-703 | `sim_session.py`, 옛 문서 833-912, 1313-1337 |
| T4 | 179-354, 803-1087 | 옛 문서 99-246 |
| T5 | 803-909, 1242-1272 | `mcp_servers/dft_judge.py`, 옛 문서 153-246 |
| T6 | 993-1087 | `pass_segmentation.py`, `world_axes.py`, `pass_coaching.py`, 옛 문서 174-187, 380-447, 919-974, `docs/superpowers/specs/2026-08-03-uv-coverage-order-coaching-design.md` §5 |
| T7 | 910-992 | `pose_similarity.py`, `mcp_servers/pose_estimation.py:126-234`, `simulator_csv.py:23-31`, 옛 문서 188-226 |
| T8 | 1088-1176 | 옛 문서 354-457, `docs/superpowers/specs/2026-08-07-query-anomaly-response-contract-design.md` §3~§6 |
| T9 | 1177-1241 | `search_index.py:80-165`, `agent_pipeline.py:495-515` |
| T10 | 725-802, 1273-1286 | `mcp_servers/*`, `agent.Dockerfile:24-37` |
| T11 | 229-378, 1242-1272 | `simulator.py`(전이 순서), `monitor.py`, T2~T8 산출물 |
| T12 | 94-228, 466-484 | `copilot_router.py:57-60,159-173`, `app.py` |
| T13 | 355-378 | `monitor_api.py`, `monitor.py`, AGENTS.md "Testing gotchas" |
| T14 | 379-465, 634-647, 1177-1272 | `copilot_router.py`, `agent_pipeline.py:677-800,1018-1212`, `openai_backend.py:1-100`, `generation_slot.py`, `live_response_contract.py`, `rag_config_spark_vllm.yaml:81-190`, 옛 문서 597-832 |
| T15 | 485-614, 634-647, 700 | `session_csv.py`, `simulator_csv.py`, `pose_sequence_report.py:66-106`, 옛 문서 833-1312, `docs/superpowers/specs/2026-08-07-session-report-response-contract-design.md` |
| T16 | 648-660 | `app.py`, `readiness.py`, `chat_service.py:113-160` |
| T17 | 67-93, 1273-1286 | `tool_calling_adapter.py`, `agent_pipeline.py:597-619,963`, `live_sequence_fallback.py`, `zone_resolver.py`, `tool_calling_protocol.json`, `tool_schemas/`, `rag_config_*.yaml` prompt 블록 |
| T18a | 94-165, 355-465, 615-633 | `demo-ui/src/api.ts`, `contracts.ts`, `simMonitorClient.ts`, `demo-ui/tsconfig.json` |
| T18b/c | 179-228, 379-614, 615-647 | `App.tsx`, `ChatPanel.tsx`, `stageLabels.ts`, `answerFormat.ts`, T18a 산출물 |
| T19 | 663-670 | T1·T18a 산출물 |
| T20 | 661-703 | T12~T16 산출물 |
| T21 | 634-647, 681-703 | `bench_latency.py`, `bench_report.py`, `bench_sse.py`, `sim_dummy_driver.py`, `deploy/spark/scripts/soak.sh` |
| T22 | 615-633, 681-690 | `playwright.config.ts`, `demo-ui/e2e/*` |
| T23 | 648-660, 691-703 | `deploy/spark/*`, `test_spark_deploy_contract.py` |
| T24 | 76-84, 379-465, 1242-1272 | `run_agentic_eval.py`(구조 참고), `generate_agentic_responses.py`, `deploy/spark/scripts/quality_gate.py`, `run_fp8_gate.sh`, 계획 2026-09-15 D4 |
| T25 | 615-633, 721 | `chat_service.py:161-180`, `stageLabels.ts`, `stageLabels.test.ts` |

## 6. 진행 기록

(웨이브마다 커밋 해시, 테스트 수, 남은 위험을 적는다.)

## 7. 계약 빈칸과 잠정 규칙 (T0에서 확정)

| # | 빈칸·모순 | 잠정 규칙(초안) |
|---|---|---|
| D-1 | §18.4 906행 "§3.7 불변 규칙"이 v2.5에 없음 | 옛 문서 227행의 등급→status 불변식을 따른다(E9) |
| D-2 | 빈 히트맵일 때 도막 판정 규칙 없음 | `analysis_status=not_available`, `reason_code=heatmap_missing` |
| D-3 | spec, targets, 구역별 기대 방향의 제공 주체 없음 | 시나리오 설정(E5) |
| D-4 | `nominal_um`과 `target_thickness_um` 이름 불일치, 처방 조합 규칙 없음, `sequence_feedback.status` enum 없음 | 공개 응답은 §7.3 이름, 내부는 §18.4 이름을 쓰고 매핑한다. 처방은 각 엔진이 자기 코드만 낸다 |
| D-5 | HTTP 응답에 anomaly 객체가 없어 HTTP·SSE 일치를 볼 수 없음. §8.2 items에 `map_num`·`uv_bin` 누락 | SSE는 §18.7 결과를 그대로 싣는다. 일치 검증은 저장된 결과 기준으로 한다 |
| D-6 | 관절명 어휘, `rot` 단위와 순서, up축 미정의 | Unity Y-up(메모리 `csv-world-is-unity-y-up`), 관절명은 PDF p.14 어휘, 회전은 도(degree). 엔진 입력 정규화에서 명시 |
| D-7 | end와 complete의 순서, 미시작·종료 세션 처리 | complete는 active·ended 모두 허용한다. 종료 뒤 sequence는 409. 없는 세션은 404 |
| D-8 | 없는 세션 404와 "시작 전 시퀀스" 409가 충돌 | 저장소에 세션이 없으면 404, 있지만 active가 아니면 409 |
| D-9 | 처리 중인 같은 request_id, 1세션 1리포트에서 재요청, start·end 멱등 | 처리 중이면 409. 같은 세션의 새 complete는 기존 report_id를 돌려준다(payload가 다르면 409). start·end는 같은 payload면 기존 결과를 돌려준다 |
| D-10 | anomaly가 pose 결과를 받는지, `anomaly_type` enum 없음 | pose는 입력으로 받되, not_available이면 무시한다. enum은 coating, trajectory, sequence, pose, mixed |
| D-11 | "Trigger ON→OFF 패스" 단위와 실측 한 버스트 평균 4패스가 충돌. bucket 계산 주체, severity enum 없음 | 버스트 단위로 판정하고 패스 수를 보고한다. bucket은 엔진이 계산한다. severity는 minor, moderate, major |
| D-12 | 동기 Query에서 "실제 단계 표시" | UI는 요청 경과 시간과 "분석 결과 조회/근거 검색/답변 생성" 고정 순서만 표시하고, 모델 초기화 문구는 쓰지 않는다 |
| D-13 | `resolved_intents` 어휘, task_type별 context | PDF 예시 어휘를 쓴다(`trajectory_recommendation`, `pose_recommendation`, `coating_error_analysis`, `knowledge_search`) |
| D-14 | degraded 제한 시간 | §14의 60초 |
| D-15 | `telemetry_csv` 형식·크기, `height_map_image` 용도 | SIM CSV 방언(`session_csv`), 상한 50MB, 이미지는 보관만 한다. 2초 접수에 업로드 시간은 넣지 않는다 |
| D-16 | failed 리포트 조회 응답, 하위 `reason_code` | failed면 200 + `report_status=failed` + reason_code. 하위 리포트마다 reason_code를 둔다 |
| D-17 | Query의 `scenario_id` 필수 여부 | 선택. 없으면 세션의 scenario_id를 쓰고, 세션도 없으면 null |
| D-18 | SSE 이벤트 id 범위·보존 | 세션별 단조 증가(저장소 자동 증가 id), 재기동 후에도 연속, 세션 종료 후 24시간 보존 |
| D-19 | `spray_angle_deg` 원천, `straightness_score` 정의 | 각도는 `tool_pos_rot.rot`에서 구하고 없으면 null. straightness는 RMS 이탈을 100점으로 선형 환산하고 임계는 시나리오 설정에 둔다 |

## 8. 알려진 위험

- 좌표축 충돌: `world_axes.py`는 Y-up인데 `pass_segmentation.py`와 `normalize_pose`는 z를 수직으로 쓴다. T6·T7은 정규화 단계에서 축을 명시하고 테스트로 고정한다.
- `/api/chat` 도구 경로를 제거하면 내부 데모의 도구 기반 답변이 바뀐다(E2). 데모 e2e(`chat-sequence.spec.ts`)를 T22에서 정리한다.
- 새 평가로 측정한 점수는 옛 6.704와 비교할 수 없다. T26 전에는 FP8 품질 판정을 새로 내리지 않는다.
- p95 목표(시퀀스 500ms 등)는 수 MB JSON 처리에 달려 있다. T11에서 이중 파싱을 피하고, T21에서 실측한다.
- x86 개발 서버에서는 arm64 이미지를 빌드할 수 없다. 실측은 T28(Spark)에서만 가능하다.
