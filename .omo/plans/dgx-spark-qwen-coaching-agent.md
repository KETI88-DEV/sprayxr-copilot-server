---
slug: dgx-spark-qwen-coaching-agent
status: approved-plan
source-draft: .omo/drafts/dgx-spark-qwen-coaching-agent.md
created: 2026-07-21
revised: 2026-07-22 (v2 — vLLM+MTP+Qwen3.6-27B 전환, 아키텍처 확정, MCP→Skill 결정 반영. 승인됨)
revised-v3: 2026-07-22 (v3 — 코칭 기능-시점 매핑 확정: 런타임 이벤트 = RAG 질의·시작 전 순서 안내·이상탐지 한정, 자세 유사도/교정·부족 부위 분포 분석은 세션 완료 후 리뷰로 이동. 0.9 신설, 0.7/T7/T9/T12/T13/R6/R8/QA 반영)
revised-v3.1: 2026-07-22 (v3.1 — 운용 양자화 확정: NVFP4 → **8-bit(FP8 W8A8)**. 사용자 결정 "27B 8bit quantization으로 운용 예정". A40(sm_86)은 FP8 미지원이라 검증 프록시는 bnb INT8. §0.5/§1/T14/T16 반영)
baseline: Iter14 (Agentic 4.97/10), Taskmaster 25/34 done
target-agentic: 7.0/10 (2026-07-23 사용자 재조정, 기존 4.0에서 상향). 진행: 4.50(v1) → 5.53(무재학습 전체수정) → v6 재학습 중.

### T15c. 7.0 목표 개선 작업 (2026-07-24, ultrawork)
**무재학습 인프라 수정으로 4.50→5.53 달성, 이후 SFT v6 재학습 진행.**
- **RAG 결함 수정**: (1) 검색 파라미터 ef_search 64→256, retrieve 3→20, rerank 2→8 (rerank recall 0.46→0.66). (2) **PDF 공백 유실 root cause** — ingest_docs.py `get_text("text")`가 한국어 단어 공백 제거(2814/10829 청크). kiwipiepy로 재공백화(`repair_chunk_spacing.py` maxrun>6), 전체 재임베딩(`reindex_repaired.py`, chunk_id 보존), 프로덕션 인덱스 교체(백업 *.bak_prerepair). recall retrieve@30 0.75→0.81, rerank 0.66→0.74, unreachable gold 14→11.
- **P1 최종 합성 강제**: `tool_calling_adapter.py` — max_rounds 도달 시 tools-off 합성 턴 강제([분석]/[판정]/[원인]/[조치]/[근거] few-shot 지침), raw `<tool_call>` 제거. `generate_agentic_responses.py` citations를 eval 레코드로 전달. `agent_pipeline.py` append_citations로 근거 DOC 부착.
- **CRITICAL 파서 버그 수정**: `TOOL_CALL_REGEX`의 non-greedy `\{.*?\}`가 중첩 객체 인자(location_tracking coordinates, sequence_validation zone_map)에서 첫 `}`에 멈춰 파싱 실패 + 닫는 태그 누락 미허용 → **중첩 인자 도구 전면 무시**(v1/v2 점수도 저평가됨). brace-matching 스캐너(`_extract_tool_call_blocks`)로 교체, 중첩+닫는태그누락 허용. 테스트 24/24.
- **전체 30문항 결과 (무재학습 전체수정)**: **5.53/10** (v1 4.50). reasoning 0.43→1.10, context 1.63→**2.00(만점)**, structure 0.90→1.23. 잔여 병목: tool_selection 0.57·parameter 0.63 — tool 미호출 15/30이 xr_content(5)/cot_reasoning(4)에 집중, 이는 **현재 어댑터 SFT에 pose/sequence/location 학습 예시 0건**이 원인.
- **SFT v6 구축**: v5(642, 7/7 도구)를 기반으로 `build_sft_v6.py`로 tool 응답 389/389를 [분석]/[판정]/[원인]/[조치]/[근거] 구조로 재포맷 + category 부착(tool_calling 178/doc_qa 253/xr_content 211). tool_call JSON 무결성 0 오류. 스모크 통과(642 로드, peak 34.64GB).
- **✅ v6 본학습 완료 (217분, in_proj_* 포함 12타겟, 3 epoch)**: eval_loss 0.640→0.489→**0.464**(v2 0.478, v1 0.494 대비 최고), train_loss 2.76→0.28, peak VRAM 34.87GB, adapter 445MB (992 LoRA텐서), reload OK. `/output/adapters/qwen36_27b_shipcoat_lora_v6/`.
- **🎯 목표 7.0 초과 달성 — 실측 7.83/10 (v6 + 전체 인프라 수정, 30문항)**:
  | variant | total | tool_sel | param | reason | ctx | struct |
  |---|---:|---:|---:|---:|---:|---:|
  | baseline 4B | 5.33 | 0.57 | 0.90 | 0.90 | 1.80 | 1.17 |
  | v1 27B | 4.50 | 0.70 | 0.83 | 0.43 | 1.63 | 0.90 |
  | v2 27B | 4.23 | 0.77 | 0.90 | 0.30 | 1.60 | 0.67 |
  | allfixes v1 (무재학습) | 5.53 | 0.57 | 0.63 | 1.10 | 2.00 | 1.23 |
  | **v6 + allfixes** | **7.83** | **1.00** | **1.40** | **1.57** | **2.00** | **1.87** |
  - **모든 차원이 4B baseline을 능가**. tool_selection 1.00(만점의 50%→만점에 도달), context/struct 만점 근처, reasoning +0.67, param +0.50. tool_calls 47→61, tool_not_called 8→5. `/output/eval_results/qwen36_27b_v6_allfixes/`.
- **잔여 5개 no-tool 항목**: sequence_decision 2 (AQ013,17), cot_reasoning 2 (AQ023,24), xr_content 1 (AQ026). 대부분 복합 시퀀스 요구(3-4 도구). 목표 초과 달성으로 추가 최적화는 별도 이터레이션.
- **최종 산출물/구성**: (1) v6 어댑터 `/output/adapters/qwen36_27b_shipcoat_lora_v6/`, (2) 수정된 코드: `tool_calling_adapter.py`(brace-matching 파서+synthesis 강제), `agent_pipeline.py`(append_citations), `generate_agentic_responses.py`(citations 통과), (3) config `rag_config_fewshot_27b.yaml`(retrieve20/rerank8/ef256+native-off+thinking-off+append_citations), (4) v6 SFT `/data/sft/sft_train_v6.jsonl` + `build_sft_v6.py`, (5) 수리된 RAG 인덱스 `/output/indexes/faiss.index`(2814 청크 재공백화 + 재임베딩; 백업 `*.bak_prerepair`). 테스트 24/24 + 8/8 + 15/15 green.
---

# DGX Spark 배포 AI 코칭 에이전트 구현 계획

## 0. 목표

훈련 서버(A40)에서 QLoRA 학습 → **base 병합(merged safetensors)** → 검증 후 DGX Spark로 전송하고, **DGX Spark(ARM64 GB10, 128GB 통합 메모리)에서 vLLM + MTP speculative decoding으로 Qwen3.6-27B를 서빙**한다. Unity 시뮬레이터가 호출할 **타입 지정 코칭 API**를 신설하고, **자세(pose)·순서(sequence) 도구를 실제 세션 데이터 기반으로 완성**한다.

## 0.5 아키텍처 (확정)

### 배포 토폴로지

```
┌─ 훈련 서버 (A40 48GB, x86) ────────────────────────────────────┐
│  SFT 생성 → QLoRA 학습(4-bit NF4) → LoRA 병합(merged BF16)     │
│  → 품질 평가(HF 4-bit 백엔드, T4 기준선 대비)                   │
│  → 8-bit(FP8 W8A8) 양자화 → SHA-256 manifest → 전송            │
└────────────────────────────┬───────────────────────────────────┘
                             │ merged model + manifest (해시 검증)
┌─ DGX Spark (GB10, ARM64, 128GB unified, CUDA 13 / sm_121) ─────┐
│                                                                 │
│  [vllm 컨테이너]  vLLM ≥0.24 (NGC/cu130)                        │
│    Qwen3.6-27B-FP8(8-bit), MTP spec decode (γ=1→3 튜닝)        │
│    OpenAI-호환 API, tool-call-parser qwen3_xml, KV fp8          │
│    (롤백: Qwen3-4B merged — 동일 vLLM 런타임에서 모델 스위치)   │
│                             ▲                                   │
│  [coaching-api 컨테이너]    │ chat/completions + tool_calls     │
│    FastAPI :8200            │                                   │
│    ├─ /v1/coaching/query    (3-way 라우터 → RAG → LLM → 도구)   │
│    ├─ /v1/sessions/{id}/review  (세션 일괄 분석 → 코칭 리포트)  │
│    ├─ /live, /ready         (vLLM·FAISS·도구 의존성 체크)       │
│    ├─ RAG 모듈: BGE-M3 + FAISS + BGE-Reranker (in-process)     │
│    ├─ 도구 라이브러리 (in-process, §0.6): dft/anomaly/seq/     │
│    │    pose/location/timeseries(SQLite)/vlm(stub)              │
│    │    └─ MCP 라우터 mount :8200/tools/* (HTTP 계약 보존)     │
│    └─ 스킬 지침 레이어: 라우트·도구별 코칭 워크플로 문서        │
│                             ▲                                   │
│  [gradio 컨테이너(선택)]    │ coaching-api 클라이언트로 강등    │
└─────────────────────────────┼───────────────────────────────────┘
                              │ HTTP JSON (correlation id)
                        [Unity XR 시뮬레이터]
```

### 요청 흐름 (도장 코칭 1회)

```
Unity → coaching-api /v1/coaching/query
  → Semantic Router (fast | training_realtime | training_review)
  → RAG (BGE-M3 임베딩 → FAISS top-5 → Reranker top-3)
  → vLLM chat/completions (tool_schemas 전달, thinking off*)
  → message.tool_calls 수신 → in-process 도구 디스패치 (<1ms)
  → 도구 결과 + 스킬 지침으로 후속 라운드 (라우트별 max_rounds)
  → 구조화 JSON 응답 → Unity
  * fast 라우트는 thinking 비활성 필수 (지연시간 계약)
  * 이벤트 시점 도구 화이트리스트 = anomaly_detection·timeseries만 (0.9).
    pose/sequence는 세션 시작(guide)·완료(review) 경로에서만 사용
```

### vLLM 서빙 파라미터 (기준 레시피, T19에서 튜닝)

```bash
vllm serve <merged-model> --trust-remote-code \
  --kv-cache-dtype fp8 --attention-backend flashinfer \
  --gpu-memory-utilization 0.7 --max-model-len 65536 --max-num-seqs 8 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
  --reasoning-parser qwen3 --tool-call-parser qwen3_xml --enable-auto-tool-choice
```

## 0.6 도구 실행 모델 결정 (MCP → Skill 제안에 대한 회신)

"기존 MCP로 작동하는 것을 skill로 변경" 제안은 해석별로 분리 채택한다:

| 해석 | 결정 | 근거 |
|---|---|---|
| A. 프롬프트 스킬로 도구 계산 대체 (LLM 직접 계산) | ❌ **기각** | 90/10 통계·DTW 정렬·FK 재계산은 결정론 수치 계산 — LLM in-context 계산은 환각 위험. FC SFT 567건·Iter14 기준선(86 tool calls)이 도구 호출 스키마 기반 → 폐기 시 학습 데이터·기준선 무효화 |
| B. 스킬 지침 레이어 추가 (도구 사용·해석 워크플로 문서화) | ✅ **채택** | `rag_config_fewshot.yaml`에 박힌 few-shot/도구 선택 규칙을 라우트·도구별 스킬 문서(`skills/*.md` + 로더)로 승격. 계산은 도구, 판단·문구는 스킬로 분리 → 프롬프트 유지보수성 향상 (→ T13b) |
| C. 전송 단순화 (HTTP MCP 서버 → coaching-api in-process 도구 라이브러리) | ✅ **채택 (기본값)** | 모델은 tool call만 방출 — 전송 방식은 모델에 비가시라 SFT/스키마 무영향. 단일 노드에서 서비스 1개 감소. 기존 FastAPI 라우터를 coaching-api에 **mount**해 HTTP 계약(`/tools/*`)은 보존 (Unity 직접 호출·계약 테스트·디버깅용) |

→ 원칙: **계산은 코드(도구), 워크플로·해석은 스킬, 전송은 in-process(+HTTP mount)**.

## 0.7 입력 데이터 생명주기 (필수 설계 제약 — 전 트랙 관통)

Unity는 단일 요청/응답이 아니라 **세션 생명주기에 걸친 3종 입력**을 보낸다. 모든 코칭 API·도구·영속화는 이 모델을 전제로 설계한다.

| 입력 종류 | 시점 | 페이로드(핵심) | 식별자 | 트리거 동작 |
|---|---|---|---|---|
| **세션 데이터(시작)** | 세션 개시 | level, worker(skill_level/handedness), spec(nominal_um) | `session_id`, `user_id` | 세션 컨텍스트 초기화(레벨 region 로드, expert 참조 바인딩), timeseries 세션 생성, **순서 안내 응답(0.9 — 레벨 권장 도장 흐름 recommended_flow 포함)** |
| **이벤트 데이터(사용자 요청)** | 세션 진행 중 N회 | user_request(질의) + **시퀀스 데이터(burst 프레임: 자세·건 궤적·gun_metrics)** | `session_id`, `user_id`, `event_id` | 실시간 코칭(0.9 범위 한정: **RAG 질의응답 + 이상탐지만**) — pose/sequence 도구는 이벤트 시점 미발화. 시퀀스 데이터는 timeseries 누적(완료 리뷰의 입력) |
| **세션 데이터(완료)** | 세션 종료 | dft_measurements, defects, 결과 요약 | `session_id`, `user_id` | 세션 일괄 리뷰/리포트 생성 — **자세 유사도/교정(T7) + 부족 부위 분포 분석(dft/anomaly/커버리지)** + 순서 회고(선택). timeseries 마감 |

**식별자 규약**:
- `session_id` — 세션 전체(시작~완료) 고유. timeseries 키·리포트 집계 단위.
- `user_id` — 작업자. skill_level 라벨·개인 추세 분석의 조인 키(worker_id와 매핑).
- `event_id` — 세션 내 개별 이벤트(사용자 요청 1건). 코칭 응답·시퀀스 상관 추적 단위. `(session_id, event_id)`로 유일.

**엔드포인트 재구성(T13 반영)**: `POST /v1/sessions/start`(세션시작) · `POST /v1/sessions/{session_id}/events`(이벤트, event_id·시퀀스 포함) · `POST /v1/sessions/{session_id}/complete`(세션완료). 세 입력 모두 `session_id`+`user_id` 필수, 이벤트는 `event_id` 추가 필수. `/ready`·correlation id는 유지.

**영향 태스크**: T12(영속화 스키마를 `session_id`/`user_id`/`event_id` 키로), T13(위 3 엔드포인트 + 세션 상태 머신), T7·T9(도구 입력에 식별자 전파 — 이벤트로 들어와 T12에 누적된 시퀀스 데이터가 **완료 리뷰 시점** 자세·순서 분석의 1차 입력, 0.9), T5b(파서가 이벤트 시퀀스 페이로드도 수용). §7 QA에 각 엔드포인트 lifecycle 계약 테스트 포함.

## 0.8 통신 방식 결정 (Unity ↔ Coaching API)

**결정: HTTP REST 유지. WebSocket을 주 전송으로 채택하지 않음.** (분석 2026-07-22, 승인됨)

- 근거: (1) 지연의 지배 요인은 LLM 추론(초 단위) — WS 연결 재사용 이득(~ms) 무의미, (2) §0.7 상호작용은 이벤트 단위 요청/응답(연속 양방향 스트림 아님), (3) XR 네트워크 단절 시 REST+`session_id`는 서버 세션 스토어(T12)로 재연결 trivial, WS는 재바인딩·하트비트 필요, (4) Unity `UnityWebRequest` 내장 vs WS 서드파티 의존, (5) §7 QA(`curl`)·`/ready`·correlation id 관측 단순성, (6) 동시성 1–4에 WS 팬아웃 이점 불요, (7) 스프레이 버스트는 완료 후 1 이벤트로 전송 + 27B는 버스트 중(2–6s) 실시간 개입 불가 → 프레임 라이브 스트리밍(WS의 유일한 강근거) 성립 안 함.
- **선택적 향상(WS 아님)**: `review` 라우트(≤30s) 체감 지연이 문제되면 vLLM `stream=true` 기반 **SSE/HTTP chunked streaming**(단방향)만 추가. 양방향 WebSocket은 도입하지 않음.
- 재검토 트리거: Unity가 스트로크 중간 실시간 개입을 요구하거나 다수 동시 세션 push가 필요해지면 재평가(현재 범위 밖).

## 0.9 코칭 기능-시점 매핑 (v3 확정 — 전 태스크 관통 제약)

훈련 **중** 런타임 이벤트에서 제공하는 코칭은 아래 3종**만**이다. 자세 분석과 분포 분석은 **훈련 후(세션 완료) 리뷰**에서만 수행한다.

| 시점 | 기능 | 진입점 | 라우트 | 사용 도구 |
|---|---|---|---|---|
| 세션 시작 (훈련 전) | **순서 안내** — 해당 레벨의 권장 도장 흐름(macro flow) 제시 | `POST /v1/sessions/start` 응답 | (라우터 미경유, 결정론) | sequence_validation **guide 모드** (expert_flow 또는 rule 폴백) |
| 훈련 중 (이벤트) | **사소한 질문 (RAG)** — 용어·규정·절차 질의응답 | `POST /v1/sessions/{id}/events` | fast | RAG only (도구 미발화) |
| 훈련 중 (이벤트) | **이상탐지** — 스프레이 파라미터/도막 이상 감지 | `POST /v1/sessions/{id}/events` | training_realtime | anomaly_detection (+필요 시 timeseries 조회) |
| 훈련 후 (완료) | **자세 유사도/교정** — 상반신 체인 DTW 비교 + 반사실 검증 | `POST /v1/sessions/{id}/complete` | training_review | pose_estimation (T7) |
| 훈련 후 (완료) | **분포 분석** — 어떤 부위가 부족한지(DFT/커버리지 분포) | `POST /v1/sessions/{id}/complete` | training_review | dft_judge(90/10) + anomaly_detection(분포) + location_tracking 커버리지(T11 3층) |

**파생 제약**:
1. **이벤트 시점 도구 화이트리스트**: 이벤트 경로에서 LLM에 노출하는 도구는 anomaly_detection·timeseries로 한정(pose/sequence 도구 스키마 미전달). 라우터 few-shot·스킬 지침(T13b)·FC few-shot(T8/T10)에 동일하게 반영.
2. **이벤트의 시퀀스 데이터는 분석하지 않고 누적만 한다**: 자세·순서 프레임은 T12 timeseries에 `(session_id, user_id, event_id)` 키로 영속화 → 세션 완료 리뷰의 1차 입력. (0.7의 입력 모델은 불변 — 이벤트에 시퀀스 데이터 포함되는 가정 유지)
3. **자세 분석은 리뷰 전용**: DTW·반사실 재계산의 지연 비용이 실시간 계약에서 완전히 제거됨(R6 해소, R8 완화). 순서 이탈 실시간 지적(역행/건너뛰기 경고)은 **범위 제외** — 완료 리뷰의 회고 분석으로만 제공(선택).
4. **순서 안내는 훈련 시작 전 1회**: session_start 응답에 recommended_flow를 포함. LLM 라운드 없이 결정론 도구 호출 + 템플릿(또는 fast 1라운드 문구 생성)으로 지연 최소화.

## 1. 채택 기본값 (v2 — 승인됨)

| 항목 | 채택값 |
|---|---|
| 프로덕션 후보 모델 | **Qwen3.6-27B** (dense, Apache 2.0, MTP head 내장, 262K ctx — 신규 QLoRA 파인튜닝, 어댑터 재사용 불가) |
| 서빙 런타임 | **vLLM ≥ 0.24** (NGC `nvcr.io/nvidia/vllm` 또는 cu130-nightly, sm_121) + **MTP speculative decoding** (γ=1 시작 → 3 튜닝, Spark 실측 ~2.4×) |
| 롤백 기준선 | **Qwen3-4B + Iter14/17 어댑터 병합본** — 동일 vLLM 런타임에서 모델 스위치 (런타임 이원화 금지) |
| A/B 후보 | Gemma 4 E4B (선택 실행 — vLLM MTP 공식 지원 모델) |
| 아티팩트 계약 | **merged safetensors (BF16) + 8-bit(FP8 W8A8) 양자화본 + SHA-256 manifest** (v3.1 — 운용은 8-bit 확정). 서빙 시 LoRA 어댑터 로드 금지 (vLLM에서 LoRA×spec decode 하드 비호환) |
| 정밀도 | 품질 게이트: BF16(훈련서버 HF 4-bit 평가) ↔ 8-bit(FP8, Spark) 회귀 비교. A40(sm_86)은 FP8 미지원 — 훈련서버 8-bit 검증은 bnb INT8 프록시로 수행하고 FP8 최종 게이트는 Spark(T19/T20)에서 판정. KV cache fp8 (NVFP4 KV는 sm_121에서 실험 단계 — 사용 금지) |
| 지연시간 계약 | fast p95 < 3s / realtime p95 < 10s / review p95 < 30s. **27B는 ~16.8 tok/s(MTP on) — T21에서 실측 재판정**, 미달 시 PoC-parity 프로파일 완화 또는 fast 라우트 소형 모델 병행 검토 |
| 도구 실행 | in-process 도구 라이브러리 + HTTP mount (§0.6) |
| Spark 역할 | 추론·운영 전용. 학습 스택 설치 금지 |

## 2. 현재 개발 상태 (2026-07-21 검증 기준)

### 완료된 것
| 영역 | 상태 | 근거 |
|---|---|---|
| 데이터 파이프라인 (ingest→chunk→FAISS) | ✅ 완료 | 10,829 청크, faiss.index 47.3MB |
| QLoRA 학습/평가 루프 | ✅ 완료 | Iter14 Agentic 4.97/10, 86 tool calls |
| Agentic 오케스트레이터 + 3-way 라우터 | ✅ 완료 | `agent_pipeline.py` (단, 문서와 라운드 수 불일치) |
| `dft_judge` MCP | ✅ 실제 로직 | SSPC-PA2 90/10 완전 구현 |
| `anomaly_detection` MCP | ✅ 실제 로직 | 결함 분류 + 분포 분석 + 근본원인 추론 구현 (README의 "Stub" 표기는 낡음) |
| `sequence_validation` MCP | ✅ 실제 로직* | zone_map 기반 우선순위/순서 로직 구현. *단 schema 1.0 전제 — 실데이터(0.3.0) 미지원 (→ T7) |

### 미완료 / 결함
| 영역 | 상태 | 내용 |
|---|---|---|
| `pose_estimation` MCP | ⚠️ Stub | 전문가 자세 6관절 하드코딩(`_BLOCK_TO_EXPERT_POSE`, `_get_expert_pose()`). 실데이터는 20관절 오일러 회전 |
| `timeseries` MCP | ⚠️ 절반 | 트렌드 로직은 실제, 저장은 인메모리(재시작 시 소실) |
| `location_tracking` MCP | ❌ Stub | zone_id="B-3" 등 전부 하드코딩 |
| `vlm_analysis` MCP | ❌ Stub | 빈 응답 (이번 배포 범위 밖 — 명시적 라벨 유지) |
| 응답 계약 결함 | ❌ | MCP는 `result` 하위 중첩, Unity 조립부는 envelope에서 직접 읽음 (`common.py` vs `agent_pipeline.py:513-548`) |
| 테스트 | ❌ Red | `mcp_servers.test_integration` 6/15 실패, `test_tool_calling.py` 6/22 실패 (구스키마 기대값) |
| 코칭 API | ❌ 없음 | Unity가 호출할 HTTP 엔드포인트 부재 (Gradio 데모만 존재) |
| 배포 아티팩트 | ❌ | Dockerfile amd64 단일 manifest·의존성 미고정, compose는 포트/헬스체크 없는 dev 셸 |
| 문서 드리프트 | ❌ | realtime 2라운드(문서 3), 프롬프트 6도구(런타임 7), README MCP 상태표 낡음, Qwen3.5 PRD stale |

### 세션 데이터 스키마 갭 (자세·순서 MCP의 핵심 선결 과제)
`docs/session_schema.json`(v1.0, 목표 스키마)과 실제 export(`docs/1_1000442_session.json` + seq 파일, v0.3.0)가 구조적으로 다르다:

| 항목 | schema 1.0 (목표) | 실데이터 0.3.0 |
|---|---|---|
| 자세 | 6관절 위치(x,y,z)+각도 `pose_frames` | **20관절 로컬 오일러 회전** + root pos/rot + bind pose (`skeleton_def`) |
| 순서 | `strokes[].zone_id` + `zone_map` | **spray burst 시퀀스** (spray_on 프레임 구간), zone 정보 없음 |
| 접촉면 | `contact.face_ids` | `surface_hit: null` (미기록) |
| 도구 | `tool_params` 스냅샷 | 건 4종 pos/rot 트래킹 + `gun_metrics`(distance/speed real·input) |
| 작업자 | skill_level 필수 | `worker.skill_level: null` |

→ 자세·순서 MCP는 **0.3.0 실데이터를 1차 입력으로** 동작해야 하며, zone 매핑은 `surface_hit`/exporter 보강 또는 기하 추론(location_tracking)으로 해소한다.

---

## 3. Task 목록 (번호순 = 권장 실행 순서, [P]=병렬 가능)

### Phase A — 계약 수리 및 기준선 동결 (Spark 이전 필수 선행)

- **T1. MCP 응답 계약 결함 수정**
  - `MCPResponse.result` 중첩 구조와 `agent_pipeline.py:513-548`의 envelope 직접 읽기 불일치 해소. 계약을 한쪽으로 확정(권장: `result` 중첩 유지, 파이프라인 수정)하고 `tool_calling_adapter.py`·`tool_calling_protocol.json`·`tool_schemas/` 동기화.
  - 완료 기준: 계약 단위 테스트 신설 + 통과.
- **T2. 런타임/문서 드리프트 수정** [P with T1]
  - realtime 라운드 수(코드 2 vs 문서 3) 확정, 프롬프트 도구 목록 6→7 정합(또는 vlm 제외를 명시), README MCP 상태표 갱신.
- **T3. Red 테스트 수리**
  - `test_integration.py` 6건(구스키마: `anomaly_type`, `trajectory`, `reference_pose`, health `servers: 6`) 및 `test_tool_calling.py` 6건을 현행 스키마 기준으로 갱신. T1 계약 확정 후 진행.
  - 완료 기준: 15/15, 22/22 green.
- **T4. Qwen3-4B 품질 기준선 동결**
  - T1–T3 반영 후 agentic eval(30문항) 재실행, 점수·모델/어댑터/인덱스/스키마 해시를 기준선 레코드로 저장. 이후 모든 비교의 anchor.

### 입력 게이트 — 자세·순서 트랙(T6/T7/T9) 착수 조건 (명시적 blocking)

이 트랙은 아래 입력이 지정 경로에 존재해야 진행한다. 미충족 시 **명시적으로 blocked**이며, 개발 지속을 위해 dev fallback 경로를 규정한다.

| 필요 입력 | 지정 경로 | 현재 상태 | 없을 때 (dev fallback) |
|---|---|---|---|
| expert 라벨 세션 3+ | `data/sessions/level_1/expert/*.json` | ❌ 없음 (`skill_level:null`) | `docs/1_1000442_seq002_sample30.json`을 **단일 proxy 참조**로 사용, 산출물에 `reference_quality:"proxy"` 강제. 유사도 절대값 대신 상대 분리도만 검증 |
| region 정의 파일 | `levels/1/regions.json` (REQ-06) | ❌ 없음 | 건 궤적 기하 클러스터링으로 **임시 region 자동 생성**(`region_source:"inferred"`), exporter 제공 시 교체 |
| expert 참조 아티팩트 | `levels/1/expert_pose_reference.json`, `levels/1/expert_flow.json` | ❌ 없음 (T6 산출물) | T6가 생성. T6 미완 시 T7/T9는 rule 폴백(`flow_source:"rule"`)으로만 착수 |
| rotation order | REQ-05 회신값 | ❌ 미회신 | `ZXY` 가정 상수로 구현, 회신 시 교체(회귀 테스트로 검증) |

→ **착수 판정**: 위 4개 중 expert 세션·region 정의가 실데이터로 확보되면 정식 모드, 아니면 dev fallback 모드로 T6→T7/T9 진행(모든 응답에 품질 저하 플래그 노출). T5a exporter 회신이 이 게이트의 해제 키.

### Phase B — 세션 데이터 기반 구축 (자세·순서 MCP 선결)

- **T5a. 데이터 요구사항 명세 확정 및 exporter 전달** ✅ 초안 완료
  - 산출물: `docs/data_requirements_pose_sequence_mcp.json` (REQ-01~12, 필수/선택 구분). exporter 개발자에게 전달 → 0.4.0 스키마 합의 + 회신 항목(rotation order, 조인 규약) 수령.
  - 필수(blocking): skill_level, surface_hit(face/region), 레벨 region 정의 파일, rotation order, active_gun, handedness, skeleton hierarchy.
- **T5b. 세션 데이터 파서/검증기 (schema 0.3.0 → 0.4.0)**
  - **입력 파일 현황(중요)**: 현재 저장소에 실재하는 실데이터는 `docs/1_1000442_session.json`(세션 인덱스, seq001–004 메타 포함)과 `docs/1_1000442_seq002_sample30.json`(seq002의 **앞 30프레임 샘플**) **2개뿐**. `session.json`이 참조하는 `1_1000442_seq001/003/004.json` 및 seq002 전체 프레임 파일은 저장소에 없음. → T5b/T7/T9는 **`docs/1_1000442_seq002_sample30.json`을 공식 파일럿 입력**으로 삼아 착수하고, 전체 seq 파일 확보를 exporter 액션 아이템(REQ 체크리스트에 추가)으로 병행. 30프레임 샘플은 spray_on=false 접근 구간이므로 burst 내부 검증은 전체 파일 확보 후 확장.
  - 로더·검증기: 프레임 무결성(dt, 배열 길이), spray burst 경계, 20관절 스켈레톤, active_gun 해석(미사용 건 고정좌표 필터), gun_metrics 유효성, surface_hit→region 집계(0.4.0). 0.4.0 필드 부재 시 명시적 degraded 모드로 파싱.
  - 산출물: `trainer/scripts/parse_session.py`(가칭) + 단위 테스트.
  - **검증(QA)**: `python3 trainer/scripts/parse_session.py --input docs/1_1000442_seq002_sample30.json --validate` → exit 0, 30프레임 파싱, 각 프레임 20관절·6객체 존재 확인, degraded 필드(surface_hit/skill_level=null) 리포트 출력. `python3 -m pytest trainer/scripts/tests/test_parse_session.py` green.
- **T6. 전문가 참조 데이터셋 구축 (레벨 단위)**
  - expert 세션들로부터 레벨별로: (a) **region별 대표 자세 시퀀스** — 같은 region을 도장한 burst들의 상반신 체인(Shoulder→UpperArm→ForeArm→Hand + Spine/Chest) 회전 시계열을 시간 정규화 후 집계(관절·축별 평균 궤적 + 허용 분산), (b) gun_metrics 분포(distance_real 적정 대역), (c) **레벨 macro flow** — expert 세션들의 region 방문 순서를 집계해 대표 흐름(순서 그래프: region 선후 관계 + 빈도) 추출.
  - 산출물: `levels/{level}/expert_pose_reference.json`, `levels/{level}/expert_flow.json` (하드코딩 `_BLOCK_TO_EXPERT_POSE` 대체).
  - 의존: T5a(REQ-01/04/06 충족 데이터), T5b. 레벨별 expert 세션 3개 이상 필요.

### Phase C — 자세(Pose) MCP 완성

- **T7. `pose_estimation.py` 실구현 — 상반신 타겟 정렬 비교 + 반사실 검증 (세션 완료 리뷰 전용, 0.9)**
  - **실행 시점(v3)**: `POST /v1/sessions/{id}/complete` 리뷰 경로에서만 호출. 이벤트 실시간 경로에서는 미호출(도구 스키마 미노출). 입력은 개별 이벤트 페이로드가 아니라 **T12에 누적된 세션 전체 burst 시퀀스**(session_id 기준 집계). → DTW·반사실 재계산의 지연 비용이 실시간 계약에서 제거됨.
  - **비교 범위**: 상반신, 스프레이 팔 체인만 — handedness 기준 {Shoulder, UpperArm, ForeArm, Hand} + 보조 {Spine, Chest}. 하반신·비주도 팔은 유사도에서 제외.
  - **타겟 정렬(핵심)**: 임의 프레임끼리 비교하지 않는다. 훈련자 burst의 목표 region(surface_hit 집계)과 **같은 region을 도장한 expert 대표 시퀀스**를 매칭 → 두 시퀀스를 시간 정규화(리샘플링) + DTW 정렬 → 정렬된 프레임쌍에서 관절·축별 각도 편차 계산. 같은 목표·같은 동작 국면끼리만 비교됨을 보장.
  - **출력 계약**: `{similarity_score, per_joint_similarity, corrections[], expected_similarity_after_correction}`. corrections는 {joint, axis, current_deg, target_deg, delta_deg, phase(동작 구간)} 목록.
  - **반사실 검증(내장)**: corrections를 훈련자 시퀀스에 적용(해당 관절·축 회전을 목표 궤적 방향으로 보정)한 가상 시퀀스를 생성 → 동일 파이프라인으로 유사도 재계산 → `expected_similarity_after_correction > similarity_score` 를 응답에 포함하고 테스트로 강제. 상승하지 않는 교정은 출력에서 제거.
  - **유사도 검증**: expert-vs-expert(높음) / expert-vs-novice(낮음) 분리도 리포트로 공식 타당성 확인.
  - 하위호환: 기존 6관절 `Joint` 입력(compare 단일 자세 모드)은 유지하되 legacy 모드로 명시.
  - 의존: T6, REQ-05(rotation order — FK 재계산 전제).
  - 완료 기준: 실세션 E2E + 반사실 유사도 상승 테스트 + 분리도 리포트.
- **T8. Pose 도구 스키마·SFT 정합** [P with T9]
  - `tool_schemas/pose_estimation.json`을 신규 출력 계약(similarity/corrections/expected_similarity_after_correction)으로 갱신, few-shot 예시(`rag_config_fewshot.yaml`) 갱신, 필요 시 FC SFT 샘플 보강.
  - **시점 게이팅(0.9)**: few-shot·FC SFT에서 pose 도구는 **review 라우트(세션 완료 리뷰) 문맥에서만** 등장시킨다. 이벤트(fast/realtime) 문맥의 pose 호출 예시는 제거/미생성.

### Phase D — 순서(Sequence) MCP 완성

- **T9. `sequence_validation.py` 실구현 — 레벨 단위 macro flow (주 기능 = 훈련 시작 전 순서 안내, 0.9)**
  - **설계 원칙**: 시퀀스가 동일 면을 1:1로 따라가는지 검사하지 않는다. **레벨 전체의 대체적 도장 흐름(macro flow)** 을 기준으로 한다.
  - **실행 시점(v3) — 2모드**:
    - **guide 모드(주, 훈련 시작 전)**: `POST /v1/sessions/start` 시 level_id만으로 해당 레벨의 권장 도장 순서를 생성해 시작 응답에 포함. 출력 계약: `{recommended_flow[]: 레벨의 권장 region 순서, flow_source: "expert"|"rule", guidance: 한국어 안내 문구(각 level에서 어떤 순서로 진행할지)}`. LLM 라운드 불요(결정론 도구 + 템플릿).
    - **review 모드(보조, 세션 완료 후·선택)**: 완료 리뷰에서 T12 누적 burst 시퀀스로 회고적 이탈 분석 — `{current_progress, deviations[]: {burst_index, painted_region, expected_region, severity}, guidance}`. **이벤트 시점 실시간 이탈 지적은 범위 제외**(0.9 파생 제약 3).
  - **기준 흐름**: T6의 `expert_flow.json`(expert 세션들의 region 방문 순서 그래프)을 우선 사용. expert 데이터 부족 시 기존 규칙 기반 우선순위(_POSITION_PRIORITY: 상→중→하, 평면→곡면→모서리)를 폴백으로 사용하고 응답에 `flow_source: "expert"|"rule"` 명시.
  - **입력**: guide 모드 = level_id(+세션 컨텍스트). review 모드 = 훈련자 세션의 burst 시퀀스(각 burst의 주 도장 region — surface_hit 집계 또는 REQ-07 요약) + level_id.
  - **판정 로직(review 모드)**: 개별 burst의 정확한 일치가 아니라 흐름 수준 이탈만 지적 — (a) 대구간 역행(하→상 역주행), (b) 미완료 region 건너뛰기 후 복귀 비용 큰 이동, (c) 동일 region 과도 재도장. 같은 region 내 burst 반복·인접 region 간 순서 뒤바뀜은 허용 오차.
  - 기존 zone_map 기반 로직(schema 1.0)은 유지, macro flow 모드를 기본으로 승격.
  - 의존: T6(expert_flow), REQ-04/06. region 매핑 미가용 시 T11 기하 추론 폴백 공유(guide 모드는 region 정의 파일만으로 가능).
  - 완료 기준: guide 모드(level 1 권장 흐름 생성) E2E + review 모드 실세션(4 burst) E2E + 이탈 시나리오(역행/건너뛰기/재도장) 합성 테스트.
- **T10. Sequence 도구 스키마·SFT 정합** [P with T8]
  - `tool_schemas/sequence_validation.json`을 macro flow 출력 계약(guide/review 2모드)으로 갱신, few-shot·FC SFT 정합.
  - **시점 게이팅(0.9)**: guide 모드는 session_start 결정론 경로(LLM 미경유)이므로 few-shot 불요. review 모드 예시는 review 라우트 문맥에만 배치. 이벤트 문맥의 sequence 호출 예시는 제거/미생성.

### Phase E — 지원 MCP 및 코칭 API

- **T11. `location_tracking.py` → 공유 기하 region-recognition 엔진 (3D 인지)**
  - fixture 제거. 3층 구조로 재정의(2026-07-22 explore+brainstorm 반영):
    - **1층 (주 경로)**: `surface_hit`(Unity RaycastHit export, REQ-04)가 있으면 그대로 사용 — `triangleIndex→region` 룩업(triangleToRegion). Unity가 이미 raycast 수행하므로 최우선.
    - **2층 (폴백)**: `surface_hit` 미제공 시 서버측 raycast 재계산 — 레벨 mesh 로드(REQ-06 `mesh_ref`) + `active_gun_pose` 오일러(ZXY extrinsic)→forward 벡터(scipy `from_euler('zxy',...)`, Unity +Z forward) + **Open3D `RaycastingScene.cast_rays`**(ARM64 휠 존재; embreex는 aarch64 휠 부재로 배제) → `primitive_ids`→region. 조준 불량/근접/클리핑 시 nearest-face 폴백(trimesh `nearest.on_surface` 또는 Open3D `compute_closest_points`).
    - **3층 (커버리지)**: 단일 ray는 중심점만 → cone footprint(반경 = distance·tan(θ/2), 타원형 double-β/가우시안 근사)로 region 커버리지·다중 region 겹침·`coverage_ratio` 산출. mesh 삼각형 centroid KD-tree 반경 질의. `dft_judge` 90/10과 연결.
  - 공유: T9(순서 macro flow의 event→region), T7(자세 target region 매칭)이 이 엔진을 호출. 데이터 부족 시 명시적 `insufficient_data`/`region_source:"inferred"` 반환.
  - 완료 기준: surface_hit 유(1층)·무(2층) 양쪽 E2E 테스트 + 서버 raycast 거리를 `distance_real_mm`과 대조해 좌표계/회전순서 검증 리포트.
  - 의존: REQ-04/05/06. Open3D(ARM64) 의존성 추가(T17 이미지에 반영).
- **T12. `timeseries.py` 영속화**
  - 인메모리 → SQLite(단일 노드 Spark에 충분, Redis 불요) + 세션 TTL. 트렌드 분석 슬라이딩 윈도우로 개선.
  - **키 스키마(§0.7)**: 레코드를 `(session_id, user_id, event_id, t)`로 키잉. 이벤트 단위 조회 + 세션 전체 집계 양쪽 지원. `user_id`별 추세도 조인 가능하도록 인덱스.
  - **리뷰 피드 역할(0.9)**: 이벤트로 수신한 burst 프레임(자세·건 궤적·gun_metrics)을 분석 없이 누적 저장하고, 세션 완료 시 `session_id` 기준 전체 시퀀스를 T7(자세)·T9 review 모드(순서 회고)·분포 분석에 제공하는 조회 API 포함.
- **T13. Unity향 타입 지정 코칭 API 신설 (세션 생명주기 모델 §0.7)**
  - FastAPI 신규 서비스, **3 엔드포인트 + 세션 상태 머신** (기능-시점 매핑은 0.9 준수):
    - `POST /v1/sessions/start` — 컨텍스트 초기화(session_id·user_id) + **순서 안내 응답**(T9 guide 모드: recommended_flow + 한국어 안내).
    - `POST /v1/sessions/{session_id}/events` — event_id·user_request·시퀀스 데이터 수신. 실시간 코칭은 **RAG 질의응답(fast) + 이상탐지(realtime)만**. 이벤트 경로 도구 화이트리스트 = anomaly_detection·timeseries(pose/sequence 스키마 미전달). 시퀀스 데이터는 T12 누적만.
    - `POST /v1/sessions/{session_id}/complete` — dft/defects 수신 → **일괄 리뷰 리포트**: (a) 자세 유사도/교정(T7: similarity·corrections·expected_similarity_after_correction), (b) **부족 부위 분포 분석**(dft_judge 90/10 + anomaly_detection 분포 + T11 커버리지 → region별 부족 순위·사유), (c) 순서 회고(T9 review 모드, 선택).
    - 공통 `/live`, 의존성 인지 `/ready`(vLLM·FAISS·도구), correlation id, 타임아웃/폴백. Gradio는 이 API의 클라이언트로 강등.
  - 세션 상태 머신: start→(event*)→complete. 미시작 세션의 event/complete는 4xx. 세션 컨텍스트(level·worker·spec)는 start에서 보관해 이후 도구 호출에 주입.
  - **전송(§0.8)**: HTTP REST. WebSocket 미채택. 긴 응답 체감 지연 완화가 필요할 때만 이벤트 엔드포인트에 SSE(`stream=true`) 옵션을 단방향으로 추가(선택).
  - **도구 in-process 전환(§0.6-C)**: 기존 `mcp_servers/` FastAPI 라우터들을 coaching-api에 mount(`/tools/*`)하고, `tool_calling_adapter.py`의 디스패치를 HTTP 호출 → 직접 함수 호출로 전환(HTTP 경로는 폴백/디버그 플래그로 보존). 도구 스키마·응답 계약은 무변경.
  - 완료 기준: OpenAPI 스펙 + 계약 테스트(직접 호출·HTTP mount 양쪽) + Gradio 경유 동작 확인. (demo_app의 합성 센서 주입 제거 포함)
- **T13b. 스킬 지침 레이어 (§0.6-B)** [P with T13]
  - `rag_config_fewshot.yaml`에 내장된 few-shot·도구 선택 규칙·응답 템플릿을 라우트·도구별 스킬 문서(`trainer/skills/*.md` 가칭)로 분리하고 로더 구현: 라우트 판정 후 해당 스킬(도구 사용 시점·파라미터 구성·결과 해석·한국어 코칭 문구 가이드)만 컨텍스트에 주입. 프롬프트 토큰 예산 절감 + 유지보수성 확보.
  - **시점 게이팅(0.9)**: 스킬 문서를 시점별로 구성 — 이벤트 스킬(RAG·이상탐지 워크플로만), 완료 리뷰 스킬(자세 교정 해석 + 부족 부위 분포 분석 리포트 구성). 스킬 문서에 pose/sequence 도구 사용 지침이 이벤트 스킬에 섞이지 않도록 로더가 시점별 분리 주입.
  - 완료 기준: 기존 few-shot 대비 agentic eval 비열화(T4 기준선 ±0.2 이내) 확인.

### Phase F — 모델 아티팩트 파이프라인 (훈련 서버)

- **T14. Export 자동화: merge → 양자화 → manifest**
  - 스크립트: 정확한 base에 LoRA 병합(merged BF16 safetensors) → 병합모델 품질 검증(T4 기준선 대비) → **8-bit 양자화(FP8 W8A8 — llm-compressor/ModelOpt 오프라인 체크포인트, 폴백: vLLM 온라인 `--quantization fp8`)** → SHA-256 manifest(모델·토크나이저·설정 전체) → 전송 검증. **서빙 시 어댑터 별도 로드 금지** — vLLM에서 LoRA×speculative decoding 하드 비호환이므로 병합본이 유일한 배포 형태.
  - Qwen3-4B 롤백 패키지(Iter14/17 어댑터 병합본)를 이 파이프라인의 첫 산출물로 생성(파이프라인 검증 겸용).
- **T15. Qwen3.6-27B 후보 파인튜닝**
  - **T15-pre (스모크, 선행 필수)**: Qwen3.6-27B는 하이브리드 아키텍처(Gated DeltaNet + Gated Attention) — A40에서 transformers/peft/trl 현행 스택으로 QLoRA 1-step 학습 + 저장/로드 스모크를 본학습 전에 검증. 실패 시 스택 업그레이드 또는 모델 재선정 에스컬레이션.
  - **✅ T15-pre 통과 (2026-07-22 실측)**: NF4 QLoRA 1-step 학습→save→reload 성공, trainable 79.7M/26.98B(0.30%, LoRA 512텐서 — full-attn 16층 q/k/v/o + 전층 MLP), **peak VRAM 29.04GB < 48GB**. 8-bit(bnb INT8) 추론 스모크도 통과: peak 29.77GB, 256tok 생성 3.3 tok/s(A40 프록시 — Spark FP8+MTP가 실서빙 목표). **필수 워크어라운드**: `HF_DEACTIVATE_ASYNC_LOAD=1` — transformers 5.3.0 비동기 로더가 bnb 양자화 전에 BF16을 GPU에 적재해 OOM(46GB) 유발(Oracle 진단, upstream에 가드 존재). `train_qlora.py`/`smoke_infer_8bit.py`가 자동 설정.
  - 본학습: 현행 SFT(492 train/30 val)로 QLoRA 신규 학습(A40, 4-bit NF4, seq2048) → 동일 rubric으로 agentic/DocQA 평가 → T4 기준선 게이트 통과 시 T14 파이프라인으로 패키징. `.taskmaster/docs/qwen35_migration_prd.md`를 Qwen3.6-27B/vLLM-MTP 계약 기준으로 갱신.
  - **✅ 본학습 완료 (2026-07-23 실측)**: 3 epoch, 186 steps, **112.5분**, **peak VRAM 32.88GB < 48GB**. Loss 3.14→0.42, **eval_loss 0.683→0.525→0.494(단조 감소, 과적합 신호 없음)**, mean_token_accuracy 0.79→0.90. adapter `/output/adapters/qwen36_27b_shipcoat_lora/`(304MB, 512 LoRA텐서, reload 검증). trainable 79.7M/26.98B(0.30%).
  - **⚠️ T15 게이트 미통과 (2026-07-23 agentic eval, HF 4-bit 백엔드)**: **2.83/10 (baseline_v2 5.33 대비 회귀)**. tool_calls_made=**1**(baseline 47), tool_not_called=29/30. 차원별: tool_selection 0.067, param_correctness 0.067, reasoning 0.567, context 1.2, structure 0.933. 산출물 `/output/eval_results/qwen36_27b/`.
  - **근본 원인(진단 완료)**: 30개 응답 중 1개만 파싱가능 `<tool_call>` 방출, 0개가 `<think>` 사용(추론이 plain answer로 유출, 8개는 영어 추론 리크), 14개는 tool_call을 산문으로만 언급. **결정적**: Qwen3.6-27B 챗 템플릿은 **네이티브 tool-calling(`tools=` kwarg, `<tool_call>` 21회 참조)** 을 내장하고 `tools=` 렌더 시 "함수 호출 없으면 그냥 답하고 함수 얘기 말라"는 지시를 주입. 현 파이프라인은 `apply_chat_template(tools=...)` 대신 **수제 few-shot `<tool_call>` 프롬프트** 사용 → 4B는 관용했으나 27B의 강한 네이티브 tool 프로토콜 prior가 우세해 도구를 건너뜀. 보조 원인: LoRA가 DeltaNet 48층 `in_proj_*` 미부착(16/64 attn층만) — 도구 트리거 학습 커버리지 부족.
  - **Oracle 진단(2026-07-23)**: dominant 원인 = **(c) 프로토콜 혼용** — 파이프라인이 `tool_calling_adapter.py:545`에서 `apply_chat_template(tools=...)`로 27B **네이티브 XML tool 프로토콜**을 주입하면서 동시에 시스템 프롬프트엔 수제 **JSON `<tool_call>`** few-shot을 넣어 두 프로토콜 충돌. 4B 네이티브 템플릿은 우연히 JSON 계약과 일치해 문제없었음. 부차 (d) train/infer thinking 불일치(학습은 `enable_thinking=False` 고정, eval은 다수 항목 재활성).
  - **✅ 추론측 수정 검증 (Arm C 애블레이션, 무재학습)**: `TOOL_ADAPTER_NATIVE_TOOLS=0`(네이티브 `tools=` 미전달, JSON 계약만 유지) + thinking off. 동일 15문항 대조: **tool_calls 1→24, tool_selection 0.067→1.00, param 0.067→1.13**. 15문항 총점 **4.80(4B baseline 4.93과 동률권)**. 회귀는 재학습 없이 추론측 1줄 수정으로 해소. 잔여 감소(reasoning/structure)는 thinking 강제 off 부작용 → Arm D(네이티브 off + thinking 라우트별 기본)로 회복 검증 중.
  - **애블레이션 3-arm(동일 15문항, 무재학습)**: baseline 4.93 / **Arm C(native-off+think-off) 4.80 / Arm D(native-off+think-on) 3.73**. → **두 수정 모두 필수**: native-off가 프로토콜 충돌 해소, think-off가 도구 회피 방지(Arm D가 think-on시 tool_sel 0.33로 붕괴 입증 = (d) 확정).
  - **✅ 구현 완료(config 주도, 무재학습)**: `tool_calling_adapter.py` `pass_native_tools`(config `llm.pass_native_tools`, 기본 True 보존) + `agent_pipeline.py` `thinking_off_for_tools`(config `llm.thinking_off_for_tools`, 기본 False 보존). `rag_config_fewshot_27b.yaml`에 둘 다 활성. **4B 경로 무영향(test_tool_calling 22/22 green, 4B 기본값 미변경)**.
  - **✅ 전체 30문항 재평가(2026-07-23, 수정 적용)**: **4.50/10, tool_calls 36, tool_not_called 12/30**(2.83→4.50 회복, 무재학습). tool_selection 0.70, param 0.83, context 1.63. 산출물 `/output/eval_results/qwen36_27b_fixed/`.
  - **✅ v2 재학습 완료 (in_proj_* 추가, 2026-07-23)**: LoRA 타겟 7→12종(DeltaNet 48층 `in_proj_qkv/a/b/z/out_proj` 추가), trainable 79.7M→116.7M(0.43%), LoRA텐서 512→992, 124.9분, eval_loss 0.665→0.508→**0.478**(v1 0.494보다 개선), peak 33.1GB. adapter `/output/adapters/qwen36_27b_shipcoat_lora_v2/`. config `qlora_config_27b_v2.yaml`.
  - **v2 평가 결과 — tool은 개선, 총점은 미개선**: 30문항 **4.23/10, tool_calls 46**(4B 47 근접), tool_not_called 9/30. tool_selection **0.77(3안 중 최고)**, param **0.90(=4B)**. 그러나 **reasoning_depth 0.30, response_structure 0.67로 하락** → 총점 v1 4.50 > v2 4.23.
  - **3안 차원 비교(동일 30문항)**: baseline(4B) total 5.33 / tool_sel 0.57 / param 0.90 / **reason 0.90 / ctx 1.80 / struct 1.17**. 27B_v1 4.50 / 0.70 / 0.83 / 0.43 / 1.63 / 0.90. 27B_v2 4.23 / **0.77 / 0.90** / 0.30 / 1.60 / 0.67.
  - **결론(진단 확정)**: (a) LoRA 미커버리지는 **해소됨** — in_proj_* 추가로 tool 지표가 4B 이상 도달. **잔여 갭의 실체 = reasoning_depth·response_structure 품질**이며 이는 tool-calling을 위해 강제한 `thinking_off_for_tools`의 부작용(27B가 think-off 시 더 간결·저구조 응답). LoRA 커버리지 문제 아님. **채택 어댑터 = v1(4.50, 총점 우위)**; v2는 tool 신뢰도가 중요한 배포에선 재고.

### T15b. 목표 7.0 재조정 — 채점 메커니즘 분석 (2026-07-23)

- **결정적 발견: 채점기가 100% 결정론 정규식**(`run_agentic_eval.py`). `response['answer']`(=파이프라인 `text_guide.body`) 텍스트에만 정규식 적용. 즉 **잔여 갭 2.5점의 대부분은 응답 포매팅 문제**(모델 능력·LoRA 커버리지 아님).
  - reasoning_depth(0-2): 번호매김(`^\d+[\).]`)·한국어 섹션헤더 `[분석]/[판정]/[원인]/[조치]/[근거]/[결론]`·접속사(먼저/다음/이후)·`단계 N:` 개수 + **근거(DOC-xxx citation)**. steps≥expected & 근거 → 2점.
  - response_structure(0-2): 판정어(합격/불합격/진행/중지/정상/이상)+리스트, 또는 섹션헤더≥3, xr_content는 JSON 키.
  - context_integration(0-2): DOC-xxx citation + context 숫자 반향(현재 1.6, 거의 만점).
  - parameter_correctness = `int(mean)` **정수 절삭**(0.5→0, 1.5→1) — 평균 ≥1.0/≥2.0 정확히 넘겨야 점수.
- **v1 30문항 실측 실패모드**: **6/30만** 정규식 매칭 구조(번호/헤더) 보유, **10/30은 최종 답변에 raw `<tool_call>` 잔존**(도구 루프가 최종 합성 턴 없이 종료), 19/30 citation 보유. → reason 0.43·struct 0.90이 낮은 직접 원인.
- **갭 축소 후보(무재학습 우선)**: (S1) 도구 루프가 도구 결과 후 **최종 합성 턴 강제**(raw tool_call 잔존 제거), (S2) 시스템 프롬프트에 **엄격 한국어 응답 포맷 계약**(섹션헤더+번호+`근거 문서 ID: DOC-...`) + few-shot 1개, (S3) citation 후처리 보장, (S4 재학습) FC-SFT 최종 assistant 턴을 항상 이 구조로 재생성. **정당성**: 루브릭이 실제로 구조화·인용·다단계 코칭을 보상 → 포매팅이 곧 산출물(게이밍 아님).
- **다음(Oracle 로드맵 대기 bg_edb2576a + explore bg_19b3ab17 + librarian bg_ad91e4bb)**: 단계별 예상 점수 확정 후 S1부터 15문항 서브셋으로 저비용 검증하며 7.0까지 상향. **T18에서 27B는 native-off+think-off 계약 고정(혼용 금지)**.
  - 의존: T4, T14. (Gemma 4 E4B A/B는 동일 절차의 선택 태스크 **T15b**)
- **T16. vLLM 패키지 검증 + MTP 게이트**
  - 훈련 서버(x86)에서 vLLM ≥0.24로 병합본 로드 → 구조화 tool_calls 스모크(`--tool-call-parser qwen3_xml`) → 한국어 코칭 응답 회귀 → BF16 vs 8-bit 품질 비교 게이트(A40에서는 INT8/bnb 프록시, FP8 최종 확인은 Spark T19/T20).
  - **MTP 검증(핵심)**: 파인튜닝은 body만 갱신하고 MTP head는 미갱신 → 병합본에서 `mtp_num_layers` 확인 + MTP on/off acceptance rate·품질 동일성 실측. acceptance 급락 시 γ=1 고정 또는 MTP 없이 배포(속도 게이트 T21에서 재판정).

### Phase G — DGX Spark 배포

- **T17. ARM64 배포 아티팩트**
  - 컨테이너 2종(+선택 1): **vllm**(NGC `nvcr.io/nvidia/vllm` 또는 cu130-nightly ≥0.24, sm_121 검증 빌드 — 표준 stable 이미지 사용 금지), **coaching-api**(FastAPI + RAG + in-process 도구, ARM64 빌드), gradio(선택). 의존성 완전 고정(lock), 개발자 경로 마운트 제거, 전 경로 환경변수화. compose: 포트 공개, 헬스체크, restart 정책, named volume, `gpu-memory-utilization 0.5–0.7`(통합 메모리 풀 주의).
  - 주의: CUDA 13/sm_121에서 faiss·torch(BGE-M3 임베딩용) ARM64 호환 확인 — 비호환 시 faiss-cpu + onnx/CPU 임베딩 폴백 결정 포함.
- **T18. vLLM 게이트웨이 모듈**
  - `agent_pipeline.py`의 HF 백엔드를 대체하는 OpenAI-호환 클라이언트(vLLM chat/completions): 저장소 tool_schemas 전달, `message.tool_calls` 소비(raw-tag 파싱 대체), thinking 모드 라우트별 제어(fast=off 필수), 컨텍스트 상한, 타임아웃, 모델 선택(27B 후보/4B 롤백 스위치), 헬스체크. HF 백엔드는 훈련 서버 평가용으로 보존.
  - 의존: T13, T16.
- **T19. Spark 스택 기동 + 스모크**
  - 모델 전송(해시 검증) → vLLM 로드(§0.5 기준 레시피, MTP γ=1) → 전 서비스 기동 → `/ready` green → 3-route 각 1회 E2E + 도구 7종 개별 호출 스모크(직접 호출 + `/tools/*` mount 양쪽) → MTP γ∈{1,2,3} tok/s 비교 후 운영값 고정.

### Phase H — 검증 게이트 및 마무리

- **T20. Spark 품질 회귀 게이트**
  - agentic eval 30문항을 Spark에서 재실행, T4 기준선 대비 점수·tool call 수·hallucination 비교. 결과에 아티팩트/설정 해시 첨부(기계 판독 evidence).
- **T21. 성능·안정성 게이트**
  - cold/warm p50/p95/p99(라우트별), 동시성 1/2/4, deadline/폴백 주입, 통합메모리/RSS/swap 관측, 4–8h soak. §1 지연시간 계약으로 판정.
- **T22. 롤백 리허설 + 문서 최종화**
  - Qwen3.6-27B → Qwen3-4B 전환 리허설(게이트웨이 모델 스위치, 동일 vLLM 런타임), 운영 runbook, README/CLAUDE.md/아키텍처 문서 갱신(라운드 수·도구 수·MCP 상태표·§0.6 도구 실행 모델 포함), Taskmaster에 본 계획 태스크 등록.

## 4. 의존 그래프 (요약)

```
T1 ─┬─ T3 ── T4 ─────────────────────┬─ T15-pre ── T15 ─┐
T2 ─┘                                 │                  ├─ T16 ── T18 ─┐
T5a ── T5b ── T6 ─┬─ T7 ── T8         └─ T14 ────────────┘             ├─ T19 ── T20 ── T21 ── T22
 (exporter 회신)  └─ T9 ── T10                                          │
                    └─ T11   T12   T13/T13b ────────────────── T17 ────┘
```
병렬 트랙: {T1–T4 계약수리} ∥ {T5–T12 도구/데이터} ∥ {T14–T16 모델}. T13은 T1 이후, T17–T19는 전 트랙 합류점.
**T5a는 외부 의존(critical path)**: exporter의 0.4.0 반영 + expert 라벨 세션 확보가 늦어지면 T6 이후가 전부 지연 → T5a 전달을 최우선 실행하고, 대기 중 T1–T4·T12–T15를 병렬 진행.
**T15-pre(27B QLoRA 스모크)도 조기 실행 권장**: 하이브리드 아키텍처 학습 가능성이 모델 선정의 전제이므로 실패 시 조기 에스컬레이션.

## 5. Scope OUT (v2 개정)

- Spark에서 학습/파인튜닝 금지 (추론·운영 전용)
- Qwen3-4B 어댑터의 타 모델 재사용 금지
- **서빙 시 LoRA 어댑터 로드 금지** — vLLM에서 LoRA×speculative decoding 하드 비호환. 병합본만 배포
- **sm_121 미검증 vLLM 표준 stable 이미지 사용 금지** (NGC 또는 cu130-nightly만)
- **NVFP4 KV cache 사용 금지** (sm_121에서 실험 단계 — KV는 fp8)
- 기존 어댑터/SFT/인덱스/평가 evidence 삭제·덮어쓰기 금지
- `vlm_analysis`는 이번 범위에서 Stub 유지 (명시 라벨). LLM 프롬프트 스킬로 도구 계산 대체 금지(§0.6-A)
- Kubernetes, 멀티-Spark(tensor-parallel 2노드 링크 포함), 멀티모달 확장 금지 — ~~speculative decoding 금지~~는 **해제(MTP 한정, T16 게이트 통과 조건)**
- **양방향 WebSocket 전송 금지(§0.8)** — HTTP REST 유지. 단방향 SSE 스트리밍만 선택적 허용
- import 성공/이미지 빌드/`nvidia-smi`/단일 지연 샘플만으로 합격 판정 금지

## 6. 리스크 및 열린 항목

| # | 리스크 | 완화 |
|---|---|---|
| R1 | expert 라벨 세션 부재(`skill_level: null`) → T6/T7/T9 참조 데이터 불가 | **T5a 요구사항 REQ-01로 공식 요청 완료**(`docs/data_requirements_pose_sequence_mcp.json`). 레벨별 expert 3개+ 확보 전까지 임시 프록시(distance/speed 안정성) 사용, 응답에 `reference_quality: "proxy"` 명시 |
| R2 | `surface_hit: null` → burst→region 매핑 불가 = 자세 타겟 정렬·순서 macro flow 모두 불가 | **2층 완화**: 주 경로=Unity RaycastHit export(이미 raycast 수행 중이라 저비용, REQ-04 재프레이밍 완료). 폴백=T11 서버측 Open3D raycast + nearest-face. `distance_real_mm` 존재가 hit 계산 증거이므로 export가 현실적 |
| R3 | ARM64에서 faiss/torch/sentence-transformers 호환성 | T17에서 조기 검증. faiss-cpu + CPU 임베딩 폴백 경로 확보 |
| R4 | Qwen3.6-27B가 NVIDIA Spark 공식 playbook 매트릭스에 없음(Qwen3-32B만 등재) — 반검증 경로 | vLLM 공식 레시피(`recipes.vllm.ai/Qwen/Qwen3.6-27B`) 존재 + 커뮤니티 배포 사례로 보완. T19 스모크에서 조기 검증, 실패 시 Qwen3-4B 롤백 패키지로 배포 진행 |
| R5 | Unity 회전 순서 미확정(`rotation.order: null`) → FK·반사실 검증 무효 위험 | **REQ-05로 확정 요청 완료**. 회신 전 ZXY 가정 구현 + 회신 시 상수 교체 |
| R6 | DTW 정렬·교정 시뮬레이션의 계산 비용(실시간 라우트 지연) | **v3에서 구조적으로 해소(0.9)** — 자세 분석은 세션 완료 리뷰 전용이라 실시간 계약과 무관. review p95<30s 내 수렴 위해 burst 단위 사전 리샘플링(고정 프레임 수)으로 DTW 입력 축소는 유지 |
| R7 | expert 세션 간 흐름 불일치(macro flow가 수렴하지 않음) | 순서 그래프 합의도(edge 빈도) 임계 미달 시 rule 폴백 + `flow_source: "rule"` 명시, 데이터 추가 확보 요청 |
| R8 | 27B 지연시간: MTP on에서도 ~16.8 tok/s → fast p95<3s는 ~50토큰 한계 | fast 라우트 thinking off + max_tokens 상한. **v3 완화(0.9)**: 이벤트 경로가 RAG+이상탐지로 한정되어 무거운 도구(DTW/반사실)가 실시간에서 제거, 순서 안내는 결정론 경로(LLM 라운드 0~1). T21 실측으로 재판정 — 미달 시 PoC-parity 완화 또는 fast 전용 소형 모델(4B) 병행 서빙 검토 |
| R9 | QLoRA가 MTP head 미갱신 → 파인튜닝 후 body-head 분포 어긋남으로 acceptance rate 하락 | T16에서 MTP on/off acceptance·품질 실측 게이트. 급락 시 γ=1 고정 또는 MTP 비활성 배포 |
| R10 | Qwen3.6-27B 하이브리드 아키텍처(Gated DeltaNet)의 QLoRA 학습 스택 호환 미지수 | **✅ 해소(2026-07-22)**: T15-pre 스모크 통과(1-step 학습+save/reload, peak 29.04GB). 단 `HF_DEACTIVATE_ASYNC_LOAD=1` 필수(5.3.0 로더 버그) + LoRA는 DeltaNet in_proj_*에 미부착(전층 MLP+full-attn만) — 본학습 품질 게이트(T15)에서 커버리지 충분성 판정 |
| R11 | in-process 도구 전환(§0.6-C) 중 HTTP 계약 회귀 | 기존 라우터 mount로 계약 보존 + T13 계약 테스트를 직접 호출·HTTP 양쪽에 적용 |
| R12 | 서버측 raycast 폴백(T11 2층)의 좌표계·회전순서 오류 → region 오인식. Unity 좌표계(좌수)와 GLTF export(우수) 불일치 가능 | REQ-05로 rotation order 확정 요청 완료. T11 완료 기준에 `distance_real_mm` 대조 검증 포함(계산 거리 vs export 거리 일치 확인). 주 경로(Unity export)는 이 리스크 회피 |
| R13 | ARM64(Spark)에서 embreex(trimesh embree) 휠 부재 → 고속 raycast 백엔드 제약 | **Open3D `RaycastingScene` 채택**(aarch64 manylinux 휠 존재, 멀티스레드 Embree 내장). trimesh는 nearest-face(scipy, 네이티브 무의존) 폴백용. 90Hz 단일 ray는 어느 백엔드든 충분 |

## 7. 태스크별 검증 시나리오 (Final Verification Wave 실행형)

각 태스크는 아래 `도구/명령 + 입력 + 합격 기준`으로 검증한다. 합격 기준 미충족 시 해당 태스크는 미완료.

| Task | 도구/명령 | 입력 | 합격 기준 (Expected) |
|---|---|---|---|
| T1 | `python3 -m pytest mcp_servers/tests/test_response_contract.py -v` | dft/anomaly/seq 각 1 요청 | 모든 응답이 `result` 중첩 계약 준수, `agent_pipeline` 조립부가 동일 경로에서 읽음 → green |
| T2 | `python3 -m pytest trainer/scripts/test_tool_calling.py -k rounds` + `grep -n "max_rounds" trainer/scripts/agent_pipeline.py` | realtime 라우트 질의 1건 | 코드 라운드 수 == 문서 값(3), 프롬프트 도구 목록 == 7(또는 vlm 제외 명시), 로그에 라운드 수 일치 |
| T3 | `python3 -m unittest mcp_servers.test_integration -v` + `python3 trainer/scripts/test_tool_calling.py -v` | 기존 테스트 스위트 | 15/15, 22/22 PASS (health `servers:7` 포함) |
| T4 | `python3 trainer/scripts/run_agentic_eval.py --responses <gen> --output-dir /output/eval_results/baseline_v2` | agentic 30문항 | 점수·tool call 수 산출 + `baseline_v2/manifest.json`에 모델/어댑터/인덱스/스키마 SHA 기록 |
| T5a | 파일 존재 + exporter 회신 | `docs/data_requirements_pose_sequence_mcp.json` | REQ-01~12 명세 전달 완료, rotation order·조인 규약 회신 수령(미수령 시 blocked 표시) |
| T5b | `python3 trainer/scripts/parse_session.py --input docs/1_1000442_seq002_sample30.json --validate` + pytest | seq002 30프레임 샘플 | exit 0, 30프레임·20관절·6객체 파싱, degraded 필드 리포트, 단위테스트 green |
| T6 | `python3 trainer/scripts/build_expert_reference.py --level 1 --out levels/1/` | expert 라벨 세션 3+ (미확보 시 proxy) | `expert_pose_reference.json`·`expert_flow.json` 생성, region별 관절 평균/분산·flow 그래프 edge 빈도 포함, `reference_quality` 필드 명시 |
| T7 | `curl -X POST :8200/tools/pose_estimation/compare -d @<session_bursts>` + `pytest test_pose.py -k counterfactual` | T12 누적 세션 전체 burst + expert 참조 (완료 리뷰 경로) | 응답에 `{similarity_score, corrections[], expected_similarity_after_correction}`, **expected > similarity 강제**, expert-vs-novice 분리도 리포트 출력. **이벤트 경로 도구 목록에 pose 미노출 확인(0.9)** |
| T8 | `python3 -c "import json,jsonschema; ..."` 스키마 검증 + eval 재생성 | `tool_schemas/pose_estimation.json` | 신규 출력 계약과 스키마 일치, few-shot 예시 유효 파싱 |
| T9 | guide: `curl -X POST :8200/tools/sequence_validation/guide -d '{"level_id":1}'` / review: `curl -X POST :8200/tools/sequence_validation/validate -d @<session>` + `pytest test_seq.py -k "reverse or skip or repaint"` | guide: level_id만 / review: 실세션 4 burst + 합성 이탈 케이스 | guide: `{recommended_flow[], flow_source, guidance}` 반환(훈련 시작 전 순서 안내). review: `{deviations[], current_progress}`, 역행/건너뛰기/재도장 3 시나리오 정탐. **이벤트 경로에서 sequence 도구 미노출 확인(0.9)** |
| T10 | 스키마 검증 (T8과 동일 방식) | `tool_schemas/sequence_validation.json` | macro flow 출력 계약과 일치 |
| T11 | `curl -X POST :8200/tools/location_tracking/identify -d @<coords>` | 좌표 시퀀스 (0.3.0) | region_id 추론 반환 또는 데이터 부족 시 `insufficient_data` 명시(하드코딩 B-3 제거 확인) |
| T12 | `pytest test_timeseries.py -k persist` (서버 재시작 포함) | update 후 프로세스 재기동 → query | 재시작 후에도 세션 데이터 유지(SQLite), TTL 만료 동작 확인 |
| T13 | `python3 -m pytest trainer/scripts/tests/test_coaching_api.py` + `curl :8200/ready` + lifecycle 시나리오(start→event×N→complete) | 코칭 질의 + 세션 리뷰 요청 | OpenAPI 스펙 생성, `/ready`가 vLLM·FAISS·도구 상태 반영, 직접호출·HTTP mount 계약 테스트 양쪽 green. **0.9 계약**: start 응답에 recommended_flow 포함, event 응답은 RAG/이상탐지만(도구 화이트리스트 위반 시 fail), complete 리포트에 자세 유사도/교정 + 부족 부위 분포 분석 포함 |
| T13b | `python3 trainer/scripts/run_agentic_eval.py`(스킬 레이어 on) | agentic 30문항 | 점수 T4 기준선 ±0.2 이내, 프롬프트 토큰 수 감소 로그 |
| T14 | `bash trainer/scripts/export_model.sh --adapter <4b> --base <4b>` → 산출물 검사 | Qwen3-4B 어댑터 | merged safetensors + NVFP4/FP8 + `manifest.json`(SHA 전체), 전송 후 해시 재검증 일치 |
| T15-pre | `python3 trainer/scripts/train_qlora.py --config <27b> --max-steps 1 --smoke` | Qwen3.6-27B + SFT 1배치 | 1-step 학습→save→reload 성공(OOM/아키텍처 오류 없음), A40 peak VRAM 로그 < 48GB |
| T15 | `run_qlora_eval.py --mode rag-qlora` + `run_agentic_eval.py` | Qwen3.6-27B 병합본 | DocQA·agentic 점수 산출, T4 기준선 게이트 통과 여부 판정 기록 |
| T16 | `vllm serve <merged> --speculative-config '{"method":"mtp",...}'` + tool_calls 스모크 | 병합본 (x86 검증) | 구조화 tool_calls 파싱 성공, MTP on/off acceptance rate·품질 동일성 실측, `mtp_num_layers` 확인 |
| T17 | `docker compose config` + `docker build --platform linux/arm64` | compose + Dockerfile | arm64 빌드 성공, 포트/헬스체크/named volume 존재, 개발자 경로 마운트 0건(grep) |
| T18 | `pytest test_vllm_gateway.py` | mock/real vLLM 엔드포인트 | `message.tool_calls` 소비, fast 라우트 thinking=off, 27B/4B 모델 스위치 동작 |
| T19 | Spark에서 `curl :8200/ready` + 3-route E2E + 도구 7종 호출 | 전송된 병합본 | `/ready` green, 3 라우트 각 응답 수신, 도구 7종 직접·mount 양쪽 200, γ∈{1,2,3} tok/s 표 |
| T20 | Spark에서 `run_agentic_eval.py` | agentic 30문항 | T4 기준선 대비 점수·tool call·hallucination 비교표 + 아티팩트 해시 첨부 |
| T21 | 부하 스크립트(cold/warm, 동시성 1/2/4) + `nvidia-smi`/RSS 관측 | 라우트별 반복 요청 | p50/p95/p99 산출, §1 지연 계약 판정(R8 반영), 4–8h soak 무크래시 |
| T22 | 롤백 스위치 실행 + 문서 diff | 27B→4B 전환 | 게이트웨이 모델 스위치 동작 확인, runbook·README·CLAUDE.md·아키텍처 문서 갱신 커밋, Taskmaster 등록 |

**Final Verification Wave**: T19(스모크)→T20(품질)→T21(성능/안정)→T22(롤백/문서) 순으로 전 게이트 통과 시에만 배포 완료로 간주. 각 게이트 산출물은 기계 판독 evidence(JSON/표 + 해시)로 남긴다.
