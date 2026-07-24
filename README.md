# 선박 도장 AI Co-Pilot PoC

## 프로젝트 개요

선박 도장 품질 관리를 위한 AI 코파일럿 PoC(Proof of Concept)이다. XR 훈련 시뮬레이터(Unity)와 연동하여 작업자에게 실시간 코칭 피드백을 제공하는 것을 목표로 한다.

**핵심 기술 스택**

- **베이스 모델**: Qwen3-4B (Apache 2.0)
- **파인튜닝**: QLoRA (rank 16, 4-bit NF4, peft + trl SFTTrainer)
- **지식 검색**: BGE-M3 임베딩 + FAISS HNSW 인덱스 + BGE-Reranker-v2-m3
- **에이전트**: ReAct 오케스트레이터 + 7개 MCP 서버 / 7개 모델-대상 도구 스키마 (FastAPI)
- **데모 UI**: Gradio 6.x 웹 인터페이스

**최종 성과 (Iter14 기준)**

| 지표 | 결과 |
|---|---|
| Agentic 평가 점수 | 4.97/10 (목표 4.0 달성) |
| RAG+QLoRA DocQA 점수 | 1.368/2.0 |
| Hallucination 비율 | 2.0% |
| Tool Calling 작동 | 86회/30문항 |
| RTX 4080 VRAM 사용 | ~8.5 GB (학습), ~4 GB (추론) |

---

## 아키텍처

### 전체 데이터 흐름

```
[문서 수집] → [파싱] → [정규화] → [중복 제거] → [토큰 청킹]
     → [BGE-M3 임베딩] → [FAISS HNSW 인덱스]
                                    ↓
[시뮬레이터 요청] → [3-Way Semantic Router]
                         ↓              ↓              ↓
                  training_realtime  training_review   fast
                  (max_rounds=2)     (max_rounds=5)  (no tools)
                         ↓
                  [RAG 검색 + Rerank]
                         ↓
                  [Qwen3-4B + QLoRA Adapter]
                         ↓
                  [Tool Calling Adapter]
                    ↙  ↙  ↙  ↙  ↙  ↘
              dft  anom seq  pose loc  ts
              judge det  val  est  trk  eries
                         ↓
                  [구조화 JSON 응답] → [시뮬레이터]
```

### 3-Way Semantic Router

| 라우트 | 판단 기준 | max_rounds | 응답 목표 |
|---|---|---|---|
| **training_realtime** | 순서, 이상탐지, 합부 판정 키워드 | 2 | < 60초 |
| **training_review** | 자세, 추세분석, 코칭, 리포트 키워드 | 5 | 제약 없음 |
| **fast** | 단순 지식 질문 (용어, 규정) | 1 | < 10초 |

### MCP 서버/도구 현황

- 마운트된 MCP 서버: 7개 (`/health` 기준, `vlm_analysis` 포함)
- 모델-대상 도구 스키마: 7개 (`trainer/configs/tool_schemas/` 기준)

| 도구명 | 엔드포인트 | 기능 | 구현 상태 |
|---|---|---|---|
| dft_judge | `/dft_judge/evaluate` | SSPC-PA2 90/10 룰 DFT 합부 판정 | 실제 로직 |
| anomaly_detection | `/anomaly_detection/detect` | 두께 편차 기반 결함 유형 분류 및 원인 추론 | 실제 로직 |
| sequence_validation | `/sequence_validation/validate` | 블록/구역 우선순위 기반 다음 도장 구역 추천 | 실제 로직 |
| pose_estimation | `/pose_estimation/compare` | 저장된 숙련자 기준 자세와 현재 자세 비교/추천 | Stub (하드코딩 기준 자세) |
| location_tracking | `/location_tracking/identify` | 3D 좌표 기반 Zone ID 및 커버리지 매핑 | Stub (fixture 응답) |
| timeseries | `/timeseries/{action}` | 작업 속도/거리/각도 트렌드 분석 | Partial (in-memory + 실제 추세 계산) |
| vlm_analysis | `/vlm/analyze` | 히트맵/스크린샷 기반 VLM 분석 | Stub (빈 결과) |

MCP 서버 기본 URL: `http://localhost:8100`

---

## 빠른 시작 (Quick Start)

### 1. Docker 환경 구성

```bash
# 이미지 빌드
docker build -t ship-paint-trainer:0.1 /workspace/trainer/

# 컨테이너 실행 (docker-compose 사용)
docker-compose -f /workspace/deploy/docker-compose.yml up -d

# 컨테이너 접속
docker exec -it ship-paint-trainer bash
```

`docker-compose.yml`은 다음 볼륨을 마운트한다:

| 호스트 경로 | 컨테이너 경로 | 용도 |
|---|---|---|
| `/data/ship-paint-ai/models` | `/models` | 베이스 모델 |
| `/data/ship-paint-ai/data` | `/data` | 학습/평가 데이터 |
| `/data/ship-paint-ai/output` | `/output` | 어댑터, 평가 결과 |
| `/data/ship-paint-ai/hf-cache` | `/hf-cache` | HuggingFace 캐시 |
| `/data/ship-paint-ai/logs` | `/logs` | 학습 로그 |
| `/home/mjkim/ship-paint-ai` | `/workspace` | 코드 |

### 2. 모델 다운로드

```bash
# Qwen3-4B 베이스 모델 (약 7.6 GB)
huggingface-cli download Qwen/Qwen3-4B --local-dir /models/Qwen3-4B

# BGE-M3 임베딩 모델 (자동 다운로드 — 최초 실행 시)
# BAAI/bge-m3, BAAI/bge-reranker-v2-m3 는 build_index.py 실행 시 자동 캐시됨
```

### 3. 데모 실행 (GPU 불필요)

```bash
# DEMO_MODE: 더미 데이터 스트리밍, 모델 로딩 없음
DEMO_MODE=true python3 /workspace/trainer/scripts/demo_app.py
# 브라우저에서 http://localhost:7860 접속
```

### 4. 실제 파이프라인 실행 (GPU 필요)

```bash
# MCP 서버 먼저 기동
uvicorn mcp_servers.main:app --host 0.0.0.0 --port 8100 &

# 실제 모델 + 어댑터 로딩
DEMO_MODE=false python3 /workspace/trainer/scripts/demo_app.py
```

---

## React 웹 · Electron 데스크톱 데모 (1차)

기존 Gradio 데모(`trainer/scripts/demo_app.py`)는 그대로 유지된다. 아래는 새로 추가된 세 인터페이스의 실행 명령이다. 1차 구현은 3D 뷰어 없이 채팅과 구조화된 도장 순서 목록만 제공한다. 3D 시각화는 `zone_id ↔ mesh/coordinate` 매핑이 확보되는 시점에 후속 spec으로 이관된다.

### 1) FastAPI 데모 백엔드

```bash
python3 -m trainer.demo_api.app --host 127.0.0.1 --port 8120
```

노출 라우트:
- `GET  /api/health`
- `GET  /api/sequence/{level_id}` — 초기 도장 순서 스냅샷
- `POST /api/chat`               — SSE 스트림 (session/status/tool_result/sequence_update/message_delta/complete/error)

CORS 허용 오리진은 환경변수 `DEMO_UI_ORIGINS` (콤마 구분, 기본값 `http://127.0.0.1:5173,http://localhost:5173`).

### 2) React 웹 데모

```bash
cd demo-ui && npm install
cd demo-ui && npm run dev:web -- --port 5173
```

`VITE_API_BASE_URL`을 통해 백엔드 위치를 지정한다 (기본은 same-origin). Vite 개발 서버 사용 시:
```bash
VITE_API_BASE_URL=http://127.0.0.1:8120 npm run dev:web -- --port 5173
```

프로덕션 정적 빌드:
```bash
cd demo-ui && npm run build:web
```
빌드 산출물은 `demo-ui/dist/`.

### 3) Electron 데스크톱 데모

```bash
cd demo-ui && npm run start:electron
```

내부적으로 `build:web` + `build:electron` 후 Electron이 `dist/index.html`을 로드한다. 개발 시 라이브 리로드:
```bash
cd demo-ui && npm run dev:electron
```

Electron 셸은 `contextIsolation: true`, `nodeIntegration: false`, `sandbox: true`로 고정되며 preload는 `window.desktop.{ platform, electronVersion }`만 노출한다.

### 4) 테스트

```bash
# Python API
python3 -m pytest trainer/demo_api/tests -q

# React (Vitest)
cd demo-ui && npm test -- --run

# 브라우저 E2E (Playwright)
cd demo-ui && npm run test:e2e
```

### 3D 뷰어 관련 참고

`SP0035_rig.glb` 같은 3D 자산은 Windows 로컬 경로에서만 접근되며 이 저장소의 컨테이너에는 마운트되어 있지 않다. 자산 매핑(`zone_id` → 메시 노드 또는 월드 좌표)이 정리되면 별도 spec에서 3D 뷰어 계층을 추가한다.

## 하드웨어 요구사항

| 구분 | 사양 |
|---|---|
| **추론 최소** | RTX 4080 16GB VRAM |
| **학습 권장** | NVIDIA A40 48GB VRAM |
| **추론 VRAM** | ~4 GB (4-bit 양자화, Qwen3-4B) |
| **학습 VRAM** | ~8.5 GB peak (Iter14 기준) |
| **RAM** | 32 GB 이상 권장 |
| **저장 공간** | 모델 8 GB + 데이터 2 GB + 인덱스 0.5 GB |
| **CUDA** | 12.4 이상 |

**VRAM 예산 (RTX 4080 16GB 기준)**

| 구성 요소 | VRAM |
|---|---|
| LLM (Qwen3-4B, 4-bit NF4) + QLoRA Adapter | ~4.5 GB |
| BGE-M3 임베딩 + FAISS HNSW 인덱스 | ~2.0 GB |
| MCP 서버 (FastAPI) | ~1.5 GB |
| XR 시뮬레이터 여유분 | ~8.0 GB |

---

## 디렉토리 구조

```
/workspace/                    # 코드 루트
  trainer/
    scripts/                   # 모든 실행 스크립트 (25개)
      ingest_docs.py           # PDF/HTML/TXT 파싱
      clean_docs.py            # 텍스트 정규화
      dedup_docs.py            # 중복 제거
      chunk_docs.py            # 시맨틱 청킹
      build_index.py           # BGE-M3 임베딩 + FAISS 인덱스 빌더
      search_index.py          # FAISS 검색 + CrossEncoder 리랭커
      generate_sft.py          # DocQA SFT 데이터 생성
      generate_fc_sft.py       # FC-aware SFT 데이터 생성
      review_sft.py            # SFT 품질 검수
      train_qlora.py           # QLoRA 학습
      qa_pipeline.py           # RAG QA 파이프라인
      run_rag_eval.py          # RAG-Only 평가
      run_qlora_eval.py        # QLoRA/RAG+QLoRA 평가
      agent_pipeline.py        # AgentPipeline + SemanticRouter
      tool_calling_adapter.py  # FC 어댑터 + MCP 디스패처
      generate_agentic_responses.py  # Agentic 응답 생성
      run_agentic_eval.py      # Agentic 평가 채점
      session_coaching.py      # 세션 코칭 응답 시스템
      demo_app.py              # Gradio 웹 데모
      demo_agent.py            # E2E 시나리오 시연
      eval_metrics.py          # ROUGE-L, BERTScore, Hallucination 탐지
      response_formatter.py    # JSON 응답 포맷팅
      generate_report_charts.py # 시각화 차트 생성
    configs/                   # 설정 파일 (9개)
    Dockerfile                 # pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime 기반
  mcp_servers/                 # MCP 도구 서버 (FastAPI)
    main.py                    # 통합 라우터 + /health 엔드포인트
    dft_judge.py               # DFT 합부 판정 (실제 로직)
    anomaly_detection.py       # 이상 탐지 (실제 로직)
    sequence_validation.py     # 순서 검증 (실제 우선순위 로직)
    pose_estimation.py         # 자세 분석 (Stub: 하드코딩 기준 자세)
    timeseries.py              # 시계열 분석 (Partial: in-memory 추세 계산)
    location_tracking.py       # 위치 추적 (Stub: fixture 응답)
    vlm_analysis.py            # VLM 분석 (Stub: 빈 결과)
    common.py                  # MCPResponse, MCPException 공통 모듈
  deploy/
    docker-compose.yml         # Docker 서비스 정의
  FINAL_INTEGRATED_PRD.md      # 통합 기술 명세서
  AgenticCopilot_Architecture.md  # 아키텍처 설계서
  PATCH_NOTE.md                # 이터레이션 히스토리 (Iter0~Iter14)

/data/                         # 데이터 (Docker 볼륨 마운트)
  raw/
    pdf/                       # 원본 PDF 문서 (110건)
    html/                      # 원본 HTML 문서 (69건)
  parsed/                      # ingest_docs.py 출력 (179건)
  cleaned/                     # clean_docs.py 출력 (179건)
  deduplicated/                # dedup_docs.py 출력 (147건)
  chunked/
    chunks.jsonl               # 토큰 기반 청크 (10,829건)
  sft/
    sft_train.jsonl            # SFT 학습 데이터 (567건, Iter14)
    sft_val.jsonl              # SFT 검증 데이터 (30건)
    sft_train_docqa_only.jsonl # DocQA 전용 백업 (380건)
  eval/
    eval_set.jsonl             # DocQA 평가셋 (50문항)
    eval_set_agentic.jsonl     # Agentic 평가셋 (30문항)
    eval_answer_key.json       # 참조 답변
  manifest.csv                 # 179건 문서 카탈로그

/output/                       # 산출물 (Docker 볼륨 마운트)
  adapters/
    qwen3_4b_shipcoat_lora/    # 현재 활성 QLoRA 어댑터 (Iter14)
  checkpoints/                 # 학습 체크포인트 (epoch별)
  indexes/
    faiss.index                # FAISS HNSW32 인덱스 (47.3 MB)
    metadata.json              # 인덱스 메타데이터 + chunk id_map
  eval_results/
    final_report.md            # 최종 평가 리포트
    iteration_history.json     # Iter0~Iter14 전체 이력
    charts/                    # 시각화 차트 PNG (5개)

/models/                       # 베이스 모델 (Docker 볼륨 마운트)
  Qwen3-4B/                    # Qwen3-4B safetensors (~7.6 GB)
```

---

## 데이터 파이프라인

179건 선박 도장 문서(PDF 110건, HTML 69건)를 처리하여 10,829개 청크를 생성하는 4단계 파이프라인이다.

### 1단계: 문서 파싱

```bash
python3 trainer/scripts/ingest_docs.py \
  --input-dir /data/raw \
  --output-dir /data/parsed \
  --manifest /data/manifest.csv
```

- PDF → PyMuPDF (페이지별 텍스트 + 표 감지 → Markdown 변환)
- HTML → BeautifulSoup (script/style 제거, 표 → Markdown)
- 출력: `/data/parsed/` (JSON 형식, 179건)

### 2단계: 텍스트 정규화

```bash
python3 trainer/scripts/clean_docs.py \
  --input-dir /data/parsed \
  --output-dir /data/cleaned \
  --rules trainer/configs/normalization_rules.yaml
```

- 10단계 정규화: NFKC → 제어문자 → 공백 → 줄바꿈 → 헤더/푸터 → 페이지번호 → 단위 → 용어 → 공백정리 → 빈줄축소
- 출력: `/data/cleaned/` (179건)

### 3단계: 중복 제거

```bash
python3 trainer/scripts/dedup_docs.py \
  --input-dir /data/cleaned \
  --output-dir /data/deduplicated
```

- 3단계 중복 제거: SHA-256 정확 중복 → MinHash+LSH 유사 문서 → 문단 수준 중복
- 출력: `/data/deduplicated/` (147건, 32건 제거)

### 4단계: 토큰 청킹

```bash
python3 trainer/scripts/chunk_docs.py \
  --input-dir /data/deduplicated \
  --output-dir /data/chunked \
  --schema trainer/configs/metadata_schema.json
```

- Qwen3-4B 토크나이저 기반 토큰 청킹 (min 50, target 300-800, max 800 tokens)
- 15% 오버랩, chunk_id 형식: `DOC-{doc_id}-C{seq:04d}`
- 출력: `/data/chunked/chunks.jsonl` (10,829건)

### 5단계: FAISS 인덱스 구축

```bash
python3 trainer/scripts/build_index.py \
  --config trainer/configs/rag_config.yaml
```

- BGE-M3 임베딩 (1024차원, batch_size=32, GPU)
- FAISS HNSW32 인덱스 (Inner Product + 정규화 = cosine similarity)
- 출력: `/output/indexes/faiss.index` (47.3 MB), `/output/indexes/metadata.json`

---

## SFT 데이터 생성

### DocQA SFT 생성

```bash
python3 trainer/scripts/generate_sft.py \
  --chunks /data/chunked/chunks.jsonl \
  --output-dir /data/sft
```

- 청크 기반 QA 쌍 생성 (6개 카테고리: 문서QA, 절차설명, 결함분석, DFT피드백, 용어/다국어, 요약/비교)
- 출력: `/data/sft/sft_train.jsonl` (380건), `/data/sft/sft_val.jsonl` (20건)

### FC-aware SFT 생성 (Tool Calling 학습 데이터 추가)

```bash
python3 trainer/scripts/generate_fc_sft.py \
  --existing-sft /data/sft/sft_train_docqa_only.jsonl \
  --output-dir /data/sft
```

- 7개 도구별 synthetic FC 데이터 생성 (single-turn + multi-turn)
- 기존 DocQA 샘플 30%를 FC system prompt로 업그레이드
- 출력: `/data/sft/sft_train.jsonl` (567건, Iter14 기준), `/data/sft/sft_val.jsonl` (30건)

**SFT 데이터 구성 (Iter14)**

| 카테고리 | 건수 |
|---|---|
| Document QA | 380건 |
| Tool Calling (FC single-turn) | 173건 |
| Tool Calling (FC multi-turn) | 14건 |
| 합계 | 567건 |

---

## QLoRA 학습

```bash
python3 trainer/scripts/train_qlora.py \
  --config trainer/configs/qlora_config.yaml
```

**주요 하이퍼파라미터**

| 파라미터 | 값 |
|---|---|
| 베이스 모델 | Qwen3-4B (4-bit NF4) |
| LoRA rank (r) | 16 |
| lora_alpha | 32 |
| lora_dropout | 0.05 |
| target_modules | q/k/v/o_proj, gate/up/down_proj |
| learning_rate | 2e-4 (cosine scheduler) |
| epochs | 3 |
| batch_size | 1 (gradient_accumulation=8) |
| max_seq_length | 2048 |
| precision | bf16 |
| optimizer | adamw_8bit |

**Iter14 학습 결과**

| 항목 | 수치 |
|---|---|
| 학습 시간 | ~23분 |
| Train Loss | 1.68 → 0.24 |
| Eval Loss (epoch 1/2/3) | 0.505 / 0.396 / 0.390 |
| Peak VRAM | 8.52 GB |
| Trainable Params | 33M / 4.05B (0.81%) |

출력 어댑터: `/output/adapters/qwen3_4b_shipcoat_lora/`

---

## 평가 실행

### RAG+QLoRA 평가 (DocQA 50문항)

```bash
python3 trainer/scripts/run_qlora_eval.py \
  --config trainer/configs/rag_config.yaml \
  --mode rag-qlora \
  --adapter-dir /output/adapters/qwen3_4b_shipcoat_lora
```

`--mode` 옵션: `qlora-only`, `rag-qlora`, `compare` (3-way 비교)

### Agentic 평가 (30문항)

**1단계: 응답 생성**

```bash
python3 trainer/scripts/generate_agentic_responses.py \
  --config trainer/configs/rag_config_fewshot.yaml \
  --adapter /output/adapters/qwen3_4b_shipcoat_lora \
  --output /output/eval_results/agentic_responses.jsonl
```

**2단계: 채점**

```bash
python3 trainer/scripts/run_agentic_eval.py \
  --responses /output/eval_results/agentic_responses.jsonl \
  --output-dir /output/eval_results/agentic_eval \
  --tool-schemas-dir trainer/configs/tool_schemas
```

### 세션 코칭 테스트

```bash
# Dry-run (위험 플래그 계산만, LLM 미사용)
python3 trainer/scripts/session_coaching.py \
  --input /data/eval/coaching_samples/session_bad_dft.json \
  --dry-run

# Full pipeline (LLM 추론 포함)
python3 trainer/scripts/session_coaching.py \
  --input /data/eval/coaching_samples/session_bad_dft.json \
  --config trainer/configs/rag_config_fewshot.yaml \
  --adapter /output/adapters/qwen3_4b_shipcoat_lora \
  --mcp-url http://localhost:8100
```

---

## 설정 파일 가이드

| 파일 | 설명 |
|---|---|
| `qlora_config.yaml` | QLoRA 학습 하이퍼파라미터. 모델 경로, LoRA rank/alpha, 학습률, epoch, 배치 크기, 출력 경로 등 |
| `rag_config.yaml` | RAG 파이프라인 설정. BGE-M3 임베딩(1024d), FAISS HNSW(m=32, efConstruction=200), top-k=5, rerank top-3, 생성 파라미터 |
| `rag_config_fewshot.yaml` | Agentic 파이프라인 설정. 7개 도구 few-shot 예시, 도구 선택 규칙, 응답 구조 템플릿, temperature=0.3, max_new_tokens=1024 |
| `normalization_rules.yaml` | 텍스트 정규화 규칙. 단위 통일(µm, °C, mm), 도메인 용어 대체(DFT, WFT, Sa 등), 헤더/푸터 패턴 |
| `metadata_schema.json` | 청크 메타데이터 JSON Schema (draft-07). chunk_id, doc_id, domain, text, token_count 등 |
| `tool_calling_protocol.json` | MCP 프로토콜 규약. 7개 등록 도구, 엔드포인트 매핑, 요청/응답 포맷, 에러 코드, retry 정책(max 1회, backoff 1000ms) |
| `eval_rubric_agentic.md` | Agentic 5차원 채점 루브릭. tool_selection_accuracy, parameter_correctness, reasoning_depth, context_integration, response_structure (각 0-2점, 최대 10점) |
| `eval_rubric.md` | DocQA 5차원 채점 루브릭. 기술 정확성, 문서 충실도, 용어 적절성, 완전성, 한국어 품질 (각 0-2점) |
| `tool_schemas/` | 7개 MCP 도구 JSON Schema (OpenAI-compatible function calling 포맷) |

---

## 환경 변수

| 변수명 | 기본값 | 설명 |
|---|---|---|
| `DEMO_MODE` | `true` | `true`: 더미 데이터 스트리밍 (GPU 불필요), `false`: 실제 파이프라인 |
| `APP_PORT` | `7860` | Gradio 웹 UI 포트 |
| `MCP_BASE_URL` | `http://localhost:8100` | MCP 서버 기본 URL |
| `CONFIG_PATH` | `/workspace/trainer/configs/rag_config_fewshot.yaml` | RAG/Agentic 설정 파일 경로 |
| `ADAPTER_PATH` | `/output/adapters/qwen3_4b_shipcoat_lora/` | QLoRA 어댑터 경로 |
| `ANTHROPIC_API_KEY` | — | Task Master AI 기능 사용 시 필요 (PoC 실행에는 불필요) |
| `HF_HOME` | `/hf-cache` | HuggingFace 캐시 루트 디렉토리 |

---

## 주요 의존성

| 패키지 | 버전 | 용도 |
|---|---|---|
| torch | 2.4.0+cu124 | 딥러닝 프레임워크 |
| transformers | 5.3.0 | 모델 로딩, 토크나이저 |
| peft | 0.18.1 | LoRA 어댑터 |
| trl | 0.29.0 | SFTTrainer |
| bitsandbytes | 0.49.2 | 4-bit 양자화 |
| accelerate | 1.13.0 | 분산 학습 지원 |
| sentence-transformers | 5.2.3 | BGE-M3 임베딩, CrossEncoder 리랭커 |
| faiss-cpu | 1.13.2 | FAISS 벡터 인덱스 |
| gradio | 6.9.0 | 웹 데모 UI |
| fastapi | 0.135.1 | MCP 서버 |
| uvicorn | 0.41.0 | ASGI 서버 |
| PyMuPDF | 1.27.1 | PDF 파싱 |
| beautifulsoup4 | — | HTML 파싱 |
| datasketch | 1.9.0 | MinHash+LSH 중복 제거 |
| pydantic | — | 데이터 검증 |
| yaml | — | 설정 파일 파싱 |

**알려진 비호환 이슈**

- `unsloth`: `transformers==5.3.0`과 충돌 → peft+trl 직접 사용
- `FlagEmbedding 1.3.5`: `transformers 5.3.0`에서 `is_torch_fx_available` 제거됨 → `sentence-transformers`로 BGE-M3 로드
- `vLLM`: RTX 4080 16GB VRAM에서 OOM → HF 4-bit 백엔드 확정

---

## 이터레이션 히스토리

| 이터레이션 | 모델 | SFT 데이터 | RAG+QLoRA | Agentic | 주요 결정 |
|---|---|---|---|---|---|
| Iter0 | Qwen2.5-7B | v0 (보일러플레이트 66%) | 1.148 | — | DROP (halluc 10%) |
| Iter1 | Qwen2.5-7B | v1 (다양성 개선) | 1.376 | — | KEEP (halluc 2%) |
| Iter3 | Qwen3-4B | v1 | 1.348 | — | REGEN (halluc 18%) |
| Iter4 | Qwen3-4B | v2 (Qwen3 RAG 기반) | 1.368 | 1.90 | FC 학습 필요 |
| Iter5 | Qwen3-8B | v2 | 1.408 | — | ARCHIVED (VRAM 초과) |
| Iter6 | Qwen3-4B | v3 (FC-aware) | — | 2.60 | tool_calls 53회 |
| Iter7 | Qwen3-4B | v3 (multi-turn 추가) | — | 2.97 | max_new_tokens 1024 |
| Iter8 | Qwen3-4B | v3 | — | 3.87 | graceful fallback 적용 |
| Iter9 | Qwen3-4B | v3 | — | 3.50 | SemanticRouter + adaptive compression |
| Iter10 | Qwen3-4B | v3 | — | 3.37 | adaptive compression 검증 |
| Iter11 | Qwen3-4B | v3 | — | 3.83 | 3-way Router, 7 tool few-shot, temp 0.3 |
| Iter12 | Qwen3-4B | v3 | — | 3.30 | temp 0.5 실험 → REVERTED |
| Iter13 | Qwen3-4B | v4 (784건, 도구 균형) | — | 4.83 | 목표 4.0 달성 |
| Iter14 | Qwen3-4B | v4 (567건, 중복 제거) | — | 4.97 | assistant_only_loss, 86 tool calls, POC_CONFIRMED |

**Qwen3-4B 선정 근거**: RTX 4080 16GB 기준 VRAM ~8.5 GB (학습), ~4 GB (추론). Qwen3-8B는 학습 시 ~16 GB로 XR 시뮬레이터 동시 가동 불가.

---

## 레이턴시 벤치마크

A40 GPU 기준 측정값이다. RTX 4080에서는 약 1.5-2배 증가 예상.

| 라우트 | 대표 질문 | 레이턴시 | LLM 라운드 | 목표 |
|---|---|---|---|---|
| fast | "에폭시 도료란?" | 9.9초 | 1 | — |
| training_realtime | "지금 도막 상태 어때?" | 55.5초 | 2 | < 60초 ✅ |
| training_review | "자세 교정 필요해?" | 85.3초 | 5 | 제약 없음 |

LLM inference가 전체 레이턴시의 95%를 차지한다. MCP 도구 호출은 <1ms (순수 계산).

---

## 라이선스 및 참고사항

| 구성 요소 | 라이선스 |
|---|---|
| Qwen3-4B | Apache 2.0 |
| BGE-M3 / BGE-Reranker-v2-m3 | MIT |
| 본 PoC 코드 | 내부 프로젝트 |

**본 저장소는 내부 PoC 용도이며 외부 배포를 금한다.**

참고 문서:
- `FINAL_INTEGRATED_PRD.md`: 통합 기술 명세서 (v1.8.8)
- `AgenticCopilot_Architecture.md`: 아키텍처 설계서
- `PATCH_NOTE.md`: 전체 이터레이션 히스토리 (Iter0~Iter14)
- `/output/eval_results/final_report.md`: 최종 평가 리포트

