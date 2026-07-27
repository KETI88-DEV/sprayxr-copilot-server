# 선박 도장 AI Co-Pilot PoC

## 프로젝트 개요

선박 도장 품질 관리를 위한 AI 코파일럿 PoC(Proof of Concept)이다. XR 훈련 시뮬레이터(Unity)와 연동하여 작업자에게 실시간 코칭 피드백을 제공하는 것을 목표로 한다.

**핵심 기술 스택**

- **베이스 모델**: Qwen3-4B (RTX 4080 기준선) / Qwen3.6-27B (Iter18+, A40 학습·평가 경로)
- **파인튜닝**: QLoRA — Qwen3-4B rank 16·4-bit NF4 / Qwen3.6-27B rank 16·4-bit NF4 v6.1 decontam, peft + trl SFTTrainer
- **지식 검색**: BGE-M3 임베딩 + FAISS HNSW 인덱스 + BGE-Reranker-v2-m3 (recall@8 0.74 after re-index)
- **에이전트**: ReAct 오케스트레이터 + 단일 FastAPI MCP 집계 서비스의 7개 tool router / 7개 모델-대상 도구 스키마
- **데모 UI**: 3가지 병행 지원 — 기존 Gradio 6.x (`trainer/scripts/demo_app.py`) / 신규 React 웹 (`demo-ui/`) / 동일 빌드를 감싸는 Electron 데스크톱 셸
- **데모 백엔드**: `trainer/demo_api/` FastAPI (SSE 이벤트 전달, `DEMO_LIVE_MODE=on`으로 실제 파이프라인 연결; 현재 live 응답 delta는 buffered compatibility 방식)
- **레이턴시 계측**: tool deadline, 라운드별 token cap, end-to-end timing SSE, `bench_latency.py` HTTP/SSE 벤치 하네스

**최종 성과 (Iter18 진행 중)**

| 지표 | 결과 |
|---|---|
| Agentic 평가 점수 | 4.97/10 (Iter14, 목표 4.0 달성) · Iter17 rescore 5.57 (Oracle 검증 후 5.30) |
| Qwen3.6-27B v6.1 3-seed 기준선 | latency token-cap 변경 전 평균 6.744, sample SD 0.139 (seeds 42/43/44; 변경 후 gate 재평가·n=5·CI 대기) |
| RAG+QLoRA DocQA 점수 | 1.368/2.0 |
| Hallucination 비율 | 2.0% |
| Tool Calling 작동 | 86회/30문항 |
| RTX 4080 VRAM 사용 | ~8.5 GB (Qwen3-4B 학습), ~4 GB (추론) |
| A40 48GB VRAM 사용 | peak 34.87 GB (Qwen3.6-27B 4-bit NF4 v6.1 학습) |
| Rerank recall@8 | 0.66 → 0.74 (공백 복원 + 재임베딩) |
| Latency 인프라 | deadline guard, tool/synthesis token cap 분리, SSE timing 계약, HTTP/SSE 벤치 하네스 완료; 실제 GPU gate는 Task 8 대기 |

**Iter18 하이라이트 (2026-07-25)**
- React 웹 + Electron 데스크톱 데모 (Gradio 유지)
- FastAPI SSE 백엔드 + `DEMO_LIVE_MODE` 스위치로 실제 AgentPipeline 연동
- 원격 접근 경로 정리 (데모 API는 기본 `0.0.0.0` 바인딩, 런타임 API base URL 주입, CORS 기본 `*`; Vite 개발 서버는 `127.0.0.1` 고정이라 필요 시 `--host`로 노출)
- Qwen3.6-27B 4-bit NF4 v6.1 재학습 및 latency 변경 전 3-seed 기준선 완료 (변경 후 gate 재평가·n=5 확장 대기)
- 도구 호출 파서 통일 및 seed-기반 재현성 확보 (`run_multiseed_eval.py`)
- RAG TTFT 30초 / tool completion 180초 예산을 위한 Tasks 1–7 계측·제어 경로 완료 (실제 GPU gate는 Task 8)
- 상세: [`PATCH_NOTE.md`](PATCH_NOTE.md) → `[Iter18]` 섹션

---

## 아키텍처

### 전체 데이터 흐름

```text
오프라인 데이터 경로
문서 수집 → 파싱 → 정규화 → 중복 제거 → 토큰 청킹
         → BGE-M3 임베딩 → FAISS HNSW + metadata

런타임 요청 경로
Unity / Gradio / React·Electron
          │
          ├─ React·Electron → FastAPI demo_api :8120 → ChatService
          │                                      ├─ demo scenario
          │                                      └─ live runner
          │
          └──────────────────────────────→ AgentPipeline
                                              │
                                      3-Way Semantic Router
                          ┌───────────────────┼───────────────────┐
                  training_realtime    training_review          fast
                    max_rounds=2         max_rounds=5       max_rounds=1
                          └───────────────────┼───────────────────┘
                                      RAG 검색 + rerank
                                              │
                       Qwen3-4B 또는 Qwen3.6-27B HF 4-bit NF4
                                              │
                                   ToolCallingAdapter → MCP :8100
                        dft / anomaly / sequence / pose / location / timeseries / vlm
                                              │
                   ┌──────────────────────────┴──────────────────────────┐
             run() 구조화 응답                           run_streaming() buffered 이벤트
       평가·시뮬레이터 호환 경로                 status/tool_result/delta/complete/error
                                                               │
                                                        FastAPI SSE → UI
```

### 3-Way Semantic Router

| 라우트 | 판단 기준 | max_rounds | 응답 목표 |
|---|---|---|---|
| **training_realtime** | 순서, 이상탐지, 합부 판정 키워드 | 2 | RAG TTFT ≤30초 / tool 완료 ≤180초 |
| **training_review** | 자세, 추세분석, 코칭, 리포트 키워드 | 5 | tool 완료 ≤180초 |
| **fast** | 단순 지식 질문 (용어, 규정) | 1 | TTFT ≤30초 |

`run()`은 평가 재현성과 기존 시뮬레이터 계약을 위해 유지한다. additive `run_streaming()`은 현재 `run()` 완료 후 답변을 24자 단위 `message_delta`로 재방출하는 buffered compatibility 경로이므로 true token streaming이 아니며 TTFT를 줄이지 않는다. 벤치 하네스는 사용자 관측 기준으로 첫 `message_delta`를 TTFT, `complete` 도착을 전체 완료 시점으로 측정한다. Task 8에서 RAG TTFT가 실패할 때만 `_stream_generate()`를 실제 live 경로에 연결한다.

### MCP 집계 서비스/도구 현황

- 런타임 서비스: `mcp_servers.main:app` 단일 FastAPI 프로세스가 7개 router를 `:8100`에 통합 마운트한다.
- `/health`의 `servers: 7`은 기존 호환 필드이며 프로세스 수가 아니라 마운트된 tool router 수를 뜻한다.
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

환경변수:
- `DEMO_UI_ORIGINS` — 콤마 구분 CORS 오리진. 미설정 시 `*` (원격 웹 접근 편의). 특정 도메인만 허용하려면 명시적으로 설정한다. 예: `https://demo.example.com,http://192.168.0.10:5173`.
- `DEMO_LIVE_MODE` — `off`(기본) 또는 `on`. `on`이면 startup 시 `build_default_live_runner`를 부착해 실제 AgentPipeline + MCP + QLoRA 어댑터를 사용한다. 잘못된 값이면 startup 실패.
- `CONFIG_PATH` / `ADAPTER_PATH` / `MCP_BASE_URL` — live 러너의 파이프라인 구성. 각각 기본값은 `/workspace/trainer/configs/rag_config_fewshot_27b.yaml`, `/output/adapters/qwen36_27b_shipcoat_lora_v6_1`, `http://localhost:8100`.

원격 접근 (LAN/서버 배포):
```bash
# 기본: 모든 인터페이스에서 수신 (--host 기본값이 0.0.0.0)
python3 -m trainer.demo_api.app --port 8120

# Live 파이프라인까지 켠 상태로 기동
python3 -m trainer.demo_api.app --port 8120 --live

# Qwen3.6-27B v6.1 live 경로
CONFIG_PATH=trainer/configs/rag_config_fewshot_27b.yaml \
ADAPTER_PATH=/output/adapters/qwen36_27b_shipcoat_lora_v6_1 \
python3 -m trainer.demo_api.app --port 8120 --live

# 특정 오리진만 허용
DEMO_UI_ORIGINS=https://demo.example.com python3 -m trainer.demo_api.app --port 8120 --live
```

### 2) React 웹 데모

```bash
cd demo-ui && npm install
cd demo-ui && npm run dev:web             # Vite 개발 서버 (127.0.0.1 전용 바인딩)
```

브라우저에서 접속:
- 로컬: `http://127.0.0.1:5173`
- 원격/LAN: `dev:web`과 `preview:web` 스크립트는 `--host 127.0.0.1`을 고정하므로 그대로는 외부에서 접속되지 않는다. 열어야 하면 명시적으로 덮어쓴다.

```bash
cd demo-ui && npm run dev:web -- --host 0.0.0.0        # 개발 서버를 모든 인터페이스에 노출
cd demo-ui && npm run preview:web -- --host 0.0.0.0    # 정적 미리보기(4173)를 외부에 노출
```

개발 서버를 외부에 여는 것은 신뢰된 네트워크에서만 권장한다. 원격 접속은 SSH 터널이 더 안전하다(아래 Windows 가이드 참고).

API 위치 지정 우선순위:
1. `window.__DEMO_API_BASE_URL__` — 런타임 주입. 빌드 시 `VITE_API_BASE_URL` 값이 `index.html`에 치환되며, 배포된 `dist/index.html`의 해당 값을 직접 수정하면 재빌드 없이 API 주소를 바꿀 수 있다.
2. 빌드 시 `VITE_API_BASE_URL` 환경변수 (JS 번들에 내장).
3. 미지정 → same-origin (리버스 프록시 뒤 배치 시 권장). **Electron은 `file://` 오리진에서 로드되므로 same-origin fallback이 동작하지 않는다. 1 또는 2를 반드시 지정한다.**

정적 빌드 + 미리보기:
```bash
cd demo-ui && npm run build:web            # 산출물: demo-ui/dist/
cd demo-ui && npm run preview:web          # 정적 서버 127.0.0.1:4173
```

원격 배포 예시 (nginx 리버스 프록시):
```nginx
location /api/ {
    proxy_pass http://127.0.0.1:8120/api/;
    proxy_http_version 1.1;
    proxy_set_header Connection "";        # SSE 안정성 확보
    proxy_buffering off;
}
location / {
    root /var/www/demo-ui/dist;
    try_files $uri /index.html;
}
```
이 구성이면 프런트는 same-origin으로 API를 호출하므로 CORS/베이스 URL 설정이 필요 없다.

UI 조작:
- `Enter` — 전송. `Shift+Enter` — 줄바꿈.
- 상단 상태바 세그먼트에서 `demo` ↔ `live` 모드 전환. `live`인데 API가 `DEMO_LIVE_MODE=off`로 떠 있으면 서버가 명확한 error 이벤트를 반환한다.
- 스트리밍 중에는 마지막 응답 말미에 커서(`▌`)가 깜박이고, 메시지가 늘면 자동으로 스크롤이 하단으로 따라간다.

### 3) Electron 데스크톱 데모

```bash
cd demo-ui && npm run start:electron
```

내부적으로 `build:web` + `build:electron` 후 Electron이 `dist/index.html`을 로드한다. 개발 시 라이브 리로드:
```bash
cd demo-ui && npm run dev:electron
```

Electron 셸은 `contextIsolation: true`, `nodeIntegration: false`, `sandbox: true`로 고정되며 preload는 `window.desktop.{ platform, electronVersion }`만 노출한다.

**실행 환경 제약**: 학습/서빙용 Docker 컨테이너에는 디스플레이가 없어 Electron 창을 직접 띄울 수 없다. 컨테이너에서는 `xvfb-run`을 통한 헤드리스 검증만 가능하고, 실제 데스크톱 실행은 Windows/macOS/데스크톱 리눅스에서 수행한다.

### 3-1) Windows에서 Electron 데모 실행 (원격 API 연결)

서버(Ubuntu + Docker)에서 API를 실행하고 Windows에서 UI만 띄우는 구성이다.

**A. Windows 사전 준비**
- Node.js 20 LTS 이상, Git 설치.
- 저장소를 Windows 로컬로 가져온다(`git clone` 또는 서버에서 복사). UI 실행에는 `demo-ui/` 디렉터리면 충분하다.

**B. 서버에서 데모 API 노출**

컨테이너는 브리지 네트워크(예: `172.21.0.2`)를 사용하므로 컨테이너 내부 포트는 호스트나 외부에서 자동으로 보이지 않는다. 다음 중 하나가 필요하다.

```bash
# 1) 컨테이너 안에서 API 기동 (모든 인터페이스)
python3 -m trainer.demo_api.app --host 0.0.0.0 --port 8120          # demo 모드 (GPU 불필요)
python3 -m trainer.demo_api.app --host 0.0.0.0 --port 8120 --live   # live 모드 (GPU 필요)

# 2) 호스트에서 컨테이너 포트 게시 상태 확인
docker port ship-paint-trainer

# 3) 게시되어 있지 않으면 deploy/docker-compose.yml 의 trainer 서비스에
#      ports:
#        - "8120:8120"
#    를 추가한 뒤 재생성
docker compose -f deploy/docker-compose.yml up -d
```

Windows에서 SSH 터널로 접근(PowerShell):
```powershell
ssh -L 8120:127.0.0.1:8120 <user>@<ubuntu-host>
```
터널을 유지하는 동안 Windows에서의 API 주소는 `http://127.0.0.1:8120`이 된다.

**C. Windows에서 API 주소 지정**

`demo-ui\.env.local` 파일을 만들고 한 줄을 넣는다.
```
VITE_API_BASE_URL=http://127.0.0.1:8120
```
SSH 터널 대신 서버 IP로 직접 접속한다면 `http://<ubuntu-host>:8120`을 쓰고, 서버 API는 `--host 0.0.0.0`으로 기동한다. `DEMO_UI_ORIGINS`를 좁게 설정하면 Electron(`file://`, Origin `null`)이 차단될 수 있으므로 기본값(`*`)을 유지하거나 프록시를 사용한다.

**D. 실행 (PowerShell)**
```powershell
cd demo-ui
npm ci
npm run start:electron     # build:web + build:electron 후 Electron 창 실행
npm run dev:electron       # 개발용 핫 리로드
npm run dev:web            # Electron 없이 브라우저만 사용할 때 http://127.0.0.1:5173
```

**E. 재빌드 없이 API 주소만 변경**

`demo-ui\dist\index.html` 상단 스크립트의 `var injected = "...";` 값을 원하는 API 주소로 바꾸면 다음 실행부터 반영된다.

**F. 문제 해결**

| 증상 | 원인 | 조치 |
|---|---|---|
| 상태바가 `오프라인` | API 주소 미지정 또는 도달 불가 | `.env.local`의 `VITE_API_BASE_URL` 확인 후 재빌드, SSH 터널·포트 게시 확인 |
| `live` 전환 시 error 이벤트 | 서버가 `DEMO_LIVE_MODE=off` 상태 | 서버를 `--live`로 재기동 |
| 브라우저는 되는데 Electron만 실패 | `file://`에는 same-origin fallback이 없음 | `VITE_API_BASE_URL`을 지정해 다시 빌드 |
| 컨테이너에서 창이 안 뜸 | 디스플레이 없음 | Windows에서 실행하거나 `xvfb-run`으로 헤드리스 검증 |

### 4) 테스트

```bash
# Python API
python3 -m pytest trainer/demo_api/tests -q

# React (Vitest)
cd demo-ui && npm test -- --run

# 브라우저 E2E (Playwright)
cd demo-ui && npm run test:e2e

# HTTP/SSE latency harness 순수 로직
python3 -m pytest trainer/scripts/tests/test_bench_latency.py -q
```

### 3D 뷰어 관련 참고

`SP0035_rig.glb` 같은 3D 자산은 Windows 로컬 경로에서만 접근되며 이 저장소의 컨테이너에는 마운트되어 있지 않다. 자산 매핑(`zone_id` → 메시 노드 또는 월드 좌표)이 정리되면 별도 spec에서 3D 뷰어 계층을 추가한다.

## 하드웨어 요구사항

| 구분 | 사양 |
|---|---|
| **4B 추론·XR 기준선** | RTX 4080 16GB VRAM |
| **27B 학습·평가 경로** | NVIDIA A40 48GB VRAM 또는 동급 |
| **4B VRAM** | 추론 ~4 GB / QLoRA 학습 ~8.5 GB peak |
| **27B VRAM** | 4-bit NF4 v6.1 QLoRA 학습 34.87 GB peak / live 추론 peak는 Task 8에서 계측 |
| **RAM** | 32 GB 이상 권장 |
| **저장 공간** | 4B 모델 약 8 GB 또는 27B 모델 약 52 GB + 데이터·인덱스·평가 산출물 |
| **CUDA** | 12.4 이상 |

**RTX 4080 기준선 VRAM 예산**

| 구성 요소 | VRAM |
|---|---|
| LLM (Qwen3-4B, 4-bit NF4) + QLoRA Adapter | ~4.5 GB |
| BGE-M3 임베딩 + FAISS HNSW 인덱스 | ~2.0 GB |
| MCP 서버 (FastAPI) | ~1.5 GB |
| XR 시뮬레이터 여유분 | ~8.0 GB |

Qwen3.6-27B는 A40 전용 경로다. RTX 4080에서는 Qwen3-4B 4-bit 기준선을 유지하며, 27B에 vLLM·Unsloth를 필수 의존성으로 추가하지 않는다.

---

## 디렉토리 구조

```text
/workspace/
  trainer/
    scripts/                   # 수집·RAG·학습·평가·agent·benchmark CLI
      ingest_docs.py → clean_docs.py → dedup_docs.py → chunk_docs.py
      build_index.py           # BGE-M3 + FAISS HNSW 인덱스
      search_index.py          # FAISS 검색 + cross-encoder rerank
      agent_pipeline.py        # SemanticRouter, run(), run_streaming()
      tool_calling_adapter.py  # tool loop, deadline, token cap, MCP dispatch
      bench_latency.py         # 실제 FastAPI SSE latency driver
      bench_report.py          # percentile, budget summary, report rendering
      generate_agentic_responses.py / run_agentic_eval.py
      run_multiseed_eval.py    # seed 반복, 재현성·CI 집계
      build_sft_v6.py / generate_fc_sft.py / generate_cot_sft.py
      train_qlora.py           # config-driven QLoRA 학습
      tests/
        test_bench_latency.py
        test_build_expert_reference.py
        test_parse_session.py
    demo_api/                  # FastAPI :8120
      app.py                   # health, sequence, chat SSE routes
      chat_service.py          # demo/live 분기와 pipeline event 전달
      contracts.py             # Pydantic 요청·SSE 계약
      scenario_adapter.py      # GPU-free demo scenario
      tests/
    configs/
      rag_config_fewshot_27b.yaml
      qlora_config_27b_v6_1.yaml
      tool_schemas/            # 7개 모델 대상 tool schema
  mcp_servers/                 # 단일 FastAPI 집계 서비스, 7개 tool router, 통합 :8100
  demo-ui/                     # React 19 + Vite + Electron
    src/contracts.ts           # backend SSE와 정렬된 strict Zod 계약
    src/api.ts                 # HTTP/SSE client
    src/App.tsx                # chat·sequence·tool log orchestration
    electron/                  # sandboxed Electron main/preload
    tests/ / e2e/              # Vitest / Playwright
  deploy/                      # NVIDIA container와 외부 volume wiring
  docs/superpowers/            # 승인된 spec과 구현 계획
  PATCH_NOTE.md                # Iter0~Iter18 변경·검증 기록

/data/                         # 문서, SFT, eval set 외부 volume
/models/Qwen3-4B/              # 4B 기준선
/models/Qwen3.6-27B/           # 27B A40 경로
/output/
  indexes/                     # FAISS index + metadata
  adapters/                    # 4B 및 27B QLoRA adapters
  eval_results/latency_bench/  # timestamp/report.json/SUMMARY.md
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

학습 스크립트는 config-driven이며 4B 기준선과 27B Iter18 경로가 동일한 `train_qlora.py`를 사용한다.

```bash
# Qwen3-4B 기준선
python3 trainer/scripts/train_qlora.py \
  --config trainer/configs/qlora_config.yaml

# Qwen3.6-27B v6.1
python3 trainer/scripts/train_qlora.py \
  --config trainer/configs/qlora_config_27b_v6_1.yaml
```

**학습 트랙**

| 항목 | Qwen3-4B 기준선 | Qwen3.6-27B v6.1 |
|---|---|---|
| 양자화 | 4-bit NF4 | 4-bit NF4 |
| 데이터 | Iter14 v4, 567건 | decontam v6.1, train 636 / val 29 |
| 실행 환경 | RTX 4080 16GB | NVIDIA A40 48GB |
| Peak VRAM | 8.52 GB | 34.87 GB |
| 학습 시간 | 약 23분 | 약 216분 |
| Eval loss | 0.505 → 0.396 → 0.390 | 0.65 → 0.50 → 0.47 |

**Qwen3-4B Iter14 상세**

| 항목 | 수치 |
|---|---|
| 학습 시간 | ~23분 |
| Train Loss | 1.68 → 0.24 |
| Eval Loss (epoch 1/2/3) | 0.505 / 0.396 / 0.390 |
| Peak VRAM | 8.52 GB |
| Trainable Params | 33M / 4.05B (0.81%) |

출력 어댑터:
- 4B: `/output/adapters/qwen3_4b_shipcoat_lora/`
- 27B v6.1: `/output/adapters/qwen36_27b_shipcoat_lora_v6_1/`

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

Qwen3.6-27B는 `trainer/configs/rag_config_fewshot_27b.yaml`과 v6.1 adapter를 사용한다. latency token-cap 변경 전 3-seed 기준선은 평균 6.744, sample SD 0.139이다. 이 값은 현재 변경의 품질 gate 통과 증거가 아니며, 변경 후 동일 평가 재실행과 seeds 45/46을 포함한 n=5·CI가 후속 단계다.

### HTTP/SSE 레이턴시 벤치

MCP와 live demo API를 먼저 기동한 뒤, pipeline 함수를 직접 호출하지 않고 사용자가 실제로 보는 `/api/chat` SSE를 측정한다.

```bash
# MCP health 확인 후 Qwen3.6-27B v6.1 live API를 :8130에 기동
curl -sSf -m3 http://localhost:8100/health
CONFIG_PATH=trainer/configs/rag_config_fewshot_27b.yaml \
ADAPTER_PATH=/output/adapters/qwen36_27b_shipcoat_lora_v6_1 \
setsid nohup python3 -m trainer.demo_api.app --port 8130 --live \
  > /tmp/bench_api.log 2>&1 < /dev/null &
sleep 90
curl -sSf -m5 http://127.0.0.1:8130/api/health

# 사용자 관측 HTTP/SSE 경계 측정
python3 trainer/scripts/bench_latency.py \
  --api-url http://127.0.0.1:8130 \
  --eval-set /data/eval/eval_set_agentic.jsonl \
  --profile mixed --repeat 1 \
  --output-root /output/eval_results/latency_bench
```

각 실행은 timestamp 디렉터리에 `report.json`과 `SUMMARY.md`를 기록한다. 사전 `rag/tools` profile은 예상 tool 목록으로 질문을 고르지만, 최종 경로와 gate는 `complete.tool_call_count`의 실제 값으로 판정한다. terminal `error`, malformed/missing terminal event, 필수 metric 누락, MCP `TIMEOUT`이 있는 행은 invalid로 보존하되 percentile과 gate 계산에서는 제외하며, 측정 가능한 행이 없으면 `0 ms pass`가 아니라 `no_data`로 보고한다.

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
| `rag_config_fewshot_27b.yaml` | Qwen3.6-27B live/eval 설정. HF 4-bit inference, `tool_deadline_sec=150`, `max_new_tokens_tool=256` 포함 |
| `qlora_config_27b_v6_1.yaml` | Qwen3.6-27B 4-bit NF4 v6.1 데이터·학습·adapter 경로 |
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
| `DEMO_LIVE_MODE` | `off` | FastAPI demo API에서 `on`: 실제 AgentPipeline, `off`: demo scenario |
| `DEMO_UI_ORIGINS` | `*` | 허용 CORS origin 목록(콤마 구분); 미설정 시 전체 허용 |
| `VITE_API_BASE_URL` | same-origin | React build/dev server가 사용할 demo API base URL |
| `APP_PORT` | `7860` | Gradio 웹 UI 포트 |
| `MCP_BASE_URL` | `http://localhost:8100` | MCP 서버 기본 URL |
| `CONFIG_PATH` | `/workspace/trainer/configs/rag_config_fewshot_27b.yaml` | FastAPI live RAG/Agentic 설정 파일 경로 |
| `ADAPTER_PATH` | `/output/adapters/qwen36_27b_shipcoat_lora_v6_1` | FastAPI live QLoRA 어댑터 경로 |
| `HF_DEACTIVATE_ASYNC_LOAD` | `1` | Transformers v5 양자화 모델 로딩의 일시적 GPU 메모리 급증 방지 |
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
| httpx | 0.28.1 | 실제 HTTP/SSE 레이턴시 벤치 client |
| React / Vite / Zod | 19 / 8 / 4 | 웹 UI, build, strict API boundary |
| Electron | 41 | sandboxed 데스크톱 shell |

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
| Iter15 | Qwen3-4B | v4 | — | — | 실험 구성 아카이브 (rag/router 튜닝) |
| Iter16 | Qwen3-4B | v4 | — | — | 실험 구성 아카이브 |
| Iter17 | Qwen3-4B | v4 | — | 5.57 | eval scorer 버그 픽스(context_integration/response_structure +0.20), Oracle rescore 5.30 |
| Iter18 | Qwen3.6-27B (4-bit NF4) | v6.1 (636건, decontam) | — | 변경 전 평균 6.744, sample SD 0.139 (3 seeds) | React/Electron/FastAPI live, tool parser 통일, rerank recall@8 0.74, latency Tasks 1–7 |

**Qwen3-4B 선정 근거**: RTX 4080 16GB 기준 VRAM ~8.5 GB (학습), ~4 GB (추론). Qwen3-8B는 학습 시 ~16 GB로 XR 시뮬레이터 동시 가동 불가.

**Qwen3.6-27B 승격 근거**: 27B 4-bit NF4 QLoRA는 A40 48GB에서 peak 34.87 GB로 학습 가능하고 latency 변경 전 3-seed agentic 평균 6.744를 기록했다. 현재 live 경로는 HF 4-bit를 유지하며, RTX 4080은 Qwen3-4B 기준선을 사용한다. latency 변경 후 ≥6.60 품질 gate는 아직 재검증되지 않았다.

---

## 레이턴시 벤치마크

> **주의**: 아래 표는 deadline guard, 라운드별 token cap, HTTP/SSE timing harness 적용 전의 과거 A40 참고치다. Task 8에서 Qwen3.6-27B v6.1 live 경로를 `bench_latency.py`로 다시 측정하기 전까지 현재 gate 달성값으로 해석하면 안 된다.

| 라우트 | 대표 질문 | 레이턴시 | LLM 라운드 | 목표 |
|---|---|---|---|---|
| fast | "에폭시 도료란?" | 9.9초 | 1 | — |
| training_realtime | "지금 도막 상태 어때?" | 55.5초 | 2 | < 60초 ✅ |
| training_review | "자세 교정 필요해?" | 85.3초 | 5 | 제약 없음 |

LLM inference가 전체 레이턴시의 95%를 차지한다. MCP 도구 호출은 <1ms (순수 계산).

### Iter18 레이턴시 예산과 계측 계약

| 실제 경로 | gate metric | 예산 | 현재 상태 |
|---|---|---:|---|
| RAG, 실제 tool call 0건 | 첫 `message_delta` wall-clock TTFT | p90 ≤30,000 ms, 위반 0건 | Task 8 실측 대기 |
| 도구 호출, 실제 tool call 1건 이상 | HTTP/SSE complete wall-clock total | p99 ≤180,000 ms, 위반 0건 | Task 8 실측 대기 |
| 품질 | Agentic total | ≥6.60 | 변경 전 평균 6.744; latency 변경 후 재평가·n=5 대기 |

- `tool_deadline_sec=150`은 1차 실행용 잠정값이다.
- Task 8에서 `synthesis_ms` p95를 측정한 뒤 `180 - ceil(p95 / 1000)`초로 재보정한다.
- `report.json`은 observed/backend TTFT·total, synthesis percentile, 실제/예상 경로, valid/invalid sample 수, budget gate 상태, deadline/degraded/MCP timeout/error count를 기록한다.
- 실제 RAG TTFT가 실패한 경우에만 true token streaming 완성 또는 kernel/backend 단계로 확장한다.

---

## 라이선스 및 참고사항

| 구성 요소 | 라이선스 |
|---|---|
| Qwen3-4B | Apache 2.0 |
| Qwen3.6-27B | 배포 전 사용한 model artifact의 model card/license 별도 확인 필요 |
| BGE-M3 / BGE-Reranker-v2-m3 | MIT |
| 본 PoC 코드 | 내부 프로젝트 |

**본 저장소는 내부 PoC 용도이며 외부 배포를 금한다.**

참고 문서:
- `FINAL_INTEGRATED_PRD.md`: 통합 기술 명세서 (v1.8.8)
- `AgenticCopilot_Architecture.md`: 아키텍처 설계서
- `PATCH_NOTE.md`: 전체 이터레이션 히스토리 (Iter0~Iter18)
- `/output/eval_results/final_report.md`: 최종 평가 리포트
