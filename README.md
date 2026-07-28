# 선박 도장 AI Co-Pilot PoC

선박 도장 품질 관리를 위한 AI 코파일럿 PoC. Unity XR 훈련 시뮬레이터와 연동해 작업자에게 순서 안내·이상 진단·자세 코칭을 제공하는 것이 목표다.

한 줄 요약: **도장 문서 RAG → QLoRA 파인튜닝 모델 → ReAct 도구 호출(MCP) → SSE 스트리밍 콘솔**.
현재 기본 경로는 Qwen3.6-27B(4-bit NF4) + v6.1 어댑터이며, Qwen3-4B는 RTX 4080 기준선으로 유지한다.

---

## 1. 지금 동작하는 범위

| 영역 | 상태 | 진입점 |
|---|---|---|
| 문서 RAG | 동작 — 10,829 청크, FAISS HNSW, rerank recall@8 0.74 | [`build_index.py`](trainer/scripts/build_index.py), [`search_index.py`](trainer/scripts/search_index.py) |
| 에이전트 | 동작 — 3-way 라우터, 도구 루프, deadline·라운드별 token cap | [`agent_pipeline.py`](trainer/scripts/agent_pipeline.py) |
| MCP 도구 | 동작 — 단일 프로세스에 7개 router (`:8100`). 실제 로직 3 / partial 1 / stub 3 | [`mcp_servers/main.py`](mcp_servers/main.py) |
| 데모 API | 동작 — `:8120` SSE, demo ↔ live 전환 | [`demo_api/app.py`](trainer/demo_api/app.py) |
| 시뮬레이터 인터페이스 | 동작 — 결정론 3단계 + I/O 모니터. **현재 작업 상태의 단일 소유자** | [`simulator.py`](trainer/demo_api/simulator.py), [`sim_session.py`](trainer/demo_api/sim_session.py), [`monitor.py`](trainer/demo_api/monitor.py) |
| React 웹 / Electron 콘솔 | 동작 | [`demo-ui/`](demo-ui) |
| Gradio 데모 | 유지(병행) | [`demo_app.py`](trainer/scripts/demo_app.py) |
| 레이턴시 예산 | 계측·제어 경로 완료, **실측 gate 미실행** | [`bench_latency.py`](trainer/scripts/bench_latency.py) |

아직 아닌 것 — 데모 시연 시 오해를 부르는 지점이라 명시한다.

- `/api/sim/*`의 **응답**은 아직 결정론 스텁이다(실제 코칭 LLM 미연결). 연결 지점은 `_run_phase(build_output=...)` 인자로 남겨 뒀다. 다만 시뮬레이터가 보낸 **상태**는 스텁이 아니며, 도장 순서 패널과 모델 `environment`에 실제로 반영된다(§4).
- 세션 슬롯은 하나다. 새 `start`가 이전 세션을 대체하며, 다중 동시 세션·DB 영속화·인증·Unity 실클라이언트 연동은 범위 밖이다.
- 좌표→구역 폴백은 배선돼 있으나 `location_tracking`이 stub(항상 `B-3`)이라 지금은 위치를 옮기지 못하고 "미매핑 구역"으로만 표시된다.
- `run_streaming()`은 `run()` 완료 후 답변을 24자 단위 `message_delta`로 재방출하는 **buffered 경로**다. true token streaming이 아니므로 TTFT를 줄이지 않는다.
- 27B live 경로의 레이턴시 gate(RAG TTFT p90 ≤30s / 도구 완료 p99 ≤180s)는 아직 **실측 전**이다.
- `pose_estimation`·`location_tracking`·`vlm_analysis`는 stub 응답이다.
- 3D 뷰어는 없다. `zone_id ↔ mesh/좌표` 매핑이 확보되면 별도 spec으로 진행한다.

---

## 2. 실행

### 2.1 컨테이너가 런타임이다

호스트 `python3`에는 의존성이 없다. **모든 Python 명령은 컨테이너 안에서, 저장소 루트(`/workspace`)에서** 실행한다. 그래야 `trainer.*` / `mcp_servers.*` import가 해석된다.

```bash
docker compose -f deploy/docker-compose.yml up -d
docker exec -it ship-paint-trainer bash                       # working_dir=/workspace

# 단발 실행
docker exec -w /workspace ship-paint-trainer python3 -m pytest trainer/demo_api/tests -q
```

컨테이너 런타임: Python 3.11.9 / fastapi 0.135.1 / pydantic 2.12.5.
compose가 `8120`을 호스트에 게시하고 live 기본값(`DEMO_LIVE_MODE=on`, 27B config·adapter 경로, `MCP_BASE_URL`)을 환경변수로 주입한다.

| 호스트 경로 | 컨테이너 경로 | 용도 |
|---|---|---|
| `/data/ship-paint-ai/models` | `/models` | 베이스 모델 |
| `/data/ship-paint-ai/data` | `/data` | 학습·평가 데이터 |
| `/data/ship-paint-ai/output` | `/output` | 인덱스, 어댑터, 평가 산출물 |
| `/data/ship-paint-ai/hf-cache` | `/hf-cache` | HuggingFace 캐시 |
| `/data/ship-paint-ai/logs` | `/logs` | 학습 로그 |
| `/home/mjkim/ship-paint-ai` | `/workspace` | 코드 |

이미지 Dockerfile에 없는 런타임 패키지는 컨테이너를 재생성하면 사라진다. 목록으로 복구한다.

```bash
docker exec -it ship-paint-trainer pip install -r /workspace/trainer/requirements-runtime.txt
```

### 2.2 백엔드 두 개

```bash
uvicorn mcp_servers.main:app --host 0.0.0.0 --port 8100 &   # 7개 도구 router
python3 -m trainer.demo_api.app --port 8120                 # demo 모드 (GPU 불필요)
python3 -m trainer.demo_api.app --port 8120 --live          # live 모드 (27B 로딩, GPU 필요)
```

```bash
curl -s http://127.0.0.1:8100/health      # {"status":"ok","servers":7,...}
curl -s http://127.0.0.1:8120/api/health  # {"status":"ok","servers":1,...}
```

- MCP `/health`의 `servers: 7`은 프로세스 수가 아니라 **마운트된 tool router 수**다(호환 필드).
- MCP `:8100`은 컨테이너 내부 전용이다. compose가 게시하지 않으므로 외부에서 붙는 것은 데모 API `:8120`뿐이다.
- `--live` 기동은 27B 로딩에 수십 초~수 분이 걸린다. `/api/health`가 응답하기 전까지 UI는 오프라인으로 표시된다.

### 2.3 UI 세 가지

```bash
# React 웹 — http://127.0.0.1:5173
cd demo-ui && npm install && npm run dev:web

# Electron 데스크톱 (build:web + build:electron 후 창 실행)
cd demo-ui && npm run start:electron

# Gradio (기존 유지) — http://localhost:7860
DEMO_MODE=true  python3 trainer/scripts/demo_app.py    # 더미, 모델 로딩 없음
DEMO_MODE=false python3 trainer/scripts/demo_app.py    # 실제 파이프라인
```

React 콘솔 구성(`demo-ui/src/components/`):

| 패널 | 내용 |
|---|---|
| `TitleBar` | 프로그램명 + 메뉴 아이콘 한 줄. 로그 토글·초기화·재연결 |
| `MonitorPanel` | 연결/모드(demo·live)/실행 상태·경과·턴 지표(TTFT·total·synthesis·도구 수·청크 수)와 도장 순서 |
| `SequencePanel` | 도장 순서 + 시뮬레이터 세션 단계·짧은 세션 ID·미매핑 구역 배지. 구역 상태는 `완료`/`현재`/`대기` |
| `ChatPanel` | 질문과 응답을 시각적으로 분리. `AnswerBody`가 제목·표·강조를 렌더링 |
| `ToolLogPanel` | 진행 단계는 한 줄이 제자리 갱신되고, 마일스톤·도구 결과·완료·오류만 누적 |
| `SimulatorMonitorPanel` | 시뮬레이터 I/O 타임라인 + `세션 시작`·`런타임`·`세션 끝` 더미 송신 버튼 |

조작: `Enter` 전송 / `Shift+Enter` 줄바꿈. 상단에서 `demo` ↔ `live` 전환. 서버가 `DEMO_LIVE_MODE=off`인데 `live`를 고르면 서버가 명시적 error 이벤트를 돌려준다.
`SP0035-L01`은 데모 레벨 식별자다(`demo-ui/src/App.tsx`의 `LEVEL_ID`). `/api/sequence/{level_id}` 스냅샷과 시뮬레이터 더미 페이로드가 같은 값을 쓴다.

컨테이너에는 디스플레이가 없어 Electron 창이 뜨지 않는다. 헤드리스 검증은 `xvfb-run`, 실제 데스크톱 실행은 Windows/macOS/데스크톱 리눅스에서 한다(§7).

### 2.4 GPU 없이 시뮬레이터 I/O 확인

```bash
python3 -m trainer.demo_api.app --port 8120 &          # demo 모드로 충분
python3 trainer/scripts/sim_dummy_driver.py --api-url http://127.0.0.1:8120 --events 3 --anomaly
curl -s 'http://127.0.0.1:8120/api/monitor/events?limit=20'
```

호출 1건마다 input 1건 + output 1건이 기록된다. `--events 3`이면 `(start in/out) (runtime in/out)×3 (session_end in/out)` = 10건이다.

UI 하단 모니터에서는 `세션 시작` → `런타임`(반복 가능) → `세션 끝` 버튼으로 같은 3단계를 단계별로 보낼 수 있다. 시작 전에는 나머지 두 버튼이 비활성이고, 런타임을 누를 때마다 다음 구역으로 `zone_id`를 보내므로 좌측 도장 순서가 실제로 이동한다. 상태만 따로 보려면:

```bash
curl -s http://127.0.0.1:8120/api/sim/session
curl -s http://127.0.0.1:8120/api/sequence/SP0035-L01
```

---

## 3. 아키텍처

```text
오프라인   ingest → clean → dedup → chunk → BGE-M3 임베딩 → FAISS HNSW + metadata

런타임     React/Electron · Gradio · Unity(예정)
                       │
                       ▼
              데모 API :8120
   ┌───────────────────┴───────────────────┐
   │ /api/chat (SSE)                       │ /api/sim/*  (시뮬레이터 3단계)
   ▼                                       ├─▶ MonitorRecorder (ring buffer 500)
ChatService ─ demo scenario                │        └─▶ /api/monitor/{events,stream}
            └ live runner                  └─▶ SimSessionStore ◀── 현재 상태의 소유자
                 └ AgentPipeline                     │
                      3-way router                   ├─▶ /api/sequence/{level_id}
                      RAG 검색 + rerank              ├─▶ /api/sim/session
                      Qwen3.6-27B + v6.1 LoRA        └─▶ live environment ──┐
                      ToolCallingAdapter → MCP :8100 (7 tools)              │
                            ▲                                               │
                            └───────────────────────────────────────────────┘
```

시뮬레이터가 보낸 값이 모델 컨텍스트와 좌측 패널의 유일한 출처다. 활성 세션이 없으면 [`default_snapshot`](trainer/demo_api/sequence.py)의 고정 fixture로 되돌아간다.

### 3.1 3-way Semantic Router

| 라우트 | 판단 기준 | max_rounds | 목표 |
|---|---|---|---|
| `training_realtime` | 순서·이상탐지·합부 판정 | 2 | RAG TTFT ≤30s / 도구 완료 ≤180s |
| `training_review` | 자세·추세·코칭·리포트 | 5 | 도구 완료 ≤180s |
| `fast` | 단순 지식 질문(용어·규정) | 1 | TTFT ≤30s |

- `run()`은 평가 재현성과 기존 시뮬레이터 계약을 위해 동작을 고정한다. live 동작은 `run_streaming()`이나 경계 어댑터로만 추가한다.
- `run_streaming()`은 instruction만으로 분류해 `force_route`로 넘긴다. live 턴에 항상 붙는 ambient environment를 의도로 오인해 모든 질문이 `training_realtime`으로 가던 회귀(Iter18-E)를 막기 위한 것이다.
- `fast` 경로이고 도구 호출이 0건이면 [`live_response_contract.py`](trainer/scripts/live_response_contract.py)가 5-섹션 진단 스캐폴드를 `[답변] + [근거]`로 정규화한다. 근거 없는 판정·원인·조치가 지식 질문에 붙는 것을 막는 계층이며, v7 SFT가 포맷 편향을 해소하면 제거한다.

### 3.2 MCP 도구 7개 (`:8100`)

| 도구 | 엔드포인트 | 기능 | 구현 |
|---|---|---|---|
| `dft_judge` | `/dft_judge/evaluate` | SSPC-PA2 90/10 룰 DFT 합부 판정 | 실제 로직 |
| `anomaly_detection` | `/anomaly_detection/detect` | 두께 편차 기반 결함 분류·원인 추론 | 실제 로직 |
| `sequence_validation` | `/sequence_validation/validate` | 블록/구역 우선순위 기반 다음 구역 추천 | 실제 로직 |
| `timeseries` | `/timeseries/{action}` | 속도·거리·각도 추세 분석 | Partial (in-memory) |
| `pose_estimation` | `/pose_estimation/compare` | 숙련자 기준 자세 대비 비교·교정 | Stub (기준 자세 하드코딩) |
| `location_tracking` | `/location_tracking/identify` | 3D 좌표 → Zone ID·커버리지 | Stub (fixture) |
| `vlm_analysis` | `/vlm/analyze` | 히트맵·스크린샷 VLM 분석 | Stub (빈 결과) |

모델에 노출되는 스키마는 [`trainer/configs/tool_schemas/`](trainer/configs/tool_schemas), 호출 규약·에러 코드·retry(최대 1회, backoff 1000ms)는 [`tool_calling_protocol.json`](trainer/configs/tool_calling_protocol.json)이 정본이다.

---

## 4. API 계약

경계의 JSON/SSE 봉투는 그대로 유지한다. 백엔드는 Pydantic v2(`extra="forbid"`), 프런트는 동형 Zod로 미러한다 — [`contracts.py`](trainer/demo_api/contracts.py) ↔ [`contracts.ts`](demo-ui/src/contracts.ts), [`sim_contracts.py`](trainer/demo_api/sim_contracts.py) ↔ [`simContracts.ts`](demo-ui/src/simContracts.ts).

| 메서드 | 경로 | 용도 |
|---|---|---|
| `GET` | `/api/health` | 헬스체크 |
| `GET` | `/api/sequence/{level_id}` | 도장 순서 초기 스냅샷 |
| `POST` | `/api/chat` | 채팅 SSE 스트림 |
| `POST` | `/api/sim/sessions/start` | 시뮬레이터 — 세션 시작 |
| `POST` | `/api/sim/sessions/{session_id}/events` | 시뮬레이터 — 런타임 이벤트 |
| `POST` | `/api/sim/sessions/{session_id}/complete` | 시뮬레이터 — 세션 종료 |
| `GET` | `/api/sim/session` | 현재 세션 상태 (read-only). 세션이 없으면 `{"session": null}` |
| `GET` | `/api/monitor/events?cursor=&limit=` | 모니터 스냅샷 (read-only) |
| `GET` | `/api/monitor/stream?cursor=` | 모니터 SSE (replay → live) |

**채팅 SSE 이벤트**: `session` · `status` · `message_delta` · `tool_result` · `sequence_update` · `complete` · `error`.
`complete`는 `ttft_ms`, `total_ms`, `synthesis_ms`, `tool_call_count`, `rag_chunk_count`, `deadline_hit`를 싣는다.

### 4.1 시뮬레이터 인풋 보내기

`start`(순서 안내) → `runtime`(RAG·이상탐지) → `session_end`(회고 리뷰) 3단계다. UI 버튼과 [`sim_dummy_driver.py`](trainer/scripts/sim_dummy_driver.py)(§2.4)가 아래와 똑같은 요청을 보낸다. 실제 Unity 클라이언트가 붙을 자리다.

**① 세션 시작** — `POST /api/sim/sessions/start`

```json
{"session_id": "demo-1", "level_id": "SP0035-L01",
 "worker": {"trainee_id": "t-001", "skill_level": "beginner", "handedness": "right"},
 "spec": {"nominal_um": 100.0}}
```

`skill_level`은 `beginner|intermediate|advanced`, `handedness`는 `left|right`만 받는다.
응답: `{"recommended_flow": ["A-1","A-2","A-3"], "flow_source": "rule", "guidance": "..."}`

**② 런타임** — `POST /api/sim/sessions/{session_id}/events`

```json
{"event_id": "ev-0", "zone_id": "A-2",
 "telemetry": {"gun_metrics": {"speed_m_per_s": 0.8, "standoff_mm": 300.0, "angle_deg": 72.0}}}
```

`event_id`만 필수이고 나머지는 전부 선택이다.

| 필드 | 효과 |
|---|---|
| `zone_id` | **현재 위치를 이 구역으로 이동**. 권장 순서 밖이면 이동 없이 `unresolved_zone_id`로만 기록 |
| `telemetry.coordinates` | `zone_id`가 없을 때만 MCP `location_tracking`으로 구역 해석 |
| `telemetry.gun_metrics` | `standoff_mm > 320` **또는** `angle_deg < 60`이면 응답이 `kind:"anomaly"` |
| `telemetry.pose_joints` | 기록만 된다 |
| `user_request` | 있으면 `kind:"rag"` 응답. 자세 판정보다 우선한다 |

**③ 세션 끝** — `POST /api/sim/sessions/{session_id}/complete`

```json
{"session_id": "demo-1", "scenario_id": "demo", "mode": "live",
 "zone_id": "A-2", "zone_type": "panel",
 "metrics": {"avg_dft": 92.0, "dft_values": [95.0, 102.0, 88.0],
             "avg_speed": 0.8, "speed_variance": 0.05,
             "avg_distance": 0.3, "distance_variance": 0.05,
             "avg_angle": 70.0, "angle_consistency": 0.9,
             "sequence_correct": true, "under_thickness_count": 1}}
```

`under_thickness_count`·`over_thickness_count`(기본 0), `env`, `timestamp`는 선택이다. 위험 플래그는 두께 초과/미달 외에 `distance_variance > 0.3`, `speed_variance > 0.15`, `angle_consistency < 0.7`, `sequence_correct = false`에서도 나온다([`compute_risk_flags`](trainer/scripts/session_coaching.py)).

**결과 확인**

```bash
curl -s http://127.0.0.1:8120/api/sim/session           # 현재 세션 상태
curl -s http://127.0.0.1:8120/api/sequence/SP0035-L01   # 좌측 패널이 읽는 값
curl -s 'http://127.0.0.1:8120/api/monitor/events?limit=20'
```

**주의**

- 경로의 `{session_id}`가 정본이다. 본문과 불일치하면 422.
- `start`·`runtime`은 `extra="forbid"`라 스키마에 없는 필드를 넣으면 422다: `{"error":"validation_error","detail":[{"type":"extra_forbidden",...}]}`. `complete`는 `SessionSummary`를 재사용해 아직 느슨하다.
- 422는 세션 상태를 전혀 바꾸지 않지만 **모니터에는 input/output 두 건이 그대로 남는다.** 거부된 요청도 관측된다.
- 세션 슬롯은 하나다. 새 `start`가 이전 세션을 덮어쓴다.

**현재 위치 이동**: `zone_id` → 없으면 `telemetry.coordinates` 해석 → 둘 다 없으면 위치 유지. 순서 밖 구역은 `unresolved_zone_id`로만 남는다.

**갱신 순서**: 상태 전이는 `output` 모니터 이벤트보다 **먼저** 확정된다. UI가 그 이벤트를 재조회 신호로 쓰기 때문에, 순서가 반대면 갱신 전 상태를 읽고 다시 받을 신호가 없다.

**세션이 없을 때**: `/api/sequence/{level_id}`와 live `environment` 모두 기존 fixture로 되돌아간다. 활성 세션의 `level_id`가 요청과 다를 때도 같다.

### 4.2 모니터

단일 프로세스 in-memory 링버퍼(최대 500건), 단조 증가 `seq`. `/stream`은 구독을 먼저 잡고 스냅샷을 뜬 뒤 `seq`로 중복을 제거하므로 그 사이 기록된 이벤트를 잃지 않는다. 버퍼가 넘치면 `gap` 프레임(`dropped_before`)을 먼저 보낸다. 읽기 전용이라 모니터가 세션 상태를 바꾸지 않는다.

클라이언트에 **재연결 로직이 없다.** 스트림이 끊기면(API 재기동, 네트워크 순단) 패널은 "아직 수신된 I/O가 없습니다"로 남고 스스로 복구하지 않는다. 화면을 새로고침하면 `cursor=0`으로 다시 구독해 버퍼에 남은 이벤트를 전부 리플레이한다.

---

## 5. 설정과 환경변수

| 파일 | 역할 |
|---|---|
| [`rag_config_fewshot_27b.yaml`](trainer/configs/rag_config_fewshot_27b.yaml) | **현재 기본 live/eval 설정.** Qwen3.6-27B HF 4-bit NF4, rerank top 20→8, `tool_deadline_sec`, `max_new_tokens_tool`, `prompt.fast_system` |
| [`qlora_config_27b_v6_1.yaml`](trainer/configs/qlora_config_27b_v6_1.yaml) | 27B v6.1 학습·데이터·어댑터 경로 |
| `rag_config_fewshot.yaml` / `rag_config.yaml` | 4B agentic / 4B RAG 기준선 |
| `qlora_config.yaml` | 4B QLoRA 기준선 |
| `tool_calling_protocol.json`, `tool_schemas/` | MCP 규약과 7개 도구 스키마 |
| `normalization_rules.yaml`, `metadata_schema.json` | 텍스트 정규화 규칙, 청크 메타데이터 스키마 |
| `eval_rubric.md`, `eval_rubric_agentic.md` | DocQA / Agentic 5차원 루브릭 (각 0–2점) |

산문과 설정이 다르면 **YAML/JSON이 정본**이다.

| 변수 | 기본값 | 설명 |
|---|---|---|
| `DEMO_LIVE_MODE` | `off` | 데모 API `on`=실제 AgentPipeline, `off`=시나리오. 다른 값이면 startup 실패 |
| `DEMO_MODE` | `true` | Gradio 전용. `true`=더미, `false`=실제 파이프라인 |
| `DEMO_UI_ORIGINS` | `*` | 콤마 구분 CORS origin. 미설정 시 전체 허용 |
| `CONFIG_PATH` | `/workspace/trainer/configs/rag_config_fewshot_27b.yaml` | live 파이프라인 설정 |
| `ADAPTER_PATH` | `/output/adapters/qwen36_27b_shipcoat_lora_v6_1` | live QLoRA 어댑터 |
| `MCP_BASE_URL` | `http://localhost:8100` | MCP 기본 URL |
| `VITE_API_BASE_URL` | same-origin | 프런트가 사용할 데모 API base URL |
| `HF_DEACTIVATE_ASYNC_LOAD` | `1` | Transformers v5 양자화 로딩의 일시적 VRAM 급증 방지 |
| `HF_HOME` | `/hf-cache` | HuggingFace 캐시 루트 |
| `APP_PORT` | `7860` | Gradio 포트 |

`.env`는 읽지도 커밋하지도 않는다. 런타임 모듈은 dotenv를 자동 로드하지 않는다.

---

## 6. 재현 순서 (데이터 → 학습 → 평가)

```bash
# 1) 문서 파이프라인: 179건 → 정규화 → 중복 제거 147건 → 청크 10,829건
python3 trainer/scripts/ingest_docs.py  --input-dir /data/raw       --output-dir /data/parsed --manifest /data/manifest.csv
python3 trainer/scripts/clean_docs.py   --input-dir /data/parsed    --output-dir /data/cleaned --rules trainer/configs/normalization_rules.yaml
python3 trainer/scripts/dedup_docs.py   --input-dir /data/cleaned   --output-dir /data/deduplicated
python3 trainer/scripts/chunk_docs.py   --input-dir /data/deduplicated --output-dir /data/chunked --schema trainer/configs/metadata_schema.json

# 2) 인덱스: BGE-M3(1024d) + FAISS HNSW32(IP·정규화 = cosine)
python3 trainer/scripts/build_index.py --config trainer/configs/rag_config.yaml

# 3) SFT: DocQA 생성 → FC-aware 병합 (SFT 생성 전에 RAG 기준선 평가를 먼저 돌려야 한다)
python3 trainer/scripts/generate_sft.py    --chunks /data/chunked/chunks.jsonl --output-dir /data/sft
python3 trainer/scripts/generate_fc_sft.py --existing-sft /data/sft/sft_train_docqa_only.jsonl --output-dir /data/sft

# 4) QLoRA 학습 (config-driven, 4B·27B 동일 스크립트)
python3 trainer/scripts/train_qlora.py --config trainer/configs/qlora_config_27b_v6_1.yaml

# 5) Agentic 평가 — MCP를 먼저 띄운 뒤 생성, 채점은 분리 실행
python3 trainer/scripts/generate_agentic_responses.py \
  --config trainer/configs/rag_config_fewshot_27b.yaml \
  --adapter /output/adapters/qwen36_27b_shipcoat_lora_v6_1 \
  --output /output/eval_results/agentic_responses.jsonl
python3 trainer/scripts/run_agentic_eval.py \
  --responses /output/eval_results/agentic_responses.jsonl \
  --output-dir /output/eval_results/agentic_eval \
  --tool-schemas-dir trainer/configs/tool_schemas

# 시드 반복·재현성 집계
python3 trainer/scripts/run_multiseed_eval.py --help
```

**HTTP/SSE 레이턴시 벤치** — 파이프라인 함수를 직접 부르지 않고 사용자가 실제로 보는 `/api/chat`을 측정한다.

```bash
curl -sSf -m3 http://localhost:8100/health
setsid nohup python3 -m trainer.demo_api.app --port 8130 --live > /tmp/bench_api.log 2>&1 < /dev/null &
sleep 90 && curl -sSf -m5 http://127.0.0.1:8130/api/health

python3 trainer/scripts/bench_latency.py \
  --api-url http://127.0.0.1:8130 \
  --eval-set /data/eval/eval_set_agentic.jsonl \
  --profile mixed --repeat 1 \
  --output-root /output/eval_results/latency_bench
```

실행마다 timestamp 디렉터리에 `report.json`과 `SUMMARY.md`가 남는다. 경로 판정은 사전 profile이 아니라 `complete.tool_call_count` 실측값으로 한다. terminal `error`, 봉투 파손, 필수 metric 누락, MCP `TIMEOUT` 행은 보존하되 percentile·gate에서 제외하고, 유효 행이 없으면 `0 ms pass`가 아니라 `no_data`로 보고한다.

---

## 7. 원격 배포와 운영

DGX Spark 등에 배포한 뒤 동일망 다른 PC에서 input/output/진행 로그를 모니터링하는 구성을 전제로 한다.

**API 주소 결정 우선순위**

1. `window.__DEMO_API_BASE_URL__` — 빌드 시 `VITE_API_BASE_URL`이 `index.html`에 치환된다. 배포된 `dist/index.html`의 값을 직접 고치면 **재빌드 없이** 주소를 바꿀 수 있다.
2. 빌드 시 `VITE_API_BASE_URL` (번들 내장).
3. 미지정 → same-origin. **Electron은 `file://`이라 same-origin fallback이 없다. 1 또는 2를 반드시 지정한다.**

**Windows에서 Electron만 띄우기**

```powershell
# 저장소를 받은 뒤
cd demo-ui
npm ci
# demo-ui\.env.local 한 줄:  VITE_API_BASE_URL=http://<서버IP>:8120
npm run start:electron
```

SSH 터널을 쓰면 `ssh -L 8120:127.0.0.1:8120 <user>@<host>` 후 주소는 `http://127.0.0.1:8120`이다.
`DEMO_UI_ORIGINS`를 좁히면 Electron(Origin `null`)이 차단될 수 있으니 기본값을 유지하거나 프록시를 둔다.

**Vite 개발 서버 노출** — `dev:web`/`preview:web`은 `--host 127.0.0.1` 고정이다. 외부에 열려면 명시적으로 덮어쓴다(신뢰된 망에서만).

```bash
cd demo-ui && npm run dev:web -- --host 0.0.0.0
```

**nginx 리버스 프록시** — 이 구성이면 프런트가 same-origin이라 CORS·base URL 설정이 필요 없다.

```nginx
location /api/ {
    proxy_pass http://127.0.0.1:8120/api/;
    proxy_http_version 1.1;
    proxy_set_header Connection "";   # SSE 유지
    proxy_buffering off;              # 델타가 뭉치지 않게
}
location / { root /var/www/demo-ui/dist; try_files $uri /index.html; }
```

**증상별 조치**

| 증상 | 원인 | 조치 |
|---|---|---|
| 상태가 계속 `오프라인` | API 주소 미지정 또는 도달 불가 | `VITE_API_BASE_URL` 확인 후 재빌드, 포트 게시·터널 확인 |
| 질문 후 응답이 오래 없음 | live 27B 로딩/추론 진행 중 | 콘솔의 경과 타이머와 진행 단계, 서버 로그의 라우터·도구 라인 확인 |
| `live` 전환 시 error | 서버가 `DEMO_LIVE_MODE=off` | `--live`로 재기동 |
| 브라우저는 되는데 Electron만 실패 | `file://`에 same-origin fallback 없음 | `VITE_API_BASE_URL` 지정 후 재빌드 |
| 컨테이너에서 Electron 창이 안 뜸 | 디스플레이 없음 | 데스크톱 OS에서 실행하거나 `xvfb-run` |
| SSE가 끊기거나 몰아서 도착 | 프록시 버퍼링 | `proxy_buffering off`, `Connection ""` |

---

## 8. 검증

```bash
# Python (컨테이너)
docker exec -w /workspace ship-paint-trainer python3 -m pytest trainer/demo_api/tests trainer/scripts/tests -q
docker exec -w /workspace ship-paint-trainer python3 mcp_servers/test_integration.py

# 웹/Electron (demo-ui)
npm test            # Vitest
npm run build:web   # tsc -b + vite build — 타입 게이트를 겸한다
npm run test:e2e    # Playwright (데모 API 8120 + Vite 5173 자동 기동)
```

2026-07-28 기준 실측: Python 115 passed, MCP 통합 15 tests OK, Vitest 79 passed / 15 files.
`basedpyright`와 `ruff`는 현재 컨테이너에 설치돼 있지 않다. 타입 게이트는 프런트가 `tsc -b`, 백엔드가 pytest다.
SSE 라우트는 종료되지 않으므로 Starlette `TestClient`로 테스트하지 않는다. 순수 async generator(`monitor_sse_body`)를 anyio로 직접 검증하거나 실제 uvicorn을 띄운다.

---

## 9. 하드웨어와 의존성

| 구분 | 사양 |
|---|---|
| 27B 학습·평가 | NVIDIA A40 48GB 이상 — v6.1 QLoRA peak 34.87 GB |
| 4B 추론·XR 기준선 | RTX 4080 16GB — 추론 ~4 GB / 학습 ~8.5 GB |
| RAM / 저장 | 32 GB 이상 / 4B 약 8 GB 또는 27B 약 52 GB + 데이터·인덱스 |
| CUDA | 12.4 이상 |

RTX 4080 VRAM 예산: LLM+어댑터 ~4.5 GB, BGE-M3+FAISS ~2.0 GB, MCP ~1.5 GB, XR 시뮬레이터 여유 ~8.0 GB.

주요 버전: torch 2.4.0+cu124 · transformers 5.3.0 · peft 0.18.1 · trl 0.29.0 · bitsandbytes 0.49.2 · sentence-transformers 5.2.3 · faiss-cpu 1.13.2 · fastapi 0.135.1 · gradio 6.9.0 · PyMuPDF 1.27.1 · httpx 0.28.1.
프런트: React 19 · Zod 4 · Vite 8 · TypeScript 7 · Vitest 4 · Electron 41.

**의존성 제약** — RTX 4080 경로는 HF 4-bit 백엔드를 유지한다. 다음을 필수 의존성으로 추가하지 않는다.

- `unsloth` — transformers/trl를 다운그레이드시켜 롤백됨
- `FlagEmbedding` — transformers 5.3.0에서 `is_torch_fx_available` 제거로 동작 불가. BGE-M3는 sentence-transformers로 로드
- `vLLM` — 16GB VRAM에서 OOM. `agent_pipeline.py`의 optional import로만 남음

---

## 10. 지표

| 지표 | 값 | 단서 |
|---|---|---|
| Agentic (Qwen3.6-27B v6.1) | 평균 **6.744**, sample SD 0.139 | seeds 42/43/44. **latency token-cap 변경 전** 값이며 변경 후 재평가·n=5·CI는 미실행 |
| Agentic (Qwen3-4B) | 4.97 (Iter14) · 5.57 (Iter17 rescore, Oracle 검증 후 5.30) | 목표 4.0 달성 |
| RAG+QLoRA DocQA | 1.368 / 2.0 | Iter4 4B |
| Hallucination | 2.0% | |
| Tool calling | 30문항에서 86회 | Iter14 |
| Rerank recall@8 | 0.66 → 0.74 | 공백 복원 + 재임베딩 |
| 레이턴시 gate | RAG TTFT p90 ≤30s / 도구 p99 ≤180s / 품질 ≥6.60 | **셋 다 실측 전** |

과거 A40 참고치(fast 9.9s / realtime 55.5s / review 85.3s)는 deadline guard·token cap·SSE timing 도입 **이전** 측정이다. 현재 gate 달성값으로 읽으면 안 된다. LLM 추론이 전체 레이턴시의 약 95%이고 MCP 도구 호출은 1ms 미만이다.

이터레이션 히스토리(Iter0~Iter18-E)의 전체 표·근거·검증 로그는 [`PATCH_NOTE.md`](PATCH_NOTE.md)에 있다.

---

## 11. 문서 맵

| 문서 | 내용 |
|---|---|
| [`AGENTS.md`](AGENTS.md) | 저장소 규약 — 실행 위치, 경계 계약, 커밋 규칙 |
| [`PATCH_NOTE.md`](PATCH_NOTE.md) | Iter0~Iter18-E 전체 변경·검증 기록 |
| `AgenticCopilot_Architecture.md` | 아키텍처 설계서. gitignore 대상이라 저장소에는 없고 작업 환경에만 존재한다 |
| [`docs/superpowers/specs/`](docs/superpowers/specs) | 승인된 설계 — Electron 데모, 레이턴시 예산, 시뮬레이터 I/O 모니터 |
| [`docs/superpowers/plans/`](docs/superpowers/plans) | 대응 TDD 구현 계획 |
| [`docs/session_data_schema.md`](docs/session_data_schema.md), [`docs/session_schema.json`](docs/session_schema.json) | 시뮬레이터 세션 데이터 스키마 |
| `.taskmaster/` | 작업 DB. Task Master 명령으로만 갱신하고 손으로 고치지 않는다 |

라이선스: Qwen3-4B Apache 2.0 / BGE-M3·Reranker MIT / Qwen3.6-27B는 배포 전 사용 artifact의 model card를 별도 확인한다. 본 PoC 코드는 내부 프로젝트다.
