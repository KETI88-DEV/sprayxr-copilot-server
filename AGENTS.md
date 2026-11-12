# AGENTS.md

Ship-coating training copilot: a Python 3.11 RAG/agent pipeline, seven FastAPI MCP tool routers, a FastAPI SSE demo API, and a separate React 19 + Electron package in `demo-ui/`. `demo-ui/` is the only real npm package; root-level personal tooling manifests are intentionally untracked. Production uses the Qwen3.6-27B v6.1 LoRA-merged BF16 model through vLLM and a separate Qwen3-8B-AWQ quick model. HF NF4 remains the training/evaluation backend; 4B configs remain for baseline reproduction.

`README.md` covers user-facing setup. This file covers what an agent would otherwise get wrong.

## Runtime: run Python in the container, never on the host

Host `python3` is 3.12 with **no project dependencies** (`import fastapi` fails). The provisioned runtime is the `ship-paint-trainer` container (Python 3.11.9, fastapi 0.135.1, pydantic 2.12.5) with the repo mounted at `/workspace`.

```bash
docker compose -f deploy/docker-compose.yml up -d
docker exec -w /workspace ship-paint-trainer python3 -m pytest trainer mcp_servers -q
```

- Always run from the repo root so `trainer.*` / `mcp_servers.*` imports resolve. There is no Python manifest or lockfile — do not invent a dependency workflow.
- Exception: `deploy/spark/requirements/*.lock.txt` pin only the Spark (aarch64) deployment images, copied from this container's installed versions; they do not change the development workflow above.
- Packages missing after a container rebuild: `pip install -r /workspace/trainer/requirements-runtime.txt` (reconstructed list; the image Dockerfile lacks them).
- `deploy/docker-compose.yml` publishes **only `8120`** and injects live defaults (`DEMO_LIVE_MODE=on`, 27B `CONFIG_PATH`/`ADAPTER_PATH`, `MCP_BASE_URL`). So `create_app()` inside the container defaults to **live**, and MCP `:8100` is reachable only from inside the container.
- `--live` startup loads a 27B model, so `/api/health` stays down for a while. That is not a hang.

## Commands

```bash
# services — inside the container, from the repo root
uvicorn mcp_servers.main:app --host 0.0.0.0 --port 8100
python3 -m trainer.demo_api.app --port 8120          # demo mode, no GPU
python3 -m trainer.demo_api.app --port 8120 --live   # live mode, GPU
DEMO_MODE=true|false python3 trainer/scripts/demo_app.py   # Gradio, separate UI entrypoint

# full Python suite — 1366 tests, ~263 s (2026-09-29 실측; 2026-08-28에는 410개·17초였다)
docker exec -w /workspace ship-paint-trainer python3 -m pytest trainer mcp_servers -q

# GPU-free smoke of the simulator interface + monitor
python3 trainer/scripts/sim_dummy_driver.py --api-url http://127.0.0.1:8120 --events 3 --anomaly
curl -s 'http://127.0.0.1:8120/api/monitor/events?limit=20'   # expect 10 events for --events 3

# frontend — on the host, inside demo-ui/
npm test            # vitest run
npm run build:web   # tsc -b + vite build   <- the type gate
npm run test:e2e    # playwright
npm run start:electron
```

`basedpyright`, `ruff`, and `task-master` are **not installed** on the host or in the container. Do not put them in a verification plan. Task Master is wired as the `task-master-ai` MCP server in `.mcp.json`.

## Verification gates that actually exist

| Change | Gate |
|---|---|
| Python | `pytest trainer mcp_servers -q` in the container |
| TypeScript/React | `npm run build:web`, then `npm test` |
| UI behavior | `npm run test:e2e` |
| Live pipeline behavior | run the real API and drive `/api/chat`; no unit test loads the model |

`demo-ui/tsconfig.json` is `strict` plus `noUncheckedIndexedAccess` and `exactOptionalPropertyTypes`, and includes **both `src` and `tests`** — test files are type-gated too. `pyrightconfig.json` includes only `trainer`, so changed `mcp_servers` files get no static check at all; cover them with tests.

## Where to look

| Task | Location | Notes |
|---|---|---|
| Agent routing / RAG / tool loop | `trainer/scripts/agent_pipeline.py` | `run()` is evaluation-compatible (`use_rag=False` skips retrieval; default unchanged). `run_streaming()` classifies the route from the instruction alone (environment ignored), passes it to `run()` as `force_route` in a worker thread under the shared generation slot, then re-chunks the finished answer — no token streaming yet. A consumer that leaves cancels the openai backend between tokens |
| OpenAI-compatible backend | `trainer/scripts/openai_backend.py`, `trainer/configs/rag_config_spark_vllm.yaml` | `llm.backend: openai`: streamed `/v1/chat/completions`, never `tools`, explicit sampling + `enable_thinking` per request. The in-process `VLLMWrapper` (`backend: vllm`) is known-broken |
| Prompt token budget | `trainer/scripts/prompt_budget.py` | Exact count through the chat template when a tokenizer is loaded (`llm.tokenizer_path` for remote backends); `llm.max_model_len` must equal vLLM `--max-model-len` |
| v2.5 공개 API 15개 | `trainer/demo_api/v1_router.py`, `v1_contracts.py`, `v1_store.py`, `anomaly_stream.py` | 정본은 `docs/XR Painting Simulator AI Co-Pilot Service API 정의서 - 대화형 API 명세서.pdf`(76쪽). 세션·사용자·리포트·이상 알림 SSE·테스트 12개가 여기, 시퀀스 2종은 `telemetry_router`, 질의는 `copilot_router`. 계약 밖 경로는 `include_in_schema=False`로 `/docs`에서 감춘다(라우트는 demo-ui·헬스체크용으로 남음). 이상 알림 `anomaly_type`은 넷이다(모의 `/test/sequence/*`의 `distance_deviation` 제외): `sequence_order_deviation`, `direction_deviation`, `coating_over_app`, `coating_under_app`(도막 둘은 서버가 떠 있는 동안 세션마다 Zone당 종류별 1회, 저도막의 `zone_id`는 방금 떠난 Zone). 분석 엔진이 없는 영역은 `unavailable`이고 수치는 `null` — 임의 값을 만들지 않는다 |
| v2.5 자세 채점 (`pose_recomm`) | `trainer/demo_api/pose_query.py` | `POST /api/v1/copilot/query` 의 `task_type=pose_recomm` 이 **저장된 시퀀스**를 채점한다(명세서 p.9). 대상은 `context`(예: `SEQ-012`) 또는 세션의 최근 시퀀스. `telemetry/sequence` 는 올리기만 한다(p.4) — 자세를 거기서 채점하지 않는다. 대상을 못 찾으면 200 + 안정적 `reason_code`. **구역 질문**(`context`가 구역 번호이거나 질문에 "구역·어디·어느")은 채점하지 않고 complete 가 리포트 DB에 남긴 구역별 결과로 답한다(`answer_zone_question`) — 가장 문제 있는 구역 = 자세 평균 유사도가 가장 낮은 구역 |
| 세션 종합 분석 (complete) | `trainer/demo_api/session_analysis.py`, `session_coating.py`, `v1_router.py` | complete 의 CSV·DB 저장 시퀀스로 자세를 구역별로 집계하고(시퀀스 = 착탄이 가장 많이 난 구역), 높이맵(없으면 시퀀스 히트맵 크롭)을 TSB 라벨맵 구역마다 목표 두께 대비 등급(`coating_assessment`)으로 판정한다. 판정한 리포트(`analysis_status="available"`)는 `comparison_basis="session_target"`이고 전문가 대비 판정은 리포트에 없다. 판정하지 못한 리포트는 `comparison_basis`가 기본값 `"expert_reference"`로 남고 `target_*` 값은 `null`이다. 판정할 Zone 목록은 여전히 전문가 프로필에서 얻는다. 전문가 프로필(`EXPERT_REFERENCE_ROOT`)은 Spark 에만 있어 개발 서버에서는 도막이 `reference_unavailable` 이다 |
| v2.5 telemetry sequence | `trainer/demo_api/telemetry_router.py`, `telemetry_contracts.py`, `sequence_store.py` | `POST /api/v1/telemetry/sequence` (정본) + `/api/v1/copilot/analyze-sequence` (벤더가 같은 본문을 두 경로로 보내므로 멱등 별칭). 최상위 `forbid`, 프레임 하위 `allow`+보고. 자세는 여기서 채점하지 않는다. analyze-sequence 는 `coating_feedback`(현재 Zone의 도막 등급·분포·조절 후보)을 채우고, 저장 전용 `telemetry/sequence` 만 `not_available`(`coating_engine_pending_integration`)이다. 엔진 장애도 200이다. 계약 §7.4 |
| 도막 판정 어댑터 | `trainer/demo_api/coating_assessment.py` | PR #8 `trainer/zone_engine/numerical.py`·`behavior.py`를 프로세스 안에서 부르는 유일한 곳(PR #8 서비스 8122·8123·`/api/v2`는 켜지 않는다). 실시간 표면은 크롭 밖을 "미도장"으로 본다(PR #8 엔진은 "모름"). PR #8과 다른 점은 모듈 독스트링 목록이 정본이다. 등급·6구간 경계 수의 사본은 표시용 `DISPLAY_BAND_FACTORS` 한 줄뿐이다. 실시간 크롭 복호·누적 교체는 `coating_progress`에 있는 사본이고, `test_live_crop_path_matches_the_engine_cumulative_crop_path`가 PR #8 경로와 값이 같은지 지킨다 |
| v2.5 copilot query | `trainer/demo_api/copilot_router.py`, `copilot_contracts.py` | `POST /api/v1/copilot/query`: only `knowledge_search` (and knowledge-routed prompts) runs the generation pipeline. `error_analysis` answers from the saved complete report's coating rows without the model or MCP (`not_available` when no report). `pose_recomm` scores stored sequences through MCP, and `trajectory_recomm` answers from the complete report when one exists. `/api/v1` validation errors use the v2.5 ERROR wrapper. Readiness: `GET /api/ready` (`readiness.py`) |
| 외국어 질의·용어집 | `trainer/demo_api/query_language.py`, `trainer/multilingual/glossary.py`, `glossary.json` | 요청의 `language`·`response_language`로 질문을 한국어로 옮겨 기존 분석을 돌리고 `answer`만 번역한다. 용어집은 핵심 5개 용어가 현지 1차 문서로 확인된 7개 언어(우즈벡어는 건조도막두께 하나)에만 적용한다(`docs/local_voice_multilingual_terminology.md` §6). 근거 없는 대역어를 용어집에 넣지 않는다. 용어는 띄어쓰기·대소문자·ё/е를 무시하고 찾으며, 번역에 대역어가 빠지면 503이다. 번역 호출은 greedy·반복 벌점 1.0이다(HF는 한자 차단도 이 호출에서만 푼다) |
| LoRA merge / FP8 export | `trainer/scripts/merge_lora_shards.py`, `quantize_fp8_block.py` | Shard-level BF16 merge with manifest; FP8 block-128 in the official Qwen3.6-FP8 layout |
| Fast-path answer contract | `trainer/scripts/live_response_contract.py` | Normalizes the 5-section diagnostic scaffold to `[답변]/[근거]` when route is `fast` and tool count is 0 |
| Model-to-MCP calls | `trainer/scripts/tool_calling_adapter.py` | Protocol from `trainer/configs/tool_calling_protocol.json` |
| Demo HTTP/SSE API | `trainer/demo_api/app.py`, `chat_service.py` | Demo/live selection, lazy pipeline singleton |
| Simulator interface + monitor | `trainer/demo_api/simulator.py`, `monitor.py`, `monitor_api.py` | Responses are still a deterministic stub — `_run_phase(build_output=...)` is the seam for a real `AgentPipeline`. Every call records one input + one output event, including on 422. **State transitions commit before the output event is published**; reversing that order lets subscribers read pre-update state with no second refresh signal |
| Current work state (who owns it) | `trainer/demo_api/sim_session.py` | Single-slot in-memory `SimSessionStore` is the only owner. `/api/sequence/{level_id}`, `/api/sim/session`, and the live chat `environment` all read from it and fall back to `default_snapshot` / `default_tool_environment` when no session matches the level. A 422 must leave it untouched |
| Zone resolution fallback | `trainer/demo_api/zone_resolver.py` | Used only when a runtime event carries no `zone_id`. Every failure returns `None` — never raises, never required by tests. MCP answers errors with HTTP 200, so the `status` field decides success |
| API event contracts | `contracts.py` ↔ `demo-ui/src/contracts.ts`, `sim_contracts.py` ↔ `simContracts.ts` | Mirrored pairs — change both sides together |
| UI HTTP/SSE client | `demo-ui/src/api.ts` | Base URL: `window.__DEMO_API_BASE_URL__`, then Vite env, then same-origin (absent under Electron's `file://`) |
| MCP routes/envelopes | `mcp_servers/main.py`, `common.py` | `/health` `servers: 7` = mounted routers, not processes |
| Current 27B inference / training | `trainer/configs/rag_config_fewshot_27b.yaml`, `qlora_config_27b_v6_1.yaml` | Tool deadline, per-round token caps, `prompt.fast_system` |
| Spark release package | `deploy/spark/` (`compose.yaml`, `README.md`) | vllm / mcp / agent-api / web images plus a sha256-manifested asset bundle; `trainer/scripts/tests/test_spark_deploy_contract.py` ties compose values to `rag_config_spark_vllm.yaml`. Do not build these images on the x86 host |
| Latency contract and bench | `docs/superpowers/specs/2026-07-25-latency-budget-live-demo-design.md`, `trainer/scripts/bench_latency.py` | SSE timing semantics; `bench_report.py` is pure aggregation |

## Conventions

- Preserve exact JSON/SSE/tool envelopes at boundaries. Parse untrusted payloads in Pydantic (`extra="forbid"`) / Zod before business logic.
- Keep `AgentPipeline.run()` behavior stable for evaluation. Add live behavior through `run_streaming()` or boundary adapters.
- `DEMO_MODE` controls Gradio only. `DEMO_LIVE_MODE` controls the FastAPI chat wiring (`--live` sets it); any other value fails startup by design.
- Treat YAML/JSON config as source of truth when prose disagrees. `/models`, `/data`, `/output`, `/hf-cache`, `/logs` are mounted artifacts, not repository data.
- Keep the HF 4-bit backend. Unsloth, FlagEmbedding, and vLLM must not become required dependencies. Production inference uses `llm.backend: openai`, which calls an external vLLM OpenAI-compatible server over HTTP (httpx; no `vllm` package). HF NF4 stays the default for training and evaluation.
- `.taskmaster/tasks/tasks.json` is tool-managed: use Task Master, then `task-master generate`. Never hand-edit.
- Never read, print, or commit `.env`. Runtime modules do not auto-load dotenv files.
- Electron keeps `contextIsolation: true`, `nodeIntegration: false`, `sandbox: true`; preload exposes only platform/version metadata. Preserve that boundary.

## Testing gotchas

- **Never drive a non-terminating SSE route through Starlette `TestClient`.** It runs the whole ASGI call to completion first, so `client.stream()` hangs with no output. Test the pure async generator (e.g. `monitor_sse_body`) directly with anyio, or run a real uvicorn.
- `npm run test:e2e` starts its backend with `python3 -m trainer.demo_api.app` from the repo root, which cannot work on the host. It passes only because `reuseExistingServer: true` finds the container's published `:8120`. Keep the container API up before running e2e.
- Google Chrome is not installed (`/opt/google/chrome/chrome` is absent), so browser tooling pinned to channel `chrome` fails. Use Playwright's bundled Chromium via `npm run test:e2e` or a script importing `@playwright/test`.
- Do not wait for `networkidle` in browser automation against this app — the SSE connection never idles. Use `domcontentloaded`.
- pytest collects both `trainer/scripts/tests/` and the loose `trainer/scripts/test_*.py`; both are real suites.
- `trainer/demo_api/scenario_adapter.py` dynamically imports `trainer.scripts.demo_app`, so demo fixtures are shared by Gradio and FastAPI demo mode.

## Version control

- When a coherent, independently describable update is complete and its relevant checks pass, create an atomic commit automatically. Do not commit incomplete or failing work.
- Before committing, inspect status, diff, and recent history. Stage only files belonging to the current update; never include unrelated concurrent changes, local artifacts, secrets, or generated outputs.
- Use Conventional Commits: `<type>(<scope>): <concise Korean summary>`. Scope is optional; keep one purpose per commit. Allowed types: `feat`, `fix`, `docs`, `test`, `refactor`, `chore`, `build`, `ci`.
- This worktree is shared with the user and other agents. Stage explicit paths — never `git add -A`. Unfamiliar modified files are someone else's in-progress work; do not revert them.
- Verify remote branch heads and SSH credentials before pushing. Local verification and commits do not prove remote delivery.

## Issue tracking

- 관련 기능 수정 요청과 재현한 오류를 의미 있는 작업 단위로 모아 GitHub Issues에 기록한다. 매 질문마다 별도 Issue를 만들지는 않는다.
- 현상, 재현 절차, 기대 동작, 영향과 완료 기준을 기록하고 실제 확인 사실에 맞춰 작성한다.
- `.github/issue-batches/`의 JSON을 `main`에 반영하면 Issue가 등록·갱신된다. 같은 `key`는 기존 Issue를 재사용한다.
- 운영 검증을 마치면 결과와 릴리스 버전을 기록하고 해당 배치의 `state`를 `closed`로 바꾼다.

## Naming conventions

- Level ID: **`TR-L01`** (Training Level 01). `SP0035`, `SP0035-L01`, `SP0035 Level 01`은 합의되지 않은 자산 경로 기반 식별자다. 코드·문서·응답 어디에도 쓰지 않는다.
- Zone ID: **레벨마다 1부터 매긴 번호 문자열** — `"1"`, `"2"`, … (명세서·실물 기준 `"1"`~`"39"`, 미판정은 `"-1"`).
  레벨 안에서만 유일하다. 같은 `"3"`이라도 레벨이 다르면 다른 구역이므로 `scenario_id`와 함께 봐야 한다.
  `A-1`·`A-2`·`A-3`은 2026-10 이전 표기다. 새 코드·문서·응답에 쓰지 않는다.

## Local-only files

`AgenticCopilot_Architecture.md`, `dataset.md`, `AgenticCopilot.png`, `.taskmaster/config.json`, `.claude/`, `.omc/`, and the `models/ data/ output/ hf-cache/` trees are gitignored. They exist in this worktree but not in the repository — do not link them from tracked docs and do not assume a fresh clone has them.

`CLAUDE.md.back` is a stale Iter14-era instruction file (it still calls Qwen3-4B current). This file supersedes it; do not mine it for guidance.

## Data, training, and evaluation order

1. `ingest_docs.py` → `clean_docs.py` → `dedup_docs.py` → `chunk_docs.py` → `build_index.py`.
2. Run baseline RAG evaluation before SFT generation; `generate_sft.py` consumes its raw response artifact.
3. Generate document QA SFT, merge function-calling SFT, then run `train_qlora.py` with the intended QLoRA config.
4. Start MCP before live agentic generation; score the generated JSONL separately with `run_agentic_eval.py`.

## Query dialogue and operator provenance

- Assistant operational QA always reuses `COPILOT-AI-TEST`, with display name `정성오`, and sends `X-Copilot-Source: assistant-test`. Do not use `1234` or the shared `COPILOT-DEMO-USER` for assistant QA. Do not create or rename a trainee per test run; use request/conversation IDs for isolation. This fixed identity and display name were requested by the user on 2026-11-11; the account ID and caller source distinguish assistant QA from TSB calls.
- The public simulation/test workflow keeps `COPILOT-DEMO-USER` (`모의훈련자`). Do not rename that shared account or historical users to claim their traffic as assistant testing.
- Motion viewer requests keep the selected worker ID and send `X-Copilot-Source: motion-viewer`. Caller channel (`3D VIEWER`, `TEST`, `TEST API`) is separate from the trainee identity.
- Do not rename historical worker `1234` or attribute its old calls to a person without evidence. Missing historical IPs stay unknown; file imports have no HTTP caller.
- IP comes from ASGI client scope. Never trust arbitrary forwarded headers or widen trusted proxies to `*` merely to populate the UI.
- Query dialogue uses persisted history scoped by user, training session and optional `X-Copilot-Conversation`. All Query `AgentPipeline.run` calls disable pipeline history; its unscoped memory is not a second source of conversation context.
- Viewer dialogue facts require the server signature. Selected Zone representative statistics are not instantaneous measurements.

## Zone completion and reference planning

- Spray advancement requires95% of official Zone pixels meeting the session target(default160µm). Crop-only mean/expert-cell coverage is not proof of whole-Zone completion. Reuse the same gate for sequence feedback, SSE, current-next queries and explicit-target reports.
- 두께는 PR #8 보정값으로 잰다: 10 µm 한 번 차감, 1600 µm 상한, 1 µm 미만은 미도장. 세션이 목표를 생략하면 160µm로 판정하고 `target_basis="default_target"`으로 드러낸다(생략 여부는 `sessions.target_basis`에 남는다). 등급(칠한 곳 평균: 목표 미만 `below`, 목표~1.5배 미만 `normal`, 1.5배 이상 `above`)은 95% 게이트와 별개 값이고 `coating_feedback`·리포트 행에 싣는다. `overpaint_risk`는 등급 `above`다. 규칙을 다시 구현하지 말고 `trainer/demo_api/coating_assessment.py`만 부른다. `trainer/zone_engine/`은 고치지 않는다.
- Registered reference-order planning does not require trainee sequences. Keep `reference_only` separate from actual visits/scores/current progress and preserve signed viewer comparisons.
- Public feedback submissions create only feedback records; processing-state writes use the existing administrator session and CSRF checks.
- The user explicitly authorized TSB corpus ingestion on 2026-11-11, superseding the earlier deferral. Preserve Windows originals and use only the original Spark (192.168.0.20). Import archives under expert_reference_data/imports/tsb-20261111 and publish record JSON through trainer.scripts.ingest_tsb_corpus; valid Zone data takes precedence per process/session. Unmapped records remain reviewable data and do not silently replace reference policies or imply LLM weight training.
