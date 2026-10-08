# AGENTS.md

Ship-coating training copilot: a Python 3.11 RAG/agent pipeline, seven FastAPI MCP tool routers, a FastAPI SSE demo API, and a separate React 19 + Electron package in `demo-ui/`. `demo-ui/` is the only real npm package; root-level personal tooling manifests are intentionally untracked. Current model path is Qwen3.6-27B 4-bit NF4 + v6.1 LoRA; 4B configs remain for baseline reproduction.

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
| v2.5 공개 API 15개 | `trainer/demo_api/v1_router.py`, `v1_contracts.py`, `v1_store.py`, `anomaly_stream.py` | 정본은 `docs/XR Painting Simulator AI Co-Pilot Service API 정의서 - 대화형 API 명세서.pdf`(76쪽). 세션·사용자·리포트·이상 알림 SSE·테스트 12개가 여기, 시퀀스 2종은 `telemetry_router`, 질의는 `copilot_router`. 계약 밖 경로는 `include_in_schema=False`로 `/docs`에서 감춘다(라우트는 demo-ui·헬스체크용으로 남음). 분석 엔진이 없는 영역은 `unavailable`이고 수치는 `null` — 임의 값을 만들지 않는다 |
| v2.5 자세 채점 (`pose_recomm`) | `trainer/demo_api/pose_query.py` | `POST /api/v1/copilot/query` 의 `task_type=pose_recomm` 이 **저장된 시퀀스**를 채점한다(명세서 p.9). 대상은 `context`(예: `SEQ-012`) 또는 세션의 최근 시퀀스. `telemetry/sequence` 는 올리기만 한다(p.4) — 자세를 거기서 채점하지 않는다. 대상을 못 찾으면 200 + 안정적 `reason_code` |
| v2.5 telemetry sequence | `trainer/demo_api/telemetry_router.py`, `telemetry_contracts.py`, `sequence_store.py` | `POST /api/v1/telemetry/sequence` (정본) + `/api/v1/copilot/analyze-sequence` (벤더가 같은 본문을 두 경로로 보내므로 멱등 별칭). 최상위 `forbid`, 프레임 하위 `allow`+보고. 자세만 MCP로 위임하고 coating·sequence는 `not_available`. 엔진 장애도 200이다. 계약 §7.4 |
| v2.5 copilot query | `trainer/demo_api/copilot_router.py`, `copilot_contracts.py` | `POST /api/v1/copilot/query`: only `knowledge_search` runs the pipeline (fast route); other task types answer 200 `not_available` without MCP. `/api/v1` validation errors use the v2.5 ERROR wrapper. Readiness: `GET /api/ready` (`readiness.py`) |
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
- `main` is far ahead of `origin/main` and pushes need repaired GitHub credentials. Local verification and commits do not prove remote delivery.

## Naming conventions

- Level ID: **`TR-L01`** (Training Level 01). `SP0035`, `SP0035-L01`, `SP0035 Level 01`은 합의되지 않은 자산 경로 기반 식별자다. 코드·문서·응답 어디에도 쓰지 않는다.
- Zone ID: `A-1`, `A-2`, `A-3` (현재 SP0035 리그 기준). 새 리그가 추가되면 이 파일을 갱신한다.

## Local-only files

`AgenticCopilot_Architecture.md`, `dataset.md`, `AgenticCopilot.png`, `.taskmaster/config.json`, `.claude/`, `.omc/`, and the `models/ data/ output/ hf-cache/` trees are gitignored. They exist in this worktree but not in the repository — do not link them from tracked docs and do not assume a fresh clone has them.

`CLAUDE.md.back` is a stale Iter14-era instruction file (it still calls Qwen3-4B current). This file supersedes it; do not mine it for guidance.

## Data, training, and evaluation order

1. `ingest_docs.py` → `clean_docs.py` → `dedup_docs.py` → `chunk_docs.py` → `build_index.py`.
2. Run baseline RAG evaluation before SFT generation; `generate_sft.py` consumes its raw response artifact.
3. Generate document QA SFT, merge function-calling SFT, then run `train_qlora.py` with the intended QLoRA config.
4. Start MCP before live agentic generation; score the generated JSONL separately with `run_agentic_eval.py`.
