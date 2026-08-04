# Findings

## Restored Original Scope
- Item 1: `git push origin main`.
- Item 2: v7 SFT development for tool-call format-emission failures that could not be solved by decoding/router/parser tuning.
- Item 3: extend multi-seed evidence with exact-sequence rate, parser rejection, and hashes.
- Item 4: run five seeds and target CI ±0.10.
- Item 5: actual Live-mode load test and latency remediation.

## Frozen Baselines
- v6.1 contamination-clean training: 636 train / 29 validation examples; eval loss 0.65 → 0.50 → 0.47; peak 34.87 GB; 216 minutes.
- v6.1 3-seed agentic score: 6.744 ± 0.139; `tool_not_called` in 17-18 of 30 items.
- Best ablation: `pass_native_tools=false`, `do_sample=true`.
- RAG-only baseline: p50 31.4 s, p90 59.1 s, max 93.0 s; 87/154 exceed 30-second TTFT target.
- Tool-path baseline: p50 124.6 s, p90 158.3 s, max 191.2 s; 4/86 exceed 180-second completion target.
- Effective generation throughput is about 4.6 tokens/s; logs report Flash Linear Attention and causal-conv1d torch fallbacks.

## Repository State at Resume
- Branch: `main`, upstream `origin/main`, 22 commits ahead before new work.
- Existing latency design commit: `05f387b`.
- Existing latency implementation plan commit: `e872af1`.
- Interrupted Task 1 left modifications in `trainer/scripts/tool_calling_adapter.py` and `trainer/scripts/test_tool_calling.py`; they require independent inspection and verification.

## Path/Patch Note Rule
- No local convention literally named `path_note` exists.
- The repository's authoritative iteration record is `/workspace/PATCH_NOTE.md`; README links it as the detailed Iter18 history, and prior work already records the v7 target there.
- This work therefore treats the user's `path_note` request as the project `PATCH_NOTE.md` rule.
- Detailed file paths, commands, hashes, metrics, errors, and decisions are mirrored in `/workspace/findings.md` and `/workspace/progress.md` after each phase.

## Recovered v7 Requirements
- Reuse existing generated data assets: CoT SFT 80 samples, tool-selection SFT 40 samples, and medium SFT 30 samples.
- Explicitly target the eight zero-tool-call questions: `AQ004`, `AQ007`, `AQ010`, `AQ014`, `AQ019`, `AQ020`, `AQ022`, `AQ024`.
- Teach `<tool_call>{...}</tool_call>` emission before prose and include hard multi-tool exemplars.
- Preserve the proven runtime contract: native tools off, tool thinking off.
- v7 is a data/config change; current parser/runtime behavior is not the training root cause.
- Required acceptance evidence includes exact sample counts, decontamination, formatting order, v7 input/config/output hashes, `tool_not_called <= 8/30`, and overall agentic score >= 6.60.

## Current Evaluation Artifact Gaps
- Existing chosen v6.1 seeds 42/43/44 score 6.7000, 6.9000, and 6.6333; mean 6.7444, std 0.1388, 95% t-CI ±0.3448.
- Missing before item 4: seeds 45/46, per-question call/no-call matrix, exact-sequence rate, parser rejection counts/reasons, MCP timeout counts, full SHA-256 for adapter/config/eval set, and structured JSON CI fields.
- Canonical current report: `/output/eval_results/qwen36_27b_v6_1_multiseed/multiseed_report.json`.

## GPU and Storage Readiness
- One NVIDIA A40 with 46,068 MiB total and 45,403 MiB free; no GPU compute processes at discovery time.
- `/models/Qwen3.6-27B` exists (52 GB), `/output/adapters` exists (3.6 GB), `/data/sft` exists (7.1 MB).
- Shared model/output/data volume has about 1.1 TB free; root/workspace has about 174 GB free.
- Long GPU work can run serially without artifact overwrite.

## Minimal v7 File Map
- Build from `trainer/scripts/build_sft_v6.py` lineage into a v7 dataset builder.
- Reuse hard multi-tool rows from `generate_fc_sft.py`, `generate_medium_sft.py`, and `generate_cot_sft.py`.
- Clone `trainer/configs/qlora_config_27b_v6_1.yaml` to a v7 config with v7 train/val and adapter paths.
- Keep `trainer/scripts/train_qlora.py` unchanged; it is already config-driven.

## Latency Task 5 Integration Seam
- The locked design requires the real path `POST /api/chat → ChatService → AgentPipeline.run_streaming()`; progress events do not count as TTFT and the first `message_delta` does.
- Current live mode still calls synchronous `AgentPipeline.run()` through `LiveRunner`, then reconstructs events in `_iter_stream_events`, so Task 4 timings cannot reach SSE.
- Task 5 must preserve demo scenario behavior while making the default live runner an async event source and mapping pipeline `delta` to API `message_delta` without recomputing complete timings.
- The `complete` SSE payload must preserve `ttft_ms`, `synthesis_ms`, `total_ms`, `tool_call_count`, `rag_chunk_count`, and `deadline_hit`; errors must preserve timing fields when supplied.
- `create_app(chat_service=...)` already provides the injection seam needed for GPU-free HTTP/SSE manual QA.

## Latency Task 6 Frontend Boundary
- The original Task 6 plan covered only timing-rich `complete`, but the real Task 4/5 stream also emits deadline `status` without `chunks` and timed `error`; the strict frontend schema rejected both.
- Keeping nested `.strict()` while adding only optional approved timing fields preserves rejection of unknown payload keys and compatibility with older backends.
- `App` must preserve exact-optional typing when mapping `chunks`/`elapsed_sec`; conditional spreads avoid assigning explicit `undefined`.
- Chrome QA exposed pre-existing out-of-scope UI debt: the 375px page retains a desktop-fixed grid and assistant Markdown is rendered as raw text. Neither was introduced by Task 6; the changed status row itself wraps cleanly.

## Task 7 Planning and Simplification Review
- Metis confirmed Task 7 should remain one typed benchmark module plus one test module, followed by an independent README architecture refresh; Task 8 measurements must not be invented early.
- The code-simplifier's proposed duplicate `route` removal was rejected after direct inspection: `metadata.route` and the top-level legacy `route` are separate compatibility surfaces.
- Preserve the approved repository stack for this task: `argparse` and installed `httpx==0.28.1`, despite generic greenfield skill preferences for Typer/httpx2.
- The eval set uses `q_id`, `question`, `tool_calls_required`, and `context_data`; profile filtering must use the required-tool list, while measured path classification must use actual `tool_call_count` from `complete`.

## Electron Demo Execution Constraints
- The Electron entry must not gate bootstrap on `process.argv[1]`; npm scripts pass a relative path, Chromium flags shift argv, and Windows uses different separators. The entry now bootstraps unconditionally.
- `index.html` must not compare the injected base URL against the same `%VITE_API_BASE_URL%` token, because Vite replaces every occurrence and the comparison then always resets the value.
- Electron loads `dist/index.html` over `file://`, so the same-origin fallback cannot work; `VITE_API_BASE_URL` at build time or a hand-edited `dist/index.html` is mandatory for remote APIs.
- Requests from `file://` carry an opaque origin, so the demo API must keep the permissive default `DEMO_UI_ORIGINS`; narrowing it can block the desktop shell.
- `dev:web` and `preview:web` pin `--host 127.0.0.1`, so remote browsers need an explicit `-- --host 0.0.0.0` or an SSH tunnel; the container also runs on a bridge network without published ports.
- The container has no display; only `xvfb-run` headless verification is possible there.

## Latency Task 7 Final Contract
- The benchmark measures the user-visible `POST /api/chat` SSE boundary: wall TTFT is the first nonempty valid `message_delta`, and wall total stops at `complete` arrival rather than connection close.
- Eval rows and metric-bearing events are parsed through strict finite Pydantic models. Every delta and nested MCP outcome is validated; contradictory success/error outcomes invalidate the sample.
- Terminal errors, malformed or missing terminals, complete without an answer delta, structured MCP `TIMEOUT`, and monotonic overall deadline expiry remain in artifacts but are excluded from percentile math.
- Reports expose total/valid/invalid/unclassified counts, expected-path coverage, per-path `no_data`/`invalid` states, and a report-level `gate_ready`; any invalid sample or missing expected path coverage blocks readiness.
- README now distinguishes one MCP aggregator with seven routers, 4B live defaults versus 27B opt-in, buffered compatibility streaming, pre-change score mean/sample SD, and Task 8's explicit 27B startup on port 8130.
- Task 7 Oracle code review and README review both ended APPROVE with no Critical or Important findings.
