# Progress

## 2026-07-25 Session Resume
- Restored the exact original items 1-5 from OpenCode session history.
- Loaded Subagent-Driven Development, planning, programming, TDD, git, and verification workflows.
- Inspected branch context and detected the repository's English semantic commit style.
- Replaced stale planning artifacts from an unrelated plugin-install task with the active Qwen3.6 execution plan.
- Started five independent discovery lanes for v7 requirements, evaluation artifacts, path-note rules, GPU readiness, and v7 training inputs.
- Inspected latency call paths through `ToolCallingAdapter.run_tool_loop` and `AgentPipeline.run` using CodeGraph.
- Found an interrupted implementation diff in the two Task 1 files; no commit has been made for it yet.
- Recovered all five discovery reports from child-session transcripts after the background result cache expired.
- Resolved the recording rule: update repository `PATCH_NOTE.md` as the authoritative iteration note, with detailed evidence mirrored in `findings.md` and `progress.md`.
- Confirmed the A40 is idle and all v7 model/data/output roots have sufficient capacity.
- Recovered the exact v7 data focus and the missing n=5 instrumentation fields.
- Completed latency Task 1 through TDD, independent spec review, and independent code-quality review.
- Rejected removal of the planned `tool_max_tokens` placeholder because Task 1 explicitly defines it and Task 2 consumes it next.
- Recorded Task 1 permanently in `PATCH_NOTE.md` under Iter18-B.
- Created atomic commits `6647435`, `d38d10f`, and `436f2fa` for Task 1 implementation, patch note, and synchronized worklogs.
- Item 1 push is blocked: the `origin` HTTPS credential is unavailable and `gh auth status` reports an invalid token.
- Completed latency Task 2: tool rounds use their own cap while both synthesis paths retain the full answer budget.
- Corrected Task 2 test-fixture drift and removed attempted cast/ClassVar type escape hatches before review.
- Completed latency Task 3: provisional deadline/tool cap config is wired through AgentPipeline and timing state is top-level for the demo API contract.
- Cleared the changed pipeline file's pre-existing implicit-relative import diagnostic with its established importlib pattern.
- Completed latency Task 4: added the staged buffered `run_streaming()` contract while preserving the frozen `run()` path.
- Hardened `_stream_generate()` so worker failures terminate the streamer, join the worker, and propagate the original exception.
- Finalized the streaming error boundary: blank instructions are non-retryable without invoking `run()`; internal failures are logged and returned as generic retryable errors without message leakage.
- Added five bounded streaming regressions and obtained final independent spec PASS and quality APPROVED verdicts.
- Drove the public async generator with GPU-free fakes and observed ordered deadline/status/tool/delta/complete events plus the blank-input error path.
- Completed latency Task 5: the real live demo path now consumes `AgentPipeline.run_streaming()` and forwards timing-bearing pipeline events to SSE.
- Offloaded lock-protected first pipeline construction from the ASGI event loop and added deterministic concurrent initialization coverage.
- Sanitized and timed adapter-generated failures while preserving pipeline-originated error payloads verbatim.
- Split Task 5 coverage into bounded event-contract and live-runner modules; obtained independent spec PASS and final quality APPROVED.
- Drove an actual uvicorn process with an injected async live runner and observed timing-bearing SSE plus FastAPI 422 validation.
- Completed latency Task 6: strict frontend Zod schemas now accept backend complete/error timings and deadline status metadata while preserving legacy payloads.
- Updated the React status consumer and tool log to display elapsed deadline state without undefined chunk placeholders.
- Confirmed RED on four real backend event shapes, then passed the full 20-test Vitest suite, production build, and two Playwright E2E scenarios.
- Drove real Chrome against FastAPI and exact timing-rich SSE; independent code review approved and fresh 1280/768/375 visual review passed for the changed status row.
- Started latency Task 7 with the user-requested Metis and code-simplifier reviews.
- Rejected a proposed `route` simplification after verifying the two occurrences belong to metadata and top-level legacy contracts.
- Locked Task 7 to a single typed `httpx` SSE benchmark module and one focused test module; comprehensive README refresh follows as a separate documentation commit.
- Completed Task 7 with separate CLI, typed SSE boundary, and pure report modules; all remain below the 250 pure-LOC ceiling.
- Closed repeated Oracle review findings for no-data false passes, complete-arrival timing, strict finite event parsing, structured MCP timeout handling, answer-delta requirements, overall heartbeat deadlines, contaminated-run readiness, expected-path coverage, and mutually exclusive MCP outcomes.
- Refreshed README architecture and Task 8 operations: one MCP aggregator/seven routers, 4B default versus 27B opt-in, buffered streaming caveat, pre-change score terminology, pending gates, and explicit 27B startup on port 8130.
- Drove the CLI against a real fake HTTP/SSE server with success, MCP timeout, terminal error, malformed complete, and delayed connection close; artifacts reported `1 valid / 3 invalid`, blocked gate readiness, and intact Markdown.

## Verification Ledger
| Phase | Command/surface | Result |
|---|---|---|
| Resume | `git status --short --branch` | `main...origin/main [ahead 22]`; two Task 1 files modified. |
| Resume | `git log -30` | 22/30 semantic English commit subjects; new commits will match. |
| Discovery | `nvidia-smi` via readiness worker | A40 idle; 45,403 MiB free; no compute process. |
| Task 1 | `python3 -m pytest trainer/scripts/test_tool_calling.py -k Deadline -q` | 3 passed, 24 deselected before spec correction. |
| Task 1 final | `python3 -m pytest trainer/scripts/test_tool_calling.py -q` | 28 passed. |
| Task 1 regression | `python3 -m pytest trainer/demo_api/tests trainer/scripts/test_tool_calling.py trainer/scripts/test_response_contract.py -q` | 51 passed; 2 pre-existing SWIG warnings. |
| Task 1 diagnostics | LSP error diagnostics on both changed Python files | 0 errors. |
| Task 1 reviews | Oracle spec + code quality | PASS / APPROVED. |
| Item 1 push | `git push origin main` | Failed: no usable HTTPS username/token; `gh` token is invalid. |
| Task 2 targeted | `python3 -m pytest trainer/scripts/test_tool_calling.py -k TokenCap -q` | 3 passed, 28 deselected. |
| Task 2 full | `python3 -m pytest trainer/scripts/test_tool_calling.py -q` | 31 passed. |
| Task 2 regression | `python3 -m pytest trainer/demo_api/tests trainer/scripts/test_tool_calling.py trainer/scripts/test_response_contract.py -q` | 54 passed, 2 pre-existing SWIG warnings. |
| Task 2 reviews | Oracle spec + quality | PASS / APPROVED. |
| Task 3 regression | `python3 -m pytest trainer/demo_api/tests trainer/scripts/test_tool_calling.py trainer/scripts/test_response_contract.py -q` | 54 passed, 2 pre-existing SWIG warnings. |
| Task 3 diagnostics | LSP error diagnostics on `agent_pipeline.py` | 0 errors. |
| Task 3 reviews | Oracle spec + quality | PASS / APPROVED. |
| Task 4 focused | `python3 -m pytest trainer/scripts/test_agent_pipeline_streaming.py -q` | 5 passed, 2 pre-existing SWIG warnings. |
| Task 4 regression | `python3 -m pytest trainer/demo_api/tests trainer/scripts/test_tool_calling.py trainer/scripts/test_response_contract.py trainer/scripts/test_agent_pipeline_streaming.py -q` | 59 passed, 2 pre-existing SWIG warnings. |
| Task 4 diagnostics | LSP error diagnostics on `agent_pipeline.py` and `test_agent_pipeline_streaming.py` | 0 errors. |
| Task 4 diff hygiene | `GIT_MASTER=1 git diff --check` | Exit 0; existing line-ending conversion warning only. |
| Task 4 reviews | Oracle spec + final quality | PASS / APPROVED; no Critical or Important findings remain. |
| Task 4 manual surface | `AgentPipeline.run_streaming()` with a fake completed result and blank input | Ordered events reconstructed the full response; complete timings/counts were present; blank input did not invoke `run()`. |
| Task 5 focused | `python3 -m pytest trainer/demo_api/tests -q` | 27 passed. |
| Task 5 diagnostics | LSP error diagnostics on `chat_service.py` and both Task 5 test modules | 0 errors. |
| Task 5 reviews | Oracle spec + final quality | PASS / APPROVED after off-loop initialization, sanitized errors, and deterministic bounded tests. |
| Task 5 manual HTTP/SSE | uvicorn on `127.0.0.1:8145` + curl `/api/health` and POST `/api/chat` | Health 200; ordered live SSE carried timing/count/deadline fields; blank message returned 422. |
| Task 6 RED | `npm test -- --run tests/contracts.test.ts` before schema changes | 4 failed for timing-rich complete, null TTFT, deadline status, and timed error. |
| Task 6 frontend gates | `npm test -- --run`; `npm run build:web`; `npm run test:e2e` | 20 passed; production build succeeded; 2 browser E2E passed. |
| Task 6 reviews | Oracle code review + dual visual/CJK review | APPROVE / PASS; no Task 6 Critical, Important, or visual blockers. |
| Task 6 browser surface | Chrome + real FastAPI demo, then exact deadline SSE at 1280×900, 768×900, 375×812 | POST 200; timing-rich complete parsed; elapsed status legible at all viewports; final console errors 0. |
| Task 7 focused | `python3 -m pytest trainer/scripts/tests/test_bench_{sse,latency,cli}.py -q` | 29 passed. |
| Task 7 regression | `python3 -m pytest trainer/demo_api/tests trainer/scripts/tests -q` | 62 passed. |
| Task 7 typing | changed-file `basedpyright` + LSP diagnostics | 0 errors, 0 warnings; all changed Python files clean. Full repository `basedpyright` still has unrelated pre-existing errors, including `train_qlora.py`. |
| Task 7 CLI surface | fake HTTP/SSE server + `bench_latency.py --profile mixed` | success stopped at complete before delayed close; timeout/error/malformed rows invalid; `gate_ready=false`; JSON/Markdown generated. |
| Task 7 reviews | Oracle harness review + README review | APPROVE / APPROVE; no Critical or Important findings remain. |

## 2026-07-27 Electron Demo Verification
- Ran the Electron shell for the first time inside the container using `xvfb-run` plus Playwright `_electron`.
- Reproduced a blocking defect: `electron dist-electron/main.js` (the npm script form) created zero CDP page targets because the argv-based entry guard never matched a relative path.
- Reproduced a second defect: the `index.html` bootstrap compared the injected value against the same placeholder token, so Vite replaced both sides and always reset `window.__DEMO_API_BASE_URL__` to undefined.
- Measured the Vite dev server binding: loopback returned 200 while the container IP was refused, contradicting the README remote-access instructions.
- Fixed both defects under TDD, corrected the README host claims, and added a Windows execution guide covering port publishing, SSH tunneling, and `.env.local`.

| Phase | Command/surface | Result |
|---|---|---|
| Electron RED | `electron dist-electron/main.js` + CDP `/json/list` | 0 page targets; absolute path produced 1. |
| Electron unit | `npx vitest run tests/electronMain.test.ts` | 1 failed before fix; suite green after. |
| index.html RED | `npx vitest run tests/htmlApiBaseUrl.test.ts` | 2 failed before fix. |
| Vite binding | `curl 127.0.0.1:5173` vs `curl 172.21.0.2:5173` | 200 vs connection refused. |
| Frontend gates | `npx vitest run`; `npm run build:web`; `npm run build:electron`; `npm run test:e2e` | 24 passed; builds succeeded; 2 E2E passed. |
| Electron surface | `xvfb-run` + Playwright `_electron` with relative arg | Window loaded; API base URL string applied; `/api/health` 200 from `file://`; chat streamed and zones updated. |

## Next Action
Start Task 8 on the A40: launch the explicit 27B live API on port 8130, run the mixed HTTP/SSE benchmark, recalibrate `tool_deadline_sec` from synthesis p95, and judge latency plus post-change quality gates. Windows-side Electron execution still needs user confirmation on a real desktop.
