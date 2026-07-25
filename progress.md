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

## Next Action
Commit latency Task 5 and its Iter18-B/worklog records, then execute latency Task 6 frontend timing contract.
