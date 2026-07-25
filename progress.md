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

## Next Action
Commit latency Task 3 and its Iter18-B/worklog records, then execute latency Task 4 streaming path.
