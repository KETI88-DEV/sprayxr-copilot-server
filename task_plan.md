# Task Plan

## Goal
Complete the user's original items 1 through 5 for Qwen3.6-27B: push the current work, build and train v7 SFT, extend multi-seed evidence, run n=5 seeds to a ±0.10 CI target, and make the live demo meet the agreed latency budgets through its real HTTP/SSE surface.

## Original Items
- [pending] 1. Push committed work to `origin/main`.
- [pending] 2. Develop and train v7 SFT for tool-call format emission failures.
- [pending] 3. Extend multi-seed reporting with exact-sequence rate, parser rejection, and reproducibility hashes.
- [pending] 4. Run n=5 seeds and reduce the reported confidence interval to ±0.10 or document the measured result if the statistical target is not attainable from five runs.
- [in_progress] 5. Implement and verify the live latency budget: RAG first `message_delta` TTFT ≤30 seconds and tool-path completion ≤180 seconds for one concurrent user.

## Execution Order
1. Recover exact v7 and `path_note`/patch-note conventions; preserve evidence in `findings.md` and `progress.md`.
2. Finish latency Tasks 1-7 under TDD, commit in atomic implementation/test units, and push durable work.
3. Run the real live benchmark (latency Task 8), recalibrate the deadline from measured synthesis p95, and apply a kernel/backend escalation only if the first-stage gate fails.
4. Implement v7 data/config changes under TDD, train v7, and record immutable input/config/output hashes.
5. Extend evaluation reporting, run five seeds, compute the CI, and enforce the ≥6.60 quality gate.
6. Update the authoritative path/patch note and README/result summaries, create final atomic documentation commits, push `main`, and independently verify all five items.

## Latency Plan Progress
- [complete] Task 1: tool-loop deadline guard and synthesis timing.
- [complete] Task 2: round-differentiated token caps.
- [complete] Task 3: config wiring.
- [complete] Task 4: additive streaming contract.
- [complete] Task 5: demo API timing propagation.
- [pending] Task 6: frontend Zod timing contract.
- [pending] Task 7: HTTP/SSE benchmark harness.
- [pending] Task 8: live GPU benchmark, deadline recalibration, and quality gate.

## Constraints
- Preserve `AgentPipeline.run()` for frozen evaluation comparability; streaming is additive.
- RAG latency gate is TTFT to first `message_delta`; progress events do not count.
- Tool latency gate is full response completion within 180 seconds.
- Quality gate is agentic total ≥6.60.
- Deadline default starts at 150 seconds and must be recalibrated to `180 - synthesis_ms_p95` after measurement.
- Do not install Flash Linear Attention, causal-conv1d, or switch to vLLM unless measured first-stage latency fails.
- Record paths, commands, hashes, metrics, errors, and decisions after each phase.
- Create atomic git commits and push; never rewrite `main` history.

## Errors Encountered
| Error | Attempt | Resolution |
|---|---:|---|
| Prior root planning files described an unrelated OpenCode installation | 1 | Replaced them with the active Qwen3.6 items 1-5 plan before further implementation. |
| Initial Task 1 delegation was interrupted while files were being modified | 1 | Treat current diff as untrusted partial work; inspect and verify before accepting or committing. |
| GPU readiness background dispatch returned a contradictory launch/failure result | 1 | Wait for the task notification; if no usable result arrives, run a direct read-only readiness check by a different route. |
| `git push origin main` failed because the HTTPS token is invalid | 1 | Continue local work and retry only after a valid preconfigured credential is available; never expose or synthesize credentials. |
| Task 2 implementer repeatedly changed existing unittest fixture scope | 1 | Restored per-instance construction, removed cast/ClassVar escape hatches, and obtained final spec/quality approval on the actual worktree. |
| First Task 4 manual driver used `importlib` without registering the module in `sys.modules`, breaking dataclass decoration | 1 | Switched to the repository's documented `trainer/scripts` import path; the public async-generator QA then passed without product-code changes. |

## Completion Gate
- All five original items have durable artifacts and recorded evidence.
- Every changed source file has clean diagnostics.
- Relevant unit/integration/build suites pass.
- v7 training and n=5 evaluation outputs exist with hashes and metrics.
- The live HTTP/SSE surface has been driven and both latency budgets measured.
- Atomic commits exist, the authoritative path/patch note is updated, and `origin/main` contains the final commits.
