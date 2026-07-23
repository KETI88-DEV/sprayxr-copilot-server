# Task Plan

## Goal
Execute T4 for /workspace without modifying production code: run the post-fix 30-item Qwen3-4B agentic evaluation against current code, score it if the scorer can run, and freeze a baseline manifest with score/tool metrics plus artifact hashes and git SHA.

## Phases
- [x] Phase 1: Confirm actual CLI flags/help for generation and scoring scripts, then start the MCP server and verify `/health` returns 200 with `DEMO_MODE=false` planned for generation.
- [x] Phase 2: Generate `/output/eval_results/agentic_responses.jsonl` for the full 30-item eval set using the current adapter/config/index stack.
- [x] Phase 3: Run the agentic scorer into `/output/eval_results/baseline_v2` or capture the exact reason rubric scoring is unavailable.
- [x] Phase 4: Write `/output/eval_results/baseline_v2/manifest.json` with score/tool-call metrics, item count, SHA-256 hashes for adapter files, FAISS artifacts, rag config, tool schemas, and git HEAD short SHA.
- [x] Phase 5: Verify outputs/counts, compare against Iter14 baseline 4.97/10, and stop the background MCP server.

## Constraints
- Evaluation-only run: do not modify production code, configs, adapters, or existing final reports.
- Full 30-item set only; no downsizing or skipped items.
- If scorer needs absent API credentials, report that limitation precisely and do not fabricate a score.

## Key Decisions
- Actual argparse/help output is the source of truth for invocation flags.
- Manifest should tie the baseline to both code state (`git rev-parse --short HEAD`) and major evaluation artifacts via SHA-256.
- Any retry must be based on observed stderr/traceback, not guesswork.

## Errors Encountered
- `session-catchup.py` first attempt failed because `$CLAUDE_PLUGIN_ROOT` was unset, yielding `/scripts/session-catchup.py` not found.
- JSON LSP diagnostics were unavailable because the configured `biome` server is not installed; manifest JSON validity was verified via Python round-trip instead.
