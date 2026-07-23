# Progress

## Session Log
- Started T4 baseline-freeze evaluation session.
- Read existing planning files from prior T2/T3 work and replaced plan with T4 phases.
- `session-catchup.py` attempt via `$CLAUDE_PLUGIN_ROOT` failed because the env var is unset; proceeding with direct project-state inspection instead.
- Verified actual argparse/help for both evaluation scripts; confirmed generation uses `--mcp-url` and scoring requires only local files plus `--tool-schemas-dir`.
- Started MCP server with uvicorn on port 8100 and confirmed `/health` returned HTTP 200 before generation.
- Ran full 30-item response generation with `DEMO_MODE=false` and `MCP_BASE_URL=http://localhost:8100`; output written to `/output/eval_results/agentic_responses.jsonl`.
- Generation summary: 30 responses, 47 total tool calls, average response length 710.23 chars.
- Ran offline agentic scorer successfully into `/output/eval_results/baseline_v2`: average total score 5.3333/10, tool_not_called_count 8.
- Wrote `/output/eval_results/baseline_v2/manifest.json` with git SHA, artifact hashes, adapter/tool-schema directory listings + per-file SHA-256 values, and per-item score/tool summaries.
- Verified MCP shutdown by checking `/health` returned HTTP 000 after killing uvicorn.
