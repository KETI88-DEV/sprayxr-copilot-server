# Findings

## T4 Evaluation Notes
- Existing planning artifacts were reused and repointed at T4.
- `session-catchup.py` could not be run through `$CLAUDE_PLUGIN_ROOT` because that env var is unset in this shell; no additional unsynced context was recovered automatically.
- `generate_agentic_responses.py` actual flags: `--eval-set`, `--config`, `--adapter`, `--qlora-config`, `--mcp-url`, `--output`, `--log-level`, `--timeout`.
- `run_agentic_eval.py` actual flags: `--eval-set`, `--rubric`, `--tool-schemas-dir`, `--responses`, `--output-dir`, `--log-level`.
- `run_agentic_eval.py` is heuristic/offline scoring only; no LLM-judge API key is required by the current script.
- Generation completed for all 30 eval items with 47 total tool calls.
- Observed generation warnings (non-fatal): tool-loop max-round hits on AQ009/AQ013/AQ021/AQ024/AQ025/AQ026/AQ028, one `vlm_analysis` auto-correction, and schema-validation warnings on `sequence_validation` (AQ017) and `pose_estimation` (AQ027).
