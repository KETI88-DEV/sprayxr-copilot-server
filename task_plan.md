# Task Plan

## Goal
Install and configure the OpenCode Ultimate edition of `oh-my-openagent` at version 4.19.1, migrating the existing `oh-my-opencode` setup without losing provider configuration, then verify with the official doctor command.

## Phases
- [complete] Phase 1: Fetch the authoritative dev-branch guide exactly, inspect prerequisites, current OpenCode/plugin configuration, and infer existing provider selections safely.
- [complete] Phase 2: Back up affected OpenCode configuration and confirm OpenCode meets the guide requirement (>=1.4.0).
- [complete] Phase 3: Run the supported non-global `bunx oh-my-openagent@4.19.1 install` path with explicit non-interactive platform/provider flags.
- [complete] Phase 4: Verify plugin registration, generated configuration, installed version, and `doctor` results; remediate any errors.
- [in_progress] Phase 5: Obtain independent verifier/Oracle review and report final state.

## Constraints
- Target exact version 4.19.1.
- Install Ultimate edition for OpenCode; do not install Codex Light unless explicitly requested.
- Do not expose credentials or alter provider authentication unnecessarily.
- Preserve existing user configuration via backup and migration rather than destructive replacement.
- Follow upstream supported `bunx` installation; no global npm/bun install of the plugin.

## Key Decisions
- The repeated OpenCode/oh-my-opencode context establishes OpenCode as the requested platform.
- Existing provider/plugin configuration will be inspected to infer non-sensitive installer flags where possible.
- Provider auth evidence supports OpenAI and Gemini; installer migration detection explicitly identified the prior Claude tier as `max20`, so preserve it on the corrective idempotent run. Disable providers with no local evidence and use `--skip-auth` to preserve existing credentials.

## Errors Encountered
- `customize-opencode` was advertised in environment metadata but unavailable through the runtime skill registry; proceeded with the authoritative guide and available planning/orchestration skills.
- Initial guide retrieval used WebFetch before its embedded agent note required raw `curl`; the raw guide was fetched with `curl` before installation and treated as source of truth.
- First install run used `--claude=yes`; installer migration detection reported prior `Claude=max20`. Resolution: rerun idempotently with `--claude=max20` to preserve the detected tier.
- Installer warned that its temporary package lacked `dist/cli/skills/ast-grep/install.sh`; pre-install doctor already confirmed AST-grep CLI is present, so final doctor will determine whether remediation is needed.
