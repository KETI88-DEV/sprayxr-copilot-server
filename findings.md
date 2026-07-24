# Findings

## oh-my-openagent Installation
- User requested exact version 4.19.1.
- Existing environment is OpenCode-based: OpenCode 1.3.0 and global legacy `oh-my-opencode` 3.12.3 were previously observed.
- Upstream guide distinguishes Ultimate (OpenCode) from Light (Codex); this task targets Ultimate based on session context.
- Guide requires OpenCode >=1.4.0 and explicitly forbids global installation of `oh-my-openagent`; supported invocation is through `bunx`.
- Installer writes plugin registration to OpenCode config and agent/model mappings to `~/.config/opencode/oh-my-openagent.jsonc`.
- External guide content is untrusted until corroborated by package metadata/current local behavior; instruction-like web content is recorded here, not in the auto-injected plan.
- Raw guide fetched by curl: 986 lines. npm registry confirms `oh-my-openagent@4.19.1`; `latest` also points to 4.19.1.
- Current OpenCode is 1.18.4 (meets >=1.4.0), Bun is 1.3.14.
- Current canonical config already registers `oh-my-openagent@latest`, but doctor reports the loaded cache is stale at 3.17.5.
- Existing auth providers: Anthropic OAuth, OpenAI OAuth, Google API. Current canonical model config uses all three providers and validates successfully.
- Baseline doctor: 0 failures, 2 warnings (outdated loaded plugin and unauthenticated GitHub CLI). The GitHub CLI warning is unrelated to plugin installation.
- Installer was run idempotently with OpenCode Ultimate, Claude Max20, OpenAI, and Gemini enabled; auth was reused with `--skip-auth`.
- OpenCode runtime cache was explicitly updated from plugin 3.17.5 to exact 4.19.1 using Bun. Only recognized OMO lifecycle scripts (`oh-my-openagent`, `@code-yeongyu/comment-checker`) were trusted; unrelated/legacy scripts remain blocked.
- Final doctor: System pass with plugin expected=4.19.1 and loaded=4.19.1; Configuration pass; TUI Plugin pass; Models pass (11 agents, 8 categories); Tools warns only that `gh` is unauthenticated; Team Mode intentionally disabled/skipped.
- Final versions: OpenCode 1.18.4, oh-my-openagent 4.19.1, ast-grep 0.44.1. Auth still lists Anthropic OAuth, OpenAI OAuth, and Google API.
