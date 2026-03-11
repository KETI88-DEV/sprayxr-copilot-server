# Claude Code Instructions

## Task Master AI Instructions
**Import Task Master's development workflow commands and guidelines, treat as if import is in the main CLAUDE.md file.**
@./.taskmaster/CLAUDE.md

## Coding Rules

### Development Process
1. **Understand** — Restate requirements, identify inputs/outputs/constraints/edge cases
2. **Design** — Describe architecture, components, data flow BEFORE writing code
3. **Implement** — Write production-ready code following standards below
4. **Self-Review** — Review own code for bugs, unclear logic, maintainability issues. Fix before delivering.
5. **Verify** — Run the code. Confirm it works with real execution, not just type checks.

### Architecture
- Layered: Domain (no framework imports) → Application (use-cases, DTOs) → Infrastructure (DB, IO, API) → Presentation (CLI, handlers)
- Dependencies point inward only. Use interfaces/ports in Domain/Application; implement in Infrastructure.
- Centralize config (env/yaml). No hard-coded secrets. Use dependency injection.

### Code Standards (Python)
- Naming: `snake_case` (vars/functions), `PascalCase` (classes), `UPPER_CASE` (constants)
- Functions: single responsibility, 20-40 lines, minimal nesting
- Type hints on all public functions. Docstrings (`"""..."""`) explain **why**, not obvious **what**.
- Error handling: domain errors (business rules), application errors (validation), infrastructure errors (IO/network). Map at boundaries.
- No duplicated logic, no magic numbers, validate inputs, handle edge cases.

### Modification Rules
- Change only the minimal code required. Do not restructure, rename, or improve unrelated code.
- Preserve existing patterns and conventions in the codebase.

### Quality Gates
- Every script: argparse CLI, logging, error handling, type hints
- Match existing `/workspace/trainer/scripts/` style
