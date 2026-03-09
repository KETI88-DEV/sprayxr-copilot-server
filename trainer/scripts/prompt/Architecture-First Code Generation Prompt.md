# Architecture-First Code Generation Prompt

## Role
Act as a **software architect and senior engineer**. Prioritize **clean architecture, modularity, testability, and long-term maintainability** over quick hacks.

## Objective
Design and implement {SYSTEM/FEATURE} so that it is:
- modular and layered (clear separation of concerns)
- easy to extend (new features without rewriting core)
- testable (units can be tested in isolation)
- observable (logging/metrics where appropriate)
- production-ready (error handling, input validation)

## Context
Use the following constraints and background:
- Domain: {DOMAIN}
- Primary use-case: {USE_CASE}
- Users/actors: {ACTORS}
- Data sources: {DB/APIs/FILES}
- Platform/runtime: {LANGUAGE + FRAMEWORK + VERSION}
- Non-functional requirements: {PERFORMANCE / SECURITY / RELIABILITY}
- Constraints: {TIME/BUDGET/COMPUTE/LEGACY}
- Out of scope: {OUT_OF_SCOPE}

## Architecture Requirements
Follow these architecture rules strictly:

1) Layers
- **Domain**: entities, value objects, domain services; no framework imports.
- **Application**: use-cases (service layer), orchestration, DTOs; depends on Domain only.
- **Infrastructure**: DB/API clients, file IO, external integrations; depends on Application/Domain via interfaces.
- **Presentation** (if applicable): controllers/handlers/CLI; depends on Application layer.

2) Dependency Rule
- Dependencies must point inward (Presentation/Infrastructure -> Application -> Domain).
- Use **interfaces/ports** in Application/Domain; implement them in Infrastructure.

3) Data Flow
- Input -> Presentation -> Application use-case -> Domain logic -> Ports -> Infrastructure adapters -> Output
- Keep side effects (IO, network) out of Domain.

4) Error Handling
- Define a consistent error model:
  - domain errors (business rules)
  - application errors (validation, not found)
  - infrastructure errors (timeouts, connection)
- Map errors at boundaries (presentation layer).

5) Configuration
- Centralize configuration (env/config file).
- No hard-coded secrets.
- Use dependency injection (explicit constructors).

## Step-by-Step Instructions
Think step by step and do the following in order:

### Step 1 — Clarify Requirements
- Restate the problem in your own words.
- List functional requirements and non-functional requirements.
- Identify assumptions and risks.

### Step 2 — Propose Architecture
Provide:
- A component diagram (text form is fine).
- A folder/package structure.
- Key interfaces (ports) and adapters.
- Data contracts (DTOs) and domain entities.

### Step 3 — Define Public APIs
Specify:
- Primary entrypoints (functions/classes/endpoints/CLI commands).
- Input/output schemas.
- Examples of usage.

### Step 4 — Implementation Plan
- Break implementation into milestones.
- Note test strategy per layer.

### Step 5 — Implement
Generate the code with:
- consistent module boundaries
- docstrings/comments explaining intent
- type hints where available
- minimal coupling
- no framework imports in Domain

### Step 6 — Tests and Quality Gates
Provide:
- unit tests for Domain and Application
- integration tests stubs for Infrastructure
- lint/format guidance
- “how to run” instructions

### Step 7 — Architecture Review (Self-critique)
Review your design like an architect:
- identify violations of dependency rule
- highlight complexity hotspots
- suggest refactors and extension points
- list trade-offs

## Output Format
Return your answer in this structure:

1) Requirements Summary
2) Architecture Proposal
   - Components
   - Dependency flow
   - Folder structure
   - Interfaces & contracts
3) API / Entrypoints
4) Implementation Milestones
5) Code (Domain / Application / Infrastructure / Presentation)
6) Tests
7) Architecture Review & Next Steps

## Constraints on Response Size (Adaptive)
- If the request is small: implement a minimal skeleton + key interfaces + 1–2 example use-cases.
- If the request is large: implement a representative slice end-to-end and clearly mark TODOs.

## Evidence and Justification
For each major architectural choice, briefly justify:
- why this pattern fits the requirements
- alternatives considered
- when you would change the approach