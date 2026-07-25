# Latency Budget Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the live demo meet RAG-only TTFT ≤ 30s and tool-path completion ≤ 180s, and prove it with a measurement harness.

**Architecture:** Add a wall-clock deadline guard and round-differentiated token caps to the existing tool loop, add a streaming execution path that emits answer tokens as they are produced, instrument TTFT/synthesis/total timings end-to-end, then measure the real HTTP/SSE path with a dedicated bench script. The existing non-streaming `run()` is preserved untouched so evaluation reproducibility is not disturbed.

**Tech Stack:** Python 3.11, transformers `TextIteratorStreamer`, FastAPI SSE, pytest, React + Zod, httpx.

**Spec:** `docs/superpowers/specs/2026-07-25-latency-budget-live-demo-design.md`

**Git note:** Commit steps are included but require explicit user authorization before running; batch them if the user prefers.

---

## Locked File Structure

### Phase A — tool loop control (no GPU required)
- `trainer/scripts/tool_calling_adapter.py` — deadline guard, token caps, `synthesis_ms` measurement
- `trainer/scripts/test_tool_calling.py` — HF-style fake model/tokenizer harness + new tests

### Phase B — streaming and instrumentation (GPU for manual check only)
- `trainer/scripts/agent_pipeline.py` — `run_streaming()` async generator, config passthrough
- `trainer/demo_api/chat_service.py` — consume streaming events, carry timings
- `trainer/demo_api/contracts.py` — `complete` event timing fields
- `trainer/demo_api/tests/test_chat_service_streaming.py` — streaming event order + timing fields
- `demo-ui/src/contracts.ts` — Zod extension for the new `complete` fields
- `demo-ui/tests/contracts.test.ts` — schema test

### Phase C — measurement
- `trainer/scripts/bench_latency.py` — drives FastAPI SSE, emits report
- `trainer/scripts/tests/test_bench_latency.py` — percentile/classification logic
- `trainer/configs/rag_config_fewshot_27b.yaml` — `tool_deadline_sec`, `max_new_tokens_tool`

---

### Task 1: Deadline guard and synthesis timing in the tool loop

**Files:**
- Modify: `trainer/scripts/tool_calling_adapter.py` (imports, `run_tool_loop`)
- Test: `trainer/scripts/test_tool_calling.py`

The loop currently runs `max_rounds` LLM passes with no wall-clock bound. Four measured runs exceeded 180s. This adds a time bound and records how long forced synthesis actually takes, which the spec requires to recalibrate the deadline default.

- [ ] **Step 1: Add the HF-style fake harness to the test file**

Append to `trainer/scripts/test_tool_calling.py` (the existing `FakeModel` has `generate_response`, which short-circuits the HF code path we need to exercise):

```python
class _FakeIds:
    shape = (1, 4)


class _FakeInputs:
    def __init__(self) -> None:
        self.input_ids = _FakeIds()

    def to(self, device: Any) -> "_FakeInputs":
        return self


class RecordingTokenizer:
    """Returns scripted decode outputs and accepts any chat template call."""

    eos_token_id = 0

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.decode_count = 0

    def apply_chat_template(self, messages: Any, **kwargs: Any) -> str:
        return "PROMPT"

    def __call__(self, texts: Any, **kwargs: Any) -> _FakeInputs:
        return _FakeInputs()

    def decode(self, ids: Any, skip_special_tokens: bool = True) -> str:
        out = self.outputs[self.decode_count]
        self.decode_count += 1
        return out


class RecordingModel:
    """HF-style fake without generate_response; records generate() kwargs."""

    device = "cpu"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def generate(self, **kwargs: Any) -> Any:
        self.calls.append(dict(kwargs))
        return [[0] * 12]
```

- [ ] **Step 2: Write the failing deadline test**

Append to `trainer/scripts/test_tool_calling.py`:

```python
class ToolLoopDeadlineTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.adapter = ToolCallingAdapter(
            schema_dir=Path("/workspace/trainer/configs/tool_schemas"),
            protocol_path=Path("/workspace/trainer/configs/tool_calling_protocol.json"),
            mcp_base_url="http://localhost:8100",
        )

    async def test_deadline_stops_further_rounds_and_forces_synthesis(self) -> None:
        call = '<tool_call>{"name": "dft_judge", "arguments": {"measurements": [95], "spec": {"nominal": 100}}}</tool_call>'
        tokenizer = RecordingTokenizer([call, call, "SYNTHESIZED"])
        model = RecordingModel()

        clock = {"now": 0.0}
        with patch.object(adapter_module.time, "monotonic", lambda: clock["now"]), \
             patch.object(self.adapter, "dispatch_tool_call", new=AsyncMock(return_value={"status": "ok"})):

            original = self.adapter._generate_model_text

            async def advancing(*args: Any, **kwargs: Any) -> str:
                clock["now"] += 100.0
                return await original(*args, **kwargs)

            with patch.object(self.adapter, "_generate_model_text", side_effect=advancing):
                result = await self.adapter.run_tool_loop(
                    messages=[{"role": "user", "content": "q"}],
                    model=model,
                    tokenizer=tokenizer,
                    max_rounds=5,
                    deadline_sec=150.0,
                )

        self.assertTrue(result["deadline_hit"])
        self.assertEqual(result["rounds"], 2)
        self.assertEqual(result["final_response"], "SYNTHESIZED")
        self.assertIn("synthesis_ms", result)

    async def test_no_deadline_hit_when_within_budget(self) -> None:
        tokenizer = RecordingTokenizer(["plain answer"])
        model = RecordingModel()
        result = await self.adapter.run_tool_loop(
            messages=[{"role": "user", "content": "q"}],
            model=model,
            tokenizer=tokenizer,
            max_rounds=5,
            deadline_sec=150.0,
        )
        self.assertFalse(result["deadline_hit"])
        self.assertEqual(result["final_response"], "plain answer")
```

- [ ] **Step 3: Run it and confirm failure**

Run: `cd /workspace && python3 -m pytest trainer/scripts/test_tool_calling.py -k Deadline -q`

Expected: FAIL — `run_tool_loop() got an unexpected keyword argument 'deadline_sec'`.

- [ ] **Step 4: Add the `time` import**

`trainer/scripts/tool_calling_adapter.py` currently does not import `time`. Add it to the stdlib import block so it sits alphabetically after `re`:

```python
import re
import time
import uuid
```

- [ ] **Step 5: Extend the `run_tool_loop` signature**

Replace the signature (currently at `trainer/scripts/tool_calling_adapter.py:702`):

```python
    async def run_tool_loop(
        self,
        messages: list[dict[str, Any]],
        model: Any,
        tokenizer: Any,
        enable_thinking: bool = False,
        max_rounds: int = 5,
        max_tokens: int | None = None,
        gen_params: dict[str, Any] | None = None,
        deadline_sec: float | None = None,
        tool_max_tokens: int | None = None,
    ) -> dict[str, Any]:
```

- [ ] **Step 6: Add loop state and the deadline check**

Immediately after the `max_rounds` validation, replace the state initialisation with:

```python
        started = time.monotonic()
        deadline_hit = False
        synthesis_ms = 0
        conversation = list(messages)
        collected_tool_results: list[dict[str, Any]] = []

        for round_idx in range(max_rounds):
            if (
                deadline_sec is not None
                and collected_tool_results
                and (time.monotonic() - started) >= deadline_sec
            ):
                deadline_hit = True
                logger.warning(
                    "Tool loop hit deadline %.1fs after %d rounds; forcing synthesis",
                    deadline_sec,
                    round_idx,
                )
                break
```

The `collected_tool_results` condition matters: never abort before at least one tool result exists, otherwise synthesis would have nothing to work from and the user gets an empty answer.

- [ ] **Step 7: Make forced synthesis handle both exits and record its duration**

The block after the `for` loop currently logs `max_rounds` unconditionally. Replace it so it serves both the deadline exit and the exhausted-rounds exit, and times the synthesis call:

```python
        if not deadline_hit:
            logger.warning(
                "Tool loop reached max_rounds=%d without final answer; forcing synthesis",
                max_rounds,
            )

        rounds_used = round_idx if deadline_hit else max_rounds

        if self.force_synthesis and collected_tool_results:
            synth_started = time.monotonic()
            final_text = await self._generate_synthesis_text(
                conversation, model, tokenizer, max_tokens=max_tokens, gen_params=gen_params,
            )
            synthesis_ms = int((time.monotonic() - synth_started) * 1000)
            conversation.append({"role": "assistant", "content": final_text})
            return {
                "final_response": final_text,
                "messages": conversation,
                "tool_results": collected_tool_results,
                "rounds": rounds_used,
                "truncated": True,
                "synthesized": True,
                "deadline_hit": deadline_hit,
                "synthesis_ms": synthesis_ms,
            }
```

- [ ] **Step 8: Add the new keys to the remaining two return sites**

The early return inside the loop (the no-tool-call branch) and the final fallback return must both carry the keys so callers never have to guard for their absence. Add to the early in-loop return dict:

```python
                    "deadline_hit": False,
                    "synthesis_ms": synthesis_ms,
```

and to the trailing fallback return dict:

```python
            "deadline_hit": deadline_hit,
            "synthesis_ms": synthesis_ms,
```

Also change that fallback's `"rounds": max_rounds` to `"rounds": rounds_used`.

- [ ] **Step 9: Run the tests**

Run: `cd /workspace && python3 -m pytest trainer/scripts/test_tool_calling.py -q`

Expected: all tests pass, including the two new deadline tests.

---

### Task 2: Round-differentiated token caps

**Files:**
- Modify: `trainer/scripts/tool_calling_adapter.py` (`run_tool_loop` body)
- Test: `trainer/scripts/test_tool_calling.py`

Tool-call JSON needs a few hundred tokens; synthesis needs the full budget. A uniform 1024 gives the model room to write prose and never emit the `<tool_call>` tag, which is the exact failure observed on AQ022.

- [ ] **Step 1: Write the failing test**

Append to `trainer/scripts/test_tool_calling.py`:

```python
class ToolLoopTokenCapTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.adapter = ToolCallingAdapter(
            schema_dir=Path("/workspace/trainer/configs/tool_schemas"),
            protocol_path=Path("/workspace/trainer/configs/tool_calling_protocol.json"),
            mcp_base_url="http://localhost:8100",
        )

    async def test_tool_round_and_synthesis_use_different_caps(self) -> None:
        call = '<tool_call>{"name": "dft_judge", "arguments": {"measurements": [95], "spec": {"nominal": 100}}}</tool_call>'
        tokenizer = RecordingTokenizer([call, "SYNTHESIZED"])
        model = RecordingModel()

        with patch.object(self.adapter, "dispatch_tool_call", new=AsyncMock(return_value={"status": "ok"})):
            await self.adapter.run_tool_loop(
                messages=[{"role": "user", "content": "q"}],
                model=model,
                tokenizer=tokenizer,
                max_rounds=1,
                max_tokens=1024,
                tool_max_tokens=256,
            )

        self.assertEqual(model.calls[0]["max_new_tokens"], 256)
        self.assertEqual(model.calls[-1]["max_new_tokens"], 1024)

    async def test_tool_max_tokens_defaults_to_max_tokens(self) -> None:
        tokenizer = RecordingTokenizer(["plain answer"])
        model = RecordingModel()
        await self.adapter.run_tool_loop(
            messages=[{"role": "user", "content": "q"}],
            model=model,
            tokenizer=tokenizer,
            max_rounds=1,
            max_tokens=777,
        )
        self.assertEqual(model.calls[0]["max_new_tokens"], 777)
```

- [ ] **Step 2: Run it and confirm failure**

Run: `cd /workspace && python3 -m pytest trainer/scripts/test_tool_calling.py -k TokenCap -q`

Expected: FAIL — the first call records 777/1024 rather than the tool cap, because `tool_max_tokens` is not yet applied.

- [ ] **Step 3: Apply the cap to tool rounds only**

Inside the `for round_idx` loop, change the `_generate_model_text` call so the tool round uses its own cap while synthesis keeps `max_tokens`:

```python
            raw_text = await self._generate_model_text(
                messages=conversation,
                model=model,
                tokenizer=tokenizer,
                enable_thinking=enable_thinking,
                max_tokens=tool_max_tokens if tool_max_tokens is not None else max_tokens,
                gen_params=gen_params,
            )
```

Leave both `_generate_synthesis_text` call sites on `max_tokens=max_tokens`.

- [ ] **Step 4: Run the tests**

Run: `cd /workspace && python3 -m pytest trainer/scripts/test_tool_calling.py -q`

Expected: all pass.

---

### Task 3: Config wiring through the pipeline

**Files:**
- Modify: `trainer/configs/rag_config_fewshot_27b.yaml`
- Modify: `trainer/scripts/agent_pipeline.py` (the `run_tool_loop` call site near line 797)

- [ ] **Step 1: Add the two settings**

In `trainer/configs/rag_config_fewshot_27b.yaml`, add to the `llm` block:

```yaml
  tool_deadline_sec: 150
```

and to the `generation` block:

```yaml
  max_new_tokens_tool: 256
```

`tool_deadline_sec: 150` is provisional. Task 8 recalibrates it from measured `synthesis_ms` p95 as `180 - p95`.

- [ ] **Step 2: Pass them at the call site**

In `trainer/scripts/agent_pipeline.py`, extend the existing `self.tool_adapter.run_tool_loop(...)` call:

```python
            loop_output = _run_coro_sync(
                self.tool_adapter.run_tool_loop(
                    messages=messages,
                    model=self.model,
                    tokenizer=self.tokenizer,
                    enable_thinking=tool_thinking,
                    max_rounds=ROUTE_MAX_ROUNDS.get(route, 5),
                    max_tokens=route_max_tokens,
                    gen_params=gen_params,
                    deadline_sec=float(self.config.get("llm", {}).get("tool_deadline_sec", 150)),
                    tool_max_tokens=int(gen_cfg.get("max_new_tokens_tool", 256)),
                )
            )
```

- [ ] **Step 3: Surface the new fields in the result**

Still in `run()`, after `rounds = int(loop_output.get("rounds", 0))`, capture:

```python
            deadline_hit = bool(loop_output.get("deadline_hit", False))
            synthesis_ms = int(loop_output.get("synthesis_ms", 0))
```

Initialise both to `False` / `0` on the non-tool branch so the names always exist, and add them to the `result` dict alongside `latency_ms`:

```python
        result = {
            ...
            "deadline_hit": deadline_hit,
            "synthesis_ms": synthesis_ms,
        }
```

- [ ] **Step 4: Confirm no regression**

Run: `cd /workspace && python3 -m pytest trainer/demo_api/tests trainer/scripts/test_tool_calling.py trainer/scripts/test_response_contract.py -q`

Expected: 47+ passed (47 existing plus the new adapter tests from Tasks 1-2).

---

### Task 4: Streaming execution path

**Files:**
- Modify: `trainer/scripts/agent_pipeline.py` (new `run_streaming` method)

`run()` stays exactly as-is so `generate_agentic_responses.py` and the frozen eval baseline are untouched. Streaming is additive.

- [ ] **Step 1: Add the streaming synthesis helper**

Add to `AgentPipeline`:

```python
    def _stream_generate(
        self,
        prompt_text: str,
        gen_cfg: dict[str, Any],
        max_new_tokens: int,
    ) -> Iterator[str]:
        """Yields decoded text chunks as the model produces them."""
        from threading import Thread

        from transformers import TextIteratorStreamer

        inputs = self.tokenizer([prompt_text], return_tensors="pt", return_attention_mask=True)
        if hasattr(inputs, "to") and hasattr(self.model, "device"):
            inputs = inputs.to(self.model.device)

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        do_sample = bool(gen_cfg.get("do_sample", True))
        kwargs: dict[str, Any] = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
            "repetition_penalty": float(gen_cfg.get("repetition_penalty", 1.15)),
            "do_sample": do_sample,
            "streamer": streamer,
        }
        if do_sample:
            kwargs["temperature"] = float(gen_cfg.get("temperature", 0.3))
            kwargs["top_p"] = float(gen_cfg.get("top_p", 0.8))
            kwargs["top_k"] = int(gen_cfg.get("top_k", 20))

        thread = Thread(target=self.model.generate, kwargs=kwargs, daemon=True)
        thread.start()
        for chunk in streamer:
            if chunk:
                yield chunk
        thread.join()
```

Add `Iterator` to the `typing` imports at the top of the file.

- [ ] **Step 2: Add `run_streaming`**

```python
    async def run_streaming(
        self,
        instruction: str,
        environment: dict[str, Any] | None = None,
        session_id: str | None = None,
        enable_thinking: bool = False,
        request_id: str | None = None,
    ) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        """Yields (kind, payload) where kind is status|tool_result|delta|complete|error.

        Falls back to the non-streaming run() if the streamer cannot be created,
        emitting the whole answer as a single delta so callers need no special case.
        """
        t0 = time.monotonic()
        ttft_ms: int | None = None

        try:
            result = await asyncio.to_thread(
                self.run,
                instruction=instruction,
                environment=environment,
                session_id=session_id,
                enable_thinking=enable_thinking,
                request_id=request_id,
            )
        except Exception as exc:
            yield "error", {"message": str(exc), "retryable": True}
            return

        yield "status", {"stage": "rag", "chunks": len(result.get("rag_chunks", []))}

        if result.get("deadline_hit"):
            yield "status", {
                "stage": "deadline_forced_synthesis",
                "elapsed_sec": int((time.monotonic() - t0)),
            }

        for call in result.get("tool_calls", []):
            yield "tool_result", {
                "name": call.get("name", ""),
                "arguments": call.get("arguments", {}) or {},
                "result": call.get("result", {}) or {},
            }

        text = str(result.get("response", ""))
        for index in range(0, len(text), 24):
            if ttft_ms is None:
                ttft_ms = int((time.monotonic() - t0) * 1000)
            yield "delta", {"text": text[index : index + 24]}

        yield "complete", {
            "ttft_ms": ttft_ms,
            "synthesis_ms": int(result.get("synthesis_ms", 0)),
            "total_ms": int((time.monotonic() - t0) * 1000),
            "tool_call_count": len(result.get("tool_calls", [])),
            "rag_chunk_count": len(result.get("rag_chunks", [])),
            "deadline_hit": bool(result.get("deadline_hit", False)),
        }
```

This first cut reuses `run()` and therefore does **not** yet reduce TTFT; it establishes the event contract and the measurement so Task 8 can quantify the gap. Task 5 replaces the synthesis segment with `_stream_generate` once the contract is proven.

Add `AsyncIterator` to the `typing` imports and ensure `asyncio` is imported.

- [ ] **Step 3: Smoke the contract without a GPU**

Run:

```bash
cd /workspace && python3 -c "
import inspect, sys
sys.path.insert(0, 'trainer/scripts')
import agent_pipeline
assert inspect.isasyncgenfunction(agent_pipeline.AgentPipeline.run_streaming)
assert hasattr(agent_pipeline.AgentPipeline, '_stream_generate')
print('streaming API present')
"
```

Expected: `streaming API present`.

---

### Task 5: Wire streaming into the demo API

**Files:**
- Modify: `trainer/demo_api/contracts.py`
- Modify: `trainer/demo_api/chat_service.py`
- Test: `trainer/demo_api/tests/test_chat_service_streaming.py`

- [ ] **Step 1: Write the failing test**

Create `trainer/demo_api/tests/test_chat_service_streaming.py`:

```python
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import TypeVar

from trainer.demo_api.chat_service import ChatService
from trainer.demo_api.contracts import ChatRequest

T = TypeVar("T")


def _collect(agen: AsyncIterator[T]) -> list[T]:
    async def _run() -> list[T]:
        return [event async for event in agen]

    return asyncio.run(_run())


def test_demo_complete_event_carries_timing_fields() -> None:
    request = ChatRequest(message="다음에 어떤 구역을 칠해야 해?", session_id="s-1")
    events = _collect(ChatService().stream(request))
    complete = [e for e in events if e.event == "complete"][-1]
    assert "ttft_ms" in complete.data
    assert "total_ms" in complete.data
    assert "synthesis_ms" in complete.data
    assert isinstance(complete.data["total_ms"], int)
```

- [ ] **Step 2: Run it and confirm failure**

Run: `cd /workspace && python3 -m pytest trainer/demo_api/tests/test_chat_service_streaming.py -q`

Expected: FAIL — `KeyError`/assertion on `ttft_ms`.

- [ ] **Step 3: Record timings in the demo path**

In `trainer/demo_api/chat_service.py`, import `time` and change `_iter_stream_events` so it accepts a start timestamp and reports timings. Replace the `complete` yield at the end of `_iter_stream_events` with:

```python
    yield StreamEvent(
        event="complete",
        data={
            "tool_call_count": len(tool_calls),
            "rag_chunk_count": len(rag_chunks),
            "ttft_ms": ttft_ms,
            "synthesis_ms": int(scenario.get("synthesis_ms", 0) or 0),
            "total_ms": int((time.monotonic() - started) * 1000),
            "deadline_hit": bool(scenario.get("deadline_hit", False)),
        },
    )
```

Set `started = time.monotonic()` at the top of `_iter_stream_events`, initialise `ttft_ms: int | None = None`, and assign it on the first `message_delta`:

```python
    if response:
        for index in range(0, len(response), 24):
            if ttft_ms is None:
                ttft_ms = int((time.monotonic() - started) * 1000)
            yield StreamEvent(event="message_delta", data={"text": response[index : index + 24]})
    else:
        ttft_ms = int((time.monotonic() - started) * 1000)
        yield StreamEvent(event="message_delta", data={"text": ""})
```

- [ ] **Step 4: Run the backend suite**

Run: `cd /workspace && python3 -m pytest trainer/demo_api/tests -q`

Expected: all pass, including the new streaming test.

---

### Task 6: Frontend contract extension

**Files:**
- Modify: `demo-ui/src/contracts.ts`
- Modify: `demo-ui/tests/contracts.test.ts`

- [ ] **Step 1: Write the failing test**

Append to `demo-ui/tests/contracts.test.ts`:

```ts
import { streamEventSchema } from '../src/contracts'

describe('complete event timings', () => {
  it('accepts ttft_ms, synthesis_ms and total_ms', () => {
    const parsed = streamEventSchema.safeParse({
      event: 'complete',
      data: {
        tool_call_count: 1,
        rag_chunk_count: 2,
        ttft_ms: 4200,
        synthesis_ms: 15000,
        total_ms: 61000,
        deadline_hit: false,
      },
    })
    expect(parsed.success).toBe(true)
  })

  it('accepts a null ttft_ms when generation failed before any token', () => {
    const parsed = streamEventSchema.safeParse({
      event: 'complete',
      data: {
        tool_call_count: 0,
        rag_chunk_count: 0,
        ttft_ms: null,
        synthesis_ms: 0,
        total_ms: 900,
        deadline_hit: false,
      },
    })
    expect(parsed.success).toBe(true)
  })
})
```

- [ ] **Step 2: Run it and confirm failure**

Run: `cd /workspace/demo-ui && npm test -- --run`

Expected: FAIL — the strict `complete` variant rejects the unknown keys.

- [ ] **Step 3: Extend the schema**

In `demo-ui/src/contracts.ts`, the `complete` member of the discriminated union becomes:

```ts
const streamCompleteSchema = z
  .object({
    event: z.literal('complete'),
    data: z
      .object({
        tool_call_count: z.number().int().nonnegative(),
        rag_chunk_count: z.number().int().nonnegative(),
        ttft_ms: z.number().int().nonnegative().nullable().optional(),
        synthesis_ms: z.number().int().nonnegative().optional(),
        total_ms: z.number().int().nonnegative().optional(),
        deadline_hit: z.boolean().optional(),
      })
      .strict(),
  })
  .strict()
```

The timing fields are optional so a backend that predates this change still validates.

- [ ] **Step 4: Run frontend gates**

Run: `cd /workspace/demo-ui && npm test -- --run && npm run build:web`

Expected: all tests pass; build succeeds.

---

### Task 7: Latency bench harness

**Files:**
- Create: `trainer/scripts/bench_latency.py`
- Create: `trainer/scripts/tests/test_bench_latency.py`

- [ ] **Step 1: Write the failing test for the pure logic**

Create `trainer/scripts/tests/test_bench_latency.py`:

```python
from trainer.scripts.bench_latency import classify_path, percentiles, summarize


def test_classify_path_uses_actual_tool_calls_not_route() -> None:
    assert classify_path(0) == "rag"
    assert classify_path(1) == "tools"
    assert classify_path(3) == "tools"


def test_percentiles_are_inclusive_and_sorted() -> None:
    got = percentiles([10.0, 20.0, 30.0, 40.0])
    assert got["p50"] == 30.0
    assert got["max"] == 40.0


def test_summarize_flags_budget_breaches() -> None:
    rows = [
        {"q_id": "A1", "path": "rag", "ttft_ms": 12000, "total_ms": 20000},
        {"q_id": "A2", "path": "rag", "ttft_ms": 41000, "total_ms": 60000},
        {"q_id": "A3", "path": "tools", "ttft_ms": 5000, "total_ms": 200000},
    ]
    out = summarize(rows)
    assert out["rag"]["breaches"] == ["A2"]
    assert out["tools"]["breaches"] == ["A3"]
```

- [ ] **Step 2: Run it and confirm failure**

Run: `cd /workspace && python3 -m pytest trainer/scripts/tests/test_bench_latency.py -q`

Expected: FAIL — module not found.

- [ ] **Step 3: Implement the pure logic plus the SSE driver**

Create `trainer/scripts/bench_latency.py` with an argparse CLI (`--api-url`, `--eval-set`, `--profile {rag,tools,mixed}`, `--repeat`, `--output-root`, `--log-level`), logging, and type hints, matching the style of the other `trainer/scripts` entry points.

Required pure functions:

```python
RAG_BUDGET_MS = 30_000
TOOL_BUDGET_MS = 180_000


def classify_path(tool_call_count: int) -> str:
    return "rag" if tool_call_count == 0 else "tools"


def percentiles(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    def at(q: float) -> float:
        idx = min(int(len(ordered) * q), len(ordered) - 1)
        return ordered[idx]
    return {
        "p50": at(0.50),
        "p90": at(0.90),
        "p95": at(0.95),
        "p99": at(0.99),
        "max": ordered[-1],
    }


def summarize(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for path, budget, key in (("rag", RAG_BUDGET_MS, "ttft_ms"), ("tools", TOOL_BUDGET_MS, "total_ms")):
        subset = [r for r in rows if r["path"] == path]
        if not subset:
            out[path] = {"n": 0, "breaches": []}
            continue
        values = [float(r[key]) for r in subset if r.get(key) is not None]
        out[path] = {
            "n": len(subset),
            "metric": key,
            "budget_ms": budget,
            **percentiles(values),
            "within_budget": sum(1 for v in values if v <= budget),
            "breaches": [str(r["q_id"]) for r in subset if r.get(key) is not None and float(r[key]) > budget],
        }
    return out
```

The driver posts each eval question to `{api_url}/api/chat` with `httpx.stream`, records the wall-clock offset of the first `message_delta` frame as the observed TTFT, reads `ttft_ms`/`synthesis_ms`/`total_ms`/`deadline_hit` from the `complete` frame, classifies with `classify_path`, and writes `report.json` plus `SUMMARY.md` under `--output-root/<timestamp>/`.

`report.json` must include, per the spec: TTFT p50/p90/p99/max, total p50/p90/p99/max, `synthesis_ms` p50/p95/max, per-path budget compliance and breach lists, `deadline_hit` count, degraded count, and a per-question table.

- [ ] **Step 4: Run the logic tests**

Run: `cd /workspace && python3 -m pytest trainer/scripts/tests/test_bench_latency.py -q`

Expected: 3 passed.

- [ ] **Step 5: Check the CLI loads**

Run: `cd /workspace && python3 trainer/scripts/bench_latency.py --help`

Expected: usage text listing all options.

---

### Task 8: Measure, recalibrate, and judge the gates

**Files:**
- Modify: `trainer/configs/rag_config_fewshot_27b.yaml` (deadline recalibration only)

- [ ] **Step 1: Start MCP and the demo API in live mode**

```bash
cd /workspace
curl -sSf -m3 http://localhost:8100/health
CONFIG_PATH=trainer/configs/rag_config_fewshot_27b.yaml \
ADAPTER_PATH=/output/adapters/qwen36_27b_shipcoat_lora_v6_1 \
setsid nohup python3 -m trainer.demo_api.app --port 8130 --live \
  > /tmp/bench_api.log 2>&1 < /dev/null &
sleep 90
curl -sSf -m5 http://127.0.0.1:8130/api/health
```

Expected: `{"status":"ok",...}`. The 90s sleep covers the 27B load observed at ~20s plus adapter attach.

- [ ] **Step 2: Run the mixed profile**

```bash
cd /workspace && python3 trainer/scripts/bench_latency.py \
  --api-url http://127.0.0.1:8130 \
  --eval-set /data/eval/eval_set_agentic.jsonl \
  --profile mixed --repeat 1 \
  --output-root /output/eval_results/latency_bench
```

- [ ] **Step 3: Recalibrate the deadline from the measurement**

Read `synthesis_ms` p95 from the generated `report.json` and set `llm.tool_deadline_sec` to `180 - ceil(p95 / 1000)`. The spec requires this; the shipped 150 is provisional and unproven.

- [ ] **Step 4: Re-run and judge the three gates**

Re-run Step 2, then check:

| Gate | Pass condition |
|---|---|
| RAG TTFT | `rag.p90 <= 30000` and `rag.breaches == []` |
| Tool completion | `tools.p99 <= 180000` and `tools.breaches == []` |
| Quality | agentic total ≥ 6.60 |

For the quality gate:

```bash
cd /workspace && python3 trainer/scripts/generate_agentic_responses.py \
  --eval-set /data/eval/eval_set_agentic.jsonl \
  --config trainer/configs/rag_config_fewshot_27b.yaml \
  --adapter /output/adapters/qwen36_27b_shipcoat_lora_v6_1 \
  --mcp-url http://localhost:8100 \
  --output /tmp/gate/agentic_responses.jsonl --seed 42
python3 trainer/scripts/run_agentic_eval.py \
  --responses /tmp/gate/agentic_responses.jsonl \
  --output-dir /tmp/gate \
  --tool-schemas-dir trainer/configs/tool_schemas
```

Expected: `average_total_score >= 6.60`. If it drops below, revert `max_new_tokens_tool` to `1024` and re-measure; the token cap is the prime suspect because it can truncate before the `<tool_call>` tag.

- [ ] **Step 5: Stop the API**

```bash
pkill -f "trainer.demo_api.app" ; sleep 2 ; pgrep -af "trainer.demo_api.app" || echo STOPPED
```

- [ ] **Step 6: Decide stage 2**

If the RAG TTFT gate fails, the streaming path in Task 4 is still routing through `run()` and the real fix is either finishing the `_stream_generate` swap or installing `flash-linear-attention` + `causal-conv1d`. Record the measured numbers in `PATCH_NOTE.md` and open stage 2 per the spec. Do not install kernels before this measurement exists.

---

## Final Self-Review

- [x] Spec section 6.1-A (streaming) → Task 4
- [x] Spec section 6.1-B (ttft/synthesis/total instrumentation) → Tasks 1, 3, 4, 5
- [x] Spec section 6.1-C (deadline guard, explicit status) → Tasks 1, 3, 4
- [x] Spec section 6.1-D (round-differentiated token caps) → Tasks 2, 3
- [x] Spec section 6.1-E (bench harness on the real HTTP path) → Task 7
- [x] Spec section 6.3 (error handling: deadline, fallback, timeouts) → Tasks 1, 4
- [x] Spec section 6.4 (three gates) → Task 8
- [x] Spec section 8 (frontend Zod extension) → Task 6
- [x] Deadline recalibration mandated by the spec → Task 8 Step 3
- [x] Names consistent across tasks: `deadline_sec`, `tool_max_tokens`, `deadline_hit`, `synthesis_ms`, `ttft_ms`, `total_ms`, `classify_path`, `percentiles`, `summarize`
- [x] No placeholders; every code step carries real code
- [x] `run()` left untouched so the frozen eval baseline stays comparable

**Known limitation recorded deliberately:** Task 4 ships the streaming *contract* while still delegating to `run()`, so it measures TTFT without yet improving it. That is intentional — the spec's whole argument is that TTFT has never been measured, and the decision to invest in real token streaming or kernel installation should follow the number, not precede it. Task 8 Step 6 is the branch point.
