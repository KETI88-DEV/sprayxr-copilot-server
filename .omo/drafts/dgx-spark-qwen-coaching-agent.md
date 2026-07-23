---
slug: dgx-spark-qwen-coaching-agent
status: awaiting-approval
intent: clear
review_required: false
pending-action: write .omo/plans/dgx-spark-qwen-coaching-agent.md
approach: Keep training and fine-tuning on the current server, export a merged and validated GGUF artifact, and run inference on DGX Spark through Ollama; use Qwen3.5-9B as the recommended production candidate, retain Qwen3-4B as rollback, and evaluate Gemma 4 E4B as an A/B candidate.
---

# Draft: dgx-spark-qwen-coaching-agent

## Components (topology ledger)
<!-- Lock the SHAPE before depth. One row per top-level component that can succeed or fail independently. -->
<!-- id | outcome (one line) | status: active|deferred | evidence path -->
| id | outcome | status | evidence path |
|---|---|---|---|
| C1 | The current training server produces a versioned merged-model GGUF artifact, manifest, and rollback package that Ollama can load deterministically | active | `trainer/scripts/train_qlora.py`; `trainer/scripts/run_qlora_eval.py`; Ollama import documentation |
| C2 | FAISS/BGE retrieval and reranking load from explicit artifacts and remain quality-equivalent on Spark | active | `trainer/scripts/agent_pipeline.py:424-442`; `trainer/scripts/search_index.py:24-165` |
| C3 | MCP tool contracts and Unity-oriented structured response fields are correct end to end | active | `mcp_servers/common.py:10-53`; `trainer/scripts/agent_pipeline.py:451-549` |
| C4 | A typed coaching API exposes the orchestrator to Unity while Gradio remains an optional demo client | active | `trainer/scripts/demo_app.py:642-820`; no coaching API route exists in repository |
| C5 | DGX Spark runs Ollama plus ARM64-compatible coaching/RAG/MCP services with readiness, persistence, and pinned dependencies; no training stack is installed on Spark | active | `trainer/Dockerfile:1-45`; `deploy/docker-compose.yml:1-24` |
| C6 | Spark-native regression, quality, latency, memory, failure, and soak gates provide deployment evidence | active | `trainer/scripts/run_agentic_eval.py`; `trainer/scripts/run_qlora_eval.py`; `PATCH_NOTE.md:3336-3419` |

## Open assumptions (announced defaults)
<!-- Record any default you adopt instead of asking, so the user can veto it at the gate. -->
<!-- assumption | adopted default | rationale | reversible? -->
| assumption | adopted default | rationale | reversible? |
|---|---|---|---|
| Model baseline | Recommend Qwen3.5-9B as the production candidate; keep the current Qwen3-4B/Iter17 result as the rollback and parity baseline; run Gemma 4 E4B as an A/B candidate | Qwen3.5 has no 8B dense checkpoint; 9B is the matching current model and preserves Qwen-family continuity | yes |
| Serving architecture | Ollama is the only LLM inference runtime on DGX Spark; the application calls Ollama's native chat API or OpenAI-compatible endpoint and consumes structured `tool_calls` | This matches the confirmed operational environment and removes the need to port the training/HF inference stack to GB10 | yes |
| Artifact transfer | Merge the LoRA/QLoRA result into the exact base model on the training server, validate it, convert the fused model to GGUF, quantize if selected, and transfer it with a hash manifest | Ollama's documented direct Safetensors adapter list does not explicitly cover Qwen3.5 or Gemma 4, and quantized adapters can vary across frameworks | yes |
| Precision | Start the quality baseline at `Q8_0`, compare BF16 for regression detection, then admit `Q4_K_M` only if it passes the same Korean coaching/tool gate | 9B-class artifacts fit easily in 128 GB; Q4 is a throughput/concurrency option rather than a capacity necessity | yes |
| API surface | Add versioned HTTP JSON coaching endpoints plus `/live` and dependency-aware `/ready`; retain Gradio as a client | Unity cannot consume the current Python object/Gradio-only path | yes |
| Test strategy | TDD for existing contract defects and new service boundaries, followed by Spark-native image, runtime, quality, latency, and soak gates | Current targeted suites already demonstrate contract drift | yes |
| Training location | All fine-tuning, merge, export, and pre-deployment evaluation remain on the current server; DGX Spark performs inference and operational services only | This is the user-confirmed topology | yes |

## Findings (cited - path:lines)

- Hardware fit is positive: DGX Spark provides 128 GB coherent unified memory and an ARM64 GB10 Blackwell GPU. NVIDIA's current stack is DGX OS 7.5.0/CUDA 13.0.2, and native CUDA builds target SM 12.1. Official references: https://docs.nvidia.com/dgx/dgx-spark/hardware.html, https://docs.nvidia.com/dgx/dgx-spark/release-notes.html, https://docs.nvidia.com/dgx/dgx-spark-porting-guide/porting/compilation.html.
- The active runtime is Qwen3-4B, not Qwen3.5: `/models/Qwen3-4B`, HF backend, NF4, BF16 compute, and optional PEFT adapter are configured in `trainer/configs/rag_config_fewshot.yaml:27-39` and loaded by `trainer/scripts/agent_pipeline.py:332-364`.
- Qwen officially requires `transformers>=4.51.0` for Qwen3. The current unpinned Docker install cannot reproduce the README's stated versions. Official reference: https://huggingface.co/Qwen/Qwen3-4B.
- The base image in `trainer/Dockerfile:1` resolves to a single Linux `amd64` manifest, while DGX Spark is ARM64. The Dockerfile leaves all Python dependencies unpinned and omits runtime packages including Gradio and FAISS (`trainer/Dockerfile:13-41`).
- Compose is not a deployment: it publishes no ports, defines no health checks, hard-codes a developer host path, and runs `sleep infinity` (`deploy/docker-compose.yml:1-24`).
- No Unity-facing coaching endpoint exists. The network surfaces are MCP FastAPI routes (`mcp_servers/main.py:23-41`) and Gradio (`trainer/scripts/demo_app.py:898-962`); the latter only displays an XR JSON blob.
- A critical response-shape defect exists: MCP success payloads nest business data below `result` (`mcp_servers/common.py:10-53`), but Unity response assembly reads fields from the envelope (`trainer/scripts/agent_pipeline.py:513-548`).
- Live Gradio injects synthetic sensor data into every request (`trainer/scripts/demo_app.py:710-754`), which forces a tool route and prevents genuine live routing evaluation.
- Runtime and documentation have drifted: realtime is coded as 2 rounds (`trainer/scripts/agent_pipeline.py:97-108`) despite documented 3-round behavior; prompts advertise 6 tools while the runtime mounts 7 (`trainer/configs/rag_config_fewshot.yaml:51-70`; `mcp_servers/main.py:26-32`).
- Direct verification is red: `python3 -m unittest mcp_servers.test_integration -v` produced 6 failures of 15; `python3 trainer/scripts/test_tool_calling.py -v` produced 6 failures of 22. Failures cover stale health counts and anomaly/pose/sequence schemas.
- The current Qwen3.5 migration PRD is stale: it uses the Iter7 score 2.97 instead of the current Iter17 record, assumes A40/Transformers v5/Unsloth, and proposes `AutoModelForCausalLM` while the current official model card uses multimodal loading and says Qwen3.5 requires current main/nightly serving packages (`.taskmaster/docs/qwen35_migration_prd.md:636-731,824-923`). Official reference: https://huggingface.co/Qwen/Qwen3.5-4B.
- NVIDIA validates Qwen3 8B/14B/32B variants in its Spark vLLM matrix, not Qwen3-4B; therefore Qwen3-4B is capacity-safe but still requires a device-local load/LoRA/tool-call smoke gate. Official reference: https://build.nvidia.com/spark/vllm/i.
- The earlier vLLM-first recommendation did not fully account for the confirmed Ollama-only production runtime. The revised boundary is current server for training/export and DGX Spark for Ollama inference.
- Qwen3.5 has no official 8B dense checkpoint in Ollama; the applicable model is `qwen3.5:9b`. Ollama publishes approximately 6.6 GB for `q4_K_M`, 11 GB for `q8_0`, and 19 GB for BF16, with a 256K context window and text/image, tool, and thinking capabilities. Official references: https://registry.ollama.com/library/qwen3.5/tags and https://huggingface.co/Qwen/Qwen3.5-9B.
- Gemma 4 E4B is an effective 4.5B model with about 8B total parameters including embeddings. It supports 128K context, text/image/audio input, function calling, thinking, and system-role messages. Ollama publishes a 9.6 GB default E4B package and a 16 GB BF16 package. Official references: https://ai.google.dev/gemma/docs/core, https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/, and https://registry.ollama.com/library/gemma4/tags.
- Both candidates are far below DGX Spark's 128 GB capacity. Model fit is not the limiting factor at 9B/E4B; context/KV cache, concurrent sessions, RAG services, and end-to-end latency still require explicit budgets.
- Ollama supports importing a fused GGUF model and quantizing FP16/FP32 inputs during `ollama create`. Its documented direct Safetensors-adapter architectures list Llama, Mistral, and Gemma 1/2, but do not explicitly list Qwen3.5 or Gemma 4. Official references: https://docs.ollama.com/import and https://docs.ollama.com/modelfile.
- The current NF4 QLoRA adapter cannot be reused across architectures or sizes: Qwen3-4B to Qwen3.5-9B or Gemma 4 E4B requires a new fine-tuning run against that exact base. For Ollama delivery, the default must be a fused-model export rather than assuming direct adapter compatibility.
- Ollama exposes structured tool calls through its chat API. The coaching pipeline should pass the repository's JSON tool schemas and consume `message.tool_calls`, replacing raw-tag parsing as the production path. Official references: https://ollama.com/blog/tool-support and https://docs.ollama.com/api/chat.

## Decisions (with rationale)

- Verdict: DGX Spark is suitable hardware for this local/offline coaching workload, but the repository is not deployable on it unchanged.
- Preferred sequence: repair functional contracts -> freeze the Qwen3-4B quality baseline -> fine-tune and evaluate Qwen3.5-9B on the current server -> merge with the exact base -> convert/quantize to GGUF -> verify a local Ollama package -> transfer with hashes -> validate end to end on DGX Spark.
- Recommend Qwen3.5-9B over Gemma 4 E4B for the first production candidate because it preserves the current Qwen training/prompt lineage and offers a larger effective dense model, broader claimed multilingual coverage, longer context, and tool-oriented behavior. Keep Gemma 4 E4B as a measured alternative for multimodal/audio and edge-oriented priorities; do not claim Korean or tool-quality superiority for either until the matched evaluation passes.
- Retain Qwen3-4B as a deployable rollback baseline. A model-family upgrade and a DGX deployment must remain separately reversible even though they can share the same Ollama gateway.
- Do not use direct Ollama `ADAPTER` import as the default for Qwen3.5/Gemma 4 until an exact-version smoke test proves it. The release artifact is a fused GGUF plus Modelfile and SHA-256 manifest.
- Use artifact/config hashes for model, adapter, FAISS index, metadata, tool schemas, and dependency lock in every benchmark result.
- Replace the single-example latency report with cold/warm p50/p95/p99 tests, concurrency 1/2/4, deadline/fallback injection, unified-memory/RSS/swap observation, and a 4-8 hour soak.

## Scope IN

- Repair current MCP/tool/Unity response contracts and their tests.
- Add a typed versioned coaching API for Unity, request correlation, session handling, readiness, and graceful dependency failures.
- Create reproducible ARM64 deployment artifacts and pinned dependency sets for orchestration, RAG, and MCP; install and pin Ollama separately as the LLM runtime.
- Add an Ollama model gateway that supports structured tool calls, bounded context, timeouts, model selection, health/readiness, and deterministic rollback.
- Add training-server export automation for exact-base merge, merged-model quality verification, GGUF conversion/quantization, Modelfile generation, SHA-256 manifesting, and transfer verification.
- Produce and compare Qwen3-4B rollback, Qwen3.5-9B candidate, and Gemma 4 E4B A/B packages under the same coaching/tool/RAG evaluation contract.
- Parameterize all model/data/index/cache/output paths and remove developer-specific mounts.
- Add DGX Spark smoke, regression, quality, latency, memory, failure-path, and soak validation with machine-readable evidence.
- Refresh the Qwen3.5 migration document against Qwen3.5-9B, Iter17, and the Ollama/GGUF delivery contract.

## Scope OUT (Must NOT have)

- No training or fine-tuning on DGX Spark; its role is Ollama inference and operational services.
- No assumption that Qwen3.5-8B exists; use the official Qwen3.5-9B identifier.
- No reuse of the Qwen3-4B adapter on Qwen3.5-9B or Gemma 4 E4B.
- No direct quantized Safetensors adapter deployment to Ollama without a version-pinned compatibility and output-parity test.
- No deletion or overwriting of existing adapters, SFT data, indexes, or evaluation evidence.
- No claim that all seven MCP tools are production implementations; existing stubs/heuristics remain explicitly labeled unless separately authorized.
- No Kubernetes, multi-Spark clustering, speculative decoding, or multimodal/VLM expansion in the initial deployment.
- No acceptance based only on imports, image build success, `nvidia-smi`, grep output, or a single latency sample.

## Open questions

1. Owner decision: approve Qwen3.5-9B as the primary production candidate, Qwen3-4B as rollback, and Gemma 4 E4B as a mandatory A/B comparison? Recommended: yes.
2. Owner decision: approve fused-GGUF transfer as the production artifact contract, with direct Ollama adapter loading treated only as an optional compatibility experiment? Recommended: yes.
3. Owner decision: which latency contract should govern the plan? Recommended production profile: fast p95 < 3 s, training_realtime p95 < 10 s, training_review p95 < 30 s; fallback PoC-parity profile: fast < 10 s, realtime < 60 s, review without a hard limit.

## Approval gate
status: awaiting-approval
approved approach: pending user decision
next workflow action: write `.omo/plans/dgx-spark-qwen-coaching-agent.md` only after explicit approval; do not implement.
<!-- When exploration is exhausted and unknowns are answered, set status: awaiting-approval. -->
<!-- That durable record is the loop guard: on a later turn read it and resume at the gate instead of re-running exploration. -->
