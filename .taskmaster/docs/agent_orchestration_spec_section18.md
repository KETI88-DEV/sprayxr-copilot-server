
# 18. Agent Orchestration Specification

## 18.1 Purpose
The AI Co-Pilot system separates deterministic analytical computation from language-based reasoning.
Analytical modules (trajectory, pose, thickness, etc.) perform numerical evaluation, while the LLM
interprets results and generates human-readable guidance.

This design improves:
- reliability
- traceability
- hallucination resistance
- deterministic evaluation

The orchestration layer coordinates interactions between user instructions, analysis tools, and the language model.

---

## 18.2 Orchestration Architecture

System execution pipeline:

User Instruction  
→ Intent Router  
→ Orchestrator  
→ Tool Selection  
→ Tool Execution  
→ Result Aggregation  
→ LLM Explanation  
→ Final Response

Key rules:

1. Numerical analysis must always be performed by tools.
2. The LLM must not invent numerical values.
3. All tool calls must be logged.

---

## 18.3 Intent Router

The router determines which analytical modules must be executed.

### Router Inputs

- User instruction
- Session metadata
- Available simulator data

### Router Outputs

- intent classification
- required tools
- fallback strategy

Example:

Instruction:
Analyze whether the current spray posture is correct.

Router Output:

intent: pose_feedback
required_tools:
- pose_analyzer
- rag_retriever

---

## 18.4 Tool Registry

The orchestrator can access the following tools.

| Tool | Purpose |
|-----|-----|
| rag_retriever | Retrieve painting guidelines and SOP documents |
| trajectory_analyzer | Evaluate spray trajectory patterns |
| pose_analyzer | Analyze worker spray posture |
| thickness_heatmap | Analyze coating thickness distribution |
| sequence_analyzer | Evaluate painting sequence correctness |
| paint_usage_analyzer | Analyze paint consumption efficiency |
| training_session_analyzer | Evaluate XR training session results |
| analytics_query | Query simulator data |
| report_generator | Generate training reports |

All tools must return structured results.

---

## 18.5 Tool Call Schema

Example tool invocation:

{
  "tool": "trajectory_analyzer",
  "arguments": {
    "session_id": "S2031"
  }
}

---

## 18.6 Tool Result Schema

Example result:

{
 "trajectory_deviation_cm": 18,
 "recommended_deviation_cm": 5,
 "spray_speed": 0.82,
 "recommended_speed_range": [0.5, 0.6]
}

LLM responses must reference these results and must not modify them.

---

## 18.7 Orchestrator Decision Process

Decision steps:

1. Identify user intent
2. Select required tools
3. Execute tools
4. Aggregate results
5. Generate explanation with LLM

Example:

User request:
Why is coating quality poor in this session?

Execution plan:
- trajectory_analyzer
- pose_analyzer
- thickness_heatmap

---

## 18.8 Multi‑Tool Execution

Some tasks require multiple tools.

Example:

User request:
Analyze the reason for coating quality problems.

Tools executed:

- trajectory_analyzer
- pose_analyzer
- thickness_heatmap

The orchestrator aggregates results before producing the final answer.

---

## 18.9 Fallback Strategy

If required data is unavailable:

1. Skip analytical tool
2. Use RAG retrieval
3. Provide guideline-based recommendations

Example:

Pose data unavailable → Use RAG posture guideline.

---

## 18.10 Execution Trace Logging

All interactions generate a trace record.

Trace schema:

{
 "instruction": "...",
 "intent": "...",
 "selected_tools": [...],
 "tool_calls": [...],
 "tool_results": [...],
 "final_response": "..."
}

Trace data is used for debugging and dataset generation.

---

## 18.11 Training Dataset Generation

Execution traces can be converted into training examples.

Dataset format:

instruction  
context  
orchestrator_decision  
tool_calls  
tool_results  
final_answer

This dataset supports:

- instruction tuning
- agent fine‑tuning
- evaluation benchmarks

---

## 18.12 Design Constraints

The orchestration system must satisfy:

1. Analytical modules must be deterministic.
2. LLM responses must reference tool outputs.
3. All responses must remain interpretable to operators.
4. Tool latency must support real‑time feedback.

---

## 18.13 Implementation Options

Possible orchestration frameworks:

- LangGraph
- MCP tool interface
- Open WebUI agent tool calling
- custom lightweight router

All implementations must support on‑premise deployment.

---

## 18.14 Expected Benefits

The orchestration architecture provides:

- reduced hallucination
- transparent reasoning
- modular extensibility
- reliable industrial AI deployment

