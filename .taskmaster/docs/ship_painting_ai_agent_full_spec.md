
# Ship Painting AI Co-Pilot
# Agent Orchestration, Intent Router, Tool Schema and Dataset Specification

---

# 1. Intent Router Design

## Purpose
The intent router classifies user instructions and determines which analytical tools should be executed.

## Inputs

- user instruction
- simulator session metadata
- available data types

## Outputs

- intent classification
- required tools
- fallback strategy

## Primary Intents

| Intent | Description |
|------|-------------|
| pose_feedback | Evaluate spray posture |
| trajectory_analysis | Analyze spray trajectory |
| thickness_analysis | Analyze coating thickness |
| sequence_validation | Validate painting sequence |
| paint_efficiency | Evaluate paint usage |
| training_feedback | XR training session analysis |
| quality_diagnosis | Diagnose coating quality |
| knowledge_query | Retrieve guideline knowledge |

## Example Routing

Instruction:
Analyze whether the spray posture is correct.

Router Output:

intent: pose_feedback  
tools: pose_analyzer, rag_retriever

---

# 2. Tool JSON Schema

## trajectory_analyzer

```json
{
  "name": "trajectory_analyzer",
  "description": "Analyze spray trajectory patterns",
  "parameters": {
    "type": "object",
    "properties": {
      "session_id": { "type": "string" },
      "panel_id": { "type": "string" }
    },
    "required": ["session_id"]
  }
}
```

## pose_analyzer

```json
{
  "name": "pose_analyzer",
  "description": "Analyze worker spray posture",
  "parameters": {
    "type": "object",
    "properties": {
      "session_id": { "type": "string" }
    },
    "required": ["session_id"]
  }
}
```

---

# 3. LangGraph Orchestrator Architecture

## Node Types

| Node | Role |
|----|-----|
| Router Node | Determine intent |
| Tool Node | Execute analytical tools |
| Aggregator Node | Combine tool outputs |
| LLM Node | Generate explanation |

## Execution Flow

User Input  
→ Router Node  
→ Tool Nodes  
→ Aggregator  
→ LLM Explanation  
→ Final Response

---

# 4. Tool Calling Prompt Template

## System Prompt

You are an industrial AI co-pilot assisting ship painting operations.

### Rules

- Never invent numerical values
- Always reference tool results
- Call tools when analytical evaluation is required

### Response Structure

Thought: reasoning for tool selection

Tool Call: selected analytical tool

Tool Result: result returned by tool

Final Answer: explanation and recommendation

---

# 5. Execution Trace Schema

```json
{
  "instruction": "...",
  "intent": "...",
  "selected_tools": [],
  "tool_calls": [],
  "tool_results": [],
  "final_response": "..."
}
```

This trace supports:

- debugging
- evaluation
- dataset generation

---

# 6. Example Orchestration Dataset

## Example Row 1

Instruction

Evaluate whether spray posture is correct.

Intent

pose_feedback

Tool Call

pose_analyzer(session_id=S2031)

Tool Result

spray_angle: 72  
recommended_angle: 90

Final Answer

The spray gun angle is too shallow. Adjust the spray gun angle closer to 90 degrees.

---

## Example Row 2

Instruction

Why is coating thickness uneven?

Intent

quality_diagnosis

Tool Calls

trajectory_analyzer  
thickness_heatmap

Tool Results

trajectory deviation: 14 cm  
thickness variance: 65 μm

Final Answer

Uneven coating thickness is likely caused by unstable spray trajectory and inconsistent spray speed.

---

# 7. Design Constraints

1. Analytical computations must be deterministic.
2. LLM outputs must reference tool results.
3. Numerical values must originate from tools.
4. Tool latency must support near real-time feedback.

---

# 8. Expected Benefits

- reduced hallucination
- transparent reasoning
- modular architecture
- reliable industrial deployment

