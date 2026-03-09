# Code Modification Prompt

## Role
Act as a careful software engineer making a **small modification to existing code**.

## Objective
Modify the provided code to implement the requested change while:

- keeping the existing structure intact
- minimizing unnecessary changes
- preserving readability and functionality

Do not rewrite the entire code unless absolutely necessary.

---

## Instructions

1. Carefully read the existing code.
2. Identify the **exact location** that needs modification.
3. Change **only the minimal amount of code required**.
4. Ensure the change does not introduce bugs or break existing logic.

---

## Output Requirements

Provide the result in the following format:

### Explanation
Briefly explain:
- what was changed
- why the change was necessary

### Modified Code
Return the updated code.

### Optional
If helpful, highlight the modified lines.

---

## Rules

- Do **not restructure the whole program**
- Do **not rename variables unnecessarily**
- Do **not introduce unrelated improvements**
- Keep the change focused on the requested modification