#!/usr/bin/env python3
"""Build sft v6: reformat v5 tool-using assistant answers to [분석]/[판정]/[원인]/[조치]/[근거] structure."""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_TOOL_CALL = re.compile(r"<tool_call>.*?(?:</tool_call>|(?=\n\n)|$)", flags=re.DOTALL)
_DOC = re.compile(r"DOC-[\w-]+-C\d+")
_VERDICT_POS = re.compile(r"정상|합격|적합|이상\s*없|양호")
_VERDICT_NEG = re.compile(r"불합격|부적합|이상\s*(?:탐지|발견|있)|결함|미달|박리|저도막|과도막")
_BRACKET = re.compile(r"\[(?:분석|판정|원인|조치|근거)\]")


def _extract_tool_call_text(answer: str) -> str:
    calls = _TOOL_CALL.findall(answer)
    return "\n".join(c if c.strip().endswith("</tool_call>") else c.strip() + "</tool_call>" for c in calls)


def _reformat_answer(answer: str) -> str:
    if _BRACKET.search(answer):
        return answer

    tool_text = _extract_tool_call_text(answer)
    prose = _TOOL_CALL.sub("", answer).strip()
    prose = re.sub(r"근거 문서 ID\s*:\s*", "", prose)
    docs = _DOC.findall(answer)

    verdict = "정상"
    if _VERDICT_NEG.search(prose):
        verdict = "이상 탐지"
    elif _VERDICT_POS.search(prose):
        verdict = "정상"

    body_lines = [ln.strip() for ln in prose.split("\n") if ln.strip() and not _DOC.search(ln)]
    body_lines = [ln for ln in body_lines if not re.match(r"^(판정 결과|분석 결과)", ln)]
    analysis = " ".join(body_lines[:2]) if body_lines else "도구 결과를 종합해 분석했습니다."
    cause = body_lines[2] if len(body_lines) > 2 else "측정값 편차 또는 작업 조건 이탈이 확인되었습니다."
    actions = body_lines[3:5] if len(body_lines) > 3 else body_lines[2:4] if len(body_lines) > 2 else []

    parts = []
    if tool_text:
        parts.append(tool_text + "\n")
    parts.append(f"[분석] {analysis}")
    parts.append(f"[판정] {verdict}")
    parts.append(f"[원인] {cause}")
    parts.append("[조치]")
    if actions:
        for i, act in enumerate(actions, 1):
            parts.append(f"{i}. {act}")
    elif verdict == "정상":
        parts.append("1. 현재 상태를 유지하고 정기 점검을 계속하십시오.")
    else:
        parts.append("1. 해당 구역을 재작업 후 재측정하십시오.")
    if docs:
        parts.append(f"[근거] 참조 문서 ID: {docs[0]}")
    return "\n".join(parts)


_TOOL_TO_CATEGORY = {
    "dft_judge": "tool_calling",
    "anomaly_detection": "anomaly_interpretation",
    "sequence_validation": "sequence_decision",
    "location_tracking": "sequence_decision",
    "pose_estimation": "cot_reasoning",
    "timeseries": "cot_reasoning",
    "vlm_analysis": "xr_content",
}


def _infer_category(record: dict[str, Any]) -> str:
    text = " ".join(m.get("content", "") for m in record.get("messages", []))
    tools = re.findall(r'"name"\s*:\s*"(\w+)"', text)
    if not tools:
        return "doc_qa"
    if len(set(tools)) >= 3:
        return "xr_content"
    return _TOOL_TO_CATEGORY.get(tools[0], "tool_calling")


def _load_eval_questions(path: Path) -> set[str]:
    questions: set[str] = set()
    if not path.is_file():
        return questions
    for line in path.open("r", encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        q = json.loads(line).get("question", "").strip()
        if len(q) >= 20:
            questions.add(q)
    return questions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reformat SFT v5 -> v6 with bracket structure + category kept.")
    parser.add_argument("--input", type=Path, default=Path("/data/sft/sft_train_v5.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("/data/sft/sft_train_v6.jsonl"))
    parser.add_argument("--eval-set", type=Path, default=Path("/data/eval/eval_set_agentic.jsonl"),
                        help="Eval set to decontaminate against (drops samples whose user turn contains an eval question).")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")

    records = [json.loads(l) for l in args.input.open("r", encoding="utf-8") if l.strip()]
    eval_questions = _load_eval_questions(args.eval_set)
    reformatted = 0
    decontaminated = 0
    kept = 0
    with args.output.open("w", encoding="utf-8") as fh:
        for rec in records:
            msgs = rec.get("messages", [])
            user_text = " ".join(m.get("content", "") for m in msgs if m.get("role") == "user")
            if any(q in user_text for q in eval_questions):
                decontaminated += 1
                continue
            is_tool = any("<tool_call>" in m.get("content", "") for m in msgs)
            if is_tool and msgs and msgs[-1].get("role") == "assistant":
                new_answer = _reformat_answer(msgs[-1]["content"])
                if new_answer != msgs[-1]["content"]:
                    msgs[-1]["content"] = new_answer
                    reformatted += 1
            rec["category"] = _infer_category(rec)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1

    logger.info(
        "Kept %d, decontaminated %d (eval overlap), reformatted %d tool answers -> %s",
        kept, decontaminated, reformatted, args.output,
    )
    print(f"kept={kept} decontaminated={decontaminated} reformatted={reformatted} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
