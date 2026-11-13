#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SessionMetrics(BaseModel):
    avg_dft: float
    dft_values: list[float]
    avg_speed: float
    speed_variance: float
    avg_distance: float
    distance_variance: float
    avg_angle: float
    angle_consistency: float = Field(ge=0.0, le=1.0)
    sequence_correct: bool
    under_thickness_count: int = 0
    over_thickness_count: int = 0


class SessionSummary(BaseModel):
    session_id: str
    scenario_id: str
    mode: str
    zone_id: str
    zone_type: str
    metrics: SessionMetrics
    env: dict[str, Any] = {}
    timestamp: str = ""


class CoachingResponse(BaseModel):
    session_id: str
    summary: str
    action_items: list[dict[str, str]]
    risk_flags: list[dict[str, Any]]
    citations: list[str]
    tool_results: list[dict[str, Any]]
    confidence: float
    model_name: str
    timestamp: str
    raw_response: str


def compute_risk_flags(metrics: SessionMetrics) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []

    if metrics.under_thickness_count > 0:
        sev = "high" if metrics.under_thickness_count >= 3 else ("medium" if metrics.under_thickness_count >= 2 else "low")
        flags.append({"flag": "under_thickness", "severity": sev, "value": metrics.under_thickness_count, "threshold": 0})

    if metrics.over_thickness_count > 0:
        sev = "high" if metrics.over_thickness_count >= 3 else ("medium" if metrics.over_thickness_count >= 2 else "low")
        flags.append({"flag": "over_thickness", "severity": sev, "value": metrics.over_thickness_count, "threshold": 0})

    if metrics.distance_variance > 0.3:
        sev = "high" if metrics.distance_variance > 0.6 else ("medium" if metrics.distance_variance > 0.45 else "low")
        flags.append({"flag": "distance_variance", "severity": sev, "value": round(metrics.distance_variance, 3), "threshold": 0.3})

    if metrics.speed_variance > 0.15:
        sev = "high" if metrics.speed_variance > 0.3 else ("medium" if metrics.speed_variance > 0.22 else "low")
        flags.append({"flag": "speed_variance", "severity": sev, "value": round(metrics.speed_variance, 3), "threshold": 0.15})

    if metrics.angle_consistency < 0.7:
        sev = "high" if metrics.angle_consistency < 0.4 else ("medium" if metrics.angle_consistency < 0.55 else "low")
        flags.append({"flag": "angle_consistency_low", "severity": sev, "value": round(metrics.angle_consistency, 3), "threshold": 0.7})

    if not metrics.sequence_correct:
        flags.append({"flag": "sequence_violation", "severity": "high", "value": 0, "threshold": 1})

    return flags


def build_coaching_instruction(session: SessionSummary, risk_flags: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    parts.append(f"세션 {session.session_id} ({session.zone_id}, {session.zone_type})의 코칭 피드백을 생성해줘.")

    if risk_flags:
        parts.append("\n감지된 위험 플래그:")
        for f in risk_flags:
            parts.append(f"  - {f['flag']}: severity={f['severity']}, value={f['value']}, threshold={f['threshold']}")

    parts.append(f"\nDFT 측정값: {session.metrics.dft_values}")
    parts.append("이 측정값으로 DFT 합부 판정을 수행하고, 이상이 있으면 anomaly_detection도 호출해줘.")

    if session.env:
        parts.append(f"환경 조건: 온도={session.env.get('temperature', 'N/A')}°C, 습도={session.env.get('humidity', 'N/A')}%")

    parts.append("\n최종 코칭 피드백에는 다음을 포함해줘:")
    parts.append("1. 세션 종합 요약")
    parts.append("2. 우선순위별 행동항목 (최대 3개)")
    parts.append("3. 관련 규정/문서 근거")

    return "\n".join(parts)


def build_environment(session: SessionSummary) -> dict[str, Any]:
    env: dict[str, Any] = {}
    env["dft_measurements"] = session.metrics.dft_values
    env["spec"] = {"nominal": 100, "tolerance_rule": "90/10"}
    env["session_id"] = session.session_id
    env["zone_id"] = session.zone_id
    env["zone_type"] = session.zone_type
    env["speed_data"] = {
        "speed": session.metrics.avg_speed,
        "distance": session.metrics.avg_distance,
        "angle": session.metrics.avg_angle,
    }
    if session.env:
        env["env"] = session.env
    return env


_DOC_ID_RE = re.compile(r"DOC-[a-zA-Z0-9_-]+-C\d+")


def extract_citations(text: str) -> list[str]:
    return list(dict.fromkeys(_DOC_ID_RE.findall(text)))


def parse_action_items(text: str) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for match in re.finditer(r"(\d+)\.\s*(.+?)(?=\n\d+\.|\n\n|$)", text, re.DOTALL):
        action_text = match.group(2).strip()
        if not action_text:
            continue
        priority = "high" if len(items) == 0 else ("medium" if len(items) == 1 else "low")
        items.append({"action": action_text[:300], "priority": priority})
        if len(items) >= 3:
            break
    if not items and text.strip():
        items.append({"action": text.strip()[:300], "priority": "medium"})
    return items


def build_coaching_response(
    session: SessionSummary,
    risk_flags: list[dict[str, Any]],
    pipeline_result: dict[str, Any],
) -> CoachingResponse:
    text_guide = pipeline_result.get("text_guide", {})
    response_text = str(text_guide.get("body", "")) if text_guide else str(pipeline_result.get("response", ""))
    tool_calls = pipeline_result.get("tool_calls", [])

    citations = pipeline_result.get("citations", [])
    if not citations:
        citations = extract_citations(response_text)
        rag_chunks = pipeline_result.get("rag_chunks", [])
        for chunk in rag_chunks:
            cid = chunk.get("chunk_id", "")
            if cid and cid not in citations:
                citations.append(cid)

    action_items = parse_action_items(response_text)
    tool_results = [
        {"name": tc.get("name"), "arguments": tc.get("arguments"), "result": tc.get("result")}
        for tc in tool_calls
    ]

    high_count = sum(1 for f in risk_flags if f.get("severity") == "high")
    confidence = 0.8 if not risk_flags else (0.5 if high_count > 0 else 0.65)

    return CoachingResponse(
        session_id=session.session_id,
        summary=response_text[:500],
        action_items=action_items,
        risk_flags=risk_flags,
        citations=citations,
        tool_results=tool_results,
        confidence=confidence,
        model_name="Qwen3-4B",
        timestamp=datetime.now(timezone.utc).isoformat(),
        raw_response=response_text,
    )


def _ensure_scripts_path() -> None:
    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Session coaching response system using AgentPipeline.")
    parser.add_argument("--config", type=Path, default=Path("/workspace/trainer/configs/rag_config_fewshot.yaml"))
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--input", type=str, required=True, help="Path to session JSON or '-' for stdin.")
    parser.add_argument("--output", type=Path, default=None, help="Output path. Defaults to stdout.")
    parser.add_argument("--mcp-url", type=str, default="http://localhost:8100")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--dry-run", action="store_true", help="Compute risk flags only, skip LLM.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")

    if args.input == "-":
        raw = sys.stdin.read()
    else:
        input_path = Path(args.input)
        if not input_path.is_file():
            logger.error("Input file not found: %s", args.input)
            return 1
        raw = input_path.read_text(encoding="utf-8")

    try:
        session = SessionSummary.model_validate_json(raw)
    except Exception:
        logger.exception("Failed to parse session summary JSON")
        return 1

    risk_flags = compute_risk_flags(session.metrics)
    logger.info("Session %s: %d risk flags detected", session.session_id, len(risk_flags))
    for f in risk_flags:
        logger.info("  %s [%s] value=%s threshold=%s", f["flag"], f["severity"], f["value"], f["threshold"])

    if args.dry_run:
        output = {
            "session_id": session.session_id,
            "risk_flags": risk_flags,
            "instruction_preview": build_coaching_instruction(session, risk_flags),
        }
        result_json = json.dumps(output, ensure_ascii=False, indent=2)
    else:
        _ensure_scripts_path()
        agent_mod = importlib.import_module("agent_pipeline")
        AgentPipeline = agent_mod.AgentPipeline

        adapter_path = str(args.adapter) if args.adapter else None
        pipeline = AgentPipeline(
            config_path=args.config,
            adapter_path=adapter_path,
            mcp_base_url=args.mcp_url,
        )

        instruction = build_coaching_instruction(session, risk_flags)
        environment = build_environment(session)

        logger.info("Running pipeline for session %s", session.session_id)
        pipeline_result = pipeline.run(
            instruction=instruction,
            environment=environment,
            session_id=session.session_id,
            enable_thinking=True,
        )

        coaching = build_coaching_response(session, risk_flags, pipeline_result)
        result_json = coaching.model_dump_json(indent=2)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result_json, encoding="utf-8")
        logger.info("Wrote coaching response to %s", args.output)
    else:
        print(result_json)

    return 0


if __name__ == "__main__":
    sys.exit(main())
