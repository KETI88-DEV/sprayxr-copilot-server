#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def _ensure_scripts_path() -> None:
    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)


def _resolve_adapter_path(adapter_arg: Path | None, qlora_config_path: Path) -> str | None:
    if adapter_arg is not None:
        return str(adapter_arg)
    if not qlora_config_path.is_file():
        return None

    with qlora_config_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    adapter_dir = cfg.get("output", {}).get("adapter_dir")
    if isinstance(adapter_dir, str) and adapter_dir:
        return adapter_dir
    return None


def _print_trace(trace: dict[str, Any]) -> None:
    print("-" * 80)
    print(f"Scenario: {trace['scenario']}")
    print(f"Mode: {trace.get('mode', 'N/A')}")
    print(f"Instruction: {trace['instruction']}")
    print(f"Latency: {trace.get('metadata', {}).get('latency_ms', 'N/A')}ms")

    print("Text Guide:")
    tg = trace.get("text_guide", {})
    if tg:
        print(f"  Language: {tg.get('language')}, Detail: {tg.get('detail_level')}")
        print(f"  Summary: {tg.get('summary', '')[:200]}")

    tools_used = trace.get("metadata", {}).get("tools_used", [])
    print(f"Tools Used: {', '.join(tools_used) if tools_used else '(none)'}")

    for section_name in ("pose_comparison", "next_path", "dft_result", "anomaly"):
        section = trace.get(section_name)
        if section is not None:
            print(f"{section_name}:")
            print(f"  {json.dumps(section, ensure_ascii=False)}")

    if trace.get("steps"):
        print(f"Steps: {json.dumps(trace['steps'], ensure_ascii=False, indent=2)}")

    print(f"Citations: {trace.get('citations', [])}")
    print("Response:")
    print(trace.get("text_guide", {}).get("body", trace.get("response", "")))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PRD 16.4 scenarios D/E/F with AgentPipeline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("/workspace/trainer/configs/rag_config.yaml"),
        help="Path to rag_config.yaml.",
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        default=None,
        help="Optional QLoRA adapter path.",
    )
    parser.add_argument(
        "--qlora-config",
        type=Path,
        default=Path("/workspace/trainer/configs/qlora_config.yaml"),
        help="Path to qlora_config.yaml for adapter fallback.",
    )
    parser.add_argument(
        "--mcp-url",
        type=str,
        default="http://localhost:8100",
        help="MCP server base URL.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/output/eval_results/demo_agent_traces.json"),
        help="Path to write scenario traces JSON.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if not args.config.is_file():
        logger.error("Config not found: %s", args.config)
        return 1

    adapter_path = _resolve_adapter_path(args.adapter, args.qlora_config)

    _ensure_scripts_path()
    agent_mod = importlib.import_module("agent_pipeline")
    AgentPipeline = agent_mod.AgentPipeline

    try:
        pipeline = AgentPipeline(
            config_path=args.config,
            adapter_path=adapter_path,
            mcp_base_url=args.mcp_url,
        )
    except Exception:
        logger.exception("Failed to initialize AgentPipeline")
        return 1

    scenarios: list[dict[str, Any]] = [
        {
            "scenario": "D",
            "instruction": "지금 내 도장 상태 어때?",
            "environment": {
                "current_pose": {
                    "joints": {
                        "wrist": {"x": 1.2, "y": 0.8, "z": 1.5},
                        "elbow": {"x": 0.8, "y": 1.0, "z": 1.2},
                    }
                },
                "reference_pose": {
                    "joints": {
                        "wrist": {"x": 1.0, "y": 1.0, "z": 1.5},
                        "elbow": {"x": 0.8, "y": 0.9, "z": 1.2},
                    }
                },
                "trajectory": [
                    {"x": 0, "y": 5, "z": 0, "timestamp": 1.0},
                    {"x": 0, "y": 3, "z": 0, "timestamp": 2.0},
                    {"x": 0, "y": 1, "z": 0, "timestamp": 3.0},
                ],
                "zone_id": "B-3",
                "zone_type": "수직 패널",
                "session_id": "demo-session-1",
                "speed_data": {"timestamp": 100, "speed": 0.48, "distance": 2.0, "angle": 15.0},
            },
            "session_id": "demo-session-1",
        },
        {
            "scenario": "E",
            "instruction": "도막 상태에 이상이 있나?",
            "environment": {
                "image_context": "좌하단 B-3-L 구역 청색 집중, 전체 대비 저도막 패턴",
                "dft_measurements": [150, 85, 92, 110, 78],
                "env": {"temperature": 28, "humidity": 65},
                "spec": {"nominal": 100, "tolerance_rule": "90/10"},
                "session_id": "demo-session-2",
            },
            "session_id": "demo-session-2",
        },
        {
            "scenario": "F",
            "instruction": "웰드 부위 스트라이프 코팅 방법 보여줘",
            "environment": {},
            "session_id": "demo-session-3",
        },
    ]

    traces: list[dict[str, Any]] = []
    for scenario in scenarios:
        try:
            result = pipeline.run(
                instruction=scenario["instruction"],
                environment=scenario["environment"],
                session_id=scenario["session_id"],
                enable_thinking=True,
            )
        except Exception:
            logger.exception("Scenario %s failed", scenario["scenario"])
            return 1

        trace = {
            "scenario": scenario["scenario"],
            "instruction": scenario["instruction"],
            "environment": scenario["environment"],
            "mode": result.get("mode"),
            "text_guide": result.get("text_guide"),
            "pose_comparison": result.get("pose_comparison"),
            "next_path": result.get("next_path"),
            "dft_result": result.get("dft_result"),
            "anomaly": result.get("anomaly"),
            "steps": result.get("steps"),
            "citations": result.get("citations", []),
            "metadata": result.get("metadata", {}),
            "session_id": result["session_id"],
            "response": result.get("response", ""),
        }
        traces.append(trace)
        _print_trace(trace)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(traces, fh, ensure_ascii=False, indent=2)

    logger.info("Wrote demo traces to %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
