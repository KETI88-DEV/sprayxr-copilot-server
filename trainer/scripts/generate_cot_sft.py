#!/usr/bin/env python3
# pyright: reportGeneralTypeIssues=false, reportExplicitAny=false, reportUnusedParameter=false, reportUnusedCallResult=false, reportAny=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportImplicitStringConcatenation=false
"""Generate Chain-of-Thought SFT dataset for ship painting copilot.

Creates 80 JSONL samples with required category distribution:
- 30 docqa_cot (no tool calls)
- 15 single_tool_cot (exactly 1 tool call)
- 15 multi_tool_cot (2-3 tool calls)
- 20 complex_tool_cot (4+ tool calls)
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, cast

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "너는 선박 도장 전문 AI Co-Pilot이다. 복잡한 질문에는 반드시 단계별로 사고하고, "
    "각 단계의 근거를 명시하라."
)

TOOLS = [
    "dft_judge",
    "anomaly_detection",
    "sequence_validation",
    "pose_estimation",
    "location_tracking",
    "timeseries",
    "vlm_analysis",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CoT SFT JSONL data")
    parser.add_argument(
        "--chunks",
        type=Path,
        default=Path("/data/chunked/chunks.jsonl"),
        help="Input chunk JSONL path",
    )
    parser.add_argument(
        "--tool-schemas-dir",
        type=Path,
        default=Path("/workspace/trainer/configs/tool_schemas"),
        help="Tool schema directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/data/sft/sft_cot.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Invalid JSON skipped: %s:%d", path, line_no)
    return items


def load_tool_schemas(tool_schemas_dir: Path) -> dict[str, dict[str, Any]]:
    schema_map: dict[str, dict[str, Any]] = {}
    for tool_name in TOOLS:
        schema_path = tool_schemas_dir / f"{tool_name}.json"
        raw = json.loads(schema_path.read_text(encoding="utf-8"))
        schema_map[tool_name] = raw["function"]["parameters"]
    return schema_map


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def sanitize_doc_excerpt(text: str, max_chars: int = 520) -> str:
    clean = normalize_text(text)
    if len(clean) <= max_chars:
        return clean
    return clean[:max_chars].rstrip() + " ..."


def validate_with_schema(value: Any, schema: dict[str, Any], path: str = "root") -> None:
    schema_type = schema.get("type")
    if schema_type == "object":
        if not isinstance(value, dict):
            raise ValueError(f"{path} must be object")
        required = schema.get("required", [])
        for key in required:
            if key not in value:
                raise ValueError(f"{path}.{key} is required")
        props = schema.get("properties", {})
        for k, v in value.items():
            if k in props:
                validate_with_schema(v, props[k], f"{path}.{k}")
    elif schema_type == "array":
        if not isinstance(value, list):
            raise ValueError(f"{path} must be array")
        item_schema = schema.get("items")
        if item_schema:
            for idx, item in enumerate(value):
                validate_with_schema(item, item_schema, f"{path}[{idx}]")
    elif schema_type == "string":
        if not isinstance(value, str):
            raise ValueError(f"{path} must be string")
        enum = schema.get("enum")
        if enum and value not in enum:
            raise ValueError(f"{path} must be one of {enum}")
    elif schema_type == "number":
        if not isinstance(value, (int, float)):
            raise ValueError(f"{path} must be number")
    elif schema_type == "boolean":
        if not isinstance(value, bool):
            raise ValueError(f"{path} must be boolean")


def to_tool_call(tool_name: str, arguments: dict[str, Any], schemas: dict[str, dict[str, Any]]) -> str:
    validate_with_schema(arguments, schemas[tool_name])
    payload = {"name": tool_name, "arguments": arguments}
    return f"<tool_call>{json.dumps(payload, ensure_ascii=False)}</tool_call>"


def make_sample(user_content: str, assistant_content: str, category_type: str) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "category_type": category_type,
    }


def pick_chunks_by_keywords(chunks: list[dict[str, Any]], keywords: list[str]) -> list[dict[str, Any]]:
    result = []
    for chunk in chunks:
        text = str(chunk.get("text", ""))
        if any(k.lower() in text.lower() for k in keywords):
            result.append(chunk)
    return result


def build_docqa_cot_samples(chunks: list[dict[str, Any]], rng: random.Random, count: int = 30) -> list[dict[str, Any]]:
    topic_cfg = [
        {
            "topic": "dft_spec",
            "keywords": ["DFT", "건조도막", "90/10", "두께", "um"],
            "q_templates": [
                "도막두께(DFT) 규격 판정 기준을 작업자가 바로 적용할 수 있게 설명해줘.",
                "90/10 규칙 기반 도막두께(DFT) 검사 포인트를 정리해줘.",
            ],
            "focus": "도막두께(DFT) 규격과 판정 기준",
        },
        {
            "topic": "paint_defect",
            "keywords": ["결함", "핀홀", "기포", "박리", "부식", "Sag", "오렌지"],
            "q_templates": [
                "문서 근거로 도장 결함(Paint Defect) 원인과 즉시 조치 절차를 알려줘.",
                "결함 유형별 점검 순서를 초급 작업자 교육용으로 정리해줘.",
            ],
            "focus": "도장 결함(Paint Defect) 원인과 조치",
        },
        {
            "topic": "safety",
            "keywords": ["안전", "보호구", "환기", "화재", "폭발", "MSDS", "가스"],
            "q_templates": [
                "안전 절차(Safety Procedure)에서 작업 전 반드시 확인할 항목을 알려줘.",
                "도장 작업 중 환기(Ventilation)와 보호구(PPE) 기준을 정리해줘.",
            ],
            "focus": "안전 절차(Safety Procedure)",
        },
        {
            "topic": "blasting",
            "keywords": ["블라스팅", "표면처리", "Sa", "조도", "연마재", "청정도"],
            "q_templates": [
                "블라스팅(Blasting) 후 표면처리 기준을 단계별로 설명해줘.",
                "표면 청정도(Surface Cleanliness)와 조도(Profile) 확인 절차를 알려줘.",
            ],
            "focus": "블라스팅(Blasting) 및 표면처리",
        },
        {
            "topic": "coating_thickness",
            "keywords": ["WFT", "DFT", "막두께", "건조", "측정", "두께계"],
            "q_templates": [
                "습도막두께(WFT)와 건조도막두께(DFT) 관리 포인트를 비교 설명해줘.",
                "도막두께(Coating Thickness) 편차 예방을 위한 관리 기준을 정리해줘.",
            ],
            "focus": "도막두께(Coating Thickness) 관리",
        },
    ]

    per_topic = count // len(topic_cfg)
    samples: list[dict[str, Any]] = []

    for cfg in topic_cfg:
        keywords = cast(list[str], cfg["keywords"])
        candidates = pick_chunks_by_keywords(chunks, keywords)
        if not candidates:
            candidates = chunks
        rng.shuffle(candidates)

        for idx in range(per_topic):
            chunk = candidates[idx % len(candidates)]
            chunk_text = str(chunk.get("text", ""))
            chunk_id = str(chunk.get("chunk_id", "DOC-unknown"))
            excerpt = sanitize_doc_excerpt(chunk_text)
            q_templates = cast(list[str], cfg["q_templates"])
            question = q_templates[idx % len(q_templates)]

            user = f"[문서] {excerpt}\n\n[질문] {question}"
            assistant = (
                "<think>\n"
                f"단계 1: 질문의 핵심은 {cfg['focus']}를 문서 근거로 정리하는 것이다.\n"
                f"단계 2: 제공 문서에서 관련 근거를 추출하면 \"{excerpt[:120]}\" 부분이 직접적인 단서다.\n"
                "단계 3: 추출한 근거를 현장 적용 관점으로 재구성해 작업 순서와 주의점을 분리한다.\n"
                "단계 4: 수치/규정 표현은 문서 범위를 벗어나지 않게 제한하고 실행 가능한 지시로 마무리한다.\n"
                "</think>\n"
                f"문서 기준으로 보면 {cfg['focus']}의 핵심은 다음과 같습니다. "
                "첫째, 작업 전 기준값과 점검 항목을 먼저 확정해야 합니다. "
                "둘째, 점검은 구역별로 동일 기준을 반복 적용해 편차를 줄여야 합니다. "
                "셋째, 이상 징후가 보이면 즉시 재점검 후 보수 절차를 진행해야 합니다.\n\n"
                f"근거 문서 ID: {chunk_id}"
            )
            samples.append(make_sample(user, assistant, "docqa_cot"))

    return samples


def dft_objects(values: list[float], zone_id: str) -> list[dict[str, Any]]:
    positions = ["upper", "middle", "lower", "left", "right", "center"]
    rows: list[dict[str, Any]] = []
    for i, v in enumerate(values):
        rows.append({"value_um": v, "zone_id": zone_id, "position": positions[i % len(positions)]})
    return rows


def make_zone_map(base_block: str, painted_idx: int = 0) -> list[dict[str, Any]]:
    return [
        {
            "zone_id": f"{base_block}-A1",
            "block_id": base_block,
            "zone_type": "수직 패널",
            "position": "upper",
            "is_painted": painted_idx >= 1,
            "area_m2": 24.0,
        },
        {
            "zone_id": f"{base_block}-A2",
            "block_id": base_block,
            "zone_type": "수직 패널",
            "position": "middle",
            "is_painted": painted_idx >= 2,
            "area_m2": 26.5,
        },
        {
            "zone_id": f"{base_block}-A3",
            "block_id": base_block,
            "zone_type": "모서리",
            "position": "lower",
            "is_painted": painted_idx >= 3,
            "area_m2": 8.5,
        },
    ]


def build_single_tool_cot_samples(rng: random.Random, schemas: dict[str, dict[str, Any]], count: int = 15) -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []

    dft_sets = [
        ([96, 101, 94, 98, 103], 100),
        ([82, 86, 79, 88, 84], 100),
        ([148, 152, 146, 151, 149], 150),
        ([112, 109, 107, 111, 108], 125),
        ([91, 93, 95, 97, 96], 100),
        ([88, 90, 85, 87, 89], 100),
        ([132, 128, 130, 131, 127], 125),
        ([72, 76, 80, 78, 74], 90),
        ([104, 102, 99, 101, 103], 100),
    ]
    for idx, (meas, nominal) in enumerate(dft_sets, start=1):
        scenarios.append(
            {
                "tool": "dft_judge",
                "args": {"measurements": meas, "spec": {"nominal": nominal, "tolerance_rule": "90/10"}},
                "user": {
                    "zone_id": f"B-{idx}-L",
                    "request": "도막두께(DFT) 합부를 판정하고 재작업 필요 여부를 알려줘.",
                },
                "result_hint": "규격 대비 하한/평균/편차를 우선 확인해 합부를 판단하고 재도장 범위를 제시",
            }
        )

    scenarios.extend(
        [
            {
                "tool": "anomaly_detection",
                "args": {
                    "dft_measurements": dft_objects([85, 91, 78, 82, 88], "VP-2"),
                    "zone_id": "VP-2",
                    "spec": {"nominal_um": 100, "tolerance_rule": "90/10"},
                    "image_context": "하부 좌측 저도막 히트맵 집중",
                },
                "user": {"zone_id": "VP-2", "request": "히트맵과 DFT 기반 이상 패턴을 확인해줘."},
                "result_hint": "저도막 군집 위치와 원인 후보를 분리해 보고",
            },
            {
                "tool": "sequence_validation",
                "args": {
                    "zone_map": make_zone_map("BLK-K1", painted_idx=1),
                    "current_zone_id": "BLK-K1-A2",
                    "structure_type": "block",
                },
                "user": {"zone_id": "BLK-K1", "request": "현재 도장 순서가 맞는지 검증해줘."},
                "result_hint": "미도장 우선순위와 이동 경로 비효율 여부를 확인",
            },
            {
                "tool": "pose_estimation",
                "args": {
                    "mode": "compare",
                    "current_pose": {
                        "joints": {
                            "wrist_r": {"x": 0.13, "y": -0.05, "z": 0.37, "roll": 7, "pitch": 19, "yaw": 8}
                        }
                    },
                    "block_id": "BLK-P2",
                },
                "user": {"zone_id": "BLK-P2", "request": "자세 편차를 숙련자 기준으로 코칭해줘."},
                "result_hint": "손목 각도와 분사 거리 편차를 중심으로 교정",
            },
            {
                "tool": "location_tracking",
                "args": {
                    "coordinates": [{"x": 412.0, "y": 118.0, "z": 276.0, "timestamp": 1712201101}],
                    "ship_model_id": "HULL-KR-310",
                },
                "user": {"zone_id": "unknown", "request": "현재 작업 구역과 커버리지 위험을 알려줘."},
                "result_hint": "현재 구역 식별 후 미도장 잔여 리스크를 제시",
            },
            {
                "tool": "timeseries",
                "args": {
                    "action": "update",
                    "session_id": "sess-cot-single-01",
                    "data": {"timestamp": 1712201160, "speed": 1.12, "distance": 325, "angle": 47},
                },
                "user": {"zone_id": "B-4-R", "request": "이번 패스를 시계열 추적에 반영해줘."},
                "result_hint": "속도/거리/각도 추세 편차 감시 포인트를 안내",
            },
            {
                "tool": "vlm_analysis",
                "args": {
                    "heatmap_path": "/data/sim/heatmaps/aq024_pass03.png",
                    "screenshot_path": "/data/sim/screenshots/aq024_pass03.jpg",
                    "report_json": {"zone_id": "AQ024-3", "note": "용접선 주변 색 편차"},
                },
                "user": {"zone_id": "AQ024-3", "request": "히트맵 시각 분석으로 결함 후보를 정리해줘."},
                "result_hint": "관심영역(ROI)과 의심 결함 타입을 우선 정리",
            },
        ]
    )

    rng.shuffle(scenarios)
    selected = scenarios[:count]
    samples: list[dict[str, Any]] = []

    for i, sc in enumerate(selected, start=1):
        tool_name = sc["tool"]
        tool_call = to_tool_call(tool_name, sc["args"], schemas)
        user_payload = {
            "zone_id": sc["user"]["zone_id"],
            "tool_input": sc["args"],
            "scenario_id": f"single-{i:02d}",
        }
        user = f"[환경 데이터]\n{json.dumps(user_payload, ensure_ascii=False)}\n\n질문: {sc['user']['request']}"
        assistant = (
            "<think>\n"
            "단계 1: 요청은 단일 의사결정 포인트를 빠르게 확인하는 성격이다.\n"
            f"단계 2: 입력 데이터 구조를 보면 {tool_name} 도구의 파라미터와 정확히 대응된다.\n"
            f"단계 3: 따라서 먼저 {tool_name}를 호출해 정량 근거를 확보하는 것이 타당하다.\n"
            "</think>\n"
            f"{tool_call}\n"
            "<think>\n"
            f"단계 4: 도구 결과에서 {sc['result_hint']}를 확인한다.\n"
            "단계 5: 확인된 결과를 작업자 행동 지시로 변환해 즉시 적용 가능하게 제시한다.\n"
            "</think>\n"
            f"분석 결과, 요청 구역에 대해 {tool_name} 기준 판정을 우선 적용해야 합니다. "
            "현장에서는 기준 이탈 구간을 먼저 표시하고 재측정-보수-재검증 순서로 진행하세요.\n\n"
            f"근거 문서 ID: DOC-COT-SINGLE-{i:04d}"
        )
        samples.append(make_sample(user, assistant, "single_tool_cot"))

    return samples


def build_multi_tool_cot_samples(rng: random.Random, schemas: dict[str, dict[str, Any]], count: int = 15) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []

    chains = [
        ["dft_judge", "anomaly_detection"],
        ["location_tracking", "sequence_validation"],
        ["timeseries", "dft_judge", "anomaly_detection"],
        ["location_tracking", "pose_estimation", "dft_judge"],
        ["vlm_analysis", "anomaly_detection"],
    ]

    for i in range(count):
        chain = chains[i % len(chains)]
        nominal = 100 + (i % 3) * 25
        meas = [nominal - 12 + (j * 4) for j in range(5)]

        tool_args: dict[str, dict[str, Any]] = {
            "dft_judge": {"measurements": meas, "spec": {"nominal": nominal, "tolerance_rule": "90/10"}},
            "anomaly_detection": {
                "dft_measurements": dft_objects([float(v) for v in meas], f"MT-{i:02d}"),
                "zone_id": f"MT-{i:02d}",
                "spec": {"nominal_um": nominal, "tolerance_rule": "90/10"},
                "image_context": "중앙부 스트라이프형 편차 패턴",
            },
            "location_tracking": {
                "coordinates": [{"x": 360 + i * 5, "y": 100 + i * 2, "z": 250 + i, "timestamp": 1712300000 + i * 30}],
                "ship_model_id": "HULL-COT-MULTI",
            },
            "sequence_validation": {
                "zone_map": make_zone_map(f"BLK-M{i%4}", painted_idx=i % 3),
                "current_zone_id": f"BLK-M{i%4}-A2",
                "structure_type": "block",
            },
            "pose_estimation": {
                "mode": "compare",
                "current_pose": {
                    "joints": {
                        "wrist_r": {"x": 0.11 + i * 0.001, "y": -0.03, "z": 0.39, "roll": 4, "pitch": 15, "yaw": 5}
                    }
                },
                "block_id": f"BLK-M{i%4}",
            },
            "timeseries": {
                "action": "update",
                "session_id": f"sess-multi-{i:02d}",
                "data": {"timestamp": 1712300500 + i * 60, "speed": 1.0 + i * 0.02, "distance": 310 + i * 3, "angle": 45 + (i % 6)},
            },
            "vlm_analysis": {
                "heatmap_path": f"/data/sim/heatmaps/multi_{i:02d}.png",
                "screenshot_path": f"/data/sim/screenshots/multi_{i:02d}.jpg",
                "report_json": {"zone": f"MT-{i:02d}", "shift": "day"},
            },
        }

        user_payload = {"scenario_id": f"multi-{i:02d}", "chain": chain, "inputs": {k: tool_args[k] for k in chain}}
        user = (
            f"[환경 데이터]\n{json.dumps(user_payload, ensure_ascii=False)}\n\n"
            "질문: 여러 신호를 연결해서 원인과 조치 우선순위를 단계별로 알려줘."
        )

        parts: list[str] = []
        step = 1
        parts.append("<think>")
        parts.append(f"단계 {step}: 입력 신호를 보면 단일 판정보다 연쇄 분석이 필요하다.")
        step += 1
        parts.append(f"단계 {step}: 첫 도구로 기초 상태를 확정한 뒤 다음 도구로 원인을 좁히는 전략을 사용한다.")
        parts.append("</think>")

        for idx, tool_name in enumerate(chain):
            parts.append(to_tool_call(tool_name, tool_args[tool_name], schemas))
            if idx < len(chain) - 1:
                step += 1
                parts.append("<think>")
                parts.append(f"단계 {step}: {tool_name} 결과를 다음 도구 입력 해석에 반영해 불확실성을 줄인다.")
                parts.append("</think>")

        step += 1
        parts.append("<think>")
        parts.append(f"단계 {step}: 최종적으로 합부, 결함 위험, 작업 순서 중 우선 조치 1개를 확정한다.")
        parts.append("</think>")
        parts.append(
            "종합 결론: 현재 시나리오는 다중 신호 상관관계가 확인되므로, "
            "재측정 구역 고정 -> 공정 교정 -> 보수 도장 순서로 실행하는 것이 적절합니다."
        )
        parts.append(f"\n근거 문서 ID: DOC-COT-MULTI-{i+1:04d}")

        assistant = "\n".join(parts)
        samples.append(make_sample(user, assistant, "multi_tool_cot"))

    return samples


def build_complex_tool_cot_samples(rng: random.Random, schemas: dict[str, dict[str, Any]], count: int = 20) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []

    chains = [
        ["location_tracking", "sequence_validation", "pose_estimation", "dft_judge", "anomaly_detection"],
        ["timeseries", "location_tracking", "dft_judge", "anomaly_detection", "vlm_analysis"],
        ["location_tracking", "timeseries", "sequence_validation", "pose_estimation", "dft_judge", "anomaly_detection"],
        ["vlm_analysis", "location_tracking", "sequence_validation", "dft_judge", "anomaly_detection", "timeseries"],
    ]

    for i in range(count):
        chain = chains[i % len(chains)]
        nominal = 100 + (i % 4) * 20
        meas = [float(nominal - 18 + j * 6 + (i % 3)) for j in range(6)]
        zone_id = f"AQ024-{i+1:02d}"
        block_id = f"BLK-AQ{i%5}"

        tool_args: dict[str, dict[str, Any]] = {
            "location_tracking": {
                "coordinates": [
                    {"x": 300 + i * 7, "y": 95 + i * 3, "z": 240 + i * 2, "timestamp": 1712400000 + i * 45},
                    {"x": 308 + i * 7, "y": 101 + i * 3, "z": 244 + i * 2, "timestamp": 1712400015 + i * 45},
                ],
                "ship_model_id": "HULL-AQ024",
            },
            "sequence_validation": {
                "zone_map": make_zone_map(block_id, painted_idx=i % 3),
                "current_zone_id": f"{block_id}-A2",
                "structure_type": "hull_section",
            },
            "pose_estimation": {
                "mode": "compare",
                "current_pose": {
                    "joints": {
                        "wrist_r": {"x": 0.12 + i * 0.001, "y": -0.04, "z": 0.38, "roll": 6, "pitch": 18, "yaw": 7},
                        "elbow_r": {"x": 0.21, "y": 0.10, "z": 0.35, "roll": 15, "pitch": 26, "yaw": 9},
                    }
                },
                "block_id": block_id,
            },
            "dft_judge": {
                "measurements": meas,
                "spec": {"nominal": nominal, "tolerance_rule": "90/10"},
            },
            "anomaly_detection": {
                "dft_measurements": dft_objects(meas, zone_id),
                "zone_id": zone_id,
                "spec": {"nominal_um": nominal, "tolerance_rule": "90/10"},
                "timeseries": {
                    "timestamps": [1712400000 + i * 45 + t for t in (0, 5, 10, 15, 20, 25)],
                    "standoff_distance_mm": [330, 328, 322, 340, 346, 338],
                    "gun_speed_m_per_s": [1.02, 1.06, 1.15, 1.18, 1.10, 1.04],
                    "spray_angle_deg": [46, 47, 49, 53, 51, 48],
                    "trajectory": [
                        {"x": 10 + j * 5, "y": 15 + j * 3, "z": 20 + j * 2}
                        for j in range(6)
                    ],
                },
                "image_context": "용접선 주변 띠 형태 편차 및 하부 저도막 혼재",
            },
            "timeseries": {
                "action": "update" if i % 2 == 0 else "query",
                "session_id": f"sess-aq024-{i:02d}",
                "data": {"timestamp": 1712400600 + i * 60, "speed": 1.08, "distance": 320 + i, "angle": 48},
            },
            "vlm_analysis": {
                "heatmap_path": f"/data/sim/heatmaps/aq024_{i:02d}.png",
                "screenshot_path": f"/data/sim/screenshots/aq024_{i:02d}.jpg",
                "report_json": {"zone_id": zone_id, "block_id": block_id, "shift": "night" if i % 2 else "day"},
            },
        }

        if tool_args["timeseries"]["action"] == "query":
            tool_args["timeseries"].pop("data", None)

        user_payload = {
            "scenario_id": f"complex-{i:02d}",
            "target": "AQ024형 복합 코칭",
            "chain": chain,
            "inputs": {k: tool_args[k] for k in chain},
        }
        user = (
            f"[환경 데이터]\n{json.dumps(user_payload, ensure_ascii=False)}\n\n"
            "질문: AQ024 유형처럼 복합 원인 시나리오를 단계별로 진단하고, 즉시 실행할 교정 계획을 제시해줘."
        )

        parts: list[str] = []
        step = 1
        parts.append("<think>")
        parts.append(f"단계 {step}: 이 요청은 위치-순서-자세-두께-이상신호를 결합한 복합 진단 문제다.")
        step += 1
        parts.append(f"단계 {step}: 단일 도구로는 원인 분리가 어려우므로 연쇄 도구 호출(4개 이상)이 필요하다.")
        parts.append("</think>")

        for idx, tool_name in enumerate(chain):
            parts.append(to_tool_call(tool_name, tool_args[tool_name], schemas))
            if idx < len(chain) - 1:
                step += 1
                parts.append("<think>")
                parts.append(
                    f"단계 {step}: {tool_name} 출력으로 기준편차를 갱신하고 다음 도구 입력 해석 우선순위를 재정렬한다."
                )
                parts.append("</think>")

        step += 1
        parts.append("<think>")
        parts.append(
            f"단계 {step}: 최종적으로 공정 순서 교정, 자세 교정, 재도장 범위, 재검증 루프를 하나의 실행 계획으로 통합한다."
        )
        parts.append("</think>")
        parts.append(
            "최종 코칭 계획: 1) 위치 기준으로 결함 집중 구역을 고정, 2) 순서/자세를 숙련자 패턴으로 교정, "
            "3) 기준 미달 구역 재도장, 4) 동일 조건에서 재측정 후 승인 판단을 수행합니다."
        )
        parts.append(f"\n근거 문서 ID: DOC-COT-COMPLEX-{i+1:04d}")

        assistant = "\n".join(parts)
        samples.append(make_sample(user, assistant, "complex_tool_cot"))

    return samples


def verify_dataset(samples: list[dict[str, Any]]) -> None:
    if len(samples) < 80:
        raise ValueError(f"Total sample count must be >= 80, got {len(samples)}")

    cat_counter = Counter(s.get("category_type", "unknown") for s in samples)
    expected = {
        "docqa_cot": 30,
        "single_tool_cot": 15,
        "multi_tool_cot": 15,
        "complex_tool_cot": 20,
    }
    for cat, exp_count in expected.items():
        actual = cat_counter.get(cat, 0)
        if actual < exp_count:
            raise ValueError(f"Category {cat}: got {actual}, expected >= {exp_count}")

    for idx, sample in enumerate(samples, start=1):
        assistant = sample["messages"][-1]["content"]
        if "<think>" not in assistant:
            raise ValueError(f"Sample {idx} has no <think> block")
        if not re.search(r"단계\s*1\s*:", assistant):
            raise ValueError(f"Sample {idx} has no numbered 단계 format")

        tool_calls = assistant.count("<tool_call>")
        category = sample.get("category_type")
        if category == "docqa_cot" and tool_calls != 0:
            raise ValueError(f"Sample {idx} docqa_cot must not include tool calls")
        if category == "single_tool_cot" and tool_calls != 1:
            raise ValueError(f"Sample {idx} single_tool_cot must include exactly 1 tool call")
        if category == "multi_tool_cot" and tool_calls not in (2, 3):
            raise ValueError(f"Sample {idx} multi_tool_cot must include 2-3 tool calls")
        if category == "complex_tool_cot" and tool_calls < 4:
            raise ValueError(f"Sample {idx} complex_tool_cot must include 4+ tool calls")


def write_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    rng = random.Random(args.seed)

    chunks = load_jsonl(args.chunks)
    logger.info("Loaded %d chunks from %s", len(chunks), args.chunks)

    schemas = load_tool_schemas(args.tool_schemas_dir)
    logger.info("Loaded %d tool schemas from %s", len(schemas), args.tool_schemas_dir)

    docqa_samples = build_docqa_cot_samples(chunks, rng, count=30)
    single_samples = build_single_tool_cot_samples(rng, schemas, count=15)
    multi_samples = build_multi_tool_cot_samples(rng, schemas, count=15)
    complex_samples = build_complex_tool_cot_samples(rng, schemas, count=20)

    all_samples = docqa_samples + single_samples + multi_samples + complex_samples
    rng.shuffle(all_samples)

    verify_dataset(all_samples)
    write_jsonl(args.output, all_samples)

    cat_counter = Counter(s["category_type"] for s in all_samples)
    avg_resp_len = mean(len(s["messages"][-1]["content"]) for s in all_samples)

    print(f"Total count: {len(all_samples)}")
    print(f"Per-category count: {dict(cat_counter)}")
    print(f"Average response length: {avg_resp_len:.2f}")
    print(f"Output: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
