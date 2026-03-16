#!/usr/bin/env python3
"""Generate tool-selection SFT samples for ship painting copilot.

Creates 40 JSONL samples with fixed distribution:
- 15 single intent -> single tool
- 15 complex intent -> multi tool
- 10 no-tool negative examples
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import TypedDict

logger = logging.getLogger(__name__)


class Message(TypedDict):
    role: str
    content: str


class SampleRow(TypedDict):
    messages: list[Message]
    category_type: str


class SingleCase(TypedDict):
    user: str
    name: str
    args: dict[str, object]
    post: str


class MultiCase(TypedDict):
    user: str
    calls: list[tuple[str, dict[str, object]]]


SYSTEM_PROMPT = (
    "너는 선박 도장 분야 전문 AI 코파일럿이다. "
    "질문 의도에 따라 도구 호출이 필요한지 먼저 판단하고, "
    "수치 판정/이상 탐지/순서 검증/자세 분석/위치 식별/시계열 분석/시각 분석 요청이면 "
    "반드시 적절한 도구를 호출한다. "
    "문서 기반 정의/용어/규정 질문은 도구 없이 답변한다."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate tool-selection SFT JSONL")
    _ = parser.add_argument(
        "--output",
        type=Path,
        default=Path("/data/sft/sft_tool_selection.jsonl"),
        help="Output JSONL path",
    )
    _ = parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    return parser.parse_args()


def _tool_call(name: str, arguments: dict[str, object]) -> str:
    payload = {"name": name, "arguments": arguments}
    return f"<tool_call>{json.dumps(payload, ensure_ascii=False)}</tool_call>"


def _sample(user_text: str, assistant_text: str, category_type: str) -> SampleRow:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ],
        "category_type": category_type,
    }


def build_single_tool_samples() -> list[SampleRow]:
    samples: list[SampleRow] = []

    singles: list[SingleCase] = [
        {
            "user": "DFT 측정값이 [95, 102, 88]이고 기준은 100um, 90/10 룰입니다. 합격인지 판정해줘.",
            "name": "dft_judge",
            "args": {"measurements": [95, 102, 88], "spec": {"nominal": 100, "tolerance_rule": "90/10"}},
            "post": "수치 판정 요청이므로 dft_judge를 호출해 판정을 진행합니다.",
        },
        {
            "user": "B-3 구역에서 저도막 패턴이 보이고 DFT 포인트는 85, 90, 78입니다. 이상 여부 분석해줘.",
            "name": "anomaly_detection",
            "args": {
                "dft_measurements": [
                    {"value_um": 85, "zone_id": "B-3", "position": "lower"},
                    {"value_um": 90, "zone_id": "B-3", "position": "middle"},
                    {"value_um": 78, "zone_id": "B-3", "position": "lower"},
                ],
                "zone_id": "B-3",
                "spec": {"nominal_um": 100, "tolerance_rule": "90/10"},
                "image_context": "좌하단 저도막 패턴",
            },
            "post": "도막 분포/패턴 분석 요청이므로 anomaly_detection을 호출합니다.",
        },
        {
            "user": "지금 A-1을 칠했어. 다음에 어느 구역을 칠해야 하는지 순서 검증해줘.",
            "name": "sequence_validation",
            "args": {
                "zone_map": [
                    {"zone_id": "A-1", "block_id": "BLK-A", "zone_type": "수직 패널", "position": "upper", "is_painted": True},
                    {"zone_id": "A-2", "block_id": "BLK-A", "zone_type": "수직 패널", "position": "middle", "is_painted": False},
                    {"zone_id": "A-3", "block_id": "BLK-A", "zone_type": "수직 패널", "position": "lower", "is_painted": False},
                ],
                "current_zone_id": "A-1",
                "structure_type": "block",
            },
            "post": "도장 순서 판단 요청이므로 sequence_validation을 호출합니다.",
        },
        {
            "user": "현재 자세가 숙련자 EP-003과 얼마나 유사한지 비교해줘.",
            "name": "pose_estimation",
            "args": {
                "mode": "compare",
                "current_pose": {
                    "joints": {
                        "wrist_r": {"x": 0.51, "y": 0.28, "z": 0.79, "roll": 7, "pitch": 21, "yaw": 10}
                    }
                },
                "expert_pose_id": "EP-003",
                "block_id": "BLK-B",
            },
            "post": "자세 비교 요청이므로 pose_estimation을 호출합니다.",
        },
        {
            "user": "현재 좌표 x=1200, y=3400, z=800에서 작업 중인데 어느 구역인지 확인해줘.",
            "name": "location_tracking",
            "args": {"coordinates": [{"x": 1200, "y": 3400, "z": 800}], "ship_model_id": "hull-001"},
            "post": "좌표 기반 구역 식별 요청이므로 location_tracking을 호출합니다.",
        },
        {
            "user": "sess-021 세션의 속도/거리/각도 추이를 조회해줘.",
            "name": "timeseries",
            "args": {"action": "query", "session_id": "sess-021"},
            "post": "시계열 조회 요청이므로 timeseries를 호출합니다.",
        },
        {
            "user": "히트맵 이미지 /tmp/heatmap_b3.png 를 시각 분석해서 문제 구역을 찾아줘.",
            "name": "vlm_analysis",
            "args": {"heatmap_path": "/tmp/heatmap_b3.png"},
            "post": "시각 분석 요청이므로 vlm_analysis를 호출합니다.",
        },
        {
            "user": "DFT [112, 108, 115, 110, 106], nominal 100um, 90/10으로 합부 확인해줘.",
            "name": "dft_judge",
            "args": {"measurements": [112, 108, 115, 110, 106], "spec": {"nominal": 100, "tolerance_rule": "90/10"}},
            "post": "합부 판정 요청이므로 dft_judge를 호출합니다.",
        },
        {
            "user": "C-2 구역에서 상단 과도막 패턴이 의심돼. DFT는 128, 132, 140이야. 이상 탐지해줘.",
            "name": "anomaly_detection",
            "args": {
                "dft_measurements": [
                    {"value_um": 128, "zone_id": "C-2", "position": "upper"},
                    {"value_um": 132, "zone_id": "C-2", "position": "middle"},
                    {"value_um": 140, "zone_id": "C-2", "position": "upper"},
                ],
                "zone_id": "C-2",
                "image_context": "상단 과도막 패턴",
            },
            "post": "이상 패턴 확인 요청이므로 anomaly_detection을 호출합니다.",
        },
        {
            "user": "블록 BLK-C 시작점 추천이 필요해. 어느 구역부터 칠하는 게 좋아?",
            "name": "sequence_validation",
            "args": {
                "zone_map": [
                    {"zone_id": "C-1", "block_id": "BLK-C", "zone_type": "곡면", "position": "upper", "is_painted": False},
                    {"zone_id": "C-2", "block_id": "BLK-C", "zone_type": "곡면", "position": "middle", "is_painted": False},
                    {"zone_id": "C-3", "block_id": "BLK-C", "zone_type": "곡면", "position": "lower", "is_painted": False},
                ],
                "structure_type": "block",
            },
            "post": "시작 구역 추천은 순서 문제이므로 sequence_validation을 호출합니다.",
        },
        {
            "user": "BLK-D에서 권장 숙련자 자세 번호를 추천해줘.",
            "name": "pose_estimation",
            "args": {"mode": "recommend", "block_id": "BLK-D"},
            "post": "자세 번호 추천 요청이므로 pose_estimation을 호출합니다.",
        },
        {
            "user": "좌표 [{x:980,y:2100,z:640},{x:1010,y:2140,z:650}] 기준으로 현재 구역을 추적해줘.",
            "name": "location_tracking",
            "args": {
                "coordinates": [
                    {"x": 980, "y": 2100, "z": 640, "timestamp": 1713000001},
                    {"x": 1010, "y": 2140, "z": 650, "timestamp": 1713000003},
                ],
                "ship_model_id": "hull-009",
            },
            "post": "위치 추적 요청이므로 location_tracking을 호출합니다.",
        },
        {
            "user": "sess-033에 timestamp 1713000123, speed 0.82, distance 345, angle 57 데이터를 업데이트해줘.",
            "name": "timeseries",
            "args": {
                "action": "update",
                "session_id": "sess-033",
                "data": {"timestamp": 1713000123, "speed": 0.82, "distance": 345, "angle": 57},
            },
            "post": "시계열 기록 요청이므로 timeseries를 호출합니다.",
        },
        {
            "user": "열화상 히트맵 /tmp/zone_a2_heat.png를 분석해서 관심영역을 뽑아줘. 스크린샷은 /tmp/shot_a2.png야.",
            "name": "vlm_analysis",
            "args": {"heatmap_path": "/tmp/zone_a2_heat.png", "screenshot_path": "/tmp/shot_a2.png"},
            "post": "이미지 기반 이슈 탐색 요청이므로 vlm_analysis를 호출합니다.",
        },
        {
            "user": "DFT가 [87, 89, 91, 84]인데 기준 100um 대비 통과인지 확인해줘.",
            "name": "dft_judge",
            "args": {"measurements": [87, 89, 91, 84], "spec": {"nominal": 100, "tolerance_rule": "90/10"}},
            "post": "판정형 수치 질문이므로 dft_judge를 호출합니다.",
        },
    ]

    for item in singles:
        assistant = f"{item['post']}\n\n{_tool_call(item['name'], item['args'])}\n\n도구 결과를 바탕으로 최종 판정을 제공하겠습니다."
        samples.append(_sample(item["user"], assistant, "single_tool_selection"))

    return samples


def build_multi_tool_samples() -> list[SampleRow]:
    samples: list[SampleRow] = []
    multi_cases: list[MultiCase] = [
        {
            "user": "B-3 구역 DFT가 [85, 90, 78]이고 히트맵에 좌하단 저도막 패턴이 있어. 합부와 원인을 같이 분석해줘.",
            "calls": [
                ("dft_judge", {"measurements": [85, 90, 78], "spec": {"nominal": 100, "tolerance_rule": "90/10"}}),
                (
                    "anomaly_detection",
                    {
                        "dft_measurements": [
                            {"value_um": 85, "zone_id": "B-3", "position": "lower"},
                            {"value_um": 90, "zone_id": "B-3", "position": "middle"},
                            {"value_um": 78, "zone_id": "B-3", "position": "lower"},
                        ],
                        "zone_id": "B-3",
                        "spec": {"nominal_um": 100, "tolerance_rule": "90/10"},
                        "image_context": "좌하단 저도막 패턴",
                    },
                ),
            ],
        },
        {
            "user": "현재 좌표가 (1210,3390,805)이고 다음 도장 구역을 추천해줘.",
            "calls": [
                ("location_tracking", {"coordinates": [{"x": 1210, "y": 3390, "z": 805}], "ship_model_id": "hull-001"}),
                (
                    "sequence_validation",
                    {
                        "zone_map": [
                            {"zone_id": "B-3", "block_id": "BLK-B", "zone_type": "수직 패널", "position": "middle", "is_painted": True},
                            {"zone_id": "B-4", "block_id": "BLK-B", "zone_type": "수직 패널", "position": "lower", "is_painted": False},
                            {"zone_id": "B-2", "block_id": "BLK-B", "zone_type": "수직 패널", "position": "upper", "is_painted": False},
                        ],
                        "current_zone_id": "B-3",
                        "structure_type": "block",
                    },
                ),
            ],
        },
        {
            "user": "세션 sess-041의 추이를 보고 현재 자세 교정이 필요한지 판단해줘.",
            "calls": [
                ("timeseries", {"action": "query", "session_id": "sess-041"}),
                (
                    "pose_estimation",
                    {
                        "mode": "compare",
                        "current_pose": {"joints": {"elbow_r": {"x": 0.21, "y": 0.11, "z": 0.36, "roll": 14, "pitch": 28, "yaw": 9}}},
                        "block_id": "BLK-D",
                    },
                ),
            ],
        },
        {
            "user": "히트맵 /tmp/hm_c2.png 분석하고 C-2 DFT [132,128,125]가 기준 120um에서 합격인지 같이 봐줘.",
            "calls": [
                ("vlm_analysis", {"heatmap_path": "/tmp/hm_c2.png"}),
                ("dft_judge", {"measurements": [132, 128, 125], "spec": {"nominal": 120, "tolerance_rule": "90/10"}}),
                (
                    "anomaly_detection",
                    {
                        "dft_measurements": [
                            {"value_um": 132, "zone_id": "C-2", "position": "upper"},
                            {"value_um": 128, "zone_id": "C-2", "position": "middle"},
                            {"value_um": 125, "zone_id": "C-2", "position": "lower"},
                        ],
                        "zone_id": "C-2",
                    },
                ),
            ],
        },
        {
            "user": "좌표 궤적 기반으로 현재 구역 확인 후, 해당 구역 기준 숙련자 자세 번호도 추천해줘.",
            "calls": [
                (
                    "location_tracking",
                    {
                        "coordinates": [
                            {"x": 900, "y": 1700, "z": 600, "timestamp": 1713100011},
                            {"x": 940, "y": 1750, "z": 605, "timestamp": 1713100014},
                        ],
                        "ship_model_id": "hull-007",
                    },
                ),
                ("pose_estimation", {"mode": "recommend", "block_id": "BLK-G"}),
            ],
        },
        {
            "user": "DFT [98,101,95,92]와 현재 속도/거리/각도 추이까지 반영해서 품질 안정성을 분석해줘.",
            "calls": [
                ("dft_judge", {"measurements": [98, 101, 95, 92], "spec": {"nominal": 100, "tolerance_rule": "90/10"}}),
                ("timeseries", {"action": "query", "session_id": "sess-052"}),
                (
                    "anomaly_detection",
                    {
                        "dft_measurements": [
                            {"value_um": 98, "zone_id": "A-4", "position": "center"},
                            {"value_um": 101, "zone_id": "A-4", "position": "upper"},
                            {"value_um": 95, "zone_id": "A-4", "position": "lower"},
                            {"value_um": 92, "zone_id": "A-4", "position": "left"},
                        ],
                        "zone_id": "A-4",
                    },
                ),
            ],
        },
        {
            "user": "A-2 작업 후 다음 구역 추천과 함께 순서가 맞는지 검증해줘. 현재 좌표는 (540,78,315).",
            "calls": [
                ("location_tracking", {"coordinates": [{"x": 540, "y": 78, "z": 315}], "ship_model_id": "hull-101"}),
                (
                    "sequence_validation",
                    {
                        "zone_map": [
                            {"zone_id": "A-1", "block_id": "BLK-A", "zone_type": "수평 패널", "position": "left", "is_painted": True},
                            {"zone_id": "A-2", "block_id": "BLK-A", "zone_type": "수평 패널", "position": "center", "is_painted": True},
                            {"zone_id": "A-3", "block_id": "BLK-A", "zone_type": "수평 패널", "position": "right", "is_painted": False},
                        ],
                        "current_zone_id": "A-2",
                        "structure_type": "deck",
                    },
                ),
            ],
        },
        {
            "user": "히트맵 /tmp/hm_b1.png와 스크린샷 /tmp/shot_b1.png를 보고 이상 구역 찾고 DFT [88,91,93] 합부도 같이 판정해줘.",
            "calls": [
                ("vlm_analysis", {"heatmap_path": "/tmp/hm_b1.png", "screenshot_path": "/tmp/shot_b1.png"}),
                ("dft_judge", {"measurements": [88, 91, 93], "spec": {"nominal": 100, "tolerance_rule": "90/10"}}),
            ],
        },
        {
            "user": "sess-060 데이터를 먼저 업데이트하고 바로 추세 조회해서 작업 안정성 피드백 줘.",
            "calls": [
                (
                    "timeseries",
                    {
                        "action": "update",
                        "session_id": "sess-060",
                        "data": {"timestamp": 1713200001, "speed": 0.91, "distance": 322, "angle": 49},
                    },
                ),
                ("timeseries", {"action": "query", "session_id": "sess-060"}),
            ],
        },
        {
            "user": "현재 자세 비교와 동시에 현재 위치 구역을 확인해서 교정 우선순위를 정해줘.",
            "calls": [
                ("location_tracking", {"coordinates": [{"x": 430, "y": 110, "z": 250}], "ship_model_id": "hull-103"}),
                (
                    "pose_estimation",
                    {
                        "mode": "compare",
                        "current_pose": {"joints": {"shoulder_r": {"x": 0.33, "y": 0.23, "z": 0.53, "roll": 17, "pitch": 30, "yaw": 12}}},
                        "expert_pose_id": "EP-010",
                        "block_id": "BLK-E",
                    },
                ),
            ],
        },
        {
            "user": "저도막 의심(B-5)인데 DFT [82,79,88]야. 이상 탐지하고 세션 추이까지 같이 봐줘.",
            "calls": [
                (
                    "anomaly_detection",
                    {
                        "dft_measurements": [
                            {"value_um": 82, "zone_id": "B-5", "position": "lower"},
                            {"value_um": 79, "zone_id": "B-5", "position": "lower"},
                            {"value_um": 88, "zone_id": "B-5", "position": "middle"},
                        ],
                        "zone_id": "B-5",
                        "image_context": "하단 저도막 의심",
                    },
                ),
                ("timeseries", {"action": "query", "session_id": "sess-063"}),
                ("dft_judge", {"measurements": [82, 79, 88], "spec": {"nominal": 100, "tolerance_rule": "90/10"}}),
            ],
        },
        {
            "user": "블록 BLK-H에서 다음 도장 순서를 추천하고, 추천된 구역의 권장 자세 번호도 알려줘.",
            "calls": [
                (
                    "sequence_validation",
                    {
                        "zone_map": [
                            {"zone_id": "H-1", "block_id": "BLK-H", "zone_type": "모서리", "position": "upper", "is_painted": True},
                            {"zone_id": "H-2", "block_id": "BLK-H", "zone_type": "모서리", "position": "middle", "is_painted": False},
                            {"zone_id": "H-3", "block_id": "BLK-H", "zone_type": "모서리", "position": "lower", "is_painted": False},
                        ],
                        "current_zone_id": "H-1",
                        "structure_type": "superstructure",
                    },
                ),
                ("pose_estimation", {"mode": "recommend", "block_id": "BLK-H"}),
            ],
        },
        {
            "user": "좌표 기반 구역 식별 후 그 구역의 DFT [104,99,96]를 판정해줘.",
            "calls": [
                ("location_tracking", {"coordinates": [{"x": 1180, "y": 3310, "z": 790}], "ship_model_id": "hull-001"}),
                ("dft_judge", {"measurements": [104, 99, 96], "spec": {"nominal": 100, "tolerance_rule": "90/10"}}),
            ],
        },
        {
            "user": "히트맵 /tmp/hm_d3.png 시각 분석 결과와 anomaly 탐지를 함께 수행해서 결함 원인을 알려줘.",
            "calls": [
                (
                    "vlm_analysis",
                    {
                        "heatmap_path": "/tmp/hm_d3.png",
                        "report_json": {"zone_id": "D-3", "note": "저도막 후보 영역 2개"},
                    },
                ),
                (
                    "anomaly_detection",
                    {
                        "dft_measurements": [
                            {"value_um": 86, "zone_id": "D-3", "position": "left"},
                            {"value_um": 89, "zone_id": "D-3", "position": "center"},
                            {"value_um": 84, "zone_id": "D-3", "position": "right"},
                        ],
                        "zone_id": "D-3",
                    },
                ),
            ],
        },
        {
            "user": "sess-077 추세를 보고 현재 DFT [108,111,105]와 함께 합격 여부와 위험 신호를 종합 판단해줘.",
            "calls": [
                ("timeseries", {"action": "query", "session_id": "sess-077"}),
                ("dft_judge", {"measurements": [108, 111, 105], "spec": {"nominal": 100, "tolerance_rule": "90/10"}}),
                (
                    "anomaly_detection",
                    {
                        "dft_measurements": [
                            {"value_um": 108, "zone_id": "E-1", "position": "upper"},
                            {"value_um": 111, "zone_id": "E-1", "position": "middle"},
                            {"value_um": 105, "zone_id": "E-1", "position": "lower"},
                        ],
                        "zone_id": "E-1",
                    },
                ),
            ],
        },
    ]

    for case in multi_cases:
        preface = "복합 판단 요청이므로 관련 도구를 순차 호출합니다."
        calls_text = "\n".join(_tool_call(name, args) for name, args in case["calls"])
        assistant = f"{preface}\n\n{calls_text}\n\n각 도구 결과를 결합해 최종 분석을 제시하겠습니다."
        samples.append(_sample(case["user"], assistant, "multi_tool_selection"))

    return samples


def build_no_tool_samples() -> list[SampleRow]:
    knowledge_pairs = [
        ("에폭시 프라이머의 주요 성분은 무엇이야?", "에폭시 프라이머는 일반적으로 에폭시 수지, 경화제, 안료, 용제를 포함합니다."),
        ("SSPC-PA2 규정은 무엇을 위한 기준이야?", "SSPC-PA2는 건조 도막 두께(DFT) 측정과 판정 절차를 정의하는 품질 기준입니다."),
        ("블라스팅 등급 Sa 2.5는 어떤 상태를 의미해?", "Sa 2.5는 거의 백색 금속면 수준으로 녹, 스케일, 이물질이 대부분 제거된 상태를 의미합니다."),
        ("도장 작업에서 이슬점 3도 규칙이 왜 중요해?", "표면 온도가 이슬점보다 최소 3도 높아야 결로를 방지하고 도막 부착 불량을 줄일 수 있기 때문입니다."),
        ("WFT와 DFT 차이를 간단히 설명해줘.", "WFT는 습윤 도막 두께, DFT는 건조 후 최종 도막 두께로 측정 시점과 목적이 다릅니다."),
        ("선박 도장에서 오버코트 간격이 뭐야?", "오버코트 간격은 하도 도막 위에 다음 층을 도장할 수 있는 최소/최대 허용 시간 범위입니다."),
        ("아연리치 프라이머를 사용하는 목적은?", "아연리치 프라이머는 희생양극 효과를 통해 강재 부식을 억제하기 위해 사용됩니다."),
        ("도장 공정에서 환기 기준을 지켜야 하는 이유는?", "유기용제 증기 축적을 줄여 화재·폭발·건강 위험을 예방하기 위해서입니다."),
        ("도막 결함 중 핀홀의 일반적 원인은 뭐야?", "핀홀은 기포 잔류, 과도한 희석, 부적절한 건조 조건 등으로 발생할 수 있습니다."),
        ("프라이머와 상도의 역할 차이를 알려줘.", "프라이머는 부착성과 방청 기반을 제공하고, 상도는 내후성·내화학성·외관을 담당합니다."),
    ]

    samples: list[SampleRow] = []
    for user_text, answer_core in knowledge_pairs:
        assistant = (
            "이 질문은 문서 기반 지식 질문이므로 도구 호출이 필요하지 않습니다. "
            f"{answer_core}"
        )
        samples.append(_sample(user_text, assistant, "no_tool_selection"))
    return samples


def validate_samples(rows: list[SampleRow]) -> None:
    if len(rows) != 40:
        raise ValueError(f"총 샘플 수가 40이 아닙니다: {len(rows)}")

    counts = Counter(row["category_type"] for row in rows)
    expected = {
        "single_tool_selection": 15,
        "multi_tool_selection": 15,
        "no_tool_selection": 10,
    }
    if counts != expected:
        raise ValueError(f"카테고리 분포 오류: actual={dict(counts)}, expected={expected}")

    for idx, row in enumerate(rows):
        messages = row["messages"]
        if len(messages) != 3:
            raise ValueError(f"샘플 {idx} 메시지 수 오류")
        roles = [m.get("role") for m in messages]
        if roles != ["system", "user", "assistant"]:
            raise ValueError(f"샘플 {idx} role 순서 오류: {roles}")


def write_jsonl(path: Path, rows: list[SampleRow]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            _ = f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def main() -> int:
    args = parse_args()
    log_level_name = str(getattr(args, "log_level", "INFO"))
    output_path = Path(str(getattr(args, "output", "/data/sft/sft_tool_selection.jsonl")))
    log_level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
    }
    logging.basicConfig(
        level=log_level_map.get(log_level_name, logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    single = build_single_tool_samples()
    multi = build_multi_tool_samples()
    no_tool = build_no_tool_samples()
    all_rows = [*single, *multi, *no_tool]

    validate_samples(all_rows)
    count = write_jsonl(output_path, all_rows)

    per_category = Counter(row["category_type"] for row in all_rows)
    logger.info("생성 완료: %s", output_path)
    logger.info("총 샘플 수: %d", count)
    for key in ["single_tool_selection", "multi_tool_selection", "no_tool_selection"]:
        logger.info("- %s: %d", key, per_category[key])

    print(f"total={count}")
    print(f"single_tool_selection={per_category['single_tool_selection']}")
    print(f"multi_tool_selection={per_category['multi_tool_selection']}")
    print(f"no_tool_selection={per_category['no_tool_selection']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
