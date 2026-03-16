#!/usr/bin/env python3
"""Generate medium-difficulty multi-tool chain SFT samples.

Creates 30 JSONL samples with 2-3 tool sequential chains:
- 10 location_tracking -> sequence_validation
- 10 dft_judge -> anomaly_detection
- 10 pose_estimation -> timeseries
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "너는 선박 도장 분야 전문 AI 코파일럿이다. "
    "복잡한 질문에는 반드시 단계별로 사고하고, 각 단계의 근거를 명시하라. "
    "여러 도구의 결과가 필요한 경우, 이전 도구 결과를 다음 도구의 입력으로 활용하라."
)


def _tc(name: str, args: dict) -> str:

    return f'<tool_call>\n{json.dumps({"name": name, "arguments": args}, ensure_ascii=False)}\n</tool_call>'


# ---------------------------------------------------------------------------
# Chain A: location_tracking -> sequence_validation  (10 samples)
# ---------------------------------------------------------------------------
LOC_SEQ_CASES: list[dict[str, Any]] = [
    {
        "user": "현재 좌표 (1200, 3400, 800)에서 다음에 칠할 구역을 알려줘.",
        "coords": [{"x": 1200, "y": 3400, "z": 800}],
        "ship": "hull-001",
        "zone_id": "A-1", "block_id": "BLK-A",
        "zone_map": [
            {"zone_id": "A-1", "block_id": "BLK-A", "zone_type": "수직 패널", "position": "upper", "is_painted": True},
            {"zone_id": "A-2", "block_id": "BLK-A", "zone_type": "수직 패널", "position": "lower", "is_painted": False},
        ],
        "structure": "block",
    },
    {
        "user": "GPS 좌표 (2500, 1800, 600)인데, 이 위치의 블록이 어디인지 확인하고 다음 도장 순서를 알려줘.",
        "coords": [{"x": 2500, "y": 1800, "z": 600}],
        "ship": "hull-002",
        "zone_id": "B-3", "block_id": "BLK-B",
        "zone_map": [
            {"zone_id": "B-1", "block_id": "BLK-B", "zone_type": "바닥 패널", "position": "center", "is_painted": True},
            {"zone_id": "B-2", "block_id": "BLK-B", "zone_type": "바닥 패널", "position": "left", "is_painted": True},
            {"zone_id": "B-3", "block_id": "BLK-B", "zone_type": "바닥 패널", "position": "right", "is_painted": False},
        ],
        "structure": "block",
    },
    {
        "user": "현재 위치가 (900, 4200, 1100)입니다. 어느 구역인지 파악하고, 이 구역 이후 도장 순서를 확인해 주세요.",
        "coords": [{"x": 900, "y": 4200, "z": 1100}],
        "ship": "hull-001",
        "zone_id": "C-2", "block_id": "BLK-C",
        "zone_map": [
            {"zone_id": "C-1", "block_id": "BLK-C", "zone_type": "곡면 패널", "position": "bow", "is_painted": True},
            {"zone_id": "C-2", "block_id": "BLK-C", "zone_type": "곡면 패널", "position": "mid", "is_painted": False},
            {"zone_id": "C-3", "block_id": "BLK-C", "zone_type": "곡면 패널", "position": "stern", "is_painted": False},
        ],
        "structure": "block",
    },
    {
        "user": "좌표 (3100, 2700, 400)에 있는데 현재 구역과 남은 도장 순서를 알려줘.",
        "coords": [{"x": 3100, "y": 2700, "z": 400}],
        "ship": "hull-003",
        "zone_id": "D-1", "block_id": "BLK-D",
        "zone_map": [
            {"zone_id": "D-1", "block_id": "BLK-D", "zone_type": "수직 패널", "position": "upper", "is_painted": False},
            {"zone_id": "D-2", "block_id": "BLK-D", "zone_type": "수직 패널", "position": "lower", "is_painted": False},
        ],
        "structure": "block",
    },
    {
        "user": "작업자 위치 (1500, 3000, 900). 현재 구역 확인 후 다음 칠할 곳 안내해줘.",
        "coords": [{"x": 1500, "y": 3000, "z": 900}],
        "ship": "hull-001",
        "zone_id": "A-3", "block_id": "BLK-A",
        "zone_map": [
            {"zone_id": "A-3", "block_id": "BLK-A", "zone_type": "수직 패널", "position": "center", "is_painted": True},
            {"zone_id": "A-4", "block_id": "BLK-A", "zone_type": "수직 패널", "position": "right", "is_painted": False},
        ],
        "structure": "block",
    },
    {
        "user": "현재 (800, 5100, 300) 좌표에 있습니다. 어느 블록 구역인지 알려주고 도장 진행 순서를 확인해줘.",
        "coords": [{"x": 800, "y": 5100, "z": 300}],
        "ship": "hull-002",
        "zone_id": "E-2", "block_id": "BLK-E",
        "zone_map": [
            {"zone_id": "E-1", "block_id": "BLK-E", "zone_type": "바닥 패널", "position": "left", "is_painted": True},
            {"zone_id": "E-2", "block_id": "BLK-E", "zone_type": "바닥 패널", "position": "center", "is_painted": False},
            {"zone_id": "E-3", "block_id": "BLK-E", "zone_type": "바닥 패널", "position": "right", "is_painted": False},
        ],
        "structure": "block",
    },
    {
        "user": "위치 좌표 (2000, 1500, 1200)에서 현재 구역 식별하고 순서 검증 부탁해.",
        "coords": [{"x": 2000, "y": 1500, "z": 1200}],
        "ship": "hull-001",
        "zone_id": "F-1", "block_id": "BLK-F",
        "zone_map": [
            {"zone_id": "F-1", "block_id": "BLK-F", "zone_type": "곡면 패널", "position": "upper", "is_painted": False},
            {"zone_id": "F-2", "block_id": "BLK-F", "zone_type": "곡면 패널", "position": "lower", "is_painted": False},
        ],
        "structure": "block",
    },
    {
        "user": "좌표 (1700, 3800, 500)에서 작업 중입니다. 현재 위치 확인하고 다음 도장 구역 순서를 알려주세요.",
        "coords": [{"x": 1700, "y": 3800, "z": 500}],
        "ship": "hull-003",
        "zone_id": "G-3", "block_id": "BLK-G",
        "zone_map": [
            {"zone_id": "G-1", "block_id": "BLK-G", "zone_type": "수직 패널", "position": "left", "is_painted": True},
            {"zone_id": "G-2", "block_id": "BLK-G", "zone_type": "수직 패널", "position": "center", "is_painted": True},
            {"zone_id": "G-3", "block_id": "BLK-G", "zone_type": "수직 패널", "position": "right", "is_painted": False},
        ],
        "structure": "block",
    },
    {
        "user": "(2800, 2200, 700) 좌표 확인하고, 해당 구역 이후 도장 순서를 알려줘.",
        "coords": [{"x": 2800, "y": 2200, "z": 700}],
        "ship": "hull-002",
        "zone_id": "H-1", "block_id": "BLK-H",
        "zone_map": [
            {"zone_id": "H-1", "block_id": "BLK-H", "zone_type": "바닥 패널", "position": "bow", "is_painted": False},
            {"zone_id": "H-2", "block_id": "BLK-H", "zone_type": "바닥 패널", "position": "stern", "is_painted": False},
        ],
        "structure": "block",
    },
    {
        "user": "현재 센서 위치 (1100, 4600, 950)입니다. 구역 식별 후 다음 순서를 확인해 주세요.",
        "coords": [{"x": 1100, "y": 4600, "z": 950}],
        "ship": "hull-001",
        "zone_id": "B-2", "block_id": "BLK-B",
        "zone_map": [
            {"zone_id": "B-1", "block_id": "BLK-B", "zone_type": "곡면 패널", "position": "upper", "is_painted": True},
            {"zone_id": "B-2", "block_id": "BLK-B", "zone_type": "곡면 패널", "position": "mid", "is_painted": True},
            {"zone_id": "B-3", "block_id": "BLK-B", "zone_type": "곡면 패널", "position": "lower", "is_painted": False},
        ],
        "structure": "block",
    },
]


def _build_loc_seq_sample(case: dict) -> dict:
    think = (
        f"<think>\n"
        f"단계 1: 사용자가 좌표 {case['coords']}에서 현재 구역을 확인하고 다음 도장 순서를 알고 싶어 한다.\n"
        f"단계 2: 먼저 location_tracking 도구로 좌표를 구역 ID로 변환해야 한다.\n"
        f"단계 3: location_tracking 결과에서 얻은 zone_id를 sequence_validation의 current_zone_id로 전달하여 순서를 검증한다.\n"
        f"</think>\n\n"
    )
    tc1 = _tc("location_tracking", {"coordinates": case["coords"], "ship_model_id": case["ship"]})
    bridge = (
        f"\n\nlocation_tracking 결과, 현재 위치는 {case['zone_id']} 구역({case['block_id']})으로 식별되었습니다. "
        f"이 zone_id를 sequence_validation에 전달하여 도장 순서를 확인합니다.\n\n"
    )
    tc2 = _tc("sequence_validation", {
        "zone_map": case["zone_map"],
        "current_zone_id": case["zone_id"],
        "structure_type": case["structure"],
    })
    unpainted = [z["zone_id"] for z in case["zone_map"] if not z["is_painted"]]
    conclusion = (
        f"\n\n[분석] 현재 위치는 {case['zone_id']} 구역({case['block_id']})입니다. "
        f"sequence_validation 결과, 미도장 구역은 {', '.join(unpainted)}입니다.\n"
        f"[조치] {'다음 도장 대상은 ' + unpainted[0] + ' 구역입니다.' if unpainted else '모든 구역 도장이 완료되었습니다.'}\n"
        f"[근거] location_tracking + sequence_validation 연계 분석"
    )
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": case["user"]},
            {"role": "assistant", "content": think + tc1 + bridge + tc2 + conclusion},
        ],
        "category_type": "medium_chain",
        "chain_type": "loc_seq",
    }


# ---------------------------------------------------------------------------
# Chain B: dft_judge -> anomaly_detection  (10 samples)
# ---------------------------------------------------------------------------
DFT_ANOMALY_CASES: list[dict[str, Any]] = [
    {"user": "DFT 측정값 [78, 85, 92, 68, 105]으로 기준 100\u00b5m 판정하고, 불합격이면 원인 분석해줘.", "meas": [78, 85, 92, 68, 105], "nom": 100, "zone": "B-3"},
    {"user": "도막두께 [95, 88, 102, 91, 87] 측정했어. 기준 100\u00b5m으로 합부 판정하고 이상 여부 확인해줘.", "meas": [95, 88, 102, 91, 87], "nom": 100, "zone": "A-1"},
    {"user": "측정값 [120, 115, 130, 125, 140]인데 기준 120\u00b5m으로 판정 후 과도막 분석해줘.", "meas": [120, 115, 130, 125, 140], "nom": 120, "zone": "C-2"},
    {"user": "DFT [65, 70, 75, 60, 80], 기준 80\u00b5m. 합격인지 확인하고 저도막이면 원인 찾아줘.", "meas": [65, 70, 75, 60, 80], "nom": 80, "zone": "D-1"},
    {"user": "측정 결과 [98, 102, 95, 110, 88]이고 스펙 100\u00b5m입니다. 판정하고 분포 이상이 있는지 분석해주세요.", "meas": [98, 102, 95, 110, 88], "nom": 100, "zone": "E-2"},
    {"user": "DFT [55, 62, 58, 70, 48]로 기준 75\u00b5m 판정 및 원인 분석 부탁합니다.", "meas": [55, 62, 58, 70, 48], "nom": 75, "zone": "F-1"},
    {"user": "도막두께 측정값이 [105, 112, 108, 115, 120]입니다. 기준 100\u00b5m 대비 판정하고 과도막 패턴 확인해줘.", "meas": [105, 112, 108, 115, 120], "nom": 100, "zone": "G-3"},
    {"user": "DFT [82, 90, 78, 85, 88], 기준 90\u00b5m으로 합부 판정 후 이상 탐지 실행해줘.", "meas": [82, 90, 78, 85, 88], "nom": 90, "zone": "H-1"},
    {"user": "측정값 [150, 145, 160, 155, 140]이고 기준 130\u00b5m입니다. 판정하고 과도막 원인을 분석해주세요.", "meas": [150, 145, 160, 155, 140], "nom": 130, "zone": "A-2"},
    {"user": "DFT [72, 68, 80, 75, 65]으로 기준 85\u00b5m 판정해주고, 저도막이면 원인 분석해줘.", "meas": [72, 68, 80, 75, 65], "nom": 85, "zone": "B-1"},
]


def _build_dft_anomaly_sample(case: dict) -> dict:
    think = (
        f"<think>\n"
        f"단계 1: 사용자가 DFT 측정값 {case['meas']}에 대해 기준 {case['nom']}\u00b5m 판정을 요청했다.\n"
        f"단계 2: dft_judge 도구로 합부 판정을 먼저 수행한다.\n"
        f"단계 3: 판정 결과에 따라, anomaly_detection 도구로 측정값의 분포 이상 여부를 분석한다. "
        f"dft_judge의 measurements를 anomaly_detection의 dft_measurements로 전달한다.\n"
        f"</think>\n\n"
    )
    tc1 = _tc("dft_judge", {
        "measurements": case["meas"],
        "spec": {"nominal": case["nom"], "tolerance_rule": "90/10"},
    })
    bridge = (
        f"\n\ndft_judge 판정 완료. 동일 측정값을 anomaly_detection에 전달하여 분포 이상을 분석합니다.\n\n"
    )
    dft_meas = [{"value_um": v, "zone_id": case["zone"], "position": "center"} for v in case["meas"]]
    tc2 = _tc("anomaly_detection", {
        "dft_measurements": dft_meas,
        "zone_id": case["zone"],
        "spec": {"nominal_um": case["nom"], "tolerance_rule": "90/10"},
    })
    below = [v for v in case["meas"] if v < case["nom"] * 0.9]
    above = [v for v in case["meas"] if v > case["nom"] * 1.1]
    conclusion = (
        f"\n\n[분석] DFT 측정값 {case['meas']}을 기준 {case['nom']}\u00b5m(90/10 규칙)으로 판정했습니다. "
        f"{'저도막 포인트: ' + str(below) if below else ''}"
        f"{'과도막 포인트: ' + str(above) if above else ''}\n"
        f"[판정] {'불합격 — 기준 미달 측정점 존재' if below or above else '합격'}\n"
        f"[원인] {'스프레이 건 속도 불균일 또는 도료 점도 부적합 가능성' if below else '도료 과다 적용 또는 겹침 도장 가능성' if above else '정상 범위 내'}\n"
        f"[조치] {'해당 구역 재도장 필요' if below else '다음 구역 진행 가능'}\n"
        f"[근거] dft_judge + anomaly_detection 연계 분석, {case['zone']} 구역"
    )
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": case["user"]},
            {"role": "assistant", "content": think + tc1 + bridge + tc2 + conclusion},
        ],
        "category_type": "medium_chain",
        "chain_type": "dft_anomaly",
    }


# ---------------------------------------------------------------------------
# Chain C: pose_estimation -> timeseries  (10 samples)
# ---------------------------------------------------------------------------
POSE_TS_CASES: list[dict[str, Any]] = [
    {"user": "현재 자세를 숙련자와 비교하고 세션 전체 개선 추이를 보여줘.", "expert": "EP-001", "block": "BLK-A", "session": "sess-001",
     "joints": {"wrist_r": {"x": 0.5, "y": 0.3, "z": 0.8, "roll": 8, "pitch": 24, "yaw": 12}}},
    {"user": "작업 자세 비교해줘. 그리고 이번 세션 동안 자세 점수가 얼마나 좋아졌는지 알려줘.", "expert": "EP-002", "block": "BLK-B", "session": "sess-002",
     "joints": {"wrist_r": {"x": 0.4, "y": 0.5, "z": 0.7, "roll": 10, "pitch": 20, "yaw": 15}}},
    {"user": "현재 포즈를 숙련자 기준과 비교 분석하고, 세션 추이도 확인해줘.", "expert": "EP-001", "block": "BLK-C", "session": "sess-003",
     "joints": {"wrist_r": {"x": 0.6, "y": 0.2, "z": 0.9, "roll": 5, "pitch": 30, "yaw": 8}}},
    {"user": "자세 교정 피드백 부탁해. 숙련자 EP-003과 비교하고, 세션 개선 추이도 보여줘.", "expert": "EP-003", "block": "BLK-D", "session": "sess-004",
     "joints": {"wrist_r": {"x": 0.3, "y": 0.6, "z": 0.5, "roll": 15, "pitch": 18, "yaw": 20}}},
    {"user": "스프레이 자세를 분석해줘. 숙련자 기준 비교 후, 이번 훈련 세션의 자세 변화 추이를 확인하고 싶어.", "expert": "EP-001", "block": "BLK-A", "session": "sess-005",
     "joints": {"wrist_r": {"x": 0.55, "y": 0.35, "z": 0.75, "roll": 7, "pitch": 22, "yaw": 10}}},
    {"user": "현재 작업 자세 점검하고, 세션 전체에서 자세가 개선되고 있는지 추이를 보여줘.", "expert": "EP-002", "block": "BLK-E", "session": "sess-006",
     "joints": {"wrist_r": {"x": 0.45, "y": 0.4, "z": 0.65, "roll": 12, "pitch": 25, "yaw": 14}}},
    {"user": "자세 비교 분석 후 시계열 추이까지 확인 부탁합니다.", "expert": "EP-001", "block": "BLK-F", "session": "sess-007",
     "joints": {"wrist_r": {"x": 0.5, "y": 0.25, "z": 0.85, "roll": 6, "pitch": 28, "yaw": 9}}},
    {"user": "작업 자세를 EP-003 기준으로 비교해줘. 그리고 sess-008 동안 개선 추이 확인해줘.", "expert": "EP-003", "block": "BLK-G", "session": "sess-008",
     "joints": {"wrist_r": {"x": 0.35, "y": 0.55, "z": 0.6, "roll": 14, "pitch": 19, "yaw": 18}}},
    {"user": "현재 포즈 점수 확인하고 이번 세션 전체 추이를 보고 싶어.", "expert": "EP-002", "block": "BLK-H", "session": "sess-009",
     "joints": {"wrist_r": {"x": 0.48, "y": 0.42, "z": 0.72, "roll": 9, "pitch": 23, "yaw": 11}}},
    {"user": "숙련자 EP-001 기준으로 자세 비교하고, 세션 추이 분석도 함께 해줘.", "expert": "EP-001", "block": "BLK-A", "session": "sess-010",
     "joints": {"wrist_r": {"x": 0.52, "y": 0.28, "z": 0.82, "roll": 7, "pitch": 26, "yaw": 10}}},
]


def _build_pose_ts_sample(case: dict) -> dict:
    think = (
        f"<think>\n"
        f"단계 1: 사용자가 현재 자세를 숙련자({case['expert']})와 비교하고 세션 추이를 확인하고 싶어 한다.\n"
        f"단계 2: pose_estimation 도구로 현재 자세와 숙련자 자세를 비교한다.\n"
        f"단계 3: pose_estimation 결과 확인 후, timeseries 도구로 세션({case['session']}) 전체 추이를 조회한다.\n"
        f"</think>\n\n"
    )
    tc1 = _tc("pose_estimation", {
        "mode": "compare",
        "current_pose": {"joints": case["joints"]},
        "expert_pose_id": case["expert"],
        "block_id": case["block"],
    })
    bridge = (
        f"\n\npose_estimation 비교 완료. 세션 {case['session']}의 자세 점수 추이를 timeseries로 조회합니다.\n\n"
    )
    tc2 = _tc("timeseries", {
        "action": "query",
        "session_id": case["session"],
    })
    conclusion = (
        f"\n\n[분석] 현재 자세를 숙련자({case['expert']}) 기준과 비교했습니다. "
        f"wrist_r 관절의 roll={case['joints']['wrist_r']['roll']}도, pitch={case['joints']['wrist_r']['pitch']}도로, "
        f"숙련자 대비 차이가 확인됩니다. timeseries 조회 결과, 세션 {case['session']} 동안의 자세 변화 추이를 확인했습니다.\n"
        f"[판정] 자세 교정 필요 — pitch 각도가 숙련자 기준보다 높음\n"
        f"[조치] 1) 손목 각도를 5~10도 낮추어 스프레이 방향 조정 2) 팔꿈치 높이를 어깨선에 맞추기\n"
        f"[근거] pose_estimation + timeseries 연계 분석, 세션 {case['session']}"
    )
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": case["user"]},
            {"role": "assistant", "content": think + tc1 + bridge + tc2 + conclusion},
        ],
        "category_type": "medium_chain",
        "chain_type": "pose_ts",
    }


def generate_all() -> list[dict]:
    samples = []
    for case in LOC_SEQ_CASES:
        samples.append(_build_loc_seq_sample(case))
    for case in DFT_ANOMALY_CASES:
        samples.append(_build_dft_anomaly_sample(case))
    for case in POSE_TS_CASES:
        samples.append(_build_pose_ts_sample(case))
    return samples


def verify(samples: list[dict]) -> None:
    assert len(samples) >= 30, f"Expected >=30 samples, got {len(samples)}"
    cats = Counter(s["chain_type"] for s in samples)
    for chain in ["loc_seq", "dft_anomaly", "pose_ts"]:
        assert cats.get(chain, 0) >= 10, f"{chain}: got {cats.get(chain, 0)}, expected >=10"
    for i, s in enumerate(samples, 1):
        content = s["messages"][-1]["content"]
        assert "<think>" in content, f"Sample {i} missing <think>"
        assert content.count("<tool_call>") >= 2, f"Sample {i} has <2 tool calls"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate medium-difficulty multi-tool chain SFT JSONL")
    p.add_argument("--output", type=str, default="/data/sft/sft_medium.jsonl")
    p.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")

    samples = generate_all()
    verify(samples)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for s in samples:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")

    cats = Counter(s["chain_type"] for s in samples)
    logger.info("Generated %d medium chain samples to %s", len(samples), out)
    for k, v in sorted(cats.items()):
        logger.info("  %s: %d", k, v)
    print(f"total={len(samples)}")
    for k, v in sorted(cats.items()):
        print(f"{k}={v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
