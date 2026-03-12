#!/usr/bin/env python3
"""Generate FC-aware SFT data for tool-calling capability.

Creates training samples that include <tool_call> patterns so the QLoRA
adapter learns to invoke MCP tools when context_data contains actionable
measurements, coordinates, or sensor data.

The output is merged with existing document QA SFT data to produce a
mixed dataset that preserves domain QA ability while adding FC capability.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import random
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


SYSTEM_PROMPT_FC = (
    "너는 선박 도장 분야 전문 AI 어시스턴트이다. "
    "제공된 문서를 근거로 정확한 기술 정보를 제공하고, "
    "답변 마지막에 근거 문서 ID를 반드시 명시한다. "
    "근거가 부족하면 솔직히 '제공된 문서에 해당 정보가 없습니다'라고 답한다. "
    "안전 관련 질문에는 항상 관련 규정을 함께 안내한다.\n\n"
    "[도구 사용 규칙]\n"
    "환경 데이터에 측정값, 좌표, 센서 데이터가 포함된 경우 반드시 적절한 도구를 호출하여 분석한다.\n"
    "도구를 호출할 때는 아래 형식을 정확히 따른다.\n\n"
    '예시 1: DFT 합부 판정 요청 시\n'
    '<tool_call>{"name": "dft_judge", "arguments": {"measurements": [95, 102, 88], '
    '"spec": {"nominal": 100, "tolerance_rule": "90/10"}}}</tool_call>\n\n'
    '예시 2: 이상 탐지 요청 시\n'
    '<tool_call>{"name": "anomaly_detection", "arguments": {"image_context": "좌하단 저도막 패턴", '
    '"dft_measurements": [85, 92, 78], "env": {"temperature": 28, "humidity": 65}}}</tool_call>\n\n'
    "도구 호출 후 결과를 종합하여 최종 답변을 생성한다. 여러 도구가 필요하면 모두 호출한다.\n\n"
    "[수치 정확성 규칙]\n"
    "수치 분석이 필요한 경우 반드시 도구를 호출하여 결과를 얻는다.\n"
    "도구가 반환한 수치를 그대로 인용하고, 수치를 직접 계산하거나 추정하지 않는다.\n"
    "도구 호출 없이 합격/불합격, 정상/이상 등의 판정을 내리지 않는다."
)


SYSTEM_PROMPT_DOCQA = (
    "너는 선박 도장 분야 전문 AI 어시스턴트이다. "
    "제공된 문서를 근거로 정확한 기술 정보를 제공하고, "
    "답변 마지막에 근거 문서 ID를 반드시 명시한다. "
    "근거가 부족하면 솔직히 '제공된 문서에 해당 정보가 없습니다'라고 답한다. "
    "안전 관련 질문에는 항상 관련 규정을 함께 안내한다."
)


def _dft_judge_samples() -> list[dict[str, Any]]:
    """Generate DFT judge tool-calling samples."""
    templates = [
        {
            "question": "DFT 측정값 {meas}으로 nominal {nom}um {rule} 기준 합부 판정해줘.",
            "context_data": {"measurements": None, "spec": None},
            "tool_name": "dft_judge",
        },
        {
            "question": "이 구역 DFT가 {meas}인데 규격 통과하는지 확인해줘. 기준은 nominal {nom}um {rule}.",
            "context_data": {"measurements": None, "spec": None},
            "tool_name": "dft_judge",
        },
        {
            "question": "측정값 {meas}, 기준 {nom}um, {rule} 룰 적용 시 합격 여부 알려줘.",
            "context_data": {"measurements": None, "spec": None},
            "tool_name": "dft_judge",
        },
        {
            "question": "DFT 검사 결과 {meas}를 받았어. nominal {nom}um에 {rule} 규정으로 판정 부탁해.",
            "context_data": {"measurements": None, "spec": None},
            "tool_name": "dft_judge",
        },
    ]

    samples = []
    measurement_sets = [
        ([95, 102, 88, 105, 99], 100, "90/10", "pass"),
        ([78, 82, 85, 79, 81], 100, "90/10", "fail"),
        ([145, 152, 138, 148, 155], 150, "90/10", "pass"),
        ([120, 115, 110, 108, 125], 125, "90/10", "fail"),
        ([198, 205, 192, 210, 200], 200, "90/10", "pass"),
        ([88, 92, 95, 90, 87], 100, "90/10", "fail"),
        ([102, 98, 105, 101, 100], 100, "90/10", "pass"),
        ([155, 160, 148, 162, 157], 150, "90/10", "pass"),
        ([72, 68, 75, 70, 65], 80, "90/10", "fail"),
        ([85, 88, 82, 90, 86], 80, "90/10", "pass"),
        ([130, 128, 135, 132, 126], 125, "90/10", "pass"),
        ([95, 90, 88, 92, 94], 100, "90/10", "fail"),
        ([175, 180, 172, 185, 178], 175, "90/10", "pass"),
        ([60, 65, 58, 62, 64], 75, "90/10", "fail"),
        ([110, 108, 115, 112, 106], 100, "90/10", "pass"),
    ]

    for meas, nom, rule, verdict in measurement_sets:
        tmpl = random.choice(templates)
        meas_str = str(meas)
        question = tmpl["question"].format(meas=meas_str, nom=nom, rule=rule)
        spec = {"nominal": nom, "tolerance_rule": rule}

        tc_json = json.dumps(
            {"name": "dft_judge", "arguments": {"measurements": meas, "spec": spec}},
            ensure_ascii=False,
        )

        if verdict == "pass":
            answer = (
                f"제공된 DFT 측정값을 분석하겠습니다.\n\n"
                f"<tool_call>{tc_json}</tool_call>\n\n"
                f"판정 결과: **합격(Pass)**\n"
                f"측정값 {meas_str}은 nominal {nom}um 기준 {rule} 룰을 충족합니다. "
                f"모든 측정값이 허용 범위 내에 있어 도막 두께가 적합합니다.\n\n"
                f"근거 문서 ID: DOC-doc_0033-C0326"
            )
        else:
            answer = (
                f"제공된 DFT 측정값을 분석하겠습니다.\n\n"
                f"<tool_call>{tc_json}</tool_call>\n\n"
                f"판정 결과: **불합격(Fail)**\n"
                f"측정값 {meas_str}은 nominal {nom}um 기준 {rule} 룰을 충족하지 못합니다. "
                f"일부 측정값이 허용 하한 미만이거나 편차가 큽니다. "
                f"해당 구역의 재도장이 필요합니다.\n\n"
                f"근거 문서 ID: DOC-doc_0033-C0326"
            )

        user_content = f"[환경 데이터]\nmeasurements: {meas_str}\nspec: {json.dumps(spec, ensure_ascii=False)}\n\n질문: {question}"

        samples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_FC},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": answer},
            ],
            "category": "tool_calling_dft",
        })

    return samples


def _anomaly_detection_samples() -> list[dict[str, Any]]:
    """Generate anomaly detection tool-calling samples."""
    scenarios = [
        {
            "image_context": "좌하단 B-3-L 구역 청색 집중, 전체 대비 저도막 패턴",
            "dft": [84, 86, 91, 89, 87],
            "env": {"temperature": 22, "humidity": 68},
            "anomaly_type": "저도막",
            "cause": "스프레이 건 거리가 과도하거나 이동 속도가 빠름",
        },
        {
            "image_context": "상단 A-2-R 구역 적색 집중, 과도막 패턴 확인",
            "dft": [135, 142, 128, 145, 138],
            "env": {"temperature": 18, "humidity": 55},
            "anomaly_type": "과도막",
            "cause": "도장 작업 시 분사량이 과다하거나 겹침 도장이 발생",
        },
        {
            "image_context": "중앙부 불균일 분포, 줄무늬 패턴",
            "dft": [95, 112, 88, 118, 92],
            "env": {"temperature": 25, "humidity": 72},
            "anomaly_type": "불균일 도막",
            "cause": "분사 패턴의 오버랩 불균일 또는 분사 압력 변동",
        },
        {
            "image_context": "전체 균일 분포, 이상 없음",
            "dft": [98, 102, 100, 99, 101],
            "env": {"temperature": 20, "humidity": 60},
            "anomaly_type": "정상",
            "cause": None,
        },
        {
            "image_context": "우측 경계부 급격한 두께 변화, 엣지 효과",
            "dft": [105, 108, 72, 68, 65],
            "env": {"temperature": 23, "humidity": 65},
            "anomaly_type": "엣지 저도막",
            "cause": "모서리 부위에서 도료 비산으로 도막 부착 불량",
        },
        {
            "image_context": "하단 전체 황색 경고, 건조 불량 의심",
            "dft": [110, 115, 108, 112, 118],
            "env": {"temperature": 15, "humidity": 88},
            "anomaly_type": "건조 불량 의심",
            "cause": "높은 습도와 낮은 온도로 인한 건조 지연",
        },
        {
            "image_context": "좌측 상단 핀홀 패턴 산재",
            "dft": [92, 95, 88, 90, 94],
            "env": {"temperature": 32, "humidity": 45},
            "anomaly_type": "핀홀 결함",
            "cause": "고온 환경에서 용제 급속 증발로 인한 핀홀 발생",
        },
        {
            "image_context": "구역 전체 흐름 자국 (sagging) 관찰",
            "dft": [125, 130, 140, 145, 150],
            "env": {"temperature": 20, "humidity": 70},
            "anomaly_type": "처짐(sagging)",
            "cause": "수직면에서 과다 도포로 인한 도료 흘러내림",
        },
        {
            "image_context": "B-1 구역 전체적으로 균일, 소규모 편차 존재",
            "dft": [97, 103, 99, 101, 98],
            "env": {"temperature": 21, "humidity": 62},
            "anomaly_type": "정상 (미세 편차)",
            "cause": None,
        },
        {
            "image_context": "용접선 주변 도막 두께 급감",
            "dft": [100, 95, 55, 48, 92],
            "env": {"temperature": 24, "humidity": 58},
            "anomaly_type": "용접부 저도막",
            "cause": "용접 비드 표면 요철로 인한 도료 부착 불량",
        },
    ]

    questions = [
        "히트맵 분석 결과 이상이 있는지 확인해줘. DFT: {dft}, 환경: 온도 {temp}°C 습도 {hum}%.",
        "이 구역 히트맵에서 {ctx} 패턴이 보여. DFT {dft}와 함께 이상 탐지 해줘.",
        "DFT가 {dft}이고 히트맵에서 {ctx} 상태야. 이상 여부 분석해줘.",
        "현재 구역 히트맵과 DFT 데이터로 도막 상태 진단해줘. DFT: {dft}.",
    ]

    samples = []
    for sc in scenarios:
        q_tmpl = random.choice(questions)
        question = q_tmpl.format(
            dft=sc["dft"], ctx=sc["image_context"],
            temp=sc["env"]["temperature"], hum=sc["env"]["humidity"],
        )

        tc_args = {
            "image_context": sc["image_context"],
            "dft_measurements": sc["dft"],
            "env": sc["env"],
        }
        tc_json = json.dumps(
            {"name": "anomaly_detection", "arguments": tc_args},
            ensure_ascii=False,
        )

        if sc["anomaly_type"] in ("정상", "정상 (미세 편차)"):
            answer = (
                f"히트맵 이미지와 DFT 데이터를 분석하겠습니다.\n\n"
                f"<tool_call>{tc_json}</tool_call>\n\n"
                f"분석 결과: **정상**\n"
                f"히트맵 분석에서 '{sc['image_context']}' 상태이며, "
                f"DFT 측정값 {sc['dft']}은 규격 범위 내입니다. "
                f"특별한 이상 징후가 발견되지 않았습니다.\n\n"
                f"근거 문서 ID: DOC-doc_0037-C0308"
            )
        else:
            answer = (
                f"히트맵 이미지와 DFT 데이터를 분석하겠습니다.\n\n"
                f"<tool_call>{tc_json}</tool_call>\n\n"
                f"분석 결과: **{sc['anomaly_type']}** 이상 탐지\n"
                f"히트맵에서 '{sc['image_context']}' 패턴이 확인되었으며, "
                f"DFT 측정값 {sc['dft']}에서 편차가 감지되었습니다.\n\n"
                f"**원인 분석**: {sc['cause']}\n"
                f"**조치 방안**: 해당 구역의 도막 상태를 재검사하고, "
                f"원인에 따른 교정 작업을 진행해야 합니다.\n\n"
                f"근거 문서 ID: DOC-doc_0037-C0308"
            )

        ctx_str = json.dumps(tc_args, ensure_ascii=False)
        user_content = f"[환경 데이터]\n{ctx_str}\n\n질문: {question}"

        samples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_FC},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": answer},
            ],
            "category": "tool_calling_anomaly",
        })

    return samples


def _location_tracking_samples() -> list[dict[str, Any]]:
    """Generate location tracking tool-calling samples."""
    scenarios = [
        {
            "coords": [{"x": 320, "y": 150, "z": 280, "timestamp": 1710000000}],
            "ship_model_id": "HULL-2024-001",
            "zone": "B-3-L",
            "zone_type": "hull_side",
            "coverage": 72.5,
        },
        {
            "coords": [{"x": 520, "y": 75, "z": 310, "timestamp": 1710001800}],
            "ship_model_id": "HULL-2024-001",
            "zone": "A-2-R",
            "zone_type": "upper_deck",
            "coverage": 45.0,
        },
        {
            "coords": [{"x": 180, "y": 200, "z": 120, "timestamp": 1710003600}],
            "ship_model_id": "HULL-2024-002",
            "zone": "C-1-B",
            "zone_type": "bottom",
            "coverage": 88.3,
        },
        {
            "coords": [{"x": 450, "y": 100, "z": 250, "timestamp": 1710005400}],
            "ship_model_id": "HULL-2024-001",
            "zone": "B-2-R",
            "zone_type": "hull_side",
            "coverage": 60.0,
        },
        {
            "coords": [{"x": 600, "y": 50, "z": 350, "timestamp": 1710007200}],
            "ship_model_id": "HULL-2024-003",
            "zone": "A-1-L",
            "zone_type": "upper_deck",
            "coverage": 30.2,
        },
    ]

    questions = [
        "현재 작업 구역이 어디인지 확인해줘.",
        "지금 어느 구역에서 작업 중인지 알려줘.",
        "현재 좌표 기준 도장 구역 위치를 파악해줘.",
        "작업자가 현재 위치한 구역 정보를 알려줘.",
    ]

    samples = []
    for sc in scenarios:
        question = random.choice(questions)
        tc_args: dict[str, Any] = {"coordinates": sc["coords"]}
        if sc.get("ship_model_id"):
            tc_args["ship_model_id"] = sc["ship_model_id"]

        tc_json = json.dumps(
            {"name": "location_tracking", "arguments": tc_args},
            ensure_ascii=False,
        )

        answer = (
            f"현재 위치를 확인하겠습니다.\n\n"
            f"<tool_call>{tc_json}</tool_call>\n\n"
            f"현재 작업 구역: **{sc['zone']}** ({sc['zone_type']})\n"
            f"도장 커버리지: {sc['coverage']}%\n"
            f"{'커버리지가 높아 마무리 단계입니다.' if sc['coverage'] > 80 else '아직 미도장 영역이 남아있으므로 계속 진행하세요.'}\n\n"
            f"근거 문서 ID: DOC-doc_0038-C0046"
        )

        ctx_str = json.dumps(tc_args, ensure_ascii=False)
        user_content = f"[환경 데이터]\n{ctx_str}\n\n질문: {question}"

        samples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_FC},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": answer},
            ],
            "category": "tool_calling_location",
        })

    return samples


def _pose_estimation_samples() -> list[dict[str, Any]]:
    """Generate pose estimation tool-calling samples."""
    scenarios = [
        {
            "current": {"joints": {"wrist_r": {"x": 0.12, "y": -0.05, "z": 0.38, "roll": 5, "pitch": 18, "yaw": 8}}},
            "reference": {"joints": {"wrist_r": {"x": 0.1, "y": -0.02, "z": 0.41, "roll": 2, "pitch": 12, "yaw": 3}}},
            "deviation": "높음",
            "correction": "손목 각도를 낮추고, 팔꿈치를 몸에 더 가까이 유지",
        },
        {
            "current": {"joints": {"wrist_r": {"x": 0.10, "y": -0.02, "z": 0.40, "roll": 3, "pitch": 13, "yaw": 4}}},
            "reference": {"joints": {"wrist_r": {"x": 0.1, "y": -0.02, "z": 0.41, "roll": 2, "pitch": 12, "yaw": 3}}},
            "deviation": "낮음",
            "correction": "현재 자세가 숙련자 패턴과 유사하여 양호",
        },
        {
            "current": {"joints": {"wrist_r": {"x": 0.15, "y": -0.08, "z": 0.35, "roll": 10, "pitch": 25, "yaw": 12}}},
            "reference": {"joints": {"wrist_r": {"x": 0.1, "y": -0.02, "z": 0.41, "roll": 2, "pitch": 12, "yaw": 3}}},
            "deviation": "매우 높음",
            "correction": "스프레이 건 각도가 크게 벗어남, 숙련자 기준 자세로 교정 필요",
        },
        {
            "current": {"joints": {"wrist_r": {"x": 0.11, "y": -0.03, "z": 0.39, "roll": 3, "pitch": 14, "yaw": 5}}},
            "reference": {"joints": {"wrist_r": {"x": 0.1, "y": -0.02, "z": 0.41, "roll": 2, "pitch": 12, "yaw": 3}}},
            "deviation": "중간",
            "correction": "z축 거리를 약간 늘리고 pitch를 줄여 균일한 분사 확보",
        },
        {
            "current": {"joints": {"wrist_r": {"x": 0.09, "y": -0.01, "z": 0.42, "roll": 1, "pitch": 11, "yaw": 2}}},
            "reference": {"joints": {"wrist_r": {"x": 0.1, "y": -0.02, "z": 0.41, "roll": 2, "pitch": 12, "yaw": 3}}},
            "deviation": "매우 낮음",
            "correction": "매우 양호한 자세, 숙련자 수준",
        },
    ]

    questions = [
        "현재 작업자의 자세가 숙련자와 비교해서 어떤지 확인해줘.",
        "스프레이 작업 자세를 분석하고 교정 방향을 알려줘.",
        "작업자의 현재 자세 편차를 숙련자 기준으로 분석해줘.",
        "자세 교정이 필요한지 확인해줘.",
    ]

    samples = []
    for sc in scenarios:
        question = random.choice(questions)
        tc_args = {
            "current_pose": sc["current"],
            "reference_pose": sc["reference"],
        }
        tc_json = json.dumps(
            {"name": "pose_estimation", "arguments": tc_args},
            ensure_ascii=False,
        )

        answer = (
            f"작업자 자세를 숙련자 패턴과 비교 분석하겠습니다.\n\n"
            f"<tool_call>{tc_json}</tool_call>\n\n"
            f"자세 편차 수준: **{sc['deviation']}**\n"
            f"교정 방향: {sc['correction']}\n\n"
            f"근거 문서 ID: DOC-doc_0044-C0227"
        )

        ctx_str = json.dumps(tc_args, ensure_ascii=False)
        user_content = f"[환경 데이터]\n{ctx_str}\n\n질문: {question}"

        samples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_FC},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": answer},
            ],
            "category": "tool_calling_pose",
        })

    return samples


def _sequence_validation_samples() -> list[dict[str, Any]]:
    """Generate sequence validation tool-calling samples."""
    scenarios = [
        {
            "trajectory": [{"x": 100, "y": 300, "z": 200}, {"x": 100, "y": 250, "z": 200}, {"x": 100, "y": 200, "z": 200}],
            "zone_id": "B-3",
            "zone_type": "hull_side",
            "ref_pattern": "top_to_bottom",
            "valid": False,
            "correction": "bottom_to_top",
        },
        {
            "trajectory": [{"x": 100, "y": 50, "z": 200}, {"x": 100, "y": 100, "z": 200}, {"x": 100, "y": 150, "z": 200}],
            "zone_id": "B-3",
            "zone_type": "hull_side",
            "ref_pattern": "bottom_to_top",
            "valid": True,
            "correction": None,
        },
        {
            "trajectory": [{"x": 50, "y": 100, "z": 200}, {"x": 100, "y": 100, "z": 200}, {"x": 150, "y": 100, "z": 200}],
            "zone_id": "A-1",
            "zone_type": "upper_deck",
            "ref_pattern": "left_to_right",
            "valid": True,
            "correction": None,
        },
        {
            "trajectory": [{"x": 200, "y": 100, "z": 200}, {"x": 150, "y": 100, "z": 200}, {"x": 100, "y": 100, "z": 200}],
            "zone_id": "A-1",
            "zone_type": "upper_deck",
            "ref_pattern": "left_to_right",
            "valid": False,
            "correction": "left_to_right",
        },
        {
            "trajectory": [{"x": 100, "y": 200, "z": 100}, {"x": 100, "y": 150, "z": 100}, {"x": 100, "y": 100, "z": 100}],
            "zone_id": "C-2",
            "zone_type": "bottom",
            "ref_pattern": "top_to_bottom",
            "valid": True,
            "correction": None,
        },
    ]

    questions = [
        "현재 도장 순서가 맞는지 확인해줘.",
        "이 구역의 stroke 순서를 숙련자 기준으로 검증해줘.",
        "작업자의 도장 방향이 올바른지 분석해줘.",
        "도장 순서 위반 여부를 확인해줘.",
    ]

    samples = []
    for sc in scenarios:
        question = random.choice(questions)
        tc_args = {
            "trajectory": sc["trajectory"],
            "zone_id": sc["zone_id"],
            "zone_type": sc["zone_type"],
            "reference_pattern": sc["ref_pattern"],
        }
        tc_json = json.dumps(
            {"name": "sequence_validation", "arguments": tc_args},
            ensure_ascii=False,
        )

        if sc["valid"]:
            answer = (
                f"도장 순서를 검증하겠습니다.\n\n"
                f"<tool_call>{tc_json}</tool_call>\n\n"
                f"검증 결과: **적합**\n"
                f"현재 작업자의 stroke 순서가 {sc['zone_id']} 구역({sc['zone_type']})의 "
                f"숙련자 참조 패턴({sc['ref_pattern']})과 일치합니다.\n\n"
                f"근거 문서 ID: DOC-doc_0044-C0227"
            )
        else:
            answer = (
                f"도장 순서를 검증하겠습니다.\n\n"
                f"<tool_call>{tc_json}</tool_call>\n\n"
                f"검증 결과: **부적합 — 순서 위반**\n"
                f"현재 stroke 방향이 {sc['zone_id']} 구역의 숙련자 패턴({sc['ref_pattern']})과 "
                f"다릅니다. 교정 방향은 '{sc['correction']}'으로 변경해야 합니다.\n\n"
                f"근거 문서 ID: DOC-doc_0044-C0227"
            )

        ctx_str = json.dumps(tc_args, ensure_ascii=False)
        user_content = f"[환경 데이터]\n{ctx_str}\n\n질문: {question}"

        samples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_FC},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": answer},
            ],
            "category": "tool_calling_sequence",
        })

    return samples


def _timeseries_samples() -> list[dict[str, Any]]:
    """Generate timeseries tool-calling samples."""
    scenarios = [
        {
            "session_id": "sess-ts-001",
            "action": "update",
            "data": {"timestamp": 1710000600, "speed": 1.05, "distance": 320, "angle": 45},
            "description": "작업 시작 후 초기 데이터",
        },
        {
            "session_id": "sess-ts-002",
            "action": "query",
            "data": None,
            "description": "현재 세션 추세 조회",
        },
        {
            "session_id": "sess-ts-003",
            "action": "update",
            "data": {"timestamp": 1710001200, "speed": 0.85, "distance": 290, "angle": 52},
            "description": "속도 감소 추세 데이터",
        },
        {
            "session_id": "sess-ts-004",
            "action": "update",
            "data": {"timestamp": 1710001800, "speed": 1.25, "distance": 380, "angle": 38},
            "description": "속도 증가, 각도 변화 데이터",
        },
        {
            "session_id": "sess-ts-005",
            "action": "query",
            "data": None,
            "description": "30분간 데이터 추세 확인",
        },
    ]

    questions_update = [
        "현재 작업 데이터를 시계열에 추가해줘.",
        "이 시점의 속도/거리/각도 데이터를 기록해줘.",
        "실시간 작업 데이터를 업데이트해줘.",
    ]

    questions_query = [
        "현재까지의 작업 추세를 확인해줘.",
        "시계열 데이터에서 현재 추세와 편차를 알려줘.",
        "작업 세션의 시계열 상태를 조회해줘.",
    ]

    samples = []
    for sc in scenarios:
        if sc["action"] == "update":
            question = random.choice(questions_update)
            tc_args: dict[str, Any] = {
                "action": "update",
                "session_id": sc["session_id"],
                "data": sc["data"],
            }
        else:
            question = random.choice(questions_query)
            tc_args = {
                "action": "query",
                "session_id": sc["session_id"],
            }

        tc_json = json.dumps(
            {"name": "timeseries", "arguments": tc_args},
            ensure_ascii=False,
        )

        if sc["action"] == "update":
            answer = (
                f"시계열 데이터를 업데이트하겠습니다.\n\n"
                f"<tool_call>{tc_json}</tool_call>\n\n"
                f"세션 {sc['session_id']}에 데이터가 추가되었습니다.\n"
                f"속도: {sc['data']['speed']}m/s, 거리: {sc['data']['distance']}mm, "
                f"각도: {sc['data']['angle']}°\n\n"
                f"근거 문서 ID: DOC-doc_0018-C0037"
            )
        else:
            answer = (
                f"현재 세션의 시계열 상태를 조회하겠습니다.\n\n"
                f"<tool_call>{tc_json}</tool_call>\n\n"
                f"세션 {sc['session_id']}의 추세를 확인했습니다.\n"
                f"현재까지의 데이터에서 속도와 각도의 편차 추세를 분석합니다.\n\n"
                f"근거 문서 ID: DOC-doc_0018-C0037"
            )

        ctx_str = json.dumps(tc_args, ensure_ascii=False)
        user_content = f"[환경 데이터]\n{ctx_str}\n\n질문: {question}"

        samples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_FC},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": answer},
            ],
            "category": "tool_calling_timeseries",
        })

    return samples


def _multi_tool_samples() -> list[dict[str, Any]]:
    """Generate samples requiring multiple tool calls."""
    samples = []

    # Scenario 1: anomaly + dft_judge
    meas1 = [84, 86, 91, 89, 87]
    spec1 = {"nominal": 100, "tolerance_rule": "90/10"}
    tc1_json = json.dumps(
        {"name": "anomaly_detection", "arguments": {
            "image_context": "하단 영역 저도막 패턴",
            "dft_measurements": meas1,
            "env": {"temperature": 22, "humidity": 68},
        }}, ensure_ascii=False,
    )
    tc2_json = json.dumps(
        {"name": "dft_judge", "arguments": {
            "measurements": meas1,
            "spec": spec1,
        }}, ensure_ascii=False,
    )
    samples.append({
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_FC},
            {"role": "user", "content": (
                "[환경 데이터]\n"
                f"image_context: \"하단 영역 저도막 패턴\"\n"
                f"dft_measurements: {meas1}\n"
                f"env: {{\"temperature\": 22, \"humidity\": 68}}\n"
                f"spec: {json.dumps(spec1, ensure_ascii=False)}\n\n"
                "질문: 이 DFT 데이터에 이상이 있는지 확인하고 합격 여부도 알려줘."
            )},
            {"role": "assistant", "content": (
                "히트맵 이상 탐지와 DFT 합부 판정을 모두 수행하겠습니다.\n\n"
                f"<tool_call>{tc1_json}</tool_call>\n\n"
                f"<tool_call>{tc2_json}</tool_call>\n\n"
                "**이상 탐지 결과**: 하단 영역에서 저도막 패턴이 확인되었습니다.\n"
                f"**DFT 판정 결과**: 불합격(Fail) — 측정값 {meas1}이 nominal 100um 90/10 기준 미달.\n\n"
                "저도막 원인은 스프레이 건 거리 과다 또는 이동 속도 과속으로 추정됩니다. "
                "해당 구역의 재도장이 필요합니다.\n\n"
                "근거 문서 ID: DOC-doc_0037-C0308, DOC-doc_0033-C0326"
            )},
        ],
        "category": "tool_calling_multi",
    })

    # Scenario 2: location + sequence_validation
    coords2 = [{"x": 320, "y": 150, "z": 280, "timestamp": 1710000000}]
    traj2 = [{"x": 100, "y": 300, "z": 200}, {"x": 100, "y": 250, "z": 200}, {"x": 100, "y": 200, "z": 200}]
    tc3_json = json.dumps(
        {"name": "location_tracking", "arguments": {
            "coordinates": coords2,
            "ship_model_id": "HULL-2024-001",
        }}, ensure_ascii=False,
    )
    tc4_json = json.dumps(
        {"name": "sequence_validation", "arguments": {
            "trajectory": traj2,
            "zone_id": "B-3",
            "zone_type": "hull_side",
            "reference_pattern": "bottom_to_top",
        }}, ensure_ascii=False,
    )
    samples.append({
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_FC},
            {"role": "user", "content": (
                "[환경 데이터]\n"
                f"coordinates: {json.dumps(coords2, ensure_ascii=False)}\n"
                f"trajectory: {json.dumps(traj2, ensure_ascii=False)}\n"
                f"ship_model_id: \"HULL-2024-001\"\n"
                f"reference_pattern: \"bottom_to_top\"\n\n"
                "질문: 현재 구역 위치 확인하고 도장 순서가 맞는지도 확인해줘."
            )},
            {"role": "assistant", "content": (
                "현재 위치 확인과 도장 순서 검증을 수행하겠습니다.\n\n"
                f"<tool_call>{tc3_json}</tool_call>\n\n"
                f"<tool_call>{tc4_json}</tool_call>\n\n"
                "**위치**: B-3-L 구역 (hull_side)\n"
                "**순서 검증**: 부적합 — 현재 top→bottom 방향이지만 숙련자 패턴은 bottom→top입니다.\n"
                "교정 방향: 아래에서 위로(bottom_to_top) 변경해주세요.\n\n"
                "근거 문서 ID: DOC-doc_0038-C0046, DOC-doc_0044-C0227"
            )},
        ],
        "category": "tool_calling_multi",
    })

    # Scenario 3: location + pose_estimation (XR content)
    coords3 = [{"x": 520, "y": 75, "z": 310, "timestamp": 1710001800}]
    cur_pose = {"joints": {"wrist_r": {"x": 0.12, "y": -0.05, "z": 0.38, "roll": 5, "pitch": 18, "yaw": 8}}}
    ref_pose = {"joints": {"wrist_r": {"x": 0.1, "y": -0.02, "z": 0.41, "roll": 2, "pitch": 12, "yaw": 3}}}
    tc5_json = json.dumps(
        {"name": "location_tracking", "arguments": {"coordinates": coords3}},
        ensure_ascii=False,
    )
    tc6_json = json.dumps(
        {"name": "pose_estimation", "arguments": {
            "current_pose": cur_pose,
            "reference_pose": ref_pose,
        }}, ensure_ascii=False,
    )
    samples.append({
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_FC},
            {"role": "user", "content": (
                "[환경 데이터]\n"
                f"coordinates: {json.dumps(coords3, ensure_ascii=False)}\n"
                f"current_pose: {json.dumps(cur_pose, ensure_ascii=False)}\n"
                f"reference_pose: {json.dumps(ref_pose, ensure_ascii=False)}\n\n"
                "질문: 이 구역의 도장 방법을 XR 가이드로 만들어줘."
            )},
            {"role": "assistant", "content": (
                "XR 가이드 생성을 위해 위치와 자세를 분석하겠습니다.\n\n"
                f"<tool_call>{tc5_json}</tool_call>\n\n"
                f"<tool_call>{tc6_json}</tool_call>\n\n"
                "**XR 가이드 — 구역 A-2-R**\n\n"
                "**Step 1**: 스프레이 건을 피도체로부터 약 41cm 거리에서 유지\n"
                "**Step 2**: 손목 pitch를 12° 이하로 유지 (현재 18° → 교정 필요)\n"
                "**Step 3**: bottom_to_top 방향으로 일정한 속도로 이동\n"
                "**Step 4**: 오버랩 50%를 유지하며 균일하게 도포\n\n"
                "**주의사항**: 현재 자세 편차가 높으므로 교정 후 작업 재개를 권장합니다.\n\n"
                "근거 문서 ID: DOC-doc_0038-C0046, DOC-doc_0044-C0227"
            )},
        ],
        "category": "tool_calling_multi",
    })

    # Scenario 4: timeseries + anomaly + dft (CoT reasoning)
    ts_args = {"action": "update", "session_id": "sess-cot-001",
               "data": {"timestamp": 1710001200, "speed": 1.12, "distance": 335, "angle": 57}}
    meas4 = [91, 108, 87, 112, 89, 105]
    spec4 = {"nominal": 100, "tolerance_rule": "90/10"}
    tc7_json = json.dumps({"name": "timeseries", "arguments": ts_args}, ensure_ascii=False)
    tc8_json = json.dumps(
        {"name": "anomaly_detection", "arguments": {
            "image_context": "구역 중앙 저도막과 과도막이 교차",
            "dft_measurements": meas4,
            "env": {"temperature": 24, "humidity": 65},
        }}, ensure_ascii=False,
    )
    tc9_json = json.dumps(
        {"name": "dft_judge", "arguments": {"measurements": meas4, "spec": spec4}},
        ensure_ascii=False,
    )
    samples.append({
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_FC},
            {"role": "user", "content": (
                "[환경 데이터]\n"
                f"session_id: \"sess-cot-001\"\n"
                f"timeseries_data: {json.dumps(ts_args['data'], ensure_ascii=False)}\n"
                f"image_context: \"구역 중앙 저도막과 과도막이 교차\"\n"
                f"dft_measurements: {meas4}\n"
                f"env: {{\"temperature\": 24, \"humidity\": 65}}\n"
                f"spec: {json.dumps(spec4, ensure_ascii=False)}\n\n"
                "질문: 최근 30분간 DFT 편차가 커졌어. 공정 최적화를 위해 원인 체인을 분석해줘."
            )},
            {"role": "assistant", "content": (
                "시계열 추세, 이상 탐지, DFT 판정을 종합하여 원인을 분석하겠습니다.\n\n"
                f"<tool_call>{tc7_json}</tool_call>\n\n"
                f"<tool_call>{tc8_json}</tool_call>\n\n"
                f"<tool_call>{tc9_json}</tool_call>\n\n"
                "**원인 체인 분석**:\n"
                "1. 시계열 데이터에서 속도 1.12m/s, 각도 57°로 기준 대비 편차 발생\n"
                "2. 히트맵 분석: 중앙부에서 저도막-과도막 교차 패턴 → 불균일 분사\n"
                "3. DFT 측정값 편차: min 87 ~ max 112 → 25um 편차 (과도)\n\n"
                "**원인 추정**: 이동 속도 불균일 + 분사 각도 변동이 겹쳐 발생\n"
                "**조치 방안**: 이동 속도를 0.9~1.0m/s로 안정화하고, "
                "분사 각도를 45° 이하로 유지하여 균일한 도막을 확보하세요.\n\n"
                "근거 문서 ID: DOC-doc_0018-C0037, DOC-doc_0037-C0308, DOC-doc_0033-C0326"
            )},
        ],
        "category": "tool_calling_multi",
    })

    # Scenario 5: full coaching (location + pose + sequence + timeseries)
    coords5 = [{"x": 400, "y": 120, "z": 260, "timestamp": 1710002400}]
    cur5 = {"joints": {"wrist_r": {"x": 0.11, "y": -0.04, "z": 0.39, "roll": 4, "pitch": 16, "yaw": 6}}}
    ref5 = {"joints": {"wrist_r": {"x": 0.1, "y": -0.02, "z": 0.41, "roll": 2, "pitch": 12, "yaw": 3}}}
    traj5 = [{"x": 100, "y": 50, "z": 200}, {"x": 100, "y": 100, "z": 200}, {"x": 100, "y": 150, "z": 200}]

    tc_loc = json.dumps({"name": "location_tracking", "arguments": {"coordinates": coords5}}, ensure_ascii=False)
    tc_pose = json.dumps({"name": "pose_estimation", "arguments": {"current_pose": cur5, "reference_pose": ref5}}, ensure_ascii=False)
    tc_seq = json.dumps({"name": "sequence_validation", "arguments": {"trajectory": traj5, "zone_id": "B-2", "zone_type": "hull_side", "reference_pattern": "bottom_to_top"}}, ensure_ascii=False)
    tc_ts = json.dumps({"name": "timeseries", "arguments": {"action": "update", "session_id": "sess-coach-001", "data": {"timestamp": 1710002400, "speed": 0.95, "distance": 310, "angle": 44}}}, ensure_ascii=False)

    samples.append({
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_FC},
            {"role": "user", "content": (
                "[환경 데이터]\n"
                f"coordinates: {json.dumps(coords5, ensure_ascii=False)}\n"
                f"current_pose: {json.dumps(cur5, ensure_ascii=False)}\n"
                f"reference_pose: {json.dumps(ref5, ensure_ascii=False)}\n"
                f"trajectory: {json.dumps(traj5, ensure_ascii=False)}\n"
                f"session_id: \"sess-coach-001\"\n"
                f"timeseries_data: {{\"timestamp\": 1710002400, \"speed\": 0.95, \"distance\": 310, \"angle\": 44}}\n\n"
                "질문: 실시간 코칭 모드로 현재 작업자에게 자세/순서/속도까지 종합 피드백을 줘."
            )},
            {"role": "assistant", "content": (
                "종합 실시간 코칭 피드백을 위해 위치, 자세, 순서, 시계열을 모두 분석하겠습니다.\n\n"
                f"<tool_call>{tc_loc}</tool_call>\n\n"
                f"<tool_call>{tc_pose}</tool_call>\n\n"
                f"<tool_call>{tc_seq}</tool_call>\n\n"
                f"<tool_call>{tc_ts}</tool_call>\n\n"
                "**종합 코칭 피드백 — B-2 구역**\n\n"
                "1. **위치**: B-2 hull_side 구역에서 작업 중\n"
                "2. **자세**: pitch 16° (기준 12°) — 약간 높음, 스프레이 건을 살짝 낮추세요\n"
                "3. **순서**: bottom_to_top 패턴 적합 ✓\n"
                "4. **속도**: 0.95m/s — 적정 범위 (0.8~1.1m/s)\n"
                "5. **거리**: 310mm — z축 39cm로 기준(41cm)보다 약간 가까움\n\n"
                "**교정 우선순위**: 자세 pitch 교정 > z축 거리 확보\n\n"
                "근거 문서 ID: DOC-doc_0038-C0046, DOC-doc_0044-C0227, DOC-doc_0018-C0037"
            )},
        ],
        "category": "tool_calling_multi",
    })

    return samples


def _no_tool_needed_samples() -> list[dict[str, Any]]:
    """Generate samples where NO tool call is needed (domain QA with FC system prompt).

    These teach the model that not every question requires a tool call,
    even when the FC-aware system prompt is active.
    """
    qa_pairs = [
        {
            "question": "에폭시 프라이머와 아연 프라이머의 차이점은?",
            "answer": (
                "에폭시 프라이머와 아연 프라이머의 주요 차이점은 다음과 같습니다:\n\n"
                "- **에폭시 프라이머**: 에폭시 수지를 기반으로 하며, 우수한 접착력과 내화학성을 제공합니다. "
                "주로 수선 하부(underwater area)에 사용됩니다.\n"
                "- **아연 프라이머(Zinc primer)**: 아연 분말을 함유하여 희생 양극 방식의 부식 방지 효과를 제공합니다. "
                "주로 블라스팅 직후 임시 방청용으로 사용됩니다.\n\n"
                "사용 환경과 목적에 따라 적합한 프라이머를 선택해야 합니다.\n\n"
                "근거 문서 ID: DOC-doc_0033-C0210, DOC-doc_0036-C0298"
            ),
        },
        {
            "question": "Sa 2.5 표면처리 등급은 어떤 상태를 의미해?",
            "answer": (
                "Sa 2.5는 ISO 8501-1에 정의된 블라스팅 표면 청결도 등급으로, "
                "'Very thorough blast-cleaning'을 의미합니다.\n\n"
                "이 등급에서는 표면의 밀스케일, 녹, 이물질이 거의 완전히 제거되어야 하며, "
                "잔류 오염은 전체 표면의 5% 이내로 허용됩니다. "
                "선박 도장에서 가장 일반적으로 요구되는 표면처리 등급입니다.\n\n"
                "근거 문서 ID: DOC-doc_0034-C0330"
            ),
        },
        {
            "question": "도장 작업 시 안전 장비는 무엇을 착용해야 해?",
            "answer": (
                "도장 작업 시 다음 안전 장비를 반드시 착용해야 합니다:\n\n"
                "1. **호흡 보호구**: 유기용제 증기 차단용 방독 마스크 (KOSHA GUIDE 준수)\n"
                "2. **보안경**: 도료 비산 방지\n"
                "3. **보호복**: 내화학성 도장복\n"
                "4. **안전장갑**: 내용제성 장갑\n"
                "5. **안전화**: 미끄럼 방지 기능\n\n"
                "밀폐 공간 작업 시에는 추가로 송기 마스크와 산소 농도 측정기가 필요합니다.\n\n"
                "근거 문서 ID: DOC-doc_0046-C0028"
            ),
        },
        {
            "question": "WFT와 DFT의 관계는 어떻게 되나요?",
            "answer": (
                "WFT(Wet Film Thickness)와 DFT(Dry Film Thickness)의 관계는 다음과 같습니다:\n\n"
                "DFT = WFT × (체적 고형분 비율 / 100)\n\n"
                "예를 들어 체적 고형분이 50%인 도료의 WFT가 200um이면 "
                "DFT는 약 100um이 됩니다. 도료의 체적 고형분 비율은 "
                "제품 기술 자료서(TDS)에서 확인할 수 있습니다.\n\n"
                "근거 문서 ID: DOC-doc_0033-C0326"
            ),
        },
        {
            "question": "PSPC란 무엇인가요?",
            "answer": (
                "PSPC(Performance Standard for Protective Coatings)는 "
                "IMO(국제해사기구)에서 제정한 방식 도장 성능 기준입니다.\n\n"
                "주요 내용:\n"
                "- 전용 해수 밸러스트 탱크와 이중 선체 공간의 도장 기준\n"
                "- 표면 처리 등급 Sa 2.5 이상 요구\n"
                "- 도막 두께, 도장 간격, 환경 조건 등 상세 기준 규정\n"
                "- 2008년부터 신조선에 의무 적용\n\n"
                "근거 문서 ID: DOC-doc_0035-C0150"
            ),
        },
    ]

    samples = []
    for qa in qa_pairs:
        user_content = f"질문: {qa['question']}"
        samples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_FC},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": qa["answer"]},
            ],
            "category": "no_tool_qa",
        })

    return samples


def _multi_turn_samples() -> list[dict[str, Any]]:
    """Generate multi-turn samples: user → assistant(tool_call) → tool(result) → assistant(final).

    Teaches the model to interpret tool results and produce structured
    Thought → Tool Call → Result → Final Answer responses.
    """
    samples = []

    # MT-1: DFT judge with tool result interpretation
    meas_mt1 = [95, 102, 88, 105, 99]
    spec_mt1 = {"nominal": 100, "tolerance_rule": "90/10"}
    tc_mt1 = json.dumps(
        {"name": "dft_judge", "arguments": {"measurements": meas_mt1, "spec": spec_mt1}},
        ensure_ascii=False,
    )
    tool_result_mt1 = json.dumps({
        "status": "success",
        "result": {
            "verdict": "pass",
            "mean": 97.8,
            "min": 88,
            "max": 105,
            "below_90pct": 1,
            "total": 5,
            "threshold_90pct": 90.0,
        },
    }, ensure_ascii=False)

    samples.append({
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_FC},
            {"role": "user", "content": (
                "[환경 데이터]\n"
                f"measurements: {meas_mt1}\n"
                f"spec: {json.dumps(spec_mt1, ensure_ascii=False)}\n\n"
                "질문: DFT 측정값으로 합부 판정해줘."
            )},
            {"role": "assistant", "content": f"DFT 합부 판정을 위해 도구를 호출합니다.\n\n<tool_call>{tc_mt1}</tool_call>"},
            {"role": "tool", "content": tool_result_mt1},
            {"role": "assistant", "content": (
                "도구 판정 결과를 바탕으로 답변합니다.\n\n"
                "**판정 결과: 합격(Pass)**\n"
                f"- 평균 DFT: 97.8um\n"
                f"- 최솟값: 88um, 최댓값: 105um\n"
                f"- 90% 하한(90.0um) 미만 측정값: 1개/5개\n"
                "- 90/10 규정 충족\n\n"
                "모든 측정값이 허용 범위 내에 있어 해당 구역은 합격입니다.\n\n"
                "근거 문서 ID: DOC-doc_0033-C0326"
            )},
        ],
        "category": "multi_turn_dft",
    })

    # MT-2: DFT judge fail case with tool result
    meas_mt2 = [78, 82, 85, 79, 81]
    spec_mt2 = {"nominal": 100, "tolerance_rule": "90/10"}
    tc_mt2 = json.dumps(
        {"name": "dft_judge", "arguments": {"measurements": meas_mt2, "spec": spec_mt2}},
        ensure_ascii=False,
    )
    tool_result_mt2 = json.dumps({
        "status": "success",
        "result": {
            "verdict": "fail",
            "mean": 81.0,
            "min": 78,
            "max": 85,
            "below_90pct": 5,
            "total": 5,
            "threshold_90pct": 90.0,
        },
    }, ensure_ascii=False)

    samples.append({
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_FC},
            {"role": "user", "content": (
                "[환경 데이터]\n"
                f"measurements: {meas_mt2}\n"
                f"spec: {json.dumps(spec_mt2, ensure_ascii=False)}\n\n"
                "질문: 이 DFT 측정값이 규격을 통과하는지 확인해줘."
            )},
            {"role": "assistant", "content": f"DFT 규격 검증을 위해 도구를 호출합니다.\n\n<tool_call>{tc_mt2}</tool_call>"},
            {"role": "tool", "content": tool_result_mt2},
            {"role": "assistant", "content": (
                "도구 판정 결과를 바탕으로 답변합니다.\n\n"
                "**판정 결과: 불합격(Fail)**\n"
                f"- 평균 DFT: 81.0um (nominal 100um 대비 19% 부족)\n"
                f"- 최솟값: 78um, 최댓값: 85um\n"
                f"- 90% 하한(90.0um) 미만 측정값: 5개/5개 (전수 미달)\n\n"
                "전체 측정값이 허용 하한 미만입니다. "
                "해당 구역은 재도장이 필요합니다.\n\n"
                "근거 문서 ID: DOC-doc_0033-C0326"
            )},
        ],
        "category": "multi_turn_dft",
    })

    # MT-3: Anomaly detection with tool result
    tc_mt3 = json.dumps(
        {"name": "anomaly_detection", "arguments": {
            "image_context": "좌하단 B-3-L 구역 청색 집중, 저도막 패턴",
            "dft_measurements": [84, 86, 91, 89, 87],
            "env": {"temperature": 22, "humidity": 68},
        }}, ensure_ascii=False,
    )
    tool_result_mt3 = json.dumps({
        "status": "success",
        "result": {
            "anomaly_detected": True,
            "anomaly_type": "low_thickness",
            "severity": "medium",
            "affected_area": "하단 20%",
            "confidence": 0.85,
        },
    }, ensure_ascii=False)

    samples.append({
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_FC},
            {"role": "user", "content": (
                "[환경 데이터]\n"
                "image_context: \"좌하단 B-3-L 구역 청색 집중, 저도막 패턴\"\n"
                "dft_measurements: [84, 86, 91, 89, 87]\n"
                "env: {\"temperature\": 22, \"humidity\": 68}\n\n"
                "질문: 히트맵에서 이상이 있는지 확인해줘."
            )},
            {"role": "assistant", "content": f"히트맵 이상 탐지를 위해 도구를 호출합니다.\n\n<tool_call>{tc_mt3}</tool_call>"},
            {"role": "tool", "content": tool_result_mt3},
            {"role": "assistant", "content": (
                "도구 분석 결과를 바탕으로 답변합니다.\n\n"
                "**이상 탐지 결과: 저도막(low_thickness) 이상 확인**\n"
                "- 심각도: 중간(medium)\n"
                "- 영향 범위: 하단 20% 영역\n"
                "- 신뢰도: 85%\n\n"
                "하단 영역에서 청색 집중 패턴이 저도막 상태를 나타냅니다. "
                "스프레이 건 거리가 과도하거나 이동 속도가 빨랐을 가능성이 있습니다. "
                "해당 구역의 추가 도장을 권장합니다.\n\n"
                "근거 문서 ID: DOC-doc_0037-C0308"
            )},
        ],
        "category": "multi_turn_anomaly",
    })

    # MT-4: Pose estimation with tool result
    cur_pose = {"joints": {"wrist_r": {"x": 0.12, "y": -0.05, "z": 0.38, "roll": 5, "pitch": 18, "yaw": 8}}}
    ref_pose = {"joints": {"wrist_r": {"x": 0.1, "y": -0.02, "z": 0.41, "roll": 2, "pitch": 12, "yaw": 3}}}
    tc_mt4 = json.dumps(
        {"name": "pose_estimation", "arguments": {"current_pose": cur_pose, "reference_pose": ref_pose}},
        ensure_ascii=False,
    )
    tool_result_mt4 = json.dumps({
        "status": "success",
        "result": {
            "overall_deviation": "high",
            "joint_deviations": {
                "wrist_r": {"pitch_error": 6, "yaw_error": 5, "z_distance_error": -0.03},
            },
            "correction_vector": {"pitch": -6, "yaw": -5, "z": 0.03},
        },
    }, ensure_ascii=False)

    samples.append({
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_FC},
            {"role": "user", "content": (
                "[환경 데이터]\n"
                f"current_pose: {json.dumps(cur_pose, ensure_ascii=False)}\n"
                f"reference_pose: {json.dumps(ref_pose, ensure_ascii=False)}\n\n"
                "질문: 현재 자세가 숙련자 기준에 맞는지 분석해줘."
            )},
            {"role": "assistant", "content": f"자세 비교 분석을 위해 도구를 호출합니다.\n\n<tool_call>{tc_mt4}</tool_call>"},
            {"role": "tool", "content": tool_result_mt4},
            {"role": "assistant", "content": (
                "도구 분석 결과를 바탕으로 답변합니다.\n\n"
                "**자세 편차: 높음(high)**\n"
                "- 손목 pitch 오차: +6° (기준 12° → 현재 18°)\n"
                "- 손목 yaw 오차: +5° (기준 3° → 현재 8°)\n"
                "- z축 거리 오차: -3cm (기준 41cm → 현재 38cm)\n\n"
                "**교정 방향**:\n"
                "1. 스프레이 건 pitch를 6° 낮춰주세요\n"
                "2. yaw 각도를 5° 줄여 정면 방향으로 맞춰주세요\n"
                "3. 피도체와의 거리를 3cm 더 확보하세요\n\n"
                "근거 문서 ID: DOC-doc_0044-C0227"
            )},
        ],
        "category": "multi_turn_pose",
    })

    # MT-5: Sequence validation with tool result
    traj_mt5 = [{"x": 100, "y": 300, "z": 200}, {"x": 100, "y": 250, "z": 200}, {"x": 100, "y": 200, "z": 200}]
    tc_mt5 = json.dumps(
        {"name": "sequence_validation", "arguments": {
            "trajectory": traj_mt5, "zone_id": "B-3", "zone_type": "hull_side",
            "reference_pattern": "bottom_to_top",
        }}, ensure_ascii=False,
    )
    tool_result_mt5 = json.dumps({
        "status": "success",
        "result": {
            "valid": False,
            "detected_pattern": "top_to_bottom",
            "reference_pattern": "bottom_to_top",
            "correction": "reverse_direction",
        },
    }, ensure_ascii=False)

    samples.append({
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_FC},
            {"role": "user", "content": (
                "[환경 데이터]\n"
                f"trajectory: {json.dumps(traj_mt5, ensure_ascii=False)}\n"
                "zone_id: \"B-3\"\nzone_type: \"hull_side\"\n"
                "reference_pattern: \"bottom_to_top\"\n\n"
                "질문: 도장 순서가 맞는지 확인해줘."
            )},
            {"role": "assistant", "content": f"도장 순서 검증을 위해 도구를 호출합니다.\n\n<tool_call>{tc_mt5}</tool_call>"},
            {"role": "tool", "content": tool_result_mt5},
            {"role": "assistant", "content": (
                "도구 검증 결과를 바탕으로 답변합니다.\n\n"
                "**순서 검증 결과: 부적합**\n"
                "- 감지된 패턴: top_to_bottom (위→아래)\n"
                "- 숙련자 기준 패턴: bottom_to_top (아래→위)\n"
                "- 교정: 방향을 반대로 전환\n\n"
                "B-3 hull_side 구역에서는 아래에서 위로(bottom_to_top) "
                "도장해야 도료 흘러내림을 방지할 수 있습니다. "
                "즉시 방향을 교정해주세요.\n\n"
                "근거 문서 ID: DOC-doc_0044-C0227"
            )},
        ],
        "category": "multi_turn_sequence",
    })

    # MT-6: Timeseries query with tool result
    tc_mt6 = json.dumps(
        {"name": "timeseries", "arguments": {"action": "query", "session_id": "sess-mt-006"}},
        ensure_ascii=False,
    )
    tool_result_mt6 = json.dumps({
        "status": "success",
        "result": {
            "session_id": "sess-mt-006",
            "point_count": 15,
            "trends": {
                "speed": {"mean": 1.05, "std": 0.12, "trend": "stable"},
                "distance": {"mean": 315, "std": 18, "trend": "increasing"},
                "angle": {"mean": 48, "std": 5.2, "trend": "stable"},
            },
        },
    }, ensure_ascii=False)

    samples.append({
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_FC},
            {"role": "user", "content": (
                "[환경 데이터]\n"
                "session_id: \"sess-mt-006\"\n\n"
                "질문: 현재 세션의 작업 추세를 확인해줘."
            )},
            {"role": "assistant", "content": f"시계열 추세를 조회합니다.\n\n<tool_call>{tc_mt6}</tool_call>"},
            {"role": "tool", "content": tool_result_mt6},
            {"role": "assistant", "content": (
                "도구 조회 결과를 바탕으로 답변합니다.\n\n"
                "**세션 sess-mt-006 추세 분석** (15개 데이터 포인트)\n"
                "- 속도: 평균 1.05m/s (표준편차 0.12) — 안정적\n"
                "- 거리: 평균 315mm (표준편차 18) — 점진적 증가 추세\n"
                "- 각도: 평균 48° (표준편차 5.2) — 안정적\n\n"
                "거리가 점진적으로 증가하고 있어 피도체와의 간격이 벌어지는 경향이 있습니다. "
                "일정 거리(300~320mm)를 유지하도록 주의해주세요.\n\n"
                "근거 문서 ID: DOC-doc_0018-C0037"
            )},
        ],
        "category": "multi_turn_timeseries",
    })

    # MT-7: Multi-tool with chained results (anomaly → dft)
    meas_mt7 = [84, 86, 91, 89, 87]
    spec_mt7 = {"nominal": 100, "tolerance_rule": "90/10"}
    tc_mt7a = json.dumps(
        {"name": "anomaly_detection", "arguments": {
            "image_context": "하단 영역 저도막",
            "dft_measurements": meas_mt7,
            "env": {"temperature": 22, "humidity": 68},
        }}, ensure_ascii=False,
    )
    tc_mt7b = json.dumps(
        {"name": "dft_judge", "arguments": {"measurements": meas_mt7, "spec": spec_mt7}},
        ensure_ascii=False,
    )
    tool_result_mt7a = json.dumps({
        "status": "success",
        "result": {"anomaly_detected": True, "anomaly_type": "low_thickness", "severity": "medium"},
    }, ensure_ascii=False)
    tool_result_mt7b = json.dumps({
        "status": "success",
        "result": {"verdict": "fail", "mean": 87.4, "below_90pct": 4, "total": 5},
    }, ensure_ascii=False)

    samples.append({
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_FC},
            {"role": "user", "content": (
                "[환경 데이터]\n"
                "image_context: \"하단 영역 저도막\"\n"
                f"dft_measurements: {meas_mt7}\n"
                "env: {\"temperature\": 22, \"humidity\": 68}\n"
                f"spec: {json.dumps(spec_mt7, ensure_ascii=False)}\n\n"
                "질문: 이상 여부와 합격 여부를 모두 확인해줘."
            )},
            {"role": "assistant", "content": (
                f"이상 탐지와 DFT 판정을 순차적으로 수행합니다.\n\n"
                f"<tool_call>{tc_mt7a}</tool_call>\n\n"
                f"<tool_call>{tc_mt7b}</tool_call>"
            )},
            {"role": "tool", "content": tool_result_mt7a},
            {"role": "tool", "content": tool_result_mt7b},
            {"role": "assistant", "content": (
                "두 도구의 분석 결과를 종합합니다.\n\n"
                "**1. 이상 탐지**: 저도막(low_thickness) 이상 감지, 심각도 중간\n"
                "**2. DFT 판정**: 불합격(Fail) — 평균 87.4um, 90% 하한 미만 4/5개\n\n"
                "하단 영역의 저도막 이상과 DFT 불합격이 동시에 확인되었습니다. "
                "해당 구역의 재도장이 필요하며, 재도장 전 스프레이 건 거리와 "
                "이동 속도를 점검해주세요.\n\n"
                "근거 문서 ID: DOC-doc_0037-C0308, DOC-doc_0033-C0326"
            )},
        ],
        "category": "multi_turn_chain",
    })

    # MT-8: Location tracking with tool result
    coords_mt8 = [{"x": 320, "y": 150, "z": 280, "timestamp": 1710000000}]
    tc_mt8 = json.dumps(
        {"name": "location_tracking", "arguments": {
            "coordinates": coords_mt8, "ship_model_id": "HULL-2024-001",
        }}, ensure_ascii=False,
    )
    tool_result_mt8 = json.dumps({
        "status": "success",
        "result": {
            "zone_id": "B-3-L",
            "zone_type": "hull_side",
            "coverage_pct": 72.5,
            "uncovered_areas": ["상단 25%", "좌측 모서리"],
        },
    }, ensure_ascii=False)

    samples.append({
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_FC},
            {"role": "user", "content": (
                "[환경 데이터]\n"
                f"coordinates: {json.dumps(coords_mt8, ensure_ascii=False)}\n"
                "ship_model_id: \"HULL-2024-001\"\n\n"
                "질문: 현재 작업 구역이 어디인지 확인해줘."
            )},
            {"role": "assistant", "content": f"현재 위치를 확인합니다.\n\n<tool_call>{tc_mt8}</tool_call>"},
            {"role": "tool", "content": tool_result_mt8},
            {"role": "assistant", "content": (
                "도구 확인 결과를 바탕으로 답변합니다.\n\n"
                "**현재 위치: B-3-L 구역 (hull_side)**\n"
                "- 도장 커버리지: 72.5%\n"
                "- 미도장 영역: 상단 25%, 좌측 모서리\n\n"
                "아직 미도장 영역이 남아있으므로 상단과 좌측 모서리를 "
                "우선적으로 도장해주세요.\n\n"
                "근거 문서 ID: DOC-doc_0038-C0046"
            )},
        ],
        "category": "multi_turn_location",
    })

    return samples


def generate_all_fc_samples(seed: int = 42) -> list[dict[str, Any]]:
    """Generate all FC-aware SFT samples."""
    random.seed(seed)

    samples: list[dict[str, Any]] = []
    samples.extend(_dft_judge_samples())
    samples.extend(_anomaly_detection_samples())
    samples.extend(_location_tracking_samples())
    samples.extend(_pose_estimation_samples())
    samples.extend(_sequence_validation_samples())
    samples.extend(_timeseries_samples())
    samples.extend(_multi_tool_samples())
    samples.extend(_multi_turn_samples())
    samples.extend(_no_tool_needed_samples())

    random.shuffle(samples)
    return samples


def load_existing_sft(path: Path) -> list[dict[str, Any]]:
    """Load existing SFT JSONL data."""
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def upgrade_docqa_system_prompt(
    samples: list[dict[str, Any]],
    fc_ratio: float = 0.3,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Upgrade a fraction of doc QA samples to use FC-aware system prompt.

    This teaches the model that the FC system prompt doesn't mean
    every response needs a tool call.
    """
    rng = random.Random(seed)
    upgraded = []
    for sample in samples:
        s = copy.deepcopy(sample)
        if rng.random() < fc_ratio:
            if s.get("messages") and s["messages"][0].get("role") == "system":
                s["messages"][0]["content"] = SYSTEM_PROMPT_FC
        upgraded.append(s)
    return upgraded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate FC-aware SFT data")
    parser.add_argument(
        "--existing-sft",
        type=Path,
        default=Path("/data/sft/sft_train.jsonl"),
        help="Path to existing SFT train JSONL.",
    )
    parser.add_argument(
        "--existing-val",
        type=Path,
        default=Path("/data/sft/sft_val.jsonl"),
        help="Path to existing SFT val JSONL.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/data/sft"),
        help="Output directory for merged SFT data.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fc-val-count",
        type=int,
        default=10,
        help="Number of FC samples to put in validation set.",
    )
    parser.add_argument(
        "--upgrade-ratio",
        type=float,
        default=0.3,
        help="Fraction of existing doc QA samples to upgrade with FC system prompt.",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        default=True,
        help="Backup existing SFT files before overwriting.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # 1. Generate FC-aware samples
    fc_samples = generate_all_fc_samples(seed=args.seed)
    logger.info("Generated %d FC-aware samples", len(fc_samples))

    from collections import Counter
    cat_counts = Counter(s.get("category", "unknown") for s in fc_samples)
    for cat, cnt in sorted(cat_counts.items()):
        logger.info("  %s: %d", cat, cnt)

    # 2. Load existing SFT data
    existing_train = load_existing_sft(args.existing_sft)
    existing_val = load_existing_sft(args.existing_val)
    logger.info("Loaded existing SFT: %d train, %d val", len(existing_train), len(existing_val))

    # 3. Upgrade a fraction of existing doc QA to FC system prompt
    upgraded_train = upgrade_docqa_system_prompt(
        existing_train, fc_ratio=args.upgrade_ratio, seed=args.seed
    )
    upgraded_val = upgrade_docqa_system_prompt(
        existing_val, fc_ratio=args.upgrade_ratio, seed=args.seed
    )

    # 4. Split FC samples into train/val
    rng = random.Random(args.seed)
    rng.shuffle(fc_samples)
    fc_val = fc_samples[: args.fc_val_count]
    fc_train = fc_samples[args.fc_val_count :]

    # 5. Merge
    merged_train = upgraded_train + fc_train
    merged_val = upgraded_val + fc_val
    rng.shuffle(merged_train)
    rng.shuffle(merged_val)

    logger.info(
        "Merged train: %d (%d existing + %d FC)",
        len(merged_train), len(upgraded_train), len(fc_train),
    )
    logger.info(
        "Merged val: %d (%d existing + %d FC)",
        len(merged_val), len(upgraded_val), len(fc_val),
    )

    # 6. Backup existing files
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.backup:
        train_path = output_dir / "sft_train.jsonl"
        val_path = output_dir / "sft_val.jsonl"
        if train_path.exists():
            backup_train = output_dir / "sft_train_docqa_only.jsonl"
            if not backup_train.exists():
                import shutil
                shutil.copy2(train_path, backup_train)
                logger.info("Backed up train to %s", backup_train)
        if val_path.exists():
            backup_val = output_dir / "sft_val_docqa_only.jsonl"
            if not backup_val.exists():
                import shutil
                shutil.copy2(val_path, backup_val)
                logger.info("Backed up val to %s", backup_val)

    # 7. Write merged datasets
    train_out = output_dir / "sft_train.jsonl"
    val_out = output_dir / "sft_val.jsonl"

    with train_out.open("w", encoding="utf-8") as fh:
        for item in merged_train:
            # Strip category field (not needed for training)
            out = {k: v for k, v in item.items() if k != "category"}
            fh.write(json.dumps(out, ensure_ascii=False) + "\n")

    with val_out.open("w", encoding="utf-8") as fh:
        for item in merged_val:
            out = {k: v for k, v in item.items() if k != "category"}
            fh.write(json.dumps(out, ensure_ascii=False) + "\n")

    logger.info("Written train: %s (%d samples)", train_out, len(merged_train))
    logger.info("Written val: %s (%d samples)", val_out, len(merged_val))

    # 8. Summary statistics
    fc_train_tools = sum(
        1 for s in fc_train
        if any("<tool_call>" in m.get("content", "") for m in s.get("messages", []))
    )
    fc_train_no_tools = len(fc_train) - fc_train_tools
    upgraded_count = sum(
        1 for s in upgraded_train
        if s.get("messages", [{}])[0].get("content", "") == SYSTEM_PROMPT_FC
    )

    print(f"\n{'='*60}")
    print("FC-aware SFT Data Generation Summary")
    print(f"{'='*60}")
    print(f"Total train samples: {len(merged_train)}")
    print(f"  - Existing doc QA: {len(upgraded_train)} ({upgraded_count} upgraded to FC prompt)")
    print(f"  - FC tool-calling: {fc_train_tools}")
    print(f"  - FC no-tool QA: {fc_train_no_tools}")
    print(f"Total val samples: {len(merged_val)}")
    print(f"  - Existing doc QA: {len(upgraded_val)}")
    print(f"  - FC samples: {len(fc_val)}")
    print(f"FC ratio in train: {len(fc_train)/len(merged_train)*100:.1f}%")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
