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
from typing import Any, cast

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


def _location_tracking_augment() -> list[dict[str, Any]]:
    """Add diverse location tracking samples to improve tool balance."""
    scenarios = [
        {
            "coordinates": [{"x": 305, "y": 142, "z": 265, "timestamp": 1710100001}],
            "ship_model_id": "HULL-2025-101",
            "zone": "B-4-L",
            "zone_type": "hull_side",
            "coverage": 61.2,
            "question": "지금 어디야?",
        },
        {
            "coordinates": [{"x": 535, "y": 68, "z": 322, "timestamp": 1710100120}],
            "ship_model_id": "HULL-2025-101",
            "zone": "A-3-R",
            "zone_type": "upper_deck",
            "coverage": 43.9,
            "question": "작업 구역 확인해줘.",
        },
        {
            "coordinates": [{"x": 175, "y": 218, "z": 108, "timestamp": 1710100240}],
            "ship_model_id": "HULL-2025-102",
            "zone": "C-2-B",
            "zone_type": "bottom",
            "coverage": 84.6,
            "question": "좌표 기준 구역 파악해줘.",
        },
        {
            "coordinates": [{"x": 448, "y": 112, "z": 246, "timestamp": 1710100360}],
            "ship_model_id": "HULL-2025-103",
            "zone": "VP-1",
            "zone_type": "수직 패널",
            "coverage": 57.0,
            "question": "현재 위치 기준으로 작업 구역 알려줘.",
        },
        {
            "coordinates": [{"x": 412, "y": 94, "z": 298, "timestamp": 1710100480}],
            "ship_model_id": "HULL-2025-104",
            "zone": "CR-2",
            "zone_type": "곡면",
            "coverage": 36.4,
            "question": "지금 좌표가 어느 곡면 구역인지 확인해줘.",
        },
        {
            "coordinates": [{"x": 622, "y": 58, "z": 336, "timestamp": 1710100600}],
            "ship_model_id": "HULL-2025-105",
            "zone": "ED-5",
            "zone_type": "모서리",
            "coverage": 28.8,
            "question": "작업 구역 확인 부탁해.",
        },
        {
            "coordinates": [{"x": 338, "y": 132, "z": 274, "timestamp": 1710100720}],
            "ship_model_id": "HULL-2025-102",
            "zone": "B-5-R",
            "zone_type": "hull_side",
            "coverage": 69.5,
            "question": "지금 어디 구역에서 작업 중이야?",
        },
        {
            "coordinates": [{"x": 560, "y": 72, "z": 314, "timestamp": 1710100840}],
            "ship_model_id": "HULL-2025-106",
            "zone": "A-1-L",
            "zone_type": "upper_deck",
            "coverage": 49.1,
            "question": "좌표 기준 구역 파악해줘.",
        },
        {
            "coordinates": [{"x": 208, "y": 232, "z": 118, "timestamp": 1710100960}],
            "ship_model_id": "HULL-2025-107",
            "zone": "C-4-B",
            "zone_type": "bottom",
            "coverage": 91.0,
            "question": "지금 어디야?",
        },
        {
            "coordinates": [{"x": 456, "y": 116, "z": 252, "timestamp": 1710101080}],
            "ship_model_id": "HULL-2025-108",
            "zone": "VP-3",
            "zone_type": "수직 패널",
            "coverage": 52.7,
            "question": "작업 구역 확인해줘.",
        },
        {
            "coordinates": [{"x": 394, "y": 88, "z": 286, "timestamp": 1710101200}],
            "ship_model_id": "HULL-2025-109",
            "zone": "CR-7",
            "zone_type": "곡면",
            "coverage": 40.3,
            "question": "좌표 기준 구역 파악해줘.",
        },
        {
            "coordinates": [{"x": 610, "y": 54, "z": 344, "timestamp": 1710101320}],
            "ship_model_id": "HULL-2025-110",
            "zone": "ED-2",
            "zone_type": "모서리",
            "coverage": 34.6,
            "question": "현재 위치로 작업 구역 확인해줘.",
        },
        {
            "coordinates": [{"x": 322, "y": 146, "z": 270, "timestamp": 1710101440}],
            "ship_model_id": "HULL-2025-111",
            "zone": "B-2-L",
            "zone_type": "hull_side",
            "coverage": 76.2,
            "question": "지금 어느 구역이야?",
        },
        {
            "coordinates": [{"x": 542, "y": 70, "z": 318, "timestamp": 1710101560}],
            "ship_model_id": "HULL-2025-112",
            "zone": "A-4-R",
            "zone_type": "upper_deck",
            "coverage": 47.4,
            "question": "작업 구역 확인 부탁해.",
        },
        {
            "coordinates": [{"x": 186, "y": 224, "z": 114, "timestamp": 1710101680}],
            "ship_model_id": "HULL-2025-113",
            "zone": "C-3-B",
            "zone_type": "bottom",
            "coverage": 82.9,
            "question": "좌표 기준으로 지금 작업 구역 파악해줘.",
        },
    ]

    samples = []
    for sc in scenarios:
        tc_args = {
            "coordinates": sc["coordinates"],
            "ship_model_id": sc["ship_model_id"],
        }
        tc_json = json.dumps(
            {"name": "location_tracking", "arguments": tc_args},
            ensure_ascii=False,
        )
        ctx_str = json.dumps(tc_args, ensure_ascii=False)
        coverage = cast(float, sc["coverage"])
        analysis = (
            f"현재 작업 구역은 **{sc['zone']}** ({sc['zone_type']})이며 커버리지는 {coverage}%입니다. "
            f"{'마무리 점검 단계로 전환하세요.' if coverage >= 80 else '미도장 영역을 우선 보완하세요.'}"
        )
        samples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_FC},
                {"role": "user", "content": f"[환경 데이터]\n{ctx_str}\n\n질문: {sc['question']}"},
                {
                    "role": "assistant",
                    "content": (
                        "현재 위치를 추적해 작업 구역을 확인하겠습니다.\n\n"
                        f"<tool_call>{tc_json}</tool_call>\n\n"
                        f"{analysis}\n\n"
                        "근거 문서 ID: DOC-doc_0038-C0046"
                    ),
                },
            ],
            "category": "tool_calling_location_aug",
        })

    return samples


def _pose_estimation_augment() -> list[dict[str, Any]]:
    """Add additional pose estimation samples with varied joints."""
    scenarios = [
        {
            "current": {"joints": {"wrist_r": {"x": 0.16, "y": -0.07, "z": 0.35, "roll": 12, "pitch": 26, "yaw": 14}}},
            "reference": {"joints": {"wrist_r": {"x": 0.10, "y": -0.02, "z": 0.41, "roll": 2, "pitch": 12, "yaw": 3}}},
            "deviation": "매우 높음",
            "correction": "손목 pitch/yaw를 크게 줄여 스프레이 건 축을 피도체 정면으로 맞추세요.",
            "question": "스프레이 건 각도 교정이 필요한지 봐줘.",
        },
        {
            "current": {"joints": {"wrist_r": {"x": 0.13, "y": -0.04, "z": 0.38, "roll": 6, "pitch": 17, "yaw": 7}}},
            "reference": {"joints": {"wrist_r": {"x": 0.10, "y": -0.02, "z": 0.41, "roll": 2, "pitch": 12, "yaw": 3}}},
            "deviation": "높음",
            "correction": "손목 각도를 완만하게 낮춰 분사 폭 편차를 줄이세요.",
            "question": "자세 교정 포인트 알려줘.",
        },
        {
            "current": {"joints": {"wrist_r": {"x": 0.11, "y": -0.03, "z": 0.39, "roll": 4, "pitch": 14, "yaw": 5}}},
            "reference": {"joints": {"wrist_r": {"x": 0.10, "y": -0.02, "z": 0.41, "roll": 2, "pitch": 12, "yaw": 3}}},
            "deviation": "중간",
            "correction": "거리(z) 2cm 확보와 pitch 2° 보정으로 균일도를 높이세요.",
            "question": "숙련자 비교해서 편차가 어느 정도야?",
        },
        {
            "current": {"joints": {"wrist_r": {"x": 0.10, "y": -0.02, "z": 0.40, "roll": 3, "pitch": 13, "yaw": 4}}},
            "reference": {"joints": {"wrist_r": {"x": 0.10, "y": -0.02, "z": 0.41, "roll": 2, "pitch": 12, "yaw": 3}}},
            "deviation": "낮음",
            "correction": "현재 자세를 유지하되 손목 yaw만 미세 조정하세요.",
            "question": "현재 자세가 안정적인지 확인해줘.",
        },
        {
            "current": {"joints": {"wrist_r": {"x": 0.09, "y": -0.01, "z": 0.42, "roll": 1, "pitch": 11, "yaw": 2}}},
            "reference": {"joints": {"wrist_r": {"x": 0.10, "y": -0.02, "z": 0.41, "roll": 2, "pitch": 12, "yaw": 3}}},
            "deviation": "매우 낮음",
            "correction": "숙련자 패턴과 거의 일치합니다. 현재 자세를 유지하세요.",
            "question": "숙련자랑 비교해 자세 수준이 어때?",
        },
        {
            "current": {"joints": {"elbow_r": {"x": 0.22, "y": 0.11, "z": 0.34, "roll": 18, "pitch": 30, "yaw": 11}}},
            "reference": {"joints": {"elbow_r": {"x": 0.18, "y": 0.08, "z": 0.38, "roll": 10, "pitch": 20, "yaw": 6}}},
            "deviation": "높음",
            "correction": "팔꿈치를 몸통 쪽으로 당겨 분사 궤적을 안정화하세요.",
            "question": "팔꿈치 중심으로 자세 교정해줘.",
        },
        {
            "current": {"joints": {"elbow_r": {"x": 0.20, "y": 0.10, "z": 0.36, "roll": 14, "pitch": 24, "yaw": 8}}},
            "reference": {"joints": {"elbow_r": {"x": 0.18, "y": 0.08, "z": 0.38, "roll": 10, "pitch": 20, "yaw": 6}}},
            "deviation": "중간",
            "correction": "팔꿈치 pitch를 3~4° 줄여 오버랩 편차를 줄이세요.",
            "question": "스프레이 작업에서 팔꿈치 각도 어때?",
        },
        {
            "current": {"joints": {"elbow_r": {"x": 0.18, "y": 0.08, "z": 0.38, "roll": 11, "pitch": 21, "yaw": 6}}},
            "reference": {"joints": {"elbow_r": {"x": 0.18, "y": 0.08, "z": 0.38, "roll": 10, "pitch": 20, "yaw": 6}}},
            "deviation": "낮음",
            "correction": "팔꿈치 자세는 양호합니다. 손목 궤적만 일정하게 유지하세요.",
            "question": "숙련자 비교 기준으로 자세 진단해줘.",
        },
        {
            "current": {"joints": {"shoulder_r": {"x": 0.34, "y": 0.24, "z": 0.52, "roll": 20, "pitch": 33, "yaw": 16}}},
            "reference": {"joints": {"shoulder_r": {"x": 0.30, "y": 0.20, "z": 0.56, "roll": 12, "pitch": 24, "yaw": 8}}},
            "deviation": "매우 높음",
            "correction": "어깨 elevation을 낮추고 상완 회전을 줄여 피로를 줄이세요.",
            "question": "장시간 작업용 자세 교정안 줘.",
        },
        {
            "current": {"joints": {"shoulder_r": {"x": 0.32, "y": 0.22, "z": 0.54, "roll": 16, "pitch": 28, "yaw": 12}}},
            "reference": {"joints": {"shoulder_r": {"x": 0.30, "y": 0.20, "z": 0.56, "roll": 12, "pitch": 24, "yaw": 8}}},
            "deviation": "높음",
            "correction": "어깨 roll을 줄여 분사선을 수평에 가깝게 유지하세요.",
            "question": "어깨 각도 기준으로 스프레이 건 자세 봐줘.",
        },
        {
            "current": {
                "joints": {
                    "wrist_r": {"x": 0.13, "y": -0.05, "z": 0.37, "roll": 7, "pitch": 19, "yaw": 8},
                    "elbow_r": {"x": 0.21, "y": 0.10, "z": 0.35, "roll": 16, "pitch": 27, "yaw": 9},
                }
            },
            "reference": {
                "joints": {
                    "wrist_r": {"x": 0.10, "y": -0.02, "z": 0.41, "roll": 2, "pitch": 12, "yaw": 3},
                    "elbow_r": {"x": 0.18, "y": 0.08, "z": 0.38, "roll": 10, "pitch": 20, "yaw": 6},
                }
            },
            "deviation": "높음",
            "correction": "손목-팔꿈치 연동 각도를 함께 낮춰 분사 편차를 최소화하세요.",
            "question": "다관절 기준으로 자세 교정해줘.",
        },
        {
            "current": {
                "joints": {
                    "wrist_r": {"x": 0.11, "y": -0.03, "z": 0.39, "roll": 4, "pitch": 14, "yaw": 5},
                    "shoulder_r": {"x": 0.31, "y": 0.21, "z": 0.55, "roll": 14, "pitch": 26, "yaw": 10},
                }
            },
            "reference": {
                "joints": {
                    "wrist_r": {"x": 0.10, "y": -0.02, "z": 0.41, "roll": 2, "pitch": 12, "yaw": 3},
                    "shoulder_r": {"x": 0.30, "y": 0.20, "z": 0.56, "roll": 12, "pitch": 24, "yaw": 8},
                }
            },
            "deviation": "중간",
            "correction": "손목은 미세 조정, 어깨는 2° 완화해 분사 안정성을 높이세요.",
            "question": "숙련자 비교해서 관절별 편차 분석해줘.",
        },
        {
            "current": {
                "joints": {
                    "elbow_r": {"x": 0.19, "y": 0.09, "z": 0.37, "roll": 12, "pitch": 22, "yaw": 7},
                    "shoulder_r": {"x": 0.30, "y": 0.20, "z": 0.56, "roll": 13, "pitch": 25, "yaw": 9},
                }
            },
            "reference": {
                "joints": {
                    "elbow_r": {"x": 0.18, "y": 0.08, "z": 0.38, "roll": 10, "pitch": 20, "yaw": 6},
                    "shoulder_r": {"x": 0.30, "y": 0.20, "z": 0.56, "roll": 12, "pitch": 24, "yaw": 8},
                }
            },
            "deviation": "낮음",
            "correction": "상완 라인 유지 상태가 양호합니다. 현재 패턴을 유지하세요.",
            "question": "스프레이 건 각도와 상체 자세 같이 봐줘.",
        },
        {
            "current": {"joints": {"shoulder_r": {"x": 0.29, "y": 0.19, "z": 0.57, "roll": 11, "pitch": 23, "yaw": 7}}},
            "reference": {"joints": {"shoulder_r": {"x": 0.30, "y": 0.20, "z": 0.56, "roll": 12, "pitch": 24, "yaw": 8}}},
            "deviation": "매우 낮음",
            "correction": "숙련자 자세와 거의 동일합니다.",
            "question": "자세가 숙련자 수준인지 최종 확인해줘.",
        },
        {
            "current": {
                "joints": {
                    "wrist_r": {"x": 0.14, "y": -0.06, "z": 0.36, "roll": 9, "pitch": 22, "yaw": 10},
                    "elbow_r": {"x": 0.23, "y": 0.12, "z": 0.33, "roll": 19, "pitch": 31, "yaw": 12},
                    "shoulder_r": {"x": 0.35, "y": 0.25, "z": 0.51, "roll": 21, "pitch": 34, "yaw": 17},
                }
            },
            "reference": {
                "joints": {
                    "wrist_r": {"x": 0.10, "y": -0.02, "z": 0.41, "roll": 2, "pitch": 12, "yaw": 3},
                    "elbow_r": {"x": 0.18, "y": 0.08, "z": 0.38, "roll": 10, "pitch": 20, "yaw": 6},
                    "shoulder_r": {"x": 0.30, "y": 0.20, "z": 0.56, "roll": 12, "pitch": 24, "yaw": 8},
                }
            },
            "deviation": "매우 높음",
            "correction": "전반적인 관절 각도가 과도합니다. 손목-팔꿈치-어깨를 순차 교정하세요.",
            "question": "초보자용 자세 교정안을 만들어줘.",
        },
    ]

    samples = []
    for sc in scenarios:
        tc_args = {
            "current_pose": sc["current"],
            "reference_pose": sc["reference"],
        }
        tc_json = json.dumps(
            {"name": "pose_estimation", "arguments": tc_args},
            ensure_ascii=False,
        )
        ctx_str = json.dumps(tc_args, ensure_ascii=False)
        samples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_FC},
                {"role": "user", "content": f"[환경 데이터]\n{ctx_str}\n\n질문: {sc['question']}"},
                {
                    "role": "assistant",
                    "content": (
                        "작업자 자세를 숙련자 기준과 비교 분석하겠습니다.\n\n"
                        f"<tool_call>{tc_json}</tool_call>\n\n"
                        f"자세 편차 수준은 **{sc['deviation']}**이며, 교정 방향은 {sc['correction']}\n\n"
                        "근거 문서 ID: DOC-doc_0044-C0227"
                    ),
                },
            ],
            "category": "tool_calling_pose_aug",
        })

    return samples


def _sequence_validation_augment() -> list[dict[str, Any]]:
    """Add sequence validation samples with required zone/pattern coverage."""
    scenarios = [
        {
            "trajectory": [
                {"x": 110, "y": 320, "z": 210, "timestamp": 1710200001},
                {"x": 110, "y": 260, "z": 210, "timestamp": 1710200003},
                {"x": 110, "y": 200, "z": 210, "timestamp": 1710200005},
            ],
            "zone_id": "VP-1",
            "zone_type": "수직 패널",
            "reference_pattern": "bottom_to_top",
            "valid": False,
            "detected": "top_to_bottom",
            "question": "수직 패널 작업 방향이 맞는지 검증해줘.",
        },
        {
            "trajectory": [
                {"x": 120, "y": 180, "z": 220, "timestamp": 1710200101},
                {"x": 120, "y": 230, "z": 220, "timestamp": 1710200103},
                {"x": 120, "y": 280, "z": 220, "timestamp": 1710200105},
            ],
            "zone_id": "VP-2",
            "zone_type": "수직 패널",
            "reference_pattern": "bottom_to_top",
            "valid": True,
            "detected": "bottom_to_top",
            "question": "순서 위반 여부 확인해줘.",
        },
        {
            "trajectory": [
                {"x": 60, "y": 95, "z": 260, "timestamp": 1710200201},
                {"x": 120, "y": 95, "z": 260, "timestamp": 1710200203},
                {"x": 180, "y": 95, "z": 260, "timestamp": 1710200205},
            ],
            "zone_id": "HP-1",
            "zone_type": "수평 패널",
            "reference_pattern": "left_to_right",
            "valid": True,
            "detected": "left_to_right",
            "question": "stroke 패턴이 기준과 같은지 봐줘.",
        },
        {
            "trajectory": [
                {"x": 190, "y": 105, "z": 255, "timestamp": 1710200301},
                {"x": 135, "y": 105, "z": 255, "timestamp": 1710200303},
                {"x": 80, "y": 105, "z": 255, "timestamp": 1710200305},
            ],
            "zone_id": "HP-2",
            "zone_type": "수평 패널",
            "reference_pattern": "left_to_right",
            "valid": False,
            "detected": "right_to_left",
            "question": "작업 방향 위반인지 알려줘.",
        },
        {
            "trajectory": [
                {"x": 240, "y": 120, "z": 290, "timestamp": 1710200401},
                {"x": 200, "y": 120, "z": 280, "timestamp": 1710200403},
                {"x": 160, "y": 120, "z": 270, "timestamp": 1710200405},
            ],
            "zone_id": "CV-1",
            "zone_type": "곡면",
            "reference_pattern": "right_to_left",
            "valid": True,
            "detected": "right_to_left",
            "question": "곡면 구역 순서 검증해줘.",
        },
        {
            "trajectory": [
                {"x": 150, "y": 130, "z": 265, "timestamp": 1710200501},
                {"x": 195, "y": 130, "z": 275, "timestamp": 1710200503},
                {"x": 240, "y": 130, "z": 285, "timestamp": 1710200505},
            ],
            "zone_id": "CV-2",
            "zone_type": "곡면",
            "reference_pattern": "right_to_left",
            "valid": False,
            "detected": "left_to_right",
            "question": "곡면 작업 stroke 패턴이 맞아?",
        },
        {
            "trajectory": [
                {"x": 300, "y": 340, "z": 230, "timestamp": 1710200601},
                {"x": 300, "y": 280, "z": 230, "timestamp": 1710200603},
                {"x": 300, "y": 220, "z": 230, "timestamp": 1710200605},
            ],
            "zone_id": "ED-1",
            "zone_type": "모서리",
            "reference_pattern": "top_to_bottom",
            "valid": True,
            "detected": "top_to_bottom",
            "question": "모서리 구역 도장 순서가 올바른지 확인해줘.",
        },
        {
            "trajectory": [
                {"x": 300, "y": 210, "z": 230, "timestamp": 1710200701},
                {"x": 300, "y": 270, "z": 230, "timestamp": 1710200703},
                {"x": 300, "y": 330, "z": 230, "timestamp": 1710200705},
            ],
            "zone_id": "ED-2",
            "zone_type": "모서리",
            "reference_pattern": "top_to_bottom",
            "valid": False,
            "detected": "bottom_to_top",
            "question": "순서 위반 체크해줘.",
        },
        {
            "trajectory": [
                {"x": 70, "y": 110, "z": 250, "timestamp": 1710200801},
                {"x": 125, "y": 110, "z": 250, "timestamp": 1710200803},
                {"x": 180, "y": 110, "z": 250, "timestamp": 1710200805},
            ],
            "zone_id": "HP-3",
            "zone_type": "수평 패널",
            "reference_pattern": "left_to_right",
            "valid": True,
            "detected": "left_to_right",
            "question": "작업 방향 검증해줘.",
        },
        {
            "trajectory": [
                {"x": 180, "y": 112, "z": 250, "timestamp": 1710200901},
                {"x": 125, "y": 112, "z": 250, "timestamp": 1710200903},
                {"x": 70, "y": 112, "z": 250, "timestamp": 1710200905},
            ],
            "zone_id": "HP-4",
            "zone_type": "수평 패널",
            "reference_pattern": "right_to_left",
            "valid": True,
            "detected": "right_to_left",
            "question": "수평 패널 stroke 패턴 확인해줘.",
        },
        {
            "trajectory": [
                {"x": 126, "y": 286, "z": 214, "timestamp": 1710201001},
                {"x": 126, "y": 236, "z": 214, "timestamp": 1710201003},
                {"x": 126, "y": 186, "z": 214, "timestamp": 1710201005},
            ],
            "zone_id": "VP-4",
            "zone_type": "수직 패널",
            "reference_pattern": "top_to_bottom",
            "valid": True,
            "detected": "top_to_bottom",
            "question": "수직 패널 작업 방향 적합한지 봐줘.",
        },
        {
            "trajectory": [
                {"x": 126, "y": 186, "z": 214, "timestamp": 1710201101},
                {"x": 126, "y": 236, "z": 214, "timestamp": 1710201103},
                {"x": 126, "y": 286, "z": 214, "timestamp": 1710201105},
            ],
            "zone_id": "VP-5",
            "zone_type": "수직 패널",
            "reference_pattern": "top_to_bottom",
            "valid": False,
            "detected": "bottom_to_top",
            "question": "순서 위반 여부 확인해줘.",
        },
        {
            "trajectory": [
                {"x": 235, "y": 124, "z": 294, "timestamp": 1710201201},
                {"x": 195, "y": 124, "z": 284, "timestamp": 1710201203},
                {"x": 155, "y": 124, "z": 274, "timestamp": 1710201205},
            ],
            "zone_id": "CV-3",
            "zone_type": "곡면",
            "reference_pattern": "right_to_left",
            "valid": True,
            "detected": "right_to_left",
            "question": "곡면 구역 작업 방향 분석해줘.",
        },
        {
            "trajectory": [
                {"x": 155, "y": 124, "z": 274, "timestamp": 1710201301},
                {"x": 195, "y": 124, "z": 284, "timestamp": 1710201303},
                {"x": 235, "y": 124, "z": 294, "timestamp": 1710201305},
            ],
            "zone_id": "CV-4",
            "zone_type": "곡면",
            "reference_pattern": "right_to_left",
            "valid": False,
            "detected": "left_to_right",
            "question": "stroke 방향 위반인지 확인해줘.",
        },
        {
            "trajectory": [
                {"x": 292, "y": 332, "z": 236, "timestamp": 1710201401},
                {"x": 292, "y": 272, "z": 236, "timestamp": 1710201403},
                {"x": 292, "y": 212, "z": 236, "timestamp": 1710201405},
            ],
            "zone_id": "ED-3",
            "zone_type": "모서리",
            "reference_pattern": "top_to_bottom",
            "valid": True,
            "detected": "top_to_bottom",
            "question": "모서리 작업 순서 검증해줘.",
        },
    ]

    samples = []
    for sc in scenarios:
        tc_args = {
            "trajectory": sc["trajectory"],
            "zone_id": sc["zone_id"],
            "zone_type": sc["zone_type"],
            "reference_pattern": sc["reference_pattern"],
        }
        tc_json = json.dumps(
            {"name": "sequence_validation", "arguments": tc_args},
            ensure_ascii=False,
        )
        ctx_str = json.dumps(tc_args, ensure_ascii=False)
        if sc["valid"]:
            analysis = (
                f"검증 결과 **적합**입니다. 감지 패턴({sc['detected']})이 참조 패턴({sc['reference_pattern']})과 일치합니다."
            )
        else:
            analysis = (
                f"검증 결과 **부적합**입니다. 감지 패턴({sc['detected']})이 참조 패턴({sc['reference_pattern']})과 다르므로 순서를 교정하세요."
            )
        samples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_FC},
                {"role": "user", "content": f"[환경 데이터]\n{ctx_str}\n\n질문: {sc['question']}"},
                {
                    "role": "assistant",
                    "content": (
                        "도장 순서와 작업 방향을 검증하겠습니다.\n\n"
                        f"<tool_call>{tc_json}</tool_call>\n\n"
                        f"{analysis}\n\n"
                        "근거 문서 ID: DOC-doc_0044-C0227"
                    ),
                },
            ],
            "category": "tool_calling_sequence_aug",
        })

    return samples


def _timeseries_augment() -> list[dict[str, Any]]:
    """Add timeseries query/update samples including environment context."""
    scenarios = [
        {
            "action": "update",
            "session_id": "sess-ts-aug-001",
            "data": {"timestamp": 1710300001, "speed": 0.82, "distance": 295, "angle": 49},
            "env": {"temperature": 20, "humidity": 63},
            "question": "속도 변화 추세 확인을 위해 현재 값을 업데이트해줘.",
        },
        {
            "action": "query",
            "session_id": "sess-ts-aug-002",
            "data": None,
            "env": None,
            "question": "최근 작업 추세 확인해줘.",
        },
        {
            "action": "update",
            "session_id": "sess-ts-aug-003",
            "data": {"timestamp": 1710300121, "speed": 1.18, "distance": 338, "angle": 41},
            "env": {"temperature": 24, "humidity": 58},
            "question": "속도/거리/각도 데이터 기록해줘.",
        },
        {
            "action": "query",
            "session_id": "sess-ts-aug-004",
            "data": None,
            "env": {"temperature": 26, "humidity": 71},
            "question": "환경 데이터 포함해서 시계열 추세 확인해줘.",
        },
        {
            "action": "update",
            "session_id": "sess-ts-aug-005",
            "data": {"timestamp": 1710300241, "speed": 0.91, "distance": 312, "angle": 53},
            "env": None,
            "question": "작업 속도 변화 분석용 업데이트 해줘.",
        },
        {
            "action": "query",
            "session_id": "sess-ts-aug-006",
            "data": None,
            "env": None,
            "question": "속도 추세와 편차 조회해줘.",
        },
        {
            "action": "update",
            "session_id": "sess-ts-aug-007",
            "data": {"timestamp": 1710300361, "speed": 1.05, "distance": 320, "angle": 47},
            "env": {"temperature": 22, "humidity": 66},
            "question": "환경 조건과 함께 현재 시점 데이터 반영해줘.",
        },
        {
            "action": "query",
            "session_id": "sess-ts-aug-008",
            "data": None,
            "env": {"temperature": 28, "humidity": 74},
            "question": "습도 상승 구간에서 속도 변화 추세 확인해줘.",
        },
        {
            "action": "update",
            "session_id": "sess-ts-aug-009",
            "data": {"timestamp": 1710300481, "speed": 0.88, "distance": 302, "angle": 56},
            "env": None,
            "question": "각도 변화가 큰 구간 데이터 업데이트해줘.",
        },
        {
            "action": "query",
            "session_id": "sess-ts-aug-010",
            "data": None,
            "env": None,
            "question": "현재 세션 추세 확인 부탁해.",
        },
        {
            "action": "update",
            "session_id": "sess-ts-aug-011",
            "data": {"timestamp": 1710300601, "speed": 1.22, "distance": 346, "angle": 39},
            "env": {"temperature": 19, "humidity": 52},
            "question": "고속 구간 데이터 시계열에 추가해줘.",
        },
        {
            "action": "query",
            "session_id": "sess-ts-aug-012",
            "data": None,
            "env": {"temperature": 17, "humidity": 80},
            "question": "저온/고습 조건에서 추세 확인해줘.",
        },
        {
            "action": "update",
            "session_id": "sess-ts-aug-013",
            "data": {"timestamp": 1710300721, "speed": 0.97, "distance": 315, "angle": 45},
            "env": None,
            "question": "중립 조건 데이터 업데이트해줘.",
        },
        {
            "action": "query",
            "session_id": "sess-ts-aug-014",
            "data": None,
            "env": None,
            "question": "속도 변화 추세를 다시 조회해줘.",
        },
        {
            "action": "update",
            "session_id": "sess-ts-aug-015",
            "data": {"timestamp": 1710300841, "speed": 1.09, "distance": 327, "angle": 43},
            "env": {"temperature": 23, "humidity": 61},
            "question": "환경 데이터 포함해서 업데이트해줘.",
        },
    ]

    samples = []
    for sc in scenarios:
        if sc["action"] == "update":
            update_data = cast(dict[str, Any], sc["data"])
            tc_args: dict[str, Any] = {
                "action": "update",
                "session_id": sc["session_id"],
                "data": update_data,
            }
            analysis = (
                f"세션 {sc['session_id']}에 속도 {update_data['speed']}m/s, "
                f"거리 {update_data['distance']}mm, 각도 {update_data['angle']}° 데이터가 반영되었습니다."
            )
        else:
            tc_args = {"action": "query", "session_id": sc["session_id"]}
            analysis = f"세션 {sc['session_id']}의 속도/거리/각도 추세를 조회해 편차 변화를 확인했습니다."

        context_data: dict[str, Any] = {"timeseries": tc_args}
        if sc["env"] is not None:
            context_data["env"] = sc["env"]

        tc_json = json.dumps({"name": "timeseries", "arguments": tc_args}, ensure_ascii=False)
        ctx_str = json.dumps(context_data, ensure_ascii=False)
        samples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_FC},
                {"role": "user", "content": f"[환경 데이터]\n{ctx_str}\n\n질문: {sc['question']}"},
                {
                    "role": "assistant",
                    "content": (
                        "시계열 데이터를 분석하겠습니다.\n\n"
                        f"<tool_call>{tc_json}</tool_call>\n\n"
                        f"{analysis}\n\n"
                        "근거 문서 ID: DOC-doc_0018-C0037"
                    ),
                },
            ],
            "category": "tool_calling_timeseries_aug",
        })

    return samples


def _xr_content_samples() -> list[dict[str, Any]]:
    """Generate XR content creation samples requiring location+pose tools."""
    scenarios = [
        ("XR 가이드 만들어줘.", "A-2-R", "upper_deck", "pitch를 4° 낮춰 정확도를 확보"),
        ("초보자용 튜토리얼 만들어줘.", "B-3-L", "hull_side", "손목 yaw를 3° 줄여 분사선을 안정화"),
        ("XR 코칭 콘텐츠 생성해줘.", "C-1-B", "bottom", "피도체 거리 2cm 추가 확보"),
        ("XR 마이크로레슨으로 정리해줘.", "VP-2", "수직 패널", "팔꿈치 각도를 완만하게 유지"),
        ("작업 구역 기반 XR 가이드 부탁해.", "CR-4", "곡면", "어깨 roll을 2° 줄여 곡면 추종"),
        ("신입용 XR 코칭 콘텐츠 만들어줘.", "ED-1", "모서리", "모서리 접근 시 손목 pitch를 낮춤"),
        ("초보자 교육용 XR 튜토리얼 구성해줘.", "A-4-L", "upper_deck", "스프레이 건 yaw 흔들림 최소화"),
        ("실습형 XR 가이드 설계해줘.", "B-1-R", "hull_side", "분사 거리 40~42cm 유지"),
        ("현장 코칭용 XR 마이크로레슨 만들어줘.", "VP-4", "수직 패널", "bottom_to_top 구간에서 손목 고정"),
        ("XR 튜토리얼로 자세랑 위치 같이 안내해줘.", "CR-7", "곡면", "곡면 전환부에서 어깨 긴장 완화"),
    ]

    samples = []
    for idx, (question, zone, zone_type, correction) in enumerate(scenarios, start=1):
        coords = [{"x": 300 + idx * 22, "y": 70 + idx * 6, "z": 250 + idx * 5, "timestamp": 1710400000 + idx * 60}]
        current_pose = {
            "joints": {
                "wrist_r": {"x": 0.12 + idx * 0.001, "y": -0.04, "z": 0.38, "roll": 5 + idx % 3, "pitch": 16 + idx % 4, "yaw": 6 + idx % 3}
            }
        }
        reference_pose = {
            "joints": {
                "wrist_r": {"x": 0.10, "y": -0.02, "z": 0.41, "roll": 2, "pitch": 12, "yaw": 3}
            }
        }
        context_data = {
            "coordinates": coords,
            "ship_model_id": f"HULL-XR-20{idx:02d}",
            "current_pose": current_pose,
            "reference_pose": reference_pose,
        }

        tc_loc = json.dumps(
            {
                "name": "location_tracking",
                "arguments": {
                    "coordinates": coords,
                    "ship_model_id": f"HULL-XR-20{idx:02d}",
                },
            },
            ensure_ascii=False,
        )
        tc_pose = json.dumps(
            {
                "name": "pose_estimation",
                "arguments": {
                    "current_pose": current_pose,
                    "reference_pose": reference_pose,
                },
            },
            ensure_ascii=False,
        )
        xr_output = (
            "XR 콘텐츠 출력:\n"
            "- steps:\n"
            f"  1) {zone} 구역({zone_type}) 진입 후 안전거리 확인\n"
            "  2) 스프레이 건을 피도체와 평행하게 유지\n"
            f"  3) 자세 교정 포인트 적용: {correction}\n"
            "  4) 스트로크 오버랩 50% 유지하며 반복\n"
            f"- highlight_zone: {zone}\n"
            "- safety_notes:\n"
            "  - 보호구 착용(방독 마스크, 보안경, 장갑)\n"
            "  - 밀폐구역 환기 및 가연성 증기 모니터링"
        )

        samples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_FC},
                {"role": "user", "content": f"[환경 데이터]\n{json.dumps(context_data, ensure_ascii=False)}\n\n질문: {question}"},
                {
                    "role": "assistant",
                    "content": (
                        "XR 가이드 생성을 위해 위치와 자세를 함께 분석하겠습니다.\n\n"
                        f"<tool_call>{tc_loc}</tool_call>\n\n"
                        f"<tool_call>{tc_pose}</tool_call>\n\n"
                        f"{xr_output}\n\n"
                        "근거 문서 ID: DOC-doc_0038-C0046, DOC-doc_0044-C0227"
                    ),
                },
            ],
            "category": "tool_calling_xr",
        })

    return samples


def _multi_tool_augment() -> list[dict[str, Any]]:
    """Add eval-focused multi-tool combinations for underrepresented patterns."""
    samples = []

    for idx in range(1, 6):
        ts_args: dict[str, Any] = {
            "action": "update" if idx % 2 else "query",
            "session_id": f"sess-mta-tsseq-{idx:02d}",
        }
        if ts_args["action"] == "update":
            ts_args["data"] = {
                "timestamp": 1710500000 + idx * 60,
                "speed": 0.84 + idx * 0.06,
                "distance": 298 + idx * 8,
                "angle": 42 + idx * 2,
            }
        seq_args = {
            "trajectory": [
                {"x": 100 + idx, "y": 170, "z": 210, "timestamp": 1710500000 + idx * 60},
                {"x": 140 + idx, "y": 170, "z": 210, "timestamp": 1710500002 + idx * 60},
                {"x": 180 + idx, "y": 170, "z": 210, "timestamp": 1710500004 + idx * 60},
            ],
            "zone_id": f"HP-{idx}",
            "zone_type": "수평 패널",
            "reference_pattern": "left_to_right",
        }
        context_data = {
            "timeseries": ts_args,
            "sequence_validation": seq_args,
            "env": {"temperature": 20 + idx, "humidity": 60 + idx * 2},
        }
        tc_ts = json.dumps({"name": "timeseries", "arguments": ts_args}, ensure_ascii=False)
        tc_seq = json.dumps({"name": "sequence_validation", "arguments": seq_args}, ensure_ascii=False)
        samples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_FC},
                {"role": "user", "content": f"[환경 데이터]\n{json.dumps(context_data, ensure_ascii=False)}\n\n질문: 환경 조건과 현재 순서를 함께 검증해줘."},
                {
                    "role": "assistant",
                    "content": (
                        "환경 조건을 고려해 시계열 추세와 작업 순서를 함께 분석하겠습니다.\n\n"
                        f"<tool_call>{tc_ts}</tool_call>\n\n"
                        f"<tool_call>{tc_seq}</tool_call>\n\n"
                        "시계열 변화와 stroke 패턴을 종합해 작업 안정성과 순서 적합성을 판단했습니다.\n\n"
                        "근거 문서 ID: DOC-doc_0018-C0037, DOC-doc_0044-C0227"
                    ),
                },
            ],
            "category": "tool_calling_multi_aug",
        })

    for idx in range(1, 4):
        meas = [82 + idx * 2, 88 + idx, 93 + idx, 86 + idx * 2, 90 + idx]
        loc_args = {
            "coordinates": [{"x": 320 + idx * 20, "y": 140 - idx * 3, "z": 270 + idx * 4, "timestamp": 1710501000 + idx * 60}],
            "ship_model_id": f"HULL-MTA-LAD-{idx:02d}",
        }
        anm_args = {
            "image_context": f"B-{idx}-L 구역 국부 저도막 패턴",
            "dft_measurements": meas,
            "env": {"temperature": 22 + idx, "humidity": 64 + idx},
        }
        dft_args = {"measurements": meas, "spec": {"nominal": 100, "tolerance_rule": "90/10"}}
        context_data = {"location": loc_args, "anomaly": anm_args, "dft": dft_args}
        tc_loc = json.dumps({"name": "location_tracking", "arguments": loc_args}, ensure_ascii=False)
        tc_anm = json.dumps({"name": "anomaly_detection", "arguments": anm_args}, ensure_ascii=False)
        tc_dft = json.dumps({"name": "dft_judge", "arguments": dft_args}, ensure_ascii=False)
        samples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_FC},
                {"role": "user", "content": f"[환경 데이터]\n{json.dumps(context_data, ensure_ascii=False)}\n\n질문: 위치별 결함 분석하고 최종 합부도 판단해줘."},
                {
                    "role": "assistant",
                    "content": (
                        "위치 식별 후 이상 탐지와 DFT 판정을 순차적으로 수행하겠습니다.\n\n"
                        f"<tool_call>{tc_loc}</tool_call>\n\n"
                        f"<tool_call>{tc_anm}</tool_call>\n\n"
                        f"<tool_call>{tc_dft}</tool_call>\n\n"
                        "구역 위치 기반 결함 패턴과 DFT 판정 결과를 결합해 재도장 우선순위를 제시했습니다.\n\n"
                        "근거 문서 ID: DOC-doc_0038-C0046, DOC-doc_0037-C0308, DOC-doc_0033-C0326"
                    ),
                },
            ],
            "category": "tool_calling_multi_aug",
        })

    for idx in range(1, 4):
        meas = [89 + idx, 111 - idx, 86 + idx, 108 - idx, 92 + idx]
        ts_args = {
            "action": "update",
            "session_id": f"sess-mta-tad-{idx:02d}",
            "data": {"timestamp": 1710502000 + idx * 60, "speed": 1.02 + idx * 0.05, "distance": 314 + idx * 7, "angle": 46 + idx * 3},
        }
        anm_args = {
            "image_context": f"중앙부 교차 패턴 idx-{idx}",
            "dft_measurements": meas,
            "env": {"temperature": 23 + idx, "humidity": 62 + idx * 2},
        }
        dft_args = {"measurements": meas, "spec": {"nominal": 100, "tolerance_rule": "90/10"}}
        context_data = {"timeseries": ts_args, "anomaly": anm_args, "dft": dft_args}
        tc_ts = json.dumps({"name": "timeseries", "arguments": ts_args}, ensure_ascii=False)
        tc_anm = json.dumps({"name": "anomaly_detection", "arguments": anm_args}, ensure_ascii=False)
        tc_dft = json.dumps({"name": "dft_judge", "arguments": dft_args}, ensure_ascii=False)
        samples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_FC},
                {"role": "user", "content": f"[환경 데이터]\n{json.dumps(context_data, ensure_ascii=False)}\n\n질문: 추세+이상+판정을 종합해 원인 체인 분석해줘."},
                {
                    "role": "assistant",
                    "content": (
                        "시계열 추세와 히트맵 이상, DFT 합부를 종합 분석하겠습니다.\n\n"
                        f"<tool_call>{tc_ts}</tool_call>\n\n"
                        f"<tool_call>{tc_anm}</tool_call>\n\n"
                        f"<tool_call>{tc_dft}</tool_call>\n\n"
                        "속도/각도 변동 추세와 이상 패턴, 판정 결과를 연결해 공정 원인 체인을 도출했습니다.\n\n"
                        "근거 문서 ID: DOC-doc_0018-C0037, DOC-doc_0037-C0308, DOC-doc_0033-C0326"
                    ),
                },
            ],
            "category": "tool_calling_multi_aug",
        })

    for idx in range(1, 3):
        loc_args = {
            "coordinates": [{"x": 350 + idx * 18, "y": 118, "z": 260 + idx * 8, "timestamp": 1710503000 + idx * 60}],
            "ship_model_id": f"HULL-MTA-LST-{idx:02d}",
        }
        seq_args = {
            "trajectory": [
                {"x": 110, "y": 290, "z": 220, "timestamp": 1710503000 + idx * 60},
                {"x": 110, "y": 240, "z": 220, "timestamp": 1710503002 + idx * 60},
                {"x": 110, "y": 190, "z": 220, "timestamp": 1710503004 + idx * 60},
            ],
            "zone_id": f"VP-{idx+5}",
            "zone_type": "수직 패널",
            "reference_pattern": "bottom_to_top",
        }
        ts_args = {
            "action": "query",
            "session_id": f"sess-mta-lst-{idx:02d}",
        }
        context_data = {"location": loc_args, "sequence": seq_args, "timeseries": ts_args}
        tc_loc = json.dumps({"name": "location_tracking", "arguments": loc_args}, ensure_ascii=False)
        tc_seq = json.dumps({"name": "sequence_validation", "arguments": seq_args}, ensure_ascii=False)
        tc_ts = json.dumps({"name": "timeseries", "arguments": ts_args}, ensure_ascii=False)
        samples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_FC},
                {"role": "user", "content": f"[환경 데이터]\n{json.dumps(context_data, ensure_ascii=False)}\n\n질문: 구역 우선순위를 위치/순서/추세 기반으로 정해줘."},
                {
                    "role": "assistant",
                    "content": (
                        "구역 우선순위 결정을 위해 위치, 순서, 시계열을 함께 분석하겠습니다.\n\n"
                        f"<tool_call>{tc_loc}</tool_call>\n\n"
                        f"<tool_call>{tc_seq}</tool_call>\n\n"
                        f"<tool_call>{tc_ts}</tool_call>\n\n"
                        "위치 기반 난이도와 순서 적합성, 추세 안정성을 결합해 우선 작업 구역을 도출했습니다.\n\n"
                        "근거 문서 ID: DOC-doc_0038-C0046, DOC-doc_0044-C0227, DOC-doc_0018-C0037"
                    ),
                },
            ],
            "category": "tool_calling_multi_aug",
        })

    for idx in range(1, 3):
        seq_args = {
            "trajectory": [
                {"x": 86, "y": 120 + idx * 2, "z": 248, "timestamp": 1710504000 + idx * 60},
                {"x": 136, "y": 120 + idx * 2, "z": 248, "timestamp": 1710504002 + idx * 60},
                {"x": 186, "y": 120 + idx * 2, "z": 248, "timestamp": 1710504004 + idx * 60},
            ],
            "zone_id": f"HP-{idx+6}",
            "zone_type": "수평 패널",
            "reference_pattern": "left_to_right",
        }
        ts_args = {
            "action": "update",
            "session_id": f"sess-mta-sta-{idx:02d}",
            "data": {"timestamp": 1710504000 + idx * 60, "speed": 0.92 + idx * 0.07, "distance": 309 + idx * 6, "angle": 50 + idx * 2},
        }
        anm_args = {
            "image_context": f"수평 패널 스트릭 패턴 idx-{idx}",
            "dft_measurements": [90 + idx, 97 + idx, 85 + idx, 104 + idx, 93 + idx],
            "env": {"temperature": 21 + idx, "humidity": 67 + idx},
        }
        context_data = {"sequence": seq_args, "timeseries": ts_args, "anomaly": anm_args}
        tc_seq = json.dumps({"name": "sequence_validation", "arguments": seq_args}, ensure_ascii=False)
        tc_ts = json.dumps({"name": "timeseries", "arguments": ts_args}, ensure_ascii=False)
        tc_anm = json.dumps({"name": "anomaly_detection", "arguments": anm_args}, ensure_ascii=False)
        samples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_FC},
                {"role": "user", "content": f"[환경 데이터]\n{json.dumps(context_data, ensure_ascii=False)}\n\n질문: 순서+추세+이상 상태를 종합 진단해줘."},
                {
                    "role": "assistant",
                    "content": (
                        "순서 검증, 시계열 추세, 이상 탐지를 통합 분석하겠습니다.\n\n"
                        f"<tool_call>{tc_seq}</tool_call>\n\n"
                        f"<tool_call>{tc_ts}</tool_call>\n\n"
                        f"<tool_call>{tc_anm}</tool_call>\n\n"
                        "작업 순서 적합성과 속도 추세, 결함 신호를 결합해 종합 리스크를 판단했습니다.\n\n"
                        "근거 문서 ID: DOC-doc_0044-C0227, DOC-doc_0018-C0037, DOC-doc_0037-C0308"
                    ),
                },
            ],
            "category": "tool_calling_multi_aug",
        })

    return samples


def _multi_turn_augment() -> list[dict[str, Any]]:
    """Add multi-turn samples focused on underrepresented tools."""
    samples = []

    for idx in range(1, 3):
        loc_args = {
            "coordinates": [{"x": 330 + idx * 15, "y": 145 - idx * 4, "z": 272 + idx * 5, "timestamp": 1710600000 + idx * 60}],
            "ship_model_id": f"HULL-MTLOC-{idx:02d}",
        }
        tool_result = json.dumps({
            "status": "success",
            "result": {
                "zone_id": f"B-{idx+2}-L",
                "zone_type": "hull_side",
                "coverage_pct": 62.0 + idx * 9.5,
                "uncovered_areas": ["상단 영역", "좌측 경계"],
            },
        }, ensure_ascii=False)
        tc_json = json.dumps({"name": "location_tracking", "arguments": loc_args}, ensure_ascii=False)
        samples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_FC},
                {"role": "user", "content": f"[환경 데이터]\n{json.dumps(loc_args, ensure_ascii=False)}\n\n질문: 지금 작업 구역 확인해줘."},
                {"role": "assistant", "content": f"현재 작업 위치를 확인하기 위해 도구를 호출합니다.\n\n<tool_call>{tc_json}</tool_call>"},
                {"role": "tool", "content": tool_result},
                {
                    "role": "assistant",
                    "content": (
                        "도구 분석 결과를 바탕으로 답변합니다.\n\n"
                        f"현재 위치는 B-{idx+2}-L(hull_side)이며 미도장 구역은 상단/좌측 경계입니다. "
                        "미도장 영역부터 우선 보완하세요.\n\n"
                        "근거 문서 ID: DOC-doc_0038-C0046"
                    ),
                },
            ],
            "category": "multi_turn_location_aug",
        })

    for idx in range(1, 3):
        seq_args = {
            "trajectory": [
                {"x": 112, "y": 304 - idx * 40, "z": 218, "timestamp": 1710601000 + idx * 60},
                {"x": 112, "y": 254 - idx * 40, "z": 218, "timestamp": 1710601002 + idx * 60},
                {"x": 112, "y": 204 - idx * 40, "z": 218, "timestamp": 1710601004 + idx * 60},
            ],
            "zone_id": f"VP-{idx+6}",
            "zone_type": "수직 패널",
            "reference_pattern": "bottom_to_top",
        }
        tool_result = json.dumps({
            "status": "success",
            "result": {
                "valid": False if idx == 1 else True,
                "detected_pattern": "top_to_bottom" if idx == 1 else "bottom_to_top",
                "reference_pattern": "bottom_to_top",
                "correction": "reverse_direction" if idx == 1 else "none",
            },
        }, ensure_ascii=False)
        tc_json = json.dumps({"name": "sequence_validation", "arguments": seq_args}, ensure_ascii=False)
        samples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_FC},
                {"role": "user", "content": f"[환경 데이터]\n{json.dumps(seq_args, ensure_ascii=False)}\n\n질문: 순서 위반 여부 확인해줘."},
                {"role": "assistant", "content": f"도장 순서 검증을 위해 도구를 호출합니다.\n\n<tool_call>{tc_json}</tool_call>"},
                {"role": "tool", "content": tool_result},
                {
                    "role": "assistant",
                    "content": (
                        "도구 분석 결과를 바탕으로 답변합니다.\n\n"
                        f"검증 결과는 {'부적합' if idx == 1 else '적합'}이며, "
                        f"감지 패턴은 {'top_to_bottom' if idx == 1 else 'bottom_to_top'}입니다.\n\n"
                        "근거 문서 ID: DOC-doc_0044-C0227"
                    ),
                },
            ],
            "category": "multi_turn_sequence_aug",
        })

    for idx in range(1, 3):
        ts_args: dict[str, Any] = {
            "action": "query" if idx == 1 else "update",
            "session_id": f"sess-mtts-{idx:02d}",
        }
        if idx == 2:
            ts_args["data"] = {"timestamp": 1710602000 + idx * 60, "speed": 1.08, "distance": 324, "angle": 46}

        tool_result = json.dumps({
            "status": "success",
            "result": {
                "session_id": f"sess-mtts-{idx:02d}",
                "trend": "stable" if idx == 1 else "updated",
                "speed": {"mean": 1.01 + idx * 0.03, "std": 0.11},
                "distance": {"mean": 312 + idx * 6, "std": 15},
                "angle": {"mean": 47 + idx, "std": 4.8},
            },
        }, ensure_ascii=False)
        tc_json = json.dumps({"name": "timeseries", "arguments": ts_args}, ensure_ascii=False)
        samples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_FC},
                {"role": "user", "content": f"[환경 데이터]\n{json.dumps(ts_args, ensure_ascii=False)}\n\n질문: 속도 변화와 추세를 확인해줘."},
                {"role": "assistant", "content": f"시계열 분석을 위해 도구를 호출합니다.\n\n<tool_call>{tc_json}</tool_call>"},
                {"role": "tool", "content": tool_result},
                {
                    "role": "assistant",
                    "content": (
                        "도구 분석 결과를 바탕으로 답변합니다.\n\n"
                        "세션 추세를 확인했으며 속도/거리/각도 편차를 기준 범위에서 관리 중입니다.\n\n"
                        "근거 문서 ID: DOC-doc_0018-C0037"
                    ),
                },
            ],
            "category": "multi_turn_timeseries_aug",
        })

    for idx in range(1, 3):
        pose_args = {
            "current_pose": {
                "joints": {
                    "wrist_r": {"x": 0.12 + idx * 0.01, "y": -0.05, "z": 0.37, "roll": 5 + idx, "pitch": 17 + idx * 2, "yaw": 7 + idx}
                }
            },
            "reference_pose": {
                "joints": {
                    "wrist_r": {"x": 0.10, "y": -0.02, "z": 0.41, "roll": 2, "pitch": 12, "yaw": 3}
                }
            },
        }
        tool_result = json.dumps({
            "status": "success",
            "result": {
                "overall_deviation": "high" if idx == 1 else "medium",
                "joint_deviations": {
                    "wrist_r": {"pitch_error": 5 + idx, "yaw_error": 3 + idx, "z_distance_error": -0.03},
                },
            },
        }, ensure_ascii=False)
        tc_json = json.dumps({"name": "pose_estimation", "arguments": pose_args}, ensure_ascii=False)
        samples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_FC},
                {"role": "user", "content": f"[환경 데이터]\n{json.dumps(pose_args, ensure_ascii=False)}\n\n질문: 자세 교정 포인트 알려줘."},
                {"role": "assistant", "content": f"자세 분석을 위해 도구를 호출합니다.\n\n<tool_call>{tc_json}</tool_call>"},
                {"role": "tool", "content": tool_result},
                {
                    "role": "assistant",
                    "content": (
                        "도구 분석 결과를 바탕으로 답변합니다.\n\n"
                        "손목 pitch/yaw 편차가 확인되어 스프레이 건 각도 교정이 필요합니다.\n\n"
                        "근거 문서 ID: DOC-doc_0044-C0227"
                    ),
                },
            ],
            "category": "multi_turn_pose_aug",
        })

    for idx in range(1, 3):
        meas = [84 + idx, 87 + idx, 92 + idx, 88 + idx, 86 + idx]
        combined_calls = [
            {
                "name": "anomaly_detection",
                "arguments": {
                    "image_context": f"하단 저도막 분포 idx-{idx}",
                    "dft_measurements": meas,
                    "env": {"temperature": 22 + idx, "humidity": 67 + idx},
                },
            },
            {
                "name": "dft_judge",
                "arguments": {
                    "measurements": meas,
                    "spec": {"nominal": 100, "tolerance_rule": "90/10"},
                },
            },
        ]
        tool_result = json.dumps({
            "status": "success",
            "results": [
                {"tool": "anomaly_detection", "anomaly_type": "low_thickness", "severity": "medium"},
                {"tool": "dft_judge", "verdict": "fail", "below_90pct": 4, "total": 5},
            ],
        }, ensure_ascii=False)
        tc_text = (
            f"<tool_call>{json.dumps(combined_calls[0], ensure_ascii=False)}</tool_call>\n\n"
            f"<tool_call>{json.dumps(combined_calls[1], ensure_ascii=False)}</tool_call>"
        )
        context_data = {
            "image_context": f"하단 저도막 분포 idx-{idx}",
            "dft_measurements": meas,
            "env": {"temperature": 22 + idx, "humidity": 67 + idx},
            "spec": {"nominal": 100, "tolerance_rule": "90/10"},
        }
        samples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_FC},
                {"role": "user", "content": f"[환경 데이터]\n{json.dumps(context_data, ensure_ascii=False)}\n\n질문: 이상과 판정을 함께 분석해 상세 원인 설명해줘."},
                {"role": "assistant", "content": f"이상 탐지와 DFT 판정을 위해 도구를 호출합니다.\n\n{tc_text}"},
                {"role": "tool", "content": tool_result},
                {
                    "role": "assistant",
                    "content": (
                        "도구 분석 결과를 바탕으로 답변합니다.\n\n"
                        "저도막 이상과 DFT 불합격이 동시에 확인되었습니다. "
                        "속도 과속 또는 분사 거리 과대로 인한 하한 미달이 원인으로 보이며, 재도장 전 분사 조건을 교정해야 합니다.\n\n"
                        "근거 문서 ID: DOC-doc_0037-C0308, DOC-doc_0033-C0326"
                    ),
                },
            ],
            "category": "multi_turn_chain_aug",
        })

    return samples


def _fc_volume_augment() -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    samples.extend(_location_tracking_augment())
    samples.extend(_pose_estimation_augment())
    samples.extend(_sequence_validation_augment())
    samples.extend(_timeseries_augment())
    samples.extend(_xr_content_samples())
    samples.extend(_multi_tool_augment())
    samples.extend(_multi_turn_augment())

    samples.extend(_dft_judge_samples())
    samples.extend(_anomaly_detection_samples())
    samples.extend(_location_tracking_samples())
    samples.extend(_pose_estimation_samples())
    samples.extend(_sequence_validation_samples())
    samples.extend(_timeseries_samples())
    samples.extend(_multi_tool_samples())
    samples.extend(_multi_turn_samples())

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
    samples.extend(_location_tracking_augment())
    samples.extend(_pose_estimation_samples())
    samples.extend(_pose_estimation_augment())
    samples.extend(_sequence_validation_samples())
    samples.extend(_sequence_validation_augment())
    samples.extend(_timeseries_samples())
    samples.extend(_timeseries_augment())
    samples.extend(_xr_content_samples())
    samples.extend(_multi_tool_samples())
    samples.extend(_multi_tool_augment())
    samples.extend(_multi_turn_samples())
    samples.extend(_multi_turn_augment())
    samples.extend(_no_tool_needed_samples())
    samples.extend(_fc_volume_augment())

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
