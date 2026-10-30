"""`POST /api/v1/copilot/query`의 `pose_recomm` — 저장된 시퀀스의 자세 채점.

명세서 p.9가 `pose_recomm`을 "전문가 작업 자세 유사도 비교"로, `context`를 "가리키는 대상
구역/시퀀스 식별자 (예: SEQ-012)"로 정한다. 즉 **채점 대상은 이미 올라간 시퀀스**이고
`POST /telemetry/sequence`는 그것을 데이터베이스에 올리는 일만 한다(p.4는 텔레메트리에
"도막 두께 및 궤적 평가"를 걸고 자세는 걸지 않는다).

이 모듈의 이전 배치가 자세를 텔레메트리 응답에 넣은 것은 틀렸다.

대상 고르는 순서:
1. `context`가 시퀀스 id를 가리키면 그것
2. 없으면 `session_id`의 가장 최근 시퀀스
3. 그래도 없으면 `not_available` — 시퀀스를 지어내지 않는다
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Final

import httpx

from trainer.demo_api.sequence_store import NON_CONTENT_FIELDS, SequenceStore
from trainer.scripts import pose_similarity

logger = logging.getLogger(__name__)

_MCP_TIMEOUT_SEC: Final[float] = 20.0


@dataclass(frozen=True)
class PoseQueryResult:
    """`pose_context`에 실을 내용과 답변에 쓸 요약.

    `analysis_status`가 `not_available`이면 수치는 없다 — 임의 값을 만들지 않는다.
    """

    analysis_status: str
    reason_code: str | None
    sequence_id: str | None
    similarity_score: float | None
    correction_mode: str | None
    corrections: list[dict[str, Any]]
    summary: str | None
    # 교정된 관절 → 숙련자와의 거리(m). 엔진 evidence 라 `pose_context`에는 싣지 않고 세션 리포트 집계만 쓴다.
    joint_distances: dict[str, float] = field(default_factory=dict)
    confidence: str | None = None
    input_quality: dict[str, Any] = field(default_factory=dict)

    def as_context(self) -> dict[str, Any]:
        context = {
            "analysis_status": self.analysis_status,
            "reason_code": self.reason_code,
            "sequence_id": self.sequence_id,
            "similarity_score": self.similarity_score,
            "correction_mode": self.correction_mode,
            "corrections": self.corrections,
        }
        if self.confidence is not None:
            context["confidence"] = self.confidence
        if self.input_quality:
            context["input_quality"] = self.input_quality
        return context


@dataclass(frozen=True)
class PoseEngineResult:
    """검증된 MCP 자세 엔진 원문과 안정적인 가용성 상태."""

    analysis_status: str
    reason_code: str | None
    sequence_id: str
    result: dict[str, Any]


def _raw_unavailable(reason: str, *, sequence_id: str) -> PoseEngineResult:
    return PoseEngineResult(
        analysis_status="not_available",
        reason_code=reason,
        sequence_id=sequence_id,
        result={},
    )


def _unavailable(reason: str, *, sequence_id: str | None = None) -> PoseQueryResult:
    return PoseQueryResult(
        analysis_status="not_available", reason_code=reason, sequence_id=sequence_id,
        similarity_score=None, correction_mode=None, corrections=[], summary=None,
    )


def _project_corrections(result: dict[str, Any]) -> list[dict[str, Any]]:
    """계약 §9.2의 `pose_context.corrections` 모양.

    엔진의 `evidence`·`target.per_frame`은 내부 인터페이스의 것이므로 공개하지 않는다.
    `target_validated`는 적용할 좌표가 실제로 있는지다 — 없으면 교정이 말뿐이다.
    """
    projected: list[dict[str, Any]] = []
    for item in result.get("corrections") or []:
        if not isinstance(item, dict):
            continue
        target = item.get("target") if isinstance(item.get("target"), dict) else {}
        projected.append(
            {
                "joint": str(item.get("joint", "")),
                "label": str(item.get("correction_text") or item.get("label") or ""),
                "target_validated": bool((target or {}).get("per_frame")),
                "angle_delta_deg": item.get("angle_delta_deg"),
                "direction": item.get("direction"),
            }
        )
    return projected


def resolve_sequence_id(
    store: SequenceStore, *, session_id: str | None, context: str | None
) -> str | None:
    """`context`가 시퀀스를 가리키면 그것, 아니면 세션의 최근 시퀀스."""
    if context:
        return context
    if session_id:
        return store.latest_sequence_id(session_id)
    return None


def score_pose(
    store: SequenceStore,
    *,
    session_id: str | None,
    context: str | None,
    mcp_base_url: str,
    transport: httpx.BaseTransport | None = None,
) -> PoseQueryResult:
    """저장된 시퀀스를 MCP 자세 엔진으로 채점한다. 실패는 200 + `not_available`이다."""
    if not session_id:
        # 세션이 없으면 어느 시퀀스를 채점할지 정할 수 없다.
        return _unavailable("session_id_missing")

    sequence_id = resolve_sequence_id(store, session_id=session_id, context=context)
    if sequence_id is None:
        return _unavailable("no_sequence_for_session")

    body = store.load_body(session_key=session_id, sequence_id=sequence_id)
    if body is None:
        return _unavailable("sequence_not_found", sequence_id=sequence_id)
    return score_body(body, sequence_id=sequence_id, mcp_base_url=mcp_base_url, transport=transport)


def score_body(
    body: dict[str, Any],
    *,
    sequence_id: str,
    mcp_base_url: str,
    transport: httpx.BaseTransport | None = None,
    timeout_sec: float = _MCP_TIMEOUT_SEC,
) -> PoseQueryResult:
    """원문 자세 엔진 결과를 Query API 계약으로 투영한다."""
    raw = score_body_raw(
        body, sequence_id=sequence_id, mcp_base_url=mcp_base_url, transport=transport,
        timeout_sec=timeout_sec,
    )
    if raw.analysis_status != "available":
        return _unavailable(raw.reason_code or "analysis_unavailable", sequence_id=sequence_id)
    result = raw.result
    summary = result.get("summary")
    return PoseQueryResult(
        analysis_status="available",
        reason_code=raw.reason_code,
        sequence_id=sequence_id,
        similarity_score=result.get("similarity_score"),
        correction_mode=result.get("correction_mode"),
        corrections=_project_corrections(result),
        summary=str(summary) if summary else None,
        confidence="limited" if raw.reason_code else None,
        input_quality=dict(result["input_quality"]) if isinstance(result.get("input_quality"), dict) else {},
        joint_distances={
            str(item["joint"]): float(item["evidence"]["distance_m"])
            for item in result.get("corrections") or []
            if isinstance(item, dict) and isinstance(item.get("evidence"), dict)
            and isinstance(item["evidence"].get("distance_m"), (int, float)) and item.get("joint")
        },
    )


def score_body_raw(
    body: dict[str, Any],
    *,
    sequence_id: str,
    mcp_base_url: str,
    transport: httpx.BaseTransport | None = None,
    timeout_sec: float = _MCP_TIMEOUT_SEC,
) -> PoseEngineResult:
    """시퀀스 하나를 MCP에 한 번 보내고 검증된 제품 엔진 원문을 돌려준다."""
    # `analyze-sequence`가 붙인 분석 파라미터(`mode_id`)는 시퀀스 내용이 아니다. MCP 자세 엔진은 최상위 모르는 필드를
    # 거절하므로(422) 빼고 보낸다 — 빼지 않으면 벤더 경로의 모든 질의가 `engine_unavailable`이 된다(2026-10-08 샘플).
    body = {key: value for key, value in body.items() if key not in NON_CONTENT_FIELDS}

    try:
        with httpx.Client(timeout=timeout_sec, transport=transport) as client:
            response = client.post(
                f"{mcp_base_url.rstrip('/')}/pose_estimation/sequence", json=body
            )
        if response.status_code != 200:
            logger.warning("pose engine HTTP %d sequence=%s", response.status_code, sequence_id)
            return _raw_unavailable("engine_unavailable", sequence_id=sequence_id)
        payload = response.json()
    except Exception:  # noqa: BLE001 - 엔진 장애는 200 + not_available 이다
        logger.exception("pose engine call failed sequence=%s", sequence_id)
        return _raw_unavailable("engine_unavailable", sequence_id=sequence_id)

    # MCP는 오류도 HTTP 200으로 답하므로 status 필드가 성공을 결정한다.
    if not isinstance(payload, dict) or payload.get("status") != "success":
        return _raw_unavailable("engine_unavailable", sequence_id=sequence_id)

    result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
    status = result.get("analysis_status")
    score = result.get("similarity_score")
    quality = result.get("input_quality")
    rejected = isinstance(quality, dict) and quality.get("status") == "reject"
    usable = isinstance(score, (int, float)) and not isinstance(score, bool) and math.isfinite(score) and 0 <= score <= 1
    if status not in {"available", "degraded"} or not usable or rejected:
        return _raw_unavailable(
            str(result.get("reason_code") or "analysis_unavailable"),
            sequence_id=sequence_id,
        )
    return PoseEngineResult(
        analysis_status="available",
        # Public availability means usable measurements, not unrestricted confidence.
        # Retain the quality reason rather than silently promoting degraded evidence.
        reason_code=str(result.get("reason_code") or "analysis_degraded") if status == "degraded" else None,
        sequence_id=sequence_id,
        result=dict(result),
    )


# --- 구역 질문: complete 가 남긴 구역별 결과로 답한다 ----------------------------------------
#
# "가장 문제 있는 구역이 어디야?"는 시퀀스 하나를 채점해서는 답할 수 없다. complete 의 세션 종합 분석이 리포트 DB에
# 남긴 구역별 자세(`pose_report.zone_summary`)와 구역별 도막(`coating_zones`)을 읽어 답한다. 가장 문제 있는 구역은
# **자세 평균 유사도가 가장 낮은 구역**이다(pose_recomm 이므로 자세 기준). 같으면 교정 발생률이 높은 쪽, 그다음 구역 순서.

# "존"은 넣지 않는다 — "기존", "존재"에 걸린다.
_ZONE_QUESTION: Final = re.compile(r"구역|zone|어디|어느", re.IGNORECASE)
_ZONE_NUMBER: Final = re.compile(
    r"(?:zone|구역)\s*(\d+)|(?:\b(\d+)\s*번?\s*(?:zone|구역))",
    re.IGNORECASE,
)

_ZONE_REASON_ANSWERS: Final[dict[str, str]] = {
    "session_id_missing": "어느 세션인지 알 수 없어 구역별 자세를 찾지 못했습니다. session_id를 함께 보내 주세요.",
    "session_report_unavailable": "이 세션에 분석할 분사 시퀀스나 완료 리포트가 없습니다. 먼저 훈련 데이터를 전송해 주세요.",
    "zone_analysis_unavailable": "이 세션에는 자세를 채점한 구역이 없습니다.",
    "zone_not_evaluated": "{zone}번 구역은 이 세션에서 자세를 채점한 분사가 없습니다.",
}


@dataclass(frozen=True)
class ZoneAnswer:
    analysis_status: str
    reason_code: str | None
    answer: str
    context: dict[str, Any]


def asks_about_zones(prompt: str, context: str | None) -> bool:
    """구역 단위 질문인가. `context`가 있으면 그것이 정한다 — 숫자면 구역 번호, 그 밖(`SEQ-012`)은 시퀀스다."""
    if context and context.strip():
        return context.strip().isdigit()
    return bool(_ZONE_QUESTION.search(prompt))


def _zone_from_prompt(prompt: str) -> str | None:
    match = _ZONE_NUMBER.search(prompt)
    return next((group for group in match.groups() if group), None) if match else None


def requested_zone(prompt: str, context: str | None) -> str | None:
    return context.strip() if context and context.strip().isdigit() else _zone_from_prompt(prompt)


def zone_unavailable(reason: str, *, zone_id: str | None = None) -> ZoneAnswer:
    return ZoneAnswer(
        analysis_status="not_available",
        reason_code=reason,
        answer=_ZONE_REASON_ANSWERS[reason].format(zone=zone_id),
        context={"analysis_status": "not_available", "reason_code": reason, "scope": "zone", "zone_id": zone_id},
    )


def _zone_order(zone: Mapping[str, Any]) -> tuple[float, float, int, str]:
    joints = zone.get("problem_joints") or []
    rate = float(joints[0].get("occurrence_rate") or 0.0) if joints else 0.0
    zone_id = str(zone.get("zone_id"))
    return (float(zone["average_similarity_score"]), -rate, int(zone_id) if zone_id.isdigit() else 1_000_000, zone_id)


def _coating_sentence(coating: Mapping[str, Any]) -> str:
    status = coating.get("status")
    thickness = coating.get("thickness_um")
    if status == "normal":
        return " 도막은 전문가 기준을 충족했습니다."
    if status == "under_app":
        return f" 도막은 전문가 기준보다 부족합니다(평균 {thickness:.0f}µm, 기준 도포율 {coating.get('coverage_percent', 0):.0f}%)."
    if status == "over_app":
        return f" 도막은 과도장입니다(평균 {thickness:.0f}µm, 과도장 셀 {coating.get('overpaint_percent', 0):.0f}%)."
    return ""


def answer_zone_question(found: tuple[str, Mapping[str, Any]] | None, *, context: str | None) -> ZoneAnswer:
    """complete 분석(`found` = (report_id, analysis))으로 구역 질문에 답한다. 구역 번호 `context`면 그 구역을 말한다."""
    target = context.strip() if context and context.strip().isdigit() else None
    if found is None:
        return zone_unavailable("session_report_unavailable", zone_id=target)
    report_id, analysis = found
    pose = analysis.get("pose_report") if isinstance(analysis.get("pose_report"), Mapping) else {}
    zones = [
        zone for zone in pose.get("zone_summary") or []
        if isinstance(zone, Mapping) and isinstance(zone.get("average_similarity_score"), (int, float))
    ]
    if not zones:
        return zone_unavailable("zone_analysis_unavailable", zone_id=target)
    if target is not None:
        zone = next((item for item in zones if str(item.get("zone_id")) == target), None)
        if zone is None:
            return zone_unavailable("zone_not_evaluated", zone_id=target)
    else:
        zone = min(zones, key=_zone_order)

    zone_id = str(zone["zone_id"])
    count = int(zone.get("evaluated_sequence_count") or 0)
    score = float(zone["average_similarity_score"])
    session = pose.get("average_similarity_score")
    answer = f"가장 문제가 큰 구역은 {zone_id}번 구역입니다. " if target is None else f"{zone_id}번 구역 결과입니다. "
    answer += f"이 구역 분사 {count}회의 평균 자세 유사도는 {score * 100:.0f}%"
    if isinstance(session, (int, float)):
        relation = "낮습니다" if score < session else "높습니다" if score > session else "같습니다"
        answer += f"로 세션 평균 {session * 100:.0f}%보다 {relation}." if relation != "같습니다" else f"로 세션 평균과 같습니다."
    else:
        answer += "입니다."

    corrections: list[dict[str, Any]] = []
    for joint in zone.get("problem_joints") or []:
        label = pose_similarity.JOINT_LABELS.get(str(joint.get("joint")), str(joint.get("joint")))
        corrections.append({
            "joint": joint.get("joint"), "label": label,
            "occurrence_count": joint.get("occurrence_count"), "occurrence_rate": joint.get("occurrence_rate"),
            "average_distance": joint.get("average_distance"),
        })
    if corrections:
        top = corrections[0]
        answer += f" {top['label']} 위치가 숙련자 범위를 벗어난 분사가 {count}회 중 {top['occurrence_count']}회였습니다"
        distance = top.get("average_distance")
        answer += f"(평균 {distance * 100:.0f}cm)." if isinstance(distance, (int, float)) else "."
    else:
        answer += " 숙련자 범위를 벗어난 관절은 없었습니다."

    coating = next(
        (item for item in analysis.get("coating_zones") or []
         if isinstance(item, Mapping) and str(item.get("zone_id")) == zone_id),
        None,
    )
    if coating is not None:
        answer += _coating_sentence(coating)

    return ZoneAnswer(
        analysis_status="available",
        reason_code=None,
        answer=answer,
        context={
            "analysis_status": "available",
            "reason_code": None,
            "scope": "zone",
            "report_id": report_id,
            "zone_id": zone_id,
            "sequence_id": None,
            "similarity_score": score,
            "evaluated_sequence_count": count,
            "correction_mode": None,
            "corrections": corrections,
            **({"coating": {
                key: coating.get(key)
                for key in ("status", "thickness_um", "completion_percent", "coverage_percent", "overpaint_percent")
            }} if coating is not None else {}),
        },
    )


def _pose_overview(found: tuple[str, Mapping[str, Any]] | None) -> ZoneAnswer:
    if found is None:
        return zone_unavailable("session_report_unavailable")
    report_id, analysis = found
    pose = analysis.get("pose_report") if isinstance(analysis.get("pose_report"), Mapping) else {}
    score = pose.get("average_similarity_score")
    zones = [
        zone for zone in pose.get("zone_summary") or []
        if isinstance(zone, Mapping) and isinstance(zone.get("average_similarity_score"), (int, float))
    ]
    if not isinstance(score, (int, float)) or not zones:
        return zone_unavailable("zone_analysis_unavailable")
    worst = min(zones, key=_zone_order)
    count = pose.get("evaluated_sequence_count")
    count_text = f" 분사 {int(count)}회를 평가했으며," if isinstance(count, (int, float)) else ""
    answer = (
        f"세션 전체 자세 유사도는 {float(score) * 100:.0f}%입니다."
        f"{count_text} 가장 낮은 구역은 {worst['zone_id']}번"
        f"({float(worst['average_similarity_score']) * 100:.0f}%)입니다."
    )
    return ZoneAnswer("available", None, answer, {
        "analysis_status": "available", "reason_code": None, "scope": "session",
        "report_id": report_id, "zone_id": None, "sequence_id": None,
        "similarity_score": float(score), "corrections": [],
    })


def _body_part_answer(
    found: tuple[str, Mapping[str, Any]] | None, *, side: str,
) -> ZoneAnswer:
    if found is None:
        return zone_unavailable("session_report_unavailable")
    report_id, analysis = found
    pose = analysis.get("pose_report") if isinstance(analysis.get("pose_report"), Mapping) else {}
    prefix = "Right" if side == "right" else "Left"
    candidates: list[tuple[float, float, Mapping[str, Any], Mapping[str, Any]]] = []
    for zone in pose.get("zone_summary") or []:
        if not isinstance(zone, Mapping):
            continue
        for joint in zone.get("problem_joints") or []:
            if not isinstance(joint, Mapping) or not str(joint.get("joint", "")).startswith(prefix):
                continue
            candidates.append((
                float(joint.get("occurrence_rate") or 0.0),
                float(joint.get("average_distance") or 0.0),
                zone,
                joint,
            ))
    side_label = "오른쪽" if side == "right" else "왼쪽"
    if not candidates:
        answer = f"{side_label} 팔·손은 전문가 기준을 벗어난 반복 문제가 발견되지 않았습니다."
        return ZoneAnswer("available", None, answer, {
            "analysis_status": "available", "reason_code": None, "scope": "body_part",
            "report_id": report_id, "zone_id": None, "sequence_id": None,
            "similarity_score": pose.get("average_similarity_score"), "corrections": [],
        })
    _, distance, zone, joint = max(candidates, key=lambda item: (item[0], item[1]))
    joint_id = str(joint.get("joint"))
    label = pose_similarity.JOINT_LABELS.get(joint_id, joint_id)
    occurrences = int(joint.get("occurrence_count") or 0)
    answer = (
        f"{label}이(가) 가장 큰 개선 대상입니다. "
        f"{zone.get('zone_id')}번 구역에서 전문가 범위를 벗어난 분사가 {occurrences}회였고, "
        f"평균 위치 차이는 {distance * 100:.0f}cm입니다. "
        f"{label} 위치를 전문가 기준 궤적에 가깝게 유지하세요."
    )
    correction = {
        "joint": joint_id, "label": label, "occurrence_count": occurrences,
        "occurrence_rate": joint.get("occurrence_rate"), "average_distance": joint.get("average_distance"),
    }
    return ZoneAnswer("available", None, answer, {
        "analysis_status": "available", "reason_code": None, "scope": "body_part",
        "report_id": report_id, "zone_id": str(zone.get("zone_id")), "sequence_id": None,
        "similarity_score": zone.get("average_similarity_score"), "corrections": [correction],
    })


def answer_pose_question(
    found: tuple[str, Mapping[str, Any]] | None, *, prompt: str, context: str | None, intent: str | None = None,
) -> ZoneAnswer:
    """완료 세션의 구조화된 자세 결과를 질문 의도에 맞게 설명한다."""
    target = requested_zone(prompt, context)
    if target is not None:
        return answer_zone_question(found, context=target)
    lowered = prompt.lower()
    if intent == "pose_right" or "오른" in lowered or "right" in lowered:
        return _body_part_answer(found, side="right")
    if intent == "pose_left" or "왼쪽" in lowered or "left" in lowered:
        return _body_part_answer(found, side="left")
    if intent == "pose_worst_zone" or any(token in lowered for token in ("문제", "안 좋", "낮", "어디", "어느", "최악")):
        return answer_zone_question(found, context=None)
    if intent == "pose_overall" or any(token in lowered for token in ("전체", "평균", "유사도", "점수", "퍼센트", "자세")):
        return _pose_overview(found)
    report_id = found[0] if found is not None else None
    return ZoneAnswer("available", None, (
        "자세 탭에서는 ‘전체 자세 유사도’, ‘가장 문제가 큰 Zone’, "
        "‘오른손을 어떻게 고쳐?’, ‘왼손을 어떻게 고쳐?’처럼 질문할 수 있습니다."
    ), {
        "analysis_status": "available", "reason_code": None, "scope": "help",
        "report_id": report_id, "zone_id": None, "sequence_id": None,
        "similarity_score": None, "corrections": [],
    })


def answer_order_question(
    found: tuple[str, Mapping[str, Any]] | None, *, prompt: str | None = None, intent: str | None = None,
) -> ZoneAnswer:
    """complete 의 순서 리포트(`sequence_report`)로 "순서 잘 지켰어?"에 답한다. 계산 결과를 문장으로 옮길 뿐이다."""
    if found is None:
        return zone_unavailable("session_report_unavailable")
    report_id, analysis = found
    report = analysis.get("sequence_report") if isinstance(analysis.get("sequence_report"), Mapping) else {}
    if report.get("analysis_status") != "available":
        reason = str(report.get("reason_code") or "sequence_report_unavailable")
        guideline = str(report.get("guideline") or "순서 기준이 없습니다.")
        return ZoneAnswer(
            "not_available", reason, f"이 세션은 순서를 분석하지 못했습니다. {guideline}",
            {"analysis_status": "not_available", "reason_code": reason, "scope": "session", "report_id": report_id},
        )
    flow = [str(zone) for zone in report.get("recommended_flow") or []]
    first = [str(zone) for zone in report.get("first_visit_flow") or []]
    evaluated = int(report.get("evaluated_visit_count") or 0)
    on_track = int(report.get("on_track_visit_count") or 0)
    rate = float(report.get("adherence_rate") or 0.0)
    answer = (
        f"권장 순서는 {' → '.join(flow)}입니다. 이번 세션에서 처음 칠한 순서는 {' → '.join(first) or '없음'}입니다. "
        f"구역 방문 {evaluated}회 중 {on_track}회가 순서에 맞아 순서 준수율은 {rate * 100:.0f}%입니다."
    )
    missing = [zone for zone in flow if zone not in first]
    if missing:
        answer += f" 칠하지 않은 구역은 {'·'.join(missing)}번입니다."
    zones = [zone for zone in report.get("zone_summary") or [] if isinstance(zone, Mapping)]
    worst = max(zones, key=lambda zone: int(zone.get("deviation_count") or 0), default=None)
    if worst is not None and int(worst.get("deviation_count") or 0) > 0:
        answer += f" 순서 이탈이 가장 많은 구역은 {worst['zone_id']}번({worst['deviation_count']}회)입니다."
    direction = report.get("direction_adherence_rate")
    question = (prompt or "").lower()
    if question:
        if intent == "order_missing" or any(token in question for token in ("누락", "안 칠", "빠진", "미수행")):
            focused = (
                f"권장 순서 중 칠하지 않은 구역은 {'·'.join(missing)}번입니다."
                if missing else "권장 순서의 모든 구역을 칠했습니다."
            )
            return ZoneAnswer("available", None, focused, {
                "analysis_status": "available", "reason_code": None, "scope": "session", "report_id": report_id,
                "recommended_flow": flow, "first_visit_flow": first, "adherence_rate": rate,
                "deviation_count": report.get("deviation_count"), "direction_adherence_rate": direction,
            })
        if intent == "order_direction" or "방향" in question:
            focused = (
                f"전문가 기준 도포 방향 준수율은 {float(direction) * 100:.0f}%입니다."
                if isinstance(direction, (int, float)) else "이 세션은 도포 방향을 평가할 수 없습니다."
            )
            return ZoneAnswer("available", None, focused, {
                "analysis_status": "available", "reason_code": None, "scope": "session", "report_id": report_id,
                "recommended_flow": flow, "first_visit_flow": first, "adherence_rate": rate,
                "deviation_count": report.get("deviation_count"), "direction_adherence_rate": direction,
            })
        if intent == "order_actual" or any(token in question for token in ("실제", "수행 순서", "칠한 순서")):
            focused = f"이번 세션에서 처음 칠한 순서는 {' → '.join(first) or '없음'}입니다."
            return ZoneAnswer("available", None, focused, {
                "analysis_status": "available", "reason_code": None, "scope": "session", "report_id": report_id,
                "recommended_flow": flow, "first_visit_flow": first, "adherence_rate": rate,
                "deviation_count": report.get("deviation_count"), "direction_adherence_rate": direction,
            })
        if intent == "order_reference" or any(token in question for token in ("권장", "추천", "정석")):
            reference_count = int(report.get("reference_session_count") or 0)
            distributions = report.get("position_distribution") or []
            if len(flow) == 1 and reference_count:
                focused = (
                    f"학습모드 1은 Zone {flow[0]} 한 곳으로 구성되어 있으며, "
                    f"전형기 반장 {reference_count}회 모두 Zone {flow[0]}부터 작업했습니다."
                )
            else:
                focused = f"권장 작업 순서는 Zone {' → '.join(flow)}입니다."
                top_choices: list[tuple[str, float]] = []
                for row in distributions[:2]:
                    if isinstance(row, list) and row and isinstance(row[0], Mapping):
                        top_choices.append((
                            str(row[0].get("zone_id")), float(row[0].get("probability") or 0),
                        ))
                if (
                    len(flow) >= 2 and len(top_choices) == 2
                    and top_choices[0][0] == flow[0] and top_choices[1][0] == flow[1]
                    and min(top_choices[0][1], top_choices[1][1]) >= 0.6
                ):
                    confidence = min(top_choices[0][1], top_choices[1][1]) * 100
                    focused += (
                        f"\n\n우선 Zone {flow[0]}에서 시작한 뒤 Zone {flow[1]}로 이동하세요. "
                        f"이 시작 순서는 전문가 작업의 약 {confidence:.0f}%에서 동일하게 나타났습니다."
                    )
                elif flow:
                    focused += f"\n\n우선 Zone {flow[0]}에서 시작하세요."
            return ZoneAnswer("available", None, focused, {
                "analysis_status": "available", "reason_code": None, "scope": "session", "report_id": report_id,
                "recommended_flow": flow, "first_visit_flow": first, "adherence_rate": rate,
                "deviation_count": report.get("deviation_count"), "direction_adherence_rate": direction,
                "reference_session_count": reference_count,
                "best_reference_similarity": report.get("best_reference_similarity"),
                "position_similarity": report.get("position_similarity"),
                "position_distribution": distributions,
            })
        if intent == "order_adherence" or any(token in question for token in ("준수", "유사", "몇 퍼센트", "점수")):
            closest = report.get("best_reference_similarity")
            positional = report.get("position_similarity")
            focused = (
                f"전문가 순서 유사도는 {rate * 100:.0f}%입니다."
            )
            if isinstance(closest, (int, float)) and isinstance(positional, (int, float)):
                focused += (
                    f" 가장 가까운 전문가 수행과 {float(closest) * 100:.0f}%, "
                    f"순번별 통계와 {float(positional) * 100:.0f}% 유사합니다."
                )
            else:
                focused = (
                    f"순서 준수율은 {rate * 100:.0f}%입니다. "
                    f"구역 방문 {evaluated}회 중 {on_track}회가 권장 순서에 맞았습니다."
                )
            return ZoneAnswer("available", None, focused, {
                "analysis_status": "available", "reason_code": None, "scope": "session", "report_id": report_id,
                "recommended_flow": flow, "first_visit_flow": first, "adherence_rate": rate,
                "deviation_count": report.get("deviation_count"), "direction_adherence_rate": direction,
            })
        return ZoneAnswer("available", None, (
            "순서 탭에서는 ‘권장 Zone 순서’, ‘실제 수행 순서’, ‘누락한 Zone’, "
            "‘순서 준수율’, ‘도포 방향’처럼 질문할 수 있습니다."
        ), {
            "analysis_status": "available", "reason_code": None, "scope": "help", "report_id": report_id,
            "recommended_flow": flow, "first_visit_flow": first, "adherence_rate": rate,
            "deviation_count": report.get("deviation_count"), "direction_adherence_rate": direction,
        })
    if isinstance(direction, (int, float)):
        answer += f" 도포 방향 준수율은 {direction * 100:.0f}%입니다."
    return ZoneAnswer("available", None, answer, {
        "analysis_status": "available", "reason_code": None, "scope": "session", "report_id": report_id,
        "recommended_flow": flow, "first_visit_flow": first, "adherence_rate": rate,
        "deviation_count": report.get("deviation_count"), "direction_adherence_rate": direction,
    })
