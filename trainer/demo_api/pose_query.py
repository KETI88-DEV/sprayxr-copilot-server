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

    def as_context(self) -> dict[str, Any]:
        return {
            "analysis_status": self.analysis_status,
            "reason_code": self.reason_code,
            "sequence_id": self.sequence_id,
            "similarity_score": self.similarity_score,
            "correction_mode": self.correction_mode,
            "corrections": self.corrections,
        }


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
) -> PoseQueryResult:
    """원문 자세 엔진 결과를 Query API 계약으로 투영한다."""
    raw = score_body_raw(
        body, sequence_id=sequence_id, mcp_base_url=mcp_base_url, transport=transport
    )
    if raw.analysis_status != "available":
        return _unavailable(raw.reason_code or "analysis_unavailable", sequence_id=sequence_id)
    result = raw.result
    summary = result.get("summary")
    return PoseQueryResult(
        analysis_status="available",
        reason_code=None,
        sequence_id=sequence_id,
        similarity_score=result.get("similarity_score"),
        correction_mode=result.get("correction_mode"),
        corrections=_project_corrections(result),
        summary=str(summary) if summary else None,
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
) -> PoseEngineResult:
    """시퀀스 하나를 MCP에 한 번 보내고 검증된 제품 엔진 원문을 돌려준다."""
    # `analyze-sequence`가 붙인 분석 파라미터(`mode_id`)는 시퀀스 내용이 아니다. MCP 자세 엔진은 최상위 모르는 필드를
    # 거절하므로(422) 빼고 보낸다 — 빼지 않으면 벤더 경로의 모든 질의가 `engine_unavailable`이 된다(2026-10-08 샘플).
    body = {key: value for key, value in body.items() if key not in NON_CONTENT_FIELDS}

    try:
        with httpx.Client(timeout=_MCP_TIMEOUT_SEC, transport=transport) as client:
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
    if status != "available":
        return _raw_unavailable(
            str(result.get("reason_code") or "analysis_unavailable"),
            sequence_id=sequence_id,
        )
    return PoseEngineResult(
        analysis_status="available",
        reason_code=None,
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

_ZONE_REASON_ANSWERS: Final[dict[str, str]] = {
    "session_id_missing": "어느 세션인지 알 수 없어 구역별 자세를 찾지 못했습니다. session_id를 함께 보내 주세요.",
    "session_report_unavailable": "구역별 자세 분석은 훈련 종료(sessions/complete) 뒤에 만들어집니다. 종료 후 다시 물어봐 주세요.",
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
