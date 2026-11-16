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
import secrets
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Final

import httpx

from trainer.demo_api.coating_assessment import basis_label, grade_phrase
from trainer.demo_api.sequence_store import NON_CONTENT_FIELDS, SequenceStore
from trainer.scripts import pose_similarity

logger = logging.getLogger(__name__)

_MCP_TIMEOUT_SEC: Final[float] = 20.0
# 자세 유사도를 말하는 답에 붙인다. 숙련자끼리 견줘도 중앙값이 69~78%라 70%를 낮은 점수로 읽기 쉽다(2026-11-14 사용자 지적).
POSE_SCALE_NOTE: Final[str] = "숙련자 평균 동작 대비 유사도이며, 숙련자도 70% 안팎입니다."
# 엔진이 교정을 내지 않은 것은 "정상"이 아니다. 엔진 안내문(session_analysis._guideline)과 같은 문장이다.
NO_CORRECTION_CAVEAT: Final[str] = "교정이 없어도 자세가 정상이라는 뜻은 아닙니다."
# 엔진 관절 이름 → 질문 낱말. 팔꿈치는 ForeArm, 손목은 Hand다(pose_similarity.JOINT_LABELS).
PART_WORDS: Final[dict[str, str]] = {"ForeArm": "팔꿈치", "Hand": "손목", "UpperArm": "위팔"}
_PART_QUESTION: Final = (("ForeArm", re.compile(r"팔꿈치|elbow", re.I)), ("Hand", re.compile(r"손목|wrist", re.I)),
                         ("UpperArm", re.compile(r"위팔|상완|upper\s*arm", re.I)))
_ANGLE_QUESTION: Final = re.compile(r"각도|몇\s*도|°|degree|angle", re.I)
# 손·팔 낱말이 있어도 관절 자세가 아닌 질문: 보호구·통증·손잡이·건(도구) 각도·압력(2026-11-14 독립 검토).
NOT_JOINT: Final = re.compile(r"보호대|보호구|장갑|착용|통증|아프|아파|부상|예방|스트레칭|손잡이|그립|잡은|쥔|쥐고|"
                              r"(?<![가-힣])건(?![가-힣])|건\s*(?:각도|을|의|이|은)|스프레이\s*건|도구|노즐|압력|안전", re.I)
ANGLE_NOTE: Final[str] = "관절 각도 수치는 없으며, 숙련자 평균 위치의 거리·방향을 비교합니다."


def wants_full_list(prompt: str, *, subject: str = r"Zone|존|구역|분사|시퀀스|sequence") -> bool:
    """Distinguish a requested breakdown from an overall average question."""
    return bool(re.search(rf"(?:{subject})\s*별|(?:모든|전체)\s*(?:{subject})|"
                          rf"(?:{subject}).{{0,30}}(?:전부|모두|각각|목록)|(?:전부|모두|전체)\s*(?:보여|알려|나열|출력)", prompt, re.I))


def order_request_fields(prompt: str) -> set[str]:
    """Fields explicitly requested together, independent of the chosen tab intent."""
    markers = {'reference':r'권장|추천|정석', 'actual':r'실제|수행\s*순서|칠한\s*순서',
               'score':r'준수|유사도|점수|수치', 'direction':r'방향', 'missing':r'누락|빠진|미방문|미수행|안\s*칠'}
    return {key for key, pattern in markers.items() if re.search(pattern, prompt, re.I)}


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
    # 교정 관절 → 교정 문장이 말한 척도("radius", 반경 차 m) 또는 ("heading", 방향 차 °). 우선 교정 문장이 같은 수를 쓴다.
    joint_measures: dict[str, tuple[str, float]] = field(default_factory=dict)
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
    # 질문에 답하려고 다시 채점할 때는 엔진의 세션 기록에 쌓지 않는다.
    return score_body(body, sequence_id=sequence_id, mcp_base_url=mcp_base_url, transport=transport, isolated=True)


def score_body(
    body: dict[str, Any],
    *,
    sequence_id: str,
    mcp_base_url: str,
    transport: httpx.BaseTransport | None = None,
    timeout_sec: float = _MCP_TIMEOUT_SEC,
    isolated: bool = False,
) -> PoseQueryResult:
    """원문 자세 엔진 결과를 Query API 계약으로 투영한다."""
    raw = score_body_raw(
        body, sequence_id=sequence_id, mcp_base_url=mcp_base_url, transport=transport,
        timeout_sec=timeout_sec, isolated=isolated,
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
        joint_measures={
            str(item["joint"]): measure
            for item in result.get("corrections") or []
            if isinstance(item, dict) and item.get("joint") and (measure := _measure(item.get("evidence"))) is not None
        },
    )


def _measure(evidence: object) -> tuple[str, float] | None:
    """교정 문장이 쓴 척도(arm_reference.Correction.text와 같은 판단: 반경 몫 σ가 크거나 같으면 반경)."""
    if not isinstance(evidence, dict):
        return None
    radial, angular = evidence.get("radial_sigma"), evidence.get("angular_sigma")
    radius, angle = evidence.get("radius_delta_m"), evidence.get("angle_from_expert_deg")
    if not isinstance(radial, (int, float)) or not isinstance(angular, (int, float)):
        return None
    if radial >= angular and isinstance(radius, (int, float)):
        return "radius", float(radius)
    if radial < angular and isinstance(angle, (int, float)):
        return "heading", float(angle)
    return None


def score_body_raw(
    body: dict[str, Any],
    *,
    sequence_id: str,
    mcp_base_url: str,
    transport: httpx.BaseTransport | None = None,
    timeout_sec: float = _MCP_TIMEOUT_SEC,
    isolated: bool = False,
) -> PoseEngineResult:
    """시퀀스 하나를 MCP에 한 번 보내고 검증된 제품 엔진 원문을 돌려준다.

    `isolated`면 일회용 엔진 세션으로 보낸다 — 엔진은 받은 시퀀스를 세션 기록에 쌓으므로,
    같은 분사를 질문마다 다시 보내면 실제 세션의 누적 기록이 부풀었다(2026-11-11 Spark).
    """
    # `analyze-sequence`가 붙인 분석 파라미터(`mode_id`)는 시퀀스 내용이 아니다. MCP 자세 엔진은 최상위 모르는 필드를
    # 거절하므로(422) 빼고 보낸다 — 빼지 않으면 벤더 경로의 모든 질의가 `engine_unavailable`이 된다(2026-10-08 샘플).
    body = {key: value for key, value in body.items() if key not in NON_CONTENT_FIELDS}
    if isolated:
        body["session_id"] = f"query:{secrets.token_hex(8)}"

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


_EXPERT_SENTENCE: Final = {"normal": "전문가 기준을 충족합니다", "under_app": "전문가보다 얇습니다", "over_app": "전문가보다 두껍습니다"}


def _coating_sentence(coating: Mapping[str, Any], target_um: object = None) -> str:
    """구역 도막 문장. 세션 목표 두께로 판정한 리포트는 전문가 기준이 아니라 그 목표를 말한다."""
    if "grade" in coating:
        basis = basis_label(coating["target_thickness_um"], coating["target_basis"])
        expert = coating.get("expert")
        compare = f" 전문가 최종 도막과 견주면 {_EXPERT_SENTENCE[expert['status']]}." if isinstance(expert, Mapping) else ""
        return f" 이 구역 도막은 {basis}으로 {grade_phrase(coating)}입니다.{compare}"
    # `grade`가 없으면 2026-11 이전에 저장된 리포트의 행이다. 그때 판정한 기준 그대로 말한다.
    status = coating.get("status")
    thickness = coating.get("thickness_um")
    targeted = isinstance(target_um, (int, float)) and not isinstance(target_um, bool)
    if status == "normal":
        return f" 이 구역 도막은 세션 목표 두께 {target_um:g}µm를 충족했습니다." if targeted else " 이 구역 도막은 전문가 기준을 충족했습니다."
    if status == "under_app":
        basis = f"세션 목표 두께 {target_um:g}µm" if targeted else "전문가 기준"
        return f" 이 구역 도막은 {basis}보다 부족합니다(평균 {thickness:.0f}µm, 기준 도포율 {coating.get('coverage_percent', 0):.0f}%)."
    if status == "over_app":
        return f" 이 구역 도막은 과도장입니다(평균 {thickness:.0f}µm, 과도장 셀 {coating.get('overpaint_percent', 0):.0f}%)."
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
        return zone_unavailable("zone_not_evaluated" if target is not None else "zone_analysis_unavailable", zone_id=target)
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
        # 답에 보이는 정수 %로 견준다. 51%가 51%보다 "높다"고 하지 않는다.
        shown, average = round(score * 100), round(session * 100)
        relation = "낮습니다" if shown < average else "높습니다" if shown > average else "같습니다"
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
        answer += f" 숙련자 범위를 벗어난 관절은 없었습니다. {NO_CORRECTION_CAVEAT}"

    coating = next(
        (item for item in analysis.get("coating_zones") or []
         if isinstance(item, Mapping) and str(item.get("zone_id")) == zone_id),
        None,
    )
    if coating is not None:
        report = analysis.get("coating_report") if isinstance(analysis.get("coating_report"), Mapping) else {}
        answer += _coating_sentence(coating, report.get("target_thickness_um") if report.get("comparison_basis") == "session_target" else None)
    answer += f" {POSE_SCALE_NOTE}"

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
                for key in (
                    "status", "thickness_um", "completion_percent", "coverage_percent", "overpaint_percent",
                    "grade", "unpainted_fraction", "target_thickness_um", "target_basis", "gate_status",
                )
            }} if coating is not None else {}),
        },
    )


_TRACKING_SENTENCE: Final = re.compile(r"\s*착탄을 몸이 따라간 기울기 [+-]?\d+(?:\.\d+)?\(시퀀스 \d+개\)\.?")
# 팔 관절 엔진이 재지 않는 부위. 팔꿈치(ForeArm)·손목(Hand)·위팔은 잰다 — 각도가 아니라 위치를(ANGLE_NOTE).
_UNSCORED_PART: Final = re.compile(r"어깨|무릎|허리|shoulder|knee", re.I)
_WHOLE_SESSION: Final = re.compile(r"전체|세션|가장|어느|어디|최악")
_ARM_WORDS: Final = {"right": ("Right", "오른팔"), "left": ("Left", "왼팔")}


def asked_arm(prompt: str, intent: str | None = None) -> str | None:
    lowered = prompt.lower()
    if intent == "pose_right" or "오른" in lowered or "right" in lowered:
        return "right"
    if intent == "pose_left" or "왼" in lowered or "left" in lowered:
        return "left"
    return None


def _has_final(word: str) -> bool:
    """끝 글자에 받침이 있는가. 숫자는 읽는 소리로 본다(1 일, 3 삼, 6 육, 7 칠, 8 팔, 0 영·십·백 — 받침 있음)."""
    if not word:
        return False
    if word[-1].isdigit():
        return word[-1] in "013678"
    code = ord(word[-1]) - 0xAC00
    return 0 <= code < 11172 and code % 28 != 0


def josa(word: str, with_final: str, without_final: str) -> str:
    """'Zone 8' + 은/는 → 'Zone 8은'."""
    return f"{word}{with_final if _has_final(word) else without_final}"


def radius_phrase(label: str, delta_m: float, *, average: bool = False) -> str:
    """반경 교정(어깨 중점에서 관절까지의 거리 차, arm_reference.Correction)을 몸을 기준으로 적고 할 일을 붙인다."""
    part = label.split()[-1]
    subject = f"{label}{'이' if _has_final(label) else '가'}"
    obj = f"{part}{'을' if _has_final(part) else '를'}"
    cm = f"{'평균 ' if average else ''}{abs(delta_m) * 100:.0f}cm"
    if delta_m < 0:
        return f"{subject} 숙련자 평균보다 몸 쪽으로 {cm} 더 붙어 있었습니다. {obj} 몸에서 더 떼어 주세요."
    return f"{subject} 숙련자 평균보다 몸 밖으로 {cm} 더 뻗어 있었습니다. {obj} 몸 쪽으로 당겨 주세요."


_RADIUS_LABEL: Final = re.compile(r"^(?P<where>스트로크\s*\d+~\d+%\s*구간에서\s*)?(?P<label>\S+\s\S+?)(?:이|가) 숙련자 평균보다 "
                                  r"(?P<cm>\d+(?:\.\d+)?)\s*cm (?P<way>멀리|가까이) 있었습니다\.?$")


def friendly_correction(label: str) -> str:
    """MCP 자세 엔진의 교정 문장을 훈련생이 바로 쓸 말로 바꾼다. 엔진 이미지는 다시 배포하지 않으므로 여기서 바꾼다.

    엔진 문장은 arm_reference.Correction.text가 반경 차의 부호(멀리/가까이)와 스트로크 구간으로 만든 것이라 그대로
    되읽을 수 있다. 스트로크 구간 표현은 그대로 둔다. 모르는 모양(방향 교정 등)은 바꾸지 않는다.
    """
    text = label.strip()
    match = _RADIUS_LABEL.match(text)
    if match:
        delta = float(match["cm"]) / 100 * (1 if match["way"] == "멀리" else -1)
        return _whole_stroke(match["where"] or "") + radius_phrase(match["label"], delta)
    return _whole_stroke(text)


def _whole_stroke(text: str) -> str:
    """스트로크 전체(0~100%)는 '스트로크 내내'. 다른 구간 표현은 그대로 둔다."""
    return re.sub(r"스트로크\s*0~100%\s*구간에서\s*", "스트로크 내내 ", text)


# 엔진이 채점을 제한한 사유(skeleton_gate·reference). 프레임 간격(R7)만 고르지 않은 분사는 채점에서 빠진 것이 없어 말하지 않는다.
_SILENT_LIMITS: Final = {"frame_interval_irregular", "analysis_degraded"}
_LIMIT_WORDS: Final = {
    "skeleton_joint_frozen": "관절 일부가 멈춘 채로 기록된",
    "skeleton_bone_length_inconsistent": "관절 위치가 흔들려 기록된",
    "skeleton_rig_mismatch": "기준과 다른 몸 모델로 기록된",
    "reference_single_worker": "비교할 숙련자 기록이 한 명뿐인",
}


def limit_reasons(item: Mapping[str, Any]) -> list[str]:
    """제한 사유 가운데 말할 것. 사유를 모르는 제한 결과는 "unknown" 하나로 센다."""
    if item.get("confidence") != "limited":
        return []
    quality = item.get("input_quality") if isinstance(item.get("input_quality"), Mapping) else {}
    codes = [str(code) for code in [*(item.get("reason_codes") or []), *(quality.get("reason_codes") or []), item.get("reason_code")] if code]
    if not codes:
        return ["unknown"]
    return [code for code in dict.fromkeys(codes) if code not in _SILENT_LIMITS]


def limited_note(items: list[Mapping[str, Any]]) -> str:
    """참고용 분사를 사유별로 한 번만, 쉬운 말로. 말할 사유가 없으면 빈 문자열."""
    counts: dict[str, int] = {}
    for item in items:
        reasons = limit_reasons(item)
        if reasons:
            key = next((code for code in reasons if code in _LIMIT_WORDS), reasons[0])
            counts[key] = counts.get(key, 0) + 1
    parts = [f"{_LIMIT_WORDS.get(code, '기록 상태가 고르지 않은')} 분사 {count}회" for code, count in counts.items()]
    return f"{', '.join(parts)}는 참고로만 봐 주세요." if parts else ""


def asked_part(prompt: str) -> str | None:
    """팔꿈치·손목·위팔을 물으면 엔진 관절 이름의 뒷부분(ForeArm·Hand·UpperArm)."""
    return next((part for part, pattern in _PART_QUESTION if pattern.search(prompt)), None)


def prefers_hand(prompt: str) -> bool:
    """오른손·왼손·손을 물었는가(손목·손잡이는 아니다)."""
    return bool(re.search(r"손(?!목|잡이|가락)|hand", prompt, re.I))


def angle_note(prompt: str) -> str:
    """각도를 물었으면 왜 각도 수치가 없는지 먼저 밝힌다(지어내지 않는다)."""
    joint = asked_part(prompt) is not None or bool(re.search(r"팔|arm", prompt, re.I))
    return ANGLE_NOTE + " " if _ANGLE_QUESTION.search(prompt) and joint and not NOT_JOINT.search(prompt) else ""


def asks_whole_session(prompt: str, intent: str | None = None) -> bool:
    """구역이나 분사 하나가 아니라 세션 전체를 묻는 질문인가."""
    return intent == "pose_worst_zone" or bool(_WHOLE_SESSION.search(prompt))


def unsupported_pose_part(prompt: str) -> str | None:
    """채점 엔진이 재지 않는 부위·각도를 물으면 지어내지 않고 잴 수 없다고 답한다."""
    match = _UNSCORED_PART.search(prompt)
    if match is None:
        return None
    return (f"질문하신 내용({match.group(0)})은 채점 결과로는 확인할 수 없습니다. "
            "채점 엔진은 양팔의 위팔·팔꿈치·손목 위치만 숙련자와 비교합니다.")


def answer_sequence_question(pose: PoseQueryResult, *, prompt: str, intent: str | None = None) -> str:
    """분사 하나의 엔진 결과에서 질문이 묻는 것만 답한다."""
    side, part = asked_arm(prompt, intent), asked_part(prompt)
    if part is not None:
        # 팔꿈치(ForeArm)·손목(Hand)·위팔: 그 관절의 교정만 옮긴다. 쪽을 안 말하면 양팔이다.
        prefixes = (_ARM_WORDS[side][0],) if side else ("Right", "Left")
        found = [str(item["label"]) for item in pose.corrections if item.get("label")
                 and str(item.get("joint", "")).startswith(prefixes) and str(item.get("joint", "")).endswith(part)]
        name = f"{'오른쪽 ' if side == 'right' else '왼쪽 ' if side == 'left' else ''}{PART_WORDS[part]}"
        return angle_note(prompt) + (" ".join(map(friendly_correction, found)) if found
                                     else f"이 분사에서 {name} 교정 항목은 나오지 않았습니다. {NO_CORRECTION_CAVEAT}")
    if side is None:
        return _TRACKING_SENTENCE.sub("", pose.summary or "").strip() or "숙련자 자세와 비교했습니다."
    prefix, arm = _ARM_WORDS[side]
    mine = [item for item in pose.corrections if str(item.get("joint", "")).startswith(prefix) and item.get("label")]
    if mine and prefers_hand(prompt):
        # 오른손·왼손을 물으면 손목(Hand) 교정을 먼저 든다. 없으면 그렇다고 밝히고 팔 교정을 든다.
        hand = [item for item in mine if str(item.get("joint", "")).endswith("Hand")]
        side_word = "오른쪽" if side == "right" else "왼쪽"
        if hand:
            return " ".join(friendly_correction(str(item["label"])) for item in hand)
        return f"{side_word} 손목 교정은 없었습니다. 대신 팔 교정이 있습니다. " + " ".join(friendly_correction(str(item["label"])) for item in mine)
    if mine:
        return " ".join(friendly_correction(str(item["label"])) for item in mine)
    others = [str(item["joint"]) for item in pose.corrections if item.get("joint")]
    if others:
        label = pose_similarity.JOINT_LABELS.get(others[0], others[0])
        return f"이 분사에서 {arm}은 교정 대상이 아닙니다. 채점 엔진이 짚은 관절은 {label}입니다."
    return f"이 분사에서 {arm} 교정 항목은 나오지 않았습니다. {NO_CORRECTION_CAVEAT}"


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
        f"({float(worst['average_similarity_score']) * 100:.0f}%)입니다. {POSE_SCALE_NOTE}"
    )
    return ZoneAnswer("available", None, answer, {
        "analysis_status": "available", "reason_code": None, "scope": "session",
        "report_id": report_id, "zone_id": None, "sequence_id": None,
        "similarity_score": float(score), "corrections": [],
    })


def _body_part_answer(
    found: tuple[str, Mapping[str, Any]] | None, *, side: str | None, zone_id: str | None = None, part: str | None = None,
) -> ZoneAnswer:
    """세션 전체, 또는 `zone_id` 구역 안에서 그쪽 팔(또는 그 관절)의 반복 문제를 답한다. 쪽을 안 말하면 양팔이다."""
    if found is None:
        return zone_unavailable("session_report_unavailable", zone_id=zone_id)
    report_id, analysis = found
    pose = analysis.get("pose_report") if isinstance(analysis.get("pose_report"), Mapping) else {}
    prefixes = ("Right",) if side == "right" else ("Left",) if side == "left" else ("Right", "Left")
    zones = [zone for zone in pose.get("zone_summary") or []
             if isinstance(zone, Mapping) and (zone_id is None or str(zone.get("zone_id")) == zone_id)]
    if zone_id is not None and not zones:
        return zone_unavailable("zone_not_evaluated", zone_id=zone_id)
    candidates: list[tuple[float, float, Mapping[str, Any], Mapping[str, Any]]] = []
    for zone in zones:
        for joint in zone.get("problem_joints") or []:
            name = str(joint.get("joint", "")) if isinstance(joint, Mapping) else ""
            if not name.startswith(prefixes) or (part is not None and not name.endswith(part)):
                continue
            candidates.append((
                float(joint.get("occurrence_rate") or 0.0),
                float(joint.get("average_distance") or 0.0),
                zone,
                joint,
            ))
    side_label = "오른쪽 " if side == "right" else "왼쪽 " if side == "left" else "양쪽 "
    if not candidates:
        where = f"{zone_id}번 구역에서 " if zone_id is not None else ""
        subject = PART_WORDS[part] if part else "팔·손"
        answer = f"{where}{side_label}{subject}은(는) 전문가 기준을 벗어난 반복 문제가 발견되지 않았습니다. {NO_CORRECTION_CAVEAT}"
        return ZoneAnswer("available", None, answer, {
            "analysis_status": "available", "reason_code": None, "scope": "body_part",
            "report_id": report_id, "zone_id": zone_id, "sequence_id": None,
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
    side, part = asked_arm(prompt, intent), asked_part(prompt)
    if target is not None:
        # 구역은 어디를 볼지만 정한다. 무엇을 답할지는 질문이 정한다.
        if side or part:
            return _body_part_answer(found, side=side, zone_id=target, part=part)
        return answer_zone_question(found, context=target)
    lowered = prompt.lower()
    if side or part:
        answer = _body_part_answer(found, side=side, part=part)
        return ZoneAnswer(answer.analysis_status, answer.reason_code, angle_note(prompt) + answer.answer, answer.context)
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
    requested = order_request_fields(question)
    if len(requested) > 1:
        parts = []
        if 'reference' in requested:
            parts.append("권장: Zone " + (' → '.join(flow) or '기준 없음'))
        if 'actual' in requested:
            parts.append("실제: " + ('Zone ' + ' → '.join(first) if first else '방문 기록 없음'))
        if 'missing' in requested:
            parts.append("미방문: " + ('Zone ' + '·'.join(missing) if missing else '없음'))
            from trainer.demo_api.viewer_progress import gate_lines
            gated = [row for row in analysis.get('coating_zones') or [] if isinstance(row, Mapping) and row.get('gate_status')]
            if gated:
                parts.extend(gate_lines(gated, order=flow, visited=set(first),
                                        footer='세션이 끝났을 때의 도막 기준입니다.')[0])
        if 'score' in requested:
            value = report.get('adherence_rate')
            scores = [f"순서 유사도 {float(value) * 100:.0f}%"] if isinstance(value, (int,float)) else ['순서 유사도 자료 없음']
            all_scores = bool(re.search(r'모두|전부|수치|통계|세부', question))
            for label, key, named in [('최근접 전문가 수행', 'best_reference_similarity', r'최근접|가장\s*가까운\s*전문가'),
                                      ('순번별 통계', 'position_similarity', r'순번\s*별')]:
                metric = report.get(key)
                if (all_scores or re.search(named, question)) and isinstance(metric, (int,float)):
                    scores.append(f'{label} {float(metric) * 100:.0f}%')
            parts.append(', '.join(scores))
        if 'direction' in requested:
            from trainer.demo_api.playback_direction import LABELS
            summary = report.get('direction_summary')
            label = LABELS.get(summary.get('classification')) if isinstance(summary, Mapping) else None
            if label:
                parts.append(f'전체 기록의 주요 도포 방향 {label}')
            if isinstance(direction, (int,float)):
                parts.append(f'도포 방향 준수율 {float(direction) * 100:.0f}%')
            elif not label:
                parts.append('도포 방향 평가 자료 없음')
        return ZoneAnswer('available', None, '; '.join(parts) + '.', {
            'analysis_status':'available', 'reason_code':None, 'scope':'session', 'report_id':report_id,
            'recommended_flow':flow, 'first_visit_flow':first, 'adherence_rate':report.get('adherence_rate'),
            'deviation_count':report.get('deviation_count'), 'direction_adherence_rate':direction,
            'best_reference_similarity':report.get('best_reference_similarity'), 'position_similarity':report.get('position_similarity'),
        })
    if question:
        if intent == "order_missing" or any(token in question for token in ("누락", "안 칠", "빠진", "빼먹", "미수행", "보완", "다시 칠")):
            focused = (
                f"권장 순서 중 칠하지 않은 구역은 {'·'.join(missing)}번입니다."
                if missing else "권장 순서의 모든 구역을 칠했습니다."
            )
            # 방문만으로는 "다 칠했다"가 아니다. 리포트 행에 최종 게이트가 있으면 보완 대상을 함께 말한다.
            from trainer.demo_api.viewer_progress import gate_lines
            gated = [row for row in analysis.get("coating_zones") or [] if isinstance(row, Mapping) and row.get("gate_status")]
            if gated:
                focused = "\n".join([focused, *gate_lines(gated, order=flow, visited=set(first), footer="세션이 끝났을 때의 도막 기준입니다.")[0]])
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
            focused = f"권장 작업 순서는 Zone {' → '.join(flow)}입니다."
            if reference_count:
                focused += f" 전문가 {reference_count}회 기준입니다."
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
