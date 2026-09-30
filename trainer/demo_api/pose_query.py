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
from dataclasses import dataclass
from typing import Any, Final

import httpx

from trainer.demo_api.sequence_store import SequenceStore

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

    def as_context(self) -> dict[str, Any]:
        return {
            "analysis_status": self.analysis_status,
            "reason_code": self.reason_code,
            "sequence_id": self.sequence_id,
            "similarity_score": self.similarity_score,
            "correction_mode": self.correction_mode,
            "corrections": self.corrections,
        }


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

    try:
        with httpx.Client(timeout=_MCP_TIMEOUT_SEC, transport=transport) as client:
            response = client.post(
                f"{mcp_base_url.rstrip('/')}/pose_estimation/sequence", json=body
            )
        if response.status_code != 200:
            logger.warning("pose engine HTTP %d sequence=%s", response.status_code, sequence_id)
            return _unavailable("engine_unavailable", sequence_id=sequence_id)
        payload = response.json()
    except Exception:  # noqa: BLE001 - 엔진 장애는 200 + not_available 이다
        logger.exception("pose engine call failed sequence=%s", sequence_id)
        return _unavailable("engine_unavailable", sequence_id=sequence_id)

    # MCP는 오류도 HTTP 200으로 답하므로 status 필드가 성공을 결정한다.
    if not isinstance(payload, dict) or payload.get("status") != "success":
        return _unavailable("engine_unavailable", sequence_id=sequence_id)

    result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
    status = result.get("analysis_status")
    if status != "available":
        return _unavailable(str(result.get("reason_code") or "analysis_unavailable"), sequence_id=sequence_id)

    summary = result.get("summary")
    return PoseQueryResult(
        analysis_status="available",
        reason_code=None,
        sequence_id=sequence_id,
        similarity_score=result.get("similarity_score"),
        correction_mode=result.get("correction_mode"),
        corrections=_project_corrections(result),
        summary=str(summary) if summary else None,
    )
