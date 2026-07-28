from __future__ import annotations

from typing import Final

from trainer.demo_api.sequence import default_snapshot
from trainer.demo_api.sim_contracts import (
    SessionEventInput,
    SessionEventOutput,
    SessionReviewOutput,
    SessionStartInput,
    SessionStartOutput,
)
from trainer.scripts.session_coaching import SessionSummary, compute_risk_flags

STANDOFF_LIMIT_MM: Final[float] = 320.0
ANGLE_MIN_DEG: Final[float] = 60.0
_MAX_ACTION_ITEMS: Final[int] = 3


def build_start_output(payload: SessionStartInput) -> SessionStartOutput:
    zones = [zone.zone_id for zone in default_snapshot(payload.level_id).zones]
    guidance = (
        f"레벨 {payload.level_id} 권장 도장 순서: {' → '.join(zones)}. "
        "상부에서 하부 순으로 진행하십시오."
    )
    return SessionStartOutput(
        session_id=payload.session_id,
        recommended_flow=zones,
        flow_source="rule",
        guidance=guidance,
    )


def build_event_output(session_id: str, payload: SessionEventInput) -> SessionEventOutput:
    if payload.user_request is not None and payload.user_request.strip():
        request = payload.user_request.strip()
        return SessionEventOutput(
            session_id=session_id,
            event_id=payload.event_id,
            kind="rag",
            message=f"질의 응답(스텁): '{request}' 관련 문서 근거를 확인하세요.",
            anomaly=None,
        )

    gun = payload.telemetry.gun_metrics if payload.telemetry is not None else None
    if gun is not None:
        standoff_breach = gun.standoff_mm > STANDOFF_LIMIT_MM
        angle_breach = gun.angle_deg < ANGLE_MIN_DEG
        if standoff_breach or angle_breach:
            anomaly: dict[str, object] = {
                "type": "spray_param_out_of_band",
                "standoff_mm": gun.standoff_mm,
                "angle_deg": gun.angle_deg,
                "severity": "high" if standoff_breach and angle_breach else "medium",
            }
            return SessionEventOutput(
                session_id=session_id,
                event_id=payload.event_id,
                kind="anomaly",
                message="스프레이 파라미터 이탈 감지(스텁).",
                anomaly=anomaly,
            )

    return SessionEventOutput(
        session_id=session_id,
        event_id=payload.event_id,
        kind="ack",
        message="이벤트 누적됨.",
        anomaly=None,
    )


def build_review_output(payload: SessionSummary) -> SessionReviewOutput:
    flags = compute_risk_flags(payload.metrics)
    if flags:
        summary = f"세션 {payload.session_id} 리뷰: 위험 플래그 {len(flags)}건 감지."
        action_items = [
            {"action": f"{flag['flag']} 개선 필요", "priority": str(flag["severity"])}
            for flag in flags[:_MAX_ACTION_ITEMS]
        ]
    else:
        summary = f"세션 {payload.session_id} 리뷰: 위험 플래그 없음. 기준 충족."
        action_items = [{"action": "현재 작업 품질 유지.", "priority": "low"}]

    return SessionReviewOutput(
        session_id=payload.session_id,
        summary=summary,
        risk_flags=flags,
        action_items=action_items,
        citations=[],
    )
