"""v2.5 공개 API — 세션·사용자·리포트·이상 알림·테스트 (명세서 15개 중 12개).

정본은 `docs/XR Painting Simulator AI Co-Pilot Service API 정의서 - 대화형 API 명세서.pdf`.
남은 3개는 `telemetry_router`(시퀀스 2종)와 `copilot_router`(질의)에 있다.

**분석 엔진이 없는 곳은 임의 값을 만들지 않는다.** 명세서 p.15가 그 경우를 규정한다 —
"실제 분석이 아직 없으면 상태는 `pending`, 각 분석 영역은 `unavailable`로 반환됩니다."
세션·사용자·리포트 상태는 실제로 저장하고 조회하므로 관측된 사실이다.
"""

from __future__ import annotations

import json
import hashlib
import logging
import os
import re
import time
import zlib
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Annotated, Any, Final

import anyio
import httpx
from anyio import to_thread
from fastapi import APIRouter, BackgroundTasks, File, Form, Path as ApiPath, Query, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from trainer.demo_api.anomaly_stream import HEARTBEAT_SEC, POLL_SEC, AnomalyBroadcaster
from trainer.demo_api.v1_contracts import (
    SessionReport,
    ApiResponse,
    CoatingReport,
    PoseReport,
    ReportDetail,
    ReportStatusData,
    SequenceReport,
    SessionCompleteData,
    SessionEndRequest,
    SessionResponse,
    SessionStartRequest,
    SessionStateData,
    UserCreateRequest,
    UserProfile,
    report_example,
)
from trainer.demo_api import coating_assessment
from trainer.demo_api.coating_progress import CoatingProgressEngine
from trainer.demo_api.session_analysis import analyze_session, inflate_csv
from trainer.demo_api.report_jobs import (
    CSV_UPLOAD_LIMIT, PNG_UPLOAD_LIMIT, InputTooLarge, ReportJobs, save_artifact,
)
from trainer.demo_api.session_coating import SessionCoating, analyze_session_coating
from trainer.demo_api.sequence_store import SequenceStore
from trainer.demo_api.sequence_analysis import SequenceAnalysisEngine
from trainer.demo_api.v1_store import SessionConflict, V1Store

logger = logging.getLogger(__name__)

_V1: Final[str] = "/api/v1"
_ENGINE_MISSING: Final[str] = "분석 엔진이 연결되지 않았습니다."

# p.21-23: 테스트 엔드포인트 3종이 주입하는 이상 종류.
_TEST_CASES: Final[dict[str, tuple[str, str]]] = {
    "normal": ("none", "정상 가로 수평 패스입니다. 이대로 유지하세요."),
    "abnormal-vertical": ("direction_deviation", "가로 모드인데 세로로 분사했습니다. 가로로 도포하세요."),
    "abnormal-horizontal": ("distance_deviation", "상하 요동과 이격거리 오차가 큽니다. 자세를 고정하세요."),
}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ok(data: Any, *, message: str = "Operation completed successfully") -> JSONResponse:
    return JSONResponse(
        status_code=200, content=ApiResponse(data=data, message=message).model_dump(mode="json")
    )


def _error(code: int, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=code,
        content=ApiResponse(status="ERROR", code=code, message=message, data=None).model_dump(mode="json"),
    )


# 영역을 못 내는 이유는 서로 다르다. 한 코드로 뭉치면 운영자가 엉뚱한 데를 찾는다.
# 도막 코드는 이제 도막 분석 결과가 저장되지 않은 리포트(도막 엔진 없이 뜬 서버, 분석 실패)에만 나온다.
# 값은 공개 사유 코드라 바꾸지 않는다. 순서는 기준(recommended_flow)이 없다.
_COATING_INDEX_UNDEFINED: Final = "coating_engine_pending_integration"
_FLOW_MISSING: Final = "recommended_flow_missing"

_ANALYSIS_FAILED: Final = "analysis_failed"
_ANALYSIS_FAILED_GUIDE: Final = "분석을 마치지 못했습니다. 입력 자료와 분석 서비스 상태를 확인해 주세요."

def _unavailable_report(
    *,
    report_id: str,
    session_id: str,
    status: str,
    generated_at: str,
    user_id: str | None = None,
    scenario_id: str | None = None,
) -> ReportDetail:
    """분석 결과가 아직 없을 때의 리포트. 대기만으로 엔진 장애를 추정하지 않는다.

    모양의 정본은 `trainer/demo_api/report_sample.json` — 머리 필드는 다른 v2.5 응답과
    같은 규약이고 본체는 `report` 아래다.
    """
    pending = status == "pending"
    guide = "분석 결과가 아직 저장되지 않았습니다. 현재 진행 여부와 지연 사유는 확인되지 않았습니다." if pending else _ENGINE_MISSING
    return ReportDetail(
        session_id=session_id,
        user_id=user_id,
        scenario_id=scenario_id,
        generated_at=generated_at,
        report=SessionReport(
            report_id=report_id,
            report_status=status,  # type: ignore[arg-type]
            coating_report=CoatingReport(
                analysis_status="unavailable", reason_code="report_pending" if pending else _COATING_INDEX_UNDEFINED, guideline=guide
            ),
            sequence_report=SequenceReport(
                analysis_status="unavailable", reason_code="report_pending" if pending else _FLOW_MISSING, guideline=guide
            ),
            pose_report=PoseReport(analysis_status="unavailable", reason_code="report_pending" if pending else None, guideline=guide),
            llm_briefing=None,
        ),
    )


async def _sse_body(
    broadcaster: AnomalyBroadcaster, session_id: str, last_event_id: int
) -> AsyncIterator[bytes]:
    """이상이 있을 때만 이벤트를, 없으면 heartbeat 를 보낸다(p.10).

    `POLL_SEC` 마다 새 이벤트를 확인하고 heartbeat 는 `HEARTBEAT_SEC` 마다 보낸다.

    `TestClient`로 이 생성기를 끝까지 돌리면 멈추지 않는다 — 순수 생성기를 anyio 로
    직접 시험한다(AGENTS.md).
    """
    cursor = last_event_id
    next_beat = 0.0  # 첫 바퀴에 바로 한 번 보낸다 — 구독이 붙었음을 알린다.
    while True:
        for event in broadcaster.since(session_id, last_event_id=cursor):
            cursor = event.id
            body = json.dumps(
                {
                    "session_id": event.session_id,
                    "sequence_id": event.sequence_id,
                    "anomaly_type": event.anomaly_type,
                    **event.payload,
                },
                ensure_ascii=False,
            )
            yield f"id: {event.id}\nevent: anomaly_detection\ndata: {body}\n\n".encode("utf-8")
        # comment heartbeat. 데이터 이벤트가 아니므로 클라이언트 상태를 바꾸지 않는다.
        # 폴링 간격과는 별개다 — 둘을 묶으면 이상이 최대 HEARTBEAT_SEC 만큼 밀려
        # 명세서 p.4의 "실시간 Push"가 되지 않는다.
        now = time.monotonic()
        if now >= next_beat:
            yield b": keep-alive\n\n"
            next_beat = now + HEARTBEAT_SEC
        await anyio.sleep(POLL_SEC)


def _report(row: Any) -> ReportDetail:
    """Overlay every completed analysis area without inventing unavailable values."""
    payload = row.payload if isinstance(row.payload, dict) else {}
    status = "pending" if row.report_status == "PENDING" else row.report_status
    detail = _unavailable_report(
        report_id=row.report_id,
        session_id=row.session_id,
        status=status,
        generated_at=row.requested_at,
        user_id=payload.get("user_id"),
        scenario_id=payload.get("scenario_id"),
    )
    analysis = payload.get("analysis")
    if status == "pending" and getattr(row, "legacy_inputs_unavailable", False):
        for area in (detail.report.coating_report, detail.report.sequence_report, detail.report.pose_report):
            area.reason_code = "legacy_report_inputs_unavailable"
            area.guideline = "이전 리포트의 입력 자료 참조 확인 불가. 원본 CSV·높이맵 재업로드 권장."
    if status == "failed":
        error = analysis.get("error") if isinstance(analysis, dict) else None
        reason = error if error in {"report_input_unavailable", "session_analysis_failed"} else _ANALYSIS_FAILED
        guide = ("리포트 입력 자료를 읽지 못했습니다. 업로드 자료와 저장소 상태를 확인해 주세요."
                 if reason == "report_input_unavailable" else _ANALYSIS_FAILED_GUIDE)
        detail.report.coating_report = CoatingReport(
            analysis_status="unavailable", reason_code=reason, guideline=guide
        )
        detail.report.sequence_report = SequenceReport(
            analysis_status="unavailable", reason_code=reason, guideline=guide
        )
        detail.report.pose_report = PoseReport(
            analysis_status="unavailable", reason_code=reason, guideline=guide
        )
    elif isinstance(analysis, dict):
        if isinstance(analysis.get("pose_report"), dict):
            detail.report.pose_report = PoseReport.model_validate(analysis["pose_report"])
        if isinstance(analysis.get("sequence_report"), dict):
            detail.report.sequence_report = SequenceReport.model_validate(analysis["sequence_report"])
        if isinstance(analysis.get("coating_report"), dict):
            detail.report.coating_report = CoatingReport.model_validate(analysis["coating_report"])
        if isinstance(analysis.get("llm_briefing"), str):
            detail.report.llm_briefing = analysis["llm_briefing"]
    if isinstance(analysis, dict):
        if isinstance(analysis.get("generated_at"), str):
            detail.generated_at = analysis["generated_at"]
        if isinstance(analysis.get("latency_ms"), int):
            detail.latency_ms = analysis["latency_ms"]
    return detail


def _briefing(
    *,
    pose_report: dict[str, Any],
    sequence_report: dict[str, Any] | None,
    coating_report: dict[str, Any] | None = None,
    counts: dict[str, int] | None = None,
    coating_zones: list[dict[str, Any]] | None = None,
) -> str | None:
    """Create a fast, deterministic narrative from available structured evidence."""
    parts: list[str] = []
    if sequence_report and sequence_report.get("analysis_status") == "available":
        adherence = sequence_report.get("adherence_rate")
        deviations = sequence_report.get("deviation_count")
        if isinstance(adherence, (int, float)):
            deviation = f"(이탈 {deviations}회)" if isinstance(deviations, int) else ""
            parts.append(f"순서 준수 {adherence * 100:.1f}%{deviation}.")
        direction = sequence_report.get("direction_adherence_rate")
        if direction is None and isinstance(sequence_report.get("direction_summary"), dict):
            direction = sequence_report["direction_summary"].get("direction_adherence_rate")
        if isinstance(direction, (int, float)):
            parts.append(f"방향 준수 {direction * 100:.1f}%.")
    if pose_report.get("analysis_status") == "available":
        similarity = pose_report.get("average_similarity_score")
        scope: list[str] = []
        evaluated = pose_report.get("evaluated_sequence_count")
        if isinstance(evaluated, int):
            scope.append(f"평가 {evaluated}회")
        for key, label in (("not_scored", "미판정"), ("no_zone", "Zone 미지정")):
            excluded = (counts or {}).get(key)
            if isinstance(excluded, int) and excluded > 0:
                scope.append(f"{label} {excluded}회 제외")
        if isinstance(similarity, (int, float)):
            denominator = f"({', '.join(scope)})" if scope else ""
            parts.append(f"자세 유사도 {similarity * 100:.1f}%{denominator}.")
        corrections = [row for row in pose_report.get("priority_corrections") or []
                       if isinstance(row, dict) and row.get("correction_guidance")]
        limited_count = pose_report.get("limited_sequence_count")
        limited = (pose_report.get("confidence") == "limited"
                   or isinstance(limited_count, int) and limited_count > 0
                   or "comparison_scope_limited" in (pose_report.get("reason_codes") or []))
        if limited:
            count_note = f"(제한 {limited_count}회)" if isinstance(limited_count, int) and limited_count > 0 else ""
            parts.append(f"자세 참고용·비교 제한{count_note}.")
            if not corrections:
                parts.append("교정 없음은 정상 보장이 아닙니다.")
        if corrections:
            top = min(corrections, key=lambda row: row.get("priority", 999))
            zone = f"Zone {top['zone_id']} " if top.get("zone_id") is not None else ""
            parts.append(f"우선 교정: {zone}{top['correction_guidance']}")
    if coating_report and coating_report.get("analysis_status") == "available":
        band = coating_report.get("normal_band_um")
        normal = (f"(정상 {band[0]:g}~{band[1]:g}µm)" if isinstance(band, list) and len(band) == 2
                  and all(isinstance(value, (int, float)) for value in band) else "")
        worst = coating_report.get("worst_zones") or []
        if worst and isinstance(worst[0], dict):
            top = worst[0]
            status = {"under_app": "저도막", "over_app": "과도막"}.get(top.get("status"), "")
            value = top.get("thickness_um")
            thickness = f" {value:g}µm" if isinstance(value, (int, float)) else ""
            parts.append(f"도막 Zone {top['zone_id']} {status}{thickness}{normal}.")
            if top.get("action"):
                parts.append(str(top["action"]))
        else:
            parts.append(f"도막 분석 완료{normal}.")
        unpainted = [row for row in coating_zones or []
                     if row.get("unpainted_fraction") == 1 and row.get("zone_id") is not None]
        if unpainted:
            labels = [f"Map {row['map_num']}·Zone {row['zone_id']}" if row.get("map_num") is not None
                      else f"Zone {row['zone_id']}" for row in unpainted[:3]]
            more = " 등" if len(unpainted) > len(labels) else ""
            parts.append(f"미도장 {len(unpainted)}개({', '.join(labels)}{more}).")
    elif parts:
        reason = coating_report.get("reason_code") if coating_report else None
        note = {
            "reference_unavailable": "도막 미평가(Zone 기준 없음).",
            "coating_reference_unavailable": "도막 미평가(Zone 기준 없음).",
            "reference_context_unavailable": "도막 미평가(공정·콘텐츠 기준 없음).",
            "heatmap_unavailable": "도막 미평가(높이맵 없음).",
            "coating_not_applicable": "도막 평가 제외(블라스팅).",
            "coating_analysis_failed": "도막 분석 실패.",
        }.get(reason, "도막 미평가.")
        parts.append(note)
    return " ".join(parts) if parts else None


def create_v1_router(
    *,
    store: V1Store | None = None,
    broadcaster: AnomalyBroadcaster | None = None,
    sequences: SequenceStore | None = None,
    mcp_base_url: str | None = None,
    transport: httpx.BaseTransport | None = None,
    sequence_engine: SequenceAnalysisEngine | None = None,
    coating_engine: CoatingProgressEngine | None = None,
) -> APIRouter:
    router = APIRouter(prefix=_V1)
    state = store if store is not None else V1Store()
    anomalies = broadcaster if broadcaster is not None else AnomalyBroadcaster()
    stored_sequences = sequences if sequences is not None else SequenceStore()
    mcp_base = mcp_base_url or os.environ.get("MCP_BASE_URL", "http://localhost:8100")

    def _settled(report_id: str) -> Any:
        """Polling observes state. Age alone is not evidence of a failed worker."""
        return state.get_completion(report_id)

    def _coating(
        bodies: list[dict[str, Any]], *, scenario_id: str, height_map: bytes | None, height_map_name: str | None,
        target_thickness_um: float | None = None,
        min_thickness_um: float | None = None,
        max_thickness_um: float | None = None,
        tool_type: str | None = None,
    ) -> SessionCoating | None:
        """세션 끝 도막(`session_coating`). 여기서 터져도 자세·순서 결과는 남긴다 — 리포트를 failed로 만들지 않는다."""
        if coating_engine is None:
            return None
        try:
            return analyze_session_coating(
                scenario_id=scenario_id, bodies=bodies, engine=coating_engine,
                height_map=height_map, height_map_name=height_map_name,
                target_thickness_um=target_thickness_um,
                min_thickness_um=min_thickness_um, max_thickness_um=max_thickness_um,
                tool_type=tool_type,
            )
        except Exception:  # noqa: BLE001 - 도막만 unavailable 로 낸다.
            logger.exception("session coating analysis failed scenario=%s", scenario_id)
            return SessionCoating({
                "analysis_status": "unavailable", "reason_code": "coating_analysis_failed",
                "guideline": "도막 분석 중 오류가 나서 결과를 내지 못했습니다.",
            })

    def _analyze(
        report_id: str,
        *,
        session_id: str,
        user_id: str,
        scenario_id: str,
        csv_bytes: bytes,
        height_map: bytes | None = None,
        height_map_name: str | None = None,
        target_thickness_um: float | None = None,
    ) -> None:
        """세션 종합 분석 — 자세(`session_analysis`)·순서·도막(`session_coating`). 응답을 보낸 뒤 스레드에서 돈다."""
        try:
            result = analyze_session(
                session_id=session_id, user_id=user_id, scenario_id=scenario_id, csv_bytes=csv_bytes,
                store=stored_sequences, mcp_base_url=mcp_base, transport=transport,
            )
            sequence_report = (
                sequence_engine.report_bodies(result.source_bodies, scenario_id=scenario_id)
                if sequence_engine is not None else
                SequenceReport(analysis_status="unavailable", reason_code=_FLOW_MISSING, guideline=_ENGINE_MISSING).model_dump(mode="json")
            )
            pose_report = result.pose_report()
            session = state.get_session(session_id)
            coating = _coating(
                result.source_bodies, scenario_id=scenario_id,
                height_map=height_map, height_map_name=height_map_name,
                target_thickness_um=target_thickness_um,
                **coating_assessment.session_limits(session),
                tool_type=session.tool_type if session is not None else None,
            )
            coating_report = coating.report if coating is not None else None
            state.finish_completion(
                report_id, status="complete",
                analysis={
                    "generated_at": _now(),
                    "pose_report": pose_report,
                    "sequence_report": sequence_report,
                    **({"coating_report": coating_report} if coating_report is not None else {}),
                    # 질의(pose_recomm)가 구역 답에 쓰는 구역별 도막 판정. 리포트 본문에는 싣지 않는다.
                    "coating_zones": coating.zones if coating is not None else [],
                    "llm_briefing": _briefing(
                        pose_report=pose_report, sequence_report=sequence_report, coating_report=coating_report,
                        counts=result.counts, coating_zones=coating.zones if coating is not None else None,
                    ),
                    "counts": result.counts,
                    "sequences": result.sequences,
                },
            )
        except Exception:  # noqa: BLE001 - 어떤 분석 영역 실패도 pending에 갇히지 않아야 한다.
            logger.exception("session analysis failed %s report=%s", session_id, report_id)
            state.finish_completion(
                report_id, status="failed",
                analysis={"generated_at": _now(), "error": "session_analysis_failed"},
            )
            return
        logger.info("session analysis %s report=%s %s", session_id, report_id, result.counts)

    jobs = ReportJobs(state, _analyze)
    router.recover_report_jobs = jobs.recover_pending  # type: ignore[attr-defined]

    # --- 1. 세션 관리 (p.1-3) ---------------------------------------------

    @router.post(
        "/sessions/start", response_model=SessionResponse, tags=["1. 세션 관리"],
        summary="1-1. 훈련 세션 시작",
        description=(
            "VR 훈련 콘텐츠가 시작될 때 한 번 호출합니다. user_id가 처음이면 사용자를 생성하고, "
            "기존 사용자면 숙련도와 주 사용 손을 갱신한 뒤 세션을 ACTIVE 상태로 저장합니다. "
            "스프레이 정상 도막은 기본 160~200µm입니다. target_thickness_min_um·target_thickness_max_um으로 "
            "경계값을 포함한 정상범위를 지정할 수 있습니다. 기존 target_thickness_um 단일 값만 보내면 "
            "그 값부터 1.25배까지 정상으로 봅니다(160 → 160~200). 모두 생략해도 동작합니다. "
            "응답 data.first_zone_id·first_zone_guidance가 선택 공정·콘텐츠의 등록된 전문가 첫 Zone을 안내합니다. "
            "tool_type은 생략하면 spray_gun이며 블라스팅 시작 안내에는 blasting_gun을 보냅니다. "
            "실전·Unit 등 등록된 순서 기준이 없는 콘텐츠는 첫 Zone을 추정하지 않고 null을 반환합니다."
        ),
    )
    async def sessions_start(body: SessionStartRequest) -> JSONResponse:
        try:
            row, session_created = await to_thread.run_sync(
                lambda: state.start_session_with_created(
                    session_id=body.session_id, user_id=body.user_id, scenario_id=body.scenario_id,
                    started_at=body.started_at,
                    target_thickness_um=(body.target_thickness_min_um if body.target_thickness_min_um is not None else body.target_thickness_um),
                    target_thickness_min_um=body.target_thickness_min_um,
                    target_thickness_max_um=body.target_thickness_max_um,
                    tool_type=body.tool_type,
                    creation_origin="real", creation_path="/api/v1/sessions/start",
                    # 생략해도 계약 기본값 160.0이 저장되므로, 세션이 목표를 정했는지를 따로 남긴다.
                    target_basis=(
                        "session_target"
                        if ("target_thickness_um" in body.model_fields_set and body.target_thickness_um is not None)
                        or body.target_thickness_min_um is not None
                        else "default_target"
                    ),
                )
            )
        except SessionConflict as exc:
            return _error(409, str(exc))
        created = await to_thread.run_sync(
            lambda: state.upsert_user(
                user_id=body.user_id,
                skill_level=body.worker.skill_level,
                dominant_hand=body.worker.handedness,
                created_at=_now(),
            )
        )
        if session_created:
            anomalies.reset(body.session_id)
        logger.info("session start %s user=%s created=%s", body.session_id, body.user_id, created)
        process = "blasting" if (row.tool_type or body.tool_type) == "blasting_gun" else "spray"
        plan = (sequence_engine.reference_plan(scenario_id=row.scenario_id, process=process)
                if sequence_engine is not None else None)
        first_zone = str(plan["flow"][0]) if plan and plan.get("flow") else None
        guidance = {
            "analysis_status": "available" if first_zone else "not_available",
            "reason_code": None if first_zone else "sequence_reference_unavailable",
            "process": process, "zone_id": first_zone,
            "recommended_flow": plan["flow"] if plan else [],
            "reference_basis": "measured_expert_runs" if first_zone else None,
            "message": (f"{first_zone}구역에서 시작하세요." if first_zone else
                        "등록된 작업 순서 기준이 없습니다."),
        }
        lower, upper = coating_assessment.resolve_band(row.target_thickness_um, **coating_assessment.session_limits(row))
        return _ok(
            SessionStateData(
                session_id=row.session_id, user_id=row.user_id, scenario_id=row.scenario_id,
                session_status=row.session_status, started_at=row.started_at, ended_at=row.ended_at,
                user_created=created,
                target_thickness_um=row.target_thickness_um,
                target_thickness_min_um=lower, target_thickness_max_um=upper,
                first_zone_id=first_zone, first_zone_guidance=guidance,
            ).model_dump(mode="json"),
            message=f"훈련 시작. {guidance['message']}",
        )

    @router.post(
        "/sessions/end", response_model=SessionResponse, tags=["1. 세션 관리"],
        summary="1-2. 훈련 세션 마감",
        description="사용자가 콘텐츠를 정상 종료하거나 중도 종료했을 때 호출합니다. 시작 API에서 사용한 session_id를 그대로 보내며, 서버는 종료 시각과 COMPLETED 상태를 기록합니다.",
    )
    async def sessions_end(body: SessionEndRequest) -> JSONResponse:
        try:
            row = await to_thread.run_sync(
                lambda: state.end_session(
                    session_id=body.session_id, user_id=body.user_id,
                    scenario_id=body.scenario_id, ended_at=body.ended_at,
                )
            )
        except SessionConflict as exc:
            return _error(409, str(exc))
        if body.worker is not None and body.worker.skill_level and body.worker.handedness:
            _ = await to_thread.run_sync(
                lambda: state.upsert_user(
                    user_id=body.user_id,
                    skill_level=body.worker.skill_level,  # type: ignore[union-attr]
                    dominant_hand=body.worker.handedness,  # type: ignore[union-attr]
                    created_at=_now(),
                )
            )
        return _ok(
            SessionStateData(
                session_id=row.session_id, user_id=row.user_id, scenario_id=row.scenario_id,
                session_status="COMPLETED", started_at=row.started_at, ended_at=row.ended_at,
                user_created=False,
                target_thickness_um=row.target_thickness_um,
                target_thickness_min_um=coating_assessment.resolve_band(row.target_thickness_um, **coating_assessment.session_limits(row))[0],
                target_thickness_max_um=coating_assessment.resolve_band(row.target_thickness_um, **coating_assessment.session_limits(row))[1],
            ).model_dump(mode="json")
        )

    # --- 5. 최종 평가와 리포트 (p.11-16) ----------------------------------

    @router.post("/sessions/complete", response_model=ApiResponse, tags=["5. 최종 평가·리포트"], summary="5-1. 훈련 최종 종료 및 비동기 분석 요청")
    async def sessions_complete(
        background: BackgroundTasks,
        session_id: Annotated[str, Form(description="sessions/start와 동일한 세션 식별자", examples=["3_11000346_1"])],
        user_id: Annotated[str, Form(description="sessions/start에서 사용한 동일 훈련자 식별자입니다.", examples=["1234"])],
        scenario_id: Annotated[str, Form(description="session_id의 첫 '_' 앞 콘텐츠 식별자", examples=["3"])],
        vr_score: Annotated[int, Form(ge=0, le=100, description="시뮬레이터가 산출한 최종 VR 수행 점수(0~100점)입니다.", examples=[54])],
        is_pass: Annotated[bool, Form(description="시뮬레이터 기준 훈련 통과 여부입니다. 통과는 true, 미통과는 false입니다.", examples=[False])],
        avg_thickness_um: Annotated[float, Form(ge=0, allow_inf_nan=False, description="세션 전체 피도면의 평균 도막 두께(µm)입니다.", examples=[138.91])],

        telemetry_csv: Annotated[UploadFile, File(description="훈련 세션 전체 프레임과 Zone ID를 담은 CSV 또는 CSV.GZ 원본 파일입니다.")],
        avg_speed_mms: Annotated[float | None, Form(ge=0, allow_inf_nan=False, description="세션 전체 도구 이동 속도 평균(mm/s)입니다. 산출하지 않으면 생략합니다.", examples=[480.0])] = None,
        avg_distance_mm: Annotated[float | None, Form(ge=0, allow_inf_nan=False, description="세션 전체 노즐-피도면 이격거리 평균(mm)입니다. 산출하지 않으면 생략합니다.", examples=[255.0])] = None,
        height_map_image: Annotated[UploadFile | None, File(description="세션 종료 시점의 전체 누적 도막 높이맵 PNG입니다. 도막 분석을 사용하지 않으면 생략합니다.")] = None,
    ) -> JSONResponse:
        """최신 TSB multipart 전송을 받고 `report_id`를 부여해 `pending`으로 시작한다.

        세션 평균 속도·이격거리는 2026-10-08 샘플에 없으므로 선택값이다. 시퀀스별
        속도·이격거리 평가는 `/copilot/analyze-sequence` 입력을 사용한다.
        """
        if not state.report_storage_available:
            return _error(503, "report storage is unavailable; retry the request")
        session = await to_thread.run_sync(lambda: state.get_session(session_id))
        if session is not None and (session.user_id, session.scenario_id) != (user_id, scenario_id):
            return _error(409, "session user_id or scenario_id does not match")
        csv_raw = await telemetry_csv.read(CSV_UPLOAD_LIMIT + 1)
        if len(csv_raw) > CSV_UPLOAD_LIMIT:
            return _error(413, "telemetry CSV exceeds upload size limit")
        try:
            # Validate the inflated size before accepting a gzip bomb. Invalid legacy CSV
            # remains compatible: stored sequences can still supply analysis input.
            await to_thread.run_sync(lambda: inflate_csv(csv_raw))
        except InputTooLarge as exc:
            return _error(413, str(exc))
        except (OSError, EOFError, zlib.error):
            pass
        csv_bytes = len(csv_raw)
        image_raw = await height_map_image.read(PNG_UPLOAD_LIMIT + 1) if height_map_image is not None else None
        if image_raw is not None and len(image_raw) > PNG_UPLOAD_LIMIT:
            return _error(413, "height map exceeds upload size limit")
        image_bytes = len(image_raw) if image_raw is not None else None
        image_name = None
        if height_map_image is not None:
            # MapNN in this metadata chooses the painted surface. Retain it across
            # recovery, while never using any client filename as a filesystem path.
            basename = (height_map_image.filename or "height_map.png").replace("\\", "/").rsplit("/", 1)[-1]
            image_name = re.sub(r"[^\w.\- ]", "_", basename)[-255:] or "height_map.png"
        payload = {
            "user_id": user_id, "scenario_id": scenario_id, "vr_score": vr_score,
            "is_pass": is_pass, "avg_thickness_um": avg_thickness_um,
            "avg_speed_mms": avg_speed_mms, "avg_distance_mm": avg_distance_mm,
            "telemetry_csv_bytes": csv_bytes, "height_map_image_bytes": image_bytes,
            # 시작 때 목표를 생략한 세션은 None으로 남겨 리포트가 기본 목표 판정임을 밝히게 한다.
            "target_thickness_um": coating_assessment.session_target(session),
            "height_map_name": image_name,
        }
        try:
            refs = {"telemetry": await to_thread.run_sync(lambda: save_artifact(state.artifact_dir, csv_raw))}
            if image_raw is not None:
                refs["height_map"] = await to_thread.run_sync(lambda: save_artifact(state.artifact_dir, image_raw))
        except OSError:
            logger.exception("report input persistence failed session=%s", session_id)
            return _error(503, "report input could not be persisted; retry the request")
        payload["artifacts"] = refs
        fingerprint = hashlib.sha256(json.dumps(
            {"session_id": session_id, **payload}, sort_keys=True, ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")).hexdigest()
        completion = await to_thread.run_sync(
            lambda: state.create_completion(
                session_id=session_id,
                requested_at=_now(),
                payload=payload, fingerprint=fingerprint,
            )
        )
        logger.info("session complete %s report=%s csv=%dB", session_id, completion.report_id, csv_bytes)
        # 세션 종합 분석은 응답 뒤에 돈다(p.12 "비동기 분석"). 결과는 리포트 폴링으로 본다.
        if completion.report_status == "pending":
            background.add_task(to_thread.run_sync, lambda: jobs.run(completion.report_id))
        return _ok(
            SessionCompleteData(
                report_id=completion.report_id, session_id=session_id,
                report_status=completion.report_status, telemetry_csv_bytes=csv_bytes,
                height_map_image_bytes=image_bytes,
            ).model_dump(mode="json")
        )

    @router.get(
        "/sessions/{session_id}/reports/{report_id}",
        response_model=ApiResponse,
        tags=["5. 최종 평가·리포트"],
        summary="5-2. 리포트 생성 상태 폴링 조회",
        description="sessions/complete에서 받은 report_id의 처리 상태를 확인합니다. 비동기 분석 중에는 pending이며, 동일 session_id와 report_id 조합을 사용해야 다른 세션의 결과를 잘못 조회하지 않습니다.",
    )
    async def report_status(
        session_id: Annotated[str, ApiPath(description="조회할 리포트가 속한 훈련 세션 식별자입니다.", examples=["3_11000346_1"])],
        report_id: Annotated[str, ApiPath(description="sessions/complete 응답으로 발급받은 리포트 식별자입니다.", examples=["RPT-123456789abc"])],
    ) -> JSONResponse:
        row = await to_thread.run_sync(lambda: _settled(report_id))
        if row is None or row.session_id != session_id:
            return _error(404, "report not found")
        return _ok(
            ReportStatusData(
                report_id=row.report_id, session_id=row.session_id,
                report_status="pending" if row.report_status == "PENDING" else row.report_status,  # type: ignore[arg-type]
                requested_at=row.requested_at,
            ).model_dump(mode="json")
        )

    @router.get(
        "/reports/{report_id}", response_model=ApiResponse, tags=["5. 최종 평가·리포트"],
        summary="5-3. 최종 종합 리포트 상세 조회",
        description=(
            "리포트 상태 확인 후 최종 평가 내용을 조회합니다. 분석 엔진 결과가 준비되지 않은 영역은 수치를 임의 생성하지 않고 unavailable과 사유를 명시해 반환합니다.\n\n"
            "아래 응답 예시는 세 영역(`coating_report`·`sequence_report`·`pose_report`)과 `llm_briefing`이 모두 채워진 **최종 형식**입니다. "
            "값은 UI 설계용 예시이며, 실제 응답에서 아직 분석하지 못한 영역은 `analysis_status=unavailable`이고 수치는 `null`입니다."
        ),
        responses={200: {"description": "완성된 리포트 예시", "content": {"application/json": {
            "example": ApiResponse(message="Session report completed", data=report_example()).model_dump(mode="json")
        }}}},
    )
    async def report_detail(
        report_id: Annotated[str, ApiPath(description="sessions/complete 응답으로 발급받은 최종 리포트 식별자입니다.", examples=["RPT-123456789abc"])]
    ) -> JSONResponse:
        row = await to_thread.run_sync(lambda: _settled(report_id))
        if row is None:
            return _error(404, "report not found")
        return _ok(_report(row).model_dump(mode="json"))

    # --- 6. 사용자 관리 (p.17-20) -----------------------------------------

    @router.get(
        "/users", response_model=ApiResponse, tags=["6. 사용자 관리"], summary="6-1. 훈련자 전체 목록 조회",
        description="등록된 훈련자 프로필을 페이지 단위로 조회합니다. skip은 건너뛸 행 수, limit은 한 번에 받을 최대 행 수이며 운영 UI의 사용자 선택 목록에 사용할 수 있습니다.",
    )
    async def users_list(
        skip: Annotated[int, Query(ge=0, description="정렬된 사용자 목록에서 건너뛸 레코드 수입니다.")] = 0,
        limit: Annotated[int, Query(ge=1, le=1000, description="한 번에 반환할 최대 사용자 수입니다(1~1000).")] = 100,
    ) -> JSONResponse:
        rows = await to_thread.run_sync(lambda: state.list_users(skip=skip, limit=limit))
        return _ok([UserProfile(**row.__dict__).model_dump(mode="json") for row in rows])

    @router.post(
        "/users", response_model=ApiResponse, tags=["6. 사용자 관리"], summary="6-2. 신규 훈련자 계정 직접 등록",
        description="훈련 시작 전 사용자를 미리 등록할 때 사용합니다. user_id만 필수이고 나머지는 기본값을 사용할 수 있으며, 이미 존재하는 ID면 중복 생성 대신 프로필을 안전하게 갱신합니다.",
    )
    async def users_create(body: UserCreateRequest) -> JSONResponse:
        created = await to_thread.run_sync(
            lambda: state.upsert_user(
                user_id=body.user_id, skill_level=body.skill_level,
                dominant_hand=body.dominant_hand, name=body.name,
                age=body.age, gender=body.gender, created_at=_now(),
            )
        )
        row = await to_thread.run_sync(lambda: state.get_user(body.user_id))
        assert row is not None  # 방금 upsert 했다
        payload = UserProfile(**row.__dict__).model_dump(mode="json")
        return JSONResponse(
            status_code=200,
            content=ApiResponse(
                message="Operation completed successfully" if created else "User already existed; profile updated",
                data=payload,
            ).model_dump(mode="json"),
        )

    @router.get(
        "/users/{user_id}", response_model=ApiResponse, tags=["6. 사용자 관리"], summary="6-3. 훈련자 프로필 조회",
        description="한 명의 훈련자 프로필을 user_id로 조회합니다. VR·Replay 화면에서 숙련도와 주 사용 손을 표시하거나 세션 시작 전 등록 여부를 확인할 때 사용합니다.",
    )
    async def users_get(
        user_id: Annotated[str, ApiPath(description="조회할 훈련자의 고유 식별자입니다.", examples=["1234"])]
    ) -> JSONResponse:
        row = await to_thread.run_sync(lambda: state.get_user(user_id))
        if row is None:
            return _error(404, "user not found")
        return _ok(UserProfile(**row.__dict__).model_dump(mode="json"))

    # --- 4. 이상 알림 SSE (p.10) ------------------------------------------

    @router.get(
        "/stream/anomalies/{session_id}", tags=["4. 실시간 알림"], summary="4-1. 실시간 이상 탐지 SSE 이벤트 스트림",
        description="진행 중인 세션의 이상 감지 알림을 Server-Sent Events로 계속 수신합니다. 클라이언트는 한 번 연결한 뒤 anomaly_detection 이벤트를 처리하고, keep-alive 주석은 화면 상태 변경 없이 무시합니다.",
    )
    async def stream_anomalies(
        session_id: Annotated[str, ApiPath(description="실시간 이상 알림을 구독할 훈련 세션 식별자입니다.", examples=["3_11000346_1"])],
        request: Request,
    ) -> StreamingResponse:
        raw = request.headers.get("last-event-id", "0")
        try:
            cursor = int(raw)
        except ValueError:
            cursor = 0
        return StreamingResponse(
            _sse_body(anomalies, session_id, cursor),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # --- 7. 테스트 전용 (p.21-24) -----------------------------------------

    for case, (anomaly_type, message) in _TEST_CASES.items():

        def _make(case: str = case, anomaly_type: str = anomaly_type, message: str = message):
            def test_context(session_id: str) -> dict[str, Any]:
                return {
                    "input_used": {
                        "query_parameters": {"session_id": session_id},
                        "case": case, "sequence_id": f"TEST-{case}",
                        "anomaly_type": None if anomaly_type == "none" else anomaly_type,
                        "payload": {"message": message},
                        "publish_anomaly": anomaly_type != "none",
                    },
                    "input_summary": "시뮬레이터 입력이나 분석 엔진 없이 정해진 예제 상태와 SSE 이벤트를 만듭니다.",
                    "evaluation_focus": "normal은 이상 이벤트를 발행하지 않습니다. 이상 사례는 같은 세션·시퀀스·종류를 한 번만 발행합니다.",
                    "evaluation_mode": "synthetic_event",
                }

            @router.post(
                f"/test/sequence/{case}",
                response_model=ApiResponse,
                tags=["0. 모의 연동"],
                summary=f"0. [{case}] 시퀀스 시뮬레이션 및 SSE 방송",
                description="기존 SSE 화면 개발을 위한 단일 사례 발생기입니다. session_id는 기본값 그대로 실행할 수 있고, normal 또는 방향·이격거리 이상 이벤트가 어떻게 전달되는지 빠르게 확인합니다.",
                responses={200: {"description": "모의 상태와 실제 사용된 자동 입력을 반환합니다. 실제 자세·도막 분석은 수행하지 않습니다.",
                    "content": {"application/json": {"example": {
                        "status": "SUCCESS", "code": 200, "message": "OK", "data": {
                            "test_context": test_context("3_11000346_1"),
                            "session_id": "3_11000346_1", "sequence_id": f"TEST-{case}", "case": case,
                            "anomaly_type": None if anomaly_type == "none" else anomaly_type,
                            "anomaly_published": anomaly_type != "none", "coach_message": message,
                        },
                    }}}}},
                name=f"test_sequence_{case.replace('-', '_')}",
            )
            async def _handler(
                session_id: Annotated[str, Query(
                    description="모의 이상 알림을 귀속시킬 테스트 세션 식별자입니다.",
                    examples=["3_11000346_1"],
                )] = "3_11000346_1",
            ) -> JSONResponse:
                sequence_id = f"TEST-{case}"
                published = None
                if anomaly_type != "none":
                    published = anomalies.publish(
                        session_id=session_id, sequence_id=sequence_id,
                        anomaly_type=anomaly_type, payload={"message": message},
                    )
                return _ok(
                    {
                        "test_context": test_context(session_id),
                        "session_id": session_id,
                        "sequence_id": sequence_id,
                        "case": case,
                        "anomaly_type": None if anomaly_type == "none" else anomaly_type,
                        "anomaly_published": published is not None,
                        "coach_message": message,
                    }
                )

            return _handler

        _ = _make()

    return router
