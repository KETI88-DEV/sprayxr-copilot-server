"""v2.5 공개 API — 세션·사용자·리포트·이상 알림·테스트 (명세서 15개 중 12개).

정본은 `docs/XR Painting Simulator AI Co-Pilot Service API 정의서 - 대화형 API 명세서.pdf`.
남은 3개는 `telemetry_router`(시퀀스 2종)와 `copilot_router`(질의)에 있다.

**분석 엔진이 없는 곳은 임의 값을 만들지 않는다.** 명세서 p.15가 그 경우를 규정한다 —
"실제 분석이 아직 없으면 상태는 `pending`, 각 분석 영역은 `unavailable`로 반환됩니다."
세션·사용자·리포트 상태는 실제로 저장하고 조회하므로 관측된 사실이다.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Annotated, Any, Final

import anyio
from anyio import to_thread
from fastapi import APIRouter, File, Form, Query, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from trainer.demo_api.anomaly_stream import HEARTBEAT_SEC, POLL_SEC, AnomalyBroadcaster
from trainer.demo_api.v1_contracts import (
    ApiResponse,
    CoatingReport,
    PoseReport,
    ReportDetail,
    ReportStatusData,
    SequenceReport,
    SessionCompleteData,
    SessionEndRequest,
    SessionStartRequest,
    SessionStateData,
    UserCreateRequest,
    UserProfile,
)
from trainer.demo_api.v1_store import V1Store

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


def _ok(data: Any) -> JSONResponse:
    return JSONResponse(
        status_code=200, content=ApiResponse(data=data).model_dump(mode="json")
    )


def _error(code: int, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=code,
        content=ApiResponse(status="ERROR", code=code, message=message, data=None).model_dump(mode="json"),
    )


def _unavailable_report(*, report_id: str, session_id: str, status: str, generated_at: str) -> ReportDetail:
    """엔진이 없을 때의 리포트. 수치는 `null`이고 상태가 그 사실을 말한다(p.15)."""
    return ReportDetail(
        report_id=report_id,
        session_id=session_id,
        report_status=status,  # type: ignore[arg-type]
        generated_at=generated_at,
        llm_briefing=None,
        coating_report=CoatingReport(analysis_status="unavailable", guideline=_ENGINE_MISSING),
        sequence_report=SequenceReport(analysis_status="unavailable", guideline=_ENGINE_MISSING),
        pose_report=PoseReport(analysis_status="unavailable", guideline=_ENGINE_MISSING),
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


def create_v1_router(
    *,
    store: V1Store | None = None,
    broadcaster: AnomalyBroadcaster | None = None,
) -> APIRouter:
    router = APIRouter(prefix=_V1, tags=["v2.5"])
    state = store if store is not None else V1Store()
    anomalies = broadcaster if broadcaster is not None else AnomalyBroadcaster()

    # --- 1. 세션 관리 (p.1-3) ---------------------------------------------

    @router.post("/sessions/start", response_model=ApiResponse, summary="훈련 세션 시작")
    async def sessions_start(body: SessionStartRequest) -> JSONResponse:
        created = await to_thread.run_sync(
            lambda: state.upsert_user(
                user_id=body.user_id,
                skill_level=body.worker.skill_level,
                dominant_hand=body.worker.handedness,
                created_at=_now(),
            )
        )
        row = await to_thread.run_sync(
            lambda: state.start_session(
                session_id=body.session_id,
                user_id=body.user_id,
                scenario_id=body.scenario_id,
                started_at=body.started_at,
            )
        )
        # 같은 세션 id 를 다시 시작하면 이전 이상 발행 이력을 지운다.
        anomalies.reset(body.session_id)
        logger.info("session start %s user=%s created=%s", body.session_id, body.user_id, created)
        return _ok(
            SessionStateData(
                session_id=row.session_id, user_id=row.user_id, scenario_id=row.scenario_id,
                session_status="ACTIVE", started_at=row.started_at, ended_at=row.ended_at,
                user_created=created,
            ).model_dump(mode="json")
        )

    @router.post("/sessions/end", response_model=ApiResponse, summary="훈련 세션 마감")
    async def sessions_end(body: SessionEndRequest) -> JSONResponse:
        if body.worker is not None and body.worker.skill_level and body.worker.handedness:
            _ = await to_thread.run_sync(
                lambda: state.upsert_user(
                    user_id=body.user_id,
                    skill_level=body.worker.skill_level,  # type: ignore[union-attr]
                    dominant_hand=body.worker.handedness,  # type: ignore[union-attr]
                    created_at=_now(),
                )
            )
        row = await to_thread.run_sync(
            lambda: state.end_session(
                session_id=body.session_id, user_id=body.user_id,
                scenario_id=body.scenario_id, ended_at=body.ended_at,
            )
        )
        return _ok(
            SessionStateData(
                session_id=row.session_id, user_id=row.user_id, scenario_id=row.scenario_id,
                session_status="COMPLETED", started_at=row.started_at, ended_at=row.ended_at,
                user_created=False,
            ).model_dump(mode="json")
        )

    # --- 5. 최종 평가와 리포트 (p.11-16) ----------------------------------

    @router.post("/sessions/complete", response_model=ApiResponse, summary="훈련 최종 종료 및 비동기 분석 요청")
    async def sessions_complete(
        session_id: Annotated[str, Form()],
        user_id: Annotated[str, Form()],
        scenario_id: Annotated[str, Form()],
        vr_score: Annotated[int, Form(ge=0, le=100)],
        is_pass: Annotated[bool, Form()],
        avg_thickness_um: Annotated[float, Form(ge=0)],
        avg_speed_mms: Annotated[float, Form(ge=0)],
        avg_distance_mm: Annotated[float, Form(ge=0)],
        telemetry_csv: Annotated[UploadFile, File()],
        height_map_image: Annotated[UploadFile | None, File()] = None,
    ) -> JSONResponse:
        """multipart/form-data (p.12). `report_id`를 부여하고 `pending`으로 시작한다."""
        csv_bytes = len(await telemetry_csv.read())
        image_bytes = len(await height_map_image.read()) if height_map_image is not None else None
        completion = await to_thread.run_sync(
            lambda: state.create_completion(
                session_id=session_id,
                requested_at=_now(),
                payload={
                    "user_id": user_id, "scenario_id": scenario_id, "vr_score": vr_score,
                    "is_pass": is_pass, "avg_thickness_um": avg_thickness_um,
                    "avg_speed_mms": avg_speed_mms, "avg_distance_mm": avg_distance_mm,
                    "telemetry_csv_bytes": csv_bytes, "height_map_image_bytes": image_bytes,
                },
            )
        )
        logger.info("session complete %s report=%s csv=%dB", session_id, completion.report_id, csv_bytes)
        return _ok(
            SessionCompleteData(
                report_id=completion.report_id, session_id=session_id,
                report_status="pending", telemetry_csv_bytes=csv_bytes,
                height_map_image_bytes=image_bytes,
            ).model_dump(mode="json")
        )

    @router.get(
        "/sessions/{session_id}/reports/{report_id}",
        response_model=ApiResponse,
        summary="리포트 생성 상태 폴링 조회",
    )
    async def report_status(session_id: str, report_id: str) -> JSONResponse:
        row = await to_thread.run_sync(lambda: state.get_completion(report_id))
        if row is None or row.session_id != session_id:
            return _error(404, "report not found")
        return _ok(
            ReportStatusData(
                report_id=row.report_id, session_id=row.session_id,
                report_status=row.report_status,  # type: ignore[arg-type]
                requested_at=row.requested_at,
            ).model_dump(mode="json")
        )

    @router.get("/reports/{report_id}", response_model=ApiResponse, summary="최종 종합 리포트 상세 조회")
    async def report_detail(report_id: str) -> JSONResponse:
        row = await to_thread.run_sync(lambda: state.get_completion(report_id))
        if row is None:
            return _error(404, "report not found")
        return _ok(
            _unavailable_report(
                report_id=row.report_id, session_id=row.session_id,
                status=row.report_status, generated_at=row.requested_at,
            ).model_dump(mode="json")
        )

    # --- 6. 사용자 관리 (p.17-20) -----------------------------------------

    @router.get("/users", response_model=ApiResponse, summary="훈련자 전체 목록 조회")
    async def users_list(
        skip: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int, Query(ge=1, le=1000)] = 100,
    ) -> JSONResponse:
        rows = await to_thread.run_sync(lambda: state.list_users(skip=skip, limit=limit))
        return _ok([UserProfile(**row.__dict__).model_dump(mode="json") for row in rows])

    @router.post("/users", response_model=ApiResponse, summary="신규 훈련자 계정 직접 등록")
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

    @router.get("/users/{user_id}", response_model=ApiResponse, summary="훈련자 프로필 조회")
    async def users_get(user_id: str) -> JSONResponse:
        row = await to_thread.run_sync(lambda: state.get_user(user_id))
        if row is None:
            return _error(404, "user not found")
        return _ok(UserProfile(**row.__dict__).model_dump(mode="json"))

    # --- 4. 이상 알림 SSE (p.10) ------------------------------------------

    @router.get("/stream/anomalies/{session_id}", summary="실시간 이상 탐지 SSE 이벤트 스트림")
    async def stream_anomalies(session_id: str, request: Request) -> StreamingResponse:
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
            @router.post(
                f"/test/sequence/{case}",
                response_model=ApiResponse,
                summary=f"[테스트] {case} 시퀀스 시뮬레이션 및 SSE 방송",
                name=f"test_sequence_{case.replace('-', '_')}",
            )
            async def _handler(session_id: str = "1_1000053_1") -> JSONResponse:
                sequence_id = f"TEST-{case}"
                published = None
                if anomaly_type != "none":
                    published = anomalies.publish(
                        session_id=session_id, sequence_id=sequence_id,
                        anomaly_type=anomaly_type, payload={"message": message},
                    )
                return _ok(
                    {
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
