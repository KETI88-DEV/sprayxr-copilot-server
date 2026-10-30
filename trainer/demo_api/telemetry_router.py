"""v2.5 telemetry storage and sequence analysis, sharing one immutable sequence key.

Telemetry stores the TSB body; analyze-sequence additionally evaluates guidance,
order/direction and coating progress. The same body may be sent to both paths.
Conflicting content and late new sequence IDs return 409 before engine mutations.
Analysis outcomes persist so retries never advance smoothing or coating twice.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Final, Iterator, Protocol

from anyio import to_thread
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from trainer.demo_api.monitor import MonitorRecorder
from trainer.demo_api.anomaly_stream import AnomalyBroadcaster
from trainer.demo_api.sequence_analysis import SequenceDecision
from trainer.demo_api.sequence_store import NON_CONTENT_FIELDS, SequenceStore
from trainer.demo_api.telemetry_contracts import (
    AnalyzeSequenceData,
    AnalyzeSequenceRequest,
    AnalyzeSequenceResponse,
    CoatingProgress,
    EngineFeedback,
    GuideItem,
    SequenceDataAck,
    SequenceGuide,
    TelemetrySequenceData,
    TelemetrySequenceRequest,
    TelemetrySequenceResponse,
)

logger = logging.getLogger(__name__)

TELEMETRY_SEQUENCE_PATH: Final[str] = "/api/v1/telemetry/sequence"
ANALYZE_SEQUENCE_PATH: Final[str] = "/api/v1/copilot/analyze-sequence"

_SUCCESS_MESSAGE: Final[str] = "Sequence evaluated"
_UNKNOWN_FIELD_CAP: Final[int] = 50
_COATING_PENDING: Final[str] = "coating_engine_pending_integration"
_SEQUENCE_UNAVAILABLE: Final[str] = "sequence_reference_unavailable"
# MonitorEvent.session_id 는 문자열이다. 귀속되지 않은 시퀀스도 모니터에서 보여야 한다.
_UNATTRIBUTED: Final[str] = "(unattributed)"


class GuidanceEngine(Protocol):
    def evaluate(self, request: AnalyzeSequenceRequest) -> SequenceGuide: ...


class SequenceEngine(Protocol):
    def evaluate(self, request: AnalyzeSequenceRequest, store: SequenceStore) -> SequenceDecision: ...


class CoatingProgressEvaluator(Protocol):
    def evaluate(self, request: AnalyzeSequenceRequest) -> CoatingProgress: ...


def _unavailable_guide() -> SequenceGuide:
    item = GuideItem(
        status=None,
        message=None,
        available=False,
        reason="reference_unavailable",
    )
    return SequenceGuide(speed=item, distance=item.model_copy())


def _unavailable_progress(zone_id: str) -> CoatingProgress:
    return CoatingProgress(
        zone_id=zone_id,
        available=False,
        status=None,
        recommendation=None,
        completion_percent=None,
        coverage_percent=None,
        overpaint_percent=None,
        next_zone_id=None,
        message=None,
        reason="coating_reference_unavailable",
    )


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# 시퀀스 내용이 아니라 분석 파라미터인 필드. 해시에서 뺀다 — 같은 시퀀스를 두 경로로
# 보낼 때 한쪽에만 붙는 값 때문에 중복 판정이 깨지면 안 된다. 목록은 저장소 모듈이 정본이다.
_NON_CONTENT_FIELDS: Final[frozenset[str]] = NON_CONTENT_FIELDS


def _body_sha256(query: TelemetrySequenceRequest) -> str:
    """시퀀스 **내용**의 해시. 같은 시퀀스를 두 경로로 보내면 같은 값이 나온다."""
    content = {
        key: value
        for key, value in query.model_dump(mode="json").items()
        if key not in _NON_CONTENT_FIELDS
    }
    canonical = json.dumps(content, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _unknown_fields(query: TelemetrySequenceRequest) -> list[str]:
    """프레임 하위의 미지 키를 JSON path로 모은다. 120FPS 세션이므로 상한을 둔다."""
    found: list[str] = []
    for index, frame in enumerate(query.sequence_info):
        for key in (frame.model_extra or {}):
            found.append(f"sequence_info[{index}].{key}")
            if len(found) >= _UNKNOWN_FIELD_CAP:
                return found
    return found





def create_telemetry_router(
    *,
    store: SequenceStore | None = None,
    recorder: MonitorRecorder | None = None,
    guidance_engine: GuidanceEngine | None = None,
    sequence_engine: SequenceEngine | None = None,
    broadcaster: AnomalyBroadcaster | None = None,
    coating_progress_engine: CoatingProgressEvaluator | None = None,
    session_store: Any = None,
) -> APIRouter:
    """``recorder``를 주면 호출당 입력 1건 + 출력 1건을 `/api/monitor/*`에 남긴다.

    최상위 검증 실패(422)는 Pydantic이 핸들러 전에 거부하므로 기록되지 않는다 —
    옛 시뮬레이터 라우터는 직접 검증해서 422도 남긴다. 그 차이를 알고 쓴다.
    """
    router = APIRouter()
    sequences = store if store is not None else SequenceStore()
    # Fixed stripes bound memory while serializing same-session state transitions.
    session_locks = [threading.Lock() for _ in range(64)]

    def _handle_sync(query: TelemetrySequenceRequest, *, analyze: bool) -> JSONResponse:
        started = time.monotonic()
        monitor_session = query.session_id or _UNATTRIBUTED
        if recorder is not None:
            # 처리 전에 남긴다 — 이후 단계가 실패해도 무엇이 들어왔는지는 남는다.
            _ = recorder.record(monitor_session, "runtime", "input", query.model_dump(mode="json"))
        frames_received = len(query.sequence_info)
        stored_body = json.dumps(query.model_dump(mode="json"), ensure_ascii=False)
        body_sha256 = _body_sha256(query)
        ack_record = sequences.record(
                session_key=query.session_id,
                sequence_id=query.sequence_id,
                body_sha256=body_sha256,
                frames=frames_received,
                received_at=_now(),
                body=stored_body,
                enforce_order=True,
            )

        def reject(reason: str) -> JSONResponse:
            payload = {"status": "ERROR", "code": 409, "message": reason, "data": None}
            if recorder is not None:
                recorder.record(monitor_session, "runtime", "output", payload)
            return JSONResponse(status_code=409, content=payload)

        if ack_record.conflict:
            return reject("sequence_body_conflict: sequence_id already has different content")
        if ack_record.out_of_order:
            return reject("sequence_out_of_order: submit new sequences in increasing sequence_id order")

        common_data: dict[str, Any] = dict(
                request_id=f"sequence-{query.sequence_id}",
                session_id=query.session_id,
                user_id=query.user_id,
                scenario_id=query.scenario_id,
                sequence_id=query.sequence_id,
                generated_at=_now(),
                latency_ms=int((time.monotonic() - started) * 1000),
                sequence_data_ack=SequenceDataAck(
                    stored=ack_record.stored,
                    duplicate=ack_record.duplicate,
                    frames_received=frames_received,
                    frames_accepted=frames_received if ack_record.stored else 0,
                ),
                coating_feedback=EngineFeedback(analysis_status="not_available", reason_code=_COATING_PENDING),
                sequence_feedback=EngineFeedback(analysis_status="not_available", reason_code=_SEQUENCE_UNAVAILABLE),
                text_guide=None,
        )
        if analyze:
            assert isinstance(query, AnalyzeSequenceRequest)
            analysis_key = json.dumps([query.resolved_mode_id, query.process_type, body_sha256])
            cached = sequences.load_analysis(
                session_key=query.session_id, sequence_id=query.sequence_id, analysis_key=analysis_key,
            )
            if cached is None and sequences.latest_sequence_id(query.session_id) != query.sequence_id:
                return reject("sequence_analysis_out_of_order: analyze each sequence before submitting a later sequence")
            if cached is None:
                guide = guidance_engine.evaluate(query) if guidance_engine is not None else _unavailable_guide()
                decision = sequence_engine.evaluate(query, sequences) if sequence_engine is not None else None
                target = None
                if session_store is not None:
                    session = session_store.get_session(query.session_id)
                    target = session.target_thickness_um if session is not None else None
                if coating_progress_engine is None:
                    coating_progress = _unavailable_progress(query.zone_id)
                elif callable(getattr(coating_progress_engine, "evaluate_replay", None)):
                    def prior_requests() -> Iterator[AnalyzeSequenceRequest]:
                        # The engine consumes this only after restart/LRU eviction.
                        for body in sequences.session_bodies_through(query.session_id, query.sequence_id):
                            if body.get("sequence_id") != query.sequence_id:
                                yield AnalyzeSequenceRequest.model_validate(body)
                    coating_progress = coating_progress_engine.evaluate_replay(
                        query, prior_requests(), target_thickness_um=target,
                    )
                else:
                    coating_progress = coating_progress_engine.evaluate(query)
                cached = {
                    "guide": guide.model_dump(mode="json"),
                    "coating_progress": coating_progress.model_dump(mode="json"),
                    "sequence_feedback": (decision.feedback.model_dump(mode="json") if decision else
                                          common_data["sequence_feedback"].model_dump(mode="json")),
                    "text_guide": decision.text_guide if decision else None,
                }
                sequences.save_analysis(
                    session_key=query.session_id, sequence_id=query.sequence_id,
                    analysis_key=analysis_key, result=cached,
                )
                if decision is not None and decision.anomaly_type is not None and broadcaster is not None:
                    broadcaster.publish(
                        session_id=query.session_id,
                        sequence_id=query.sequence_id,
                        anomaly_type=decision.anomaly_type,
                        payload={
                            "reason_code": decision.feedback.reason_code,
                            "message": decision.text_guide,
                        },
                    )
            common_data["sequence_feedback"] = cached["sequence_feedback"]
            common_data["text_guide"] = cached["text_guide"]
            envelope = AnalyzeSequenceResponse(
                status="SUCCESS",
                code=200,
                message=_SUCCESS_MESSAGE,
                data=AnalyzeSequenceData(
                    **common_data,
                    guide=cached["guide"],
                    coating_progress=cached["coating_progress"],
                ),
            )
        else:
            envelope = TelemetrySequenceResponse(
                status="SUCCESS",
                code=200,
                message=_SUCCESS_MESSAGE,
                data=TelemetrySequenceData(**common_data),
            )
        unknown = _unknown_fields(query)
        if unknown:
            # 응답에는 싣지 않는다(계약 §7.3의 12키). 120FPS 세션이므로 요청당 1건만 남긴다.
            logger.warning(
                "telemetry sequence unknown frame fields sequence=%s count=%d sample=%s",
                query.sequence_id,
                len(unknown),
                ",".join(unknown[:5]),
            )
        assert envelope.data is not None
        envelope.data.latency_ms = int((time.monotonic() - started) * 1000)
        payload = envelope.model_dump(mode="json")
        if recorder is not None:
            # 저장(상태 전이)이 위에서 끝난 뒤에 출력을 발행한다 — 순서를 뒤집으면
            # 구독자가 갱신 전 상태를 읽고 두 번째 신호를 받지 못한다(AGENTS.md).
            _ = recorder.record(monitor_session, "runtime", "output", payload)
        logger.info(
            "telemetry sequence session=%s sequence=%s frames=%d stored=%s duplicate=%s conflict=%s",
            query.session_id,
            query.sequence_id,
            frames_received,
            ack_record.stored,
            ack_record.duplicate,
            ack_record.conflict,
        )
        return JSONResponse(status_code=200, content=payload)

    async def _handle(query: TelemetrySequenceRequest, *, analyze: bool) -> JSONResponse:
        def work() -> JSONResponse:
            with session_locks[hash(query.session_id) % len(session_locks)]:
                return _handle_sync(query, analyze=analyze)
        return await to_thread.run_sync(work)

    @router.post(
        TELEMETRY_SEQUENCE_PATH,
        response_model=TelemetrySequenceResponse,
        tags=["2. 시퀀스 수집·피드백"],
        summary="2-1. 현재 시퀀스 텔레메트리 저장",
        description=(
            "Trigger ON→OFF 한 번의 현재 시퀀스를 저장합니다. "
            "cur_sequence_heatmap이 포함된 경우 크롭 내 픽셀은 해당 시점의 전체 누적 두께입니다."
        ),
    )
    async def telemetry_sequence(query: TelemetrySequenceRequest) -> JSONResponse:
        return await _handle(query, analyze=False)

    @router.post(
        ANALYZE_SEQUENCE_PATH,
        response_model=AnalyzeSequenceResponse,
        tags=["2. 시퀀스 수집·피드백"],
        summary="2-2. 시퀀스 분석 및 속도·이격거리·현재 Zone 도막 가이드",
        description=(
            "전문가 기준과 비교하여 매 시퀀스마다 speed(slow/ok/fast)와 "
            "distance(near/ok/far), Zone 작업 순서와 도포 방향 판정을 구조화해 반환합니다. "
            "이상 판정은 같은 세션의 SSE 채널에도 한 번만 발행됩니다. 학습 모드는 scenario_id, "
            "스프레이/블라스팅 구분은 sequence_setting.tool_type에서 자동 선택합니다. "
            "스프레이의 cur_sequence_heatmap은 해당 시점 피도면의 전체 누적 두께입니다. "
            "이미 누적된 값을 더하지 않고, 현재 시퀀스가 도포한 크롭 영역의 최신 절대 두께로 갱신하여 "

            "추가 도장/다음 Zone 이동/과도장 중단 여부를 coating_progress로 반환합니다."
        ),
    )
    async def analyze_sequence(query: AnalyzeSequenceRequest) -> JSONResponse:
        return await _handle(query, analyze=True)

    return router
