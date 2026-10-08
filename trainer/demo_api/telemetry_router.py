"""v2.5 시퀀스 엔드포인트 둘.

정본은 `docs/XR Painting Simulator AI Co-Pilot Service API 정의서 - 대화형 API 명세서.pdf`다.

* `POST /api/v1/telemetry/sequence` (p.4-5) — 90 FPS 배치를 **데이터베이스에 올린다**.
  p.4 는 여기에 "도막 두께 및 궤적 평가"를 걸고 **자세는 걸지 않는다**. 자세 유사도 비교는
  `copilot/query` 의 `pose_recomm` 이 `context` 로 시퀀스를 가리켜 수행한다(p.9) —
  이 모듈의 이전 판이 자세를 여기서 채점한 것은 틀렸다
* `POST /api/v1/copilot/analyze-sequence` — 1회 패스 궤적 비교 + 원포인트 레슨.
  최신 TSB 전송처럼 `scenario_id`와 `sequence_setting.tool_type`으로 전문가 기준을 선택한다.

**둘은 별개 엔드포인트다.** 이 모듈의 이전 판이 뒤쪽을 "계약에 없는 벤더 중복 전송"으로
보고 멱등 별칭으로 받은 것은 틀렸다 — 근거로 삼은 260813 PDF에 그 항목이 없었을 뿐이다.
같은 본문을 둘에 보내는 것이 설계이므로, 저장소는 두 경로를 **서로 다른 시퀀스 기록**으로
보지 않고 같은 키로 본다. 두 번째 도착은 `duplicate`가 되고 다시 채점하지 않는다 —
궤적 비교(`sequence_analysis`)가 구현되면 그때 분석 종류별로 갈라야 한다.

엔진 상태: `coating_analysis`·`sequence_analysis` 둘 다 구현되지 않았다. 임의 값을 만들지
않고 `analysis_status="not_available"`과 안정적인 `reason_code`를 낸다. 이상 탐지와 SSE
발행(p.4의 "이상 탐지 시 SSE 채널로 Push")도 아직 없다.

**본문을 저장한다.** 나중에 `pose_recomm` 질의가 채점할 대상이므로 해시만으로는 부족하다.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Final, Protocol

from anyio import to_thread
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from trainer.demo_api.monitor import MonitorRecorder
from trainer.demo_api.sequence_store import SequenceStore
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
_ENGINE_MISSING: Final[str] = "engine_not_implemented"
# MonitorEvent.session_id 는 문자열이다. 귀속되지 않은 시퀀스도 모니터에서 보여야 한다.
_UNATTRIBUTED: Final[str] = "(unattributed)"


class GuidanceEngine(Protocol):
    def evaluate(self, request: AnalyzeSequenceRequest) -> SequenceGuide: ...


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
# 보낼 때 한쪽에만 붙는 값 때문에 중복 판정이 깨지면 안 된다.
_NON_CONTENT_FIELDS: Final[frozenset[str]] = frozenset({"mode_id"})


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
    coating_progress_engine: CoatingProgressEvaluator | None = None,
) -> APIRouter:
    """``recorder``를 주면 호출당 입력 1건 + 출력 1건을 `/api/monitor/*`에 남긴다.

    최상위 검증 실패(422)는 Pydantic이 핸들러 전에 거부하므로 기록되지 않는다 —
    옛 시뮬레이터 라우터는 직접 검증해서 422도 남긴다. 그 차이를 알고 쓴다.
    """
    router = APIRouter()
    sequences = store if store is not None else SequenceStore()
    guidance_cache: dict[tuple[str, str, int | None, str | None, str], SequenceGuide] = {}
    coating_progress_cache: dict[
        tuple[str, str, int | None, str | None, str], CoatingProgress
    ] = {}

    async def _handle(query: TelemetrySequenceRequest, *, analyze: bool) -> JSONResponse:
        """두 엔드포인트가 같은 처리를 쓴다. 분석 종류별 분기는 엔진이 생기면 넣는다."""
        started = time.monotonic()
        monitor_session = query.session_id or _UNATTRIBUTED
        if recorder is not None:
            # 처리 전에 남긴다 — 이후 단계가 실패해도 무엇이 들어왔는지는 남는다.
            _ = recorder.record(monitor_session, "runtime", "input", query.model_dump(mode="json"))
        frames_received = len(query.sequence_info)
        stored_body = json.dumps(query.model_dump(mode="json"), ensure_ascii=False)
        body_sha256 = _body_sha256(query)
        ack_record = await to_thread.run_sync(
            lambda: sequences.record(
                session_key=query.session_id,
                sequence_id=query.sequence_id,
                body_sha256=body_sha256,
                frames=frames_received,
                received_at=_now(),
                body=stored_body,
            )
        )

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
                coating_feedback=EngineFeedback(analysis_status="not_available", reason_code=_ENGINE_MISSING),
                sequence_feedback=EngineFeedback(analysis_status="not_available", reason_code=_ENGINE_MISSING),
                text_guide=None,
        )
        if analyze:
            assert isinstance(query, AnalyzeSequenceRequest)
            cache_key = (
                query.session_id,
                query.sequence_id,
                query.resolved_mode_id,
                query.process_type,
                body_sha256,
            )
            guide = guidance_cache.get(cache_key)
            if guide is None:
                guide = guidance_engine.evaluate(query) if guidance_engine is not None else _unavailable_guide()
                guidance_cache[cache_key] = guide
            coating_progress = coating_progress_cache.get(cache_key)
            if coating_progress is None:
                coating_progress = (
                    coating_progress_engine.evaluate(query)
                    if coating_progress_engine is not None
                    else _unavailable_progress(query.zone_id)
                )
                coating_progress_cache[cache_key] = coating_progress
            envelope = AnalyzeSequenceResponse(
                status="SUCCESS",
                code=200,
                message=_SUCCESS_MESSAGE,
                data=AnalyzeSequenceData(
                    **common_data,
                    guide=guide,
                    coating_progress=coating_progress,
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

    @router.post(
        TELEMETRY_SEQUENCE_PATH,
        response_model=TelemetrySequenceResponse,
        tags=["2. 시퀀스 수집·피드백"],
        summary="2-1. 현재 시퀀스 텔레메트리 저장",
        description="Trigger ON→OFF 한 번의 현재 시퀀스를 저장합니다. 누적 히트맵은 입력하지 않습니다.",
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
            "distance(near/ok/far)를 구조화해 반환합니다. 학습 모드는 scenario_id, "
            "스프레이/블라스팅 구분은 sequence_setting.tool_type에서 자동 선택합니다. "
            "스프레이의 cur_sequence_heatmap은 서버가 세션·Zone별로 누적하여 "
            "추가 도장/다음 Zone 이동/과도장 중단 여부를 coating_progress로 반환합니다."
        ),
    )
    async def analyze_sequence(query: AnalyzeSequenceRequest) -> JSONResponse:
        return await _handle(query, analyze=True)

    return router
