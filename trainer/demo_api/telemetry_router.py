"""`POST /api/v1/telemetry/sequence` (v2.5 계약 §7, PDF p.8 "시퀀스 전송").

정본 경로 하나와 멱등 별칭 하나를 등록한다. 별칭 `/api/v1/copilot/analyze-sequence`는
계약에 없지만 벤더 클라이언트가 **같은 본문을 두 경로로** 보낸다(델타 B18). 404를 주면
현장에서 실패하므로 받되, 저장소가 두 번째를 `duplicate`로 식별한다.

엔진 상태(계획 E1·E4·E5):
* `pose_analysis`는 MCP `/pose_estimation/sequence`로 위임한다. E4는 결국 같은 엔진 모듈을
  agent-api 안에서 직접 부르는 형태를 요구하는데, 그 모듈은 아직 `mcp_servers` 아래에 있고
  agent-api 이미지가 `mcp_servers`를 복사하지 않는다. HTTP 위임은 그 사이의 형태다.
* `coating_analysis`·`sequence_analysis`는 구현되지 않았다. 임의 값을 만들지 않고
  `analysis_status="not_available"`과 안정적인 `reason_code`를 낸다(계약 §7.3 말미).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Final

import httpx
from anyio import to_thread
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from trainer.demo_api.sequence_store import SequenceStore
from trainer.demo_api.telemetry_contracts import (
    EngineFeedback,
    PoseCorrection,
    PoseFeedback,
    SequenceDataAck,
    TelemetrySequenceData,
    TelemetrySequenceRequest,
    TelemetrySequenceResponse,
)

logger = logging.getLogger(__name__)

TELEMETRY_SEQUENCE_PATH: Final[str] = "/api/v1/telemetry/sequence"
# 계약에 없는 벤더 중복 경로. 정본은 위쪽이다.
ANALYZE_SEQUENCE_ALIAS: Final[str] = "/api/v1/copilot/analyze-sequence"

_SUCCESS_MESSAGE: Final[str] = "Sequence evaluated"
_MCP_TIMEOUT_SEC: Final[float] = 10.0
_UNKNOWN_FIELD_CAP: Final[int] = 50
_ENGINE_MISSING: Final[str] = "engine_not_implemented"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _body_sha256(query: TelemetrySequenceRequest) -> str:
    """정규화한 본문의 해시. 같은 시퀀스를 두 경로로 보내면 같은 값이 나온다."""
    canonical = json.dumps(query.model_dump(mode="json"), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
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


def _pose_payload(query: TelemetrySequenceRequest, session_id: str) -> dict[str, Any]:
    """MCP `SequenceRequest`의 모양. `level`·`floor_height_m`는 벤더가 아니라 우리가 채운다."""
    payload: dict[str, Any] = {
        "session_id": session_id,
        "sequence_id": query.sequence_id,
        "user_id": query.user_id,
        "scenario_id": query.scenario_id,
        "zone_id": query.zone_id,
        "sequence_info": [frame.model_dump(mode="json") for frame in query.sequence_info],
    }
    if query.sequence_setting is not None:
        payload["sequence_setting"] = query.sequence_setting
    if query.sequence_result is not None:
        payload["sequence_result"] = query.sequence_result
    return payload


def _project_pose(result: dict[str, Any]) -> PoseFeedback:
    """MCP 결과에서 계약 §7.3이 정한 필드만 뽑는다.

    `evidence`와 `target.per_frame`은 내부 인터페이스의 것이므로 공개하지 않는다.
    `target_validated`는 **적용할 좌표가 실제로 있는지**다 — `per_frame`이 비어 있으면
    교정이 말뿐이므로 검증되지 않은 것이다.
    """
    corrections: list[PoseCorrection] = []
    for item in result.get("corrections") or []:
        if not isinstance(item, dict):
            continue
        target = item.get("target") if isinstance(item.get("target"), dict) else {}
        corrections.append(
            PoseCorrection(
                joint=str(item.get("joint", "")),
                label=str(item.get("correction_text") or item.get("label") or ""),
                target_validated=bool((target or {}).get("per_frame")),
                angle_delta_deg=item.get("angle_delta_deg"),
                direction=item.get("direction"),
            )
        )
    status = result.get("analysis_status")
    return PoseFeedback(
        analysis_status="available" if status == "available" else "not_available",
        reason_code=result.get("reason_code"),
        similarity_score=result.get("similarity_score"),
        correction_mode=result.get("correction_mode"),
        corrections=corrections,
    )


def _unavailable(reason: str) -> PoseFeedback:
    return PoseFeedback(
        analysis_status="not_available", reason_code=reason,
        similarity_score=None, correction_mode=None, corrections=[],
    )


def create_telemetry_router(
    *,
    store: SequenceStore | None = None,
    mcp_base_url: str | None = None,
    transport: httpx.BaseTransport | None = None,
) -> APIRouter:
    router = APIRouter()
    sequences = store if store is not None else SequenceStore()
    base = (mcp_base_url or os.environ.get("MCP_BASE_URL", "http://localhost:8100")).rstrip("/")

    def _call_pose(payload: dict[str, Any]) -> tuple[PoseFeedback, str | None, bool]:
        """`(feedback, text_guide, degraded)`. 엔진 장애가 수신을 실패로 만들지 않는다."""
        try:
            with httpx.Client(timeout=_MCP_TIMEOUT_SEC, transport=transport) as client:
                response = client.post(f"{base}/pose_estimation/sequence", json=payload)
            if response.status_code != 200:
                logger.warning("pose engine HTTP %d", response.status_code)
                return _unavailable("engine_unavailable"), None, True
            body = response.json()
        except Exception:  # noqa: BLE001 - 엔진 장애는 200 + not_available 이다
            logger.exception("pose engine call failed")
            return _unavailable("engine_unavailable"), None, True
        # MCP는 오류도 HTTP 200으로 답하므로 status 필드가 성공을 결정한다.
        if not isinstance(body, dict) or body.get("status") != "success":
            return _unavailable("engine_unavailable"), None, True
        result = body.get("result") if isinstance(body.get("result"), dict) else {}
        summary = result.get("summary")
        return _project_pose(result), (str(summary) if summary else None), False

    async def _handle(query: TelemetrySequenceRequest) -> JSONResponse:
        started = time.monotonic()
        frames_received = len(query.sequence_info)
        ack_record = await to_thread.run_sync(
            lambda: sequences.record(
                session_key=query.session_id or "",
                sequence_id=query.sequence_id,
                body_sha256=_body_sha256(query),
                frames=frames_received,
                received_at=_now(),
            )
        )

        if query.session_id is None:
            # 없는 세션을 지어내면 자세 누적이 엉뚱한 곳에 쌓인다. 저장은 하고 채점은 하지 않는다.
            pose, text_guide, degraded = _unavailable("session_id_missing"), None, False
        elif ack_record.duplicate:
            pose, text_guide, degraded = _unavailable("duplicate_sequence"), None, False
        elif frames_received == 0:
            pose, text_guide, degraded = _unavailable("no_frames"), None, False
        else:
            pose, text_guide, degraded = await to_thread.run_sync(
                lambda: _call_pose(_pose_payload(query, query.session_id or ""))
            )

        envelope = TelemetrySequenceResponse(
            status="SUCCESS",
            code=200,
            message=_SUCCESS_MESSAGE,
            data=TelemetrySequenceData(
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
                pose_feedback=pose,
                text_guide=text_guide,
            ),
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
        logger.info(
            "telemetry sequence session=%s sequence=%s frames=%d stored=%s duplicate=%s"
            " conflict=%s pose=%s degraded=%s",
            query.session_id,
            query.sequence_id,
            frames_received,
            ack_record.stored,
            ack_record.duplicate,
            ack_record.conflict,
            pose.analysis_status,
            degraded or ack_record.conflict,
        )
        return JSONResponse(status_code=200, content=envelope.model_dump(mode="json"))

    router.post(TELEMETRY_SEQUENCE_PATH)(_handle)
    router.post(ANALYZE_SEQUENCE_ALIAS)(_handle)
    return router
