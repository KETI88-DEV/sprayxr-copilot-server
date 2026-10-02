"""Multipart voice transport for the Co-Pilot query pipeline."""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable, Mapping
from typing import Annotated, Literal

from anyio import to_thread
from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse

from trainer.demo_api.copilot_contracts import CopilotQueryRequest, TaskType
from trainer.demo_api.copilot_router import (
    PENDING_TASK_ANSWERS,
    PipelineProvider,
    error_response,
    execute_copilot_query,
)
from trainer.demo_api.sequence_store import SequenceStore
from trainer.demo_api.speech_to_text import (
    InvalidAudioError,
    SpeechToText,
    SpeechToTextUnavailableError,
    VoiceLanguage,
    validate_wav,
)
from trainer.demo_api.voice_contracts import VoiceCopilotResponse, VoiceQueryData

logger = logging.getLogger(__name__)

VOICE_QUERY_PATH = "/api/v1/copilot/query-voice"
MAX_UPLOAD_BYTES = 1_048_576
SpeechToTextProvider = Callable[[], SpeechToText]


def create_voice_router(
    pipeline_provider: PipelineProvider | None,
    *,
    stt_provider: SpeechToTextProvider,
    pending_answers: Mapping[str, str] = PENDING_TASK_ANSWERS,
    store: SequenceStore | None = None,
    mcp_base_url: str | None = None,
    transport: object | None = None,
) -> APIRouter:
    """Create the independent WAV-to-query endpoint used by VR clients."""
    router = APIRouter()
    sequences = store if store is not None else SequenceStore()
    mcp_base = mcp_base_url or os.environ.get("MCP_BASE_URL", "http://localhost:8100")

    @router.post(
        VOICE_QUERY_PATH,
        response_model=VoiceCopilotResponse,
        tags=["3. AI Co-Pilot 질의"],
        summary="3-2. VR 음성 Co-Pilot 질의",
        description="HMD에서 녹음한 WAV를 STT로 변환한 뒤 기존 Query와 동일한 경로로 답변합니다. WAV PCM16·16kHz·mono·15초 이하만 허용합니다.",
    )
    async def voice_query(
        audio: Annotated[UploadFile, File(description="WAV, PCM 16-bit, 16 kHz, mono, maximum 15 seconds")],
        user_id: Annotated[str, Form(min_length=1, description="Training user identifier", examples=["W-017"])],
        language: Annotated[VoiceLanguage, Form(description="Spoken language", examples=["ko"])],
        session_id: Annotated[str | None, Form(description="Optional active training session")] = None,
        scenario_id: Annotated[str | None, Form(description="Optional scenario identifier")] = None,
        request_id: Annotated[str | None, Form(description="Optional caller request identifier")] = None,
        task_type: Annotated[TaskType, Form(description="Same task type as the text Query API")] = "knowledge_search",
        use_rag: Annotated[Literal["Y", "N"], Form(description="Use the knowledge base", examples=["Y"])]= "Y",
        context: Annotated[str | None, Form(description="Optional JSON or short context text")] = None,
    ) -> JSONResponse:
        started = time.monotonic()
        payload = await audio.read(MAX_UPLOAD_BYTES + 1)
        if len(payload) > MAX_UPLOAD_BYTES:
            return error_response(413, "Audio file is too large")

        try:
            validated = validate_wav(payload)
        except InvalidAudioError:
            return error_response(422, "Invalid audio")

        stt_started = time.monotonic()
        try:
            speech_engine = await to_thread.run_sync(stt_provider)
            transcript_result = await to_thread.run_sync(
                lambda: speech_engine.transcribe(validated.audio, language)
            )
        except SpeechToTextUnavailableError:
            logger.warning("voice query STT unavailable request_id=%s language=%s", request_id, language)
            return error_response(503, "Speech recognition unavailable")
        except Exception:  # noqa: BROAD_EXCEPT_OK - do not expose or log audio/transcript internals
            logger.warning("voice query STT failed request_id=%s language=%s", request_id, language)
            return error_response(503, "Speech recognition failed")

        stt_latency_ms = max(0, int((time.monotonic() - stt_started) * 1000))
        transcript = transcript_result.text.strip()
        if not transcript:
            return error_response(422, "No speech was recognized")

        query = CopilotQueryRequest(
            request_id=request_id,
            user_id=user_id,
            session_id=session_id,
            scenario_id=scenario_id,
            task_type=task_type,
            prompt=transcript,
            use_rag=use_rag,
            context=context,
        )
        response = await execute_copilot_query(
            query,
            pipeline_provider=pipeline_provider,
            pending_answers=pending_answers,
            sequences=sequences,
            mcp_base_url=mcp_base,
            transport=transport,
            started=started,
        )
        if response.status_code != 200:
            return response

        body = json.loads(response.body)
        body["data"] = VoiceQueryData(
            **body["data"],
            transcript=transcript,
            language=language,
            stt_latency_ms=stt_latency_ms,
        ).model_dump(mode="json")
        logger.info(
            "voice query done request_id=%s language=%s stt_latency_ms=%d",
            query.request_id,
            language,
            stt_latency_ms,
        )
        return JSONResponse(status_code=200, content=body)

    return router
