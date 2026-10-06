"""Multipart voice transport for the Co-Pilot query pipeline."""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Annotated, Literal

from anyio import to_thread
from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse, Response

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
TEST_VOICE_QUERY_PATH = "/api/v1/test/query-voice"
TEST_VOICE_ASSET_PATH = "/api/v1/test/assets/query-voice-ko.wav"
MAX_UPLOAD_BYTES = 1_048_576
SpeechToTextProvider = Callable[[], SpeechToText]

_TEST_USER_ID = "TSB-VOICE-DEMO-001"
_TEST_SESSION_ID = "SWAGGER-VOICE-DEMO"
_TEST_SCENARIO_ID = "training-mode-1"
_TEST_REQUEST_ID = "REQ-VOICE-TEST-001"


def _sample_wav() -> bytes:
    return (Path(__file__).parent / "assets" / "query-voice-ko.wav").read_bytes()


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

    async def _run_voice_query(
        payload: bytes,
        *,
        user_id: str,
        language: VoiceLanguage,
        session_id: str | None,
        scenario_id: str | None,
        request_id: str | None,
        task_type: TaskType,
        use_rag: Literal["Y", "N"],
        context: str | None,
    ) -> JSONResponse:
        started = time.monotonic()
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

    @router.post(
        VOICE_QUERY_PATH,
        response_model=VoiceCopilotResponse,
        tags=["3. AI Co-Pilot 질의"],
        summary="3-2. VR 음성 Co-Pilot 질의",
        description="HMD에서 녹음한 WAV를 STT로 변환한 뒤 기존 Query와 동일한 경로로 답변합니다. WAV PCM16·16kHz·mono·15초 이하만 허용합니다.",
    )
    async def voice_query(
        audio: Annotated[UploadFile, File(description="WAV, PCM 16-bit, 16 kHz, mono, maximum 15 seconds")],
        user_id: Annotated[str, Form(min_length=1, description="훈련자 식별자", examples=["W-017"])],
        language: Annotated[VoiceLanguage, Form(description="음성 언어", examples=["ko"])],
        session_id: Annotated[str | None, Form(description="진행 중인 세션 식별자", examples=["TSB-DEMO-001"])] = None,
        scenario_id: Annotated[str | None, Form(description="콘텐츠 식별자", examples=["training-mode-3"])] = None,
        request_id: Annotated[str | None, Form(description="호출 추적용 식별자", examples=["REQ-VOICE-001"])] = None,
        task_type: Annotated[TaskType, Form(description="텍스트 Query API와 동일한 작업 유형", examples=["knowledge_search"])] = "knowledge_search",
        use_rag: Annotated[Literal["Y", "N"], Form(description="Use the knowledge base", examples=["Y"])]= "Y",
        context: Annotated[str | None, Form(description="선택 JSON 또는 짧은 상황 설명", examples=["현재 훈련모드 3의 Zone 3을 도장 중입니다."])] = None,
    ) -> JSONResponse:
        payload = await audio.read(MAX_UPLOAD_BYTES + 1)
        return await _run_voice_query(
            payload,
            user_id=user_id,
            language=language,
            session_id=session_id,
            scenario_id=scenario_id,
            request_id=request_id,
            task_type=task_type,
            use_rag=use_rag,
            context=context,
        )

    @router.post(
        TEST_VOICE_QUERY_PATH,
        response_model=VoiceCopilotResponse,
        tags=["0. Swagger 모의 연동"],
        summary="0-5. 무입력 음성질의 전체 경로 테스트",
        description=(
            "파라미터나 파일을 입력하지 않고 서버에 포함된 한국어 WAV를 실제 STT와 Query 경로로 처리합니다. "
            "응답의 transcript가 인식 문장이고 answer가 Co-Pilot 답변입니다. 사용된 음성은 "
            f"`GET {TEST_VOICE_ASSET_PATH}`에서 내려받아 실제 3-2 API에도 업로드할 수 있습니다."
        ),
    )
    async def test_voice_query() -> JSONResponse:
        return await _run_voice_query(
            _sample_wav(),
            user_id=_TEST_USER_ID,
            language="ko",
            session_id=_TEST_SESSION_ID,
            scenario_id=_TEST_SCENARIO_ID,
            request_id=_TEST_REQUEST_ID,
            task_type="knowledge_search",
            use_rag="Y",
            context="Swagger 음성질의 모의연동",
        )

    @router.get(
        TEST_VOICE_ASSET_PATH,
        tags=["0. Swagger 모의 연동"],
        summary="0-6. query-voice 테스트용 한국어 WAV 다운로드",
        description=(
            "실제 3-2 query-voice의 audio 필드에 업로드할 수 있는 WAV 파일을 내려받습니다. "
            "형식은 PCM16·16 kHz·mono이며 문장은 ‘스프레이 도장 시 적정 이격거리를 알려주세요.’입니다."
        ),
        response_class=Response,
    )
    def test_voice_asset() -> Response:
        return Response(
            content=_sample_wav(),
            media_type="audio/wav",
            headers={"Content-Disposition": 'attachment; filename="query-voice-ko.wav"'},
        )

    return router
