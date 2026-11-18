"""Multipart voice transport for the Co-Pilot query pipeline."""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Annotated

from anyio import to_thread
from fastapi import APIRouter, File, Form, Query, Request, UploadFile
from fastapi.responses import JSONResponse, Response

from trainer.demo_api.copilot_contracts import CopilotQueryRequest, RagUsage, ResponseMode, TaskType, SUMMARY_DESCRIPTION
from trainer.demo_api.copilot_router import (
    PENDING_TASK_ANSWERS,
    PipelineProvider,
    error_response,
    execute_copilot_query,
)
from trainer.demo_api.guided_workflow import DEMO_USER_ID, ensure_demo_user
from trainer.demo_api.dialogue import conversation_key
from trainer.demo_api.query_language import LANGUAGE_DESCRIPTION, QueryLanguage
from trainer.demo_api.sequence_store import SequenceStore
from trainer.demo_api.sequence_analysis import SequenceAnalysisEngine
from trainer.demo_api.v1_store import V1Store
from trainer.demo_api.speech_to_text import (
    InvalidAudioError,
    MAX_WAV_BYTES,
    SpeechToText,
    SpeechToTextUnavailableError,
    VoiceLanguage,
    normalize_wav,
    validate_wav,
)
from trainer.demo_api.voice_contracts import VoiceCopilotResponse, VoiceQueryData

logger = logging.getLogger(__name__)

VOICE_QUERY_PATH = "/api/v1/copilot/query-voice"
TEST_TEXT_QUERY_PATH = "/api/v1/test/query"
TEST_VOICE_QUERY_PATH = "/api/v1/test/query-voice"
TEST_VOICE_ASSET_PATH = "/api/v1/test/assets/query-voice-ko.wav"
MAX_UPLOAD_BYTES = MAX_WAV_BYTES
SpeechToTextProvider = Callable[[], SpeechToText]
_AUDIO_ERRORS = {
    "audio_must_be_mono": "Invalid audio: WAV must be mono (1 channel)",
    "audio_sample_rate_out_of_range": "Invalid audio: WAV sample rate must be 8000 to 192000 Hz",
    "audio_too_large": "Invalid audio: WAV must be 6 MiB or smaller",
    "invalid_wav_metadata": "Invalid audio: WAV header and PCM metadata do not agree",
    "audio_must_be_pcm16": "Invalid audio: WAV must contain uncompressed 16-bit PCM samples",
    "audio_is_empty": "Invalid audio: WAV contains no samples",
    "audio_is_truncated": "Invalid audio: WAV sample data is truncated; upload the complete recording",
    "audio_too_long": "Invalid audio: recording must be 15 seconds or shorter",
}

_TEST_USER_ID = DEMO_USER_ID
_TEST_SESSION_ID = "SWAGGER-VOICE-DEMO"
_TEST_SCENARIO_ID = "training-mode-1"
_TEST_REQUEST_ID = "REQ-VOICE-TEST-001"
_TEST_SPOKEN_TEXT = "스프레이 도장 시 적정 이격거리를 알려주세요."
_TEST_TEXT_USER_ID = DEMO_USER_ID
_TEST_TEXT_SESSION_ID = "SWAGGER-TEXT-DEMO"
_TEST_TEXT_SCENARIO_ID = "training-mode-1"
_TEST_TEXT_REQUEST_ID = "REQ-TEXT-TEST-001"
_TEST_TEXT_PROMPT = "스프레이 도장 시 적정 이격거리를 알려주세요."


def _text_query_test_context(
    prompt: str,
    task_type: TaskType,
    response_mode: ResponseMode,
    use_rag: RagUsage,
    language: QueryLanguage = "ko",
    response_language: QueryLanguage | None = None,
    summarize: bool = True,
) -> dict[str, object]:
    return {
        "input_used": {
            "equivalent_real_api": "/api/v1/copilot/query",
            "query_parameters": {
                "prompt": prompt,
                "task_type": task_type,
                "response_mode": response_mode,
                "summarize": summarize,
                "use_rag": use_rag,
                "language": language,
                "response_language": response_language,
            },
            "fixed_context": {
                "user_id": _TEST_TEXT_USER_ID,
                "session_id": _TEST_TEXT_SESSION_ID,
                "scenario_id": _TEST_TEXT_SCENARIO_ID,
                "request_id": _TEST_TEXT_REQUEST_ID,
                "context": "Swagger 텍스트 Query 모의연동",
            },
        },
        "input_summary": (
            f"‘{prompt}’ 질문을 실제 텍스트 Query 처리 경로에 전달하며, "
            f"답변 모델은 {response_mode}, 요약 여부는 {str(summarize).lower()}, RAG 사용 여부는 {use_rag}입니다. "
            "모의훈련자 계정을 재사용하며 훈련 세션은 생성하지 않습니다."
        ),
        "evaluation_focus": (
            "answer가 질문에 직접 답하는지 확인하고, response_mode로 모델을, summarize로 핵심 요약과 전체 본문을 비교하세요."
        ),
    }


_TEXT_QUERY_TEST_RESPONSE = {
    200: {
        "description": "성공. 실제 Query 처리 결과와 테스트에 사용된 입력값을 함께 반환합니다.",
        "content": {
            "application/json": {
                "example": {
                    "status": "SUCCESS",
                    "code": 200,
                    "message": "Copilot query completed",
                    "data": {
                        "test_context": _text_query_test_context(
                            _TEST_TEXT_PROMPT, "knowledge_search", "quick", "Y"
                        ),
                        "request_id": _TEST_TEXT_REQUEST_ID,
                        "session_id": _TEST_TEXT_SESSION_ID,
                        "user_id": _TEST_TEXT_USER_ID,
                        "scenario_id": _TEST_TEXT_SCENARIO_ID,
                        "generated_at": "2026-10-27T05:30:00Z",
                        "latency_ms": 850,
                        "degraded": False,
                        "answer": "스프레이건과 피도면 사이의 권장 이격거리를 일정하게 유지하세요.",
                        "resolved_intents": ["knowledge"],
                        "context_status": "not_available",
                        "sequence_context": None,
                        "coating_context": None,
                        "pose_context": None,
                        "retrieval_used": True,
                        "citations": ["DOC-doc_0001-C0001"],
                    },
                }
            }
        },
    }
}


def _voice_test_context(response_language: QueryLanguage | None = None, summarize: bool = True) -> dict[str, object]:
    return {
        "input_used": {
            "equivalent_real_api": VOICE_QUERY_PATH,
            "audio": {
                "filename": "query-voice-ko.wav",
                "spoken_text": _TEST_SPOKEN_TEXT,
                "format": "WAV PCM16 mono",
                "sample_rate_hz": 16_000,
                "max_duration_seconds": 15,
                "download_url": TEST_VOICE_ASSET_PATH,
            },
            "form_fields": {
                "user_id": _TEST_USER_ID,
                "language": "ko",
                "response_language": response_language,
                "session_id": _TEST_SESSION_ID,
                "scenario_id": _TEST_SCENARIO_ID,
                "request_id": _TEST_REQUEST_ID,
                "task_type": "knowledge_search",
                "response_mode": "quick",
                "summarize": summarize,
                "use_rag": "Y",
                "context": "Swagger 음성질의 모의연동",
            },
        },
        "input_summary": (
            f"‘{_TEST_SPOKEN_TEXT}’라고 말한 한국어 WAV를 STT로 변환하고, "
            "변환 문장을 knowledge_search 질의로 전달합니다. "
            "모의훈련자 계정을 재사용하며 훈련 세션은 생성하지 않습니다."
        ),
        "evaluation_focus": (
            "transcript가 음성 문장과 일치하는지, answer가 적정 이격거리 질문에 답하는지, "
            "stt_latency_ms와 citations가 반환되는지 확인합니다."
        ),
    }


_VOICE_TEST_RESPONSE = {
    200: {
        "description": (
            "성공. `data.test_context.input_used`에서 자동 사용된 WAV와 폼 값을 확인하고, "
            "같은 응답의 `transcript`, `answer`, `citations`와 비교합니다."
        ),
        "content": {
            "application/json": {
                "example": {
                    "status": "SUCCESS",
                    "code": 200,
                    "message": "Copilot query completed",
                    "data": {
                        "test_context": _voice_test_context(),
                        "request_id": _TEST_REQUEST_ID,
                        "session_id": _TEST_SESSION_ID,
                        "user_id": _TEST_USER_ID,
                        "scenario_id": _TEST_SCENARIO_ID,
                        "transcript": _TEST_SPOKEN_TEXT,
                        "language": "ko",
                        "stt_latency_ms": 650,
                        "answer": "에어 스프레이 작업에서는 피도면과 약 15~20cm 거리를 유지하세요.",
                        "retrieval_used": True,
                        "citations": ["DOC-doc_0001-C0001"],
                    },
                }
            }
        },
    }
}


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
    reports: V1Store | None = None,
    sequence_engine: SequenceAnalysisEngine | None = None,
    coating_engine: object | None = None,
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
        use_rag: RagUsage,
        context: str | None,
        response_mode: ResponseMode = "quick",
        summarize: bool = True,
        response_language: QueryLanguage | None = None,
        conversation_id: str | None = None,
    ) -> JSONResponse:
        started = time.monotonic()
        if len(payload) > MAX_UPLOAD_BYTES:
            return error_response(413, "Audio file is too large")

        try:
            validated = await to_thread.run_sync(validate_wav, payload)
        except InvalidAudioError as exc:
            return error_response(422, _AUDIO_ERRORS.get(str(exc), "Invalid audio"))

        stt_started = time.monotonic()
        try:
            normalized = await to_thread.run_sync(normalize_wav, validated)
            speech_engine = await to_thread.run_sync(stt_provider)
            transcript_result = await to_thread.run_sync(
                lambda: speech_engine.transcribe(normalized.audio, language)
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
            language=language,
            response_language=response_language,
            use_rag=use_rag,
            context=context,
            response_mode=response_mode,
            summarize=summarize,
        )
        response = await execute_copilot_query(
            query,
            pipeline_provider=pipeline_provider,
            pending_answers=pending_answers,
            sequences=sequences,
            mcp_base_url=mcp_base,
            transport=transport,
            started=started,
            reports=reports,
            sequence_engine=sequence_engine,
            coating_engine=coating_engine,
            conversation_id=conversation_id,
        )
        if response.status_code != 200:
            return response

        body = json.loads(response.body)
        body["data"] = VoiceQueryData(**{
            **body["data"], "transcript": transcript, "language": language,
            "stt_latency_ms": stt_latency_ms,
        }).model_dump(mode="json")
        if language == "ko" and response_language is None:
            body["data"].pop("response_language", None)
        logger.info(
            "voice query done request_id=%s language=%s stt_latency_ms=%d",
            query.request_id,
            language,
            stt_latency_ms,
        )
        return JSONResponse(status_code=200, content=body, headers={
            key: value for key, value in response.headers.items() if key.lower().startswith("x-copilot-")
        })

    @router.post(
        VOICE_QUERY_PATH,
        response_model=VoiceCopilotResponse,
        tags=["3. AI Co-Pilot 질의"],
        summary="3-2. VR 음성 Co-Pilot 질의",
        description="HMD에서 녹음한 WAV를 STT로 변환한 뒤 기존 Query와 동일한 경로로 답변합니다. WAV PCM16·mono·15초 이하, 6MiB까지 허용합니다. 입력 샘플레이트는 8~192kHz이며 44.1kHz·48kHz 등은 서버에서 16kHz로 자동 변환합니다. 이미 16kHz인 음성은 그대로 사용합니다. 샘플레이트는 WAV 헤더에서 읽으므로 별도 입력값이 필요 없습니다. stt_latency_ms에는 변환·음성 인식 시간이 포함됩니다. language는 음성 인식 언어이며, response_language를 생략하면 같은 언어로 답합니다. 답변은 텍스트이고 TTS 음성 생성은 포함하지 않습니다. 413은 파일 크기, 422는 WAV 형식·샘플레이트·길이·인식된 발화 유무를 확인하세요. 503의 Speech recognition은 변환·STT 단계, Model/Query translation은 인식 이후 답변 단계의 실패입니다. 선택 헤더 X-Copilot-Conversation으로 텍스트 Query와 같은 대화 이력을 이어갈 수 있습니다.",
    )
    async def voice_query(
        request: Request,
        audio: Annotated[UploadFile, File(description="WAV PCM16·mono·최대 15초·6MiB. 8~192kHz 입력을 서버에서 16kHz로 자동 변환합니다(예: 44.1kHz, 48kHz). 샘플레이트는 WAV 헤더에서 읽습니다.")],
        user_id: Annotated[str, Form(min_length=1, description="음성 질문을 요청한 훈련자의 고유 식별자입니다.", examples=["1234"])],
        language: Annotated[VoiceLanguage, Form(description="업로드한 음성의 언어 코드(STT)입니다. " + LANGUAGE_DESCRIPTION, examples=["ko"])],
        session_id: Annotated[str | None, Form(description="진행 중인 세션 식별자(CSV 파일명에서 확장자 제외)", examples=["3_11000346_1"])] = None,
        scenario_id: Annotated[str | None, Form(description="session_id의 첫 '_' 앞 콘텐츠 식별자", examples=["3"])] = None,
        request_id: Annotated[str | None, Form(description="클라이언트 로그와 서버 응답을 연결하기 위한 호출 추적용 식별자입니다.", examples=["REQ-VOICE-001"])] = None,
        task_type: Annotated[TaskType, Form(description="텍스트 Query API와 동일한 작업 유형", examples=["knowledge_search"])] = "knowledge_search",
        response_mode: Annotated[ResponseMode, Form(description="quick은 Qwen3-8B, detailed는 Qwen3.6-27B 모델을 선택합니다. 답변 분량은 summarize로 선택합니다.")] = "quick",
        summarize: Annotated[bool, Form(description=SUMMARY_DESCRIPTION)] = True,
        use_rag: Annotated[RagUsage, Form(
            description="Y는 지식 검색 사용, N은 미사용입니다. 소문자 y/n도 허용하며 생략 시 Y입니다.",
            examples=["Y"],
        )] = "Y",
        response_language: Annotated[QueryLanguage | None, Form(
            description="답변 언어입니다. 생략 시 음성 language와 같은 언어로 답합니다. " + LANGUAGE_DESCRIPTION,
            examples=["ko"],
        )] = None,
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
            response_mode=response_mode,
            summarize=summarize,
            response_language=response_language,
            conversation_id=conversation_key(request.headers.get("x-copilot-conversation")),
        )

    @router.post(
        TEST_TEXT_QUERY_PATH,
        tags=["0. 모의 연동"],
        summary="0-5. 기본값이 채워진 텍스트 Query 테스트",
        description=(
            "실제 `POST /api/v1/copilot/query`와 동일한 처리 경로를 Swagger에서 간단히 시험합니다. "
            "모든 입력에는 기본값이 있어 수정하지 않고 바로 Execute할 수 있습니다.\n\n"
            f"모의훈련자(`{DEMO_USER_ID}`) 계정을 공통으로 사용하며 계정이 없으면 등록합니다. 훈련 세션은 생성하지 않습니다.\n\n"
            "### 기본 입력\n"
            f"질문은 **‘{_TEST_TEXT_PROMPT}’**, `task_type=knowledge_search`, "
            "`response_mode=quick`, `summarize=true`이며 RAG 기본값은 Y(지식 검색 사용)입니다.\n\n"
            "### 사용 방법\n"
            "먼저 기본값으로 실행한 뒤 `prompt`를 자유롭게 바꿔 보세요. "
            "`response_mode`로 모델을 바꾸고 `summarize=false`로 전체 본문을 비교할 수 있습니다.\n\n"
            "### 결과 읽는 법\n"
            "`data.test_context.input_used`에서 실제 사용된 값을 확인하고, "
            "`answer`, `retrieval_used`, `citations`를 함께 확인하세요."
        ),
        responses=_TEXT_QUERY_TEST_RESPONSE,
    )
    async def test_text_query(
        prompt: Annotated[str, Query(
            min_length=1,
            max_length=4000,
            description="시험할 질문입니다. 기본 질문을 그대로 실행하거나 원하는 문장으로 바꿀 수 있습니다.",
        )] = _TEST_TEXT_PROMPT,
        task_type: Annotated[TaskType, Query(
            description="실제 Query API와 동일한 작업 유형입니다.",
        )] = "knowledge_search",
        response_mode: Annotated[ResponseMode, Query(
            description="quick은 Qwen3-8B, detailed는 Qwen3.6-27B 모델입니다. 답변 분량은 summarize로 선택합니다.",
        )] = "quick",
        summarize: Annotated[bool, Query(description=SUMMARY_DESCRIPTION)] = True,
        language: Annotated[QueryLanguage, Query(
            description="질문 언어입니다. 생략 시 ko입니다. " + LANGUAGE_DESCRIPTION,
        )] = "ko",
        response_language: Annotated[QueryLanguage | None, Query(
            description="답변 언어입니다. 생략 시 질문 언어와 같습니다. " + LANGUAGE_DESCRIPTION,
            examples=["ko"],
        )] = None,
        use_rag: Annotated[RagUsage, Query(
            description="Y는 지식 검색 사용, N은 미사용입니다. 소문자 y/n도 허용하며 생략 시 Y입니다.",
        )] = "Y",
    ) -> JSONResponse:
        if reports is not None:
            ensure_demo_user(reports)
        query = CopilotQueryRequest(
            user_id=_TEST_TEXT_USER_ID,
            session_id=_TEST_TEXT_SESSION_ID,
            scenario_id=_TEST_TEXT_SCENARIO_ID,
            request_id=_TEST_TEXT_REQUEST_ID,
            prompt=prompt,
            language=language,
            response_language=response_language,
            task_type=task_type,
            response_mode=response_mode,
            summarize=summarize,
            use_rag=use_rag,
            context="Swagger 텍스트 Query 모의연동",
        )
        response = await execute_copilot_query(
            query,
            pipeline_provider=pipeline_provider,
            pending_answers=pending_answers,
            sequences=sequences,
            mcp_base_url=mcp_base,
            transport=transport,
            reports=reports,
            sequence_engine=sequence_engine,
            coating_engine=coating_engine,
        )
        if response.status_code != 200:
            return response
        body = json.loads(response.body)
        body["data"] = {
            "test_context": _text_query_test_context(
                prompt, task_type, response_mode, use_rag, language, response_language, summarize
            ),
            **body["data"],
        }
        return JSONResponse(status_code=200, content=body)

    @router.post(
        TEST_VOICE_QUERY_PATH,
        response_model=VoiceCopilotResponse,
        tags=["0. 모의 연동"],
        summary="0-6. 무입력 음성질의 전체 경로 테스트",
        description=(
            "파라미터나 파일을 입력하지 않고 서버에 포함된 한국어 WAV를 실제 STT와 Query 경로로 처리합니다.\n\n"
            "### 자동 사용 입력\n"
            f"음성 문장은 **‘{_TEST_SPOKEN_TEXT}’**이며, PCM16·16 kHz·mono WAV입니다. "
            f"모의훈련자 계정 `user_id={_TEST_USER_ID}`, `language=ko`, `task_type=knowledge_search`, `summarize=true`를 사용하며 RAG는 Y(지식 검색 사용)로 처리합니다. "
            "계정이 없으면 최초 등록하며 훈련 세션은 생성하지 않습니다.\n\n"
            "### 결과 읽는 법\n"
            "응답의 `data.test_context.input_used`에서 전체 입력을 먼저 보고, `transcript`에서 STT 결과, "
            "`answer`에서 Co-Pilot 답변, `citations`에서 검색 근거를 확인하세요.\n\n"
            "### 같은 입력으로 실제 API 시험\n"
            f"`GET {TEST_VOICE_ASSET_PATH}`에서 WAV를 내려받아 실제 3-2 API의 `audio` 필드에 업로드할 수 있습니다."
        ),
        responses=_VOICE_TEST_RESPONSE,
    )
    async def test_voice_query(
        summarize: Annotated[bool, Query(description=SUMMARY_DESCRIPTION)] = True,
        response_language: Annotated[QueryLanguage | None, Query(
            description="한국어 예제 음성에 대한 답변 언어입니다. 생략 시 한국어입니다. " + LANGUAGE_DESCRIPTION,
            examples=["ko"],
        )] = None,
    ) -> JSONResponse:
        if reports is not None:
            ensure_demo_user(reports)
        response = await _run_voice_query(
            _sample_wav(),
            user_id=_TEST_USER_ID,
            language="ko",
            session_id=_TEST_SESSION_ID,
            scenario_id=_TEST_SCENARIO_ID,
            request_id=_TEST_REQUEST_ID,
            task_type="knowledge_search",
            use_rag="Y",
            context="Swagger 음성질의 모의연동",
            response_language=response_language,
            summarize=summarize,
        )
        if response.status_code != 200:
            return response
        body = json.loads(response.body)
        body["data"] = {"test_context": _voice_test_context(response_language, summarize), **body["data"]}
        return JSONResponse(status_code=200, content=body)

    @router.get(
        TEST_VOICE_ASSET_PATH,
        tags=["0. 모의 연동"],
        summary="0-7. query-voice 테스트용 한국어 WAV 다운로드",
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
