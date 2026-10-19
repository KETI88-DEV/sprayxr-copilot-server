# pyright: reportUnusedFunction=false, reportAny=false, reportUnusedCallResult=false
from __future__ import annotations

import argparse
import logging
import os
import re
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from trainer.demo_api import chat_service as chat_service_mod
from trainer.demo_api.chat_service import ChatService, build_default_live_runner
from trainer.demo_api.contracts import ChatRequest
from trainer.demo_api.copilot_router import create_copilot_router, install_copilot_error_handlers
from trainer.demo_api.coating_progress import CoatingProgressEngine
from trainer.demo_api.db_viewer import create_db_viewer_router
from trainer.demo_api.expert_guidance import ExpertGuidanceEngine
from trainer.demo_api.guided_workflow import create_guided_workflow_router
from trainer.demo_api.telemetry_router import create_telemetry_router
from trainer.demo_api.anomaly_stream import AnomalyBroadcaster
from trainer.demo_api.sequence_analysis import SequenceAnalysisEngine
from trainer.demo_api.sequence_store import SequenceStore
from trainer.demo_api.v1_router import create_v1_router
from trainer.demo_api.v1_store import V1Store
from trainer.demo_api.monitor import MonitorRecorder
from trainer.demo_api.monitor_api import create_monitor_router
from trainer.demo_api.motion_viewer import create_motion_viewer_router
from trainer.demo_api.readiness import evaluate_live_readiness
from trainer.demo_api.sequence import default_snapshot, session_snapshot
from trainer.demo_api.sim_session import SimSessionStore
from trainer.demo_api.simulator import create_simulator_router
from trainer.demo_api.speech_to_text import LazySpeechToText, TransformersWhisperSpeechToText
from trainer.demo_api.voice_router import create_voice_router
from trainer.demo_api.zone_resolver import McpZoneResolver

logger = logging.getLogger(__name__)
_LEVEL_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9\-_.]{0,63}$")

OPENAPI_TAGS = [
    {"name": "0. 모의 연동", "description": "시뮬레이터 없이 0-reset부터 4-report까지 순서대로 시험합니다."},
    {"name": "1. 세션 관리", "description": "훈련 세션 시작·종료 API입니다."},
    {"name": "2. 시퀀스 수집·피드백", "description": "현재 시퀀스 전송과 속도·이격거리 가이드를 제공합니다."},
    {"name": "3. AI Co-Pilot 질의", "description": "텍스트 질의와 VR용 음성 질의 API입니다."},
    {"name": "4. 실시간 알림", "description": "SSE 이상 알림 스트림입니다."},
    {"name": "5. 최종 평가·리포트", "description": "훈련 완료와 리포트 조회 API입니다."},
    {"name": "6. 사용자 관리", "description": "훈련자 등록·조회 API입니다."},
]

def _openapi_description() -> str:
    public_base = os.environ.get(
        "PUBLIC_BASE_URL", "http://keti-xr-lab.iptime.org:58000"
    ).rstrip("/")
    return f"""
## TSB 연동용 AI Co-Pilot API

이 문서는 **이 서버를 처음 보는 개발자**도 VR 시뮬레이터 없이 전체 흐름을 이해하고 시험할 수 있도록 구성했습니다.

### 바로 시작하기

1. 아래 **0. 모의 연동**을 펼칩니다.
2. `0-0 → 0-1 → 0-2 → 0-3 → 0-4` 순서로 각 API의 **Try it out → Execute**를 누릅니다.
3. 입력값은 필요 없습니다. 각 응답의 `what_happened`, `beginner_tip`, `next_endpoint`를 확인합니다.
4. 저장 결과는 [Co-Pilot DB 뷰어]({public_base}/admin)에서 확인합니다.
5. 음성 기능은 `0-5`를 실행하면 파일 업로드 없이 STT부터 Query 답변까지 확인할 수 있습니다. `0-6`에서는 실제 업로드용 WAV를 내려받을 수 있습니다.

> **DB 뷰어 바로가기:** [{public_base}/admin]({public_base}/admin) — 읽기 전용이며 데이터를 수정하지 않습니다.

> **전문가 3D 모션 뷰어:** [{public_base}/motion-viewer]({public_base}/motion-viewer) — 전형기 반장님의 대표 실측 모션과 반복 수행의 통계 범위를 확인합니다.

### 실제 연동 API와 테스트 API의 차이

| 구분 | 용도 | 데이터 |
|---|---|---|
| `0. 모의 연동` | 사람의 기능 확인과 UI 개발 학습 | 서버가 안전한 예제값을 자동 생성 |
| `1~6. 실제 연동 API` | TSB VR·Replay가 운영 중 호출 | 시뮬레이터가 실제 사용자·시퀀스 값을 전송 |

테스트 API는 호출 방법을 익히기 위한 것이며, 실제 훈련 결과로 사용하지 않습니다.

### 실제 연동 기본 흐름

`세션 시작 → 현재 시퀀스 전송 → 시퀀스 분석/가이드 → 텍스트·음성 질의 → 세션 종료 → 최종 리포트`

- **예시 세션:** 최신 TSB 전체 세션 샘플과 동일한 `session_id=3_11000346_1`, `user_id=1234`, `scenario_id=3`을 사용합니다.
- **세션 식별자:** `session_id`는 TSB가 생성하며 완료 시 첨부할 CSV 파일명에서 확장자를 뺀 값과 같습니다. `scenario_id`는 `session_id`의 첫 `_` 앞 콘텐츠 식별자입니다.
- **시퀀스 입력:** TSB는 Trigger ON부터 OFF까지의 **현재 시퀀스 텍스트 데이터**와 해당 시퀀스가 도포한 영역의 히트맵 크롭을 전송합니다.
- **누적 히트맵:** `cur_sequence_heatmap`의 크롭 내 픽셀값은 해당 시점 피도면의 **전체 누적 두께**입니다. 서버는 이전 값에 더하지 않고 최신 절대값으로 갱신합니다.
- **속도 상태:** `slow / ok / fast`, 이격거리 상태: `near / ok / far`입니다.
- **음성 입력:** WAV PCM16, 16 kHz, mono, 최대 15초입니다. STT 뒤 기존 Query와 같은 답변 경로를 사용합니다.
- **공통 응답:** `status / code / message / data`이며, 성공은 `status=SUCCESS`, 검증 실패는 주로 HTTP 422입니다.
- **멱등성:** 동일한 `session_id + sequence_id`와 동일 본문을 재전송하면 중복 저장하지 않습니다.

각 API의 상세 설명에는 **언제 호출하는지, 입력값 의미, 응답 활용법과 주의사항**을 적었습니다.
"""


def create_app(
    chat_service: ChatService | None = None,
    *,
    probe_transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    recorder = MonitorRecorder()
    store = SimSessionStore()
    mcp_base_url = os.environ.get("MCP_BASE_URL", "http://localhost:8100")
    resolver = McpZoneResolver(mcp_base_url)
    expert_guidance = ExpertGuidanceEngine(
        Path(os.environ.get("EXPERT_REFERENCE_ROOT", "/expert_reference_data"))
    )
    coating_progress = CoatingProgressEngine(expert_guidance.reference_root)
    v1_store = V1Store()
    sequence_store = SequenceStore()
    anomaly_broadcaster = AnomalyBroadcaster()
    sequence_analysis = SequenceAnalysisEngine(expert_guidance.reference_root)
    speech_to_text = LazySpeechToText(
        lambda: TransformersWhisperSpeechToText(
            os.environ.get("STT_MODEL_PATH", "/stt-model"),
            device=os.environ.get("STT_DEVICE", "cuda"),
        )
    )

    if chat_service is not None:
        service = chat_service
    else:
        live_mode = os.environ.get("DEMO_LIVE_MODE", "off")
        if live_mode not in {"off", "on"}:
            raise RuntimeError("DEMO_LIVE_MODE must be 'off' or 'on'")

        logger.info("DEMO_LIVE_MODE=%s", live_mode)

        if live_mode == "on":
            service = ChatService(live_runner=build_default_live_runner(store))
        else:
            service = ChatService()
    live = service.live_runner is not None

    @asynccontextmanager
    async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
        pipeline_init_thread: threading.Thread | None = None
        if live and os.environ.get("LIVE_EAGER_INIT") == "1":
            pipeline_init_thread = chat_service_mod.start_eager_pipeline_init()
            app.state.eager_init_thread = pipeline_init_thread
        if live and os.environ.get("STT_EAGER_INIT", "0") == "1":
            def _preload_stt() -> None:
                try:
                    # Both runtimes import transformers. Importing its lazy modules from
                    # two startup threads can expose a partially initialized module.
                    if pipeline_init_thread is not None:
                        pipeline_init_thread.join()
                    speech_to_text.preload()
                except Exception:
                    logger.error("STT preload failed")

            app.state.stt_init_thread = threading.Thread(
                target=_preload_stt,
                name="stt-preload",
                daemon=True,
            )
            app.state.stt_init_thread.start()
        yield

    app = FastAPI(
        title="XR Painting Simulator AI Co-Pilot API",
        description=_openapi_description(),
        version="2.6.0",
        openapi_tags=OPENAPI_TAGS,
        lifespan=_lifespan,
    )
    app.state.chat_service = service
    app.state.eager_init_thread = None
    app.state.stt_init_thread = None
    app.state.expert_guidance = expert_guidance
    app.state.coating_progress = coating_progress
    app.state.v1_store = v1_store
    app.state.sequence_store = sequence_store
    app.state.speech_to_text = speech_to_text

    raw_allow_origins = os.environ.get("DEMO_UI_ORIGINS")
    if raw_allow_origins is None:
        allow_origins = ["*"]
    else:
        allow_origins = [origin.strip() for origin in raw_allow_origins.split(",") if origin.strip()]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # 계약 밖. Swagger 는 계약 15개만 보여야 한다 — demo-ui 와 compose 헬스체크가 쓰므로 라우트는 남긴다.
    @app.get("/api/health", include_in_schema=False)
    def health() -> dict[str, object]:
        return {
            "status": "ok",
            "servers": 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    @app.get("/api/ready", include_in_schema=False)
    async def ready() -> JSONResponse:
        body: dict[str, object] = {"ready": True, "mode": "live" if live else "demo", "checks": {}}
        if live:
            is_ready, checks = await evaluate_live_readiness(
                pipeline=chat_service_mod.loaded_pipeline(),
                mcp_base_url=mcp_base_url,
                transport=probe_transport,
            )
            body.update(ready=is_ready, checks=checks)
            if os.environ.get("STT_REQUIRED", "0") == "1":
                stt_state = "ready" if speech_to_text.ready else ("failed" if speech_to_text.failed else "loading")
                cast(dict[str, object], body["checks"])["speech_to_text"] = {
                    "ok": speech_to_text.ready,
                    "state": stt_state,
                }
                body["ready"] = bool(body["ready"]) and speech_to_text.ready
        body["timestamp"] = datetime.now(timezone.utc).isoformat()
        return JSONResponse(status_code=200 if body["ready"] else 503, content=body)

    @app.get("/api/sequence/{level_id}", include_in_schema=False)
    def sequence(level_id: str) -> dict[str, object]:
        if _LEVEL_ID_PATTERN.fullmatch(level_id) is None:
            raise HTTPException(status_code=400, detail="invalid level_id")
        state = store.current()
        if state is not None and state.level_id == level_id:
            return session_snapshot(state).model_dump(mode="json")
        return default_snapshot(level_id).model_dump(mode="json")

    @app.post("/api/chat", include_in_schema=False)
    async def chat(request: ChatRequest) -> StreamingResponse:
        service = cast(ChatService, app.state.chat_service)

        async def _sse_body() -> AsyncIterator[bytes]:
            async for event in service.stream(request):
                frame = f"event: {event.event}\ndata: {event.model_dump_json()}\n\n"
                yield frame.encode("utf-8")

        return StreamingResponse(
            _sse_body(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    app.state.monitor_recorder = recorder
    app.state.sim_session_store = store
    # v2.5 이전 데모 인터페이스. 계약의 /sessions/*·/telemetry/sequence 가 대응한다.
    app.include_router(create_simulator_router(recorder, store, resolver), include_in_schema=False)
    app.include_router(create_monitor_router(recorder), include_in_schema=False)
    app.include_router(create_db_viewer_router())
    app.include_router(
        create_motion_viewer_router(
            expert_guidance.reference_root,
            sequence_store=sequence_store,
            mcp_base_url=mcp_base_url,
            reports=v1_store,
        )
    )
    # knowledge_search shares the live pipeline singleton; resolved per call so it is built once, lazily.
    app.include_router(
        create_copilot_router(
            (lambda: chat_service_mod._get_or_build_pipeline()) if live else None,
            mcp_base_url=mcp_base_url,
            store=sequence_store,
            reports=v1_store,
            sequence_engine=sequence_analysis,
        )
    )
    # 명세서 15개 중 12개: 세션·사용자·리포트·이상 알림 SSE·테스트.
    guided_router, guided_workflow = create_guided_workflow_router(v1_store, expert_guidance)
    app.state.guided_workflow = guided_workflow
    app.include_router(guided_router)
    # 모의연동 0-0~0-4 다음에 음성 0-5~0-6이 보이도록 guided 뒤에 등록한다.
    app.include_router(
        create_voice_router(
            (lambda: chat_service_mod._get_or_build_pipeline()) if live else None,
            stt_provider=lambda: speech_to_text,
            mcp_base_url=mcp_base_url,
            store=sequence_store,
            reports=v1_store,
        )
    )
    app.include_router(create_v1_router(
        store=v1_store,
        broadcaster=anomaly_broadcaster,
        sequences=sequence_store,
        sequence_engine=sequence_analysis,
        coating_engine=coating_progress,
        mcp_base_url=mcp_base_url,
    ))
    # POST /api/v1/telemetry/sequence: 시퀀스를 데이터베이스에 올린다. 자세 채점은
    # copilot/query 의 pose_recomm 이 한다(명세서 p.4 vs p.9).
    # recorder 를 함께 준다 — v2.5 요청·응답이 /api/monitor/{events,stream} 에 보인다.
    app.include_router(
        create_telemetry_router(
            store=sequence_store,
            recorder=recorder,
            guidance_engine=expert_guidance,
            sequence_engine=sequence_analysis,
            broadcaster=anomaly_broadcaster,
            coating_progress_engine=coating_progress,
        )
    )
    install_copilot_error_handlers(app)

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--host", default="0.0.0.0")
    _ = parser.add_argument("--port", default=8120, type=int)
    _ = parser.add_argument("--reload", action="store_true")
    _ = parser.add_argument("--live", action="store_true")
    raw_args = cast(dict[str, object], vars(parser.parse_args()))
    host = cast(str, raw_args.get("host", "0.0.0.0"))
    port = cast(int, raw_args.get("port", 8120))
    reload_mode = cast(bool, raw_args.get("reload", False))
    live_mode = cast(bool, raw_args.get("live", False))

    if live_mode:
        os.environ["DEMO_LIVE_MODE"] = "on"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logger.info("Starting demo API on %s:%s", host, port)
    uvicorn.run("trainer.demo_api.app:app", host=host, port=port, reload=reload_mode)


app = create_app()


if __name__ == "__main__":
    main()
