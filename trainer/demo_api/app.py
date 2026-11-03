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
from typing import Any, cast

import httpx
import uvicorn
from anyio import to_thread
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from trainer.demo_api import chat_service as chat_service_mod
from trainer.demo_api.access_control import AccessControlMiddleware, OperatorCredentials
from trainer.demo_api.api_call_log import ApiCallStore, ApiCallLoggingMiddleware
from trainer.demo_api.chat_service import ChatService, build_default_live_runner
from trainer.demo_api.contracts import ChatRequest
from trainer.demo_api.copilot_router import create_copilot_router, install_copilot_error_handlers
from trainer.demo_api.coating_progress import CoatingProgressEngine
from trainer.demo_api.db_viewer import create_db_viewer_router
from trainer.demo_api.expert_guidance import ExpertGuidanceEngine
from trainer.demo_api.guided_workflow import create_guided_workflow_router
from trainer.demo_api.telemetry_router import create_telemetry_router
from trainer.demo_api.test_api_docs import add_test_input_overviews
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
_OPENAPI_METHODS = {"get", "post", "put", "patch", "delete", "options", "head"}

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
5. 텍스트 질의는 `0-5`, 음성 질의는 `0-6`에서 시험합니다. `0-7`에서는 실제 업로드용 WAV를 내려받을 수 있습니다.

> **DB 뷰어 바로가기:** [{public_base}/admin]({public_base}/admin) — 운영자 로그인 후 읽기 전용으로 조회합니다.

> **전문가 3D 모션 뷰어:** [{public_base}/motion-viewer]({public_base}/motion-viewer) — 전형기 반장님의 대표 실측 모션과 반복 수행의 통계 범위를 확인합니다.

### 실제 연동 API와 테스트 API의 차이

| 구분 | 용도 | 데이터 |
|---|---|---|
| `0. 모의 연동` | 사람의 기능 확인과 UI 개발 학습 | 서버가 안전한 예제값을 자동 생성 |
| `1~6. 실제 연동 API` | TSB VR·Replay가 운영 중 호출 | 시뮬레이터가 실제 사용자·시퀀스 값을 전송 |

테스트 API는 호출 방법을 익히기 위한 것이며, 실제 훈련 결과로 사용하지 않습니다.

### 실제 연동 기본 흐름

`세션 시작 → 현재 시퀀스 저장 또는 분석/가이드 → 텍스트·음성 질의 → 세션 종료 → 최종 리포트`

- **예시 세션:** 최신 TSB 전체 세션 샘플과 동일한 `session_id=3_11000346_1`, `user_id=1234`, `scenario_id=3`을 사용합니다.
- **세션 식별자:** `session_id`는 TSB가 생성하며 완료 시 첨부할 CSV 파일명에서 확장자를 뺀 값과 같습니다. `scenario_id`는 `session_id`의 첫 `_` 앞 콘텐츠 식별자입니다.
- **시퀀스 입력:** TSB는 Trigger ON부터 OFF까지의 **현재 시퀀스 텍스트 데이터**와 해당 시퀀스가 도포한 영역의 히트맵 크롭을 전송합니다.
- **누적 히트맵:** `cur_sequence_heatmap`의 크롭 내 픽셀값은 해당 시점 피도면의 **전체 누적 두께**입니다. 서버는 이전 값에 더하지 않고 최신 절대값으로 갱신합니다.
- **속도 상태:** `slow / ok / fast`, 이격거리 상태: `near / ok / far`입니다.
- **음성 입력:** WAV PCM16, 16 kHz, mono, 최대 15초입니다. STT 뒤 기존 Query와 같은 답변 경로를 사용합니다.
- **공통 응답:** `status / code / message / data`이며, 성공은 `status=SUCCESS`, 검증 실패는 주로 HTTP 422입니다.
- **멱등성:** 동일한 `session_id + sequence_id`와 동일 본문을 재전송하면 중복 저장하지 않습니다.
- **충돌 처리:** 같은 ID의 내용이 다르거나 새 시퀀스가 이미 처리한 시퀀스보다 이전 순서이면 HTTP 409를 반환합니다. 같은 내용의 정상 재시도는 기존 결과를 반환합니다.
- **리포트 재시도·복구:** 동일 완료 요청은 같은 report_id를 반환합니다. 원본 CSV와 높이맵을 보관하고 서버 재시작 후 중단된 분석을 다시 수행합니다.
- **목표 도막:** 세션 시작의 target_thickness_um을 입력하면 목표의 90~110% 두께 범위로 판정합니다. 생략하면 전문가 최종 도막 기준을 사용합니다. comparison_basis로 어느 기준인지 구분합니다.
- **훈련 중 질의:** 자세 중간 평가는 최근 분사 최대 3회의 표본임을 표시하고, Zone 순서는 현재까지 저장된 시퀀스를 사용합니다. 완료 리포트는 전체 세션을 평가합니다.
- **지원 범위:** 실시간 전문가 기준 안내는 학습모드 1~10을 대상으로 합니다. 실전·Unit은 데이터를 수신·보관하되 기준이 없는 분석은 not_available과 사유를 반환합니다. 블라스팅에는 도막 두께 판정을 적용하지 않습니다.
- **업로드 제한:** CSV/CSV.GZ 64MiB, 압축 해제 CSV 256MiB, 최종 높이맵 PNG 16MiB입니다.
- **접근 제어:** DB 뷰어는 로그인 없이 읽기 전용으로 조회합니다. 운영 로그는 운영자 인증이 필요합니다. TSB API키는 활성화한 서버에서만 X-API-Key 헤더로 전송하며 JSON 본문은 동일합니다.

각 API의 상세 설명에는 **언제 호출하는지, 입력값 의미, 응답 활용법과 주의사항**을 적었습니다.
"""


def _resolve_openapi_schema(
    node: dict[str, Any], components: dict[str, Any]
) -> dict[str, Any]:
    ref = node.get("$ref")
    if isinstance(ref, str):
        return cast(dict[str, Any], components.get(ref.rsplit("/", 1)[-1], node))
    return node


def _openapi_type_label(node: dict[str, Any], components: dict[str, Any]) -> str:
    node = _resolve_openapi_schema(node, components)
    if node.get("format") == "binary" or node.get("contentMediaType"):
        return "file"
    branches = [
        cast(dict[str, Any], branch)
        for branch in node.get("anyOf", [])
        if isinstance(branch, dict) and branch.get("type") != "null"
    ]
    if branches:
        labels = list(dict.fromkeys(_openapi_type_label(branch, components) for branch in branches))
        return " | ".join(labels)
    value_type = node.get("type")
    if value_type == "array":
        items = node.get("items")
        if isinstance(items, dict):
            item_type = _openapi_type_label(items, components)
        else:
            prefix_items = [
                item for item in node.get("prefixItems", []) if isinstance(item, dict)
            ]
            prefix_types = list(
                dict.fromkeys(_openapi_type_label(item, components) for item in prefix_items)
            )
            item_type = " | ".join(prefix_types) if prefix_types else "value"
        return f"array<{item_type}>"
    if value_type == "object" or "properties" in node:
        return "object"
    return str(value_type or "value")


def _clean_openapi_description(value: object, fallback_name: str) -> str:
    description = " ".join(str(value or "").split())
    return description or f"{fallback_name} 입력값입니다."


_OPENAPI_VALUE_GUIDANCE: dict[str, dict[str, str]] = {
    "response_mode": {
        "quick": "훈련 중 바로 확인할 핵심 판단과 권장 조치를 짧게 반환합니다.",
        "detailed": "현재 상황, 판단 근거, 원인과 권장 조치를 구조화해 자세히 반환합니다.",
    },
    "task_type": {
        "trajectory_recomm": "Zone 순서와 도장 경로를 안내합니다.",
        "pose_recomm": "현재 자세와 전문가 자세의 차이 및 개선 방향을 안내합니다.",
        "error_analysis": "현재 세션에서 감지된 이상 상태와 원인을 분석합니다.",
        "knowledge_search": "도장·블라스팅 관련 일반 질문에 답변합니다.",
    },
    "use_rag": {
        "auto": "서버가 질문 유형과 처리 경로에 따라 RAG 사용 여부를 자동 결정합니다.",
        "Y": "관련 지식과 현재 세션 문맥을 검색해 답변에 활용합니다.",
        "N": "별도 지식 검색 없이 전달된 현재 문맥을 중심으로 답변합니다.",
    },
}

_OPENAPI_OMISSION_GUIDANCE = {
    "use_rag": "서버가 질문 유형과 처리 경로에 따라 RAG 사용 여부를 자동 결정합니다.",
}

_OPENAPI_DYNAMIC_OBJECT_FIELDS: dict[str, tuple[tuple[str, str, str], ...]] = {
    "sequence_setting": (
        ("tool_type", "string", "사용 도구입니다. 스프레이는 spray_gun, 블라스팅은 blasting_gun을 사용합니다."),
        ("month", "integer", "시뮬레이터가 기록한 훈련 월입니다."),
        ("nozzle_size", "number", "시뮬레이터에 설정된 노즐 크기입니다."),
        ("thinning_ratio", "number", "시뮬레이터에 설정된 도료 희석 비율입니다."),
        ("pressure", "number", "시뮬레이터에 설정된 분사 압력입니다."),
        ("viscosity", "number", "시뮬레이터에 설정된 도료 점도입니다."),
    ),
    "tool_pos_rot": (
        ("position", "array<number>", "도구 또는 HMD의 월드 위치 [X,Y,Z]입니다."),
        ("rotation", "array<number>", "도구 또는 HMD의 회전값입니다."),
    ),
    "skeleton_joints": (
        ("{joint_name}", "array<number> | object", "관절 이름별 위치 또는 위치·회전 정보입니다."),
    ),
}


def _openapi_enum_values(node: dict[str, Any], components: dict[str, Any]) -> list[object]:
    node = _resolve_openapi_schema(node, components)
    values = node.get("enum")
    if isinstance(values, list):
        return values
    collected: list[object] = []
    for branch in node.get("anyOf", []):
        if not isinstance(branch, dict):
            continue
        resolved = _resolve_openapi_schema(branch, components)
        branch_values = resolved.get("enum")
        if isinstance(branch_values, list):
            collected.extend(branch_values)
    return list(dict.fromkeys(collected))


def _openapi_parameter_details(
    name: str, node: dict[str, Any], components: dict[str, Any]
) -> list[str]:
    node = _resolve_openapi_schema(node, components)
    details: list[str] = []
    if "default" in node and node["default"] is not None:
        details.append(f"    - 기본값: `{node['default']}`")
    omission = _OPENAPI_OMISSION_GUIDANCE.get(name)
    if omission:
        details.append(f"    - 생략 시: {omission}")

    values = _openapi_enum_values(node, components)
    guidance = _OPENAPI_VALUE_GUIDANCE.get(name, {})
    if guidance:
        for value in values:
            explanation = guidance.get(str(value))
            if explanation:
                details.append(f"    - `{value}`: {explanation}")
    elif values:
        allowed = ", ".join(f"`{value}`" for value in values)
        details.append(f"    - 허용값: {allowed}")
    return details


def _openapi_structured_schema(
    node: dict[str, Any], components: dict[str, Any]
) -> dict[str, Any]:
    resolved = _resolve_openapi_schema(node, components)
    if "properties" in resolved or resolved.get("type") == "array":
        return resolved
    for branch in resolved.get("anyOf", []):
        if not isinstance(branch, dict) or branch.get("type") == "null":
            continue
        candidate = _resolve_openapi_schema(branch, components)
        if "properties" in candidate or candidate.get("type") in {"array", "object"}:
            return candidate
    return resolved


def _openapi_nested_parameter_lines(
    name: str,
    node: dict[str, Any],
    components: dict[str, Any],
    *,
    depth: int = 0,
) -> list[str]:
    if depth > 4:
        return []
    schema = _openapi_structured_schema(node, components)
    if schema.get("type") == "array":
        items = schema.get("items")
        if isinstance(items, dict):
            return _openapi_nested_parameter_lines(
                f"{name}[]", items, components, depth=depth + 1
            )
        return []

    properties = schema.get("properties")
    if not isinstance(properties, dict) or not properties:
        dynamic = _OPENAPI_DYNAMIC_OBJECT_FIELDS.get(name.rsplit(".", 1)[-1], ())
        return [
            f"    - `{name}.{field_name}({field_type})` : {description} **[선택]**"
            for field_name, field_type, description in dynamic
        ]

    required_names = set(schema.get("required", []))
    lines: list[str] = []
    for child_name, child in properties.items():
        if not isinstance(child, dict):
            continue
        child_path = f"{name}.{child_name}"
        marker = "필수" if child_name in required_names else "선택"
        description = _clean_openapi_description(child.get("description"), child_path)
        constraints: list[str] = []
        resolved_child = _resolve_openapi_schema(child, components)
        if "default" in resolved_child and resolved_child["default"] is not None:
            constraints.append(f"기본값 `{resolved_child['default']}`")
        values = _openapi_enum_values(child, components)
        if values:
            constraints.append("허용값 " + ", ".join(f"`{value}`" for value in values))
        if constraints:
            description += " (" + "; ".join(constraints) + ")"
        lines.append(
            f"    - `{child_path}({_openapi_type_label(child, components)})` : "
            f"{description} **[{marker}]**"
        )
        lines.extend(
            _openapi_nested_parameter_lines(
                child_path, child, components, depth=depth + 1
            )
        )
    return lines


def _operation_input_lines(
    operation: dict[str, Any], components: dict[str, Any]
) -> list[str]:
    lines: list[str] = []
    seen: set[str] = set()

    for parameter in operation.get("parameters", []):
        if not isinstance(parameter, dict):
            continue
        parameter = _resolve_openapi_schema(parameter, components)
        name = str(parameter.get("name", "parameter"))
        value_schema = parameter.get("schema", {})
        if not isinstance(value_schema, dict):
            value_schema = {}
        required = bool(parameter.get("required"))
        marker = "필수" if required else "선택"
        description = _clean_openapi_description(parameter.get("description"), name)
        lines.append(
            f"- `{name}({_openapi_type_label(value_schema, components)})` : {description} **[{marker}]**"
        )
        lines.extend(_openapi_parameter_details(name, value_schema, components))
        lines.extend(_openapi_nested_parameter_lines(name, value_schema, components))
        seen.add(name)

    request_body = operation.get("requestBody", {})
    if isinstance(request_body, dict):
        content = request_body.get("content", {})
        if isinstance(content, dict) and content:
            preferred = next(
                (content[key] for key in ("application/json", "multipart/form-data") if key in content),
                next(iter(content.values())),
            )
            body_schema = preferred.get("schema", {}) if isinstance(preferred, dict) else {}
            if isinstance(body_schema, dict):
                body_schema = _resolve_openapi_schema(body_schema, components)
                required_names = set(body_schema.get("required", []))
                properties = body_schema.get("properties", {})
                if isinstance(properties, dict) and properties:
                    for name, prop in properties.items():
                        if name in seen or not isinstance(prop, dict):
                            continue
                        marker = "필수" if name in required_names else "선택"
                        description = _clean_openapi_description(prop.get("description"), name)
                        lines.append(
                            f"- `{name}({_openapi_type_label(prop, components)})` : {description} **[{marker}]**"
                        )
                        lines.extend(_openapi_parameter_details(name, prop, components))
                        lines.extend(_openapi_nested_parameter_lines(name, prop, components))
                elif body_schema:
                    marker = "필수" if request_body.get("required") else "선택"
                    description = _clean_openapi_description(body_schema.get("description"), "body")
                    lines.append(
                        f"- `body({_openapi_type_label(body_schema, components)})` : {description} **[{marker}]**"
                    )
    return lines


def _add_openapi_input_overviews(schema: dict[str, Any]) -> None:
    components = cast(dict[str, Any], schema.get("components", {}).get("schemas", {}))
    for path_item in schema.get("paths", {}).values():
        if not isinstance(path_item, dict):
            continue
        for method, operation in path_item.items():
            if method not in _OPENAPI_METHODS or not isinstance(operation, dict):
                continue
            description = str(operation.get("description", "")).rstrip()
            if "### 입력 파라미터" in description:
                continue
            lines = _operation_input_lines(operation, components)
            overview = "\n".join(lines) if lines else "- 없음"
            operation["description"] = f"{description}\n\n### 입력 파라미터\n\n{overview}".strip()


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
    api_call_store = ApiCallStore(SequenceStore._default_path())
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
    protect_admin = os.environ.get("COPILOT_PROTECT_ADMIN", "1" if live else "0") == "1"
    api_key = os.environ.get("COPILOT_API_KEY") or None
    credentials = OperatorCredentials(Path(os.environ.get(
        "COPILOT_OPERATOR_CREDENTIALS_PATH",
        str(Path(SequenceStore._default_path()).parent / "operator_credentials.json"),
    )))

    @asynccontextmanager
    async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
        if protect_admin:
            await to_thread.run_sync(credentials.ensure)
        await to_thread.run_sync(app.state.report_job_recovery)
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
    app.state.api_call_store = api_call_store
    app.state.speech_to_text = speech_to_text
    app.state.access_control = {"admin_protected": False, "monitor_protected": protect_admin, "api_key_required": bool(api_key)}
    app.add_middleware(AccessControlMiddleware, credentials=credentials, protect_admin=protect_admin, api_key=api_key)

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
    app.add_middleware(ApiCallLoggingMiddleware, store=api_call_store)

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
    # 모의연동 0-0~0-4 다음에 텍스트·음성 Query 0-5~0-7이 보이도록 guided 뒤에 등록한다.
    app.include_router(
        create_voice_router(
            (lambda: chat_service_mod._get_or_build_pipeline()) if live else None,
            stt_provider=lambda: speech_to_text,
            mcp_base_url=mcp_base_url,
            store=sequence_store,
            reports=v1_store,
            sequence_engine=sequence_analysis,
        )
    )
    v1_router = create_v1_router(
        store=v1_store,
        broadcaster=anomaly_broadcaster,
        sequences=sequence_store,
        sequence_engine=sequence_analysis,
        coating_engine=coating_progress,
        mcp_base_url=mcp_base_url,
    )
    app.state.report_job_recovery = v1_router.recover_report_jobs
    app.include_router(v1_router)
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
            session_store=v1_store,
        )
    )
    install_copilot_error_handlers(app)

    default_openapi = app.openapi

    def _openapi_with_input_overviews() -> dict[str, Any]:
        schema = default_openapi()
        _add_openapi_input_overviews(schema)
        add_test_input_overviews(schema)
        schema.setdefault("components", {}).setdefault("securitySchemes", {})["TSBApiKey"] = {
            "type": "apiKey", "in": "header", "name": "X-API-Key",
            "description": "서버에서 API 키를 활성화한 경우 사용합니다. JSON 본문은 변경하지 않습니다.",
        }
        if api_key:
            for path, operations in schema.get("paths", {}).items():
                if path.startswith("/api/v1/"):
                    for method, operation in operations.items():
                        if method in _OPENAPI_METHODS:
                            operation["security"] = [{"TSBApiKey": []}]
        return schema

    app.openapi = _openapi_with_input_overviews

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
