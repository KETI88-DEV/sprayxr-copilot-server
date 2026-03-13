#!/usr/bin/env python3
"""Ship Painting AI Co-Pilot Gradio Demo Application.

A web-based demonstration of the agentic pipeline for ship coating quality
analysis, real-time coaching, and tool integration. Supports DEMO_MODE (dummy
data, no GPU) and LIVE mode (real AgentPipeline with QLoRA + MCP).

Usage:
    DEMO_MODE=true  python3 demo_app.py   # default — dummy data
    DEMO_MODE=false python3 demo_app.py   # real pipeline (GPU required)
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any, Generator

import gradio as gr
from gradio import ChatMessage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEMO_MODE: bool = os.environ.get("DEMO_MODE", "true").lower() in ("true", "1", "yes")
APP_PORT: int = int(os.environ.get("APP_PORT", "7860"))
MCP_BASE_URL: str = os.environ.get("MCP_BASE_URL", "http://localhost:8100")
CONFIG_PATH: str = os.environ.get(
    "CONFIG_PATH", "/workspace/trainer/configs/rag_config_fewshot.yaml"
)
ADAPTER_PATH: str = os.environ.get(
    "ADAPTER_PATH", "/output/adapters/qwen3_4b_shipcoat_lora/"
)
MODEL_NAME: str = "Qwen3-4B (4-bit QLoRA)"

# ---------------------------------------------------------------------------
# Theme — dark navy / steel industrial
# ---------------------------------------------------------------------------

CUSTOM_THEME = gr.themes.Base(
    primary_hue=gr.themes.Color(
        c50="#e8eaf6", c100="#c5cae9", c200="#9fa8da", c300="#7986cb",
        c400="#5c6bc0", c500="#0f3460", c600="#0d2d56", c700="#0b264b",
        c800="#091e3d", c900="#07162e", c950="#051020",
    ),
    secondary_hue=gr.themes.Color(
        c50="#fce4ec", c100="#f8bbd0", c200="#f48fb1", c300="#f06292",
        c400="#ec407a", c500="#e94560", c600="#d81b60", c700="#c2185b",
        c800="#ad1457", c900="#880e4f", c950="#6a0a3d",
    ),
    neutral_hue=gr.themes.Color(
        c50="#e8eaf6", c100="#c5cae9", c200="#9fa8da", c300="#7986cb",
        c400="#5c6bc0", c500="#3f4865", c600="#353d58", c700="#2b314b",
        c800="#1a1a2e", c900="#16213e", c950="#0f1628",
    ),
    font=gr.themes.GoogleFont("Inter"),
).set(
    body_background_fill="#1a1a2e",
    body_background_fill_dark="#0f1628",
    body_text_color="#e0e0e0",
    body_text_color_dark="#e0e0e0",
    block_background_fill="#16213e",
    block_background_fill_dark="#16213e",
    block_border_color="#0f3460",
    block_border_color_dark="#0f3460",
    block_label_text_color="#9fa8da",
    block_label_text_color_dark="#9fa8da",
    block_title_text_color="#e0e0e0",
    block_title_text_color_dark="#e0e0e0",
    input_background_fill="#1a1a2e",
    input_background_fill_dark="#1a1a2e",
    input_border_color="#0f3460",
    input_border_color_dark="#0f3460",
    button_primary_background_fill="#0f3460",
    button_primary_background_fill_dark="#0f3460",
    button_primary_background_fill_hover="#e94560",
    button_primary_background_fill_hover_dark="#e94560",
    button_primary_text_color="#ffffff",
    button_primary_text_color_dark="#ffffff",
    button_secondary_background_fill="#16213e",
    button_secondary_background_fill_dark="#16213e",
    button_secondary_text_color="#9fa8da",
    button_secondary_text_color_dark="#9fa8da",
    border_color_primary="#0f3460",
    border_color_primary_dark="#0f3460",
    color_accent_soft="#0f3460",
    color_accent_soft_dark="#0f3460",
)

CUSTOM_CSS = """
/* Global overrides */
.gradio-container { max-width: 1400px !important; }
footer { display: none !important; }

/* Sidebar panel */
.sidebar-panel {
    background: #16213e !important;
    border: 1px solid #0f3460 !important;
    border-radius: 12px !important;
    padding: 16px !important;
}
.sidebar-panel .label-wrap { color: #9fa8da !important; }

/* Status badges */
.status-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.85em;
    font-weight: 600;
}
.status-demo {
    background: #e94560;
    color: #fff;
}
.status-live {
    background: #00c853;
    color: #1a1a2e;
}

/* Example buttons */
.example-btn {
    text-align: left !important;
    font-size: 0.9em !important;
    border: 1px solid #0f3460 !important;
    background: #1a1a2e !important;
    color: #c5cae9 !important;
    border-radius: 8px !important;
    padding: 10px 14px !important;
    transition: all 0.2s !important;
}
.example-btn:hover {
    background: #0f3460 !important;
    border-color: #e94560 !important;
    color: #ffffff !important;
}

/* Chatbot styling */
.chatbot-container .message {
    border-radius: 12px !important;
}

/* Title styling */
.title-section {
    text-align: center;
    padding: 12px 0 4px 0;
}
.title-section h1 {
    color: #e0e0e0 !important;
    margin-bottom: 2px !important;
    font-size: 1.8em !important;
}
.title-section p {
    color: #7986cb !important;
    font-size: 1.05em !important;
    margin-top: 0 !important;
}
"""

# ---------------------------------------------------------------------------
# Demo Scenarios — pre-computed Korean domain responses
# ---------------------------------------------------------------------------

EXAMPLE_QUESTIONS: list[dict[str, str]] = [
    {
        "text": "DFT 측정값 [95, 102, 88, 105, 99]로 nominal 100um 기준 합부 판정해줘",
        "display_text": "DFT 합부 판정 요청",
        "icon": "🎯",
    },
    {
        "text": "히트맵에서 좌하단에 저도막 패턴이 보여. DFT는 [85, 92, 78]이고 온도 28도, 습도 65%. 원인 분석해줘",
        "display_text": "이상 탐지 · 원인 분석",
        "icon": "🔥",
    },
    {
        "text": "현재 구역의 도장 순서가 맞는지 확인해줘",
        "display_text": "도장 순서 검증",
        "icon": "📋",
    },
    {
        "text": "실시간 코칭 모드로 현재 작업자에게 자세/순서/속도 종합 피드백을 줘",
        "display_text": "실시간 종합 코칭",
        "icon": "🏋️",
    },
    {
        "text": "습도가 85%이고 이슬점 차이가 2°C일 때 도장 작업을 진행해도 되는가?",
        "display_text": "작업 환경 적합성 판단",
        "icon": "🌡️",
    },
]

ChatHistory = list[ChatMessage]


def _classify_question(text: str) -> str:
    """Classify question into a demo scenario type."""
    t = text.lower()
    if any(k in t for k in ("dft", "합부", "판정", "측정값", "nominal")):
        return "dft_judge"
    if any(k in t for k in ("이상", "히트맵", "패턴", "원인", "anomaly")):
        return "anomaly"
    if any(k in t for k in ("순서", "검증", "sequence", "공정")):
        return "sequence"
    if any(k in t for k in ("코칭", "자세", "포즈", "피드백", "속도")):
        return "coaching"
    if any(k in t for k in ("습도", "이슬점", "온도", "환경", "작업 가능")):
        return "environment"
    return "general"


DEMO_SCENARIOS: dict[str, dict[str, Any]] = {
    "dft_judge": {
        "rag_chunks": [
            {
                "chunk_id": "SPEC-DFT-001",
                "text": "SSPC-PA2 기준 DFT 합부 판정: nominal ±20% 이내, 개별 측정값 중 90% 이상이 규격 이내이며 최소값은 nominal의 80% 이상이어야 합격.",
                "rerank_score": 0.94,
            },
            {
                "chunk_id": "QC-PROC-012",
                "text": "건조 도막 두께(DFT) 측정 시 게이지 교정 확인 필수. 각 구역 당 최소 5회 측정, 평균 및 개별 편차 기록.",
                "rerank_score": 0.87,
            },
        ],
        "tool_calls": [
            {
                "name": "dft_judge",
                "args": {
                    "measurements": [95, 102, 88, 105, 99],
                    "nominal": 100,
                    "tolerance_pct": 20,
                },
                "result": {
                    "verdict": "PASS",
                    "mean": 97.8,
                    "std": 6.14,
                    "min": 88,
                    "max": 105,
                    "out_of_spec_count": 0,
                    "details": "전 측정값이 80~120um 범위 이내. 평균 97.8um (nominal 대비 -2.2%). 합격.",
                },
            }
        ],
        "response": (
            "## DFT 합부 판정 결과\n\n"
            "**판정: ✅ 합격 (PASS)**\n\n"
            "| 항목 | 값 |\n|---|---|\n"
            "| 측정값 | [95, 102, 88, 105, 99] um |\n"
            "| 평균 DFT | 97.8 um |\n"
            "| 표준편차 | 6.14 um |\n"
            "| 최소/최대 | 88 / 105 um |\n"
            "| 규격 범위 | 80 ~ 120 um (nominal 100 ± 20%) |\n"
            "| 규격 이탈 | 0건 |\n\n"
            "SSPC-PA2 기준에 따라 모든 측정값이 허용 범위(80~120um) 이내이며, "
            "평균값 97.8um은 nominal 100um 대비 -2.2%로 양호합니다.\n\n"
            "**권고사항:** 현재 도막 상태 양호하나, 88um 측정 지점은 하한에 가까우므로 "
            "해당 구역 재측정 후 필요 시 보수 도장을 고려하세요."
        ),
    },
    "anomaly": {
        "rag_chunks": [
            {
                "chunk_id": "DEFECT-ANALYSIS-003",
                "text": "저도막 패턴 원인 분류: (1) 스프레이 건 이동속도 과다, (2) 도료 점도 부족 (희석 과다), (3) 노즐 마모에 의한 패턴 편향, (4) 도장 거리 부적합.",
                "rerank_score": 0.91,
            },
            {
                "chunk_id": "ENV-GUIDE-007",
                "text": "온도 28°C, 습도 65% 조건은 대부분 도료의 적용 가능 범위이나, 고온 시 도료 건조가 빨라져 도막 두께 불균일 위험 증가. 수성 도료는 습도 60% 초과 시 주의.",
                "rerank_score": 0.85,
            },
        ],
        "tool_calls": [
            {
                "name": "anomaly_detection",
                "args": {
                    "dft_values": [85, 92, 78],
                    "nominal": 100,
                    "env_temperature": 28,
                    "env_humidity": 65,
                },
                "result": {
                    "anomaly_detected": True,
                    "anomaly_type": "저도막 (under-thickness)",
                    "severity": "WARNING",
                    "affected_measurements": [85, 78],
                    "mean_dft": 85.0,
                    "deviation_from_nominal": -15.0,
                    "probable_causes": [
                        "스프레이 이동속도 과다 (좌하단 편향 패턴)",
                        "도장 거리 초과 가능성",
                    ],
                },
            },
        ],
        "response": (
            "## 이상 탐지 분석 결과\n\n"
            "**상태: ⚠️ 저도막 이상 감지 (WARNING)**\n\n"
            "### 측정 데이터 분석\n"
            "| 항목 | 값 |\n|---|---|\n"
            "| 측정값 | [85, 92, 78] um |\n"
            "| 평균 DFT | 85.0 um |\n"
            "| Nominal 대비 편차 | -15.0% |\n"
            "| 이상 측정값 | 85um, 78um |\n"
            "| 환경 조건 | 온도 28°C / 습도 65% |\n\n"
            "### 원인 분석\n"
            "히트맵 좌하단 저도막 패턴과 DFT 데이터를 종합 분석한 결과:\n\n"
            "1. **스프레이 건 이동속도 과다** — 좌하단 구역에서 이동속도가 빨라 "
            "도막이 충분히 형성되지 않은 것으로 판단\n"
            "2. **도장 거리 초과 가능성** — 해당 구역이 수직 패널 하부로, "
            "작업자의 팔 도달 거리 한계로 인해 노즐 거리가 과도했을 수 있음\n\n"
            "### 조치 권고\n"
            "- 좌하단 구역 보수 도장 실시 (목표 DFT: 100um 이상)\n"
            "- 해당 구역 작업 시 이동속도 20% 감소 및 오버랩 비율 50% → 60% 조정\n"
            "- 온도 28°C 조건에서 도료 점도 재확인 (희석률 점검)"
        ),
    },
    "sequence": {
        "rag_chunks": [
            {
                "chunk_id": "PROC-SEQ-001",
                "text": "표준 도장 공정 순서: (1) 표면처리(블라스팅) → (2) 분진 제거 → (3) 프라이머 도장 → (4) 건조 확인 → (5) 중도 도장 → (6) 건조 확인 → (7) 상도 도장 → (8) 최종 검사",
                "rerank_score": 0.96,
            },
        ],
        "tool_calls": [
            {
                "name": "sequence_validation",
                "args": {
                    "zone_id": "B-3",
                    "current_step": "중도 도장",
                    "completed_steps": ["표면처리", "분진 제거", "프라이머 도장", "건조 확인"],
                },
                "result": {
                    "valid": True,
                    "current_step": "중도 도장",
                    "step_number": 5,
                    "total_steps": 8,
                    "next_step": "건조 확인 (중도 후)",
                    "skipped_steps": [],
                    "message": "공정 순서 적합. 현재 5단계 진행 중.",
                },
            },
        ],
        "response": (
            "## 도장 순서 검증 결과\n\n"
            "**상태: ✅ 공정 순서 적합**\n\n"
            "### 현재 진행 현황\n"
            "| 단계 | 공정 | 상태 |\n|---|---|---|\n"
            "| 1 | 표면처리 (블라스팅) | ✅ 완료 |\n"
            "| 2 | 분진 제거 | ✅ 완료 |\n"
            "| 3 | 프라이머 도장 | ✅ 완료 |\n"
            "| 4 | 건조 확인 | ✅ 완료 |\n"
            "| **5** | **중도 도장** | **🔄 진행 중** |\n"
            "| 6 | 건조 확인 (중도 후) | ⬜ 대기 |\n"
            "| 7 | 상도 도장 | ⬜ 대기 |\n"
            "| 8 | 최종 검사 | ⬜ 대기 |\n\n"
            "현재 B-3 구역은 5단계 '중도 도장' 진행 중이며, 이전 공정이 모두 "
            "정상적으로 완료되었습니다. 건너뛴 공정 없음.\n\n"
            "**다음 단계:** 중도 도장 완료 후 건조 상태 확인 (건조 시간 최소 4시간, "
            "온도 및 습도 기준 충족 후 상도 도장 진행)"
        ),
    },
    "coaching": {
        "rag_chunks": [
            {
                "chunk_id": "COACH-POSE-002",
                "text": "스프레이 도장 표준 자세: 팔꿈치 각도 90~110°, 손목 중립, 도장면과 스프레이 건 수직 유지. 이동 시 어깨 중심 회전, 허리 비틀림 최소화.",
                "rerank_score": 0.89,
            },
            {
                "chunk_id": "COACH-SPEED-005",
                "text": "적정 스프레이 이동속도: 0.3~0.5 m/s. 0.5 m/s 초과 시 저도막 위험, 0.3 m/s 미만 시 과도막 및 흘림(sagging) 발생 가능.",
                "rerank_score": 0.86,
            },
        ],
        "tool_calls": [
            {
                "name": "pose_estimation",
                "args": {
                    "current_pose": {"wrist_angle": 15.2, "elbow_angle": 95.0, "distance_to_surface": 28.5},
                    "reference_pose": {"wrist_angle": 0.0, "elbow_angle": 100.0, "distance_to_surface": 25.0},
                },
                "result": {
                    "score": 72.5,
                    "deviations": {
                        "wrist": {"angle_diff": 15.2, "status": "⚠️ 손목 회전 과다"},
                        "elbow": {"angle_diff": 5.0, "status": "✅ 적정 범위"},
                        "distance": {"diff_cm": 3.5, "status": "⚠️ 도장 거리 초과"},
                    },
                },
            },
            {
                "name": "location_tracking",
                "args": {"zone_id": "B-3", "session_id": "coaching-session-1"},
                "result": {
                    "current_zone": "B-3",
                    "zone_type": "수직 패널",
                    "coverage": 68.5,
                    "remaining_area": 3.2,
                    "estimated_time": "12분",
                },
            },
            {
                "name": "timeseries",
                "args": {"metric": "speed", "session_id": "coaching-session-1", "window": 30},
                "result": {
                    "mean_speed": 0.48,
                    "max_speed": 0.62,
                    "min_speed": 0.31,
                    "trend": "상승",
                    "alert": "최근 30초간 이동속도 상승 추세 — 저도막 위험",
                },
            },
        ],
        "response": (
            "## 실시간 종합 코칭 피드백\n\n"
            "**종합 점수: 72.5 / 100**\n\n"
            "### 🏋️ 자세 피드백\n"
            "| 항목 | 현재 | 기준 | 상태 |\n|---|---|---|---|\n"
            "| 손목 각도 | 15.2° | 0° (중립) | ⚠️ 과도 회전 |\n"
            "| 팔꿈치 각도 | 95° | 100° | ✅ 적정 |\n"
            "| 도장 거리 | 28.5cm | 25cm | ⚠️ 거리 초과 |\n\n"
            "**교정 필요:** 손목을 중립 위치로 교정하고, 도장면과의 거리를 "
            "약 3.5cm 줄여주세요.\n\n"
            "### 📍 작업 진행 현황\n"
            "- 현재 구역: B-3 (수직 패널)\n"
            "- 도장 진척률: 68.5%\n"
            "- 잔여 면적: 3.2 m²\n"
            "- 예상 소요 시간: 약 12분\n\n"
            "### ⚡ 속도 분석\n"
            "- 평균 속도: 0.48 m/s (적정 범위 상한)\n"
            "- 최근 추세: **상승** ⚠️\n"
            "- **경고:** 최근 30초간 이동속도가 상승하고 있습니다. "
            "현재 속도를 0.4 m/s 수준으로 낮춰주세요. 과속 시 저도막 발생 위험이 있습니다."
        ),
    },
    "environment": {
        "rag_chunks": [
            {
                "chunk_id": "ENV-SPEC-001",
                "text": "도장 작업 환경 기준: 상대 습도 85% 이하, 이슬점 차이 3°C 이상, 표면 온도 5~40°C. 기준 미달 시 도장 작업 중단 및 환경 개선 후 재개.",
                "rerank_score": 0.97,
            },
            {
                "chunk_id": "ENV-SPEC-003",
                "text": "이슬점 차이(표면온도 - 이슬점온도)가 3°C 미만일 경우 결로 위험으로 도장 금지. 습도 85% 초과 시에도 결로 위험으로 작업 중단 권고.",
                "rerank_score": 0.93,
            },
        ],
        "tool_calls": [
            {
                "name": "dft_judge",
                "args": {
                    "env_check": True,
                    "humidity": 85,
                    "dewpoint_delta": 2.0,
                    "surface_temp": 22,
                },
                "result": {
                    "work_permitted": False,
                    "violations": [
                        "습도 85% — 기준치 85% 초과 경계",
                        "이슬점 차이 2°C — 기준 3°C 미달 (결로 위험)",
                    ],
                    "recommendation": "작업 중단 권고. 제습 또는 환기 후 이슬점 차이 3°C 이상 확보 시 재개.",
                },
            },
        ],
        "response": (
            "## 작업 환경 적합성 판단\n\n"
            "**판정: 🚫 작업 중단 권고**\n\n"
            "### 환경 조건 점검\n"
            "| 항목 | 현재값 | 기준 | 판정 |\n|---|---|---|---|\n"
            "| 상대 습도 | 85% | ≤ 85% | ⚠️ 경계 |\n"
            "| 이슬점 차이 | 2°C | ≥ 3°C | ❌ 미달 |\n"
            "| 표면 온도 | 22°C | 5~40°C | ✅ 적합 |\n\n"
            "### 위반 사항\n"
            "1. **이슬점 차이 부족** (2°C < 3°C) — 결로 발생 위험이 높아 "
            "도막 부착력 저하 및 들뜸(blistering) 가능성\n"
            "2. **습도 경계값** (85%) — 수성 도료 사용 시 건조 지연 및 "
            "품질 저하 위험\n\n"
            "### 조치 사항\n"
            "- **즉시 도장 작업 중단**\n"
            "- 제습기 가동 또는 환기 실시\n"
            "- 이슬점 차이 3°C 이상 확보 확인 후 작업 재개\n"
            "- 재개 전 표면 결로 여부 육안 및 기기 점검 필수"
        ),
    },
    "general": {
        "rag_chunks": [
            {
                "chunk_id": "GENERAL-001",
                "text": "선박 도장은 부식 방지 및 해양 생물 부착 방지를 위한 핵심 공정으로, 표면처리, 프라이머, 중도, 상도의 다층 도장 체계로 구성됩니다.",
                "rerank_score": 0.78,
            },
        ],
        "tool_calls": [],
        "response": (
            "질문을 분석했습니다. 보다 정확한 답변을 위해 구체적인 정보를 "
            "알려주시면 도움이 됩니다.\n\n"
            "예를 들어:\n"
            "- DFT 측정값과 기준 nominal 값\n"
            "- 현재 환경 조건 (온도, 습도, 이슬점)\n"
            "- 작업 구역 및 공정 단계\n\n"
            "위 정보를 포함하여 다시 질문해주세요."
        ),
    },
}

_TOOL_EMOJI: dict[str, str] = {
    "dft_judge": "📏",
    "anomaly_detection": "🔍",
    "sequence_validation": "📋",
    "pose_estimation": "🏋️",
    "location_tracking": "📍",
    "timeseries": "📈",
}

_TOOL_DISPLAY: dict[str, str] = {
    "dft_judge": "DFT 합부 판정",
    "anomaly_detection": "이상 탐지 분석",
    "sequence_validation": "도장 순서 검증",
    "pose_estimation": "자세 추정 분석",
    "location_tracking": "위치 추적",
    "timeseries": "시계열 분석",
}

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

_session_id: str = str(uuid.uuid4())[:8]


def _new_session() -> str:
    global _session_id
    _session_id = str(uuid.uuid4())[:8]
    return _session_id


# ---------------------------------------------------------------------------
# Demo mode streaming generator
# ---------------------------------------------------------------------------


def _demo_stream(
    question: str,
    history: ChatHistory,
) -> Generator[ChatHistory, None, None]:
    """Simulate the agentic pipeline with pre-computed data and delays."""
    scenario_type = _classify_question(question)
    scenario = DEMO_SCENARIOS.get(scenario_type, DEMO_SCENARIOS["general"])

    step_id = 0

    # --- Step 1: RAG Retrieval ---
    step_id += 1
    rag_chunks = scenario["rag_chunks"]
    history.append(
        ChatMessage(
            role="assistant",
            content="",
            metadata={"title": "🔍 RAG 검색 중...", "status": "pending", "id": step_id},
        )
    )
    yield history
    time.sleep(0.5)

    rag_text_parts = []
    for chunk in rag_chunks:
        score = chunk.get("rerank_score", 0)
        rag_text_parts.append(
            f"**[{chunk['chunk_id']}]** (유사도: {score:.2f})\n{chunk['text']}"
        )
    rag_content = "\n\n".join(rag_text_parts)

    history[-1] = ChatMessage(
        role="assistant",
        content=f"검색된 문서 {len(rag_chunks)}건:\n\n{rag_content}",
        metadata={
            "title": f"🔍 RAG 검색 완료 — {len(rag_chunks)}건 검색됨",
            "status": "done",
            "id": step_id,
        },
    )
    yield history

    # --- Step 2: MCP Tool Calls ---
    tool_calls = scenario["tool_calls"]
    for tc in tool_calls:
        step_id += 1
        tool_name = tc["name"]
        emoji = _TOOL_EMOJI.get(tool_name, "🛠️")
        display_name = _TOOL_DISPLAY.get(tool_name, tool_name)

        history.append(
            ChatMessage(
                role="assistant",
                content="",
                metadata={
                    "title": f"{emoji} MCP: {display_name} 호출 중...",
                    "status": "pending",
                    "id": step_id,
                },
            )
        )
        yield history
        time.sleep(0.3)

        args_str = json.dumps(tc["args"], ensure_ascii=False, indent=2)
        result_str = json.dumps(tc["result"], ensure_ascii=False, indent=2)
        tool_content = (
            f"**입력 파라미터:**\n```json\n{args_str}\n```\n\n"
            f"**실행 결과:**\n```json\n{result_str}\n```"
        )

        history[-1] = ChatMessage(
            role="assistant",
            content=tool_content,
            metadata={
                "title": f"{emoji} MCP: {display_name} 완료 ✅",
                "status": "done",
                "id": step_id,
            },
        )
        yield history

    # --- Step 3: QLoRA Inference ---
    step_id += 1
    history.append(
        ChatMessage(
            role="assistant",
            content="",
            metadata={
                "title": "🧠 QLoRA 추론 중...",
                "status": "pending",
                "id": step_id,
            },
        )
    )
    yield history
    time.sleep(1.0)

    history[-1] = ChatMessage(
        role="assistant",
        content=f"모델: {MODEL_NAME}\n컨텍스트: RAG {len(rag_chunks)}건 + 도구 결과 {len(tool_calls)}건 통합 추론 완료",
        metadata={
            "title": "🧠 QLoRA 추론 완료 ✅",
            "status": "done",
            "id": step_id,
        },
    )
    yield history

    # --- Step 4: Streaming final answer ---
    response = scenario["response"]
    history.append(ChatMessage(role="assistant", content=""))
    yield history

    accumulated = ""
    chunk_size = 3
    for i in range(0, len(response), chunk_size):
        accumulated += response[i : i + chunk_size]
        history[-1] = ChatMessage(role="assistant", content=accumulated)
        yield history
        time.sleep(0.01)

    history[-1] = ChatMessage(role="assistant", content=response)
    yield history


# ---------------------------------------------------------------------------
# Live mode streaming generator
# ---------------------------------------------------------------------------


def _live_stream(
    question: str,
    history: ChatHistory,
) -> Generator[ChatHistory, None, None]:
    """Run the real AgentPipeline and stream tool call progress."""
    import importlib
    import sys
    from pathlib import Path

    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    agent_mod = importlib.import_module("agent_pipeline")
    AgentPipeline = agent_mod.AgentPipeline

    step_id = 1
    history.append(
        ChatMessage(
            role="assistant",
            content="파이프라인 초기화 중...",
            metadata={"title": "⚙️ AgentPipeline 로드", "status": "pending", "id": step_id},
        )
    )
    yield history

    try:
        pipeline = AgentPipeline(
            config_path=CONFIG_PATH,
            adapter_path=ADAPTER_PATH,
            mcp_base_url=MCP_BASE_URL,
        )
    except Exception as e:
        history[-1] = ChatMessage(
            role="assistant",
            content=f"파이프라인 초기화 실패: {e}",
            metadata={"title": "⚙️ AgentPipeline 오류 ❌", "status": "done", "id": step_id},
        )
        yield history
        return

    history[-1] = ChatMessage(
        role="assistant",
        content="모델 및 RAG 인덱스 로드 완료",
        metadata={"title": "⚙️ AgentPipeline 준비 완료 ✅", "status": "done", "id": step_id},
    )
    yield history

    step_id += 1
    history.append(
        ChatMessage(
            role="assistant",
            content="",
            metadata={"title": "🔄 에이전트 파이프라인 실행 중...", "status": "pending", "id": step_id},
        )
    )
    yield history

    try:
        result = pipeline.run(
            instruction=question,
            session_id=_session_id,
            enable_thinking=True,
        )
    except Exception as e:
        history[-1] = ChatMessage(
            role="assistant",
            content=f"실행 오류: {e}",
            metadata={"title": "🔄 파이프라인 오류 ❌", "status": "done", "id": step_id},
        )
        yield history
        return

    meta = result.get("metadata", {})
    route = meta.get("route", result.get("mode", "N/A"))
    rounds = meta.get("rounds", 0)
    latency = meta.get("latency_ms", 0)
    history[-1] = ChatMessage(
        role="assistant",
        content=f"모드: {route} | 라운드: {rounds} | 지연: {latency}ms",
        metadata={"title": "🔄 파이프라인 실행 완료 ✅", "status": "done", "id": step_id},
    )
    yield history

    rag_chunks = result.get("rag_chunks", [])
    if rag_chunks:
        step_id += 1
        rag_parts = []
        for chunk in rag_chunks:
            score = chunk.get("rerank_score")
            score_text = f" (유사도: {score:.2f})" if score else ""
            rag_parts.append(f"**[{chunk.get('chunk_id', '')}]**{score_text}\n{chunk.get('text', '')}")

        history.append(
            ChatMessage(
                role="assistant",
                content=f"검색된 문서 {len(rag_chunks)}건:\n\n" + "\n\n".join(rag_parts),
                metadata={
                    "title": f"🔍 RAG 검색 결과 — {len(rag_chunks)}건",
                    "status": "done",
                    "id": step_id,
                },
            )
        )
        yield history

    tool_calls = result.get("tool_calls", [])
    for tc in tool_calls:
        step_id += 1
        name = tc.get("name", "unknown")
        emoji = _TOOL_EMOJI.get(name, "🛠️")
        display_name = _TOOL_DISPLAY.get(name, name)
        args_str = json.dumps(tc.get("arguments", {}), ensure_ascii=False, indent=2)
        result_str = json.dumps(tc.get("result", {}), ensure_ascii=False, indent=2)

        history.append(
            ChatMessage(
                role="assistant",
                content=(
                    f"**입력 파라미터:**\n```json\n{args_str}\n```\n\n"
                    f"**실행 결과:**\n```json\n{result_str}\n```"
                ),
                metadata={
                    "title": f"{emoji} MCP: {display_name} ✅",
                    "status": "done",
                    "id": step_id,
                },
            )
        )
        yield history

    for section_name, section_label in [
        ("dft_result", "📊 DFT 판정 결과"),
        ("anomaly", "⚠️ 이상 탐지 결과"),
        ("pose_comparison", "🤸 자세 비교 결과"),
        ("next_path", "🗺️ 다음 도장 경로"),
    ]:
        section = result.get(section_name)
        if section is not None:
            step_id += 1
            history.append(
                ChatMessage(
                    role="assistant",
                    content=f"```json\n{json.dumps(section, ensure_ascii=False, indent=2)}\n```",
                    metadata={"title": f"{section_label} ✅", "status": "done", "id": step_id},
                )
            )
            yield history

    text_guide = result.get("text_guide", {})
    response_text = text_guide.get("body", "") if text_guide else result.get("response", "응답을 생성하지 못했습니다.")
    history.append(ChatMessage(role="assistant", content=""))
    yield history

    accumulated = ""
    chunk_size = 3
    for i in range(0, len(response_text), chunk_size):
        accumulated += response_text[i : i + chunk_size]
        history[-1] = ChatMessage(role="assistant", content=accumulated)
        yield history
        time.sleep(0.01)

    history[-1] = ChatMessage(role="assistant", content=response_text)
    yield history

    citations = result.get("citations", [])
    if citations:
        step_id += 1
        history.append(
            ChatMessage(
                role="assistant",
                content=f"**참조 문서:** {', '.join(citations)}",
                metadata={"title": "📚 인용 문서", "status": "done", "id": step_id},
            )
        )
        yield history


# ---------------------------------------------------------------------------
# Chat handler
# ---------------------------------------------------------------------------


def respond(
    message: str,
    history: ChatHistory,
) -> Generator[ChatHistory, None, None]:
    """Main chat handler that dispatches to demo or live mode."""
    if not message.strip():
        yield history
        return

    history.append(ChatMessage(role="user", content=message))
    yield history

    if DEMO_MODE:
        yield from _demo_stream(message, history)
    else:
        yield from _live_stream(message, history)


def handle_example_click(
    example_text: str,
    history: ChatHistory,
) -> Generator[ChatHistory, None, None]:
    """Handle clicking on an example question button."""
    yield from respond(example_text, history)


# ---------------------------------------------------------------------------
# Build the Gradio UI
# ---------------------------------------------------------------------------


def build_sidebar() -> dict[str, Any]:
    """Build sidebar information panel. Returns component references."""
    mode_badge = (
        '<span class="status-badge status-demo">DEMO</span>'
        if DEMO_MODE
        else '<span class="status-badge status-live">LIVE</span>'
    )

    gr.HTML(
        f"""
        <div style="text-align:center; margin-bottom:12px;">
            <span style="font-size:1.3em; font-weight:700; color:#e0e0e0;">시스템 상태</span>
            <br/>{mode_badge}
        </div>
        """
    )

    with gr.Group():
        gr.Markdown("#### 파이프라인 구성요소")
        if DEMO_MODE:
            gr.Markdown(
                "- RAG 검색 엔진  ✅ 시뮬레이션\n"
                "- QLoRA 모델  ✅ 시뮬레이션\n"
                "- MCP 도구 서버  ✅ 시뮬레이션"
            )
        else:
            gr.Markdown(
                "- RAG 검색 엔진  🔄 연결 중\n"
                "- QLoRA 모델  🔄 연결 중\n"
                "- MCP 도구 서버  🔄 연결 중"
            )

    with gr.Group():
        gr.Markdown("#### 세션 정보")
        gr.Markdown(f"- **세션 ID:** `{_session_id}`")
        gr.Markdown(f"- **모델:** {MODEL_NAME}")
        gr.Markdown(f"- **MCP 서버:** `{MCP_BASE_URL}`")

    with gr.Group():
        gr.Markdown("#### 도구 목록 (MCP)")
        tool_list = "\n".join(
            f"- {_TOOL_EMOJI.get(k, '🛠️')} {v}"
            for k, v in _TOOL_DISPLAY.items()
        )
        gr.Markdown(tool_list)

    with gr.Group():
        gr.Markdown("#### 운용 모드")
        mode_selector = gr.Radio(
            choices=["자동 감지", "훈련 중 (실시간)", "훈련 후 (리뷰)"],
            value="자동 감지",
            label="모드 선택",
            info="훈련 중: 순서/이상탐지 (<60초) | 훈련 후: 자세/분석 (제한없음)",
        )

    with gr.Group():
        gr.Markdown("#### 사용 안내")
        gr.Markdown(
            "아래 예시 질문을 클릭하거나\n"
            "직접 입력하여 AI 코파일럿과\n"
            "대화를 시작하세요.\n\n"
            "각 응답은 도구 호출 과정을\n"
            "접이식 패널로 표시합니다."
        )

    return {"mode_selector": mode_selector}


def create_app() -> gr.Blocks:
    """Create and return the Gradio Blocks application."""
    with gr.Blocks(
        title="선박 도장 AI Co-Pilot",
    ) as app:
        gr.HTML(
            """
            <div class="title-section">
                <h1>🚢 선박 도장 AI Co-Pilot</h1>
                <p>도장 품질 분석 · 실시간 코칭 · 도구 연동</p>
            </div>
            """
        )

        with gr.Row():
            with gr.Column(scale=1, min_width=250):
                build_sidebar()

                new_session_btn = gr.Button(
                    "🔄 새 세션 시작",
                    variant="secondary",
                    size="sm",
                )

            # --- Main Chat Area ---
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(
                    value=[],
                    height=580,
                    placeholder=(
                        "🚢 선박 도장 AI Co-Pilot에 질문하세요.\n\n"
                        "DFT 판정, 이상 탐지, 공정 순서 검증,\n"
                        "실시간 코칭 등을 지원합니다."
                    ),
                    layout="panel",
                    show_label=False,
                    elem_classes=["chatbot-container"],
                    buttons=["copy"],
                )

                # --- Input Area ---
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="도장 관련 질문을 입력하세요...",
                        show_label=False,
                        scale=6,
                        container=False,
                        autofocus=True,
                    )
                    send_btn = gr.Button(
                        "전송",
                        variant="primary",
                        scale=1,
                        min_width=80,
                    )
                    clear_btn = gr.Button(
                        "초기화",
                        variant="secondary",
                        scale=1,
                        min_width=80,
                    )

                # --- Example Questions ---
                gr.Markdown(
                    "#### 💡 예시 질문",
                    elem_classes=["example-section-title"],
                )

                with gr.Row():
                    example_btns = []
                    for ex in EXAMPLE_QUESTIONS[:3]:
                        btn = gr.Button(
                            f"{ex['icon']}  {ex['display_text']}",
                            variant="secondary",
                            size="sm",
                            elem_classes=["example-btn"],
                        )
                        example_btns.append((btn, ex["text"]))

                with gr.Row():
                    for ex in EXAMPLE_QUESTIONS[3:]:
                        btn = gr.Button(
                            f"{ex['icon']}  {ex['display_text']}",
                            variant="secondary",
                            size="sm",
                            elem_classes=["example-btn"],
                        )
                        example_btns.append((btn, ex["text"]))

        # --- Event handlers ---
        msg_input.submit(
            fn=respond,
            inputs=[msg_input, chatbot],
            outputs=[chatbot],
        ).then(fn=lambda: "", outputs=[msg_input])

        send_btn.click(
            fn=respond,
            inputs=[msg_input, chatbot],
            outputs=[chatbot],
        ).then(fn=lambda: "", outputs=[msg_input])

        clear_btn.click(fn=lambda: [], outputs=[chatbot])

        def _reset_session() -> tuple[list[Any], str]:
            sid = _new_session()
            return [], f"새 세션이 시작되었습니다. (ID: {sid})"

        new_session_btn.click(
            fn=_reset_session,
            outputs=[chatbot, msg_input],
        )

        # Wire up example buttons
        for btn, text in example_btns:
            btn.click(
                fn=handle_example_click,
                inputs=[gr.State(text), chatbot],
                outputs=[chatbot],
            )

    return app


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    mode_label = "DEMO" if DEMO_MODE else "LIVE"
    logger.info("Starting Ship Painting AI Co-Pilot in %s mode on port %d", mode_label, APP_PORT)

    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=APP_PORT,
        share=False,
        show_error=True,
        theme=CUSTOM_THEME,
        css=CUSTOM_CSS,
    )
