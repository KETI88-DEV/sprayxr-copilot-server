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

DEMO_MODE: bool = os.environ.get("DEMO_MODE", "true").lower() in ("true", "1", "yes")
APP_PORT: int = int(os.environ.get("APP_PORT", "7860"))
MCP_BASE_URL: str = os.environ.get("MCP_BASE_URL", "http://localhost:8100")
CONFIG_PATH: str = os.environ.get(
    "CONFIG_PATH", "/workspace/trainer/configs/rag_config_fewshot.yaml"
)
ADAPTER_PATH: str = os.environ.get(
    "ADAPTER_PATH", "/output/adapters/qwen3_4b_shipcoat_lora/"
)
MODEL_NAME: str = "Qwen3-4B QLoRA (Iter17 — Dual Gate PASS)"

# ---------------------------------------------------------------------------
# Theme — high-contrast industrial slate
# ---------------------------------------------------------------------------

from gradio.themes import Base as _ThemeBase, GoogleFont as _GFont
_zinc = gr.themes.colors.zinc
_orange = gr.themes.colors.orange

CUSTOM_THEME = _ThemeBase(
    primary_hue=_zinc,
    secondary_hue=_orange,
    neutral_hue=_zinc,
    font=_GFont("IBM Plex Sans"),
    font_mono=_GFont("IBM Plex Mono"),
).set(
    body_background_fill="#FAFAFA",
    body_background_fill_dark="#FAFAFA",
    body_text_color="#18181B",
    body_text_color_dark="#18181B",
    block_background_fill="#FFFFFF",
    block_background_fill_dark="#FFFFFF",
    block_border_color="#E4E4E7",
    block_border_color_dark="#E4E4E7",
    block_label_text_color="#71717A",
    block_label_text_color_dark="#71717A",
    block_title_text_color="#18181B",
    block_title_text_color_dark="#18181B",
    input_background_fill="#FFFFFF",
    input_background_fill_dark="#FFFFFF",
    input_border_color="#D4D4D8",
    input_border_color_dark="#D4D4D8",
    input_placeholder_color="#A1A1AA",
    input_placeholder_color_dark="#A1A1AA",
    button_primary_background_fill="#ea580c",
    button_primary_background_fill_dark="#ea580c",
    button_primary_background_fill_hover="#f97316",
    button_primary_background_fill_hover_dark="#f97316",
    button_primary_text_color="#ffffff",
    button_primary_text_color_dark="#ffffff",
    button_secondary_background_fill="#F4F4F5",
    button_secondary_background_fill_dark="#F4F4F5",
    button_secondary_text_color="#3F3F46",
    button_secondary_text_color_dark="#3F3F46",
    border_color_primary="#E4E4E7",
    border_color_primary_dark="#E4E4E7",
    color_accent_soft="#F4F4F5",
    color_accent_soft_dark="#F4F4F5",
)

CUSTOM_CSS = """
.gradio-container { max-width: 1400px !important; }
footer { display: none !important; }

.sidebar-panel {
    background: #FFFFFF !important;
    border: 1px solid #E4E4E7 !important;
    border-radius: 8px !important;
    padding: 14px !important;
}
.sidebar-panel .label-wrap { color: #71717A !important; }

.status-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 4px;
    font-size: 0.8em;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.status-demo { background: #5E6AD2; color: #fff; }
.status-live { background: #2EB88A; color: #fff; }

.example-btn {
    text-align: left !important;
    font-size: 0.88em !important;
    border: 1px solid #E4E4E7 !important;
    background: #FFFFFF !important;
    color: #3F3F46 !important;
    border-radius: 6px !important;
    padding: 10px 14px !important;
    transition: border-color 0.15s !important;
}
.example-btn:hover {
    border-color: #5E6AD2 !important;
    color: #18181B !important;
    background: #F4F4F5 !important;
}

.chatbot-container .message { border-radius: 8px !important; }

.chatbot-container .bot, .chatbot-container .panel .bot {
    background-color: #FFFFFF !important;
    color: #18181B !important;
    border: 1px solid #E4E4E7 !important;
}
.chatbot-container .user, .chatbot-container .panel .user {
    background-color: #EEF2FF !important;
    color: #18181B !important;
    border: 1px solid #C7D2FE !important;
}
.chatbot-container .bot *, .chatbot-container .user * {
    color: inherit !important;
}
.chatbot-container .bot h1, .chatbot-container .bot h2, .chatbot-container .bot h3,
.chatbot-container .user h1, .chatbot-container .user h2, .chatbot-container .user h3 {
    color: #18181B !important;
}
.chatbot-container .bot strong, .chatbot-container .user strong {
    color: #18181B !important;
}
.chatbot-container .bot code, .chatbot-container .user code {
    background-color: #F4F4F5 !important;
    color: #5E6AD2 !important;
}
.chatbot-container .bot pre, .chatbot-container .user pre {
    background-color: #FAFAFA !important;
    border: 1px solid #E4E4E7 !important;
}
.chatbot-container .bot a, .chatbot-container .user a {
    color: #5E6AD2 !important;
}
.chatbot-container .bot table th {
    background-color: #F4F4F5 !important;
    color: #18181B !important;
}
.chatbot-container .bot table td {
    border-color: #E4E4E7 !important;
    color: #3F3F46 !important;
}
.chatbot-container .bot-row, .chatbot-container .panel.bot-row {
    background: transparent !important;
}
.chatbot-container .user-row, .chatbot-container .panel.user-row {
    background: transparent !important;
}
.chatbot-container .message-bubble-border {
    border-color: #E4E4E7 !important;
}
.chatbot-container .pending {
    background-color: #F4F4F5 !important;
    color: #71717A !important;
}
.chatbot-container [data-testid="chatbot"] {
    background-color: #FAFAFA !important;
}
.thought-group {
    background: #F4F4F5 !important;
    border: 1px solid #E4E4E7 !important;
}
.thought-group .title {
    color: #5E6AD2 !important;
}
.thought-group .title .md {
    color: #5E6AD2 !important;
}
.thought-group .thought {
    color: #3F3F46 !important;
}
.thought-group .thought * {
    color: #3F3F46 !important;
}
.thought-group .thought strong {
    color: #18181B !important;
}
.thought-group .thought code {
    background-color: #FFFFFF !important;
    color: #5E6AD2 !important;
}
.thought-group .thought pre {
    background-color: #FFFFFF !important;
    border: 1px solid #E4E4E7 !important;
}

.title-section h1 { color: #18181B !important; }
.title-section p { color: #71717A !important; }
"""

# ---------------------------------------------------------------------------
# Demo Scenarios
# ---------------------------------------------------------------------------

EXAMPLE_QUESTIONS: list[dict[str, str]] = [
    {
        "text": "DFT 측정값 [95, 102, 88, 105, 99]로 nominal 100um 기준 합부 판정해줘",
        "display_text": "DFT 합부 판정 요청",
    },
    {
        "text": "B-3 구역 하단에 저도막 패턴이 보여. DFT [85, 92, 78]이고 이동속도가 점점 빨라졌어. 원인 분석해줘",
        "display_text": "도막 분포 이상 탐지",
    },
    {
        "text": "현재 A-1 구역을 칠하고 있는데 다음에 어디를 칠해야 해?",
        "display_text": "도장 순서 안내",
    },
    {
        "text": "실시간 코칭 모드로 현재 작업자에게 자세/순서/속도 종합 피드백을 줘",
        "display_text": "실시간 종합 코칭",
    },
    {
        "text": "습도가 85%이고 이슬점 차이가 2도C일 때 도장 작업을 진행해도 되는가?",
        "display_text": "작업 환경 적합성 판단",
    },
]

ChatHistory = list[ChatMessage]

_TOOL_DISPLAY: dict[str, str] = {
    "dft_judge": "DFT 합부 판정",
    "anomaly_detection": "도막 분포 이상 탐지",
    "sequence_validation": "도장 순서 안내",
    "pose_estimation": "숙련자 자세 비교",
    "location_tracking": "위치 추적",
    "timeseries": "시계열 분석",
}


def _classify_question(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ("dft", "합부", "판정", "측정값", "nominal")):
        return "dft_judge"
    if any(k in t for k in ("이상", "히트맵", "패턴", "원인", "저도막", "과도막", "분포")):
        return "anomaly"
    if any(k in t for k in ("다음", "순서", "어디", "블록")):
        return "sequence"
    if any(k in t for k in ("코칭", "자세", "포즈", "피드백", "속도")):
        return "coaching"
    if any(k in t for k in ("습도", "이슬점", "온도", "환경", "작업 가능")):
        return "environment"
    if any(k in t for k in ("블라스팅", "표면처리", "표면 처리")):
        return "qa_blasting"
    if any(k in t for k in ("프라이머", "방청", "primer")):
        return "qa_primer"
    if any(k in t for k in ("결로", "dew point", "이슬점이 뭐")):
        return "qa_dewpoint"
    if any(k in t for k in ("도막", "두께", "film thickness", "dft가 뭐", "dft란")):
        return "qa_dft"
    if any(k in t for k in ("도장 종류", "도장 공정", "도장이란", "도장이 뭐")):
        return "qa_painting"
    return "general"


DEMO_SCENARIOS: dict[str, dict[str, Any]] = {
    "dft_judge": {
        "rag_chunks": [
            {"chunk_id": "SPEC-DFT-001", "text": "SSPC-PA2 기준 DFT 합부 판정: nominal +/-20% 이내, 개별 측정값 중 90% 이상이 규격 이내이며 최소값은 nominal의 80% 이상이어야 합격.", "rerank_score": 0.94},
            {"chunk_id": "QC-PROC-012", "text": "건조 도막 두께(DFT) 측정 시 게이지 교정 확인 필수. 각 구역 당 최소 5회 측정, 평균 및 개별 편차 기록.", "rerank_score": 0.87},
        ],
        "tool_calls": [
            {"name": "dft_judge", "args": {"measurements": [95, 102, 88, 105, 99], "nominal": 100, "tolerance_pct": 20}, "result": {"verdict": "PASS", "mean": 97.8, "std": 6.14, "min": 88, "max": 105, "out_of_spec_count": 0, "details": "전 측정값이 80~120um 범위 이내. 평균 97.8um (nominal 대비 -2.2%). 합격."}},
        ],
        "xr_commands": [
            {"action": "mark_zone", "zone_id": "B-3", "status": "pass", "color": "#16a34a", "label": "DFT PASS (97.8um)"},
        ],
        "response": (
            "## DFT 합부 판정 결과\n\n"
            "**판정: PASS (합격)**\n\n"
            "| 항목 | 값 |\n|---|---|\n"
            "| 측정값 | [95, 102, 88, 105, 99] um |\n"
            "| 평균 DFT | 97.8 um |\n"
            "| 표준편차 | 6.14 um |\n"
            "| 최소/최대 | 88 / 105 um |\n"
            "| 규격 범위 | 80 ~ 120 um (nominal 100 +/- 20%) |\n"
            "| 규격 이탈 | 0건 |\n\n"
            "SSPC-PA2 기준에 따라 모든 측정값이 허용 범위(80~120um) 이내이며, "
            "평균값 97.8um은 nominal 100um 대비 -2.2%로 양호합니다.\n\n"
            "**권고사항:** 현재 도막 상태 양호하나, 88um 측정 지점은 하한에 가까우므로 "
            "해당 구역 재측정 후 필요 시 보수 도장을 고려하세요."
        ),
    },
    "anomaly": {
        "rag_chunks": [
            {"chunk_id": "DEFECT-ANALYSIS-003", "text": "저도막 패턴 원인 분류: (1) 스프레이 건 이동속도 과다, (2) 도료 점도 부족, (3) 노즐 마모에 의한 패턴 편향, (4) 도장 거리 부적합.", "rerank_score": 0.91},
            {"chunk_id": "ENV-GUIDE-007", "text": "고온 시 도료 건조가 빨라져 도막 두께 불균일 위험 증가. 수성 도료는 습도 60% 초과 시 주의.", "rerank_score": 0.85},
        ],
        "tool_calls": [
            {"name": "anomaly_detection", "args": {"dft_measurements": [{"value_um": 85, "zone_id": "B-3", "position": "middle"}, {"value_um": 92, "zone_id": "B-3", "position": "upper"}, {"value_um": 78, "zone_id": "B-3", "position": "lower"}, {"value_um": 72, "zone_id": "B-3", "position": "lower"}], "timeseries": {"gun_speed_m_per_s": [0.4, 0.45, 0.58, 0.65], "standoff_distance_mm": [300, 310, 340, 370], "spray_angle_deg": [72, 70, 60, 52]}, "zone_id": "B-3", "spec": {"nominal_um": 100, "tolerance_rule": "90/10"}}, "result": {"anomaly_detected": True, "anomaly_count": 2, "distribution_pattern": "하부 영역에 저도막 집중 (2건)", "root_causes": [{"cause": "후반부 이동속도 증가 (0.43 -> 0.62 m/s)", "confidence": 0.85}, {"cause": "분사각도 부족 구간 2건 (52도~)", "confidence": 0.70}], "stats": {"mean_um": 81.8, "nominal_um": 100}}},
        ],
        "xr_commands": {"mode": "training_review", "anomaly": {"detected": True, "type": "저도막", "zone": "B-3", "positions": ["lower"], "severity": "warning", "distribution": "하부 영역 집중"}, "text_guide": {"summary": "하부 저도막 집중 - 보수 도장 필요", "body": "이동속도 감속 및 분사각도 교정 후 하부 보수 도장"}, "steps": [{"step": 1, "action": "B-3 하부 구역 보수 도장 (목표 100um)"}, {"step": 2, "action": "이동속도 0.4 m/s 이하 유지"}, {"step": 3, "action": "분사각도 65도 이상 유지"}]},
        "response": (
            "## 도막 분포 이상 탐지 결과\n\n"
            "**상태: WARNING — 하부 저도막 집중**\n\n"
            "### 분포 분석\n"
            "| 항목 | 값 |\n|---|---|\n"
            "| 측정값 | [85, 92, 78, 72] um |\n"
            "| 평균 DFT | 81.8 um (nominal 100um 대비 -18.2%) |\n"
            "| 이상 건수 | 2건 (하부 집중) |\n"
            "| 분포 패턴 | 하부 영역에 저도막 집중 |\n\n"
            "### 근본 원인 분석\n"
            "시계열 데이터 상관관계 분석 결과:\n\n"
            "1. **이동속도 증가** (신뢰도 85%) — 후반부(하부) 작업 시 이동속도가 "
            "0.43 m/s에서 0.62 m/s로 증가하여 도막이 충분히 형성되지 않음\n"
            "2. **분사각도 부족** (신뢰도 70%) — 하부 구역에서 분사각도가 52도까지 감소, "
            "적정 범위(65~75도) 미달\n\n"
            "### 조치 권고\n"
            "- 하부 구역 보수 도장 실시 (목표 DFT: 100um 이상)\n"
            "- 하부 작업 시 이동속도 0.4 m/s 이하 유지\n"
            "- 분사각도 65도 이상 유지 (팔 높이 조정)"
        ),
    },
    "sequence": {
        "rag_chunks": [
            {"chunk_id": "PROC-SEQ-001", "text": "도장 순서 결정 시 중력 방향 오염 방지 원칙: 상부 구역을 먼저 도장하여 하부 오염을 최소화한다. 평면 패널 우선, 곡면/모서리 후순위.", "rerank_score": 0.96},
        ],
        "tool_calls": [
             {"name": "sequence_validation", "args": {"zone_map": [{"zone_id": "A-1", "block_id": "BLK-A", "zone_type": "수직 패널", "position": "upper", "is_painted": True, "area_m2": 12.0}, {"zone_id": "A-2", "block_id": "BLK-A", "zone_type": "수직 패널", "position": "middle", "is_painted": False, "area_m2": 15.0}, {"zone_id": "A-3", "block_id": "BLK-A", "zone_type": "곡면", "position": "lower", "is_painted": False, "area_m2": 8.0}, {"zone_id": "B-1", "block_id": "BLK-B", "zone_type": "수직 패널", "position": "upper", "is_painted": False, "area_m2": 14.0}], "current_zone_id": "A-1", "structure_type": "block"}, "result": {"current_block_id": "BLK-A", "next_block": "BLK-A", "next_zone": "A-2", "zone_reason": "중간부 위치 (중력 방향 오염 방지), 평면 구역 우선", "current_block_remaining": ["A-2", "A-3"], "block_sequence": ["BLK-A", "BLK-B"], "progress": "2/4 구역 완료", "message": "현재 블록(BLK-A) 내 다음 구역: A-2 (중간부 위치). 블록 내 잔여 2개."}},
        ],
        "xr_commands": {"mode": "training_realtime", "next_path": {"current_zone": "A-1", "next_zone": "A-2", "block_id": "BLK-A", "highlight_color": "#f97316"}, "text_guide": {"summary": "다음 구역: A-2 (중간부)", "body": "BLK-A 내 상부->중간부 이동. 잔여 2개 구역."}, "steps": [{"step": 1, "action": "A-1 완료 확인"}, {"step": 2, "action": "A-2 (중간부 수직 패널) 이동"}, {"step": 3, "action": "A-3 (하부 곡면) 대기"}]},
        "response": (
            "## 도장 순서 안내\n\n"
            "**현재 위치: BLK-A / A-1 (상부, 도장 완료)**\n\n"
            "### 블록 내 순서 (BLK-A)\n"
            "| 순서 | 구역 | 위치 | 유형 | 상태 |\n|---|---|---|---|---|\n"
            "| 1 | A-1 | 상부 | 수직 패널 | 완료 |\n"
            "| **2** | **A-2** | **중간부** | **수직 패널** | **다음** |\n"
            "| 3 | A-3 | 하부 | 곡면 | 대기 |\n\n"
            "### 블록 간 순서\n"
            "| 순서 | 블록 | 잔여 구역 |\n|---|---|---|\n"
            "| **현재** | **BLK-A** | 2개 (A-2, A-3) |\n"
            "| 다음 | BLK-B | 1개 (B-1) |\n\n"
            "**다음 도장 구역: A-2** (중간부 수직 패널)\n"
            "- 근거: 상부→중간부→하부 순서 (중력 오염 방지)\n"
            "- 평면 패널 우선, 곡면(A-3) 후순위\n"
            "- 현재 블록 완료 후 BLK-B로 이동"
        ),
    },
    "coaching": {
        "rag_chunks": [
            {"chunk_id": "COACH-POSE-002", "text": "스프레이 도장 표준 자세: 팔꿈치 각도 90~110도, 손목 중립, 도장면과 스프레이 건 수직 유지.", "rerank_score": 0.89},
            {"chunk_id": "COACH-SPEED-005", "text": "적정 스프레이 이동속도: 0.3~0.5 m/s. 0.5 m/s 초과 시 저도막 위험.", "rerank_score": 0.86},
        ],
        "tool_calls": [
            {"name": "pose_estimation", "args": {"mode": "compare", "current_pose": {"joints": {"wrist_r": {"x": 0.12, "y": -0.08, "z": 0.41, "roll": 8.0, "pitch": 24.0, "yaw": 17.0}}}, "expert_pose_id": "EP-001", "block_id": "BLK-A"}, "result": {"mode": "compare", "expert_pose_id": "EP-001", "similarity_score": 0.72, "corrections": [{"joint": "wrist_r", "label": "오른쪽 손목", "angle_corrections": {"roll": 6.0, "pitch": 12.0, "yaw": 12.0}, "correction_text": "오른쪽 손목: roll 6도 과다, pitch 12도 과다, yaw 12도 과다"}], "summary": "숙련자 자세(EP-001) 대비 유사도 72%. 교정 필요 관절 1개."}},
            {"name": "location_tracking", "args": {"coordinates": [{"x": 910, "y": -45, "z": 250}], "ship_model_id": "hull-001"}, "result": {"current_zone": "B-3", "zone_type": "수직 패널", "coverage": 68.5, "remaining_area": 3.2, "estimated_time": "12분"}},
            {"name": "timeseries", "args": {"action": "query", "session_id": "sess-coach-001"}, "result": {"mean_speed": 0.48, "max_speed": 0.62, "min_speed": 0.31, "trend": "상승", "alert": "최근 30초간 이동속도 상승 추세 — 저도막 위험"}},
        ],
        "xr_commands": {"mode": "training_realtime", "pose_comparison": {"expert_pose_id": "EP-001", "similarity": 0.72, "corrections": [{"joint": "wrist_r", "roll": 6.0, "pitch": 12.0, "yaw": 12.0}]}, "anomaly": {"detected": True, "type": "speed_warning", "severity": "warning"}, "text_guide": {"summary": "손목 교정 + 속도 감속", "body": "EP-001 대비 유사도 72%. 이동속도 상승 추세 감지."}, "steps": [{"step": 1, "action": "손목 중립 교정 (roll/pitch/yaw 과다)"}, {"step": 2, "action": "이동속도 0.4 m/s로 감속"}, {"step": 3, "action": "B-3 잔여 3.2m2 도장 계속"}]},
        "response": (
            "## 실시간 종합 코칭 피드백\n\n"
            "**종합 점수: 72 / 100**\n\n"
            "### 자세 피드백 (숙련자 EP-001 대비)\n"
            "| 관절 | 항목 | 편차 | 판정 |\n|---|---|---|---|\n"
            "| 오른쪽 손목 | roll | +6도 | WARN — 과다 회전 |\n"
            "| 오른쪽 손목 | pitch | +12도 | WARN — 과다 |\n"
            "| 오른쪽 손목 | yaw | +12도 | WARN — 과다 |\n\n"
            "**교정 필요:** 손목을 중립 위치로 교정하세요. "
            "숙련자 자세 EP-001 기준 유사도 72%입니다.\n\n"
            "### 작업 진행 현황\n"
            "- 현재 구역: B-3 (수직 패널)\n"
            "- 도장 진척률: 68.5%\n"
            "- 잔여 면적: 3.2 m2\n"
            "- 예상 소요 시간: 약 12분\n\n"
            "### 속도 분석\n"
            "- 평균 속도: 0.48 m/s (적정 범위 상한)\n"
            "- 최근 추세: 상승 — WARNING\n"
            "- 최근 30초간 이동속도가 상승하고 있습니다. "
            "0.4 m/s 수준으로 낮춰주세요."
        ),
    },
    "environment": {
        "rag_chunks": [
            {"chunk_id": "ENV-SPEC-001", "text": "도장 작업 환경 기준: 상대 습도 85% 이하, 이슬점 차이 3도C 이상, 표면 온도 5~40도C.", "rerank_score": 0.97},
            {"chunk_id": "ENV-SPEC-003", "text": "이슬점 차이(표면온도 - 이슬점온도)가 3도C 미만일 경우 결로 위험으로 도장 금지.", "rerank_score": 0.93},
        ],
        "tool_calls": [
            {"name": "dft_judge", "args": {"env_check": True, "humidity": 85, "dewpoint_delta": 2.0, "surface_temp": 22}, "result": {"work_permitted": False, "violations": ["습도 85% — 기준치 경계", "이슬점 차이 2도C — 기준 3도C 미달 (결로 위험)"], "recommendation": "작업 중단 권고. 제습 또는 환기 후 이슬점 차이 3도C 이상 확보 시 재개."}},
        ],
        "xr_commands": {"mode": "training_realtime", "anomaly": {"detected": True, "type": "env_violation", "severity": "critical"}, "text_guide": {"summary": "작업 중단 — 결로 위험", "body": "이슬점 차이 2도C < 3도C 기준 미달. 즉시 작업 중단."}, "steps": [{"step": 1, "action": "도장 작업 즉시 중단"}, {"step": 2, "action": "제습기 가동 또는 환기"}, {"step": 3, "action": "이슬점 차이 3도C 이상 확인 후 재개"}]},
        "response": (
            "## 작업 환경 적합성 판단\n\n"
            "**판정: FAIL — 작업 중단 권고**\n\n"
            "### 환경 조건 점검\n"
            "| 항목 | 현재값 | 기준 | 판정 |\n|---|---|---|---|\n"
            "| 상대 습도 | 85% | 85% 이하 | WARN — 경계 |\n"
            "| 이슬점 차이 | 2도C | 3도C 이상 | FAIL — 미달 |\n"
            "| 표면 온도 | 22도C | 5~40도C | PASS |\n\n"
            "### 위반 사항\n"
            "1. **이슬점 차이 부족** (2도C < 3도C) — 결로 발생 위험이 높아 "
            "도막 부착력 저하 및 들뜸(blistering) 가능성\n"
            "2. **습도 경계값** (85%) — 수성 도료 사용 시 건조 지연 위험\n\n"
            "### 조치 사항\n"
            "- 즉시 도장 작업 중단\n"
            "- 제습기 가동 또는 환기 실시\n"
            "- 이슬점 차이 3도C 이상 확보 확인 후 작업 재개\n"
            "- 재개 전 표면 결로 여부 육안 및 기기 점검 필수"
        ),
    },
    "qa_blasting": {
        "rag_chunks": [
            {"chunk_id": "PROC-BL-001", "text": "블라스팅(Blasting)은 연마재를 고압으로 분사하여 강재 표면의 녹, 밀스케일, 구도막을 제거하는 표면처리 공정이다. ISO 8501-1 기준 Sa 2.5 등급 이상 달성 시 도막 부착력이 최적화된다.", "rerank_score": 0.95},
            {"chunk_id": "PROC-BL-003", "text": "블라스팅 연마재 종류: 스틸 그릿(GP/GH), 스틸 샷, 알루미나, 가넷. 연마재 선택은 요구 조도(profile depth), 강재 두께, 재사용성에 따라 결정된다. 표준 조도 범위: 40~75um.", "rerank_score": 0.88},
        ],
        "tool_calls": [],
        "xr_commands": [],
        "response": (
            "## 블라스팅 (Blasting)\n\n"
            "블라스팅은 **연마재를 고압 공기로 분사하여 강재 표면을 처리하는 공정**입니다.\n\n"
            "### 목적\n"
            "- 녹(rust), 밀스케일(mill scale), 기존 도막 제거\n"
            "- 도막 부착력 확보를 위한 표면 조도(profile) 형성\n"
            "- ISO 8501-1 기준 **Sa 2.5 등급** 이상 달성이 표준\n\n"
            "### 연마재 종류\n"
            "| 연마재 | 특징 | 용도 |\n|---|---|---|\n"
            "| 스틸 그릿 | 각진 형태, 높은 절삭력 | 밀스케일/구도막 제거 |\n"
            "| 스틸 샷 | 구형, 균일 조도 | 피닝 효과 필요 시 |\n"
            "| 가넷 | 일회용, 분진 적음 | 개방 환경 작업 |\n\n"
            "### 품질 기준\n"
            "- 표면 조도: 40~75um\n"
            "- 청정도: Sa 2.5 이상 (육안 검사 + 비교판)"
        ),
    },
    "qa_primer": {
        "rag_chunks": [
            {"chunk_id": "COAT-PR-001", "text": "프라이머(Primer)는 강재 표면에 최초로 적용되는 방청 도료로, 아연말(zinc-rich) 프라이머가 선박 도장에서 가장 널리 사용된다. 건조 도막 두께(DFT)는 통상 75~100um.", "rerank_score": 0.93},
        ],
        "tool_calls": [],
        "xr_commands": [],
        "response": (
            "## 프라이머 (Primer)\n\n"
            "프라이머는 **블라스팅 직후 강재에 최초 도포하는 방청 도료**입니다.\n\n"
            "### 역할\n"
            "- 강재와 후속 도막 간 부착력 확보\n"
            "- 부식 방지 (아연 희생양극 원리)\n\n"
            "### 주요 종류\n"
            "| 종류 | 설명 |\n|---|---|\n"
            "| 아연말 프라이머 (Zinc-Rich) | 선박 도장 표준, 희생양극 방식 |\n"
            "| 에폭시 프라이머 | 내화학성, 밸러스트 탱크 등 |\n"
            "| 숍 프라이머 | 블라스팅 직후 임시 방청 (15~25um) |\n\n"
            "### DFT 기준\n"
            "- 표준: 75~100um\n"
            "- 숍 프라이머: 15~25um (임시)"
        ),
    },
    "qa_dewpoint": {
        "rag_chunks": [
            {"chunk_id": "ENV-DEW-001", "text": "결로(Condensation)는 표면 온도가 이슬점 온도 이하로 내려갈 때 대기 중 수분이 표면에 응결되는 현상이다. 도장 시 표면온도와 이슬점 차이가 3도C 미만이면 결로 위험으로 작업을 금지한다.", "rerank_score": 0.96},
        ],
        "tool_calls": [],
        "xr_commands": [],
        "response": (
            "## 결로 (Condensation)\n\n"
            "결로는 **표면 온도가 이슬점 이하일 때 수분이 표면에 응결되는 현상**입니다.\n\n"
            "### 도장에 미치는 영향\n"
            "- 도막 부착력 저하\n"
            "- 들뜸(blistering), 박리 발생\n"
            "- 도막 내 수분 포집으로 조기 부식\n\n"
            "### 판정 기준\n"
            "| 항목 | 기준 | 비고 |\n|---|---|---|\n"
            "| 표면온도 - 이슬점 | 3도C 이상 | 미만 시 작업 금지 |\n"
            "| 상대습도 | 85% 이하 | 초과 시 작업 중단 |\n\n"
            "### 대응 방법\n"
            "- 제습기 가동, 환기\n"
            "- 표면 예열 (히터)\n"
            "- 이슬점 측정기로 지속 모니터링"
        ),
    },
    "qa_dft": {
        "rag_chunks": [
            {"chunk_id": "QC-DFT-001", "text": "DFT(Dry Film Thickness, 건조 도막 두께)는 도막 건조 후 측정하는 두께로, 단위는 um(마이크로미터)이다. 전자식 DFT 게이지로 측정하며, SSPC-PA2 기준에 따라 합부를 판정한다.", "rerank_score": 0.94},
        ],
        "tool_calls": [],
        "xr_commands": [],
        "response": (
            "## DFT (Dry Film Thickness, 건조 도막 두께)\n\n"
            "DFT는 **도료가 건조된 후의 도막 두께**로, 도장 품질의 핵심 지표입니다.\n\n"
            "### 측정 방법\n"
            "- 전자식 DFT 게이지 (자성체/비자성체 구분)\n"
            "- 구역당 최소 5회 측정, 평균 및 개별 편차 기록\n\n"
            "### 합부 판정 (SSPC-PA2)\n"
            "| 규칙 | 설명 |\n|---|---|\n"
            "| 90% 규칙 | 측정값의 90%가 nominal DFT 이상 |\n"
            "| 10% 규칙 | 나머지 10%도 nominal의 80% 이상 |\n\n"
            "### 예시\n"
            "nominal 100um 기준:\n"
            "- [95, 102, 88, 105, 99] → 평균 97.8um, 최소 88um > 80um → PASS\n"
            "- [95, 70, 88, 105, 99] → 70um < 80um → FAIL"
        ),
    },
    "qa_painting": {
        "rag_chunks": [
            {"chunk_id": "COAT-SYS-001", "text": "선박 도장 체계는 표면처리(블라스팅) → 프라이머 → 중도(intermediate coat) → 상도(topcoat)의 다층 구조로 구성된다. 각 층은 고유한 기능(방청, 내수, 내후, 방오)을 수행한다.", "rerank_score": 0.92},
        ],
        "tool_calls": [],
        "xr_commands": [],
        "response": (
            "## 선박 도장 공정 개요\n\n"
            "선박 도장은 **다층 도막 체계**로 부식 방지와 방오 기능을 제공합니다.\n\n"
            "### 도장 체계 (Layer System)\n"
            "| 순서 | 층 | 역할 | 일반 DFT |\n|---|---|---|---|\n"
            "| 1 | 프라이머 | 방청 (아연 희생양극) | 75~100um |\n"
            "| 2 | 중도 (Intermediate) | 차단층, 내수성 | 125~200um |\n"
            "| 3 | 상도 (Topcoat) | 내후성, 미관, 방오 | 75~150um |\n\n"
            "### 도장 전 표면처리\n"
            "- 블라스팅: Sa 2.5 이상\n"
            "- 조도: 40~75um\n\n"
            "### 총 도막 두께\n"
            "- 외판 (수선부): 350~450um\n"
            "- 밸러스트 탱크: 320um 이상\n"
            "- 상부 구조물: 200~300um"
        ),
    },
    "general": {
        "rag_chunks": [
            {"chunk_id": "GENERAL-001", "text": "선박 도장은 부식 방지 및 해양 생물 부착 방지를 위한 핵심 공정으로, 다층 도장 체계로 구성됩니다.", "rerank_score": 0.78},
        ],
        "tool_calls": [],
        "xr_commands": [],
        "response": (
            "질문을 분석했습니다. 보다 정확한 답변을 위해 구체적인 정보를 "
            "알려주시면 도움이 됩니다.\n\n"
            "예를 들어:\n"
            "- DFT 측정값과 기준 nominal 값\n"
            "- 현재 환경 조건 (온도, 습도, 이슬점)\n"
            "- 작업 구역 및 블록 정보\n\n"
            "위 정보를 포함하여 다시 질문해주세요.\n\n"
            "일반 도장 지식 질문 예시:\n"
            "- 블라스팅이 뭐야?\n"
            "- DFT가 뭐야?\n"
            "- 프라이머란?\n"
            "- 결로가 뭐야?"
        ),
    },
}

_session_id: str = str(uuid.uuid4())[:8]


def _new_session() -> str:
    global _session_id
    _session_id = str(uuid.uuid4())[:8]
    return _session_id


def _build_xr_json(scenario: dict[str, Any]) -> str:
    xr = scenario.get("xr_commands", [])
    if not xr:
        return ""
    return json.dumps(xr, ensure_ascii=False, indent=2)


def _demo_stream(
    question: str,
    history: ChatHistory,
) -> Generator[tuple[ChatHistory, str], None, None]:
    scenario_type = _classify_question(question)
    scenario = DEMO_SCENARIOS.get(scenario_type, DEMO_SCENARIOS["general"])
    xr_json = _build_xr_json(scenario)
    step_id = 0

    step_id += 1
    rag_chunks = scenario["rag_chunks"]
    history.append(ChatMessage(role="assistant", content="", metadata={"title": "RAG 검색 중...", "status": "pending", "id": step_id}))
    yield history, xr_json
    time.sleep(0.5)

    rag_text_parts = []
    for chunk in rag_chunks:
        score = chunk.get("rerank_score", 0)
        rag_text_parts.append(f"**[{chunk['chunk_id']}]** (유사도: {score:.2f})\n{chunk['text']}")
    rag_content = "\n\n".join(rag_text_parts)
    history[-1] = ChatMessage(role="assistant", content=f"검색된 문서 {len(rag_chunks)}건:\n\n{rag_content}", metadata={"title": f"RAG 검색 완료 -- {len(rag_chunks)}건", "status": "done", "id": step_id})
    yield history, xr_json

    tool_calls = scenario["tool_calls"]
    for tc in tool_calls:
        step_id += 1
        tool_name = tc["name"]
        display_name = _TOOL_DISPLAY.get(tool_name, tool_name)
        history.append(ChatMessage(role="assistant", content="", metadata={"title": f"MCP: {display_name} 호출 중...", "status": "pending", "id": step_id}))
        yield history, xr_json
        time.sleep(0.3)
        args_str = json.dumps(tc["args"], ensure_ascii=False, indent=2)
        result_str = json.dumps(tc["result"], ensure_ascii=False, indent=2)
        tool_content = f"**입력 파라미터:**\n```json\n{args_str}\n```\n\n**실행 결과:**\n```json\n{result_str}\n```"
        history[-1] = ChatMessage(role="assistant", content=tool_content, metadata={"title": f"MCP: {display_name} 완료", "status": "done", "id": step_id})
        yield history, xr_json

    step_id += 1
    history.append(ChatMessage(role="assistant", content="", metadata={"title": "QLoRA 추론 중...", "status": "pending", "id": step_id}))
    yield history, xr_json
    time.sleep(1.0)
    history[-1] = ChatMessage(role="assistant", content=f"모델: {MODEL_NAME}\n컨텍스트: RAG {len(rag_chunks)}건 + 도구 결과 {len(tool_calls)}건 통합 추론 완료", metadata={"title": "QLoRA 추론 완료", "status": "done", "id": step_id})
    yield history, xr_json

    response = scenario["response"]
    history.append(ChatMessage(role="assistant", content=""))
    yield history, xr_json
    accumulated = ""
    chunk_size = 3
    for i in range(0, len(response), chunk_size):
        accumulated += response[i : i + chunk_size]
        history[-1] = ChatMessage(role="assistant", content=accumulated)
        yield history, xr_json
        time.sleep(0.01)
    history[-1] = ChatMessage(role="assistant", content=response)
    yield history, xr_json


def _live_stream(
    question: str,
    history: ChatHistory,
) -> Generator[tuple[ChatHistory, str], None, None]:
    xr_json = ""
    import importlib
    import sys
    from pathlib import Path

    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    agent_mod = importlib.import_module("agent_pipeline")
    AgentPipeline = agent_mod.AgentPipeline

    step_id = 1
    history.append(ChatMessage(role="assistant", content="파이프라인 초기화 중...", metadata={"title": "AgentPipeline 로드", "status": "pending", "id": step_id}))
    yield history, xr_json

    try:
        pipeline = AgentPipeline(config_path=CONFIG_PATH, adapter_path=ADAPTER_PATH, mcp_base_url=MCP_BASE_URL)
    except Exception as e:
        history[-1] = ChatMessage(role="assistant", content=f"파이프라인 초기화 실패: {e}", metadata={"title": "AgentPipeline 오류", "status": "done", "id": step_id})
        yield history, xr_json
        return

    history[-1] = ChatMessage(role="assistant", content="모델 및 RAG 인덱스 로드 완료", metadata={"title": "AgentPipeline 준비 완료", "status": "done", "id": step_id})
    yield history, xr_json

    step_id += 1
    history.append(ChatMessage(role="assistant", content="", metadata={"title": "에이전트 파이프라인 실행 중...", "status": "pending", "id": step_id}))
    yield history, xr_json

    try:
        result = pipeline.run(instruction=question, session_id=_session_id, enable_thinking=True)
    except Exception as e:
        history[-1] = ChatMessage(role="assistant", content=f"실행 오류: {e}", metadata={"title": "파이프라인 오류", "status": "done", "id": step_id})
        yield history, xr_json
        return

    meta = result.get("metadata", {})
    route = meta.get("route", result.get("mode", "N/A"))
    rounds = meta.get("rounds", 0)
    latency = meta.get("latency_ms", 0)
    history[-1] = ChatMessage(role="assistant", content=f"모드: {route} | 라운드: {rounds} | 지연: {latency}ms", metadata={"title": "파이프라인 실행 완료", "status": "done", "id": step_id})
    yield history, xr_json

    rag_chunks = result.get("rag_chunks", [])
    if rag_chunks:
        step_id += 1
        rag_parts = []
        for chunk in rag_chunks:
            score = chunk.get("rerank_score")
            score_text = f" (유사도: {score:.2f})" if score else ""
            rag_parts.append(f"**[{chunk.get('chunk_id', '')}]**{score_text}\n{chunk.get('text', '')}")
        history.append(ChatMessage(role="assistant", content=f"검색된 문서 {len(rag_chunks)}건:\n\n" + "\n\n".join(rag_parts), metadata={"title": f"RAG 검색 결과 -- {len(rag_chunks)}건", "status": "done", "id": step_id}))
        yield history, xr_json

    tool_calls = result.get("tool_calls", [])
    for tc in tool_calls:
        step_id += 1
        name = tc.get("name", "unknown")
        display_name = _TOOL_DISPLAY.get(name, name)
        args_str = json.dumps(tc.get("arguments", {}), ensure_ascii=False, indent=2)
        result_str = json.dumps(tc.get("result", {}), ensure_ascii=False, indent=2)
        history.append(ChatMessage(role="assistant", content=f"**입력 파라미터:**\n```json\n{args_str}\n```\n\n**실행 결과:**\n```json\n{result_str}\n```", metadata={"title": f"MCP: {display_name}", "status": "done", "id": step_id}))
        yield history, xr_json

    for section_name, section_label in [("dft_result", "DFT 판정 결과"), ("anomaly", "이상 탐지 결과"), ("pose_comparison", "자세 비교 결과"), ("next_path", "다음 도장 경로")]:
        section = result.get(section_name)
        if section is not None:
            step_id += 1
            history.append(ChatMessage(role="assistant", content=f"```json\n{json.dumps(section, ensure_ascii=False, indent=2)}\n```", metadata={"title": section_label, "status": "done", "id": step_id}))
            yield history, xr_json

    text_guide = result.get("text_guide", {})
    response_text = text_guide.get("body", "") if text_guide else result.get("response", "응답을 생성하지 못했습니다.")
    history.append(ChatMessage(role="assistant", content=""))
    yield history, xr_json
    accumulated = ""
    chunk_size = 3
    for i in range(0, len(response_text), chunk_size):
        accumulated += response_text[i : i + chunk_size]
        history[-1] = ChatMessage(role="assistant", content=accumulated)
        yield history, xr_json
        time.sleep(0.01)
    history[-1] = ChatMessage(role="assistant", content=response_text)
    yield history, xr_json

    citations = result.get("citations", [])
    if citations:
        step_id += 1
        history.append(ChatMessage(role="assistant", content=f"**참조 문서:** {', '.join(citations)}", metadata={"title": "인용 문서", "status": "done", "id": step_id}))
        yield history, xr_json


def respond(message: str, history: ChatHistory) -> Generator[tuple[ChatHistory, str], None, None]:
    if not message.strip():
        yield history, ""
        return
    history.append(ChatMessage(role="user", content=message))
    yield history, ""
    if DEMO_MODE:
        yield from _demo_stream(message, history)
    else:
        yield from _live_stream(message, history)


def handle_example_click(example_text: str, history: ChatHistory) -> Generator[tuple[ChatHistory, str], None, None]:
    yield from respond(example_text, history)


def build_sidebar() -> dict[str, Any]:
    mode_badge = '<span class="status-badge status-demo">DEMO</span>' if DEMO_MODE else '<span class="status-badge status-live">LIVE</span>'
    gr.HTML(f'<div style="text-align:center; margin-bottom:12px;"><span style="font-size:1.2em; font-weight:600; color:#e4e4e7;">시스템 상태</span><br/>{mode_badge}</div>')

    with gr.Group():
        gr.Markdown("#### 파이프라인 구성")
        if DEMO_MODE:
            gr.Markdown("- RAG 검색 엔진 — 시뮬레이션\n- QLoRA 모델 — 시뮬레이션\n- MCP 도구 서버 — 시뮬레이션")
        else:
            gr.Markdown("- RAG 검색 엔진 — 연결 중\n- QLoRA 모델 — 연결 중\n- MCP 도구 서버 — 연결 중")

    with gr.Group():
        gr.Markdown("#### 세션 정보")
        gr.Markdown(f"- **세션 ID:** `{_session_id}`")
        gr.Markdown(f"- **모델:** {MODEL_NAME}")
        gr.Markdown(f"- **MCP:** `{MCP_BASE_URL}`")

    with gr.Group():
        gr.Markdown("#### 도구 목록")
        tool_list = "\n".join(f"- {v}" for v in _TOOL_DISPLAY.values())
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
        gr.Markdown("아래 예시 질문을 클릭하거나\n직접 입력하여 AI 코파일럿과\n대화를 시작하세요.")

    return {"mode_selector": mode_selector}


def create_app() -> gr.Blocks:
    with gr.Blocks(title="선박 도장 AI Co-Pilot") as app:
        gr.HTML('<div class="title-section"><h1>선박 도장 AI Co-Pilot</h1><p>도장 품질 분석 / 실시간 코칭 / 도구 연동</p></div>')

        with gr.Row():
            with gr.Column(scale=1, min_width=250):
                build_sidebar()
                new_session_btn = gr.Button("새 세션", variant="secondary", size="sm")

            with gr.Column(scale=4):
                chatbot = gr.Chatbot(
                    value=[],
                    height=580,
                    placeholder="선박 도장 AI Co-Pilot에 질문하세요.\nDFT 판정, 도막 분포 분석, 도장 순서, 실시간 코칭 등을 지원합니다.",
                    layout="panel",
                    show_label=False,
                    elem_classes=["chatbot-container"],
                    buttons=["copy"],
                )

                with gr.Row():
                    msg_input = gr.Textbox(placeholder="도장 관련 질문을 입력하세요...", show_label=False, scale=6, container=False, autofocus=True)
                    send_btn = gr.Button("전송", variant="primary", scale=1, min_width=80)
                    clear_btn = gr.Button("초기화", variant="secondary", scale=1, min_width=80)

                gr.Markdown("#### XR Simulator Output")
                xr_panel = gr.Code(value="", language="json", label="XR Simulator Protocol", lines=12, interactive=False)

                gr.Markdown("#### 예시 질문", elem_classes=["example-section-title"])

                with gr.Row():
                    example_btns = []
                    for ex in EXAMPLE_QUESTIONS[:3]:
                        btn = gr.Button(ex["display_text"], variant="secondary", size="sm", elem_classes=["example-btn"])
                        example_btns.append((btn, ex["text"]))

                with gr.Row():
                    for ex in EXAMPLE_QUESTIONS[3:]:
                        btn = gr.Button(ex["display_text"], variant="secondary", size="sm", elem_classes=["example-btn"])
                        example_btns.append((btn, ex["text"]))

        msg_input.submit(fn=respond, inputs=[msg_input, chatbot], outputs=[chatbot, xr_panel]).then(fn=lambda: "", outputs=[msg_input])
        send_btn.click(fn=respond, inputs=[msg_input, chatbot], outputs=[chatbot, xr_panel]).then(fn=lambda: "", outputs=[msg_input])
        clear_btn.click(fn=lambda: ([], ""), outputs=[chatbot, xr_panel])

        def _reset_session() -> tuple[list[Any], str, str]:
            sid = _new_session()
            return [], "", f"새 세션이 시작되었습니다. (ID: {sid})"

        new_session_btn.click(fn=_reset_session, outputs=[chatbot, xr_panel, msg_input])

        for btn, text in example_btns:
            btn.click(fn=handle_example_click, inputs=[gr.State(text), chatbot], outputs=[chatbot, xr_panel])

    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    mode_label = "DEMO" if DEMO_MODE else "LIVE"
    logger.info("Starting Ship Painting AI Co-Pilot in %s mode on port %d", mode_label, APP_PORT)
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=APP_PORT, share=False, show_error=True, theme=CUSTOM_THEME, css=CUSTOM_CSS)
