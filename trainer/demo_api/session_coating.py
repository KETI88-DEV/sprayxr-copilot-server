"""`POST /api/v1/sessions/complete`의 세션 종합 도막 분석 — 리포트의 `coating_report`.

세션이 끝났을 때의 두께를 구역마다 전문가 최종 도막과 비교한다. 판정은 시퀀스마다 도막을 안내하는
`CoatingProgressEngine`과 **같은 기준·같은 코드**다(`measure_zone`) — 훈련 중 "충족"이던 구역이 리포트에서
"부족"으로 갈리지 않게. 구역 경계는 TSB 라벨맵이다(프로필이 `tsb_color_label_map`일 때).

두께를 얻는 순서:
1. complete의 `height_map_image` — 세션 끝 두께 그대로다. 파일명의 `MapNN`이 면 번호(없으면 1).
2. 없거나 읽을 수 없으면 저장된 시퀀스의 `cur_sequence_heatmap` 크롭을 시간 순으로 덮어쓴다. 크롭 값이 그 시점의
   **누적** 두께라 더하지 않는다.

내지 않는 것: 6구간 면적비율(`zone_summary`)과 `coating_distribution_index`는 구간 경계·계산식을 받지 못했다.
`reachability`·`root_cause`도 근거가 없다. 전부 null로 둔다 — 값을 지어내지 않는다.
"""

from __future__ import annotations

import io
import logging
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np
from PIL import Image, UnidentifiedImageError

from trainer.demo_api.coating_progress import CoatingProgressEngine, ZoneJudgment
from trainer.demo_api.telemetry_contracts import learning_mode_from_scenario, process_from_tool_type

logger = logging.getLogger(__name__)

ALGORITHM_VERSION: Final = "coating-session-v1"
# TSB 높이맵의 RG16 눈금(2026-10-08 샘플 notes: um = (R/255 + G/65025) * 20000).
HEIGHTMAP_SCALE_UM: Final = 20000.0
ATLAS_SIZE: Final = 2048
WORST_ZONE_LIMIT: Final = 3
SOURCE_CROPS: Final = "cur_sequence_heatmap"
_MAP_NUMBER: Final = re.compile(r"Map(\d+)", re.IGNORECASE)
# `CoatingProgressEngine` 구역 상태 → 계약의 도막 상태(응답 계약 §3.7).
_STATUS: Final = {"complete": "normal", "needs_more": "under_app", "overpaint_risk": "over_app"}
_STATUS_LABEL: Final = {"under_app": "부족", "over_app": "과도장"}


@dataclass
class SessionCoating:
    """`report`는 `CoatingReport` 모양, `zones`는 질의가 쓰는 구역별 판정(리포트 본문에는 싣지 않는다)."""

    report: dict[str, Any]
    zones: list[dict[str, Any]] = field(default_factory=list)


def _unavailable(reason: str, guideline: str) -> SessionCoating:
    return SessionCoating({"analysis_status": "unavailable", "reason_code": reason, "guideline": guideline})


def surfaces_from_height_map(raw: bytes, *, filename: str | None) -> dict[int, np.ndarray]:
    """높이맵 PNG → {면 번호: 좌하단 원점 두께(µm)}. 읽을 수 없으면 빈 dict."""
    try:
        pixels = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.float32)
    except (UnidentifiedImageError, OSError, ValueError):
        logger.warning("height map unreadable (%d bytes)", len(raw))
        return {}
    thickness = (pixels[:, :, 0] / 255.0 + pixels[:, :, 1] / 65025.0) * HEIGHTMAP_SCALE_UM
    match = _MAP_NUMBER.search(filename or "")
    # PNG은 맨 윗행이 v=1이고 판정 표면은 좌하단 원점이다.
    return {int(match.group(1)) if match else 1: np.ascontiguousarray(np.flipud(thickness))}


def surfaces_from_crops(bodies: Iterable[Mapping[str, Any]]) -> dict[int, np.ndarray]:
    """시퀀스 크롭을 주어진 순서대로 덮어쓴 세션 끝 두께. 나중 크롭이 같은 화소의 앞 값을 대신한다."""
    surfaces: dict[int, np.ndarray] = {}
    for body in bodies:
        result = body.get("sequence_result")
        items = result.get("cur_sequence_heatmap") if isinstance(result, Mapping) else None
        for item in items if isinstance(items, list) else []:
            if not isinstance(item, dict):
                continue
            crop, _ = CoatingProgressEngine._decode_crop(  # noqa: SLF001 - 실시간 안내와 같은 디코더
                item, atlas_width=ATLAS_SIZE, atlas_height=ATLAS_SIZE
            )
            if crop is None:
                continue
            try:
                map_num = int(item.get("map_num", 1))
            except (TypeError, ValueError):
                continue
            surface = surfaces.setdefault(map_num, np.zeros((ATLAS_SIZE, ATLAS_SIZE), dtype=np.float32))
            rows, columns = np.nonzero(crop.valid)
            if rows.size:
                surface[crop.y_indices[rows], crop.x_indices[columns]] = crop.thickness_um[rows, columns]
    return surfaces


def _process(bodies: Iterable[Mapping[str, Any]]) -> str | None:
    for body in bodies:
        setting = body.get("sequence_setting")
        process = process_from_tool_type(setting.get("tool_type")) if isinstance(setting, Mapping) else None
        if process is not None:
            return process
    return None


def _zone_row(judgment: ZoneJudgment) -> dict[str, Any]:
    coverage = judgment.coverage
    return {
        "zone_id": judgment.zone_id,
        "map_num": judgment.map_num,
        "status": _STATUS[coverage.status],
        "thickness_um": round(coverage.mean_um, 1),
        "completion_percent": coverage.completion_percent,
        "coverage_percent": coverage.coverage_percent,
        "overpaint_percent": coverage.overpaint_percent,
        # 판정 셀 중 기준을 벗어난(부족 + 과도장) 비율. 구역끼리 심한 정도를 견준다.
        "out_of_band_percent": round((coverage.below + coverage.above) / coverage.cells * 100.0, 1),
    }


def _guideline(zones: list[dict[str, Any]], worst: list[dict[str, Any]]) -> str:
    def ids(status: str) -> list[str]:
        return [zone["zone_id"] for zone in zones if zone["status"] == status]

    under, over = ids("under_app"), ids("over_app")
    parts = [f"전문가 최종 도막과 비교한 {len(zones)}개 구역 중 기준 충족 {len(zones) - len(under) - len(over)}개"]
    if under:
        parts.append(f"부족 {len(under)}개({'·'.join(under)}번)")
    if over:
        parts.append(f"과도장 {len(over)}개({'·'.join(over)}번)")
    text = ", ".join(parts) + "입니다."
    if worst:
        top = worst[0]
        text += (
            f" 가장 벗어난 곳은 {top['zone_id']}번 구역({_STATUS_LABEL[top['status']]}, "
            f"평균 {top['thickness_um']:.0f}µm)입니다."
        )
    return text


def analyze_session_coating(
    *,
    scenario_id: str,
    bodies: list[Mapping[str, Any]],
    engine: CoatingProgressEngine,
    height_map: bytes | None = None,
    height_map_name: str | None = None,
    target_thickness_um: float | None = None,
) -> SessionCoating:
    """세션 끝 도막을 구역별로 판정한다(모듈 독스트링의 순서). `bodies`는 시간 순서여야 한다."""
    mode_id = learning_mode_from_scenario(scenario_id)
    if mode_id is None:
        return _unavailable("reference_context_unavailable", "학습 콘텐츠가 아니어서 전문가 도막 기준이 없습니다.")
    process = _process(bodies)
    if process == "blasting":
        return _unavailable("coating_not_applicable", "블라스팅 세션은 도막 두께를 판정하지 않습니다.")
    if process is None:
        return _unavailable("reference_context_unavailable", "분사 공정을 알 수 없어 도막 기준을 고르지 못했습니다.")

    surfaces: dict[int, np.ndarray] = {}
    source = SOURCE_CROPS
    if height_map:
        surfaces = surfaces_from_height_map(height_map, filename=height_map_name)
        source = height_map_name or "height_map_image"
    if not surfaces:
        surfaces, source = surfaces_from_crops(bodies), SOURCE_CROPS
    if not surfaces:
        return _unavailable("heatmap_unavailable", "세션 끝 도막 두께(높이맵·시퀀스 히트맵)가 없습니다.")

    judgments, reason = engine.judge_zones(
        scenario_id=scenario_id, process=process, mode_id=mode_id, surfaces=surfaces,
        target_thickness_um=target_thickness_um,
    )
    if not judgments:
        return _unavailable(reason or "coating_reference_unavailable", "이 콘텐츠의 전문가 도막 기준이 없습니다.")

    zones = [_zone_row(judgment) for judgment in judgments]
    # 안정 정렬이라 같은 비율이면 구역 순서를 지킨다.
    worst = sorted(
        (zone for zone in zones if zone["status"] != "normal"), key=lambda zone: -zone["out_of_band_percent"]
    )[:WORST_ZONE_LIMIT]
    below = sum(judgment.coverage.below for judgment in judgments)
    above = sum(judgment.coverage.above for judgment in judgments)
    cells = sum(judgment.coverage.cells for judgment in judgments)
    report = {
        "comparison_basis": "session_target" if target_thickness_um is not None else "expert_reference",
        "target_thickness_um": target_thickness_um,
        "analysis_status": "available",
        "reason_code": None,
        "algorithm_version": ALGORITHM_VERSION,
        "source_artifact_id": source,
        "guideline": _guideline(zones, worst),
        "distribution": {"below": below, "in_band": cells - below - above, "above": above},
        "worst_zones": [
            {
                "zone_id": zone["zone_id"],
                "map_num": zone["map_num"],
                "thickness_um": zone["thickness_um"],
                "status": zone["status"],
            }
            for zone in worst
        ],
    }
    if target_thickness_um is not None:
        report["guideline"] = (
            f"세션 목표 두께 {target_thickness_um:g}µm의 90~110% 범위를 기준으로 평가했습니다. "
            + report["guideline"].replace("전문가 최종 도막과 비교한", "목표 도막과 비교한")
        )
    return SessionCoating(report, zones)
