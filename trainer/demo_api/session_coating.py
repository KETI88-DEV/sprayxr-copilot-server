"""`POST /api/v1/sessions/complete`의 세션 종합 도막 분석 — 리포트의 `coating_report`.

세션이 끝났을 때의 두께를 구역마다 목표 두께에 대고 판정한다. 기준은 PR #8 도막 기준이고 계산은
`coating_assessment` 어댑터가 한다: 보정 두께로 재고, 칠한 곳의 평균으로 저도막·정상·과도막을 가른다(경계 값은
어댑터에만 있다). 세션이 목표를 명시하지 않았으면 기본 목표로 판정하고 `target_basis="default_target"`으로 밝힌다.
구역 경계와 판정할 구역 목록은 TSB 공식 라벨맵이다. 표면이 있는 면의 모든 공식 구역을 판정하며,
전문가 프로필은 별도의 참고 비교에만 사용한다.

두께를 얻는 순서:
1. complete의 `height_map_image` — 세션 끝 누적 두께로 읽는다(회차 합산 없음). 파일명의 `MapNN`이 면 번호(없으면 1).
2. 없거나 읽을 수 없으면 저장된 시퀀스의 `cur_sequence_heatmap` 크롭을 시간 순으로 덮어쓴다. 크롭 값이 그 시점의
   **누적** 두께라 더하지 않는다.

내는 것: 구역별 등급·칠한 곳 평균·미도장률, 칠한 면적의 6구간 비율(`zone_summary`), 3분류 화소 수(`distribution`),
가장 벗어난 구역과 조절 후보, 참고로 전문가 최종 도막 칸과 견준 구역 상태(`expert_comparison`, 보정 전 두께). 조절 후보는 규칙에 따른 후보이고 확정된 원인이 아니다.
내지 않는 것: `coating_distribution_index`(계산식을 받지 못했다), `reachability`·`root_cause`(근거가 없다), 세션 등급
(PR #8에도 없다). 전부 null로 둔다 — 값을 지어내지 않는다.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np

from trainer.demo_api import coating_assessment
from trainer.demo_api.coating_progress import CoatingProgressEngine, ZoneCoverage, expert_status
from trainer.demo_api.telemetry_contracts import learning_mode_from_scenario, process_from_tool_type

logger = logging.getLogger(__name__)

ALGORITHM_VERSION: Final = f"coating-session-v2/{coating_assessment.CRITERION_VERSION}"
ATLAS_SIZE: Final = 2048
WORST_ZONE_LIMIT: Final = 3
SOURCE_CROPS: Final = "cur_sequence_heatmap"
_MAP_NUMBER: Final = re.compile(r"Map(\d+)", re.IGNORECASE)
# 계약의 6영역 이름(`CoatingZoneRatio`) — PR #8 6구간과 같은 순서다.
_RATIO_KEYS: Final = ("under2", "under1", "normal", "over1", "over2", "over3")
# 목표 대비 등급 낱말(저도막·과도막)과 섞이지 않게 따로 둔다.
EXPERT_WORDS: Final = {
    "normal": "전문가 기준 충족", "under_app": "전문가보다 얇은 구역", "over_app": "전문가보다 두꺼운 구역",
}
_CAUSE_WORDS: Final = {"speed_mm_s": "이동 속도", "distance_mm": "이격거리"}


@dataclass
class SessionCoating:
    """`report`는 `CoatingReport` 모양, `zones`는 질의가 쓰는 구역별 판정(리포트 본문에는 싣지 않는다)."""

    report: dict[str, Any]
    zones: list[dict[str, Any]] = field(default_factory=list)


def _unavailable(reason: str, guideline: str) -> SessionCoating:
    return SessionCoating({"analysis_status": "unavailable", "reason_code": reason, "guideline": guideline})


def surfaces_from_height_map(raw: bytes, *, filename: str | None) -> dict[int, np.ndarray]:
    """높이맵 PNG → {면 번호: 좌하단 원점 원시 두께(µm), 칠한 자리가 아니면 0}. 읽을 수 없으면 빈 dict."""
    try:
        thickness = coating_assessment.painted_surface_from_png(raw)
    except ValueError:
        logger.warning("height map unreadable (%d bytes)", len(raw))
        return {}
    match = _MAP_NUMBER.search(filename or "")
    # PNG은 맨 윗행이 v=1이고 판정 표면은 좌하단 원점이다.
    return {int(match.group(1)) if match else 1: np.ascontiguousarray(np.flipud(thickness))}


def surfaces_from_crops(bodies: Iterable[Mapping[str, Any]]) -> dict[int, np.ndarray]:
    """시퀀스 크롭을 주어진 순서대로 덮어쓴 세션 끝 두께. 나중 크롭이 같은 화소의 앞 값을 대신한다.

    PR #8 `cumulative_crop` 교체 규칙의 main 쪽 사본이다. 다른 점은 크롭 안의 B=0 화소가 앞선 값을 지우지 않는 것
    (`coating_assessment` 독스트링 4번).
    """
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


def _expert_row(coverage: ZoneCoverage | None) -> dict[str, Any] | None:
    if coverage is None:
        return None
    return {
        "status": expert_status(coverage),
        "coverage_percent": coverage.coverage_percent,
        "completion_percent": coverage.completion_percent,
        "overpaint_percent": coverage.overpaint_percent,
    }


def _expert_comparison(zones: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [zone for zone in zones if zone.get("expert") is not None]
    counts = {key: sum(zone["expert"]["status"] == key for zone in rows) for key in EXPERT_WORDS}
    return {
        "comparison_basis": "expert_reference",
        "analysis_status": "available" if rows else "unavailable",
        "reason_code": None if rows else "expert_cells_unavailable",
        "zone_count": len(rows),
        **counts,
        "zones": [{"zone_id": zone["zone_id"], "map_num": zone["map_num"], **zone["expert"]} for zone in rows],
    }


def _expert_sentence(comparison: Mapping[str, Any]) -> str:
    if comparison["analysis_status"] != "available":
        return ""
    parts = [f"{EXPERT_WORDS['normal']} {comparison['normal']}개"]
    for key in ("under_app", "over_app"):
        found = [row["zone_id"] for row in comparison["zones"] if row["status"] == key]
        if found:
            parts.append(f"{EXPERT_WORDS[key]} {len(found)}개({'·'.join(found)}번)")
    return f" 전문가 최종 도막 대비: {comparison['zone_count']}개 구역, {', '.join(parts)}(보정 전 두께 기준)."


def _guideline(
    zones: list[dict[str, Any]], worst: list[dict[str, Any]], summary: Mapping[str, Any], expert: Mapping[str, Any]
) -> str:
    def ids(grade: str | None) -> list[str]:
        return [zone["zone_id"] for zone in zones if zone["grade"] == grade]

    parts = [f"정상 {len(ids('normal'))}개"]
    for label, grade in (("저도막", "below"), ("과도막", "above"), ("칠한 곳 없음", None)):
        found = ids(grade)
        if found:
            parts.append(f"{label} {len(found)}개({'·'.join(found)}번)")
    basis = coating_assessment.basis_label(summary["target_thickness_um"], summary["target_basis"])
    lower, upper = summary["normal_band_um"]
    text = (f"{basis}, 정상범위 {lower:g}~{upper:g}µm(경계 포함): {len(zones)}개 구역, "
            f"{', '.join(parts)}. 칠한 곳 평균 기준(하한 미만 저도막·상한 초과 과도막).")
    if worst:
        top = worst[0]
        text += (
            f" 최대 이탈: {top['zone_id']}번 구역({'저도막' if top['status'] == 'under_app' else '과도막'}, "
            f"칠한 곳 평균 {coating_assessment.mean_text(top['thickness_um'], top['target_thickness_um'])}µm)."
        )
    if summary["painted_mean_um"] is not None:
        # 손대지 않은 구역도 미도장으로 들어간 값이라 범위를 "판정한 구역 전체"라고 밝힌다.
        text += (
            f" 판정 구역 전체: 칠한 곳 평균 {summary['painted_mean_um']:.0f}µm, "
            f"미도장 {summary['unpainted_fraction'] * 100:.0f}%."
        )
    return text + _expert_sentence(expert)


def _worst_zone(zone: Mapping[str, Any]) -> dict[str, Any]:
    variables = [_CAUSE_WORDS[item["variable"]] for item in zone["adjustments"]]
    return {
        "zone_id": zone["zone_id"],
        "map_num": zone["map_num"],
        "thickness_um": zone["thickness_um"],
        "status": zone["status"],
        "action": coating_assessment.adjustment_sentence(zone["adjustments"]),
        "cause": f"원인 후보: {', '.join(variables)} (확정 아님)" if variables else None,
    }


def analyze_session_coating(
    *,
    scenario_id: str,
    bodies: list[Mapping[str, Any]],
    engine: CoatingProgressEngine,
    height_map: bytes | None = None,
    height_map_name: str | None = None,
    target_thickness_um: float | None = None,
    min_thickness_um: float | None = None,
    max_thickness_um: float | None = None,
    tool_type: str | None = None,
) -> SessionCoating:
    """세션 끝 도막을 구역별로 판정한다(모듈 독스트링의 순서). `bodies`는 시간 순서여야 한다."""
    mode_id = learning_mode_from_scenario(scenario_id)
    if mode_id is None:
        return _unavailable("reference_context_unavailable", "학습 콘텐츠가 아니어서 도막을 판정할 Zone 기준 자료가 없습니다.")
    # 관절 CSV를 복원하지 못해도 시작 시 확정한 공정과 최종 높이맵은 유효하다.
    # 옛 세션에는 tool_type이 없으므로 원래 시퀀스 본문의 공정 추론을 유지한다.
    process = process_from_tool_type(tool_type) or _process(bodies)
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
        target_thickness_um=target_thickness_um, min_thickness_um=min_thickness_um,
        max_thickness_um=max_thickness_um, behavior=coating_assessment.behavior_by_zone(bodies),
    )
    if not judgments:
        return _unavailable(
            reason or "zone_reference_unavailable", "이 콘텐츠의 공식 Zone 자료가 없어 도막을 판정하지 못했습니다."
        )

    zones = [
        {**coating_assessment.zone_row(judgment.map_num, judgment.zone_id, judgment.assessment),
         "expert": _expert_row(judgment.expert)}
        for judgment in judgments
    ]
    expert_comparison = _expert_comparison(zones)
    summary = coating_assessment.session_summary(judgment.assessment for judgment in judgments)
    # 안정 정렬이라 벗어난 정도가 같으면 구역 순서를 지킨다.
    worst = sorted(
        (zone for zone in zones if zone["status"] in {"under_app", "over_app"}),
        key=lambda zone: -abs(zone["thickness_um"] - zone["target_thickness_um"]) / zone["target_thickness_um"],
    )[:WORST_ZONE_LIMIT]
    bands = [summary["six_band_counts"][key] for key in coating_assessment.SIX_BAND_KEYS]
    painted = summary["painted_pixel_count"]
    report = {
        "comparison_basis": "session_target",
        "target_thickness_um": summary["target_thickness_um"],
        "target_basis": summary["target_basis"],
        "normal_band_um": summary["normal_band_um"],
        "analysis_status": "available",
        "reason_code": None,
        "algorithm_version": ALGORITHM_VERSION,
        "source_artifact_id": source,
        "guideline": _guideline(zones, worst, summary, expert_comparison),
        # 단위는 칠한 화소 수다. 6구간을 목표 미만 / 정상 구간 / 과도막 경계 이상으로 묶는다.
        "distribution": {"below": sum(bands[:2]), "in_band": bands[2], "above": sum(bands[3:])},
        "zone_summary": dict(zip(_RATIO_KEYS, (count / painted for count in bands))) if painted else None,
        "worst_zones": [_worst_zone(zone) for zone in worst],
        "expert_comparison": expert_comparison,
    }
    return SessionCoating(report, zones)
