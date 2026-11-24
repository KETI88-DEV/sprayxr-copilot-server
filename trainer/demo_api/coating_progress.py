"""Current-Zone completion: cumulative crops, official Zone masks, PR #8 corrected thickness vs the target."""

from __future__ import annotations

import base64
import binascii
import io
import json
import math
import threading
from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image, UnidentifiedImageError

from trainer.demo_api import coating_assessment
from trainer.demo_api.telemetry_contracts import AnalyzeSequenceRequest, CoatingProgress
from trainer.zone_reference import ZoneReferenceCatalog

# Persisted responses include this revision so a retry cannot reuse coverage
# calculated by the old one-sample/one-atlas-pixel placement.
CROP_PLACEMENT_VERSION = "atlas-nearest-20261124"

_MAX_SESSIONS = 16
DEFAULT_TARGET_THICKNESS_UM = coating_assessment.DEFAULT_TARGET_UM
# 1차(b39ea5f) 이전 main의 전문가 대비 상태 규칙. PR #8 기준이 아니다. 리포트의 참고 판정에만 쓴다.
_EXPERT_OVERPAINT_PERCENT = 25.0
_EXPERT_COMPLETE_PERCENT = 90.0
_EXPERT_COVERAGE_PERCENT = 85.0


@dataclass(frozen=True)
class _DecodedCrop:
    x_indices: np.ndarray
    y_indices: np.ndarray
    thickness_um: np.ndarray
    valid: np.ndarray


@dataclass(frozen=True)
class ZoneCoverage:
    """전문가 셀(q1·q3)에 대고 잰 한 구역의 도막. 셀 수는 판정에 쓴 셀만이다."""

    cells: int
    below: int  # q1 미달 셀
    above: int  # 과도장 셀
    completion_percent: float
    coverage_percent: float
    overpaint_percent: float
    mean_um: float


def expert_status(coverage: ZoneCoverage) -> Literal["normal", "under_app", "over_app"]:
    if coverage.overpaint_percent >= _EXPERT_OVERPAINT_PERCENT:
        return "over_app"
    if coverage.completion_percent >= _EXPERT_COMPLETE_PERCENT and coverage.coverage_percent >= _EXPERT_COVERAGE_PERCENT:
        return "normal"
    return "under_app"


@dataclass(frozen=True)
class ZoneJudgment:
    map_num: int
    zone_id: str
    assessment: dict[str, Any]  # `coating_assessment.assess_zone`의 반올림하지 않은 값
    expert: ZoneCoverage | None = None  # 전문가 칸과 견준 참고 판정(보정 전 두께). 등급·게이트에는 쓰지 않는다.


def measure_zone(
    surface: np.ndarray,
    target_cells: dict[str, Any],
    *,
    atlas_width: int,
    atlas_height: int,
    grid_size: int,
    zone_mask: np.ndarray | None,
) -> ZoneCoverage | None:
    """좌하단 원점 두께 표면을 전문가 셀에 대고 잰다. 잴 셀이 하나도 없으면 None."""
    completion_values: list[float] = []
    actual_values: list[float] = []
    covered = 0
    overpainted = 0
    for key, target in target_cells.items():
        if not isinstance(target, dict):
            continue
        try:
            q1 = float(target["q1_um"])
            q3 = float(target["q3_um"])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(q1) or q1 <= 0 or not math.isfinite(q3):
            continue
        try:
            gx_text, gy_text = key.split(":", 1)
            gx, gy = int(gx_text), int(gy_text)
        except (AttributeError, TypeError, ValueError):
            continue
        left = math.floor(gx * atlas_width / grid_size)
        right = math.ceil((gx + 1) * atlas_width / grid_size)
        bottom = math.floor(gy * atlas_height / grid_size)
        top = math.ceil((gy + 1) * atlas_height / grid_size)
        if left >= right or bottom >= top:
            continue
        actual_region = surface[bottom:top, left:right]
        if zone_mask is not None:
            mask_region = zone_mask[bottom:top, left:right]
            if not np.any(mask_region):
                continue
            actual = float(actual_region[mask_region].mean())
        else:
            actual = float(actual_region.mean())
        completion_values.append(min(actual / q1, 1.0))
        actual_values.append(actual)
        # RG16 at scale_um=20000 has about 0.31 um quantization. Allow
        # a 0.5 um boundary tolerance so an encoded 100 um target does
        # not become 99.96 um and incorrectly miss coverage.
        covered += int(actual >= q1 - 0.5)
        overpainted += int(actual > q3 + max(q3 - q1, q3 * 0.1, 5.0) + .5)
    if not completion_values:
        return None
    total = len(completion_values)
    return ZoneCoverage(
        cells=total,
        below=total - covered,
        above=overpainted,
        completion_percent=round(sum(completion_values) / total * 100.0, 1),
        coverage_percent=round(covered / total * 100.0, 1),
        overpaint_percent=round(overpainted / total * 100.0, 1),
        mean_um=float(np.mean(actual_values)),
    )


class CoatingProgressEngine:
    """Merge TSB absolute cumulative crops and judge the official Zone against the target.

    Thickness is corrected and graded by `coating_assessment` (PR #8 rules); the
    advance gate is 95% of the official Zone area meeting the target and a painted
    mean that is not below the target (PR #8 grade other than ``below``). Unlike the
    PR #8 engine, pixels no crop has touched count as unpainted (not unknown) and a
    missing target is judged against 160 µm and reported as `default_target`.

    ``cur_sequence_heatmap`` is a crop of the area touched by the current
    sequence, but every encoded pixel is the surface's absolute cumulative
    thickness at that time. Therefore a later crop overwrites the same atlas
    pixels; it must never be added to the preceding crop. State is isolated per
    training session. Blasting is excluded because its blue channel represents a
    different physical quantity than spray coating thickness.
    """

    def __init__(
        self,
        reference_root: Path,
        *,
        zone_reference_root: Path | None = None,
    ) -> None:
        self._root = Path(reference_root)
        try:
            self._zone_catalog = (
                ZoneReferenceCatalog(Path(zone_reference_root))
                if zone_reference_root is not None
                else ZoneReferenceCatalog.default()
            )
        except (OSError, KeyError, TypeError, ValueError):
            self._zone_catalog = None
        self._profiles: dict[tuple[str, int], tuple[tuple[int, int], dict[str, Any]]] = {}
        self._sessions: OrderedDict[
            tuple[str, str, int], dict[int, np.ndarray]
        ] = OrderedDict()
        self._applied: dict[tuple[str, str, int], set[str]] = {}
        # 세션이 마지막으로 판정받은 Zone. 다른 Zone으로 넘어간 시퀀스에서 떠난 Zone의 등급을 보려고 둔다.
        self._last_zone: dict[tuple[str, str, int], str] = {}
        self._visited: dict[tuple[str, str, int], set[str]] = {}  # 세션에서 칠한 구역(다음 구역 추천에서 뺀다)
        self._results: dict[tuple[str, str, int, str, float, float], CoatingProgress] = {}
        self._guard = threading.RLock()

    @property
    def reference_root(self) -> Path:
        return self._root

    @staticmethod
    def _unavailable(zone_id: str, reason: str) -> CoatingProgress:
        return CoatingProgress(
            zone_id=zone_id,
            available=False,
            status=None,
            recommendation=None,
            completion_percent=None,
            coverage_percent=None,
            overpaint_percent=None,
            next_zone_id=None,
            message=None,
            reason=reason,
        )

    def _load_profile(self, process: str, mode_id: int) -> tuple[dict[str, Any] | None, str | None]:
        path = (
            self._root
            / "generated"
            / "profiles"
            / process
            / "learning"
            / f"mode_{mode_id:02d}.json"
        )
        key = (process, mode_id)
        cached = self._profiles.get(key)
        try:
            stat = path.stat()
            stamp = (stat.st_mtime_ns, stat.st_size)
            if cached is not None and cached[0] == stamp:
                return cached[1], None
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload["aggregate"]["zones"], dict):
                raise ValueError("aggregate.zones must be an object")
            self._profiles[key] = (stamp, payload)
            return payload, None
        except FileNotFoundError:
            return (cached[1], None) if cached is not None else (None, "reference_unavailable")
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return (cached[1], None) if cached is not None else (None, "reference_invalid")

    @staticmethod
    def _zone(profile: dict[str, Any], zone_id: str) -> dict[str, Any] | None:
        zones = profile["aggregate"]["zones"]
        direct = zones.get(zone_id)
        if isinstance(direct, dict):
            return direct
        matches = [zone for zone in zones.values() if str(zone.get("zone_id")) == zone_id]
        return matches[0] if len(matches) == 1 else None

    @staticmethod
    def _scale(format_value: Any) -> float | None:
        if not isinstance(format_value, str):
            return None
        fields = {}
        for field in format_value.split(";"):
            key, separator, value = field.partition("=")
            if separator:
                fields[key.strip().lower()] = value.strip()
        if fields.get("thickness") != "RG16" or fields.get("alpha") != "B8":
            return None
        try:
            value = float(fields["scale_um"])
        except (KeyError, ValueError):
            return None
        return value if math.isfinite(value) and value > 0 else None

    @classmethod
    def _decode_crop(
        cls,
        item: dict[str, Any],
        *,
        atlas_width: int,
        atlas_height: int,
    ) -> tuple[_DecodedCrop | None, str | None]:
        # PR #8 복호(`numerical.decode_height_png`)의 main 쪽 사본이다. 다른 점은 요청의 `scale_um`을 쓰는 것과
        # 최근접 배치다(`coating_assessment` 독스트링 6번). 값이 같음은
        # `test_live_crop_path_matches_the_engine_cumulative_crop_path`가 지킨다.
        if item.get("encoding") != "png;base64":
            return None, "heatmap_encoding_unsupported"
        scale_um = cls._scale(item.get("format"))
        if scale_um is None:
            return None, "heatmap_format_unsupported"
        try:
            x0 = float(item["x0"])
            y0 = float(item["y0"])
            width = float(item["width"])
            height = float(item["height"])
            img_width = int(item["img_width"])
            img_height = int(item["img_height"])
            if (not all(math.isfinite(value) and value.is_integer() for value in (x0, y0, width, height))
                    or not (0 <= x0 < atlas_width and 0 <= y0 < atlas_height)
                    or min(width, height) <= 0
                    or min(img_width, img_height) <= 0):
                return None, "heatmap_dimensions_invalid"
            encoded = item["data"]
            if not isinstance(encoded, str):
                raise ValueError
            raw = base64.b64decode(encoded, validate=True)
            image = Image.open(io.BytesIO(raw)).convert("RGB")
            if image.size != (img_width, img_height):
                return None, "heatmap_dimensions_invalid"
            # Bound allocation to the atlas intersection, including legacy crops
            # whose declared rectangle extends beyond its right/top edge.
            footprint = (min(int(width), atlas_width - int(x0)), min(int(height), atlas_height - int(y0)))
            if image.size != footprint or footprint != (width, height):
                # width/height describe atlas coverage, not PNG sample count.
                # Resize packed channels together with NEAREST: interpolation
                # would fabricate RG16 thickness and B8 validity at edges.
                source_box = (0.0, img_height * (1.0 - footprint[1] / height),
                              img_width * footprint[0] / width, float(img_height))
                image = image.resize(footprint, Image.Resampling.NEAREST, box=source_box)
            img_width, img_height = footprint
            width, height = float(img_width), float(img_height)
        except (KeyError, TypeError, ValueError, binascii.Error, UnidentifiedImageError, OSError):
            return None, "heatmap_invalid"

        pixels = np.asarray(image, dtype=np.uint16)
        thickness = (
            pixels[:, :, 0].astype(np.float32) / 255.0
            + pixels[:, :, 1].astype(np.float32) / 65025.0
        ) * np.float32(scale_um)
        # TSB's B8 channel is the validity/alpha channel for the packed RG16
        # thickness. Black pixels outside the painted crop must not erase an
        # earlier absolute observation.
        valid = pixels[:, :, 2] > 0
        x_indices = np.floor(
            x0 + (np.arange(img_width, dtype=np.float64) + 0.5) * width / img_width
        ).astype(np.int64)
        # PNG rows are top-down while the atlas uses a bottom-left origin.
        y_indices = np.floor(
            y0
            + (np.arange(img_height - 1, -1, -1, dtype=np.float64) + 0.5)
            * height
            / img_height
        ).astype(np.int64)
        valid &= (x_indices[np.newaxis, :] >= 0) & (x_indices[np.newaxis, :] < atlas_width)
        valid &= (y_indices[:, np.newaxis] >= 0) & (y_indices[:, np.newaxis] < atlas_height)
        return _DecodedCrop(x_indices, y_indices, thickness, valid), None

    def _atlas_zone_mask(
        self,
        coating: dict[str, Any],
        *,
        scenario_id: str,
        map_num: int,
        zone_id: str,
        atlas_width: int,
        atlas_height: int,
        require_official: bool = False,
    ) -> tuple[np.ndarray | None, str | None]:
        """TSB 라벨맵 기준 프로필이면 그 구역 마스크(좌하단 원점). 옛 프로필은 마스크 없이 (None, None)."""
        if not require_official and coating.get("mask_source") != "tsb_color_label_map":
            return None, None
        model_key = (
            self._zone_catalog.resolve_model(scenario_id)
            if self._zone_catalog is not None
            else None
        )
        mask_image = (
            self._zone_catalog.zone_mask(
                model_key,
                map_num,
                zone_id,
                size=(atlas_width, atlas_height),
            )
            if self._zone_catalog is not None and model_key is not None
            else None
        )
        if mask_image is None:
            return None, "zone_reference_unavailable"
        # Runtime atlas rows use bottom-left coordinates; vendor label
        # maps and height-map PNG files store the top row as v=1.
        return np.flipud(np.asarray(mask_image, dtype=np.uint8) > 0), None

    @staticmethod
    def _map_dimensions(profile: dict[str, Any]) -> dict[int, tuple[int, int]]:
        dimensions: dict[int, tuple[int, int]] = {}
        for zone in profile["aggregate"]["zones"].values():
            if not isinstance(zone, dict) or not isinstance(zone.get("coating"), dict):
                continue
            coating = zone["coating"]
            try:
                map_num = int(zone.get("map_num", 1))
                width, height = int(coating["atlas_width"]), int(coating["atlas_height"])
            except (KeyError, TypeError, ValueError):
                continue
            if 0 < width <= 4096 and 0 < height <= 4096:
                dimensions[map_num] = (width, height)
        return dimensions

    @staticmethod
    def _merge_crop(
        surfaces: dict[int, np.ndarray], map_num: int, crop: _DecodedCrop,
        dimensions: tuple[int, int],
    ) -> None:
        # PR #8 누적 교체(`numerical.overwrite_cumulative_crops`)의 main 쪽 사본이다. 다른 점은 float32 저장과
        # B=0 화소가 앞선 값을 지우지 않는 것이다(`coating_assessment` 독스트링 4번). 값이 같음은
        # `test_live_crop_path_matches_the_engine_cumulative_crop_path`가 지킨다.
        rows, columns = np.nonzero(crop.valid)
        if not rows.size:
            return
        width, height = dimensions
        surface = surfaces.get(map_num)
        if surface is None or surface.shape != (height, width):
            surface = np.zeros((height, width), dtype=np.float32)
            surfaces[map_num] = surface
        surface[crop.y_indices[rows], crop.x_indices[columns]] = crop.thickness_um[rows, columns]

    def _remember_session(
        self, key: tuple[str, str, int], surfaces: dict[int, np.ndarray], applied: set[str],
    ) -> None:
        """Every surface insertion shares one bound, including unavailable replay results."""
        self._sessions[key] = surfaces
        self._sessions.move_to_end(key)
        self._applied[key] = applied
        while len(self._sessions) > _MAX_SESSIONS:
            expired, _ = self._sessions.popitem(last=False)
            self._applied.pop(expired, None)
            self._last_zone.pop(expired, None)
            self._visited.pop(expired, None)
            self._results = {item: value for item, value in self._results.items() if item[:3] != expired}

    def judge_zones(
        self,
        *,
        scenario_id: str,
        process: str,
        mode_id: int,
        surfaces: dict[int, np.ndarray],
        target_thickness_um: float | None = None,
        min_thickness_um: float | None = None,
        max_thickness_um: float | None = None,
        behavior: Mapping[str, Mapping[str, Sequence[float]]] | None = None,
    ) -> tuple[list[ZoneJudgment], str | None]:
        """세션 끝 두께가 있는 면의 공식 구역 전체를 목표 두께에 대고 판정한다.

        표면은 칠한 자리의 원시 두께이고 그 밖은 0이어야 한다. 목표가 없으면 기본 목표 160µm다.
        표면이 없는 면의 구역은 판정하지 않는다 — 높이맵이 한 면만 오면 다른 면을 칠했는지 모른다.
        전문가 프로필은 `expert` 참고 판정(보정 전 두께)에만 필요하다. 등급·게이트·평가 구역은 제한하지 않는다.
        `behavior`는 구역별 속도·거리 표본(`coating_assessment.behavior_by_zone`)이고 조절 후보에만 쓴다.
        """
        catalog = self._zone_catalog
        # 학습 모드로 정규화된 ID를 사용해 TR-L03 등 기존 허용 별칭도 같은 공식 Zone을 본다.
        model_key = catalog.resolve_model(str(mode_id)) if catalog is not None else None
        if catalog is None or model_key is None:
            return [], "zone_reference_unavailable"
        with self._guard:
            profile, _ = self._load_profile(process, mode_id)
        references: dict[tuple[int, str], tuple[int, int, int, dict[str, Any]]] = {}
        profile_zones = profile["aggregate"]["zones"] if profile is not None else {}
        for key, zone in profile_zones.items():
            if not isinstance(zone, dict):
                continue
            coating = zone.get("coating")
            if not isinstance(coating, dict) or coating.get("analysis_status") != "available":
                continue
            try:
                map_num = int(zone.get("map_num", 1))
                width, height = int(coating.get("atlas_width", 0)), int(coating.get("atlas_height", 0))
                grid = int(coating.get("grid_size", 0))
            except (TypeError, ValueError):
                continue
            cells = coating.get("cells")
            if 0 < width <= 4096 and 0 < height <= 4096 and grid > 0 and isinstance(cells, dict):
                zone_id = str(zone.get("zone_id") or str(key).partition(":")[2])
                references[(map_num, zone_id)] = (width, height, grid, cells)
        judgments: list[ZoneJudgment] = []
        for map_num, raw_surface in surfaces.items():
            for zone in catalog.zones(model_key, map_num):
                zone_id = zone.zone_id
                mask_image = catalog.zone_mask(model_key, map_num, zone_id)
                if mask_image is None:
                    continue
                width, height = mask_image.size
                zone_mask = np.flipud(np.asarray(mask_image, dtype=np.uint8) > 0)
                # 공식 라벨맵의 화소가 분모다. TSB 안내대로 해상도가 다르면 최근접으로 맞춘다.
                surface = (raw_surface if raw_surface.shape == (height, width) else np.asarray(
                    Image.fromarray(raw_surface.astype(np.float32)).resize((width, height), Image.Resampling.NEAREST)
                ))
                assessment = coating_assessment.assess_zone(
                    surface, zone_mask, target_thickness_um, min_thickness_um=min_thickness_um,
                    max_thickness_um=max_thickness_um, **(behavior or {}).get(zone_id, {}),
                )
                if assessment is None:
                    continue
                expert = None
                reference = references.get((map_num, zone_id))
                if reference is not None:
                    ref_width, ref_height, grid, cells = reference
                    expert_surface, expert_mask = surface, zone_mask
                    if surface.shape != (ref_height, ref_width):
                        expert_surface = np.asarray(Image.fromarray(surface.astype(np.float32)).resize(
                            (ref_width, ref_height), Image.Resampling.NEAREST))
                        expert_mask = np.asarray(Image.fromarray(zone_mask.astype(np.uint8)).resize(
                            (ref_width, ref_height), Image.Resampling.NEAREST)) > 0
                    expert = measure_zone(expert_surface, cells, atlas_width=ref_width, atlas_height=ref_height,
                                          grid_size=grid, zone_mask=expert_mask)
                judgments.append(ZoneJudgment(map_num, zone_id, assessment, expert))
        judgments.sort(
            key=lambda item: (item.map_num, int(item.zone_id) if item.zone_id.isdigit() else 1_000_000, item.zone_id)
        )
        return judgments, (None if judgments else "zone_reference_unavailable")

    def _left_zone_row(
        self, profile: dict[str, Any], session: dict[int, np.ndarray], zone_id: str,
        *, scenario_id: str, target_thickness_um: float | None,
        min_thickness_um: float | None = None, max_thickness_um: float | None = None,
    ) -> dict[str, Any] | None:
        """방금 떠난 Zone을 지금의 세션 표면에서 평가한 행. 못 재면 None — 알림용이라 본 판정을 막지 않는다."""
        try:
            zone = self._zone(profile, zone_id)
            coating = zone.get("coating") if zone is not None else None
            if zone is None or not isinstance(coating, dict):
                return None
            map_num = int(zone.get("map_num", 1))
            atlas_width, atlas_height = int(coating.get("atlas_width", 0)), int(coating.get("atlas_height", 0))
            surface = session.get(map_num)
            if surface is None or min(atlas_width, atlas_height) <= 0:
                return None
            zone_mask, mask_reason = self._atlas_zone_mask(
                coating, scenario_id=scenario_id, map_num=map_num, zone_id=zone_id,
                atlas_width=atlas_width, atlas_height=atlas_height, require_official=True,
            )
            if mask_reason is not None:
                return None
            # 조절 후보 표본은 주지 않는다(알림 본문에 싣지 않는다).
            assessment = coating_assessment.assess_zone(surface, zone_mask, target_thickness_um,
                min_thickness_um=min_thickness_um, max_thickness_um=max_thickness_um)
        except (TypeError, ValueError):
            return None
        return None if assessment is None else coating_assessment.zone_row(map_num, zone_id, assessment)

    @staticmethod
    def _next_zone(profile: dict[str, Any], map_num: int, zone_id: str, visited: Iterable[str] = ()) -> str | None:
        """전문가 순서에서 지금 구역 뒤의, 이번 세션에 아직 칠하지 않은 첫 구역(뒤에 없으면 앞에서 찾는다).

        이미 칠한 구역을 다음으로 권하면 안 된다(2026-11-24 현장 피드백). 모두 칠했으면 None.
        """
        done = {str(zone) for zone in visited} | {zone_id}
        for candidate in profile["aggregate"].get("macro_orders", []):
            order = candidate.get("order")
            if not isinstance(order, list):
                continue
            for index, item in enumerate(order):
                if int(item.get("map_num", 1)) == map_num and str(item.get("zone_id")) == zone_id:
                    later = order[index + 1:] + order[:index]
                    return next((str(step.get("zone_id")) for step in later if str(step.get("zone_id")) not in done), None)
        return None

    def evaluate_replay(
        self,
        request: AnalyzeSequenceRequest,
        prior_requests: Iterable[AnalyzeSequenceRequest],
        *,
        target_thickness_um: float | None = None,
        min_thickness_um: float | None = None,
        max_thickness_um: float | None = None,
    ) -> CoatingProgress:
        """Restore missing crops, including telemetry-only records, without reapplying older pixels."""
        with self._guard:
            from trainer.demo_api.sequence_store import sequence_order_key

            key = (request.session_id, request.process_type, request.resolved_mode_id)
            try:
                lower, upper = coating_assessment.resolve_band(target_thickness_um,
                    min_thickness_um=min_thickness_um, max_thickness_um=max_thickness_um)
            except (TypeError, ValueError):
                return self._unavailable(request.zone_id, "target_thickness_invalid")
            cached = self._results.get((*key, request.sequence_id, lower, upper))
            if cached is not None:
                return cached.model_copy(deep=True)
            if request.process_type == "spray" and request.resolved_mode_id is not None:
                profile, _ = self._load_profile("spray", request.resolved_mode_id)
                if profile is not None:
                    prior = sorted((previous for previous in prior_requests
                        if (previous.session_id, previous.process_type, previous.resolved_mode_id) == key
                        and sequence_order_key(previous.sequence_id) < sequence_order_key(request.sequence_id)),
                        key=lambda previous: sequence_order_key(previous.sequence_id))
                    self._visited[key] = {str(previous.zone_id) for previous in prior if str(previous.zone_id) not in {"-1", "0"}}
                    applied = self._applied.get(key, set())
                    if any(previous.sequence_id not in applied for previous in prior):
                        # Rebuild from persisted absolute snapshots, so a late-restored old
                        # crop cannot overwrite newer thickness in a warm engine.
                        surfaces: dict[int, np.ndarray] = {}
                        applied = set()
                        dimensions = self._map_dimensions(profile)
                        for previous in prior:
                            if previous.sequence_id in applied:
                                continue
                            result = previous.sequence_result
                            payload = result.model_dump(exclude_none=True) if hasattr(result, "model_dump") else result
                            for item in payload.get("cur_sequence_heatmap") or []:
                                map_num = int(item.get("map_num", 1))
                                if map_num not in dimensions:
                                    continue
                                width, height = dimensions[map_num]
                                if not (0 < width <= 4096 and 0 < height <= 4096):
                                    continue
                                crop, _ = self._decode_crop(item, atlas_width=width, atlas_height=height)
                                if crop is None:
                                    continue
                                self._merge_crop(surfaces, map_num, crop, (width, height))
                            applied.add(previous.sequence_id)
                        if surfaces:
                            self._remember_session(key, surfaces, applied)
            return self.evaluate(request, target_thickness_um=target_thickness_um,
                min_thickness_um=min_thickness_um, max_thickness_um=max_thickness_um)

    def evaluate(
        self, request: AnalyzeSequenceRequest, *, target_thickness_um: float | None = None,
        min_thickness_um: float | None = None, max_thickness_um: float | None = None,
    ) -> CoatingProgress:
        with self._guard:
            process = request.process_type
            mode_id = request.resolved_mode_id
            if process == "blasting":
                return self._unavailable(request.zone_id, "coating_not_applicable")
            if process != "spray" or mode_id is None:
                return self._unavailable(request.zone_id, "reference_context_unavailable")
            if request.zone_id in {"-1", "0"}:
                return self._unavailable(request.zone_id, "zone_unavailable")

            requested = target_thickness_um
            target_thickness_um, target_basis = coating_assessment.resolve_target(requested)
            try:
                target_thickness_um, upper = coating_assessment.resolve_band(requested,
                    min_thickness_um=min_thickness_um, max_thickness_um=max_thickness_um)
            except (TypeError, ValueError):
                return self._unavailable(request.zone_id, "target_thickness_invalid")
            if min_thickness_um is not None or max_thickness_um is not None:
                target_basis = "session_target"
            if not math.isfinite(target_thickness_um) or target_thickness_um <= 0:
                return self._unavailable(request.zone_id, "target_thickness_invalid")
            result_key = (request.session_id, process, mode_id, request.sequence_id, target_thickness_um, upper)
            cached = self._results.get(result_key)
            if cached is not None:
                return cached.model_copy(deep=True)

            profile, reason = self._load_profile(process, mode_id)
            if profile is None:
                return self._unavailable(request.zone_id, reason or "reference_unavailable")
            zone = self._zone(profile, request.zone_id)
            if zone is None:
                return self._unavailable(request.zone_id, "reference_unavailable")
            coating = zone.get("coating")
            if not isinstance(coating, dict) or coating.get("analysis_status") != "available":
                return self._unavailable(request.zone_id, "coating_reference_unavailable")

            sequence_result = request.sequence_result
            result_payload = (
                sequence_result.model_dump(exclude_none=True)
                if hasattr(sequence_result, "model_dump")
                else sequence_result
            )
            heatmaps = result_payload.get("cur_sequence_heatmap")
            if not isinstance(heatmaps, list) or not heatmaps:
                return self._unavailable(request.zone_id, "heatmap_unavailable")
            map_num = int(zone.get("map_num", 1))
            atlas_width = int(coating.get("atlas_width", 0))
            atlas_height = int(coating.get("atlas_height", 0))
            grid_size = int(coating.get("grid_size", 0))
            target_cells = coating.get("cells")
            if min(atlas_width, atlas_height, grid_size) <= 0 or not isinstance(target_cells, dict):
                return self._unavailable(request.zone_id, "coating_reference_invalid")
            zone_mask, mask_reason = self._atlas_zone_mask(
                coating,
                scenario_id=request.scenario_id,
                map_num=map_num,
                zone_id=request.zone_id,
                atlas_width=atlas_width,
                atlas_height=atlas_height,
                require_official=True,
            )
            if mask_reason is not None:
                return self._unavailable(request.zone_id, mask_reason)

            dimensions = self._map_dimensions(profile)
            if map_num not in dimensions:
                return self._unavailable(request.zone_id, "coating_reference_invalid")
            decoded: list[tuple[int, _DecodedCrop]] = []
            matching = 0
            for item in heatmaps:
                if not isinstance(item, dict):
                    continue
                item_map = int(item.get("map_num", 1))
                if item_map not in dimensions:
                    continue
                if item_map == map_num:
                    matching += 1
                width, height = dimensions[item_map]
                crop, decode_reason = self._decode_crop(
                    item, atlas_width=width, atlas_height=height,
                )
                if crop is None:
                    if item_map == map_num:
                        return self._unavailable(request.zone_id, decode_reason or "heatmap_invalid")
                    continue
                decoded.append((item_map, crop))
            if matching == 0:
                return self._unavailable(request.zone_id, "heatmap_map_unavailable")

            session_key = (request.session_id, process, mode_id)
            session = self._sessions.get(session_key, {})
            for item_map, crop in decoded:
                self._merge_crop(session, item_map, crop, dimensions[item_map])
            surface = session.get(map_num)
            if surface is None:
                surface = np.zeros((atlas_height, atlas_width), dtype=np.float32)
                session[map_num] = surface
            applied = self._applied.get(session_key, set())
            applied.add(request.sequence_id)
            self._remember_session(session_key, session, applied)

            # 조절 후보의 표본은 지금 시퀀스 한 건의 평균 속도·거리뿐이다(리포트는 그 Zone의 저장된 시퀀스 전부).
            behavior = coating_assessment.behavior_by_zone(
                [{"zone_id": request.zone_id, "sequence_result": result_payload}])
            assessment = coating_assessment.assess_zone(
                surface, zone_mask, requested, min_thickness_um=min_thickness_um,
                max_thickness_um=max_thickness_um, **behavior.get(request.zone_id, {}))
            if assessment is None:
                return self._unavailable(request.zone_id, "coating_reference_invalid")

            # 전문가 칸 비교는 보정 전 표면에 대는 참고 값이다. 진행 판정에는 쓰지 않는다.
            expert_coverage = measure_zone(
                surface, target_cells, atlas_width=atlas_width, atlas_height=atlas_height,
                grid_size=grid_size, zone_mask=zone_mask,
            )

            coverage_percent = assessment["coverage_percent"]
            status = assessment["gate_status"]
            row = coating_assessment.zone_row(map_num, request.zone_id, assessment)
            overpaint_action = "잠시 분사를 멈추고 도막 분포를 확인한 뒤 부족한 구간만 보완하세요."
            if status == "overpaint_risk":
                recommendation = "stop_current_zone"
                next_zone_id = None
                # 같은 시퀀스의 과도막 알림(SSE)과 숫자가 같도록 행의 값을 적는다.
                message = (f"{request.zone_id}구역 칠한 곳 평균 {row['thickness_um']:.0f}µm(상한 초과·과도막), "
                           f"{target_thickness_um:g}µm 이상 면적 {coverage_percent:.1f}%. {overpaint_action}")
            elif status == "complete":
                recommendation = "move_next_zone"
                next_zone_id = self._next_zone(profile, map_num, request.zone_id, self._visited.get(session_key, ()))
                message = f"{request.zone_id}구역 공식 면적 95% 이상 충족."
                if assessment["grade"] == "above":
                    # 과도막도 완료다. 더 칠하면 더 두꺼워질 뿐이라 다음 구역으로 보낸다(2026-11-24 사용자 결정).
                    message = (f"{request.zone_id}구역 공식 면적 95% 이상 충족, 칠한 곳 평균 {row['thickness_um']:.0f}µm(과도막). "
                               "더 칠하지 마세요.")
                if next_zone_id is not None:
                    message += f" {next_zone_id}구역으로 이동하세요."
            elif coverage_percent >= coating_assessment.GATE_COVERAGE_PERCENT:
                # 면적은 채웠지만 저도막이라 게이트가 막았다. 떠날 때의 저도막 알림과 같은 숫자를 적는다.
                recommendation = "continue_current_zone"
                next_zone_id = None
                message = (f"{request.zone_id}구역 충족 면적 {coverage_percent:.1f}%, "
                           f"칠한 곳 평균 {coating_assessment.mean_text(row['thickness_um'], target_thickness_um)}µm(하한 미만·저도막). "
                           "도막 분포 확인 후 보완하세요.")
            else:
                recommendation = "continue_current_zone"
                next_zone_id = None
                message = (f"{request.zone_id}구역 {target_thickness_um:g}µm 이상 면적 {coverage_percent:.1f}%. "
                           "도막 분포 확인 후 보완하세요.")
                if assessment["overpaint_percent"] > 0:
                    message = (f"{request.zone_id}구역 {target_thickness_um:g}µm 이상 면적 {coverage_percent:.1f}%, "
                               f"과도막 면적 {assessment['overpaint_percent']:.1f}%. {overpaint_action}")

            message = f"정상범위 {target_thickness_um:g}~{upper:g}µm: {message}"

            previous = self._last_zone.get(session_key)
            if previous is not None and previous != request.zone_id:
                left = self._left_zone_row(
                    profile, session, previous, scenario_id=request.scenario_id, target_thickness_um=requested,
                    min_thickness_um=min_thickness_um, max_thickness_um=max_thickness_um)
                if left is not None:
                    row["left_zone"] = left
            self._last_zone[session_key] = request.zone_id
            self._visited.setdefault(session_key, set()).add(str(request.zone_id))

            result = CoatingProgress(
                comparison_basis="session_target",
                target_thickness_um=target_thickness_um,
                target_basis=target_basis,
                assessment=row,
                expert_completion_percent=expert_coverage.completion_percent if expert_coverage is not None else None,
                zone_id=request.zone_id,
                available=True,
                status=status,
                recommendation=recommendation,
                completion_percent=assessment["completion_percent"],
                coverage_percent=coverage_percent,
                normal_band_percent=assessment["normal_band_percent"],
                overpaint_percent=assessment["overpaint_percent"],
                next_zone_id=next_zone_id,
                message=message,
                reason=None,
            )
            self._results[result_key] = result
            while len(self._results) > 8192:
                self._results.pop(next(iter(self._results)))
            return result.model_copy(deep=True)
