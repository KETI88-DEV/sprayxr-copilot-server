"""Current-Zone completion from cumulative crops and official Zone masks."""

from __future__ import annotations

import base64
import binascii
import io
import json
import math
import threading
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image, UnidentifiedImageError

from trainer.demo_api.telemetry_contracts import AnalyzeSequenceRequest, CoatingProgress
from trainer.zone_reference import ZoneReferenceCatalog

_MAX_SESSIONS = 16
# 구역 판정 기준. 시퀀스 안내(`evaluate`)와 세션 리포트(`judge_zones`)가 같은 값을 쓴다 — 갈리면 훈련 중
# "충족"이던 구역이 리포트에서 "부족"이 된다.
_OVERPAINT_RISK_PERCENT = 25.0
_COMPLETE_PERCENT = 90.0
_COVERAGE_PERCENT = 85.0
DEFAULT_TARGET_THICKNESS_UM = 160.0
_TARGET_COVERAGE_PERCENT = 95.0


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
    required_coverage_percent: float = _COVERAGE_PERCENT

    @property
    def status(self) -> Literal["overpaint_risk", "complete", "needs_more"]:
        if self.overpaint_percent >= _OVERPAINT_RISK_PERCENT:
            return "overpaint_risk"
        if self.completion_percent >= _COMPLETE_PERCENT and self.coverage_percent >= self.required_coverage_percent:
            return "complete"
        return "needs_more"


@dataclass(frozen=True)
class ZoneJudgment:
    map_num: int
    zone_id: str
    coverage: ZoneCoverage


def measure_zone(
    surface: np.ndarray,
    target_cells: dict[str, Any],
    *,
    atlas_width: int,
    atlas_height: int,
    grid_size: int,
    zone_mask: np.ndarray | None,
    target_thickness_um: float | None = None,
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
            q1 = float(target["q1_um"]) if target_thickness_um is None else target_thickness_um * .9
            q3 = float(target["q3_um"]) if target_thickness_um is None else target_thickness_um * 1.1
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
        upper = q3 if target_thickness_um is not None else q3 + max(q3 - q1, q3 * 0.1, 5.0)
        overpainted += int(actual > upper + .5)
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


def measure_target_zone(
    surface: np.ndarray, zone_mask: np.ndarray | None, *, target_thickness_um: float,
) -> ZoneCoverage | None:
    """Count every official Zone pixel, including unobserved pixels as unmet.

    Expert grid cells and crop averages are diagnostics, never completion proof.
    A sub-micrometre RG16 boundary tolerance is allowed at the target.
    """
    if zone_mask is None or zone_mask.shape != surface.shape or not np.any(zone_mask):
        return None
    values = surface[zone_mask]
    finite = np.isfinite(values)
    covered = int(np.count_nonzero(finite & (values >= target_thickness_um - 0.5)))
    overpainted = int(np.count_nonzero(finite & (values > target_thickness_um * 1.1 + 0.5)))
    total = int(values.size)
    return ZoneCoverage(
        cells=total, below=total - covered, above=overpainted,
        completion_percent=float(np.minimum(np.where(finite, values, 0) / target_thickness_um, 1).mean() * 100),
        coverage_percent=covered / total * 100, overpaint_percent=overpainted / total * 100,
        mean_um=float(np.where(finite, values, 0).mean()),
        required_coverage_percent=_TARGET_COVERAGE_PERCENT,
    )


class CoatingProgressEngine:
    """Merge TSB absolute cumulative crops and compare official pixels with the target.

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
        self._results: dict[tuple[str, str, int, str, float | None], CoatingProgress] = {}
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
            encoded = item["data"]
            if not isinstance(encoded, str):
                raise ValueError
            raw = base64.b64decode(encoded, validate=True)
            image = Image.open(io.BytesIO(raw)).convert("RGB")
            if image.size != (img_width, img_height):
                return None, "heatmap_dimensions_invalid"
            if min(width, height, img_width, img_height) <= 0:
                return None, "heatmap_dimensions_invalid"
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
            self._results = {item: value for item, value in self._results.items() if item[:3] != expired}

    def judge_zones(
        self,
        *,
        scenario_id: str,
        process: str,
        mode_id: int,
        surfaces: dict[int, np.ndarray],
        target_thickness_um: float | None = None,
    ) -> tuple[list[ZoneJudgment], str | None]:
        """세션 끝 두께(면 번호 → 좌하단 원점 표면)를 프로필의 모든 구역에 대고 판정한다.

        표면이 없는 면의 구역은 판정하지 않는다 — 높이맵이 한 면만 오면 다른 면을 칠했는지 모른다.
        """
        with self._guard:
            profile, reason = self._load_profile(process, mode_id)
        if profile is None:
            return [], reason or "reference_unavailable"
        judgments: list[ZoneJudgment] = []
        for key, zone in profile["aggregate"]["zones"].items():
            if not isinstance(zone, dict):
                continue
            coating = zone.get("coating")
            if not isinstance(coating, dict) or coating.get("analysis_status") != "available":
                continue
            map_num = int(zone.get("map_num", 1))
            zone_id = str(zone.get("zone_id") or str(key).partition(":")[2])
            atlas_width = int(coating.get("atlas_width", 0))
            atlas_height = int(coating.get("atlas_height", 0))
            grid_size = int(coating.get("grid_size", 0))
            target_cells = coating.get("cells")
            surface = surfaces.get(map_num)
            if surface is None or min(atlas_width, atlas_height, grid_size) <= 0 or not isinstance(target_cells, dict):
                continue
            if surface.shape != (atlas_height, atlas_width):
                # TSB 안내: 해상도가 다르면 최근접으로 맞춘다.
                surface = np.asarray(
                    Image.fromarray(surface.astype(np.float32)).resize(
                        (atlas_width, atlas_height), Image.Resampling.NEAREST
                    )
                )
            zone_mask, mask_reason = self._atlas_zone_mask(
                coating,
                scenario_id=scenario_id,
                map_num=map_num,
                zone_id=zone_id,
                atlas_width=atlas_width,
                atlas_height=atlas_height,
                require_official=target_thickness_um is not None,
            )
            if mask_reason is not None:
                continue
            coverage = measure_target_zone(
                surface, zone_mask, target_thickness_um=target_thickness_um,
            ) if target_thickness_um is not None else measure_zone(
                surface,
                target_cells,
                atlas_width=atlas_width,
                atlas_height=atlas_height,
                grid_size=grid_size,
                zone_mask=zone_mask,
                target_thickness_um=target_thickness_um,
            )
            if coverage is not None:
                judgments.append(ZoneJudgment(map_num, zone_id, coverage))
        judgments.sort(
            key=lambda item: (item.map_num, int(item.zone_id) if item.zone_id.isdigit() else 1_000_000, item.zone_id)
        )
        return judgments, (None if judgments else "coating_reference_invalid")

    @staticmethod
    def _next_zone(profile: dict[str, Any], map_num: int, zone_id: str) -> str | None:
        for candidate in profile["aggregate"].get("macro_orders", []):
            order = candidate.get("order")
            if not isinstance(order, list):
                continue
            for index, item in enumerate(order[:-1]):
                if int(item.get("map_num", 1)) == map_num and str(item.get("zone_id")) == zone_id:
                    return str(order[index + 1].get("zone_id"))
        return None

    def evaluate_replay(
        self,
        request: AnalyzeSequenceRequest,
        prior_requests: Iterable[AnalyzeSequenceRequest],
        *,
        target_thickness_um: float | None = None,
    ) -> CoatingProgress:
        """Restore missing crops, including telemetry-only records, without reapplying older pixels."""
        with self._guard:
            from trainer.demo_api.sequence_store import sequence_order_key

            key = (request.session_id, request.process_type, request.resolved_mode_id)
            effective_target = DEFAULT_TARGET_THICKNESS_UM if target_thickness_um is None else target_thickness_um
            cached = self._results.get((*key, request.sequence_id, effective_target))
            if cached is not None:
                return cached.model_copy(deep=True)
            if request.process_type == "spray" and request.resolved_mode_id is not None:
                profile, _ = self._load_profile("spray", request.resolved_mode_id)
                if profile is not None:
                    prior = sorted((previous for previous in prior_requests
                        if (previous.session_id, previous.process_type, previous.resolved_mode_id) == key
                        and sequence_order_key(previous.sequence_id) < sequence_order_key(request.sequence_id)),
                        key=lambda previous: sequence_order_key(previous.sequence_id))
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
            return self.evaluate(request, target_thickness_um=target_thickness_um)

    def evaluate(
        self, request: AnalyzeSequenceRequest, *, target_thickness_um: float | None = None,
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

            target_thickness_um = (DEFAULT_TARGET_THICKNESS_UM
                                   if target_thickness_um is None else target_thickness_um)
            if not math.isfinite(target_thickness_um) or target_thickness_um <= 0:
                return self._unavailable(request.zone_id, "target_thickness_invalid")
            result_key = (request.session_id, process, mode_id, request.sequence_id, target_thickness_um)
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

            coverage = measure_target_zone(
                surface, zone_mask, target_thickness_um=target_thickness_um,
            )
            if coverage is None:
                return self._unavailable(request.zone_id, "coating_reference_invalid")

            expert_coverage = measure_zone(
                surface, target_cells, atlas_width=atlas_width, atlas_height=atlas_height,
                grid_size=grid_size, zone_mask=zone_mask,
            ) if target_thickness_um is not None else coverage

            completion_percent = coverage.completion_percent
            coverage_percent = coverage.coverage_percent
            overpaint_percent = coverage.overpaint_percent
            status = coverage.status
            if status == "overpaint_risk":
                recommendation = "stop_current_zone"
                next_zone_id = None
                message = (f"Zone {request.zone_id}의 과도장 영역 분사를 중단하세요. "
                           "목표에 미달한 영역만 확인하고 이미 두꺼운 영역의 재도장을 피하세요.")
            elif status == "complete":
                recommendation = "move_next_zone"
                next_zone_id = self._next_zone(profile, map_num, request.zone_id)
                message = f"Zone {request.zone_id}의 공식 구역 면적 중 95% 이상이 목표 도막을 충족했습니다."
                if next_zone_id is not None:
                    message += f" 다음 Zone {next_zone_id}로 이동하세요."
            else:
                recommendation = "continue_current_zone"
                next_zone_id = None
                message = (f"Zone {request.zone_id}의 목표 도막 충족 면적은 {coverage_percent:.1f}%입니다. "
                           "현재 Zone의 목표 미달 영역만 보완하고 이미 두꺼운 영역은 피하세요.")

            if target_thickness_um is not None:
                message = f"세션 목표 {target_thickness_um:g}µm 기준: " + message.replace("전문가 기준 도막", "목표 도막")

            result = CoatingProgress(
                comparison_basis="session_target" if target_thickness_um is not None else "expert_reference",
                target_thickness_um=target_thickness_um,
                expert_completion_percent=expert_coverage.completion_percent if expert_coverage is not None else None,
                zone_id=request.zone_id,
                available=True,
                status=status,
                recommendation=recommendation,
                completion_percent=completion_percent,
                coverage_percent=coverage_percent,
                overpaint_percent=overpaint_percent,
                next_zone_id=next_zone_id,
                message=message,
                reason=None,
            )
            self._results[result_key] = result
            while len(self._results) > 8192:
                self._results.pop(next(iter(self._results)))
            return result.model_copy(deep=True)
