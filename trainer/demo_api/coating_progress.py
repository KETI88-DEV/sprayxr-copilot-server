"""Stateful current-Zone coating progress from expert final height maps."""

from __future__ import annotations

import base64
import binascii
import io
import json
import math
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from trainer.demo_api.telemetry_contracts import AnalyzeSequenceRequest, CoatingProgress

_MAX_SESSIONS = 16


@dataclass(frozen=True)
class _DecodedCrop:
    x_indices: np.ndarray
    y_indices: np.ndarray
    thickness_um: np.ndarray
    valid: np.ndarray


class CoatingProgressEngine:
    """Merge TSB absolute cumulative crops and compare them with expert targets.

    ``cur_sequence_heatmap`` is a crop of the area touched by the current
    sequence, but every encoded pixel is the surface's absolute cumulative
    thickness at that time. Therefore a later crop overwrites the same atlas
    pixels; it must never be added to the preceding crop. State is isolated per
    training session. Blasting is excluded because its blue channel represents a
    different physical quantity than spray coating thickness.
    """

    def __init__(self, reference_root: Path) -> None:
        self._root = Path(reference_root)
        self._profiles: dict[tuple[str, int], tuple[tuple[int, int], dict[str, Any]]] = {}
        self._sessions: OrderedDict[
            tuple[str, str, int], dict[int, np.ndarray]
        ] = OrderedDict()
        self._results: dict[tuple[str, str, int, str], CoatingProgress] = {}
        self._guard = threading.Lock()

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

    def evaluate(self, request: AnalyzeSequenceRequest) -> CoatingProgress:
        with self._guard:
            process = request.process_type
            mode_id = request.resolved_mode_id
            if process == "blasting":
                return self._unavailable(request.zone_id, "coating_not_applicable")
            if process != "spray" or mode_id is None:
                return self._unavailable(request.zone_id, "reference_context_unavailable")
            if request.zone_id in {"-1", "0"}:
                return self._unavailable(request.zone_id, "zone_unavailable")

            result_key = (request.session_id, process, mode_id, request.sequence_id)
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

            decoded: list[_DecodedCrop] = []
            matching = 0
            for item in heatmaps:
                if not isinstance(item, dict) or int(item.get("map_num", 1)) != map_num:
                    continue
                matching += 1
                crop, decode_reason = self._decode_crop(
                    item,
                    atlas_width=atlas_width,
                    atlas_height=atlas_height,
                )
                if crop is None:
                    return self._unavailable(request.zone_id, decode_reason or "heatmap_invalid")
                decoded.append(crop)
            if matching == 0:
                return self._unavailable(request.zone_id, "heatmap_map_unavailable")

            session_key = (request.session_id, process, mode_id)
            session = self._sessions.setdefault(session_key, {})
            self._sessions.move_to_end(session_key)
            while len(self._sessions) > _MAX_SESSIONS:
                expired, _ = self._sessions.popitem(last=False)
                self._results = {key: value for key, value in self._results.items() if key[:3] != expired}
            surface = session.get(map_num)
            if surface is None or surface.shape != (atlas_height, atlas_width):
                surface = np.zeros((atlas_height, atlas_width), dtype=np.float32)
                session[map_num] = surface
            for crop in decoded:
                rows, columns = np.nonzero(crop.valid)
                if rows.size:
                    surface[crop.y_indices[rows], crop.x_indices[columns]] = crop.thickness_um[
                        rows, columns
                    ]

            completion_values: list[float] = []
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
                actual = float(surface[bottom:top, left:right].mean())
                completion_values.append(min(actual / q1, 1.0))
                # RG16 at scale_um=20000 has about 0.31 um quantization. Allow
                # a 0.5 um boundary tolerance so an encoded 100 um target does
                # not become 99.96 um and incorrectly miss coverage.
                covered += int(actual >= q1 - 0.5)
                upper = q3 + max(q3 - q1, q3 * 0.1, 5.0)
                overpainted += int(actual > upper)
            if not completion_values:
                return self._unavailable(request.zone_id, "coating_reference_invalid")

            total = len(completion_values)
            completion_percent = round(sum(completion_values) / total * 100.0, 1)
            coverage_percent = round(covered / total * 100.0, 1)
            overpaint_percent = round(overpainted / total * 100.0, 1)
            if overpaint_percent >= 25.0:
                status = "overpaint_risk"
                recommendation = "stop_current_zone"
                next_zone_id = None
                message = f"Zone {request.zone_id}의 과도장 위험이 높습니다. 현재 Zone 분사를 중단하세요."
            elif completion_percent >= 90.0 and coverage_percent >= 85.0:
                status = "complete"
                recommendation = "move_next_zone"
                next_zone_id = self._next_zone(profile, map_num, request.zone_id)
                message = f"Zone {request.zone_id}의 전문가 기준 도막이 충족되었습니다."
                if next_zone_id is not None:
                    message += f" 다음 Zone {next_zone_id}로 이동하세요."
            else:
                status = "needs_more"
                recommendation = "continue_current_zone"
                next_zone_id = None
                message = f"Zone {request.zone_id}의 도막이 아직 부족합니다. 현재 Zone을 추가 도장하세요."

            result = CoatingProgress(
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
            return result.model_copy(deep=True)
