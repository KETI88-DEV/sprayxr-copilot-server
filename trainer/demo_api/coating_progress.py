"""Stateful current-Zone coating progress from expert final height maps."""

from __future__ import annotations

import base64
import binascii
import io
import json
import math
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from trainer.demo_api.telemetry_contracts import AnalyzeSequenceRequest, CoatingProgress

_MAX_SESSIONS = 256


class CoatingProgressEngine:
    """Accumulate current-sequence crops and compare them with expert Zone targets.

    The input contract supplies one trigger ON→OFF heat map. State is therefore
    held per session and Zone. Blasting is deliberately excluded because its blue
    channel represents a different physical quantity than spray coating thickness.
    """

    def __init__(self, reference_root: Path) -> None:
        self._root = Path(reference_root)
        self._profiles: dict[tuple[str, int], tuple[tuple[int, int], dict[str, Any]]] = {}
        self._sessions: OrderedDict[
            tuple[str, str, int], dict[tuple[int, str], dict[str, float]]
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
    def _crop_cells(
        cls,
        item: dict[str, Any],
        *,
        atlas_width: int,
        atlas_height: int,
        grid_size: int,
    ) -> tuple[dict[str, float] | None, str | None]:
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

        cell_area = (atlas_width / grid_size) * (atlas_height / grid_size)
        pixel_area = (width / img_width) * (height / img_height)
        sums: dict[str, float] = {}
        for row in range(img_height):
            local_y = img_height - 1 - row
            gy = int((y0 + (local_y + 0.5) * height / img_height) / atlas_height * grid_size)
            if not 0 <= gy < grid_size:
                continue
            for column in range(img_width):
                # getdata is row-major; obtain the current row pixel directly.
                red, green, _blue = image.getpixel((column, row))
                thickness = (red / 255.0 + green / 65025.0) * scale_um
                if thickness <= 0:
                    continue
                gx = int((x0 + (column + 0.5) * width / img_width) / atlas_width * grid_size)
                if 0 <= gx < grid_size:
                    key = f"{gx}:{gy}"
                    sums[key] = sums.get(key, 0.0) + thickness * pixel_area / cell_area
        return sums, None

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

            heatmaps = request.sequence_result.get("cur_sequence_heatmap")
            if not isinstance(heatmaps, list) or not heatmaps:
                return self._unavailable(request.zone_id, "heatmap_unavailable")
            map_num = int(zone.get("map_num", 1))
            atlas_width = int(coating.get("atlas_width", 0))
            atlas_height = int(coating.get("atlas_height", 0))
            grid_size = int(coating.get("grid_size", 0))
            target_cells = coating.get("cells")
            if min(atlas_width, atlas_height, grid_size) <= 0 or not isinstance(target_cells, dict):
                return self._unavailable(request.zone_id, "coating_reference_invalid")

            delta: dict[str, float] = {}
            matching = 0
            for item in heatmaps:
                if not isinstance(item, dict) or int(item.get("map_num", 1)) != map_num:
                    continue
                matching += 1
                cells, decode_reason = self._crop_cells(
                    item,
                    atlas_width=atlas_width,
                    atlas_height=atlas_height,
                    grid_size=grid_size,
                )
                if cells is None:
                    return self._unavailable(request.zone_id, decode_reason or "heatmap_invalid")
                for key, value in cells.items():
                    delta[key] = delta.get(key, 0.0) + value
            if matching == 0:
                return self._unavailable(request.zone_id, "heatmap_map_unavailable")

            session_key = (request.session_id, process, mode_id)
            session = self._sessions.setdefault(session_key, {})
            self._sessions.move_to_end(session_key)
            while len(self._sessions) > _MAX_SESSIONS:
                expired, _ = self._sessions.popitem(last=False)
                self._results = {key: value for key, value in self._results.items() if key[:3] != expired}
            accumulated = session.setdefault((map_num, request.zone_id), {})
            for key, value in delta.items():
                accumulated[key] = accumulated.get(key, 0.0) + value

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
                actual = accumulated.get(key, 0.0)
                completion_values.append(min(actual / q1, 1.0))
                covered += int(actual >= q1)
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
