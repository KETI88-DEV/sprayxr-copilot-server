"""Decode saved sequence heatmaps without trusting their transport or geometry.

TSB v2 stores physical thickness in RG and observation validity in B. Crop
coordinates describe a 2048-square atlas with a bottom-left origin, while PNG
rows and official masks have a top-left origin. Coverage here describes only
sampled target pixels in this crop, never the area of an entire official zone.
"""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
import io
import math
import re
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

if __package__:
    from .training_data_coating import PREVIEW_BACKGROUND, PREVIEW_COLORS, TARGET_TOLERANCE_PERCENT, _histogram, _summary
else:
    from training_data_coating import PREVIEW_BACKGROUND, PREVIEW_COLORS, TARGET_TOLERANCE_PERCENT, _histogram, _summary


ATLAS_SIZE = 2048
MAX_IMAGE_DIMENSION = 2048
MAX_IMAGE_PIXELS = 2048 * 2048
MAX_ENCODED_BYTES = 16 * 1024 * 1024
MAX_DECODED_BYTES = 12 * 1024 * 1024
# RG <= (1 + 1/255)*scale. Bound the largest possible sum of squared
# deviations before calling the shared population-standard-deviation helper.
_MAX_NUMERIC_SCALE = math.sqrt(np.finfo(np.float64).max / MAX_IMAGE_PIXELS) / 4
_FORMAT = re.compile(
    r"v2;thickness=RG16;alpha=B8;scale_um="
    r"([+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?)"
)


@dataclass(frozen=True)
class PreparedHeatmap:
    """JSON-safe analytics and original/derived PNGs for separate HTTP routes."""

    payload: dict[str, Any]
    original_png: bytes | None
    thickness_png: bytes | None


def _finite_positive(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _integer(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return int(number) if math.isfinite(number) and number.is_integer() else None


def _physical_summary(
    values: np.ndarray | None, *, target_pixels: int | None,
    target_thickness_um: float | None,
) -> dict[str, Any]:
    # Reuse the shared statistics; avoid target*10 overflowing before /100.
    result = _summary(values, target_pixels=target_pixels, target_thickness_um=None)
    result["target_thickness_um"] = target_thickness_um
    if target_thickness_um is not None:
        result["target_tolerance_percent"] = TARGET_TOLERANCE_PERCENT
        if values is not None and values.size:
            tolerance = target_thickness_um * (TARGET_TOLERANCE_PERCENT / 100.0)
            within = (values >= target_thickness_um - tolerance) & (values <= target_thickness_um + tolerance)
            result["within_target_percent"] = float(np.count_nonzero(within) / values.size * 100.0)
    return result


def _initial_payload(target_thickness_um: Any) -> dict[str, Any]:
    target = _finite_positive(target_thickness_um)
    warnings = []
    if target_thickness_um is not None and target is None:
        warnings.append("invalid_target_thickness_ignored")
    return {
        "status": "unavailable",
        "reason": None,
        "summary": _physical_summary(None, target_pixels=None, target_thickness_um=target),
        "histogram": [],
        "zones": [],
        "statistics_basis": "painted_pixels",
        "coverage_domain": "unknown",
        "distribution_domain": "unknown",
        "coverage_scope": "sampled_target_pixels_within_crop",
        "validity_convention": None,
        "scale_um": None,
        "preview_max_um": None,
        "preview_colors": list(PREVIEW_COLORS),
        "preview_cap_basis": "painted_pixels_p99",
        "preview_cap_affects_statistics": False,
        "image_width": None,
        "image_height": None,
        "warnings": warnings,
    }


def _decode_png(item: dict[str, Any]) -> tuple[bytes | None, tuple[int, int] | None, str | None]:
    """Check transport, PNG integrity, and header size before any conversion."""
    if item.get("encoding") != "png;base64":
        return None, None, "unsupported_heatmap_encoding"
    encoded = item.get("data")
    if not isinstance(encoded, str) or not encoded:
        return None, None, "missing_heatmap_data"
    if len(encoded) > MAX_ENCODED_BYTES:
        return None, None, "encoded_heatmap_exceeds_limit"
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        return None, None, "invalid_heatmap_base64"
    if len(raw) > MAX_DECODED_BYTES:
        return None, None, "decoded_heatmap_exceeds_limit"
    if not raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return None, None, "invalid_heatmap_png"
    try:
        with Image.open(io.BytesIO(raw)) as image:
            if image.format != "PNG":
                return None, None, "invalid_heatmap_png"
            width, height = image.size
            if (
                width < 1 or height < 1
                or width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION
                or width * height > MAX_IMAGE_PIXELS
            ):
                return None, None, "image_dimensions_exceed_limit"
            image.verify()
        # PNG verify checks chunk CRCs but does not decompress IDAT. Even raw
        # previews must be decodable images; dimensions were bounded above.
        with Image.open(io.BytesIO(raw)) as image:
            image.load()
    except (OSError, ValueError, SyntaxError, UnidentifiedImageError, Image.DecompressionBombError):
        return None, None, "invalid_heatmap_png"
    return raw, (width, height), None


def _physical_scale(format_value: Any) -> float | None:
    if not isinstance(format_value, str):
        return None
    match = _FORMAT.fullmatch(format_value)
    return _finite_positive(match.group(1)) if match else None


def _crop_indices(
    item: dict[str, Any], image_size: tuple[int, int],
) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
    image_width, image_height = image_size
    if (_integer(item.get("img_width")), _integer(item.get("img_height"))) != image_size:
        return None, None, "image_dimensions_mismatch"
    try:
        geometry = tuple(float(item[key]) for key in ("x0", "y0", "width", "height"))
    except (KeyError, TypeError, ValueError, OverflowError):
        return None, None, "invalid_crop_geometry"
    x0, y0, width, height = geometry
    if (
        not all(math.isfinite(value) for value in geometry)
        or x0 < 0 or y0 < 0 or width <= 0 or height <= 0
        or x0 + width > ATLAS_SIZE or y0 + height > ATLAS_SIZE
    ):
        return None, None, "invalid_crop_geometry"
    columns = np.floor(x0 + (np.arange(image_width) + 0.5) * width / image_width).astype(np.intp)
    bottom_y = np.floor(
        y0 + (image_height - np.arange(image_height) - 0.5) * height / image_height
    ).astype(np.intp)
    rows = ATLAS_SIZE - 1 - bottom_y
    if np.any(columns < 0) or np.any(columns >= ATLAS_SIZE) or np.any(rows < 0) or np.any(rows >= ATLAS_SIZE):
        return None, None, "invalid_crop_geometry"
    return rows, columns, None


def _default_catalog() -> Any:
    try:
        from trainer.zone_reference import ZoneReferenceCatalog
    except ImportError:
        from zone_reference import ZoneReferenceCatalog
    return ZoneReferenceCatalog.default()


def _project_official_masks(
    item: dict[str, Any], *, scenario_id: str, catalog: Any,
    rows: np.ndarray, columns: np.ndarray, thickness: np.ndarray,
    painted: np.ndarray, payload: dict[str, Any],
) -> tuple[np.ndarray | None, str | None]:
    """Return a complete projected union, or None when coverage is unknown."""
    target = payload["summary"]["target_thickness_um"]
    warnings = payload["warnings"]
    map_num = _integer(item.get("map_num"))
    if map_num is None or map_num < 0:
        return None, "invalid_map_number"
    try:
        if catalog is None:
            catalog = _default_catalog()
        model = catalog.resolve_model(str(scenario_id))
        if model is None:
            return None, "scenario_model_unavailable"
        definitions = tuple(catalog.zones(model, map_num))
    except (ImportError, OSError, ValueError, KeyError, TypeError, AttributeError):
        return None, "official_zone_catalog_unavailable"
    if not definitions:
        return None, "official_zones_unavailable"
    domain = np.zeros(painted.shape, dtype=bool)
    complete = True
    for definition in definitions:
        zone_id = str(definition.zone_id)
        try:
            mask_image = catalog.zone_mask(model, map_num, zone_id, size=(ATLAS_SIZE, ATLAS_SIZE))
            if mask_image is None or mask_image.size != (ATLAS_SIZE, ATLAS_SIZE):
                raise ValueError("Official atlas mask is unavailable or has an unexpected size")
            atlas_mask = np.asarray(mask_image.convert("L"), dtype=np.uint8)
            mask = atlas_mask[np.ix_(rows, columns)] > 0
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            complete = False
            warnings.append(f"official_zone_mask_unavailable: zone {zone_id}")
            continue
        target_pixels = int(np.count_nonzero(mask))
        # An unrelated official zone is not a zero-coverage zone in this crop.
        if not target_pixels:
            continue
        domain |= mask
        payload["zones"].append({
            "zone_id": zone_id,
            "hex_color": definition.hex_color,
            "coverage_scope": "sampled_target_pixels_within_crop",
            "summary": _physical_summary(
                thickness[painted & mask], target_pixels=target_pixels,
                target_thickness_um=target,
            ),
        })
    return (domain, None) if complete else (None, "official_zone_mask_unavailable")


def _render_preview(thickness: np.ndarray, painted: np.ndarray, vmax_um: float | None) -> bytes:
    """Color only valid observations; preview saturation never changes values."""
    output = np.empty((*painted.shape, 3), dtype=np.uint8)
    output[:] = PREVIEW_BACKGROUND
    if np.any(painted):
        colors = np.asarray([
            tuple(int(color[index:index + 2], 16) for index in (1, 3, 5))
            for color in PREVIEW_COLORS
        ], dtype=np.float64)
        levels = np.linspace(0, 1, len(colors))
        # A valid all-zero distribution has p99=0. Use the low ramp stop while
        # retaining the actual zero p99 in the payload.
        normalized = np.clip(thickness[painted] / vmax_um, 0, 1) if vmax_um else np.zeros(np.count_nonzero(painted))
        for channel in range(3):
            output[..., channel][painted] = np.rint(np.interp(normalized, levels, colors[:, channel])).astype(np.uint8)
    stream = io.BytesIO()
    Image.fromarray(output).save(stream, format="PNG")
    return stream.getvalue()


def prepare_heatmap(
    item: dict[str, Any], *, scenario_id: str, process: str = "spray",
    target_thickness_um: float | None = None, catalog: Any = None,
) -> PreparedHeatmap:
    """Prepare an untrusted DB heatmap for viewing and physical analysis.

    Only the exact ``v2;thickness=RG16;alpha=B8;scale_um=<positive number>``
    grammar enables physical decoding. B>0 is always authoritative, including
    an entirely zero B plane; no legacy RG fallback is applied. Unknown formats
    and blasting retain a verified raw PNG without physical statistics. Scales
    that exceed the float64 variance accumulator range retain raw preview only.

    Means, quantiles, population standard deviation, CV, and the 24-bin
    histogram use painted pixels inside the complete official crop mask union.
    When geometry or any official mask is unavailable, distribution statistics
    use painted PNG pixels and coverage is null. Known zone summaries describe
    only their target pixels sampled by this crop. A finite positive explicit
    target enables a +/-10% band; no nominal target is inferred.

    The derived PNG retains source dimensions/orientation. Its color cap is
    p99 of all valid PNG pixels and affects display only. No-paint numeric
    distributions and preview_max_um are null; the histogram is empty.
    """
    payload = _initial_payload(target_thickness_um)
    if not isinstance(item, dict):
        payload["reason"] = "invalid_heatmap_item"
        return PreparedHeatmap(payload, None, None)
    original, image_size, reason = _decode_png(item)
    if original is None:
        payload["reason"] = reason
        return PreparedHeatmap(payload, None, None)
    payload["image_width"], payload["image_height"] = image_size
    normalized_process = str(process).lower()
    if normalized_process != "spray":
        payload["reason"] = "physical_thickness_not_applicable" if normalized_process == "blasting" else "process_unknown"
        return PreparedHeatmap(payload, original, None)
    scale = _physical_scale(item.get("format"))
    if scale is None:
        payload["reason"] = "unsupported_heatmap_format"
        return PreparedHeatmap(payload, original, None)
    if scale > _MAX_NUMERIC_SCALE:
        payload["reason"] = "unsupported_thickness_scale"
        return PreparedHeatmap(payload, original, None)
    # Pillow exposes 16-bit truecolor PNGs as RGB after discarding low bytes.
    # The TSB RG16/B8 contract requires two separate 8-bit R/G channels.
    if (original[24], original[25]) not in ((8, 2), (8, 6)):
        payload["reason"] = "unsupported_physical_png_mode"
        return PreparedHeatmap(payload, original, None)
    try:
        with Image.open(io.BytesIO(original)) as image:
            if image.mode not in ("RGB", "RGBA"):
                payload["reason"] = "unsupported_physical_png_mode"
                return PreparedHeatmap(payload, original, None)
            pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
    except (OSError, ValueError, SyntaxError):
        payload["reason"] = "invalid_heatmap_png"
        return PreparedHeatmap(payload, None, None)
    with np.errstate(over="ignore", invalid="ignore"):
        thickness = (
            pixels[..., 0].astype(np.float64) / 255.0
            + pixels[..., 1].astype(np.float64) / 65025.0
        ) * scale
    if not np.all(np.isfinite(thickness)):
        payload["reason"] = "unsupported_thickness_scale"
        return PreparedHeatmap(payload, original, None)
    painted = pixels[..., 2] > 0
    payload["validity_convention"] = "tsb_b_channel"
    payload["scale_um"] = scale
    preview_values = thickness[painted]
    if preview_values.size:
        payload["preview_max_um"] = float(np.percentile(preview_values, 99))
    preview = _render_preview(thickness, painted, payload["preview_max_um"])
    rows, columns, reason = _crop_indices(item, image_size)
    domain = None
    if reason is None:
        domain, reason = _project_official_masks(
            item, scenario_id=scenario_id, catalog=catalog, rows=rows,
            columns=columns, thickness=thickness, painted=painted, payload=payload,
        )
    if domain is not None:
        selected = thickness[painted & domain]
        target_pixels = int(np.count_nonzero(domain))
        payload["status"] = "available"
        payload["coverage_domain"] = "official_zone_masks_in_crop"
        payload["distribution_domain"] = "official_zone_masks_in_crop"
        if not target_pixels:
            reason = "official_target_domain_empty_in_crop"
            payload["warnings"].append(reason)
    else:
        selected = thickness[painted]
        target_pixels = None
        payload["status"] = "partial"
        payload["distribution_domain"] = "painted_image_pixels"
        payload["warnings"].append(
            "coverage_unknown: complete official target masks in this crop unavailable; "
            "distribution uses painted PNG pixels"
        )
    payload["reason"] = reason
    payload["summary"] = _physical_summary(
        selected, target_pixels=target_pixels,
        target_thickness_um=payload["summary"]["target_thickness_um"],
    )
    payload["histogram"] = _histogram(selected)
    return PreparedHeatmap(payload, original, preview)
