"""Deterministic physical thickness statistics for saved TSB height-map PNGs.

PNG rows and official label masks both retain their top-left origin. Coverage
uses the union of exact official zone masks; thickness distributions use coated
pixels within that domain. When the domain cannot be established completely,
global coverage is unknown and global distributions use coated image pixels.
No rectangle is substituted for an unavailable official mask.

Thickness is the PR #8 corrected thickness (raw - 10 um, capped); a pixel is
painted when its corrected thickness is at least 1 um.
"""

from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from trainer.demo_api import coating_assessment


HEIGHTMAP_SCALE_UM = 20000.0
HISTOGRAM_BINS = 24
_ZONE_JUDGEMENT_KEYS = ("grade", "status", "unpainted_fraction", "painted_mean_um", "gate_status")
PREVIEW_BACKGROUND = (16, 19, 26)
# Shared evenly spaced stops, low to high; callers can reproduce this legend.
PREVIEW_COLORS = ("#253494", "#2c7fb8", "#41b6c4", "#a1dab4", "#ffffcc", "#fe9929", "#d7301f")


def _read_heightmap(path: str | Path) -> tuple[np.ndarray, np.ndarray, str, np.ndarray]:
    """Return (corrected thickness, painted, convention, raw thickness where source-painted else 0)."""
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
    # RG16 is packed in two 8-bit channels, rather than a grayscale uint16 PNG.
    thickness = (
        pixels[..., 0].astype(np.float64) / 255.0
        + pixels[..., 1].astype(np.float64) / 65025.0
    ) * HEIGHTMAP_SCALE_UM
    if np.any(pixels[..., 2]):
        # B is authoritative, including valid observations of zero thickness.
        painted = pixels[..., 2] > 0
        convention = "tsb_b_channel"
    else:
        # Legacy final/expert maps have no B alpha at all. Pure black remains
        # uncoated; decoding this convention must be visible to consumers.
        painted = (pixels[..., 0] > 0) | (pixels[..., 1] > 0)
        convention = "legacy_rg_nonzero"
    corrected, painted_now = coating_assessment.corrected_painted(thickness, painted)
    return corrected, painted_now, convention, np.where(painted, thickness, 0.0)


def _preview_max(thickness: np.ndarray, painted: np.ndarray) -> float:
    values = thickness[painted]
    if not values.size:
        return 200.0
    maximum = float(np.percentile(values, 99))
    return maximum if maximum > 0 else 200.0


def _summary(
    values: np.ndarray | None,
    *,
    target_pixels: int | None,
    target_um: float,
    target_basis: str,
) -> dict[str, Any]:
    painted_pixels = int(values.size) if values is not None else None
    summary: dict[str, Any] = {
        "target_pixels": target_pixels,
        "painted_pixels": painted_pixels,
        "coverage_percent": (
            painted_pixels / target_pixels * 100.0
            if target_pixels and painted_pixels is not None else None
        ),
        "mean_um": None,
        "median_um": None,
        "p10_um": None,
        "p25_um": None,
        "p75_um": None,
        "p90_um": None,
        "std_um": None,
        "cv_percent": None,
        "target_thickness_um": target_um,
        "target_basis": target_basis,
        "normal_band_um": coating_assessment.normal_band_um(target_um),
        "six_band_counts": coating_assessment.band_counts(
            np.empty(0) if values is None else values, target_um),
    }
    if values is None or not values.size:
        return summary
    p10, p25, median, p75, p90 = np.percentile(values, (10, 25, 50, 75, 90))
    mean, std = float(np.mean(values)), float(np.std(values))
    summary.update(
        mean_um=mean,
        median_um=float(median),
        p10_um=float(p10),
        p25_um=float(p25),
        p75_um=float(p75),
        p90_um=float(p90),
        std_um=std,
        cv_percent=std / mean * 100.0 if mean > 0 else None,
    )
    return summary


def _histogram(values: np.ndarray) -> list[dict[str, Any]]:
    if not values.size:
        return []
    counts, edges = np.histogram(values, bins=HISTOGRAM_BINS, range=(0.0, max(float(values.max()), 1.0)))
    return [
        {
            "low_um": float(edges[index]),
            "high_um": float(edges[index + 1]),
            "count": int(count),
            "percent": float(count / values.size * 100.0),
        }
        for index, count in enumerate(counts)
    ]


def analyze_heightmap(
    path: str | Path,
    *,
    scenario_id: str,
    map_num: int,
    catalog: Any = None,
    target_thickness_um: float | None = None,
) -> dict[str, Any]:
    """Analyze one final TSB map; all numeric thickness values are micrometers.

    Statistics use corrected thickness of painted pixels; ``six_band_counts``
    and each zone's ``grade`` follow PR #8. A missing target is judged at the
    default and reported as ``target_basis="default_target"``. ``std_um`` is
    population standard deviation. Unavailable coverage and empty distributions are null.

    ``preview_max_um`` matches ``render_thickness_preview``'s default: the 99th
    percentile over coated image pixels, or 200 when it is not positive. The
    preview may saturate outliers; histogram/statistics never clip them.
    """
    warnings: list[str] = []
    if target_thickness_um is not None:
        target_thickness_um = float(target_thickness_um)
        if not math.isfinite(target_thickness_um) or target_thickness_um <= 0:
            target_thickness_um = None
            warnings.append("invalid_target_thickness_ignored")
    target_um, target_basis = coating_assessment.resolve_target(target_thickness_um)
    result: dict[str, Any] = {
        "status": "unavailable",
        "reason": None,
        "criterion_version": coating_assessment.ASSESSMENT_VERSION,
        "thickness_basis": "corrected",
        "basis_label": coating_assessment.basis_label(target_um, target_basis),
        "band_labels": list(coating_assessment.BAND_LABELS),
        "grade_labels": dict(coating_assessment.GRADE_SHORT_LABELS),
        "rule_note": coating_assessment.RULE_NOTE,
        "summary": _summary(None, target_pixels=None, target_um=target_um, target_basis=target_basis),
        "histogram": [],
        "zones": [],
        "statistics_basis": "painted_pixels",
        "coverage_domain": "unknown",
        "distribution_domain": "painted_image_pixels",
        "validity_convention": None,
        "preview_max_um": 200.0,
        "preview_colors": list(PREVIEW_COLORS),
        "warnings": warnings,
    }
    try:
        thickness, painted, convention, source_raw = _read_heightmap(path)
    except (OSError, ValueError, UnidentifiedImageError):
        result["reason"] = "heightmap_unavailable"
        return result
    result["validity_convention"] = convention
    if convention == "legacy_rg_nonzero":
        warnings.append("legacy_rg_nonzero_validity: B is entirely zero; nonzero RG pixels are treated as coated")
    result["preview_max_um"] = _preview_max(thickness, painted)

    domain = np.zeros(painted.shape, dtype=bool)
    definitions: tuple[Any, ...] = ()
    model = None
    reason = None
    if catalog is None:
        try:
            from trainer.zone_reference import ZoneReferenceCatalog

            catalog = ZoneReferenceCatalog.default()
        except (ImportError, OSError, ValueError):
            reason = "official_zone_catalog_unavailable"
    if catalog is not None:
        try:
            model = catalog.resolve_model(scenario_id)
            if model is None:
                reason = "scenario_model_unavailable"
            else:
                definitions = tuple(catalog.zones(model, map_num))
                if not definitions:
                    reason = "official_zones_unavailable"
        except (OSError, ValueError, KeyError):
            reason = "official_zone_catalog_unavailable"

    masks_complete = bool(definitions)
    for definition in definitions:
        mask = None
        try:
            mask_image = catalog.zone_mask(model, map_num, definition.zone_id, size=(painted.shape[1], painted.shape[0]))
            if mask_image is not None:
                candidate = np.asarray(mask_image.convert("L")) > 0
                if candidate.shape == painted.shape:
                    mask = candidate
        except (OSError, ValueError, KeyError):
            pass
        if mask is None:
            masks_complete = False
            reason = "official_zone_mask_unavailable"
            warnings.append(f"official_zone_mask_unavailable: zone {definition.zone_id}")
            zone_summary = _summary(None, target_pixels=None, target_um=target_um, target_basis=target_basis)
            judgement = None
        else:
            domain |= mask
            zone_summary = _summary(
                thickness[painted & mask],
                target_pixels=int(np.count_nonzero(mask)),
                target_um=target_um, target_basis=target_basis,
            )
            judgement = coating_assessment.assess_zone(source_raw, mask, target_um)
        zone_summary.update({key: None if judgement is None else judgement[key] for key in _ZONE_JUDGEMENT_KEYS})
        result["zones"].append({
            "zone_id": str(definition.zone_id),
            "hex_color": definition.hex_color,
            "summary": zone_summary,
        })

    if masks_complete:
        selected = thickness[painted & domain]
        target_pixels = int(np.count_nonzero(domain))
        result["status"] = "available"
        result["coverage_domain"] = "official_zone_masks"
        result["distribution_domain"] = "official_zone_masks"
        if not target_pixels:
            reason = "official_target_domain_empty"
            warnings.append("official_target_domain_empty")
    else:
        selected = thickness[painted]
        target_pixels = None
        result["status"] = "partial"
        warnings.append("coverage_unknown: official target domain unavailable; distribution uses coated image pixels")
    result["reason"] = reason
    result["summary"] = _summary(selected, target_pixels=target_pixels, target_um=target_um, target_basis=target_basis)
    result["histogram"] = _histogram(selected)
    return result


def render_thickness_preview(path: str | Path, *, vmax_um: float | None = None) -> bytes:
    """Return an RGB PNG with a physical-thickness ramp and dark uncoated pixels.

    Orientation and dimensions exactly match the source PNG. Values >= vmax_um
    use the final color stop; callers should display the scale alongside it.
    Pass analysis['preview_max_um'] to make an explicit matching legend.
    """
    thickness, painted, _, _ = _read_heightmap(path)
    if vmax_um is None:
        vmax_um = _preview_max(thickness, painted)
    vmax_um = float(vmax_um)
    if not math.isfinite(vmax_um) or vmax_um <= 0:
        raise ValueError("vmax_um must be finite and positive")
    colors = np.asarray([
        tuple(int(color[index:index + 2], 16) for index in (1, 3, 5))
        for color in PREVIEW_COLORS
    ], dtype=np.float64)
    levels = np.linspace(0.0, 1.0, len(colors))
    normalized = np.clip(thickness / vmax_um, 0.0, 1.0)
    output = np.empty((*painted.shape, 3), dtype=np.uint8)
    for channel in range(3):
        output[..., channel] = np.rint(np.interp(normalized, levels, colors[:, channel])).astype(np.uint8)
    output[~painted] = PREVIEW_BACKGROUND
    stream = io.BytesIO()
    Image.fromarray(output).save(stream, format="PNG")
    return stream.getvalue()
