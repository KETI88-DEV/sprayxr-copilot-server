"""TSB corrected thickness arithmetic with explicit observation coverage."""

from __future__ import annotations

import base64
import hashlib
import io
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from PIL import Image

from .contracts import HeightPng
from .errors import EngineError


DEAD_VOLUME_UM = 10.0
MAX_CORRECTED_UM = 1600.0
MIN_PAINTED_UM = 1.0

SIX_BAND_KEYS = (
    "lt_0_9t",
    "from_0_9t_to_lt_1_0t",
    "from_1_0t_to_lt_1_5t",
    "from_1_5t_to_lt_2_0t",
    "from_2_0t_to_lt_2_5t",
    "gte_2_5t",
)


@dataclass(frozen=True)
class DecodedCrop:
    raw_um: np.ndarray
    source_painted: np.ndarray
    artifact_sha256: str


@dataclass(frozen=True)
class Surface:
    raw_um: np.ndarray
    known: np.ndarray
    source_painted: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.raw_um.shape[0]), int(self.raw_um.shape[1])


def decode_height_png(spec: HeightPng) -> DecodedCrop:
    """Decode the provider's R/G fixed-point thickness and explicit support."""
    try:
        raw_bytes = base64.b64decode(spec.data_base64, validate=True)
    except (ValueError, TypeError) as exc:
        raise EngineError("height_base64_invalid", "height_png is not valid base64") from exc
    actual_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    if actual_sha256 != spec.sha256:
        raise EngineError("height_hash_mismatch", "height PNG sha256 does not match bytes")
    try:
        with Image.open(io.BytesIO(raw_bytes)) as image:
            if image.format != "PNG":
                raise EngineError("height_not_png", "height artifact must be PNG")
            if image.size != (spec.width, spec.height):
                raise EngineError("height_geometry_mismatch", "decoded PNG dimensions changed")
            if image.mode not in {"RGB", "RGBA"}:
                raise EngineError(
                    "height_color_mode_invalid",
                    "height PNG must expose 8-bit R/G/B channels",
                )
            pixels = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    except OSError as exc:
        raise EngineError("height_png_invalid", "height artifact cannot be decoded") from exc

    red = pixels[..., 0].astype(np.float64)
    green = pixels[..., 1].astype(np.float64)
    raw_um = (red / 255.0 + green / 65025.0) * 20000.0
    if spec.encoding == "rg16_b8":
        source_painted = pixels[..., 2] > 0
    elif spec.encoding == "legacy_rg_positive":
        source_painted = (pixels[..., 0] > 0) | (pixels[..., 1] > 0)
    else:  # protected by the strict wire contract
        raise EngineError("height_encoding_unknown", "unsupported height encoding")
    return DecodedCrop(
        raw_um=raw_um,
        source_painted=source_painted,
        artifact_sha256=actual_sha256,
    )


def place_crop(decoded: DecodedCrop, spec: HeightPng) -> Surface:
    """Place native crop pixels on an atlas, leaving all other pixels unknown."""
    geometry = spec.crop
    shape = (geometry.atlas_height, geometry.atlas_width)
    raw_um = np.zeros(shape, dtype=np.float64)
    known = np.zeros(shape, dtype=bool)
    source_painted = np.zeros(shape, dtype=bool)
    rows = slice(geometry.y0, geometry.y0 + geometry.height)
    cols = slice(geometry.x0, geometry.x0 + geometry.width)
    raw_um[rows, cols] = decoded.raw_um
    source_painted[rows, cols] = decoded.source_painted
    known[rows, cols] = True
    return Surface(raw_um=raw_um, known=known, source_painted=source_painted)


def combine_current_rounds(surfaces: Iterable[Surface]) -> Surface:
    """Sum aligned current-round raw surfaces; apply no correction here.

    A pixel is numerically known only when every required round observes it.
    Painted support is the union across those aligned rounds.
    """
    items = list(surfaces)
    if not items:
        raise EngineError("round_lineage_empty", "no current-round Height is available")
    shape = items[0].shape
    if any(item.shape != shape for item in items):
        raise EngineError("round_geometry_mismatch", "round Height atlases are not aligned")
    raw_um = np.zeros(shape, dtype=np.float64)
    known = np.ones(shape, dtype=bool)
    source_painted = np.zeros(shape, dtype=bool)
    for item in items:
        raw_um += item.raw_um
        known &= item.known
        source_painted |= item.source_painted
    raw_um[~known] = 0.0
    source_painted &= known
    return Surface(raw_um=raw_um, known=known, source_painted=source_painted)


def overwrite_cumulative_crops(crops: Iterable[Surface]) -> Surface:
    """Overlay absolute cumulative crops in event order without summation."""
    items = list(crops)
    if not items:
        raise EngineError("cumulative_lineage_empty", "no cumulative crop is available")
    shape = items[0].shape
    if any(item.shape != shape for item in items):
        raise EngineError("cumulative_geometry_mismatch", "cumulative atlases are not aligned")
    raw_um = np.zeros(shape, dtype=np.float64)
    known = np.zeros(shape, dtype=bool)
    source_painted = np.zeros(shape, dtype=bool)
    for item in items:
        replace = item.known
        raw_um[replace] = item.raw_um[replace]
        source_painted[replace] = item.source_painted[replace]
        known[replace] = True
    return Surface(raw_um=raw_um, known=known, source_painted=source_painted)


def corrected_values(surface: Surface) -> tuple[np.ndarray, np.ndarray]:
    """Subtract DV10 once, cap at 1600, and identify corrected painted pixels."""
    corrected = np.minimum(surface.raw_um - DEAD_VOLUME_UM, MAX_CORRECTED_UM)
    painted = surface.known & surface.source_painted & (corrected >= MIN_PAINTED_UM)
    return corrected, painted


def grade_for_mean(mean_um: float, target_um: float) -> str:
    if mean_um < target_um:
        return "below"
    if mean_um < 1.5 * target_um:
        return "normal"
    return "above"


def six_band_counts(values: np.ndarray, target_um: float) -> dict[str, int]:
    edges = np.asarray(
        [0.9 * target_um, target_um, 1.5 * target_um, 2.0 * target_um, 2.5 * target_um],
        dtype=np.float64,
    )
    indices = np.searchsorted(edges, values, side="right")
    counts = np.bincount(indices, minlength=6)
    return {key: int(counts[index]) for index, key in enumerate(SIX_BAND_KEYS)}


def assess_zone(
    surface: Surface,
    zone_mask: np.ndarray,
    target_um: float | None,
) -> dict[str, Any]:
    """Measure only exact Zone pixels and withhold whole-Zone claims if partial."""
    if zone_mask.dtype != bool or zone_mask.shape != surface.raw_um.shape:
        raise EngineError("zone_mask_geometry_mismatch", "Zone mask and Height atlas differ")
    zone_pixel_count = int(np.count_nonzero(zone_mask))
    if zone_pixel_count <= 0:
        raise EngineError("zone_mask_empty", "exact Zone mask has no pixels")
    observed = zone_mask & surface.known
    observed_zone_pixel_count = int(np.count_nonzero(observed))
    fraction = observed_zone_pixel_count / zone_pixel_count
    full_zone = observed_zone_pixel_count == zone_pixel_count
    corrected, painted_all = corrected_values(surface)
    painted = observed & painted_all
    painted_values = corrected[painted]
    painted_pixel_count = int(painted_values.size)
    unpainted_pixel_count = observed_zone_pixel_count - painted_pixel_count
    painted_sum_um = float(np.sum(painted_values, dtype=np.float64))
    painted_mean_um = (
        float(painted_sum_um / painted_pixel_count) if painted_pixel_count else None
    )
    bands = (
        six_band_counts(painted_values, target_um)
        if target_um is not None and painted_pixel_count
        else ({key: 0 for key in SIX_BAND_KEYS} if target_um is not None else None)
    )
    observed_statistics = None
    if observed_zone_pixel_count:
        observed_statistics = {
            "painted_mean_um": painted_mean_um,
            "painted_sum_um": painted_sum_um,
            "painted_pixel_count": painted_pixel_count,
            "unpainted_pixel_count": unpainted_pixel_count,
            "six_band_counts": bands,
        }

    if not full_zone:
        analysis_status = "partial"
        reason_codes = ["partial_zone_observation"]
        grade = None
        quality_mean = None
        quality_painted_count = 0
        unpainted_fraction = None
        quality_bands = None
        scope = "observed_only"
    elif target_um is None:
        analysis_status = "unavailable"
        reason_codes = ["target_thickness_missing"]
        grade = None
        quality_mean = painted_mean_um
        quality_painted_count = painted_pixel_count
        unpainted_fraction = unpainted_pixel_count / zone_pixel_count
        quality_bands = None
        scope = "whole_zone"
    elif painted_mean_um is None:
        analysis_status = "unavailable"
        reason_codes = ["no_valid_painted_pixels"]
        grade = None
        quality_mean = None
        quality_painted_count = 0
        unpainted_fraction = 1.0
        quality_bands = bands
        scope = "whole_zone"
    else:
        analysis_status = "available"
        reason_codes = []
        grade = grade_for_mean(painted_mean_um, target_um)
        quality_mean = painted_mean_um
        quality_painted_count = painted_pixel_count
        unpainted_fraction = unpainted_pixel_count / zone_pixel_count
        quality_bands = bands
        scope = "whole_zone"

    return {
        "quality": {
            "analysis_status": analysis_status,
            "reason_codes": reason_codes,
            "grade": grade,
            "target_thickness_um": target_um,
            "painted_mean_um": quality_mean,
            "painted_pixel_count": quality_painted_count,
            "zone_pixel_count": zone_pixel_count,
            "unpainted_fraction": unpainted_fraction,
            "six_band_counts": quality_bands,
            "statistics_scope": scope,
        },
        "metrics": {
            "observed_statistics": observed_statistics,
            "coverage": {
                "zone_pixel_count": zone_pixel_count,
                "observed_zone_pixel_count": observed_zone_pixel_count,
                "fraction": fraction,
                "full_zone": full_zone,
            },
            "provider_official_summary": None,
        },
    }
