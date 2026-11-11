from __future__ import annotations

import numpy as np

from trainer.zone_engine.contracts import HeightPng
from trainer.zone_engine.numerical import (
    Surface,
    assess_zone,
    combine_current_rounds,
    corrected_values,
    decode_height_png,
    overwrite_cumulative_crops,
    place_crop,
    six_band_counts,
)

from .conftest import height_png


def _one_pixel(raw_um: float, *, supported: bool = True) -> Surface:
    return Surface(
        raw_um=np.asarray([[raw_um]], dtype=np.float64),
        known=np.asarray([[True]]),
        source_painted=np.asarray([[supported]]),
    )


def test_pure_float_oracle_dv_cap_and_exact_grade_boundaries() -> None:
    cases = [
        (10.5, 0.5, None),
        (170.0, 160.0, "normal"),
        (250.0, 240.0, "above"),
        (1700.0, 1600.0, "above"),
    ]
    for raw_um, expected_corrected, expected_grade in cases:
        surface = _one_pixel(raw_um)
        corrected, painted = corrected_values(surface)
        assert corrected[0, 0] == expected_corrected
        result = assess_zone(surface, np.asarray([[True]]), 160.0)
        assert result["quality"]["grade"] == expected_grade
        assert bool(painted[0, 0]) is (expected_corrected >= 1.0)


def test_png_quantization_oracle_straddles_160_and_240_corrected_boundaries() -> None:
    # Expected values are computed directly from declared provider bytes, not from
    # a nominal target thickness passed through the implementation.
    for red, green, expected_grade in [
        (2, 42, "below"),
        (2, 43, "normal"),
        (3, 47, "normal"),
        (3, 48, "above"),
    ]:
        spec = HeightPng.model_validate(
            height_png(width=1, height=1, atlas_width=1, atlas_height=1, rgb=(red, green, 255))
        )
        decoded = decode_height_png(spec)
        expected_raw = (red / 255.0 + green / 65025.0) * 20000.0
        assert decoded.raw_um[0, 0] == expected_raw
        result = assess_zone(place_crop(decoded, spec), np.asarray([[True]]), 160.0)
        assert result["quality"]["grade"] == expected_grade


def test_current_rounds_sum_then_subtract_dead_volume_once() -> None:
    first = _one_pixel(100.0)
    second = _one_pixel(80.0)
    combined = combine_current_rounds([first, second])
    corrected, painted = corrected_values(combined)
    assert combined.raw_um[0, 0] == 180.0
    assert corrected[0, 0] == 170.0
    assert painted[0, 0]


def test_cumulative_crops_replace_same_pixel_instead_of_adding() -> None:
    first = _one_pixel(180.0)
    second = _one_pixel(200.0)
    combined = overwrite_cumulative_crops([first, second])
    corrected, _painted = corrected_values(combined)
    assert combined.raw_um[0, 0] == 200.0
    assert corrected[0, 0] == 190.0


def test_rg16_b8_and_explicit_legacy_support_are_separate() -> None:
    current_spec = HeightPng.model_validate(
        height_png(width=1, height=1, atlas_width=1, atlas_height=1, rgb=(2, 43, 0))
    )
    legacy_spec = HeightPng.model_validate(
        height_png(
            width=1,
            height=1,
            atlas_width=1,
            atlas_height=1,
            rgb=(2, 43, 0),
            encoding="legacy_rg_positive",
        )
    )
    assert not decode_height_png(current_spec).source_painted[0, 0]
    assert decode_height_png(legacy_spec).source_painted[0, 0]


def test_partial_observation_never_becomes_whole_zone_grade() -> None:
    surface = Surface(
        raw_um=np.full((2, 2), 170.0),
        known=np.asarray([[True, True], [False, False]]),
        source_painted=np.ones((2, 2), dtype=bool),
    )
    result = assess_zone(surface, np.ones((2, 2), dtype=bool), 160.0)
    assert result["quality"]["analysis_status"] == "partial"
    assert result["quality"]["grade"] is None
    assert result["quality"]["unpainted_fraction"] is None
    assert result["quality"]["six_band_counts"] is None
    assert result["metrics"]["coverage"]["fraction"] == 0.5
    assert result["metrics"]["observed_statistics"]["painted_pixel_count"] == 2


def test_missing_target_keeps_measurement_but_withholds_grade() -> None:
    result = assess_zone(_one_pixel(170.0), np.asarray([[True]]), None)
    assert result["quality"]["analysis_status"] == "unavailable"
    assert result["quality"]["grade"] is None
    assert result["quality"]["painted_mean_um"] == 160.0
    assert result["quality"]["six_band_counts"] is None


def test_six_band_exact_edges_use_left_closed_intervals() -> None:
    target = 100.0
    values = np.asarray([89.9, 90.0, 99.9, 100.0, 149.9, 150.0, 199.9, 200.0, 249.9, 250.0])
    assert six_band_counts(values, target) == {
        "lt_0_9t": 1,
        "from_0_9t_to_lt_1_0t": 2,
        "from_1_0t_to_lt_1_5t": 2,
        "from_1_5t_to_lt_2_0t": 2,
        "from_2_0t_to_lt_2_5t": 2,
        "gte_2_5t": 1,
    }
