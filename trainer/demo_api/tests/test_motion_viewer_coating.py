"""Physical coating cells and the scale exposed to the read-only viewer."""

import pytest
from PIL import Image

from trainer.demo_api.motion_viewer import _coating_reference
from trainer.scripts.build_zone_guidance_profiles import _cell_mean_um


def _zone(values, *, semantics="physical_thickness_um"):
    coating = {
        "analysis_status": "available",
        "cells": {f"{index}:0": {"median_um": value} for index, value in enumerate(values)},
    }
    if semantics is not None:
        coating["display_semantics"] = semantics
    return {"coating": coating}


def test_decoded_png_cells_stay_in_micrometers_and_share_a_profile_scale():
    # Top PNG row is UV v=1; blue marks coverage, not thickness.
    image = Image.new("RGB", (1, 2))
    image.putdata([(1, 0, 255), (0, 128, 1)])
    top_um = _cell_mean_um(image, 0, 1, 2)
    bottom_um = _cell_mean_um(image, 0, 0, 2)
    assert top_um == pytest.approx(78.431372549)
    assert bottom_um == pytest.approx(39.369473280)

    references = _coating_reference({"aggregate": {"zones": {
        "1:1": _zone([top_um, 200]),
        "1:2": _zone([100, 1000]),
    }}})
    first, second = references["1:1"], references["1:2"]
    assert first["cells"][0] == [0, 0, 78.4314]  # No second RG decoding.
    assert first["cell_unit"] == "um"
    assert first["physical_calibration"] == "available"
    assert first["color_scale_um"] == second["color_scale_um"]
    assert first["color_scale_um"] == {
        "minimum": 0.0, "maximum": 1000.0, "unit": "um",
        "scope": "loaded_profile", "basis": "maximum_cell_thickness",
        "clamp": True, "clipped_cell_count": 0,
    }


def test_unknown_and_relative_cells_cannot_set_a_physical_scale():
    references = _coating_reference({"aggregate": {"zones": {
        "1:1": _zone([100, 200]),
        "1:2": _zone([1, 99999], semantics="relative_height"),
        "1:3": _zone([25, 50000], semantics=None),
    }}})
    assert references["1:1"]["color_scale_um"]["maximum"] == 200
    for key in ("1:2", "1:3"):
        assert references[key]["cells"]
        assert references[key]["cell_unit"] is None
        assert references[key]["physical_calibration"] == "unavailable"
        assert references[key]["color_scale_um"] is None
        assert references[key]["thickness_um"] is None


def test_scale_marker_supports_profiles_without_zone_semantics():
    references = _coating_reference({
        "quality": {"heightmap_scale_status": "rg16_scale_20000"},
        "aggregate": {"zones": {"1:1": _zone([123.25], semantics=None)}},
    })
    assert references["1:1"]["cell_unit"] == "um"
    assert references["1:1"]["thickness_um"]["median"] == 123.25


def test_empty_missing_and_invalid_cells_do_not_produce_a_scale():
    assert _coating_reference({}) == {}
    references = _coating_reference({"aggregate": {"zones": {
        "1:1": _zone([None, float("nan"), float("inf"), -1, 0]),
        "1:2": {"coating": {"analysis_status": "not_available"}},
    }}})
    assert references == {}


def test_null_optional_profile_sections_are_treated_as_missing():
    assert _coating_reference({"aggregate": None}) == {}
    references = _coating_reference({
        "quality": None,
        "aggregate": {"zones": {"1:1": _zone([100], semantics=None)}},
    })
    assert references["1:1"]["physical_calibration"] == "unavailable"
