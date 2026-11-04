"""Physical TSB thickness, official domain coverage, and preview orientation."""

from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image


_MODULE_PATH = Path(__file__).resolve().parents[1] / "training_data_coating.py"
_SPEC = importlib.util.spec_from_file_location("training_data_coating", _MODULE_PATH)
coating = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(coating)


class OfficialCatalog:
    """Tiny exact label masks, including a hole a bounding rectangle would fill."""

    def __init__(self, masks, *, missing=(), model="Learn03"):
        self.masks = masks
        self.missing = set(missing)
        self.model = model

    def resolve_model(self, scenario_id):
        return self.model if scenario_id == "3" else None

    def zones(self, model, map_num):
        return tuple(
            SimpleNamespace(zone_id=zone_id, hex_color=color)
            for zone_id, color in (("1", "#ff0000"), ("2", "#00ff00"))
            if (map_num, zone_id) in self.masks
        )

    def zone_mask(self, model, map_num, zone_id, *, size=None):
        if (map_num, zone_id) in self.missing:
            raise OSError("official label map is absent")
        values = self.masks[(map_num, zone_id)]
        image = Image.fromarray(np.asarray(values, dtype=np.uint8) * 255)
        return image if image.size == size else image.resize(size, Image.Resampling.NEAREST)


class TrainingDataCoatingTest(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "HeightMap_Map01.png"

    def save(self, rows):
        Image.fromarray(np.asarray(rows, dtype=np.uint8)).save(self.path)
        return self.path

    def test_rg_channels_decode_to_micrometers_and_ignore_invalid_blue(self):
        # Replacing RG decoding with raw RGB magnitude or including B=0 fails.
        self.save([[(1, 0, 255), (0, 255, 255), (255, 255, 0)]])
        catalog = OfficialCatalog({(1, "1"): [[1, 1, 1]]})
        result = coating.analyze_heightmap(self.path, scenario_id="3", map_num=1, catalog=catalog)
        self.assertEqual(result["status"], "available")
        self.assertEqual(result["summary"]["painted_pixels"], 2)
        self.assertAlmostEqual(result["summary"]["mean_um"], 78.43137254901961)
        self.assertAlmostEqual(result["summary"]["coverage_percent"], 200 / 3)

    def test_exact_masks_exclude_outside_paint_and_count_inside_black(self):
        # The center hole and bottom row are outside the official target domain.
        self.save([
            [(1, 0, 255), (200, 0, 255), (0, 0, 0)],
            [(0, 255, 255), (0, 0, 0), (0, 255, 255)],
            [(0, 0, 0), (200, 0, 255), (0, 0, 0)],
        ])
        catalog = OfficialCatalog({
            (1, "1"): [[1, 0, 1], [0, 0, 0], [0, 0, 0]],
            (1, "2"): [[0, 0, 0], [1, 0, 1], [0, 0, 0]],
        })
        result = coating.analyze_heightmap(self.path, scenario_id="3", map_num=1, catalog=catalog)
        self.assertEqual(result["summary"]["target_pixels"], 4)
        self.assertEqual(result["summary"]["painted_pixels"], 3)
        self.assertEqual(result["summary"]["coverage_percent"], 75.0)
        self.assertAlmostEqual(result["summary"]["mean_um"], 78.43137254901961)
        self.assertEqual(result["zones"][0]["hex_color"], "#ff0000")
        self.assertEqual(result["zones"][0]["summary"]["coverage_percent"], 50.0)
        self.assertEqual(result["zones"][1]["summary"]["coverage_percent"], 100.0)
        self.assertEqual(result["statistics_basis"], "painted_pixels")

    def test_distribution_reports_painted_quantiles_not_zero_background(self):
        self.save([[(0, 255, 255), (1, 255, 255), (2, 255, 255), (0, 0, 0)]])
        result = coating.analyze_heightmap(
            self.path, scenario_id="3", map_num=1,
            catalog=OfficialCatalog({(1, "1"): [[1, 1, 1, 1]]}),
        )
        summary = result["summary"]
        self.assertAlmostEqual(summary["median_um"], 156.86274509803923)
        self.assertAlmostEqual(summary["p10_um"], 94.11764705882354)
        self.assertAlmostEqual(summary["p25_um"], 117.64705882352942)
        self.assertAlmostEqual(summary["p75_um"], 196.07843137254903)
        self.assertAlmostEqual(summary["p90_um"], 219.60784313725492)
        self.assertAlmostEqual(summary["std_um"], 64.03894752374435)
        self.assertAlmostEqual(summary["cv_percent"], 40.8248290463863)
        self.assertEqual(sum(item["count"] for item in result["histogram"]), 3)
        self.assertAlmostEqual(sum(item["percent"] for item in result["histogram"]), 100)
        self.assertEqual(len(result["histogram"]), 24)

    def test_missing_mask_never_claims_target_area_or_global_coverage(self):
        self.save([[(1, 0, 255), (0, 0, 0), (2, 0, 255)]])
        catalog = OfficialCatalog(
            {(1, "1"): [[1, 0, 0]], (1, "2"): [[0, 1, 0]]}, missing={(1, "2")},
        )
        result = coating.analyze_heightmap(self.path, scenario_id="3", map_num=1, catalog=catalog)
        self.assertEqual(result["status"], "partial")
        self.assertIsNone(result["summary"]["target_pixels"])
        self.assertIsNone(result["summary"]["coverage_percent"])
        self.assertEqual(result["summary"]["painted_pixels"], 2)
        self.assertEqual(result["zones"][0]["summary"]["coverage_percent"], 100)
        self.assertIsNone(result["zones"][1]["summary"]["painted_pixels"])
        self.assertIsNone(result["zones"][1]["summary"]["coverage_percent"])
        self.assertTrue(result["reason"])
        self.assertTrue(result["warnings"])

    def test_unknown_scenario_returns_painted_statistics_with_null_coverage(self):
        self.save([[(1, 0, 255), (0, 0, 0)]])
        result = coating.analyze_heightmap(
            self.path, scenario_id="unknown", map_num=1, catalog=OfficialCatalog({}),
        )
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["summary"]["painted_pixels"], 1)
        self.assertAlmostEqual(result["summary"]["mean_um"], 78.43137254901961)
        self.assertIsNone(result["summary"]["coverage_percent"])
        self.assertEqual(result["zones"], [])

    def test_map_number_selects_an_independent_domain_without_filename_inference(self):
        self.save([[(1, 0, 255), (0, 0, 0)]])
        catalog = OfficialCatalog({(1, "1"): [[1, 0]], (2, "1"): [[0, 1]]})
        result = coating.analyze_heightmap(self.path, scenario_id="3", map_num=2, catalog=catalog)
        self.assertEqual(result["summary"]["coverage_percent"], 0)
        self.assertEqual(result["summary"]["painted_pixels"], 0)

    def test_no_target_does_not_invent_a_target_or_pass_rate(self):
        self.save([[(1, 0, 255)]])
        result = coating.analyze_heightmap(
            self.path, scenario_id="3", map_num=1, catalog=OfficialCatalog({(1, "1"): [[1]]}),
        )
        for summary in (result["summary"], result["zones"][0]["summary"]):
            self.assertIsNone(summary["target_thickness_um"])
            self.assertIsNone(summary["within_target_percent"])
            self.assertIsNone(summary["target_tolerance_percent"])

    def test_supplied_target_uses_explicit_ten_percent_tolerance(self):
        self.save([[(1, 0, 255), (2, 0, 255), (0, 0, 0)]])
        result = coating.analyze_heightmap(
            self.path, scenario_id="3", map_num=1, target_thickness_um=80,
            catalog=OfficialCatalog({(1, "1"): [[1, 1, 1]]}),
        )
        self.assertEqual(result["summary"]["target_thickness_um"], 80)
        self.assertEqual(result["summary"]["target_tolerance_percent"], 10)
        self.assertEqual(result["summary"]["within_target_percent"], 50)

    def test_all_black_target_has_zero_coverage_and_no_distribution(self):
        self.save([[(0, 0, 0), (0, 0, 0)]])
        result = coating.analyze_heightmap(
            self.path, scenario_id="3", map_num=1, target_thickness_um=100,
            catalog=OfficialCatalog({(1, "1"): [[1, 1]]}),
        )
        self.assertEqual(result["summary"]["target_pixels"], 2)
        self.assertEqual(result["summary"]["painted_pixels"], 0)
        self.assertEqual(result["summary"]["coverage_percent"], 0)
        for key in ("mean_um", "median_um", "p10_um", "p90_um", "std_um", "cv_percent", "within_target_percent"):
            self.assertIsNone(result["summary"][key])
        self.assertEqual(result["histogram"], [])
        self.assertGreater(result["preview_max_um"], 0)
        json.dumps(result, allow_nan=False)

    def test_blue_validity_is_authoritative_even_at_zero_thickness(self):
        self.save([[(0, 0, 255), (0, 0, 0), (2, 0, 0)]])
        result = coating.analyze_heightmap(
            self.path, scenario_id="3", map_num=1,
            catalog=OfficialCatalog({(1, "1"): [[1, 1, 1]]}),
        )
        self.assertEqual(result["validity_convention"], "tsb_b_channel")
        self.assertEqual(result["summary"]["painted_pixels"], 1)
        self.assertAlmostEqual(result["summary"]["coverage_percent"], 100 / 3)
        self.assertEqual(result["summary"]["mean_um"], 0)
        self.assertIsNone(result["summary"]["cv_percent"])
        self.assertGreater(result["preview_max_um"], 0)
        self.assertAlmostEqual(sum(item["percent"] for item in result["histogram"]), 100)

    def test_legacy_all_zero_blue_uses_rg_nonzero_with_warning(self):
        self.save([[(1, 0, 0), (0, 255, 0), (0, 0, 0)]])
        result = coating.analyze_heightmap(
            self.path, scenario_id="3", map_num=1,
            catalog=OfficialCatalog({(1, "1"): [[1, 1, 1]]}),
        )
        self.assertEqual(result["validity_convention"], "legacy_rg_nonzero")
        self.assertTrue(result["warnings"])
        self.assertEqual(result["summary"]["painted_pixels"], 2)
        self.assertAlmostEqual(result["summary"]["coverage_percent"], 200 / 3)
        self.assertAlmostEqual(result["summary"]["mean_um"], 78.43137254901961)

    def test_empty_official_mask_has_known_zero_area_without_coverage_percentage(self):
        self.save([[(1, 0, 255)]])
        result = coating.analyze_heightmap(
            self.path, scenario_id="3", map_num=1,
            catalog=OfficialCatalog({(1, "1"): [[0]]}),
        )
        self.assertEqual(result["summary"]["target_pixels"], 0)
        self.assertEqual(result["summary"]["painted_pixels"], 0)
        self.assertIsNone(result["summary"]["coverage_percent"])

    def test_unreadable_png_returns_unavailable_instead_of_fabricated_statistics(self):
        self.path.write_bytes(b"not a png")
        result = coating.analyze_heightmap(self.path, scenario_id="3", map_num=1, catalog=OfficialCatalog({}))
        self.assertEqual(result["status"], "unavailable")
        self.assertIsNone(result["summary"]["coverage_percent"])
        self.assertIsNone(result["summary"]["mean_um"])
        self.assertEqual(result["histogram"], [])

    def test_preview_retains_top_left_orientation_and_black_background(self):
        self.save([[(1, 0, 255), (0, 0, 0)], [(0, 0, 0), (2, 0, 255)]])
        preview = Image.open(io.BytesIO(coating.render_thickness_preview(self.path, vmax_um=200))).convert("RGB")
        pixels = np.asarray(preview)
        self.assertEqual(preview.size, (2, 2))
        self.assertLess(int(pixels[0, 1].max()), 35)
        self.assertTrue(np.array_equal(pixels[0, 1], pixels[1, 0]))
        self.assertFalse(np.array_equal(pixels[0, 0], pixels[0, 1]))
        self.assertFalse(np.array_equal(pixels[0, 0], pixels[1, 1]))

    def test_preview_default_scale_matches_analysis_and_clamps_high_values(self):
        self.save([[(1, 0, 255), (255, 255, 255)]])
        result = coating.analyze_heightmap(
            self.path, scenario_id="3", map_num=1, catalog=OfficialCatalog({(1, "1"): [[1, 1]]}),
        )
        implicit = coating.render_thickness_preview(self.path)
        explicit = coating.render_thickness_preview(self.path, vmax_um=result["preview_max_um"])
        self.assertEqual(implicit, explicit)
        clamped = np.asarray(Image.open(io.BytesIO(coating.render_thickness_preview(self.path, vmax_um=1))))
        self.assertTrue(np.array_equal(clamped[0, 0], clamped[0, 1]))


if __name__ == "__main__":
    unittest.main()
