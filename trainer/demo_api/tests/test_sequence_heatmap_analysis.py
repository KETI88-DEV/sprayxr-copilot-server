"""Sequence PNG safety, physical RG16 decoding, and atlas crop projection."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
import struct
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zlib

import numpy as np
from PIL import Image


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sequence_heatmap_analysis as analysis


FORMAT = "v2;thickness=RG16;alpha=B8;scale_um=20000"


def png_bytes(rows):
    stream = io.BytesIO()
    Image.fromarray(np.asarray(rows, dtype=np.uint8)).save(stream, format="PNG")
    return stream.getvalue()


def item_for(rows, **changes):
    raw = png_bytes(rows)
    image_height, image_width = np.asarray(rows).shape[:2]
    item = {
        "map_num": 1, "x0": 100, "y0": 200,
        "width": image_width, "height": image_height,
        "img_width": image_width, "img_height": image_height,
        "format": FORMAT, "encoding": "png;base64",
        "data": base64.b64encode(raw).decode("ascii"),
    }
    item.update(changes)
    return item


def atlas_mask(*points):
    """Points are explicit top-left label-map (row, column) coordinates."""
    mask = np.zeros((2048, 2048), dtype=np.uint8)
    for row, column in points:
        mask[row, column] = 255
    return mask


class OfficialCatalog:
    """A complete catalog boundary with exact masks, never rectangle inference."""

    def __init__(self, masks, *, missing=()):
        self.masks = masks
        self.missing = set(missing)

    def resolve_model(self, scenario_id):
        return "Learn03" if scenario_id == "3" else None

    def zones(self, model_key, map_num):
        return tuple(
            SimpleNamespace(zone_id=zone_id, hex_color="#ff0000")
            for map_id, zone_id in self.masks if map_id == map_num
        )

    def zone_mask(self, model_key, map_num, zone_id, *, size=None):
        if size != (2048, 2048):
            raise ValueError("The crop must be projected from the full atlas mask")
        if (map_num, zone_id) in self.missing:
            raise OSError("missing official mask")
        return Image.fromarray(self.masks[(map_num, zone_id)])


class SequenceHeatmapAnalysisTest(unittest.TestCase):
    def prepare(self, item, *, catalog=None, **kwargs):
        if catalog is None:
            catalog = OfficialCatalog({(1, "1"): np.full((2048, 2048), 255, dtype=np.uint8)})
        return analysis.prepare_heatmap(item, scenario_id="3", catalog=catalog, **kwargs)

    def test_rg16_numeric_values_and_blue_validity(self):
        # Raw RGB magnitudes, green /255, and including invalid pixels all fail.
        item = item_for([[(1, 0, 255), (0, 255, 1), (255, 255, 0)]])
        prepared = self.prepare(item)
        summary = prepared.payload["summary"]
        self.assertEqual(prepared.payload["status"], "available")
        self.assertEqual(summary["painted_pixels"], 2)
        # Both valid pixels are raw 78.43 um; the corrected value is 10 um lower.
        self.assertAlmostEqual(summary["mean_um"], 68.43137254901961)
        self.assertAlmostEqual(summary["coverage_percent"], 200 / 3)
        self.assertEqual(prepared.original_png, base64.b64decode(item["data"]))

    def test_all_zero_blue_is_authoritatively_unpainted_in_v2(self):
        result = self.prepare(item_for([[(200, 255, 0), (1, 0, 0)]])).payload
        self.assertEqual(result["validity_convention"], "tsb_b_channel")
        self.assertEqual(result["summary"]["painted_pixels"], 0)
        self.assertEqual(result["summary"]["coverage_percent"], 0)
        self.assertIsNone(result["summary"]["mean_um"])
        self.assertIsNone(result["preview_max_um"])
        self.assertEqual(result["histogram"], [])
        self.assertFalse(any("legacy" in warning for warning in result["warnings"]))

    def test_zero_thickness_with_nonzero_blue_is_unpainted_after_correction(self):
        # PR #8: painted needs corrected >= 1 um, so B>0 with raw 0 is not painted.
        result = self.prepare(item_for([[(0, 0, 1), (255, 255, 0)]])).payload
        self.assertEqual(result["summary"]["painted_pixels"], 0)
        self.assertEqual(result["summary"]["coverage_percent"], 0)
        self.assertIsNone(result["summary"]["mean_um"])
        self.assertEqual(result["histogram"], [])
        json.dumps(result, allow_nan=False)

    def test_crop_uses_bottom_left_atlas_origin_but_keeps_png_top_left(self):
        # Crop is x=[100,102), y=[200,202); PNG top-left -> atlas (100,201).
        item = item_for([
            [(1, 0, 255), (100, 0, 255)],
            [(3, 0, 255), (0, 0, 0)],
        ])
        catalog = OfficialCatalog({
            (1, "1"): atlas_mask((1846, 100), (1847, 101)),
            (1, "2"): atlas_mask((0, 0)),
        })
        result = self.prepare(item, catalog=catalog).payload
        self.assertEqual(result["summary"]["target_pixels"], 2)
        self.assertEqual(result["summary"]["painted_pixels"], 1)
        self.assertEqual(result["summary"]["coverage_percent"], 50)
        self.assertAlmostEqual(result["summary"]["mean_um"], 68.43137254901961)
        self.assertEqual([zone["zone_id"] for zone in result["zones"]], ["1"])
        self.assertEqual(result["coverage_domain"], "official_zone_masks_in_crop")
        self.assertEqual(result["distribution_domain"], "official_zone_masks_in_crop")

    def test_downsampled_crop_projects_each_pixel_center_into_atlas_mask(self):
        # 2x2 PNG over a 4x6 crop: centers x=(101,103), bottom-y=(204,201).
        item = item_for([
            [(1, 0, 255), (2, 0, 255)],
            [(3, 0, 255), (4, 0, 255)],
        ], width=4, height=6)
        catalog = OfficialCatalog({(1, "1"): atlas_mask((1843, 103), (1846, 101))})
        result = self.prepare(item, catalog=catalog).payload
        self.assertEqual(result["summary"]["target_pixels"], 2)
        self.assertEqual(result["summary"]["painted_pixels"], 2)
        self.assertAlmostEqual(result["summary"]["mean_um"], 186.07843137254903)
        self.assertEqual(result["zones"][0]["summary"]["coverage_percent"], 100)

    def test_exact_mask_hole_does_not_become_target_area(self):
        item = item_for([[(1, 0, 255), (200, 0, 255), (0, 0, 0)]])
        catalog = OfficialCatalog({(1, "1"): atlas_mask((1847, 100), (1847, 102))})
        result = self.prepare(item, catalog=catalog).payload
        self.assertEqual(result["summary"]["target_pixels"], 2)
        self.assertEqual(result["summary"]["coverage_percent"], 50)
        self.assertAlmostEqual(result["summary"]["mean_um"], 68.43137254901961)

    def test_incomplete_masks_use_image_paint_and_never_claim_global_coverage(self):
        item = item_for([[(1, 0, 255), (0, 0, 0), (3, 0, 255)]])
        catalog = OfficialCatalog({
            (1, "1"): atlas_mask((1847, 100)),
            (1, "2"): atlas_mask((1847, 101)),
        }, missing={(1, "2")})
        result = self.prepare(item, catalog=catalog).payload
        self.assertEqual(result["status"], "partial")
        self.assertIsNone(result["summary"]["target_pixels"])
        self.assertIsNone(result["summary"]["coverage_percent"])
        self.assertEqual(result["summary"]["painted_pixels"], 2)
        self.assertAlmostEqual(result["summary"]["mean_um"], 146.86274509803923)
        self.assertEqual(result["coverage_domain"], "unknown")
        self.assertEqual(result["distribution_domain"], "painted_image_pixels")
        self.assertTrue(any("coverage_unknown" in warning for warning in result["warnings"]))
        self.assertEqual([zone["zone_id"] for zone in result["zones"]], ["1"])

    def test_map_num_selects_the_correct_masks(self):
        item = item_for([[(1, 0, 255), (0, 0, 0)]], map_num=2)
        catalog = OfficialCatalog({
            (1, "1"): atlas_mask((1847, 100)),
            (2, "1"): atlas_mask((1847, 101)),
        })
        result = self.prepare(item, catalog=catalog).payload
        self.assertEqual(result["summary"]["coverage_percent"], 0)

    def test_invalid_crop_retains_physical_distribution_with_unknown_coverage(self):
        for changes in ({"x0": -1}, {"width": 0}, {"y0": 2048}, {"width": float("nan")}):
            with self.subTest(changes=changes):
                result = self.prepare(item_for([[(1, 0, 255)]], **changes)).payload
                self.assertEqual(result["status"], "partial")
                self.assertIsNone(result["summary"]["coverage_percent"])
                self.assertAlmostEqual(result["summary"]["mean_um"], 68.43137254901961)
                self.assertEqual(result["reason"], "invalid_crop_geometry")

    def test_no_target_is_judged_at_the_default_and_says_so(self):
        result = self.prepare(item_for([[(1, 0, 255)]])).payload
        for summary in (result["summary"], result["zones"][0]["summary"]):
            self.assertEqual(summary["target_thickness_um"], 160.0)
            self.assertEqual(summary["target_basis"], "default_target")
            self.assertNotIn("target_tolerance_percent", summary)
            self.assertNotIn("within_target_percent", summary)
        self.assertEqual(result["basis_label"], "기본 목표 160µm 기준")
        self.assertEqual(result["thickness_basis"], "corrected")

    def test_explicit_target_is_a_session_target(self):
        result = self.prepare(item_for([[(1, 0, 255), (2, 0, 255)]]), target_thickness_um=80).payload
        self.assertEqual(result["summary"]["target_thickness_um"], 80.0)
        self.assertEqual(result["summary"]["target_basis"], "session_target")
        self.assertEqual(result["basis_label"], "세션 목표 80µm 기준")
        # corrected 68.43 and 146.86 against edges 72, 80, 120, 160, 200
        self.assertEqual(list(result["summary"]["six_band_counts"].values()), [1, 0, 0, 1, 0, 0])

    def test_invalid_targets_fall_back_to_the_default_without_nonfinite_json(self):
        for target in (0, -1, float("nan"), float("inf"), "bad"):
            with self.subTest(target=target):
                result = self.prepare(item_for([[(1, 0, 255)]]), target_thickness_um=target).payload
                self.assertEqual(result["summary"]["target_thickness_um"], 160.0)
                self.assertEqual(result["summary"]["target_basis"], "default_target")
                self.assertIn("invalid_target_thickness_ignored", result["warnings"])
                json.dumps(result, allow_nan=False)

    def test_crop_zone_summaries_carry_no_grade(self):
        result = self.prepare(item_for([[(1, 0, 255), (0, 0, 0)]])).payload
        self.assertEqual(result["grade_scope"], "not_graded_partial_crop")
        self.assertTrue(result["zones"])
        for summary in [result["summary"], *(zone["summary"] for zone in result["zones"])]:
            self.assertNotIn("grade", summary)
        self.assertEqual(result["zones"][0]["summary"]["sampled_unpainted_fraction"], 0.5)

    def test_crop_bands_match_the_engine_rule(self):
        # R=1,2,3 -> raw 78.4, 156.9, 235.3 -> corrected 68.4, 146.9, 225.3.
        # Edges for target 160: 144, 160, 240, 320, 400 -> bands 0, 1, 2.
        result = self.prepare(item_for([[(1, 0, 255), (2, 0, 255), (3, 0, 255)]])).payload
        counts = result["summary"]["six_band_counts"]
        self.assertEqual(list(counts.values()), [1, 1, 1, 0, 0, 0])
        corrected = np.asarray([68.43137254901961, 146.86274509803923, 225.29411764705884])
        self.assertEqual(counts, analysis.coating_assessment.band_counts(corrected, 160.0))

    def test_quantiles_histogram_and_p99_preview_do_not_clip_statistics(self):
        item = item_for([[(0, 255, 255), (1, 255, 255), (2, 255, 255), (0, 0, 0)]])
        prepared = self.prepare(item)
        result = prepared.payload
        self.assertAlmostEqual(result["summary"]["p10_um"], 84.11764705882354)
        self.assertAlmostEqual(result["summary"]["p25_um"], 107.64705882352942)
        self.assertAlmostEqual(result["summary"]["median_um"], 146.86274509803923)
        self.assertAlmostEqual(result["summary"]["p75_um"], 186.07843137254903)
        self.assertAlmostEqual(result["summary"]["p90_um"], 209.60784313725492)
        self.assertAlmostEqual(result["summary"]["std_um"], 64.03894752374435)
        self.assertAlmostEqual(result["summary"]["cv_percent"], 43.6046238145648)
        self.assertEqual(len(result["histogram"]), 24)
        self.assertEqual(sum(entry["count"] for entry in result["histogram"]), 3)
        self.assertAlmostEqual(sum(entry["percent"] for entry in result["histogram"]), 100)
        self.assertAlmostEqual(result["preview_max_um"], 223.72549019607843)
        self.assertGreater(result["histogram"][-1]["high_um"], result["preview_max_um"])
        self.assertEqual(result["preview_cap_basis"], "painted_pixels_p99")
        self.assertFalse(result["preview_cap_affects_statistics"])
        self.assertEqual(result["statistics_basis"], "painted_pixels")

    def test_preview_preserves_orientation_dimensions_and_dark_invalid_pixels(self):
        prepared = self.prepare(item_for([
            [(1, 0, 255), (255, 255, 0)],
            [(0, 0, 0), (2, 0, 255)],
        ]))
        preview = Image.open(io.BytesIO(prepared.thickness_png))
        pixels = np.asarray(preview)
        self.assertEqual(preview.size, (2, 2))
        self.assertTrue(np.array_equal(pixels[0, 1], (16, 19, 26)))
        self.assertTrue(np.array_equal(pixels[1, 0], (16, 19, 26)))
        self.assertFalse(np.array_equal(pixels[0, 0], pixels[1, 1]))

    def test_unknown_or_ambiguous_format_has_raw_preview_only(self):
        for format_value in ("", "RG16", FORMAT + ";extra=yes", FORMAT.replace("B8", "A8")):
            with self.subTest(format=format_value):
                prepared = self.prepare(item_for([[(1, 0, 255)]], format=format_value))
                self.assertIsNotNone(prepared.original_png)
                self.assertIsNone(prepared.thickness_png)
                self.assertEqual(prepared.payload["status"], "unavailable")
                self.assertEqual(prepared.payload["reason"], "unsupported_heatmap_format")
                self.assertIsNone(prepared.payload["summary"]["mean_um"])
                self.assertIsNone(prepared.payload["preview_max_um"])

    def test_exact_format_supports_an_explicit_finite_positive_scale(self):
        item = item_for([[(1, 0, 255)]], format=FORMAT.replace("20000", "10000.0"))
        result = self.prepare(item).payload
        self.assertAlmostEqual(result["summary"]["mean_um"], 29.21568627450981)
        self.assertEqual(result["scale_um"], 10000)

    def test_nonfinite_and_nonpositive_scales_are_not_physical_thickness(self):
        for scale in ("0", "-1", "NaN", "Infinity", "1e309"):
            with self.subTest(scale=scale):
                prepared = self.prepare(item_for([[(1, 0, 255)]], format=FORMAT.replace("20000", scale)))
                self.assertIsNotNone(prepared.original_png)
                self.assertIsNone(prepared.thickness_png)
                self.assertIsNone(prepared.payload["summary"]["mean_um"])

    def test_blasting_retains_raw_png_without_coating_thickness(self):
        prepared = self.prepare(item_for([[(1, 0, 255)]]), process="blasting")
        self.assertIsNotNone(prepared.original_png)
        self.assertIsNone(prepared.thickness_png)
        self.assertEqual(prepared.payload["reason"], "physical_thickness_not_applicable")
        self.assertIsNone(prepared.payload["summary"]["mean_um"])
        self.assertIsNone(prepared.payload["summary"]["coverage_percent"])
        self.assertEqual(prepared.payload["histogram"], [])

    def test_unknown_process_retains_raw_png_without_assuming_spray(self):
        for process in (None, "", "unknown", "blasting_gun", "other"):
            with self.subTest(process=process):
                prepared = self.prepare(item_for([[(1, 0, 255)]]), process=process)
                self.assertIsNotNone(prepared.original_png)
                self.assertIsNone(prepared.thickness_png)
                self.assertEqual(prepared.payload["reason"], "process_unknown")

    def test_non_rgb_png_cannot_be_interpreted_as_packed_rg16(self):
        raw = png_bytes([[127]])
        prepared = self.prepare(item_for([[(1, 0, 255)]], data=base64.b64encode(raw).decode("ascii")))
        self.assertIsNotNone(prepared.original_png)
        self.assertIsNone(prepared.thickness_png)
        self.assertEqual(prepared.payload["reason"], "unsupported_physical_png_mode")

    def test_extreme_finite_scale_never_emits_nonfinite_analytics(self):
        item = item_for([[(128, 0, 255), (255, 0, 255)]], format=FORMAT.replace("20000", "1e308"))
        prepared = self.prepare(item)
        json.dumps(prepared.payload, allow_nan=False)

    def test_extreme_finite_target_keeps_json_finite(self):
        prepared = self.prepare(item_for([[(1, 0, 255)]]), target_thickness_um=1e308)
        self.assertEqual(prepared.payload["summary"]["six_band_counts"]["lt_0_9t"], 1)
        json.dumps(prepared.payload, allow_nan=False)

    def test_missing_or_tampered_base64_is_unavailable(self):
        for bad_data in (None, "", "!!!!", "aGVs bG8=", "한글"):
            with self.subTest(data=bad_data):
                prepared = self.prepare(item_for([[(1, 0, 255)]], data=bad_data))
                self.assertEqual(prepared.payload["status"], "unavailable")
                self.assertIsNone(prepared.original_png)
                self.assertIsNone(prepared.thickness_png)
        item = item_for([[(1, 0, 255)]])
        del item["data"]
        self.assertIsNone(self.prepare(item).original_png)

    def test_non_png_or_corrupt_png_is_never_returned_as_a_raw_image(self):
        stream = io.BytesIO()
        Image.new("RGB", (1, 1)).save(stream, format="JPEG")
        for raw in (b"not a PNG", stream.getvalue(), png_bytes([[(1, 0, 255)]])[:-8]):
            with self.subTest(raw_size=len(raw)):
                prepared = self.prepare(item_for([[(1, 0, 255)]], data=base64.b64encode(raw).decode("ascii")))
                self.assertEqual(prepared.payload["status"], "unavailable")
                self.assertIsNone(prepared.original_png)

    def test_unknown_format_still_requires_decodable_png_pixel_data(self):
        original = png_bytes([[(1, 0, 255)]])
        data = b"invalid zlib stream"
        idat = struct.pack(">I", len(data)) + b"IDAT" + data + struct.pack(">I", zlib.crc32(b"IDAT" + data))
        raw = original[:33] + idat + original[-12:]
        prepared = self.prepare(item_for(
            [[(1, 0, 255)]], format="unknown", data=base64.b64encode(raw).decode("ascii"),
        ))
        self.assertIsNone(prepared.original_png)
        self.assertEqual(prepared.payload["reason"], "invalid_heatmap_png")

    def test_unsupported_encoding_does_not_decode(self):
        prepared = self.prepare(item_for([[(1, 0, 255)]], encoding="jpeg;base64"))
        self.assertEqual(prepared.payload["status"], "unavailable")
        self.assertIsNone(prepared.original_png)

    def test_oversized_encoded_and_decoded_payloads_are_rejected(self):
        encoded = "A" * (16 * 1024 * 1024 + 1)
        self.assertIsNone(self.prepare(item_for([[(1, 0, 255)]], data=encoded)).original_png)
        # Base64 of this value fits the encoded limit but exceeds decoded limit.
        raw = b"x" * (12 * 1024 * 1024 + 1)
        prepared = self.prepare(item_for([[(1, 0, 255)]], data=base64.b64encode(raw).decode("ascii")))
        self.assertIsNone(prepared.original_png)
        self.assertEqual(prepared.payload["status"], "unavailable")

    def test_real_png_dimensions_and_pixel_budget_are_checked_before_conversion(self):
        # A correctly checksummed oversized IHDR exercises pre-decompression checks.
        original = png_bytes([[(1, 0, 255)]])
        for width, height in ((2049, 1), (1, 2049), (4096, 4096)):
            with self.subTest(size=(width, height)):
                data = struct.pack(">II", width, height) + original[24:29]
                ihdr = struct.pack(">I", 13) + b"IHDR" + data + struct.pack(">I", zlib.crc32(b"IHDR" + data))
                raw = original[:8] + ihdr + original[33:]
                item = item_for([[(1, 0, 255)]], data=base64.b64encode(raw).decode("ascii"))
                with patch.object(Image.Image, "convert", side_effect=AssertionError("must check dimensions first")):
                    prepared = self.prepare(item)
                self.assertIsNone(prepared.original_png)
                self.assertEqual(prepared.payload["reason"], "image_dimensions_exceed_limit")

    def test_metadata_image_size_mismatch_cannot_claim_crop_coverage(self):
        prepared = self.prepare(item_for([[(1, 0, 255)]], img_width=2))
        self.assertIsNotNone(prepared.original_png)
        self.assertEqual(prepared.payload["status"], "partial")
        self.assertEqual(prepared.payload["reason"], "image_dimensions_mismatch")
        self.assertIsNone(prepared.payload["summary"]["coverage_percent"])

    def test_truecolor_16bit_png_keeps_raw_without_truncated_physical_statistics(self):
        def chunk(kind, data):
            return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
        header = struct.pack('>IIBBBBB', 1, 1, 16, 2, 0, 0, 0)
        raw = (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', header)
               + chunk(b'IDAT', zlib.compress(b'\x00' + struct.pack('>HHH', 0x100, 0, 0x100)))
               + chunk(b'IEND', b''))
        prepared = self.prepare(item_for([[(1, 0, 1)]], data=base64.b64encode(raw).decode('ascii')))
        self.assertEqual(prepared.original_png, raw)
        self.assertIsNone(prepared.thickness_png)
        self.assertEqual(prepared.payload['status'], 'unavailable')


if __name__ == "__main__":
    unittest.main()
