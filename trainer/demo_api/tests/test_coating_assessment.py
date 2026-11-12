from __future__ import annotations

import base64
import hashlib
import io
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from trainer.demo_api import coating_assessment as ca
from trainer.zone_engine import numerical
from trainer.zone_engine.contracts import HeightPng


def _png(raw_um: np.ndarray, blue: np.ndarray) -> bytes:
    """원시 두께(µm)와 B 채널로 TSB 높이맵 PNG를 만든다."""
    code = np.rint(np.asarray(raw_um, dtype=np.float64) / 20000.0 * 65025.0).astype(np.int64)
    pixels = np.stack([code // 255, code % 255, blue, np.full(code.shape, 255)], axis=-1).astype(np.uint8)
    stream = io.BytesIO()
    Image.fromarray(pixels, "RGBA").save(stream, format="PNG")
    return stream.getvalue()


def _zone(raw_um: float, *, count: int = 16, unpainted: int = 0, **kwargs):
    surface = np.full((1, count + unpainted), raw_um)
    surface[0, count:] = 0.0
    return ca.assess_zone(surface, np.ones(surface.shape, bool), **kwargs)


def test_matches_the_engine_height_png_path() -> None:
    rng = np.random.default_rng(20261112)
    raw_um = rng.uniform(0.0, 600.0, (64, 64))
    blue = np.where(rng.random((64, 64)) < 0.2, 0, 255)
    yy, xx = np.mgrid[:64, :64]
    mask = ((yy - 33) ** 2 + (xx - 29) ** 2 < 380) & (rng.random((64, 64)) > 0.1)
    png = _png(raw_um, blue)

    mine = ca.assess_zone(ca.painted_surface_from_png(png), mask, 160.0)

    spec = HeightPng.model_validate({
        "data_base64": base64.b64encode(png).decode("ascii"), "sha256": hashlib.sha256(png).hexdigest(),
        "encoding": "rg16_b8", "basis": "current_round", "width": 64, "height": 64, "origin": "top_left",
        "crop": {"x0": 0, "y0": 0, "width": 64, "height": 64, "atlas_width": 64, "atlas_height": 64}})
    theirs = numerical.assess_zone(numerical.place_crop(numerical.decode_height_png(spec), spec), mask, 160.0)["quality"]

    assert mine is not None and theirs["grade"] is not None and 0.0 < theirs["unpainted_fraction"] < 1.0
    for key in ("grade", "painted_mean_um", "painted_pixel_count", "zone_pixel_count",
                "unpainted_fraction", "six_band_counts"):
        assert mine[key] == theirs[key], key


def test_correction_follows_the_author_oracle() -> None:
    empty = _zone(10.5, count=1)
    assert (empty["grade"], empty["status"], empty["painted_mean_um"]) == (None, None, None)
    assert empty["reason_code"] == "no_valid_painted_pixels"
    assert (empty["unpainted_fraction"], empty["gate_status"]) == (1.0, "needs_more")
    for raw_um, mean_um, grade in [(170.0, 160.0, "normal"), (250.0, 240.0, "above"), (1700.0, 1600.0, "above")]:
        result = _zone(raw_um, count=1)
        assert (result["painted_mean_um"], result["grade"]) == (mean_um, grade)
    assert _zone(169.9, count=1)["grade"] == "below"


def test_missing_target_uses_160_and_says_so() -> None:
    default = _zone(171.0)
    assert (default["target_thickness_um"], default["target_basis"]) == (160.0, "default_target")
    assert (default["grade"], default["status"], default["painted_mean_um"]) == ("normal", "normal", 161.0)
    assert (default["unpainted_fraction"], default["coverage_percent"], default["gate_status"]) == (0.0, 100.0, "complete")
    assert default["six_band_counts"]["from_1_0t_to_lt_1_5t"] == 16
    assert default["normal_band_um"] == [160.0, 240.0]

    explicit = _zone(171.0, target_um=200.0)
    assert (explicit["target_thickness_um"], explicit["target_basis"]) == (200.0, "session_target")
    assert (explicit["grade"], explicit["status"]) == ("below", "under_app")
    assert (explicit["coverage_percent"], explicit["gate_status"]) == (0.0, "needs_more")
    assert ca.resolve_target(None) == (160.0, "default_target")


def test_gate_is_area_while_grade_is_mean() -> None:
    short = _zone(171.0, count=94, unpainted=6)
    assert (short["grade"], short["coverage_percent"], short["gate_status"]) == ("normal", 94.0, "needs_more")
    assert short["completion_percent"] == 94.0
    assert _zone(171.0, count=95, unpainted=5)["gate_status"] == "complete"
    thick = _zone(251.0, count=100)
    assert (thick["grade"], thick["coverage_percent"], thick["overpaint_percent"]) == ("above", 100.0, 100.0)
    assert thick["gate_status"] == "overpaint_risk"
    # 목표의 절반만 쌓인 면: 채운 면적은 0%, 진행률은 50%
    half = _zone(90.0)
    assert (half["coverage_percent"], half["completion_percent"]) == (0.0, 50.0)


def test_old_png_without_blue_channel_is_not_all_unpainted() -> None:
    raw_um = np.full((4, 4), 200.0)
    old = ca.painted_surface_from_png(_png(raw_um, np.zeros((4, 4), int)))
    assert old.shape == (4, 4) and (old > 0).all()

    blue = np.full((4, 4), 255)
    blue[0, :] = 0
    mixed = ca.painted_surface_from_png(_png(raw_um, blue))
    assert (mixed[0] == 0).all() and (mixed[1:] > 0).all()


def test_unreadable_height_png_raises_value_error() -> None:
    gray = io.BytesIO()
    Image.new("L", (4, 4), 9).save(gray, format="PNG")
    jpeg = io.BytesIO()
    Image.new("RGB", (4, 4), (9, 9, 9)).save(jpeg, format="JPEG")
    for raw in (b"not a png", gray.getvalue(), jpeg.getvalue()):
        with pytest.raises(ValueError):
            ca.painted_surface_from_png(raw)


def test_cropping_to_the_zone_box_keeps_the_numbers() -> None:
    rng = np.random.default_rng(7)
    surface = rng.uniform(0.0, 500.0, (9, 13))
    mask = rng.random((9, 13)) > 0.3
    big_surface = rng.uniform(0.0, 500.0, (40, 50))
    big_mask = np.zeros((40, 50), bool)
    big_surface[11:20, 17:30] = surface
    big_mask[11:20, 17:30] = mask
    assert ca.assess_zone(big_surface, big_mask, 160.0) == ca.assess_zone(surface, mask, 160.0)


def test_session_mean_is_weighted_by_painted_area() -> None:
    summary = ca.session_summary([_zone(110.0, count=10), _zone(210.0, count=30), _zone(0.0, count=10)])
    assert summary["painted_mean_um"] == 175.0
    assert summary["grade_counts"] == {"below": 1, "normal": 1, "above": 0, "none": 1}
    assert (summary["zone_count"], summary["painted_pixel_count"], summary["zone_pixel_count"]) == (3, 40, 50)
    assert summary["unpainted_fraction"] == 0.2
    assert sum(summary["six_band_counts"].values()) == 40
    assert (summary["target_thickness_um"], summary["target_basis"]) == (160.0, "default_target")
    assert ca.session_summary([])["painted_mean_um"] is None


def test_adjustments_follow_the_grade_and_need_samples() -> None:
    kind = "quality_rule_candidate"
    below = _zone(100.0, speed_mm_s=[400.0])
    assert below["adjustments"] == [
        {"variable": "speed_mm_s", "direction": "decrease", "priority": 1, "evidence_kind": kind}]
    assert below["adjustment_reason"] is None
    above = _zone(300.0, speed_mm_s=[400.0], distance_mm=[260.0])
    assert above["adjustments"] == [
        {"variable": "speed_mm_s", "direction": "increase", "priority": 1, "evidence_kind": kind},
        {"variable": "distance_mm", "direction": "increase", "priority": 2, "evidence_kind": kind}]
    normal = _zone(171.0, speed_mm_s=[400.0])
    assert (normal["adjustments"], normal["adjustment_reason"]) == ([], "normal_grade_no_corrective_candidate")
    missing = _zone(100.0)
    assert (missing["adjustments"], missing["adjustment_reason"]) == ([], "behavior_evidence_missing")
    assert _zone(0.0, speed_mm_s=[400.0])["adjustment_reason"] == "no_valid_painted_pixels"


def test_display_edges_match_the_engine() -> None:
    assert len(ca.BAND_LABELS) == len(numerical.SIX_BAND_KEYS)
    assert ca.normal_band_um(160.0) == [160.0, 240.0]
    assert ca.band_edges_um(160.0) == [144.0, 160.0, 240.0, 320.0, 400.0]
    for target in (160.0, 200.0, 137.3):
        band = ca.normal_band_um(target)
        assert numerical.grade_for_mean(band[0], target) == "normal"
        assert numerical.grade_for_mean(np.nextafter(band[0], 0), target) == "below"
        assert numerical.grade_for_mean(np.nextafter(band[1], 0), target) == "normal"
        assert numerical.grade_for_mean(band[1], target) == "above"
        for index, edge in enumerate(ca.band_edges_um(target)):
            at = numerical.six_band_counts(np.array([edge]), target)
            under = numerical.six_band_counts(np.array([np.nextafter(edge, 0)]), target)
            assert at[numerical.SIX_BAND_KEYS[index + 1]] == 1
            assert under[numerical.SIX_BAND_KEYS[index]] == 1


def test_session_target_reads_the_stored_basis() -> None:
    assert ca.session_target(None) is None
    assert ca.session_target(SimpleNamespace(target_thickness_um=160.0, target_basis="default_target")) is None
    assert ca.session_target(SimpleNamespace(target_thickness_um=160.0, target_basis="session_target")) == 160.0
    assert ca.session_target(SimpleNamespace(target_thickness_um=160.0)) == 160.0


def test_trainee_sentences_use_no_internal_words() -> None:
    rows = [ca.zone_row("1", "3", assessment) for assessment in (
        _zone(108.0, count=22, unpainted=3, speed_mm_s=[400.0], distance_mm=[260.0]),
        _zone(181.0), _zone(311.0, count=97, unpainted=3), _zone(0.0))]
    assert set(rows[0]) == {
        "zone_id", "map_num", "status", "grade", "thickness_um", "target_thickness_um", "target_basis",
        "normal_band_um", "unpainted_fraction", "six_band_counts", "coverage_percent", "overpaint_percent",
        "completion_percent", "gate_status", "adjustments", "adjustment_reason"}
    assert (rows[0]["map_num"], rows[0]["zone_id"], rows[0]["thickness_um"]) == (1, "3", 98.0)
    assert (rows[3]["thickness_um"], rows[3]["status"], rows[3]["grade"]) == (None, None, None)

    phrases = [ca.grade_phrase(row) for row in rows]
    assert phrases == ["저도막(칠한 곳 평균 98µm, 미도장 12%)", "정상(칠한 곳 평균 171µm, 미도장 0%)",
                       "과도막(칠한 곳 평균 301µm, 미도장 3%)", "칠한 곳 없음(미도장 100%)"]
    adjustment = ca.adjustment_sentence(rows[0]["adjustments"])
    assert adjustment == "조절 후보: 이동 속도를 낮추는 쪽, 이격거리를 줄이는 쪽. 원인으로 확정된 것은 아닙니다."
    assert ca.adjustment_sentence([]) is None
    distribution = ca.distribution_sentence(
        dict(zip(numerical.SIX_BAND_KEYS, (10, 10, 50, 20, 5, 5), strict=True)))
    assert distribution == ("칠한 면적의 두께 분포: 목표의 90% 미만 10%, 90~100% 10%, 100~150% 50%, "
                            "150~200% 20%, 200~250% 5%, 250% 이상 5%.")
    assert ca.distribution_sentence(rows[3]["six_band_counts"]) is None
    assert ca.basis_label(160.0, "default_target") == "기본 목표 160µm 기준"
    assert ca.basis_label(200.0, "session_target") == "세션 목표 200µm 기준"
    assert ca.OVER_PHRASE == "목표의 1.5배 이상"
    assert ca.RULE_NOTE == "등급은 칠한 곳의 평균 두께로 정합니다(목표 미만은 저도막, 목표의 1.5배 이상은 과도막)."
    assert ca.GRADE_SHORT_LABELS == {"below": "목표 미만", "normal": "정상", "above": "1.5배 이상", "none": "미도장"}

    for text in (*phrases, adjustment, distribution, ca.RULE_NOTE, ca.basis_label(160.0, "default_target")):
        assert not any(word in text for word in ("픽셀", "화소", "마스크", "revision"))


def test_mean_next_to_the_target_is_not_written_as_the_target() -> None:
    """평균 159.6을 '160µm'로 적으면 목표 160에 '저도막'이 모순으로 읽힌다. 그때만 소수 한 자리로 적는다."""
    row = ca.zone_row("1", "3", _zone(169.6))
    assert (row["grade"], row["thickness_um"]) == ("below", 159.6)
    assert ca.grade_phrase(row) == "저도막(칠한 곳 평균 159.6µm, 미도장 0%)"
    assert [ca.mean_text(mean, 160.0) for mean in (98.0, 159.4, 160.0, 160.3, 240.2)] == ["98", "159", "160", "160.3", "240"]


def test_mismatched_or_empty_mask_returns_none() -> None:
    surface = np.full((4, 4), 171.0)
    assert ca.assess_zone(surface, np.ones((4, 5), bool)) is None
    assert ca.assess_zone(surface, np.zeros((4, 4), bool)) is None


def test_behavior_by_zone_collects_sequence_averages() -> None:
    bodies = [
        {"zone_id": "3", "sequence_result": {"avg_speed_mms": 450, "avg_distance_mm": 260.0}},
        {"zone_id": 3, "sequence_result": {"avg_speed_mms": 410.0, "avg_distance_mm": None}},
        {"zone_id": "4", "sequence_result": {"avg_speed_mms": float("nan"), "avg_distance_mm": -1.0}},
        {"zone_id": "-1", "sequence_result": {"avg_speed_mms": 300.0}},
        {"zone_id": "0", "sequence_result": {"avg_speed_mms": 300.0}},
        {"zone_id": "", "sequence_result": {"avg_speed_mms": 300.0}},
        {"zone_id": "5"},
    ]
    samples = ca.behavior_by_zone(bodies)
    assert samples["3"] == {"speed_mm_s": [450.0, 410.0], "distance_mm": [260.0]}
    assert not {"-1", "0", ""} & set(samples)
    assert not any(samples.get(zone, {}).get(name) for zone in ("4", "5") for name in ("speed_mm_s", "distance_mm"))
