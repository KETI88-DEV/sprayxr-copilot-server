from __future__ import annotations

import base64
import csv
import io
import json
from pathlib import Path

from PIL import Image
import pytest

from trainer.demo_api.coating_progress import CoatingProgressEngine
from trainer.demo_api.telemetry_contracts import AnalyzeSequenceRequest


def _rg_for_um(value: float) -> tuple[int, int, int, int]:
    raw = round(value / 20000.0 * 65025.0)
    return raw // 255, raw % 255, 255, 255


def _heatmap(value: float) -> dict[str, object]:
    image = Image.new("RGBA", (4, 4), _rg_for_um(value))
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return {
        "map_num": 1,
        "x0": 0,
        "y0": 0,
        "width": 4,
        "height": 4,
        "img_width": 4,
        "img_height": 4,
        "format": "v2;thickness=RG16;alpha=B8;scale_um=20000",
        "encoding": "png;base64",
        "data": base64.b64encode(stream.getvalue()).decode("ascii"),
    }


def _engine(root: Path) -> CoatingProgressEngine:
    return CoatingProgressEngine(root, zone_reference_root=root / "fixture_official")


def _profile(root: Path) -> None:
    official = root / "fixture_official"
    (official / "color").mkdir(parents=True)
    (official / "palette.csv").write_text("zone_id,hex,r,g,b\n1,#FF0000,255,0,0\n", encoding="utf-8")
    (official / "scenario_map.csv").write_text("scenario_id,model_key,note\n3,Learn03,test\n", encoding="utf-8")
    (official / "zone_rects.csv").write_text(
        "model_key,map_num,zone_id,rect_x,rect_y,rect_w,rect_h,px_x,px_y,px_w,px_h,u_min,v_min,u_max,v_max,area_percent\n"
        "Learn03,01,1,0,0,1,1,0,0,4,4,0,0,1,1,100\n", encoding="utf-8")
    Image.new("RGB", (4, 4), (255, 0, 0)).save(official / "color/Learn03_01.png")
    target = root / "generated" / "profiles" / "spray" / "learning"
    target.mkdir(parents=True)
    (target / "mode_03.json").write_text(
        json.dumps(
            {
                "aggregate": {
                    "zones": {
                        "1:1": {
                            "map_num": 1,
                            "zone_id": "1",
                            "coating": {
                                "analysis_status": "available",
                                "atlas_width": 4,
                                "atlas_height": 4,
                                "grid_size": 2,
                                "cells": {
                                    "0:0": {"q1_um": 100.0, "median_um": 110.0, "q3_um": 120.0},
                                    "1:0": {"q1_um": 100.0, "median_um": 110.0, "q3_um": 120.0},
                                    "0:1": {"q1_um": 100.0, "median_um": 110.0, "q3_um": 120.0},
                                    "1:1": {"q1_um": 100.0, "median_um": 110.0, "q3_um": 120.0},
                                },
                            },
                        }
                    },
                    "macro_orders": [
                        {
                            "order": [
                                {"map_num": 1, "zone_id": "1"},
                                {"map_num": 1, "zone_id": "2"},
                            ],
                            "session_count": 2,
                            "fraction": 1.0,
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )


def _request(*, sequence_id: str, thickness_um: float, tool_type: str = "spray_gun") -> AnalyzeSequenceRequest:
    return AnalyzeSequenceRequest.model_validate(
        {
            "session_id": "SESSION-1",
            "user_id": "worker",
            "scenario_id": "3",
            "zone_id": "1",
            "sequence_id": sequence_id,
            "sequence_setting": {"tool_type": tool_type},
            "sequence_info": [],
            "sequence_result": {"cur_sequence_heatmap": [_heatmap(thickness_um)]},
        }
    )


def test_cumulative_maps_replace_previous_absolute_thickness_instead_of_adding(tmp_path: Path) -> None:
    """TSB sends the current absolute surface thickness, not a per-sequence delta."""
    _profile(tmp_path)
    engine = _engine(tmp_path)

    first = engine.evaluate(_request(sequence_id="SEQ-001", thickness_um=90.0))
    second = engine.evaluate(_request(sequence_id="SEQ-002", thickness_um=106.0))
    third = engine.evaluate(_request(sequence_id="SEQ-003", thickness_um=171.0))

    assert first.status == "needs_more"
    assert first.recommendation == "continue_current_zone"
    assert first.completion_percent == pytest.approx(50.0, abs=0.2)
    assert second.status == "needs_more"
    assert second.completion_percent == pytest.approx(60.0, abs=0.2)
    assert third.status == "complete"
    assert third.recommendation == "move_next_zone"
    assert third.next_zone_id == "2"
    assert third.completion_percent == pytest.approx(100.0, abs=0.2)


def _subsample_crop(request: AnalyzeSequenceRequest, size: tuple[int, int]) -> None:
    """TSB may send a smaller PNG for an unchanged atlas-space rectangle."""
    crop = request.sequence_result.cur_sequence_heatmap[0]
    with Image.open(io.BytesIO(base64.b64decode(crop.data))) as source:
        image = source.resize(size, Image.Resampling.NEAREST)
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    crop.data = base64.b64encode(stream.getvalue()).decode("ascii")
    crop.img_width, crop.img_height = size


@pytest.mark.parametrize("size", [(4, 4), (2, 2), (1, 1)])
def test_subsampled_png_covers_the_same_declared_atlas_rectangle(tmp_path: Path, size: tuple[int, int]) -> None:
    _profile(tmp_path)
    request = _request(sequence_id="SEQ-001", thickness_um=171.0)
    _subsample_crop(request, size)
    result = _engine(tmp_path).evaluate(request)
    assert result.coverage_percent == 100.0
    assert result.assessment["unpainted_fraction"] == 0.0
    assert result.assessment["thickness_um"] == pytest.approx(161.0, abs=0.2)
    assert result.status == "complete"
    assert result.next_zone_id == "2"


def test_subsampled_png_preserves_validity_when_expanded(tmp_path: Path) -> None:
    _profile(tmp_path)
    request = _request(sequence_id="SEQ-001", thickness_um=171.0)
    image = Image.new("RGBA", (2, 2), _rg_for_um(171.0))
    image.putpixel((1, 0), (255, 255, 0, 255))
    image.putpixel((1, 1), (255, 255, 0, 255))
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    crop = request.sequence_result.cur_sequence_heatmap[0]
    crop.data = base64.b64encode(stream.getvalue()).decode("ascii")
    crop.img_width = crop.img_height = 2
    result = _engine(tmp_path).evaluate(request)
    assert result.coverage_percent == 50.0
    assert result.assessment["unpainted_fraction"] == 0.5
    assert result.overpaint_percent == 0.0


def test_replay_expands_subsampled_partial_crops_before_absolute_merge(tmp_path: Path) -> None:
    _profile(tmp_path)
    first = _request(sequence_id="SEQ-001", thickness_um=171.0)
    second = _request(sequence_id="SEQ-002", thickness_um=171.0)
    for request, x0 in ((first, 0), (second, 2)):
        crop = request.sequence_result.cur_sequence_heatmap[0]
        crop.x0, crop.width = x0, 2
        _subsample_crop(request, (1, 2))
    result = _engine(tmp_path).evaluate_replay(second, [first])
    assert result.status == "complete"
    assert result.coverage_percent == 100.0
    assert result.assessment["thickness_um"] == pytest.approx(161.0, abs=0.2)


def test_oversized_atlas_rectangle_is_clipped_to_bounded_output_before_expansion() -> None:
    crop = _heatmap(171.0)
    crop["width"] = 10**9
    decoded, reason = CoatingProgressEngine._decode_crop(crop, atlas_width=4, atlas_height=4)
    assert reason is None
    assert decoded is not None
    assert decoded.thickness_um.shape == decoded.valid.shape == (4, 4)
    assert decoded.valid.all()
    assert decoded.x_indices.tolist() == decoded.y_indices[::-1].tolist() == [0, 1, 2, 3]


def test_expansion_clips_the_atlas_edge_without_changing_png_origin() -> None:
    request = _request(sequence_id="SEQ-001", thickness_um=171.0)
    _subsample_crop(request, (2, 2))
    crop = request.sequence_result.cur_sequence_heatmap[0].model_dump(mode="json")
    image = Image.new("RGBA", (2, 2), _rg_for_um(500.0))
    image.putpixel((0, 1), _rg_for_um(171.0))  # PNG bottom-left remains atlas bottom-left.
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    crop.update(x0=2, y0=2, data=base64.b64encode(stream.getvalue()).decode("ascii"))
    decoded, reason = CoatingProgressEngine._decode_crop(crop, atlas_width=4, atlas_height=4)
    assert reason is None
    assert decoded is not None
    assert decoded.thickness_um.shape == (2, 2)
    assert decoded.thickness_um.min() == pytest.approx(171.0, abs=0.2)
    assert decoded.thickness_um.max() == pytest.approx(171.0, abs=0.2)
    assert decoded.x_indices.tolist() == [2, 3]
    assert decoded.y_indices.tolist() == [3, 2]


@pytest.mark.parametrize("axis", ["x", "y"])
def test_expanded_cumulative_crop_keeps_neighbor_overpaint_on_its_official_zone(tmp_path: Path, axis: str) -> None:
    from trainer.demo_api.telemetry_router import coating_alerts
    _profile(tmp_path)
    official = tmp_path / "fixture_official"
    (official / "palette.csv").write_text(
        "zone_id,hex,r,g,b\n1,#FF0000,255,0,0\n2,#00FF00,0,255,0\n", encoding="utf-8")
    label = Image.new("RGB", (4, 4), (0, 255, 0))
    for y in range(4):
        for x in range(4):
            if (axis == "x" and x < 2) or (axis == "y" and y >= 2):
                label.putpixel((x, y), (255, 0, 0))
    label.save(official / "color/Learn03_01.png")
    with (official / "zone_rects.csv").open("a", encoding="utf-8") as stream:
        stream.write("Learn03,01,2,0.5,0,0.5,1,2,0,2,4,0.5,0,1,1,50\n")
    profile_path = tmp_path / "generated/profiles/spray/learning/mode_03.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile["aggregate"]["zones"]["1:2"] = {
        **profile["aggregate"]["zones"]["1:1"], "zone_id": "2",
    }
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    image = Image.new("RGBA", (2, 2), _rg_for_um(171.0))
    for point in ([(1, 0), (1, 1)] if axis == "x" else [(0, 0), (1, 0)]):
        image.putpixel(point, _rg_for_um(500.0))
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    engine = _engine(tmp_path)
    for number, zone in ((1, "1"), (2, "2")):
        request = _request(sequence_id=f"SEQ-{number:03d}", thickness_um=171.0)
        request.zone_id = zone
        crop = request.sequence_result.cur_sequence_heatmap[0]
        crop.img_width = crop.img_height = 2
        crop.data = base64.b64encode(stream.getvalue()).decode("ascii")
        result = engine.evaluate(request)
        alerts = coating_alerts(result.assessment)
        assert result.zone_id == result.assessment["zone_id"] == zone
        if zone == "1":
            assert result.status == "complete"
            assert result.overpaint_percent == 0.0
            assert alerts == []
        else:
            assert result.status == "complete"
            assert result.overpaint_percent == 100.0
            assert [(item["anomaly_type"], item["payload"]["zone_id"], item["payload"]["map_num"])
                    for item in alerts] == [("coating_over_app", "2", 1)]


def test_thickness_is_corrected_before_the_progress_gate(tmp_path: Path) -> None:
    """원시 140µm는 보정하면 130µm라 160µm 목표의 진행률이 약 81%다."""
    _profile(tmp_path)
    engine = _engine(tmp_path)

    below = engine.evaluate(_request(sequence_id="SEQ-001", thickness_um=140.0))
    met = engine.evaluate(_request(sequence_id="SEQ-002", thickness_um=171.0))

    assert below.status == "needs_more"
    assert below.coverage_percent == 0
    assert below.completion_percent == pytest.approx(81.25, abs=0.2)
    assert met.status == "complete"


def test_result_carries_the_assessment_for_the_router_but_not_in_json(tmp_path: Path) -> None:
    _profile(tmp_path)

    result = _engine(tmp_path).evaluate(_request(sequence_id="SEQ-001", thickness_um=171.0))

    assert result.assessment is not None
    assert result.assessment["grade"] == "normal"
    assert "assessment" not in result.model_dump(mode="json")


def test_duplicate_sequence_does_not_double_count_the_same_heatmap(tmp_path: Path) -> None:
    """A transport retry returns the original result and does not mutate surface state."""
    _profile(tmp_path)
    engine = _engine(tmp_path)
    request = _request(sequence_id="SEQ-001", thickness_um=106.0)

    first = engine.evaluate(request)
    duplicate = engine.evaluate(request)

    assert first == duplicate
    assert duplicate.status == "needs_more"
    assert duplicate.completion_percent == pytest.approx(60.0, abs=0.2)


def test_overpainted_zone_is_complete_and_tells_the_worker_to_stop_painting(tmp_path: Path) -> None:
    _profile(tmp_path)

    result = _engine(tmp_path).evaluate(
        _request(sequence_id="SEQ-001", thickness_um=251.0)
    )

    assert result.status == "complete"
    assert result.recommendation == "move_next_zone"
    assert result.overpaint_percent == 100.0
    assert "더 칠하지 마세요" in result.message


def test_session_target_changes_compliance_without_changing_expert_reference(tmp_path: Path) -> None:
    _profile(tmp_path)
    request = _request(sequence_id="SEQ-001", thickness_um=110.0)
    expert = _engine(tmp_path).evaluate(request)
    target = _engine(tmp_path).evaluate(request, target_thickness_um=200.0)
    assert expert.status == "needs_more"
    assert expert.target_thickness_um == 160.0
    assert expert.target_basis == "default_target"
    assert expert.expert_completion_percent == 100.0
    assert target.status == "needs_more"
    assert target.comparison_basis == "session_target"
    assert target.target_thickness_um == 200.0
    assert target.target_basis == "session_target"
    assert target.expert_completion_percent == 100.0
    assert target.completion_percent < expert.completion_percent


def test_replay_restores_other_crop_pixels_after_engine_restart(tmp_path: Path) -> None:
    _profile(tmp_path)
    first = _request(sequence_id="SEQ-001", thickness_um=171.0)
    second = _request(sequence_id="SEQ-002", thickness_um=171.0)
    for request, x0 in ((first, 0), (second, 2)):
        crop = request.sequence_result.cur_sequence_heatmap[0]
        crop.x0 = x0
        crop.width = 2
    result = _engine(tmp_path).evaluate_replay(second, [first])
    assert result.status == "complete"
    assert result.coverage_percent == 100.0


def test_replay_cannot_leak_thickness_from_a_different_session(tmp_path: Path) -> None:
    _profile(tmp_path)
    first = _request(sequence_id="SEQ-001", thickness_um=251.0)
    second = _request(sequence_id="SEQ-002", thickness_um=10.0)
    first.session_id = "OTHER-SESSION"
    crop = second.sequence_result.cur_sequence_heatmap[0]
    crop.width = 2
    result = _engine(tmp_path).evaluate_replay(second, [first])
    assert result.status == "needs_more"
    assert result.overpaint_percent == 0.0


def test_warm_replay_applies_telemetry_only_crops_between_analyzed_sequences(tmp_path: Path) -> None:
    _profile(tmp_path)
    engine = _engine(tmp_path)
    first = _request(sequence_id="SEQ-001", thickness_um=10.0)
    second = _request(sequence_id="SEQ-002", thickness_um=171.0)
    third = _request(sequence_id="SEQ-003", thickness_um=171.0)
    first.sequence_result.cur_sequence_heatmap[0].width = 2
    second.sequence_result.cur_sequence_heatmap[0].width = 2
    third.sequence_result.cur_sequence_heatmap[0].x0 = 2
    third.sequence_result.cur_sequence_heatmap[0].width = 2
    engine.evaluate_replay(first, [])
    assert engine.evaluate_replay(third, [first, second]).status == "complete"


def test_corrected_thickness_at_the_target_is_complete_and_not_overpaint(tmp_path: Path) -> None:
    _profile(tmp_path)
    result = _engine(tmp_path).evaluate(
        _request(sequence_id="SEQ-001", thickness_um=110.0), target_thickness_um=100.0,
    )
    assert result.overpaint_percent == 0.0
    assert result.status == "complete"


def test_missing_heatmap_and_blasting_return_honest_unavailable_results(tmp_path: Path) -> None:
    _profile(tmp_path)
    engine = _engine(tmp_path)
    missing = _request(sequence_id="SEQ-001", thickness_um=80.0)
    missing.sequence_result = {}

    assert engine.evaluate(missing).reason == "heatmap_unavailable"
    assert engine.evaluate(
        _request(sequence_id="SEQ-002", thickness_um=80.0, tool_type="blasting_gun")
    ).reason == "coating_not_applicable"


def test_invalid_height_encoding_is_not_silently_scored(tmp_path: Path) -> None:
    _profile(tmp_path)
    request = _request(sequence_id="SEQ-001", thickness_um=80.0)
    request.sequence_result.cur_sequence_heatmap[0].format = "unknown"

    result = _engine(tmp_path).evaluate(request)

    assert result.available is False
    assert result.reason == "heatmap_format_unsupported"


def test_official_zone_mask_excludes_neighbor_zone_thickness(tmp_path: Path) -> None:
    _profile(tmp_path)
    profile_path = tmp_path / "generated" / "profiles" / "spray" / "learning" / "mode_03.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    coating = profile["aggregate"]["zones"]["1:1"]["coating"]
    coating["grid_size"] = 1
    coating["mask_source"] = "tsb_color_label_map"
    coating["cells"] = {
        "0:0": {"q1_um": 100.0, "median_um": 110.0, "q3_um": 120.0}
    }
    profile_path.write_text(json.dumps(profile), encoding="utf-8")

    reference = tmp_path / "zone_reference"
    (reference / "color").mkdir(parents=True)
    with (reference / "palette.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["zone_id", "hex", "r", "g", "b"])
        writer.writerow([1, "#FF0000", 255, 0, 0])
        writer.writerow([2, "#00FF00", 0, 255, 0])
    (reference / "scenario_map.csv").write_text(
        "scenario_id,model_key,note\n3,Learn03,학습\n", encoding="utf-8"
    )
    (reference / "zone_rects.csv").write_text(
        "model_key,map_num,zone_id,rect_x,rect_y,rect_w,rect_h,px_x,px_y,px_w,px_h,u_min,v_min,u_max,v_max,area_percent\n"
        "Learn03,01,1,0,0,1,1,0,0,4,4,0,0,1,1,25\n",
        encoding="utf-8",
    )
    label = Image.new("RGB", (4, 4), (0, 255, 0))
    for index in range(4):
        label.putpixel((index, index), (255, 0, 0))
    label.save(reference / "color" / "Learn03_01.png")

    heatmap = _heatmap(500.0)
    encoded = base64.b64decode(str(heatmap["data"]))
    image = Image.open(io.BytesIO(encoded)).convert("RGBA")
    for index in range(4):
        image.putpixel((index, index), _rg_for_um(111.0))
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    request = _request(sequence_id="SEQ-001", thickness_um=500.0)
    request.sequence_result.cur_sequence_heatmap[0].data = base64.b64encode(
        stream.getvalue()
    ).decode("ascii")

    result = CoatingProgressEngine(
        tmp_path, zone_reference_root=reference
    ).evaluate(request, target_thickness_um=100.0)

    assert result.status == "complete"
    assert result.completion_percent == pytest.approx(100.0, abs=0.2)
    assert result.overpaint_percent == 0.0
