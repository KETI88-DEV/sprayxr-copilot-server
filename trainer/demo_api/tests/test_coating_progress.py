from __future__ import annotations

import base64
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


def _profile(root: Path) -> None:
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


def test_current_sequence_maps_accumulate_until_expert_zone_target_is_reached(tmp_path: Path) -> None:
    """Dropping session accumulation would keep the second pass at 60% instead of complete."""
    _profile(tmp_path)
    engine = CoatingProgressEngine(tmp_path)

    first = engine.evaluate(_request(sequence_id="SEQ-001", thickness_um=50.0))
    second = engine.evaluate(_request(sequence_id="SEQ-002", thickness_um=60.0))

    assert first.status == "needs_more"
    assert first.recommendation == "continue_current_zone"
    assert first.completion_percent == pytest.approx(50.0, abs=0.2)
    assert second.status == "complete"
    assert second.recommendation == "move_next_zone"
    assert second.next_zone_id == "2"
    assert second.completion_percent == 100.0


def test_duplicate_sequence_does_not_double_count_the_same_heatmap(tmp_path: Path) -> None:
    """A transport retry must not turn one 60um pass into an apparent 120um completion."""
    _profile(tmp_path)
    engine = CoatingProgressEngine(tmp_path)
    request = _request(sequence_id="SEQ-001", thickness_um=60.0)

    first = engine.evaluate(request)
    duplicate = engine.evaluate(request)

    assert first == duplicate
    assert duplicate.status == "needs_more"
    assert duplicate.completion_percent == pytest.approx(60.0, abs=0.2)


def test_overpainted_zone_tells_the_worker_to_stop(tmp_path: Path) -> None:
    _profile(tmp_path)

    result = CoatingProgressEngine(tmp_path).evaluate(
        _request(sequence_id="SEQ-001", thickness_um=200.0)
    )

    assert result.status == "overpaint_risk"
    assert result.recommendation == "stop_current_zone"
    assert result.overpaint_percent == 100.0


def test_missing_heatmap_and_blasting_return_honest_unavailable_results(tmp_path: Path) -> None:
    _profile(tmp_path)
    engine = CoatingProgressEngine(tmp_path)
    missing = _request(sequence_id="SEQ-001", thickness_um=50.0)
    missing.sequence_result = {}

    assert engine.evaluate(missing).reason == "heatmap_unavailable"
    assert engine.evaluate(
        _request(sequence_id="SEQ-002", thickness_um=50.0, tool_type="blasting")
    ).reason == "coating_not_applicable"


def test_invalid_height_encoding_is_not_silently_scored(tmp_path: Path) -> None:
    _profile(tmp_path)
    request = _request(sequence_id="SEQ-001", thickness_um=50.0)
    request.sequence_result["cur_sequence_heatmap"][0]["format"] = "unknown"

    result = CoatingProgressEngine(tmp_path).evaluate(request)

    assert result.available is False
    assert result.reason == "heatmap_format_unsupported"
