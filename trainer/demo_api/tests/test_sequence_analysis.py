from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from trainer.demo_api.sequence_analysis import SequenceAnalysisEngine
from trainer.demo_api.sequence_store import SequenceStore
from trainer.demo_api.telemetry_contracts import AnalyzeSequenceRequest


def _profile(root: Path) -> None:
    path = root / "generated" / "profiles" / "spray" / "learning" / "mode_03.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "aggregate": {
            "macro_orders": [{"order": [
                {"map_num": 1, "zone_id": "1"},
                {"map_num": 1, "zone_id": "2"},
                {"map_num": 1, "zone_id": "3"},
            ], "session_count": 3, "fraction": 1.0}],
            "zones": {
                "1:1": {"zone_id": "1", "expected_direction": "horizontal"},
                "1:2": {"zone_id": "2", "expected_direction": "vertical"},
                "1:3": {"zone_id": "3", "expected_direction": "horizontal"},
            },
        }
    }), encoding="utf-8")


def _body(sequence_id: str, zone_id: str, *, vertical: bool = False) -> dict[str, Any]:
    points = [[100.0, 100.0], [100.0, 300.0]] if vertical else [[100.0, 100.0], [300.0, 100.0]]
    return {
        "session_id": "3_11000346_1", "user_id": "1234", "scenario_id": "3",
        "zone_id": zone_id, "sequence_id": sequence_id,
        "sequence_setting": {"tool_type": "spray_gun"},
        "sequence_info": [
            {"index": i, "gun_metrics": {"raytexmapnum": 1, "raytexuv": point}}
            for i, point in enumerate(points, start=1)
        ],
        "sequence_result": {},
    }


def _record(store: SequenceStore, body: dict[str, Any]) -> None:
    store.record(
        session_key=body["session_id"], sequence_id=body["sequence_id"],
        body_sha256=body["sequence_id"], frames=len(body["sequence_info"]),
        received_at=f"2026-10-13T00:00:{body['sequence_id'][-3:]}Z",
        body=json.dumps(body),
    )


def test_repaint_in_the_current_zone_is_normal_but_skipping_a_zone_is_anomaly(tmp_path: Path) -> None:
    _profile(tmp_path)
    engine = SequenceAnalysisEngine(tmp_path)
    store = SequenceStore(":memory:")
    first = _body("SEQ-001", "1")
    repaint = _body("SEQ-002", "1")
    skipped = _body("SEQ-003", "3")
    for body in (first, repaint):
        _record(store, body)
        assert engine.evaluate(AnalyzeSequenceRequest.model_validate(body), store).feedback.reason_code is None
    _record(store, skipped)

    decision = engine.evaluate(AnalyzeSequenceRequest.model_validate(skipped), store)

    assert decision.feedback.analysis_status == "available"
    assert decision.feedback.reason_code == "zone_skipped"
    assert decision.anomaly_type == "sequence_order_deviation"
    assert "2번" in decision.text_guide


def test_report_keeps_order_and_direction_scores_separate(tmp_path: Path) -> None:
    _profile(tmp_path)
    engine = SequenceAnalysisEngine(tmp_path)
    bodies = [_body("SEQ-001", "1"), _body("SEQ-002", "2", vertical=True), _body("SEQ-003", "3", vertical=True)]

    report = engine.report_bodies(bodies, scenario_id="3")

    assert report["analysis_status"] == "available"
    assert report["recommended_flow"] == ["1", "2", "3"]
    assert report["first_visit_flow"] == ["1", "2", "3"]
    assert report["adherence_rate"] == 1.0
    assert report["direction_adherence_rate"] == 0.667
    assert report["direction_summary"]["direction_error_count"] == 1


def test_missing_direction_reference_does_not_invent_a_direction_score(tmp_path: Path) -> None:
    _profile(tmp_path)
    path = tmp_path / "generated" / "profiles" / "spray" / "learning" / "mode_03.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    for zone in payload["aggregate"]["zones"].values():
        zone.pop("expected_direction")
    path.write_text(json.dumps(payload), encoding="utf-8")

    report = SequenceAnalysisEngine(tmp_path).report_bodies([_body("SEQ-001", "1")], scenario_id="3")

    assert report["analysis_status"] == "available"
    assert report["direction_adherence_rate"] is None
    assert report["direction_summary"] is None
