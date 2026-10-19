from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from trainer.demo_api.sequence_analysis import (
    SequenceAnalysisEngine,
    expert_reference_flows,
    score_visit_flow,
)
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

    retried_first = engine.evaluate(AnalyzeSequenceRequest.model_validate(first), store)
    assert retried_first.feedback.reason_code is None
    assert retried_first.anomaly_type is None


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


def test_unresolved_zone_is_not_scored_as_zero_or_guided_as_normal(tmp_path: Path) -> None:
    _profile(tmp_path)
    engine = SequenceAnalysisEngine(tmp_path)
    store = SequenceStore(":memory:")
    body = _body("SEQ-001", "-1")
    _record(store, body)

    decision = engine.evaluate(AnalyzeSequenceRequest.model_validate(body), store)
    report = engine.report_bodies([body], scenario_id="3")

    assert decision.feedback.reason_code == "zone_unresolved"
    assert decision.text_guide is None
    assert report["analysis_status"] == "unavailable"
    assert report["reason_code"] == "sequence_data_unavailable"
    assert "adherence_rate" not in report


def test_off_order_trainee_gets_the_next_zone_by_expert_transitions(tmp_path: Path) -> None:
    """순서 판정은 권장 순서(1→2→3)로 하되, 3부터 칠한 훈련생의 다음 추천은 3에서 전문가가 가장 많이 간 안 칠한 구역.

    깨뜨리는 법: 권장 순서만 쓰면 '2번 먼저'(이미 지나온 순서의 앞)를 권하고 전이(3→1, 5회)를 무시한다.
    """
    _profile(tmp_path)
    path = tmp_path / "generated" / "profiles" / "spray" / "learning" / "mode_03.json"
    profile = json.loads(path.read_text(encoding="utf-8"))
    profile["aggregate"]["transitions"] = [
        {"from": {"map_num": 1, "zone_id": "3"}, "to": {"map_num": 1, "zone_id": "1"}, "session_count": 5},
        {"from": {"map_num": 1, "zone_id": "3"}, "to": {"map_num": 1, "zone_id": "2"}, "session_count": 2},
    ]
    path.write_text(json.dumps(profile), encoding="utf-8")
    engine = SequenceAnalysisEngine(tmp_path)
    store = SequenceStore(":memory:")
    first = _body("SEQ-001", "3")
    _record(store, first)

    decision = engine.evaluate(AnalyzeSequenceRequest.model_validate(first), store)

    assert decision.feedback.reason_code is not None  # 판정은 여전히 권장 순서 이탈
    assert decision.text_guide == "권장 순서에서 이탈했습니다. 지금 위치에서는 다음으로 1번 구역을 권합니다."


def test_reference_flows_ignore_legacy_all_worker_override_and_expand_sessions() -> None:
    profile = {
        "aggregate": {
            "macro_orders": [
                {"source": "expert_zone_orders", "order": [{"zone_id": "9"}, {"zone_id": "8"}], "session_count": 99},
                {"order": [{"zone_id": "1"}, {"zone_id": "2"}, {"zone_id": "3"}], "session_count": 3},
                {"order": [{"zone_id": "1"}, {"zone_id": "3"}, {"zone_id": "2"}], "session_count": 2},
                {"order": [{"zone_id": "3"}, {"zone_id": "1"}, {"zone_id": "2"}], "session_count": 1},
            ]
        }
    }

    flows = expert_reference_flows(profile)

    assert len(flows) == 6
    assert flows.count(["1", "2", "3"]) == 3
    assert flows.count(["1", "3", "2"]) == 2
    assert flows.count(["3", "1", "2"]) == 1
    assert ["9", "8"] not in flows


def test_exact_minority_expert_run_remains_a_high_similarity_reference() -> None:
    references = [["1", "2", "3"]] * 3 + [["1", "3", "2"]] * 2 + [["3", "1", "2"]]

    score = score_visit_flow(["3", "1", "2"], references)

    assert score["best_reference_similarity"] == 1.0
    assert score["position_similarity"] == 0.511
    assert score["adherence_rate"] == 0.853
    assert score["position_distribution"][0] == [
        {"zone_id": "1", "count": 5, "probability": 0.833},
        {"zone_id": "3", "count": 1, "probability": 0.167},
    ]


def test_report_scores_an_alternate_expert_order_instead_of_one_rigid_flow(tmp_path: Path) -> None:
    _profile(tmp_path)
    path = tmp_path / "generated" / "profiles" / "spray" / "learning" / "mode_03.json"
    profile = json.loads(path.read_text(encoding="utf-8"))
    profile["aggregate"]["macro_orders"] = [
        {"order": [{"zone_id": "1"}, {"zone_id": "2"}, {"zone_id": "3"}], "session_count": 3},
        {"order": [{"zone_id": "1"}, {"zone_id": "3"}, {"zone_id": "2"}], "session_count": 3},
    ]
    path.write_text(json.dumps(profile), encoding="utf-8")

    report = SequenceAnalysisEngine(tmp_path).report_bodies(
        [_body("SEQ-001", "1"), _body("SEQ-002", "3"), _body("SEQ-003", "2")],
        scenario_id="3",
    )

    assert report["reference_session_count"] == 6
    assert report["best_reference_similarity"] == 1.0
    assert report["adherence_rate"] == 1.0
    assert "6회" in report["guideline"]
