"""TSB normal interval and session-start first-Zone contract regressions."""
import numpy as np
import base64
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from trainer.demo_api import coating_assessment as coating
from trainer.demo_api.sequence_store import SequenceStore
from trainer.demo_api.v1_contracts import SessionStartRequest
from trainer.demo_api.v1_router import create_v1_router
from trainer.demo_api.v1_store import SessionConflict, V1Store
from trainer.demo_api.tests.test_zone_completion_gate import analyze, request


@pytest.mark.parametrize("thickness,grade", [(159.9, "below"), (160, "normal"), (180, "normal"),
                                             (200, "normal"), (200.1, "above")])
def test_default_inclusive_range(thickness, grade):
    result = coating.assess_readings([thickness])
    assert result["normal_band_um"] == [160.0, 200.0]
    assert result["grade"] == grade
    assert result["point_grades"] == [grade]


def test_legacy_single_160_keeps_200_normal():
    result = coating.assess_readings([200], 160)
    assert result["grade"] == "normal"
    assert result["gate_status"] == "complete"


def test_wide_explicit_range_bins_never_double_count_upper_boundary():
    result = coating.assess_readings([100, 400, 401, 500], min_thickness_um=100, max_thickness_um=400)
    assert result["point_grades"] == ["normal", "normal", "above", "above"]
    assert sum(result["six_band_counts"].values()) == 4


def test_explicit_range_reaches_pixel_gate_and_summary():
    low, high = 100, 120
    normal = coating.assess_zone(np.full((10, 10), 130.), np.ones((10, 10), bool),
                                  min_thickness_um=low, max_thickness_um=high)
    assert normal["grade"] == "normal"
    assert normal["gate_status"] == "complete"
    assert normal["coverage_percent"] == 100
    assert coating.session_summary([normal])["normal_band_um"] == [100, 120]
    sparse = np.zeros((10, 10))
    sparse[0, :] = 125  # corrected 115, but only 10% of the whole Zone painted
    result = coating.assess_zone(sparse, np.ones((10, 10), bool), min_thickness_um=low, max_thickness_um=high)
    assert result["gate_status"] == "needs_more"
    assert result["coverage_percent"] == 10


def test_overpaint_area_counts_toward_completion_gate():
    raw = np.full((10, 10), 190.)  # corrected180
    raw[:2, :] = 220.  # corrected210; mean186, 20% above the upper bound but still >= the lower bound
    result = coating.assess_zone(raw, np.ones((10, 10), bool))
    assert result["grade"] == "normal"
    assert result["coverage_percent"] == 100
    assert result["normal_band_percent"] == 80
    assert result["gate_status"] == "complete"


def test_progress_api_separates_normal_area_from_target_coverage(tmp_path):
    before = analyze(tmp_path / "before", request(width=4))["coating_progress"]
    after = analyze(tmp_path / "after", request(width=6, valid_count=30, rest=250))["coating_progress"]

    assert before["normal_band_percent"] == 40
    assert before["coverage_percent"] == 40
    assert before["overpaint_percent"] == 0
    assert after["normal_band_percent"] == 30
    assert after["coverage_percent"] == 60
    assert after["overpaint_percent"] == 30
    assert after["next_zone_id"] is None


def test_progress_api_normal_area_is_null_without_heatmap(tmp_path):
    sequence = request()
    sequence.sequence_result.cur_sequence_heatmap = None
    progress = analyze(tmp_path, sequence)["coating_progress"]

    assert progress["available"] is False
    assert progress["normal_band_percent"] is None


def _body(**updates):
    return {"session_id": "3_RANGE_TEST_1", "user_id": "COPILOT-AI-TEST", "scenario_id": "3",
            "started_at": "2026-11-16T00:00:00Z", "worker": {"skill_level": "novice"}, **updates}


@pytest.mark.parametrize("updates", [dict(target_thickness_min_um=160),
    dict(target_thickness_max_um=200), dict(target_thickness_min_um=201, target_thickness_max_um=200)])
def test_invalid_ranges_are_rejected(updates):
    with pytest.raises(ValueError):
        SessionStartRequest.model_validate(_body(**updates))


class Reference:
    def reference_plan(self, *, scenario_id, process):
        assert scenario_id == "3"
        flow = ["8", "3"] if process == "spray" else ["2", "1"]
        return {"flow": flow, "process": process, "reference_session_count": 6}


def _client(store, engine=None):
    app = FastAPI()
    app.include_router(create_v1_router(store=store, sequences=SequenceStore(":memory:"), sequence_engine=engine))
    return TestClient(app)


def test_session_start_persists_explicit_range_and_returns_actual_first_zone():
    store = V1Store(":memory:")
    client = _client(store, Reference())
    result = client.post("/api/v1/sessions/start", json=_body(
        target_thickness_min_um=180, target_thickness_max_um=220)).json()["data"]
    assert (result["target_thickness_min_um"], result["target_thickness_max_um"]) == (180, 220)
    assert result["first_zone_id"] == "8"
    assert result["first_zone_guidance"]["recommended_flow"] == ["8", "3"]
    row = store.get_session("3_RANGE_TEST_1")
    assert coating.session_limits(row) == {"min_thickness_um": 180, "max_thickness_um": 220}
    assert row.target_basis == "session_target"
    ended = client.post("/api/v1/sessions/end", json={
        "session_id": row.session_id, "user_id": row.user_id, "scenario_id": "3",
        "ended_at": "2026-11-16T00:01:00Z"}).json()["data"]
    assert ended["target_thickness_max_um"] == 220


def test_blasting_start_uses_blasting_reference():
    client = _client(V1Store(":memory:"), Reference())
    result = client.post("/api/v1/sessions/start", json=_body(tool_type="blasting_gun")).json()["data"]
    assert result["first_zone_id"] == "2"
    assert result["first_zone_guidance"]["process"] == "blasting"


def test_changing_start_tool_on_retry_is_conflict():
    store = V1Store(":memory:")
    client = _client(store, Reference())
    assert client.post("/api/v1/sessions/start", json=_body()).status_code == 200
    response = client.post("/api/v1/sessions/start", json=_body(tool_type="blasting_gun"))
    assert response.status_code == 409
    assert store.get_session("3_RANGE_TEST_1").tool_type == "spray_gun"


def test_missing_reference_never_invents_first_zone():
    result = _client(V1Store(":memory:")).post("/api/v1/sessions/start", json=_body()).json()["data"]
    assert result["first_zone_id"] is None
    assert result["first_zone_guidance"]["reason_code"] == "sequence_reference_unavailable"
    assert [result["target_thickness_min_um"], result["target_thickness_max_um"]] == [160, 200]


def test_range_change_is_not_an_idempotent_start_retry():
    store = V1Store(":memory:")
    args = {"session_id": "3_RANGE_TEST_1", "user_id": "COPILOT-AI-TEST", "scenario_id": "3",
            "started_at": "2026-11-16T00:00:00Z", "target_thickness_min_um": 160, "target_thickness_max_um": 200}
    store.start_session(**args)
    with pytest.raises(SessionConflict):
        store.start_session(**{**args, "target_thickness_max_um": 220})


def test_realtime_engine_and_cache_use_both_range_limits(tmp_path):
    from trainer.demo_api.tests.test_coating_progress import _profile, _engine, _request
    _profile(tmp_path)
    engine = _engine(tmp_path)
    request = _request(sequence_id="SEQ-001", thickness_um=201)
    normal = engine.evaluate(request, min_thickness_um=160, max_thickness_um=200)
    excess = engine.evaluate(request, min_thickness_um=160, max_thickness_um=180)
    assert normal.status == "complete"
    assert normal.assessment["normal_band_um"] == [160, 200]
    # 하한 이상 면적이 95% 이상이면 칠한 곳 평균이 상한을 넘어도(과도막) 완료다.
    assert excess.status == "complete"
    assert excess.assessment["grade"] == "above"
    assert excess.assessment["normal_band_um"] == [160, 180]


def test_session_report_applies_explicit_upper_bound(tmp_path):
    from trainer.demo_api.tests.test_coating_progress import _profile, _engine, _request
    from trainer.demo_api.session_coating import analyze_session_coating
    _profile(tmp_path)
    engine = _engine(tmp_path)
    body = _request(sequence_id="SEQ-001", thickness_um=201).model_dump(mode="json")
    report = analyze_session_coating(scenario_id="3", bodies=[body], engine=engine,
                                     height_map=base64.b64decode(body["sequence_result"]["cur_sequence_heatmap"][0]["data"]),
                                     min_thickness_um=160, max_thickness_um=180)
    assert report.report["normal_band_um"] == [160, 180]
    assert report.zones[0]["grade"] == "above"
    assert "정상범위 160~180µm" in report.report["guideline"]
