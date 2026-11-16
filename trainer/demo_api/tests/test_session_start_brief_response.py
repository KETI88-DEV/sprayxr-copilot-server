"""Visible first-Zone guidance and concise, evidence-preserving session prose."""

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from trainer.demo_api.sequence_analysis import SequenceAnalysisEngine
from trainer.demo_api.sequence_store import SequenceStore
from trainer.demo_api.v1_router import _briefing, create_v1_router
from trainer.demo_api.v1_store import V1Store


class Reference:
    def reference_plan(self, *, scenario_id, process):
        assert scenario_id == "3"
        return {"flow": ["8", "3"] if process == "spray" else ["2", "1"]}


def _client(engine=None):
    app = FastAPI()
    store = V1Store(":memory:")
    app.include_router(create_v1_router(
        store=store, sequences=SequenceStore(":memory:"), sequence_engine=engine,
    ))
    return TestClient(app), store


def _body(**updates):
    return {
        "session_id": "3_AI_START_BRIEF_1", "user_id": "COPILOT-AI-TEST",
        "scenario_id": "3", "started_at": "2026-11-16T00:00:00Z",
        "worker": {"skill_level": "novice"}, **updates,
    }


@pytest.mark.parametrize("tool_type,zone", [("spray_gun", "8"), ("blasting_gun", "2")])
def test_start_envelope_says_the_measured_first_zone_and_preserves_idempotency(tool_type, zone):
    client, store = _client(Reference())
    body = _body(tool_type=tool_type)
    first = client.post("/api/v1/sessions/start", json=body).json()
    repeated = client.post("/api/v1/sessions/start", json=body).json()
    assert first["status"] == repeated["status"] == "SUCCESS"
    assert first["message"] == repeated["message"] == f"훈련 시작. Zone {zone}에서 시작하세요."
    assert first["data"]["first_zone_id"] == zone
    guidance = first["data"]["first_zone_guidance"]
    assert guidance["zone_id"] == zone
    assert guidance["recommended_flow"][0] == zone
    assert guidance["message"] == f"Zone {zone}에서 시작하세요."
    assert store.get_session(body["session_id"]).tool_type == tool_type


@pytest.mark.parametrize("scenario", ["3", "1-R-A", "2-ABCDE", "Unit12"])
def test_start_without_a_registered_reference_still_succeeds_without_inventing_a_zone(tmp_path, scenario):
    client, _ = _client(SequenceAnalysisEngine(tmp_path))
    result = client.post("/api/v1/sessions/start", json=_body(scenario_id=scenario)).json()
    assert result["status"] == "SUCCESS"
    assert result["data"]["first_zone_id"] is None
    assert result["data"]["first_zone_guidance"]["recommended_flow"] == []
    assert result["data"]["first_zone_guidance"]["reason_code"] == "sequence_reference_unavailable"
    assert result["message"] == "훈련 시작. 등록된 작업 순서 기준이 없습니다."


def test_start_and_end_openapi_expose_typed_first_zone_fields():
    client, _ = _client(Reference())
    schema = client.get("/openapi.json").json()
    for endpoint in ("start", "end"):
        response = schema["paths"][f"/api/v1/sessions/{endpoint}"]["post"]["responses"]["200"]
        model = response["content"]["application/json"]["schema"]["$ref"].rsplit("/", 1)[-1]
        session_data = schema["components"]["schemas"][model]["properties"]["data"]
        assert session_data == {"$ref": "#/components/schemas/SessionStateData"}
    fields = schema["components"]["schemas"]["SessionStateData"]["properties"]
    assert fields["first_zone_id"]["description"]
    guidance = fields["first_zone_guidance"]
    assert {"$ref": "#/components/schemas/FirstZoneGuidance"} in guidance["anyOf"]
    assert "recommended_flow" in schema["components"]["schemas"]["FirstZoneGuidance"]["properties"]


def test_briefing_keeps_metrics_and_priority_action_without_repeating_area_guidelines():
    pose_guide = "자세 평가의 상세 근거와 모든 관절의 차이는 하위 자료를 참고하세요."
    coating_guide = "도막의 모든 구역별 판정과 전문가 비교 결과를 길게 설명합니다."
    result = _briefing(
        pose_report={"analysis_status": "available", "average_similarity_score": .789,
                     "guideline": pose_guide, "priority_corrections": [
                         {"priority": 1, "zone_id": "2", "correction_guidance": "오른손을 11.9cm 올리세요."},
                     ]},
        sequence_report={"analysis_status": "available", "adherence_rate": .784,
                         "deviation_count": 8, "direction_adherence_rate": .75},
        coating_report={"analysis_status": "available", "normal_band_um": [160, 200],
                        "guideline": coating_guide, "worst_zones": [
                            {"zone_id": "3", "status": "under_app", "thickness_um": 89.9,
                             "action": "도포 속도를 10% 낮추세요."},
                        ]},
    )
    assert all(value in result for value in ("78.4%", "8회", "75.0%", "78.9%", "11.9cm", "89.9", "160~200", "10%"))
    assert pose_guide not in result and coating_guide not in result
    assert len(result) < 230


def test_briefing_keeps_coating_unavailability_reason_short_without_echoing_the_guideline():
    result = _briefing(
        pose_report={"analysis_status": "available", "average_similarity_score": .9},
        sequence_report=None,
        coating_report={"analysis_status": "unavailable", "reason_code": "reference_unavailable",
                        "guideline": "이 콘텐츠의 Zone 기준 자료가 없어 도막을 판정하지 못했습니다."},
    )
    assert "90.0%" in result
    assert "도막 미평가(Zone 기준 없음)." in result
    assert len(result) < 65


def test_all_unavailable_report_does_not_invent_a_briefing():
    assert _briefing(pose_report={"analysis_status": "unavailable"}, sequence_report=None) is None


@pytest.mark.parametrize("limits", [
    {"confidence": "limited", "limited_sequence_count": 2, "reason_codes": ["comparison_scope_limited"]},
    {"confidence": "limited"},
])
def test_limited_pose_briefing_preserves_interpretation_limits_and_scored_denominator(limits):
    result = _briefing(
        pose_report={"analysis_status": "available", "average_similarity_score": .8,
                     "evaluated_sequence_count": 3, "priority_corrections": [], **limits},
        sequence_report=None,
    )
    assert "80.0%" in result and "평가 3회" in result
    assert "참고용" in result and "비교 제한" in result
    assert "교정 없음" in result and "정상 보장" in result
    if "limited_sequence_count" in limits:
        assert "제한 2회" in result


def test_pose_briefing_keeps_known_excluded_counts_without_treating_limited_results_as_excluded():
    result = _briefing(
        pose_report={"analysis_status": "available", "average_similarity_score": .8,
                     "evaluated_sequence_count": 3, "confidence": "limited", "limited_sequence_count": 2},
        sequence_report=None, counts={"total": 6, "scored": 3, "not_scored": 2, "no_zone": 1},
    )
    assert "평가 3회" in result and "미판정 2회 제외" in result and "Zone 미지정 1회 제외" in result
    assert "제한 2회" in result


def test_report_example_uses_expert_comparison_terms_separately_from_target_grades():
    sample = json.loads((Path(__file__).parents[1] / "report_sample.json").read_text(encoding="utf-8"))
    guideline = sample["data"]["report"]["coating_report"]["guideline"]
    expert = guideline.split("전문가 대비", 1)[1]
    assert "저도막" not in expert and "과도막" not in expert
    assert "전문가 기준 충족 1개" in expert
    assert "전문가보다 얇은 구역 1개(Zone 2)" in expert
    assert "전문가보다 두꺼운 구역 1개(Zone 3)" in expert
    assert "보정 전" in expert


def test_coating_briefing_keeps_unpainted_zones_even_when_no_painted_zone_is_out_of_band():
    result = _briefing(
        pose_report={"analysis_status": "unavailable"}, sequence_report=None,
        coating_report={"analysis_status": "available", "normal_band_um": [160, 200], "worst_zones": []},
        coating_zones=[
            {"map_num": 1, "zone_id": "1", "grade": "normal", "unpainted_fraction": 0},
            {"map_num": 1, "zone_id": "2", "grade": None, "unpainted_fraction": 1},
            {"map_num": 1, "zone_id": "3", "grade": None, "unpainted_fraction": 1},
        ],
    )
    assert "160~200µm" in result and "미도장 2개" in result
    assert "Zone 2" in result and "Zone 3" in result
    assert "전체 정상" not in result
