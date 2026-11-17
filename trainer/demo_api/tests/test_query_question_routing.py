"""Questions retain their meaning across tabs and detailed formatting."""
import json

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from trainer.demo_api.copilot_router import create_copilot_router
from trainer.demo_api.sequence_store import SequenceStore
from trainer.demo_api.tests.document_evidence import supported_document_result
from trainer.demo_api.viewer_context import sign_context


class Pipeline:
    def __init__(self):
        self.calls = []
    def classify_copilot_intent(self, task, prompt):
        return "order_reference"  # Reproduces the real classifier's incorrect choice.
    def run(self, **kwargs):
        self.calls.append(kwargs)
        return {"response": "[답변] 권장 작업 순서는 Zone 1 → 2입니다.", "rag_chunks": []}


def target_assessment(**overrides):
    """서명 전 목표 대비 평가(T6 모양). Zone 1 저도막(속도 표본 있음), Zone 2 과도막."""
    from trainer.demo_api import coating_assessment as ca
    mask = np.ones((4, 4), bool)
    rows = [ca.zone_row(1, "1", ca.assess_zone(np.full((4, 4), 90.0), mask, speed_mm_s=[400.0])),
            ca.zone_row(1, "2", ca.assess_zone(np.full((4, 4), 300.0), mask))]
    return {"analysis_status": "available", "reason_code": None, "target_thickness_um": 160.0,
            "target_basis": "default_target", "basis_label": ca.basis_label(160.0, "default_target"),
            "zones": rows, **overrides}


def context(task="trajectory_recomm", coating=True, target=None, expert=True):
    analysis = {"recommended_flow": ["1", "2"], "selected_flow": ["1"],
                "adherence_rate": 0.8, "reference_session_count": 6,
                "direction_summary": {"classification": "horizontal"}, "zone_keys": ["1:1", "1:2"]}
    if task == "pose_recomm":
        analysis = {"status": "available", "similarity_percent": 90,
                    "zones": {"1:1": {"zone_id": "1", "similarity_percent": 90}}, "corrections": []}
    payload = {"analysis_context": analysis, "signature": sign_context(task, analysis),
               "playback_context": {"playhead_seconds": 1, "source_index": 0, "map_num": "1", "zone_id": "1"}}
    if coating:
        claims = {"analysis_status": "available", "scope": "final_record", "comparison_basis": "expert_zone_median",
                  "zones": [{"map_num": "1", "zone_id": "1", "actual_um": 180, "reference_um": 120},
                            {"map_num": "1", "zone_id": "2", "actual_um": 80, "reference_um": 120}] if expert else []}
        if target is not None:
            claims["target"] = target
        payload.update(coating_context=claims, coating_signature=sign_context("coating_status", claims))
    return "motion_viewer_analysis:" + json.dumps(payload)


def client(pipeline):
    app = FastAPI()
    app.include_router(create_copilot_router(lambda: pipeline, store=SequenceStore(":memory:")))
    return TestClient(app)


@pytest.mark.parametrize("task", ["knowledge_search", "pose_recomm", "trajectory_recomm"])
def test_posture_types_are_explained_instead_of_scoring_current_pose(task):
    class Knowledge(Pipeline):
        def plan_copilot_dialogue(self, *args, **kwargs):
            raise AssertionError("A standalone taxonomy question does not need dialogue classification")
        def run(self, **kwargs):
            self.calls.append(kwargs)
            return supported_document_result("[답변] 도장 작업 자세는 선 자세, 몸을 낮춘 자세, 쪼그려 앉은 자세 등으로 나눌 수 있습니다.")
    pipeline = Knowledge()
    response = client(pipeline).post("/api/v1/copilot/query", json={
        "user_id": "routing-test", "task_type": task,
        "prompt": "자세 종류로는 어떤게있어?", "context": context("pose_recomm")})
    assert response.status_code == 200
    data = response.json()["data"]
    assert "선 자세" in data["answer"]
    assert "유사도" not in data["answer"]
    assert data["retrieval_used"] is True
    assert len(pipeline.calls) == 1


@pytest.mark.parametrize("task", ["knowledge_search", "pose_recomm", "trajectory_recomm"])
@pytest.mark.parametrize("mode", ["quick", "detailed"])
def test_coating_question_answers_signed_coating_values_in_every_tab(task, mode):
    pipeline = Pipeline()
    response = client(pipeline).post("/api/v1/copilot/query", json={
        "user_id": "routing-test", "task_type": task, "response_mode": mode,
        "prompt": "도막이 과하거나 모자란 영역이 있어?", "context": context(task)})
    assert response.status_code == 200
    data = response.json()["data"]
    assert "두꺼운" in data["answer"] and "얇은" in data["answer"]
    assert "최종" in data["answer"]
    assert "180" in data["answer"] and "80" in data["answer"] and "120" in data["answer"]
    assert "권장 Zone 순서" not in data["answer"] and "1 → 2" not in data["answer"]
    assert data["coating_context"]["scope"] == "final_record"
    assert data["retrieval_used"] is False and pipeline.calls == []


@pytest.mark.parametrize("tamper", [False, True])
def test_missing_or_forged_coating_values_do_not_become_order_or_normal_answers(tamper):
    value = context(coating=tamper)
    if tamper:
        value = value.replace('"actual_um": 180', '"actual_um": 9999')
    data = client(Pipeline()).post("/api/v1/copilot/query", json={
        "user_id": "routing-test", "task_type": "trajectory_recomm", "response_mode": "detailed",
        "prompt": "도막 두께가 이상한 존이 있어?", "context": value}).json()["data"]
    assert data["coating_context"]["analysis_status"] == "not_available"
    assert "판단" in data["answer"] and "도막" in data["answer"]
    assert "권장" not in data["answer"] and "9999" not in data["answer"]


@pytest.mark.parametrize("question,expected", [("누락한 Zone은?", "칠하지 않은"), ("실제 수행 순서는?", "칠한 순서"),
                                               ("순서 준수율은?", "유사도"), ("도포 방향은?", "방향")])
def test_detailed_order_answer_preserves_the_specific_question(question, expected):
    data = client(Pipeline()).post("/api/v1/copilot/query", json={
        "user_id": "routing-test", "task_type": "trajectory_recomm", "response_mode": "detailed",
        "prompt": question, "context": context()}).json()["data"]
    assert expected in data["answer"]
    assert "[권장 Zone 순서]" not in data["answer"]
    if "누락" in question:
        assert "2번" in data["answer"] and "모든 구역을 칠했습니다" not in data["answer"]



def test_viewer_coating_uses_only_available_physical_thickness_metrics():
    from trainer.demo_api.coating_query import viewer_coating_context, report_coating_context
    payload = {"pose_evaluation": {"zones": {
        "1:1": {"map_num": 1, "zone_id": "1", "metrics": [{"key": "coating_thickness", "available": True, "actual": 0, "reference": 120}]},
        "1:2": {"map_num": 1, "zone_id": "2", "metrics": [{"key": "coating_thickness", "available": False, "actual": 150, "reference": 120}]},
        "2:1": {"map_num": 2, "zone_id": "1", "metrics": [{"key": "coating_thickness", "available": True, "actual": 180, "reference": 120}]},
    }}}
    result = viewer_coating_context(payload)
    assert {(z["map_num"], z["zone_id"]) for z in result["zones"]} == {("1", "1"), ("2", "1")}
    assert result["zones"][0]["actual_um"] == 0
    assert report_coating_context(("R", {"coating_zones": [{"zone_id": "1", "status": "unavailable"}]}))["analysis_status"] == "not_available"



def test_equal_coating_values_are_reported_as_measurements():
    from trainer.demo_api.coating_query import answer_coating_question
    text = answer_coating_question({"analysis_status": "available", "comparison_basis": "expert_zone_median",
        "zones": [{"map_num": "1", "zone_id": "1", "actual_um": 120, "reference_um": 120}]}, "Zone 1 도막 두께는?", detailed=True)
    assert "120µm" in text and "같은" in text


def test_coating_selection_respects_map_number_when_zone_ids_repeat():
    from trainer.demo_api.coating_query import answer_coating_question
    text = answer_coating_question({"analysis_status": "available", "comparison_basis": "expert_zone_median",
        "zones": [{"map_num": "1", "zone_id": "1", "actual_um": 180, "reference_um": 120},
                  {"map_num": "2", "zone_id": "1", "actual_um": 80, "reference_um": 120}]}, "Map 2 Zone 1 도막 두께는?", detailed=True)
    assert "Map 2" in text and "80µm" in text and "180µm" not in text



@pytest.mark.parametrize("task", ["pose_recomm", "trajectory_recomm"])
@pytest.mark.parametrize("mode", ["quick", "detailed"])
def test_coating_how_to_question_uses_rag_and_verified_data_instead_of_fixed_status(task, mode):
    class Advice(Pipeline):
        def run(self, **kwargs):
            self.calls.append(kwargs)
            return supported_document_result("[답변] 균일한 도막을 위해 분사 거리와 이동 속도를 일정하게 유지하세요.")
    pipeline = Advice()
    response = client(pipeline).post("/api/v1/copilot/query", json={
        "user_id": "routing-test", "task_type": task, "response_mode": mode,
        "prompt": "이 Zone에서 도막을 균일하게 만드는 방법은?", "context": context(task)})
    data = response.json()["data"]
    assert response.status_code == 200 and data["retrieval_used"] is True
    assert "일정하게" in data["answer"] and "[최종 도막 두께 비교]" not in data["answer"]
    assert data["coating_context"]["playback_context"] == {"map_num": "1", "zone_id": "1"}
    assert len(pipeline.calls) == 1
    assert "검증된 최종 도막 비교 자료" in pipeline.calls[0]["instruction"]
    assert "180" in pipeline.calls[0]["instruction"] and "120" in pipeline.calls[0]["instruction"]


def test_pose_tab_fallback_does_not_claim_a_missing_height_map():
    from trainer.demo_api.coating_query import viewer_coating_context
    payload = {"pose_evaluation": {"zones": {
        "1:1": {"map_num": 1, "zone_id": "1", "metrics": [{"key": "coating_thickness", "available": True, "actual": 150, "reference": 120}]}}}}
    result = viewer_coating_context(payload)
    assert result["target"] == {"analysis_status": "not_available", "reason_code": "target_assessment_not_loaded"}
    assert result["analysis_status"] == "available" and result["zones"]


def ask(task, prompt, value, *, mode="quick", pipeline=None, reports=None):
    app = FastAPI()
    app.include_router(create_copilot_router(lambda: pipeline or Pipeline(), store=SequenceStore(":memory:"), reports=reports))
    response = TestClient(app).post("/api/v1/copilot/query", json={
        "user_id": "routing-test", "session_id": "S-1", "task_type": task, "response_mode": mode,
        "prompt": prompt, "context": value})
    assert response.status_code == 200, response.text
    return response.json()["data"]


@pytest.mark.parametrize("task", ["knowledge_search", "pose_recomm", "trajectory_recomm"])
@pytest.mark.parametrize("mode", ["quick", "detailed"])
def test_signed_target_grades_answer_the_coating_question_in_every_tab(task, mode):
    pipeline = Pipeline()
    data = ask(task, "도막이 과하거나 모자란 영역이 있어?", context(task, target=target_assessment()), mode=mode, pipeline=pipeline)
    answer = data["answer"]
    assert pipeline.calls == [] and data["retrieval_used"] is False
    assert "저도막: 1개" in answer and "과도막: 1개" in answer and "기본 목표 160µm 기준" in answer
    assert "전문가 대비" in answer and "최종 높이맵 기준(재생 시점 아님)" in answer
    assert "두꺼운 영역" not in answer and "[" not in answer and "픽셀" not in answer
    assert pipeline.calls == [] and data["retrieval_used"] is False


def test_zone_question_gets_the_one_signed_target_row():
    answer = ask("knowledge_search", "Zone 2 도막은 어때?", context(target=target_assessment()), mode="detailed")["answer"]
    assert "Zone 2: 과도막(칠한 곳 평균 290µm" in answer and "Zone 1" not in answer
    quick = ask("knowledge_search", "Zone 2 도막은 어때?", context(target=target_assessment()))["answer"]
    assert "Zone 2: 과도막(칠한 곳 평균 290µm" in quick and "Zone 1" not in quick


def test_question_about_the_expert_keeps_the_expert_comparison():
    # "두꺼운"만으로는 도막 질문으로 판별되지 않는다(copilot_intent). 도막을 말한 질문으로 잰다.
    answer = ask("knowledge_search", "전문가보다 도막이 두꺼운 곳은 어디야?", context(target=target_assessment()), mode="detailed")["answer"]
    assert "전문가 기준보다 두꺼운 영역" in answer and "저도막: 1개" not in answer
    assert not answer.startswith("이 질문에는")


@pytest.mark.parametrize("target,expected", [
    ({"analysis_status": "not_available", "reason_code": "height_map_unavailable"}, "높이맵 원본 없음: 목표 대비 판정 불가."),
    ({"analysis_status": "not_available", "reason_code": "height_map_size_mismatch"}, "높이맵·Zone 기준 크기 불일치: 목표 대비 판정 불가."),
    ({"analysis_status": "not_available", "reason_code": "zone_reference_unavailable"}, "Zone 기준 없음: 목표 대비 판정 불가."),
    ({"analysis_status": "not_available", "reason_code": "target_assessment_not_loaded"}, "목표 대비 판정 자료가 없습니다."),
    (None, "목표 대비 판정 자료가 없습니다."),
])
def test_record_without_target_says_so_before_the_expert_comparison(target, expected):
    answer = ask("knowledge_search", "도막이 과하거나 모자란 영역이 있어?", context(target=target), mode="detailed")["answer"]
    quick = ask("knowledge_search", "도막이 과하거나 모자란 영역이 있어?", context(target=target))["answer"]
    assert quick.startswith(expected) and "아래는 전문가 기록과 견준 값입니다." in quick
    assert answer.startswith(expected) and "아래는 전문가 기록과 견준 값입니다." in answer
    assert "두꺼운" in answer and "180" in answer
    if expected == "목표 대비 판정 자료가 없습니다.":
        assert "높이맵 원본 없음" not in answer


def test_forged_target_grades_are_not_used():
    value = context(target=target_assessment()).replace('"grade": "above"', '"grade": "normal"')
    data = ask("knowledge_search", "도막이 과하거나 모자란 영역이 있어?", value)
    assert data["coating_context"]["analysis_status"] == "not_available"
    assert "과도막" not in data["answer"] and "정상: 2개" not in data["answer"]


def test_error_analysis_in_the_viewer_uses_the_signed_target():
    quick = ask("error_analysis", "도막 분포 이상 원인은?", context(target=target_assessment()))["answer"]
    assert "Zone 1은 저도막입니다" in quick and "조절 후보" in quick and "확정된 것은 아닙니다" in quick
    data = ask("error_analysis", "도막 분포 이상 원인은?", context(target=target_assessment()), mode="detailed")
    answer = data["answer"]
    assert "조절 후보" in answer and "확정된 것은 아닙니다" in answer and "Zone 1은 저도막입니다" in answer
    assert data["coating_context"]["comparison_basis"] == "expert_zone_median" and data["retrieval_used"] is False


def test_error_analysis_with_only_expert_rows_falls_back_to_the_completed_report():
    """알려진 결함: 뷰어의 전문가 대비 자료가 먼저 쓰여 완료 리포트가 있어도 "결과가 없어"라고 답했다."""
    from trainer.demo_api import coating_assessment as ca
    row = ca.zone_row(1, "3", ca.assess_zone(np.full((4, 4), 300.0), np.ones((4, 4), bool)))

    class Reports:
        def latest_complete_analysis(self, _session_id):
            return "RPT-1", {"coating_zones": [row]}

    data = ask("error_analysis", "도막 분포 이상 원인은?",
               context(target={"analysis_status": "not_available", "reason_code": "height_map_unavailable"}), reports=Reports(), mode="detailed")
    assert "Zone 3은 과도막입니다" in data["answer"] and "완료 분석의 도막 결과가 없어" not in data["answer"]
    assert data["coating_context"]["comparison_basis"] == "saved_coating_report"
    without = ask("error_analysis", "도막 분포 이상 원인은?", context(), mode="detailed")
    assert "완료 분석의 도막 결과가 없어" in without["answer"]


def test_advice_prompt_gets_the_compact_target_facts():
    class Advice(Pipeline):
        def run(self, **kwargs):
            self.calls.append(kwargs)
            return supported_document_result("[답변] 분사 거리와 이동 속도를 일정하게 유지하세요.")
    pipeline = Advice()
    ask("pose_recomm", "이 Zone에서 도막을 균일하게 만드는 방법은?", context("pose_recomm", target=target_assessment()), pipeline=pipeline)
    instruction = pipeline.calls[0]["instruction"]
    assert "검증된 최종 도막 비교 자료" in instruction and "목표 대비 판정" in instruction and "저도막" in instruction
    assert "six_band_counts" not in instruction and "전문가 대비 대표 두께 비교" not in instruction


def test_viewer_note_points_to_the_expert_basis_only_when_expert_rows_exist():
    """전문가 대비 행이 없는 기록(전문가 본인 기록 등)에서 '전문가 대비'를 보라고 하지 않는다."""
    with_rows = ask("knowledge_search", "도막이 과하거나 모자란 영역이 있어?", context(target=target_assessment()), mode="detailed")["answer"]
    without = ask("knowledge_search", "도막이 과하거나 모자란 영역이 있어?", context(target=target_assessment(), expert=False), mode="detailed")["answer"]
    assert "'전문가 대비'" in with_rows
    assert "저도막: 1개" in without and "최종 높이맵 기준(재생 시점 아님)" in without
    assert "전문가" not in without


def test_playback_zone_is_kept_when_only_the_target_has_that_zone():
    from trainer.demo_api.coating_query import parse_viewer_coating_context, target_advice_facts
    parsed = parse_viewer_coating_context(context(target=target_assessment(), expert=False), "knowledge_search")
    assert parsed["playback_context"] == {"map_num": "1", "zone_id": "1"}
    assert target_advice_facts(parsed)["zones"][0]["zone_id"] == "1"
    # 서명된 어느 행에도 없는 Zone은 붙이지 않는다(화면이 값을 만들지 못한다).
    value = json.loads(context(target=target_assessment(), expert=False).split(":", 1)[1])
    value["playback_context"]["zone_id"] = "9"
    assert "playback_context" not in parse_viewer_coating_context("motion_viewer_analysis:" + json.dumps(value), "knowledge_search")
