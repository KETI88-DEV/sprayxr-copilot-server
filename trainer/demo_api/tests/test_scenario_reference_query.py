"""Reference work plans need a mode, not trainee telemetry or a report."""
from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from trainer.demo_api.copilot_router import create_copilot_router
from trainer.demo_api.sequence_analysis import SequenceAnalysisEngine
from trainer.demo_api.sequence_store import SequenceStore

URL = "/api/v1/copilot/query"
QUESTION = "이 피도면의 작업 순서를 구역 단위로 알려주세요."



def _flow(answer: str) -> str:
    """빠른 안내는 화살표를 붙여 쓴다(Zone 7→4→8, 2026-11-16). 두 수준을 같은 꼴로 견준다."""
    return answer.replace(" → ", "→")

def profile(root: Path, process: str, flow: list[str], mode: int = 3) -> None:
    path = root / "generated" / "profiles" / process / "learning" / f"mode_{mode:02d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"aggregate": {"macro_orders": [{
        "order": [{"map_num": 1, "zone_id": zone} for zone in flow], "session_count": 3,
    }], "zones": {}}}), encoding="utf-8")


class Pipeline:
    def __init__(self, decision=None):
        self.decision = decision or {"kind": "chat", "intent": "knowledge", "scope": "unspecified"}
        self.calls = []

    def plan_copilot_dialogue(self, *args, **kwargs):
        self.calls.append("plan")
        return self.decision

    def converse_copilot(self, *args, **kwargs):
        self.calls.append("converse")
        return "일반 대화 답변입니다."

    def run(self, **kwargs):
        self.calls.append("run")
        return {"response": "[답변] 일반 지식 답변입니다.", "rag_chunks": []}


def client(root, *, pipeline=None, store=None, reports=None, **dependencies):
    app = FastAPI()
    app.include_router(create_copilot_router(
        (lambda: pipeline) if pipeline is not None else None,
        store=store or SequenceStore(":memory:"), reports=reports,
        sequence_engine=SequenceAnalysisEngine(root),
        **dependencies,
    ))
    return TestClient(app)


def query(c, **overrides):
    body = {"user_id": "COPILOT-DEMO-USER", "task_type": "knowledge_search", "scenario_id": "3", "prompt": QUESTION}
    body.update(overrides)
    response = c.post(URL, json=body)
    assert response.status_code == 200, response.text
    return response.json()["data"]


@pytest.mark.parametrize("task", ["knowledge_search", "trajectory_recomm"])
def test_mode_only_reference_question_bypasses_empty_facts_conversation(tmp_path, task):
    profile(tmp_path, "spray", ["1", "3", "2"])
    pipeline = Pipeline()
    data = query(client(tmp_path, pipeline=pipeline), task_type=task, context="스프레이 도장")
    assert "Zone 1→3→2" in _flow(data["answer"])
    assert "현재 등록된 기준 순서" in data["answer"]
    assert "전문가(스프레이 3회) 기준이며 실제 수행·완료 판정은 아닙니다." in data["answer"]
    assert "학습 모드 3" in data["answer"]
    assert "스프레이" in data["answer"]
    assert data["context_status"] == "available"
    assert data["retrieval_used"] is False and data["citations"] == []
    context = data["sequence_context"]
    assert context["scope"] == "reference_only" and context["can_compare"] is False
    assert context["recommended_flow"] == ["1", "3", "2"]
    for key in ("adherence_rate", "position_similarity", "best_reference_similarity", "evaluated_visit_count", "on_track_visit_count", "first_visit_flow"):
        assert context[key] is None
    assert pipeline.calls == []


def test_unknown_process_shows_both_registered_plans_without_selecting_spray(tmp_path):
    profile(tmp_path, "spray", ["1", "3", "2"])
    profile(tmp_path, "blasting", ["2", "1", "3"])
    data = query(client(tmp_path))
    assert "스프레이" in data["answer"] and "블라스팅" in data["answer"]
    assert "1→3→2" in _flow(data["answer"]) and "2→1→3" in _flow(data["answer"])
    assert data["sequence_context"]["process"] is None
    assert data["sequence_context"]["recommended_flow"] is None
    assert {p["process"] for p in data["sequence_context"]["reference_plans"]} == {"spray", "blasting"}


def test_unknown_process_with_only_blasting_labels_the_available_process(tmp_path):
    profile(tmp_path, "blasting", ["3", "1", "2"])
    data = query(client(tmp_path))
    assert "블라스팅" in _flow(data["answer"]) and "3→1→2" in _flow(data["answer"])
    assert data["sequence_context"]["process"] == "blasting"


def test_started_session_without_sequences_resolves_saved_mode(tmp_path):
    profile(tmp_path, "spray", ["1", "3", "2"])
    reports = SimpleNamespace(get_session=lambda session_id: SimpleNamespace(scenario_id="3"))
    data = query(client(tmp_path, reports=reports), scenario_id=None, session_id="STARTED-ONLY", context="spray")
    assert data["scenario_id"] == "3"
    assert data["sequence_context"]["recommended_flow"] == ["1", "3", "2"]


def test_explicit_mode_and_prompt_process_override_stored_sequence(tmp_path):
    profile(tmp_path, "spray", ["2", "1"], mode=2)
    profile(tmp_path, "blasting", ["3", "1", "2"])
    store = SequenceStore(":memory:")
    body = {"scenario_id": "3", "sequence_setting": {"tool_type": "blasting_gun"}}
    store.record(session_key="SAVED", sequence_id="SEQ-1", body_sha256="x", frames=0,
                 received_at="2026-11-11T00:00:00Z", body=json.dumps(body))
    data = query(client(tmp_path, store=store), scenario_id="2", session_id="SAVED", prompt="스프레이 작업 순서를 구역 단위로 알려줘")
    assert data["scenario_id"] == "2"
    assert data["sequence_context"]["process"] == "spray"
    assert data["sequence_context"]["recommended_flow"] == ["2", "1"]


def test_saved_sequence_supplies_mode_and_blasting_process(tmp_path):
    profile(tmp_path, "spray", ["1", "3", "2"])
    profile(tmp_path, "blasting", ["3", "1", "2"])
    store = SequenceStore(":memory:")
    store.record(session_key="SAVED", sequence_id="SEQ-1", body_sha256="x", frames=0,
                 received_at="2026-11-11T00:00:00Z", body=json.dumps({
                     "scenario_id": "3", "sequence_setting": {"tool_type": "blasting_gun"}}))
    data = query(client(tmp_path, store=store), scenario_id=None, session_id="SAVED")
    assert data["scenario_id"] == "3" and data["sequence_context"]["process"] == "blasting"
    assert "3→1→2" in _flow(data["answer"])


@pytest.mark.parametrize("scenario,reason,needed", [(None, "scenario_id_missing", "모드"), ("11", "sequence_reference_unavailable", "기준")])
def test_missing_mode_or_reference_explains_the_actual_requirement(tmp_path, scenario, reason, needed):
    profile(tmp_path, "spray", ["1", "2"])
    data = query(client(tmp_path), scenario_id=scenario, context="스프레이")
    assert data["context_status"] == "not_available"
    assert data["sequence_context"]["reason_code"] == reason
    assert needed in data["answer"]
    assert "시퀀스를 전송" not in data["answer"] and "리포트" not in data["answer"]


def test_semantic_order_reference_lookup_works_across_default_tab(tmp_path):
    profile(tmp_path, "spray", ["1", "3", "2"])
    pipeline = Pipeline({"kind": "lookup", "intent": "order_reference", "scope": "record"})
    data = query(client(tmp_path, pipeline=pipeline), prompt="이 피도면을 어디부터 시작하면 좋을까요?", context="spray")
    assert "1→3→2" in _flow(data["answer"])
    assert pipeline.calls == ["plan"]


def test_semantic_tab_classifier_can_request_a_reference_plan_without_a_session(tmp_path):
    profile(tmp_path, "spray", ["1", "3", "2"])

    class Classifier:
        def classify_copilot_intent(self, task_type, prompt):
            return "order_reference"

    data = query(client(tmp_path, pipeline=Classifier()), task_type="trajectory_recomm",
                 prompt="이 피도면을 어디부터 시작하면 좋을까요?", context="spray")
    assert "1→3→2" in _flow(data["answer"])


def test_bad_semantic_reference_label_does_not_replace_an_explicit_actual_question(tmp_path):
    profile(tmp_path, "spray", ["1", "3", "2"])
    pipeline = Pipeline({"kind": "lookup", "intent": "order_reference", "scope": "record"})
    data = query(client(tmp_path, pipeline=pipeline), task_type="trajectory_recomm",
                 session_id="EMPTY", prompt="실제 수행 순서를 알려줘")
    assert data["context_status"] == "not_available"
    assert data["sequence_context"].get("scope") != "reference_only"


@pytest.mark.parametrize("prompt", ["실제 수행 순서는?", "내가 누락한 구역은?", "순서 준수율은?"])
def test_actual_history_questions_still_require_actual_data(tmp_path, prompt):
    profile(tmp_path, "spray", ["1", "3", "2"])
    data = query(client(tmp_path), task_type="trajectory_recomm", session_id="EMPTY", prompt=prompt)
    assert data["context_status"] == "not_available"
    assert data["sequence_context"].get("scope") != "reference_only"
    assert "1→3→2" not in _flow(data["answer"])


@pytest.mark.parametrize("prompt", ["안녕", "다음 구역이란 무엇인가요?"])
def test_general_or_realtime_questions_are_not_replaced_with_a_reference_plan(tmp_path, prompt):
    profile(tmp_path, "spray", ["1", "3", "2"])
    data = query(client(tmp_path, pipeline=Pipeline()), prompt=prompt)
    assert data["sequence_context"] is None
    assert data["answer"] == "일반 대화 답변입니다."


def test_painting_order_term_question_is_answered_as_knowledge_not_a_reference_plan(tmp_path):
    profile(tmp_path, "spray", ["1", "3", "2"])
    data = query(client(tmp_path, pipeline=Pipeline()), prompt="도장 작업 순서란 무엇인가요?")
    assert data["sequence_context"] is None
    assert data["answer"] == "일반 지식 답변입니다."


def test_detailed_reference_answer_is_short_and_has_reference_basis(tmp_path):
    profile(tmp_path, "spray", ["1", "3", "2"])
    data = query(client(tmp_path), context=json.dumps({"process": "spray"}), response_mode="detailed")
    assert "1→3→2" in _flow(data["answer"]) and "3회" in _flow(data["answer"])
    assert data["sequence_context"]["reference_basis"] == "measured_expert_runs"
    assert len(data["answer"]) < 400
    assert "준수율" not in data["answer"] and "다음 Zone" not in data["answer"]


@pytest.mark.parametrize("prompt,position", [
    ("학습 2에서 뭐부터 칠해? 아직 시작 버튼도 안 눌렀어.", ("first", None)),
    ("나 지금 방금 켰는데 바로 어디부터 분사해?", ("first", None)),
    ("아니 작업한 데 말고, 가야 하는 첫 구역 번호만.", ("first", None)),
    ("2번 끝내면 어디로 가는 거였더라?", ("after", "2")),
    ("3번 하고 나면 그 뒤는요?", ("after", "3")),
    ("그럼 학습모드 11도 2번부터 칠하면 되는 거야?", ("first", None)),
    ("지금 여기는 끝났어? 다음 어디로 옮겨?", None),
    ("표면처리는 뭐부터 해야 해?", None),
    ("어디부터 세척해야 녹이 없어지나요?", None),
    ("10번 하고 나면 도막을 얼마나 말려야 돼?", None),
    ("그 다음은?", None),
])
def test_reference_position_recognizes_bounded_planning_questions(prompt, position):
    from trainer.demo_api.reference_order_query import reference_position_question

    assert reference_position_question(prompt) == position


def test_numbered_reference_position_stays_separate_from_actual_move(tmp_path):
    from trainer.demo_api.reference_order_query import reference_order_answer

    profile(tmp_path, "spray", ["2", "3", "1"])
    profile(tmp_path, "blasting", ["1", "2", "3"])
    engine = SequenceAnalysisEngine(tmp_path)
    answer, state = reference_order_answer(engine, scenario_id="3", process=None, position=("first", None))
    assert "스프레이" in answer and "Zone 2" in answer
    assert "블라스팅" in answer and "Zone 1" in answer
    assert state["scope"] == "reference_only" and state["recommended_flow"] is None
    assert state.get("next_zone_id") is None

    answer, state = reference_order_answer(engine, scenario_id="3", process="spray", position=("after", "3"))
    assert "Zone 3 다음: Zone 1" in answer
    assert state["recommended_flow"] == ["2", "3", "1"] and state.get("next_zone_id") is None

    answer, _ = reference_order_answer(engine, scenario_id="3", process="spray", position=("after", "1"))
    assert "Zone 1이 마지막 구역" in answer


@pytest.mark.parametrize("prompt,expected", [
    ("방금 내가 얼마나 칠했는지 퍼센트 찍어줘.", True),
    ("지금 구역 진행률 몇 퍼센트야?", True),
    ("현재까지 칠한 진행률 숫자로 알려줘.", True),
    ("지금 얼마나 더 해야 돼?", True),
    ("목표 두께 도달 면적은 50%라는데 그럼 아직도 못 가는 거 아냐?", True),
    ("2번 구역에서 160 넘는 면적은 50%라는데, 그러면 진행률도 50%인 거야?", True),
    ("아직 빨간 표시도 아니고 평균은 정상인 줄 알았는데 왜 붙잡아 둬?", True),
    ("작업 속도는 얼마나 빨라야 돼?", False),
    ("구역이 85프로 넘으면 도막 목표를 85마이크론으로 바꾸라는 뜻인가?", False),
    ("진행률은 어떻게 계산해?", False),
    ("칠한 거", False),
])
def test_current_progress_questions_separate_live_values_from_definitions(prompt, expected):
    from trainer.demo_api.reference_order_query import is_current_progress_question

    assert is_current_progress_question(prompt) is expected


@pytest.mark.parametrize("prompt,expected", [
    ("저 이제 건 내려도 돼요, 다음 존 찍어주세요.", True),
    ("화면엔 85로 반올림될 텐데 바로 넘어가면 안 돼?", True),
    ("2조는 끝나요? 이제 다음 쪼 ㄴ가?", True),
    ("건은 멈추고 3조 쪽으로 넘어감 되냐?", True),
    ("2번 구역 진행률이 벌써 95퍼니까 건 내려놓고 3번으로 가면 되죠?", True),
    ("2번 존 끝? 하고 3번 존 가도 됨? 한 줄만.", True),
    ("이 정도면 옆 칸으로 넘어가도 될까?", True),
    ("이제 3번으로 넘어가도 되겠어요?", True),
    ("지금 밥 먹으러 가도 돼?", False),
    ("그다음 챕터로 넘어가도 될까?", False),
    ("옆 칸으로 넘어갈 때 어떤 안전 장비가 필요해?", False),
    ("다음 구역에서 준비해야 할 안전 장비는?", False),
])
def test_current_move_paraphrases_dont_capture_unrelated_questions(prompt, expected):
    from trainer.demo_api.reference_order_query import is_current_next_question

    assert is_current_next_question(prompt) is expected


def test_completion_words_inside_a_rewrite_request_do_not_imply_a_result():
    from trainer.demo_api.reference_order_query import is_current_completion_question

    prompt = "'Zone 7을 완료했습니다'는 내가 적어본 UI 문구인데, 철자 자연스럽게 바꿔 줘."
    assert not is_current_completion_question(prompt)


@pytest.mark.parametrize("prompt,expected", [
    ("지금 이 정도면 옆 칸으로 넘어가도 될까?", True),
    ("여긴 그만 칠해도 돼?", True),
    ("이 구역은 이쯤에서 마쳐도 될까요?", True),
    ("이제 3번으로 넘어가도 되겠어요?", True),
    ("어느 구역부터 칠하면 돼?", False),
    ("다음 구역은 어디야?", False),
    ("여기서 그만 칠하면 다음 공정까지 시간은?", False),
    ("지금 도장 작업에 쓰는 건을 내려놔도 돼?", False),
])
def test_live_completion_includes_permission_but_excludes_plan_and_procedure(prompt, expected):
    from trainer.demo_api.reference_order_query import is_current_completion_question

    assert is_current_completion_question(prompt) is expected


def test_position_explicit_mode_overrides_selected_mode_without_falling_back(tmp_path):
    from trainer.demo_api.reference_order_query import reference_order_answer, resolve_reference_target

    profile(tmp_path, "spray", ["2", "3", "1"])
    request = SimpleNamespace(context=None, scenario_id="3", session_id=None,
                              prompt="그럼 학습모드 11도 2번부터 칠하면 되는 거야?")
    position = ("first", None)
    scenario, process = resolve_reference_target(request, SequenceStore(":memory:"), position=position)
    answer, state = reference_order_answer(SequenceAnalysisEngine(tmp_path), scenario_id=scenario,
                                          process=process, position=position)
    assert scenario == "11" and "11" in answer
    assert state["reason_code"] == "sequence_reference_unavailable"


@pytest.mark.parametrize("prompt", ["현재 작업 순서에서 다음 구역은 어디인가요?", "작업 순서상 다음 Zone은 어디야?", "지금 다음 Zone으로 이동해도 되나요?"])
def test_next_zone_without_current_data_never_becomes_a_whole_reference_plan(tmp_path, prompt):
    profile(tmp_path, "spray", ["1", "3", "2"])
    pipeline = Pipeline({"kind": "lookup", "intent": "order_reference", "scope": "record"})
    data = query(client(tmp_path, pipeline=pipeline), prompt=prompt)
    assert data["sequence_context"]["scope"] == "current_progress"
    assert data["sequence_context"]["next_zone_id"] is None
    assert data["context_status"] == "not_available"
    assert "확인할 수 없" in data["answer"]
    assert "1→3→2" not in _flow(data["answer"])
    assert pipeline.calls == []


@pytest.mark.parametrize("mode", ["quick", "detailed"])
def test_verified_csv_only_viewer_keeps_its_rich_analysis_before_registered_reference(tmp_path, mode):
    from trainer.demo_api.viewer_context import sign_context
    profile(tmp_path, "spray", ["9", "8"])
    analysis = {"status": "available", "recommended_flow": ["1", "3", "2"],
                "selected_flow": ["1", "2", "3"], "adherence_rate": .67,
                "reference_session_count": 4, "best_reference_similarity": .75,
                "position_similarity": .6, "zone_keys": ["1:1", "1:2", "1:3"]}
    context = "motion_viewer_analysis:" + json.dumps({"analysis_context": analysis,
        "signature": sign_context("trajectory_recomm", analysis)})
    data = query(client(tmp_path), task_type="trajectory_recomm", scenario_id=None,
                 prompt="권장 Zone 순서는?", context=context, response_mode=mode)
    assert data["sequence_context"]["selected_flow"] == ["1", "2", "3"]
    assert data["sequence_context"]["adherence_rate"] == .67
    assert data["context_status"] == "available"
    assert "1→3→2" in _flow(data["answer"]) and "9→8" not in _flow(data["answer"])
    assert "67%" not in data["answer"]  # 권장 순서만 물었고 유사도는 구조화된 근거에 보존한다.


def test_invalid_signed_viewer_playback_cannot_be_hidden_by_reference_shortcut(tmp_path):
    from trainer.demo_api.viewer_context import sign_context
    profile(tmp_path, "spray", ["1", "2"])
    analysis = {"status": "available", "recommended_flow": ["1", "2"], "zone_keys": ["1:1"]}
    context = "motion_viewer_analysis:" + json.dumps({"analysis_context": analysis,
        "signature": sign_context("trajectory_recomm", analysis),
        "playback_context": {"playhead_seconds": -1}})
    response = client(tmp_path).post(URL, json={"user_id": "COPILOT-DEMO-USER",
        "task_type": "trajectory_recomm", "scenario_id": "3", "prompt": "권장 Zone 순서는?", "context": context})
    assert response.status_code == 422


def test_forged_viewer_reference_flow_is_never_used(tmp_path):
    profile(tmp_path, "spray", ["1", "2"])
    context = "motion_viewer_analysis:" + json.dumps({"analysis_context": {
        "status": "available", "recommended_flow": ["99", "98"]}, "signature": "forged"})
    data = query(client(tmp_path), task_type="trajectory_recomm", prompt="권장 Zone 순서는?", context=context)
    assert "99→98" not in _flow(data["answer"])
    assert data["sequence_context"]["scope"] == "reference_only"


@pytest.mark.parametrize("coverage,status,next_zone", [(None, None, None), (50, "needs_more", None), (100, "complete", "2")])
def test_next_zone_uses_the_shared_official_pixel_completion_gate(tmp_path, coverage, status, next_zone):
    from trainer.demo_api.tests.test_zone_completion_gate import engine, request
    progress = engine(tmp_path)
    req = request(valid_count=coverage)
    if coverage is None:
        req.sequence_result.cur_sequence_heatmap = None
    store = SequenceStore(":memory:")
    store.record(session_key=req.session_id, sequence_id=req.sequence_id, body_sha256="next",
                 frames=0, received_at="now", body=req.model_dump_json())
    data = query(client(tmp_path, store=store, coating_engine=progress),
                 session_id=req.session_id, task_type="trajectory_recomm",
                 prompt="현재 작업 순서에서 다음 구역은 어디인가요?")
    assert data["sequence_context"]["scope"] == "current_progress"
    assert data["sequence_context"]["next_zone_id"] == next_zone
    assert data["coating_context"]["status"] == status
    if next_zone is None:
        assert "다음 권장 구역" not in data["answer"]
    else:
        assert "2" in data["answer"]


def test_current_progress_does_not_apply_another_scenario_to_saved_pixels(tmp_path):
    from trainer.demo_api.tests.test_zone_completion_gate import engine, request
    progress = engine(tmp_path)
    req = request()
    store = SequenceStore(":memory:")
    store.record(session_key=req.session_id, sequence_id=req.sequence_id, body_sha256="next",
                 frames=0, received_at="now", body=req.model_dump_json())
    data = query(client(tmp_path, store=store, coating_engine=progress), scenario_id="2",
                 session_id=req.session_id, prompt="작업 순서상 다음 Zone은 어디야?")
    assert data["sequence_context"]["next_zone_id"] is None
    assert data["context_status"] == "not_available"


def test_next_zone_respects_explicit_saved_session_target(tmp_path):
    from trainer.demo_api.tests.test_zone_completion_gate import engine, request
    progress = engine(tmp_path)
    req = request(value=160)
    store = SequenceStore(":memory:")
    store.record(session_key=req.session_id, sequence_id=req.sequence_id, body_sha256="target",
                 frames=0, received_at="now", body=req.model_dump_json())
    reports = SimpleNamespace(get_session=lambda _: SimpleNamespace(target_thickness_um=200))
    data = query(client(tmp_path, store=store, reports=reports, coating_engine=progress),
                 session_id=req.session_id, prompt="작업 순서상 다음 Zone은 어디야?")
    assert data["coating_context"]["target_thickness_um"] == 200
    assert data["sequence_context"]["next_zone_id"] is None


@pytest.mark.parametrize("prompt,value,next_zone", [
    ("이 정도면 옆 칸으로 넘어가도 될까?", 145.8, None),
    ("이 정도면 옆 칸으로 넘어가도 될까?", 171, "2"),
    ("여긴 그만 칠해도 돼?", 145.8, None),
    ("현재 3번으로 넘어가도 되겠어요?", 171, "2"),
])
def test_spoken_move_permission_uses_measured_gate(tmp_path, prompt, value, next_zone):
    from trainer.demo_api.tests.test_zone_completion_gate import engine, request

    progress = engine(tmp_path)
    req = request(value=value)
    store = SequenceStore(":memory:")
    store.record(session_key=req.session_id, sequence_id=req.sequence_id, body_sha256="move",
                 frames=0, received_at="now", body=req.model_dump_json())
    data = query(client(tmp_path, store=store, coating_engine=progress), session_id=req.session_id,
                 user_id=req.user_id, prompt=prompt, summarize=False)

    assert data["sequence_context"]["current_zone_id"] == "1"
    assert data["sequence_context"]["next_zone_id"] == next_zone
    expected = "완료 기준을 넘었습니다" if next_zone else "아직 완료되지 않았습니다"
    assert expected in data["answer"]
    if "3번" in prompt:
        assert "지금 안내되는 다음 구역은 2구역" in data["answer"]


def test_spoken_move_permission_does_not_use_another_workers_session(tmp_path):
    from trainer.demo_api.tests.test_zone_completion_gate import engine, request

    progress = engine(tmp_path)
    req = request()
    store = SequenceStore(":memory:")
    store.record(session_key=req.session_id, sequence_id=req.sequence_id, body_sha256="owner",
                 frames=0, received_at="now", body=req.model_dump_json())
    reports = SimpleNamespace(get_session=lambda _: SimpleNamespace(user_id=req.user_id, scenario_id=req.scenario_id))
    data = query(client(tmp_path, store=store, reports=reports, coating_engine=progress),
                 session_id=req.session_id, user_id="different-user", prompt="이제 2번으로 넘어가도 될까요?")

    assert data["context_status"] == "not_available"
    assert data["sequence_context"]["reason_code"] == "session_mismatch"
    assert data["sequence_context"]["next_zone_id"] is None
    assert data["coating_context"] is None


@pytest.mark.parametrize("with_sequence", [False, True])
def test_live_thickness_uses_recent_heatmap_instead_of_report(tmp_path, with_sequence):
    from trainer.demo_api.tests.test_zone_completion_gate import engine, request

    progress = engine(tmp_path)
    req = request(value=171)
    store = SequenceStore(":memory:")
    if with_sequence:
        store.record(session_key=req.session_id, sequence_id=req.sequence_id, body_sha256="measured",
                     frames=0, received_at="now", body=req.model_dump_json())
    data = query(client(tmp_path, store=store, coating_engine=progress), session_id=req.session_id,
                 user_id=req.user_id, prompt="지금 칠한 도막은 몇 마이크론이야?", summarize=False)

    if with_sequence:
        assert data["sequence_context"]["scope"] == "current_coating_measurement"
        measured = re.search(r"누적 히트맵 기준 1구역의 도포부 평균 두께는 ([\d.]+)µm", data["answer"])
        assert measured and float(measured[1]) == pytest.approx(161, abs=0.3)
    else:
        assert data["sequence_context"]["reason_code"] == "no_sequence_for_session"
        assert data["context_status"] == "not_available"


def test_current_thickness_and_session_target_keep_their_sources_separate(tmp_path):
    from trainer.demo_api.tests.test_zone_completion_gate import engine, request

    progress = engine(tmp_path)
    req = request(value=171)
    store = SequenceStore(":memory:")
    store.record(session_key=req.session_id, sequence_id=req.sequence_id, body_sha256="measure-vs-target",
                 frames=0, received_at="now", body=req.model_dump_json())
    reports = SimpleNamespace(get_session=lambda _: SimpleNamespace(
        user_id=req.user_id, scenario_id=req.scenario_id, target_thickness_um=145,
    ))
    data = query(client(tmp_path, store=store, reports=reports, coating_engine=progress), session_id=req.session_id,
                 user_id=req.user_id, prompt="실제로 칠한 평균 두께랑 목표는 각각 얼마야?", summarize=False)

    measured = re.search(r"도포부 평균 두께는 ([\d.]+)µm", data["answer"])
    assert measured and float(measured[1]) == pytest.approx(161, abs=0.3)
    assert "목표 두께는 145µm" in data["answer"]
    assert data["sequence_context"]["scope"] == "current_coating_measurement"


def test_next_zone_replays_saved_absolute_crops_after_engine_restart(tmp_path):
    from trainer.demo_api.tests.test_zone_completion_gate import engine, request
    progress = engine(tmp_path)
    store = SequenceStore(":memory:")
    left, right = request(1, width=5), request(2, width=5, x0=5)
    for req in (left, right):
        store.record(session_key=req.session_id, sequence_id=req.sequence_id, body_sha256=req.sequence_id,
                     frames=0, received_at="now", body=req.model_dump_json())
    data = query(client(tmp_path, store=store, coating_engine=progress),
                 session_id=right.session_id, prompt="작업 순서상 다음 Zone은 어디야?")
    assert data["coating_context"]["coverage_percent"] == 100
    assert data["sequence_context"]["next_zone_id"] == "2"


def test_blasting_next_guidance_keeps_its_existing_order_engine(tmp_path):
    from trainer.demo_api.tests.test_zone_completion_gate import fixture, request
    fixture(tmp_path)
    source = tmp_path / "generated/profiles/spray/learning/mode_03.json"
    destination = tmp_path / "generated/profiles/blasting/learning/mode_03.json"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(source.read_bytes())
    req = request()
    req.sequence_setting["tool_type"] = "blasting_gun"
    store = SequenceStore(":memory:")
    store.record(session_key=req.session_id, sequence_id=req.sequence_id, body_sha256="blasting",
                 frames=0, received_at="now", body=req.model_dump_json())
    data = query(client(tmp_path, store=store), session_id=req.session_id,
                 prompt="작업 순서상 다음 Zone은 어디야?")
    assert "다음 권장 2구역" in data["answer"]
    assert data["coating_context"] is None


@pytest.mark.parametrize("prompt, fragments", [
    ("여기 4번 먼저 갔다가 2번 가는 게 맞아?", ("스프레이", "2→4→5", "블라스팅", "4→2→5")),
    ("작업 순서가 정해진 이유와 이 모드의 순서도 알려줘.", ("2→4→5", "4→2→5", "전문가")),
])
def test_reference_checks_proposed_order_and_its_basis(tmp_path, prompt, fragments):
    profile(tmp_path, "spray", ["2", "4", "5"])
    profile(tmp_path, "blasting", ["4", "2", "5"])
    response = query(client(tmp_path, pipeline=Pipeline()), prompt=prompt, summarize=False)
    answer = _flow(response["answer"])
    assert response["sequence_context"]["scope"] == "reference_only"
    assert all(fragment in answer for fragment in fragments)


@pytest.mark.parametrize("prompt, value, fragments", [
    ("화면에는 85로 반올림되는데 넘어가면 안 돼?", 145.8, ("반올림", "85%", "1구역")),
    ("평균은 정상으로 보이는데 왜 붙잡아 둬?", 145.8, ("85%", "남았습니다", "평균 두께", "저도막")),
    ("다시 작업 얘기. 다음 순서는 뭐였지?", 171, ("2구역",)),
    ("다음 존 말고 지금 내가 어디를 칠하고 있었어?", 171, ("1번",)),
    ("지금 작업하던 구역이랑 처음 칠해야 했던 구역을 알려줘.", 171, ("최근 분사", "첫 구역")),
    ("다음 데 가면 된다고 했잖아. 아직 덜 했다고 하는데?", 145.8, ("85%", "남았습니다")),
    ("구역 진행률 85퍼센트가 목표 두께 85마이크론으로 바꾸라는 뜻이야?", 171, ("85%", "160µm")),
    ("그럼 도막 품질도 합격이란 뜻이야?", 150, ("이동 기준", "저도막")),
    ("만약 구역 전체를 40마이크론씩만 칠하면 얼마나 진행돼?", 171, ("25%", "160µm", "가정")),
])
def test_live_query_paraphrases_keep_their_measured_scope(tmp_path, prompt, value, fragments):
    from trainer.demo_api.tests.test_zone_completion_gate import engine, request

    progress = engine(tmp_path)
    req = request(value=value)
    store = SequenceStore(":memory:")
    store.record(session_key=req.session_id, sequence_id=req.sequence_id, body_sha256="live",
                 frames=0, received_at="now", body=req.model_dump_json())
    response = query(client(tmp_path, store=store, coating_engine=progress, pipeline=Pipeline()),
                     session_id=req.session_id, user_id=req.user_id, prompt=prompt, summarize=False)
    assert all(fragment in response["answer"] for fragment in fragments)
    if "반올림" in prompt:
        actual = response["coating_context"]["completion_percent"]
        assert f"{actual:.2f}%" in response["answer"]
        assert actual < 85 and response["sequence_context"]["next_zone_id"] is None
    if "말고 지금" in prompt:
        assert "2번" not in response["answer"]


@pytest.mark.parametrize("prompt, task_type, fragments", [
    ("아직 분사를 안 했는데 이상 자세였나요?", "pose_recomm", ("자세", "분사 기록")),
    ("Replay 채점표에서 제일 많이 감점된 항목은?", "error_analysis", ("완료 리포트", "항목")),
    ("리포트 점수랑 지금 화면 점수 왜 달라요?", "error_analysis", ("완료 리포트", "두 점수")),
    ("센서 수치 없어도 12번 구역의 거리가 멀었다고 결론 내줘.", "knowledge_search", ("12번 구역", "이격거리", "실측")),
    ("48번 구역 비교 화면에서 자세 유사도가 왜 0점인가요?", "pose_recomm", ("48번 구역", "자세 유사도", "확인")),
])
def test_session_query_explains_which_evidence_is_missing(tmp_path, prompt, task_type, fragments):
    response = query(client(tmp_path, pipeline=Pipeline()), session_id="STARTED-ONLY",
                     task_type=task_type, prompt=prompt, summarize=False)
    assert response["context_status"] == "not_available"
    assert all(fragment in response["answer"] for fragment in fragments)


@pytest.mark.parametrize("session_owner, scenario_id, should_read", [
    ("COPILOT-DEMO-USER", "TR-L03", True),
    ("different-user", "3", False),
    ("COPILOT-DEMO-USER", "2", False),
    (None, "3", False),
])
def test_pose_report_lookup_requires_verified_session_owner_and_mode(tmp_path, session_owner, scenario_id, should_read):
    reads = []

    def latest_complete(session_id):
        reads.append(session_id)
        return "RPT-COMPLETE", {"pose_report": {"zone_summary": [{
            "zone_id": "1", "average_similarity_score": 0.72, "evaluated_sequence_count": 2,
        }]}}

    session = SimpleNamespace(user_id=session_owner, scenario_id="3") if session_owner else None
    reports = SimpleNamespace(get_session=lambda _: session, latest_complete_analysis=latest_complete)
    response = query(client(tmp_path, reports=reports, pipeline=Pipeline()),
                     session_id="SAVED", scenario_id=scenario_id, task_type="pose_recomm",
                     prompt="1번 구역 비교 화면에서 자세 유사도가 왜 0점인가요?", summarize=False)
    assert reads == (["SAVED"] if should_read else [])
    assert ("72%" in response["answer"]) is should_read
    assert response["context_status"] == ("available" if should_read else "not_available")


def test_progress_correction_preserves_overpaint_pause(tmp_path):
    from trainer.demo_api.tests.test_zone_completion_gate import engine, request

    progress = engine(tmp_path)
    req = request(value=280, valid_count=70)
    store = SequenceStore(":memory:")
    store.record(session_key=req.session_id, sequence_id=req.sequence_id, body_sha256="risk",
                 frames=0, received_at="now", body=req.model_dump_json())
    response = query(client(tmp_path, store=store, coating_engine=progress, pipeline=Pipeline()),
                     session_id=req.session_id, user_id=req.user_id, summarize=False,
                     prompt="다음 데 가면 된다고 했잖아. 아직 덜 했다고 하는데?")
    assert response["sequence_context"]["next_zone_id"] is None
    assert response["coating_context"]["status"] == "overpaint_risk"
    assert "잠시 분사를 멈추고" in response["answer"]
