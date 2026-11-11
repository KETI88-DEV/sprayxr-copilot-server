"""세션 내용을 묻는 자유 질문은 넘겨준 자료에 있는 측정값만 말한다(2026-11-11 지시: 근거 없는 내용 금지)."""
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from trainer.demo_api import dialogue
from trainer.demo_api.copilot_router import create_copilot_router
from trainer.demo_api.copilot_session_context import summarize_session
from trainer.demo_api.sequence_store import SequenceStore


def _body(sequence_id: str, zone: str, speed: float, distance: float) -> dict:
    return {"session_id": "S-1", "scenario_id": "3", "sequence_id": sequence_id, "zone_id": zone,
            "sequence_setting": {"tool_type": "spray_gun"}, "sequence_info": [],
            "sequence_result": {"avg_speed_mms": speed, "avg_distance_mm": distance, "avg_thickness_um": 25.0}}


BODIES = [_body("SEQ-001", "9", 712.4, 753.15), _body("SEQ-002", "5", 775.77, 419.11), _body("SEQ-003", "5", 1291.84, 378.33),
          _body("SEQ-004", "5", 845.02, 445.36), _body("SEQ-005", "5", 900.0, 450.0)]


def test_summary_carries_the_sequence_the_question_points_at() -> None:
    summary = json.loads(summarize_session(BODIES, selected_sequence_id="SEQ-001"))

    assert summary["selected_sequence"] == {"sequence_id": "SEQ-001", "zone_id": "9", "avg_speed_mms": "712.4 mm/s",
                                            "avg_distance_mm": "753.15 mm", "avg_thickness_um": "25 µm"}
    assert "selected_sequence" not in json.loads(summarize_session(BODIES))
    assert "selected_sequence" not in json.loads(summarize_session(BODIES, selected_sequence_id="SEQ-999"))


FACTS = {"training_summary": summarize_session(BODIES, selected_sequence_id="SEQ-001")}


@pytest.mark.parametrize("answer", [
    "이 시퀀스의 평균 속도는 712.4 mm/s, 평균 거리는 753.15 mm입니다.",
    "평균 속도는 약 712 mm/s입니다.",  # 반올림은 같은 값이다
    "한국전쟁은 1950년 6월 25일에 시작됐습니다.",  # 측정 단위가 없는 수는 검사하지 않는다
    "1과 3을 더하면 4입니다.",
])
def test_measurements_from_the_facts_or_plain_numbers_pass(answer: str) -> None:
    assert dialogue.ungrounded_measurements(answer, facts=FACTS, prompt="") == []


@pytest.mark.parametrize(("answer", "invented"), [
    ("평균 속도: 905 mm/s (표준 대비 +4%)", ["4%"]),
    ("속도를 5~10% 낮추어 작업해 보세요.", ["5~10%"]),
    ("팔꿈치 각도 28°는 과도한 굴곡 상태입니다.", ["28°"]),
    ("평균 속도는 896 m/s입니다.", ["896 m/s"]),
])
def test_measurements_missing_from_the_facts_are_flagged(answer: str, invented: list[str]) -> None:
    assert dialogue.ungrounded_measurements(answer, facts=FACTS, prompt="") == invented


def test_a_number_the_user_gave_is_not_invented() -> None:
    assert dialogue.ungrounded_measurements("300 mm면 기준보다 가깝습니다.", facts=FACTS, prompt="거리 300 mm는 어때?") == []


def test_dropping_keeps_the_grounded_part_of_the_answer() -> None:
    cleaned = dialogue.drop_ungrounded("평균 속도: 905 mm/s (표준 대비 +4%). 속도를 5~10% 낮추어 작업해 보세요.", facts=FACTS, prompt="")

    assert cleaned == "평균 속도: 905 mm/s."
    assert dialogue.drop_ungrounded("속도를 5~10% 낮추세요.", facts=FACTS, prompt="") == "세션 기록으로는 확인할 수 없습니다."


class _Pipeline:
    def __init__(self, answers: list[str]) -> None:
        self.answers, self.facts = list(answers), []

    def plan_copilot_dialogue(self, prompt, **kwargs):
        return {"kind": "chat", "intent": "knowledge", "scope": "unspecified"}

    def converse_copilot(self, prompt, *, recent_turns, facts, response_mode, **kwargs):
        self.facts.append(facts)
        return self.answers.pop(0)


def _ask(pipeline: _Pipeline, **over) -> dict:
    store = SequenceStore(":memory:")
    for index, body in enumerate(BODIES):
        store.record(session_key="S-1", sequence_id=body["sequence_id"], body_sha256=str(index), frames=0, received_at="now", body=json.dumps(body))
    app = FastAPI()
    app.include_router(create_copilot_router(lambda: pipeline, store=store))
    return TestClient(app).post("/api/v1/copilot/query", json={"user_id": "u", "session_id": "S-1", "task_type": "knowledge_search",
                                                               "prompt": "이 시퀀스에서 분사 속도와 거리는 어땠어?", **over}).json()["data"]


def test_the_model_gets_the_pointed_sequence_and_an_invented_figure_is_regenerated_once() -> None:
    pipeline = _Pipeline(["평균 속도: 712.4 mm/s (표준 대비 +4%)", "이 시퀀스의 평균 속도는 712.4 mm/s입니다."])

    data = _ask(pipeline, context="SEQ-001")

    assert data["answer"] == "이 시퀀스의 평균 속도는 712.4 mm/s입니다."
    assert len(pipeline.facts) == 2
    assert json.loads(pipeline.facts[0]["training_summary"])["selected_sequence"]["sequence_id"] == "SEQ-001"


def test_a_figure_invented_twice_is_removed_from_the_answer() -> None:
    pipeline = _Pipeline(["평균 속도: 712.4 mm/s (표준 대비 +4%)", "평균 속도: 712.4 mm/s (기준 대비 -6%)."])

    assert _ask(pipeline, context="SEQ-001")["answer"] == "평균 속도: 712.4 mm/s."
