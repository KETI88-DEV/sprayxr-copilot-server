"""Record-specific evaluation cannot fall back to a fictional diagnosis."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from trainer.demo_api.copilot_router import create_copilot_router
from trainer.demo_api.query_chat_data import QueryChatDatasets
from trainer.demo_api.sequence_store import SequenceStore
from trainer.demo_api.tests.test_query_chat_data import datasets
from trainer.demo_api.tests.test_voice_session_routing import Planner


WORST = "선택한 훈련에서 가장 문제가 있는 Zone은 어디인가요?"
SUMMARY = "선택한 훈련 기록에서 우선 확인해야 할 문제를 알려주세요."
POSE = "선택한 훈련 기록의 자세를 확인하고 개선 방향을 알려주세요."


def _ask(prompt, *, reports=None, store=None, **fields):
    class GenerativePlanner(Planner):
        def plan_copilot_dialogue(self, prompt, **kwargs):
            self.plans.append(prompt)
            return {"kind": "chat", "intent": "knowledge", "scope": "record"}

        def converse_copilot(self, prompt, **kwargs):
            self.conversations.append(prompt)
            return "가장 문제가 있는 Zone은 3번입니다. 분사 각도가 문제입니다."

    pipeline = GenerativePlanner()
    app = FastAPI()
    app.include_router(create_copilot_router(lambda: pipeline, reports=reports, store=store or SequenceStore(":memory:")))
    response = TestClient(app).post("/api/v1/copilot/query",
        headers={"X-Copilot-Source": "assistant-test"},
        json={"user_id": "COPILOT-AI-TEST", "task_type": "knowledge_search", "prompt": prompt, **fields})
    assert response.status_code == 200
    assert pipeline.plans == [] and pipeline.runs == [] and pipeline.conversations == []
    return response


@pytest.mark.parametrize("prompt,missing_subject", [(WORST, "구역별"), (SUMMARY, "구역별"), (POSE, "자세 채점")])
def test_selected_training_record_does_not_invent_evaluation(datasets, prompt, missing_subject):
    selection = QueryChatDatasets(datasets).context("00000000000000000000000000000002")
    response = _ask(prompt, **{key: selection[key] for key in ("session_id", "scenario_id", "context")})
    data = response.json()["data"]
    assert response.headers["X-Copilot-Unavailable-Reason"] == "analysis_unavailable"
    assert data["context_status"] == "not_available"
    assert missing_subject in data["answer"] and "선택한 훈련 기록" in data["answer"]
    assert "3번" not in data["answer"] and "분사 각도" not in data["answer"]


def test_missing_session_for_explicit_record_gives_selection_hint():
    response = _ask(WORST)
    assert response.headers["X-Copilot-Unavailable-Reason"] == "session_required"
    assert "훈련 기록을 선택" in response.json()["data"]["answer"]


def test_unconfirmed_viewer_cannot_borrow_an_unrelated_report():
    def wrong_report(_):
        pytest.fail("An invalid viewer context must not evaluate another record")

    response = _ask(WORST, session_id="unrelated", context="motion_viewer_playback:{}",
                    reports=SimpleNamespace(latest_complete_analysis=wrong_report))
    assert response.headers["X-Copilot-Unavailable-Reason"] == "analysis_unavailable"
    assert "다시 불러" in response.json()["data"]["answer"]


def test_report_in_current_session_supplies_real_evaluation():
    analysis = {"pose_report": {"analysis_status": "available", "zone_summary": [
        {"zone_id": "7", "average_similarity_score": 0.4, "evaluated_sequence_count": 2},
    ]}}
    reports = SimpleNamespace(latest_complete_analysis=lambda _: ("TEST-REPORT", analysis))
    response = _ask(WORST, reports=reports, session_id="test-complete")
    data = response.json()["data"]
    assert "X-Copilot-Unavailable-Reason" not in response.headers
    assert data["context_status"] == "available" and "7" in data["answer"]


@pytest.mark.parametrize("prompt", ["그건 어때?", "그럼 왜?"])
def test_unanchored_followup_explains_which_context_is_missing(prompt):
    response = _ask(prompt)
    assert response.headers["X-Copilot-Unavailable-Reason"] == "clarification_required"
    assert "앞 질문" in response.json()["data"]["answer"]


def test_current_live_pose_with_sequences_uses_scoring_without_report(monkeypatch):
    from trainer.demo_api import copilot_router
    from trainer.demo_api.tests.test_voice_session_routing import _store
    from trainer.demo_api.tests.test_zone_completion_gate import request

    monkeypatch.setattr(copilot_router, "live_analysis", lambda *args, **kwargs: (None, {
        "pose_report": {"analysis_status": "available", "average_similarity_score": 0.8,
                        "evaluated_sequence_count": 1, "zone_summary": [
                            {"zone_id": "1", "average_similarity_score": 0.8, "evaluated_sequence_count": 1},
                        ]},
    }))
    sequence = request()
    response = _ask("이번 세션 전체 자세 유사도는 몇 퍼센트야?", session_id=sequence.session_id,
                    reports=SimpleNamespace(latest_complete_analysis=lambda _: None), store=_store(sequence))
    assert response.json()["data"]["pose_context"]["similarity_score"] == pytest.approx(0.8)
    assert "X-Copilot-Unavailable-Reason" not in response.headers
