"""Operator feedback: simulator mode numbers must not become education formats."""
import pytest

from trainer.demo_api.simulator_context import SIMULATOR_CONTEXT, wants_simulator_context
from trainer.demo_api.tests.test_query_evidence_fallback import Pipeline, ask, client


@pytest.mark.parametrize("prompt", ["훈련모드는 몇가지가있어", "훈련 모드는 몇 가지가 있어?", "학습모드는 몇 개야?"])
def test_mode_questions_receive_the_confirmed_simulator_modes(prompt):
    assert wants_simulator_context(prompt)
    assert "훈련모드" in SIMULATOR_CONTEXT and "1~10" in SIMULATOR_CONTEXT


def test_missing_documents_fallback_still_carries_simulator_mode_facts():
    pipeline = Pipeline(missing=True)
    ask(client(pipeline), prompt="훈련모드는 몇가지가있어")
    assert pipeline.conversations[-1]["simulator_context"] == SIMULATOR_CONTEXT
    assert "1~10" in pipeline.conversations[-1]["simulator_context"]


def test_unrelated_education_and_daily_topics_do_not_get_simulator_modes():
    assert not wants_simulator_context("원격 교육은 어떻게 진행돼?")
    assert not wants_simulator_context("점심으로 뭘 먹을까?")
