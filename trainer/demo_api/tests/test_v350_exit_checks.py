import pytest
from trainer.demo_api.record_query import RecordFacts, ground_generated
from trainer.demo_api.tests.viewer_record import AT_3S, AT_24S, AT_35S, RecordingPipeline, ask, context, payload


def facts(playback):
    return RecordFacts.from_context(context("knowledge_search", payload(), playback))


CASES = [  # (G, 답, playback, zone, 있어야 할 낱말, 없어야 할 낱말)
    ("G3", "제일 얇게 나온 부분은 9번 구역입니다.", AT_35S, None, ["Zone 8", "77.2"], "9번 구역"),
    ("G4 far", "숙련자보다 멀리서 뿌린 Zone은 없습니다.", AT_24S, None, ["Zone 4"], "없습니다"),
    ("G4 over", "내 기록에서도 과도막이 발생한 데가 있습니다.", AT_24S, None, ["0곳"], "발생한 데가 있습니다"),
    ("G5 pass", "합격입니다. 현재 데이터에서 미도장 영역은 0%입니다.", AT_35S, None, ["판정할 수 없"], "합격입니다"),
    ("G5 normal", "목표 두께(160µm)에 가까워 정상 범위입니다.", AT_24S, "1", ["47.9%"], "정상 범위입니다"),
    ("G6", "현재 작업 기준에서 권장 이격거리가 445.6mm입니다.", AT_24S, "1", ["숙련자 측정 중앙값"], "권장 이격거리"),
]


@pytest.mark.parametrize("g,answer,playback,zone,needed,forbidden", CASES, ids=[c[0] for c in CASES])
def test_record_claims_in_generated_answers_are_replaced(g, answer, playback, zone, needed, forbidden) -> None:
    out = ground_generated(answer, facts(playback), zone=zone)
    assert all(word in out for word in needed) and forbidden not in out, out


@pytest.mark.parametrize("answer", ["도막 두께는 습도막 두께와 고형분 용적비로 정해집니다.", "적정 이격거리는 30cm입니다."])
def test_general_sentences_pass(answer: str) -> None:
    assert ground_generated(answer, facts(AT_3S), zone="8") == answer


def test_drop_only_removes_instead_of_replacing() -> None:
    out = ground_generated("제일 얇게 나온 부분은 9번 구역입니다. 겹침은 50%로 합니다.", facts(AT_35S), zone=None, drop_only=True)
    assert out == "겹침은 50%로 합니다."


def test_document_answer_about_the_record_is_checked_against_signed_data() -> None:
    pipeline = RecordingPipeline(plan={"kind": "coating_knowledge", "intent": "knowledge", "scope": "unspecified"},
                                 document="[답변] 이 기록에서 제일 얇은 곳은 Zone 9입니다. 덧칠은 50% 겹쳐 합니다.",
                                 chunks=[{"chunk_id": "DOC-x", "text": "덧칠은 50% 겹쳐 칠한다.", "rerank_score": 9.0}])
    data = ask("얇은 도막은 어떻게 보완해?", tab="knowledge", playback=AT_35S, pipeline=pipeline, mode="detailed")  # 상세 분석의 전체 답. 빠른 안내는 한 줄 60자다(2026-11-16)
    assert "Zone 8" in data["answer"] and "Zone 9입니다" not in data["answer"] and "50% 겹쳐" in data["answer"], data["answer"]


def test_document_numbers_must_come_from_chunks_or_signed_data() -> None:
    pipeline = RecordingPipeline(document="[답변] 건을 1.5m 떨어뜨리세요. 거리는 약 30cm로 유지하세요.",
                                 chunks=[{"chunk_id": "DOC-x", "text": "피도체와의 거리는 30 cm 정도로 균일하게", "rerank_score": 9.0}])
    data = ask("건 거리는 어떻게 유지해?", tab="knowledge", playback=AT_35S, pipeline=pipeline)
    assert "1.5m" not in data["answer"] and "30cm" in data["answer"], data["answer"]


def test_public_document_answers_are_not_checked_against_a_record() -> None:
    pipeline = RecordingPipeline(document="[답변] 에어리스 건 이동 속도는 매초 30~60cm입니다.")
    data = ask("에어리스 건 이동 속도는 어느 정도야?", tab="knowledge", raw_context="", pipeline=pipeline)
    assert "30~60cm" in data["answer"], data["answer"]


def test_general_document_sentences_without_a_record_reference_pass_unchanged() -> None:  # P13
    pipeline = RecordingPipeline(document="[답변] 건조가 완료된 뒤 덧칠합니다.")
    data = ask("건조 후 덧칠 시점은 어떻게 판단해?", tab="knowledge", playback=AT_35S, pipeline=pipeline)
    assert "건조가 완료된 뒤 덧칠합니다." in data["answer"], data["answer"]


def test_verdict_word_without_a_record_marker_is_left_alone() -> None:  # P13: 표지 없는 질문의 일반 문장은 조건절이 없어도 통과
    pipeline = RecordingPipeline(document="[답변] 건조 완료 상태에서 덧칠합니다.")
    data = ask("건조 후 덧칠 시점은 어떻게 판단해?", tab="knowledge", playback=AT_35S, pipeline=pipeline)
    assert "건조 완료 상태에서 덧칠합니다." in data["answer"], data["answer"]


# ---- fix round 1
@pytest.mark.parametrize("answer", ["Zone 3은 완료된 뒤 덧칠하면 됩니다.", "Zone 3은 이미 완료되어 덧칠하면 됩니다.", "Zone 3은 완료됐을 때 덧칠합니다."])
def test_false_completion_claim_with_a_conditional_word_is_still_replaced(answer: str) -> None:
    out = ground_generated(answer, facts(AT_35S), zone=None)
    assert out != answer and "Zone 3" in out and "보완" in out, out


def test_completion_inside_a_conditional_clause_passes() -> None:
    answer = "Zone 3이 완료되면 다음 Zone으로 갑니다."
    assert ground_generated(answer, facts(AT_35S), zone=None) == answer


@pytest.mark.parametrize("answer", ["건조 완료 후 덧칠합니다.", "도막이 정상 범위인지 확인하세요."])
def test_general_verdict_words_pass_even_in_a_marked_question(answer: str) -> None:
    assert ground_generated(answer, facts(AT_35S), zone=None, marked=True) == answer
    assert ground_generated(answer, facts(AT_35S), zone=None, marked=True, drop_only=True) == answer


@pytest.mark.parametrize("answer,changed", [("숙련자보다 멀리서 뿌린 Zone은 있습니다.", False), ("숙련자보다 멀리서 뿌린 Zone은 없습니다.", True),
                                            ("과도막은 없습니다.", False), ("과도막은 있습니다.", True), ("과도막이 발생한 Zone은 있습니다.", True)])
def test_existence_claims_are_checked_in_both_directions(answer: str, changed: bool) -> None:
    assert (ground_generated(answer, facts(AT_24S), zone=None) != answer) is changed


def _query(prompt: str):
    from trainer.demo_api.copilot_contracts import CopilotQueryRequest
    data = payload()
    return CopilotQueryRequest(user_id="COPILOT-AI-TEST", session_id=data["selected_session_id"], scenario_id=str(data["mode_id"]),
                               task_type="knowledge_search", prompt=prompt, response_mode="quick",
                               context=context("knowledge_search", data, AT_35S))


def test_safety_questions_skip_the_record_claim_checks() -> None:
    from trainer.demo_api.copilot_router import _checked_generation
    query = _query("내 기록에서 이 구역은 어때?")
    assert "합격" not in _checked_generation("Zone 3은 합격입니다.", query)
    assert _checked_generation("Zone 3은 합격입니다.", query, safety=True) == "Zone 3은 합격입니다."


def test_everything_dropped_gives_the_deterministic_answer_or_the_no_answer_line() -> None:
    from trainer.demo_api.copilot_router import _checked_generation
    drop = "Zone 3은 속도를 숙련자 수준으로 올리세요."
    assert "Zone 8" in _checked_generation(drop, _query("칠 다 했는데 어디가 제일 얇게 나왔어?")) and "숙련자 수준" not in _checked_generation(drop, _query("칠 다 했는데 어디가 제일 얇게 나왔어?"))
    assert _checked_generation(drop, _query("건 잡는 법 알려줘")) == "제공된 자료로는 답할 수 없습니다."
    assert _checked_generation(drop, _query("건 잡는 법 알려줘"), drop_only=True) == ""


@pytest.mark.parametrize("answer", ["두꺼운 도막은 균열이 생길 수 있습니다.", "과도막이 생기면 흘러내림이 있습니다."])
def test_general_possibility_sentences_are_not_existence_claims(answer: str) -> None:
    assert ground_generated(answer, facts(AT_24S), zone=None, marked=True) == answer
