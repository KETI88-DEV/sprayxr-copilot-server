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
    data = ask("얇은 도막은 어떻게 보완해?", tab="knowledge", playback=AT_35S, pipeline=pipeline)
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
