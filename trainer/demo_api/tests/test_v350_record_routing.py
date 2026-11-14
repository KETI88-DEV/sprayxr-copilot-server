"""v3.5.0 라우터: 기록 질문은 계획 모델 앞에서 결정적으로 답하고, 섞인 질문은 기록 문장 뒤에 문서 문장을 붙인다."""
import json

import pytest

from trainer.demo_api.copilot_router import MIXED_DOCUMENT_RULE, MIXED_NO_EVIDENCE
from trainer.demo_api.sequence_store import SequenceStore
from trainer.demo_api.tests.viewer_record import AT_3S, AT_24S, AT_28S, AT_35S, RecordingPipeline, ask, payload

DETERMINISTIC = [  # (번호, tab, playback, mode, 질문, 꼭 있어야 할 낱말)
    ("T20", "pose", AT_3S, "quick", "제 자세 괜찮아요? 잘하고 있는 거예요?", ["54%", "참고용", "70%"]),
    ("T21", "pose", AT_3S, "quick", "숙련자도 70% 정도라던데, 그럼 저는 나쁜 거예요?", ["54%", "참고용"]),
    ("T27", "pose", AT_3S, "quick", "화면에 87% 떠 있던데, 내 자세 유사도가 87%면 숙련자(70%)보다 잘한 거 맞지?", ["아니요", "작업 유사도", "54%"]),
    ("T28", "pose", AT_3S, "quick", "건 각도는 괜찮았어?", ["100%", "5.2°", "3.1°"]),
    ("T31", "order", AT_3S, "quick", "칠하는 순서가 정해져 있어요? 아무 데서나 시작하면 안 돼요?", ["Zone 7", "6회 모두", "45%"]),
    ("T32", "order", AT_3S, "detailed", "순서 준수율 45%는 어떻게 나온 거고, 숙련자들끼리는 순서가 얼마나 일치해?", ["45.1%", "기록에 없"]),
    ("T33", "order", AT_3S, "quick", "Zone 10이랑 11은 왜 안 칠했다고 나와요?", ["1~9"]),
    ("T34", "order", AT_35S, "quick", "다 칠했으니까 이제 끝난 거죠?", ["아니요", "95%"]),
    ("T35", "order", AT_3S, "quick", "숙련자는 보통 어느 Zone에서 시작해서 어디서 끝내요?", ["Zone 7", "6회", "3회"]),
    ("T36", "coating", AT_35S, "quick", "칠 다 했는데 어디가 제일 얇게 나왔어?", ["Zone 8", "77.2"]),
    ("T39", "coating", AT_3S, "quick", "이 Zone 속도랑 이격거리 숙련자랑 비교해줘.", ["2,230", "3,514", "251 mm", "54%"]),
    ("T42", "coating", AT_24S, "quick", "두껍게 너무 많이 뿌린 데도 있어요?", ["0곳", "10.1%"]),
    ("T43", "coating", AT_24S, "quick", "Zone 1은 정상이라고 나왔으니까 다 끝난 거지? 거긴 다시 안 가도 되지?", ["아니요", "47.9%"]),
    ("T44", "coating", AT_3S, "quick", "나 숙련자보다 느리던데, 속도를 숙련자만큼 올려야 해?", ["이전 장비"]),
    ("T46", "knowledge", AT_24S, "quick", "그럼 숙련자보다 멀리서 뿌린 Zone은 있었어?", ["Zone 4"]),
    ("T49", "knowledge", AT_3S, "quick", "뭐부터 고쳐야 돼? 딱 하나만 콕 집어줘.", ["Zone 3", "95%"]),
    ("v19", "coating", AT_3S, "quick", "왜 이렇게 얇게 나왔어?", ["Zone 8", "거리로는 설명하기 어렵"]),
]
SESSION = payload()["selected_session_id"]
USER = "COPILOT-AI-TEST"


def _turn(store: SequenceStore, prompt: str, answer: str, source: str, task_type: str = "knowledge_search") -> None:
    store.record_query_turn(session_key=SESSION, user_id=USER, prompt=prompt, answer=answer, task_type=task_type,
                            created_at="2026-11-14T00:00:00Z", answer_source=source)


def _sent(pipeline: RecordingPipeline) -> str:
    return json.dumps(pipeline.kwargs, ensure_ascii=False, default=str)


@pytest.mark.parametrize("item", DETERMINISTIC, ids=[d[0] for d in DETERMINISTIC])
def test_record_questions_are_answered_without_the_model(item) -> None:
    _, tab, playback, mode, prompt, needed = item
    data = ask(prompt, tab=tab, playback=playback, mode=mode)
    assert data["source"] == "analysis" and data["calls"] == []
    for word in needed:
        assert word in data["answer"], (word, data["answer"])


@pytest.mark.parametrize("prompt,tab,playback,needed", [
    ("그럼 뭐부터 고치면 돼요?", "pose", AT_3S, "오른쪽 팔꿈치"),   # T22
    ("왼팔은?", "pose", AT_28S, "왼"),                              # T24
    ("그럼 오른쪽 팔꿈치는 몇 도 더 펴야 돼?", "pose", AT_28S, "팔꿈치"),  # T25
    ("자세가 제일 문제였던 구역이 어디야?", "pose", AT_24S, "Zone 9"),   # T26
    ("다음엔 어디 칠해야 돼?", "order", AT_3S, "Zone 5"),             # T29
])
def test_existing_correct_answers_keep_their_paths(prompt, tab, playback, needed) -> None:
    data = ask(prompt, tab=tab, playback=playback)
    assert data["source"] == "analysis" and needed in data["answer"], data["answer"]


def test_follow_up_pronoun_uses_the_previous_deterministic_zone() -> None:
    store = SequenceStore(":memory:")
    ask("칠 다 했는데 어디가 제일 얇게 나왔어?", tab="coating", playback=AT_35S, store=store)   # T36
    data = ask("왜 그렇게 얇게 나온 거야?", tab="coating", playback=AT_35S, store=store)       # T37
    assert "Zone 8" in data["answer"] and "Zone 9" not in data["answer"], data["answer"]


def test_mixed_question_puts_record_sentence_first_and_documents_after() -> None:
    store = SequenceStore(":memory:")
    ask("칠 다 했는데 어디가 제일 얇게 나왔어?", tab="coating", playback=AT_35S, store=store)
    ask("왜 그렇게 얇게 나온 거야?", tab="coating", playback=AT_35S, store=store)
    pipeline = RecordingPipeline(document="[답변] 거리 약 30cm를 일정하게 유지하고 50% 겹쳐 칠합니다.",
                                 chunks=[{"chunk_id": "doc-1", "text": "스프레이 거리는 약 30cm를 유지하고 50% 겹쳐 칠한다.", "rerank_score": 9.0}])
    data = ask("그럼 거기부터 다시 칠하면 돼? 어떻게 칠해?", tab="coating", playback=AT_35S, store=store, pipeline=pipeline)  # T38
    first, _, rest = data["answer"].partition("\n")
    assert "Zone 8" in first and "1.9%" in first, data["answer"]
    assert "30cm" in rest, data["answer"]
    assert data["source"] == "documents" and data["calls"] == ["run"]
    instruction = str(dict(pipeline.kwargs)["run"]["instruction"])
    assert "77.2" not in instruction and '"zones"' not in instruction
    assert "어떻게 칠해?" in instruction and MIXED_DOCUMENT_RULE in instruction


def test_mixed_question_with_an_empty_document_answer_says_no_evidence() -> None:
    pipeline = RecordingPipeline(document="[답변] ")
    data = ask("이 기록 도막을 PSPC 90/10 규칙으로 판정하면 합격이에요?", tab="knowledge", playback=AT_35S, mode="detailed", pipeline=pipeline)
    assert "판정할 수 없" in data["answer"] and data["answer"].endswith(MIXED_NO_EVIDENCE), data["answer"]
    assert data["answer"].count("판정할 수 없") == 1


def test_mixed_question_without_a_model_still_returns_the_record_part() -> None:
    data = ask("이 기록 도막을 PSPC 90/10 규칙으로 판정하면 합격이에요?", tab="knowledge", playback=AT_35S, mode="detailed",
               offline=True)  # T50
    assert "판정할 수 없" in data["answer"] and data["source"] == "analysis"


def test_mixed_question_with_a_failing_model_still_returns_the_record_part() -> None:
    class Failing(RecordingPipeline):
        def run(self, **kwargs: object):
            raise RuntimeError("down")
    data = ask("이 기록 도막을 PSPC 90/10 규칙으로 판정하면 합격이에요?", tab="knowledge", playback=AT_35S, mode="detailed",
               pipeline=Failing())
    assert "판정할 수 없" in data["answer"] and data["source"] == "analysis"


def test_correction_reruns_the_previous_record_question() -> None:
    store = SequenceStore(":memory:")
    _turn(store, "왜 이렇게 얇게 나왔어?", "가까이서 칠해 얇아졌습니다.", "documents")
    pipeline = RecordingPipeline(plan={"kind": "correction", "intent": "coating_status", "scope": "playback"})
    data = ask("근데 가까이서 천천히 뿌리면 오히려 두꺼워지는 거 아냐? 아까 답 이상한데", tab="coating", playback=AT_3S,
               store=store, pipeline=pipeline)  # v340 #42
    assert "거리로는 설명하기 어렵" in data["answer"], data["answer"]
    assert data["source"] == "analysis" and data["calls"] == ["plan"]


def test_correction_after_a_safety_turn_is_not_reclassified() -> None:
    store = SequenceStore(":memory:")
    store.record_query_turn(session_key=SESSION, user_id=USER, prompt="왜 이렇게 얇게 나왔어?", answer="문서 답",
                            task_type="knowledge_search", created_at="t", answer_source="documents", safety_class="pressure")
    pipeline = RecordingPipeline(plan={"kind": "correction", "intent": "coating_status", "scope": "playback"})
    data = ask("아까 답 이상한데", tab="coating", playback=AT_3S, store=store, pipeline=pipeline)
    assert "거리로는 설명하기 어렵" not in data["answer"], data["answer"]


def test_record_or_domain_question_never_reaches_the_conversation_model() -> None:
    pipeline = RecordingPipeline(plan={"kind": "chat", "intent": "knowledge", "scope": "unspecified"})
    data = ask("내 기록에서 제일 오래 칠한 Zone은?", tab="knowledge", playback=AT_3S, pipeline=pipeline)
    assert "converse" not in data["calls"], data


def test_domain_question_classified_as_chat_goes_to_documents() -> None:
    pipeline = RecordingPipeline(plan={"kind": "chat", "intent": "knowledge", "scope": "unspecified"})
    data = ask("도막 두께는 왜 재요?", tab="knowledge", playback=AT_3S, pipeline=pipeline)
    assert "converse" not in data["calls"] and "run" in data["calls"], data


def test_generated_answers_are_not_sent_back_to_the_model() -> None:
    store = SequenceStore(":memory:")
    _turn(store, "어디 다시 칠해?", "9번 구역에서 다시 칠해야 합니다.", "documents")
    pipeline = RecordingPipeline()
    ask("얇은 도막은 어떻게 보완해?", tab="knowledge", playback=AT_3S, store=store, pipeline=pipeline)
    assert pipeline.kwargs and "9번 구역" not in _sent(pipeline)


def test_off_topic_chat_keeps_its_history() -> None:
    store = SequenceStore(":memory:")
    _turn(store, "점심 뭐 먹지?", "김밥을 추천합니다.", "conversation")
    pipeline = RecordingPipeline(plan={"kind": "chat", "intent": "knowledge", "scope": "unspecified"})
    data = ask("그럼 저녁은?", tab="knowledge", playback=AT_3S, store=store, pipeline=pipeline)
    assert data["calls"] == ["plan", "converse"]
    turns = dict(pipeline.kwargs)["converse"]["recent_turns"]
    assert any(turn["answer"] == "김밥을 추천합니다." for turn in turns)


def test_hazard_question_about_the_record_keeps_the_safety_path() -> None:
    # 안전 판정이 기록 분류보다 먼저다: 기록 질문(완료 판정)으로도 분류되는 문장이지만 문서 경로 + 답 검사(guard_answer)로 간다.
    pipeline = RecordingPipeline(document="[답변] 그냥 바꿔도 됩니다.")
    data = ask("Zone 8 다 칠했으니 이제 압력 안 빼고 팁 바꿔도 되지?", tab="coating", playback=AT_3S, pipeline=pipeline)
    assert data["calls"] == ["run"] and data["source"] == "documents"
    assert "남은 압력" in data["answer"] and "그냥 바꿔도" not in data["answer"], data["answer"]


# ---- fix round 1
@pytest.mark.parametrize("prompt", ["저 오늘 너무 피곤해요", "제 이름 뭐예요?"])
def test_first_person_small_talk_with_a_viewer_record_stays_in_conversation(prompt) -> None:
    pipeline = RecordingPipeline(plan={"kind": "chat", "intent": "knowledge", "scope": "unspecified"})
    data = ask(prompt, tab="knowledge", playback=AT_3S, pipeline=pipeline, mode="detailed")
    assert data["source"] == "conversation" and data["calls"] == ["plan", "converse"], data


def test_mixed_question_instruction_has_no_session_summary() -> None:
    store = SequenceStore(":memory:")
    body = {"session_id": SESSION, "scenario_id": "3", "sequence_id": "SEQ-001", "zone_id": "8", "sequence_setting": {"tool_type": "spray_gun"},
            "sequence_info": [], "sequence_result": {"avg_speed_mms": 987.6, "avg_distance_mm": 432.1, "avg_thickness_um": 25.0}}
    store.record(session_key=SESSION, sequence_id="SEQ-001", body_sha256="x", frames=0, received_at="now", body=json.dumps(body))
    pipeline = RecordingPipeline(document="[답변] 50% 겹쳐 칠합니다.")
    ask("그럼 거기부터 다시 칠하면 돼? 어떻게 칠해?", tab="coating", playback=AT_35S, store=store, pipeline=pipeline)
    instruction = str(dict(pipeline.kwargs)["run"]["instruction"])
    assert "987.6" not in instruction and "432.1" not in instruction


def test_mixed_record_only_answer_keeps_its_sequence_context() -> None:
    data = ask("이 기록에서 다음엔 어디 칠해? 거기는 어떻게 칠해?", tab="knowledge", playback=AT_3S, offline=True)
    assert data["source"] == "analysis" and data["sequence_context"] is not None


def test_correction_after_an_old_hazard_row_without_a_class_is_not_reclassified() -> None:
    # v3.4.4 이전 행에는 safety_class가 없다. 앞 질문 문장 자체가 위험 질문이면 다시 분류하지 않는다.
    store = SequenceStore(":memory:")
    _turn(store, "Zone 8 다 칠했으니 이제 압력 안 빼고 팁 바꿔도 되지?", "문서 답", "documents")
    pipeline = RecordingPipeline(plan={"kind": "correction", "intent": "coating_status", "scope": "playback"})
    data = ask("아까 답 이상한데", tab="coating", playback=AT_3S, store=store, pipeline=pipeline)
    assert data["source"] != "analysis" and "95%" not in data["answer"], data


def test_progress_shortcut_is_skipped_for_a_hazard_question_on_error_analysis(monkeypatch) -> None:
    # error_analysis에서는 safety가 거짓이어도 위험 종류가 있으면 진행 짧은 답을 주지 않는다(P14와 같은 조건).
    from trainer.demo_api.tests import viewer_record
    monkeypatch.setitem(viewer_record.TASKS, "error", "error_analysis")
    data = ask("누락한 Zone은? 압력 안 빼고 팁 바꿔도 돼?", tab="error", playback=AT_3S)
    assert "1~9" not in data["answer"] and "모두 칠했" not in data["answer"], data["answer"]
