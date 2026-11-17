"""실시간 안내의 한 줄 유틸리티와 요약하지 않는 Query 답변을 구분한다."""
from __future__ import annotations

import logging

import pytest

from trainer.demo_api.dialogue import conversation_messages
from trainer.demo_api.copilot_session_context import build_query_instruction
from trainer.demo_api.quick_answer import QUICK_LIMIT, one_line
from trainer.demo_api.reference_order_query import reference_order_answer
from trainer.demo_api.safety_guard import HIGH_PRESSURE_ANSWER, fixed_answer_translation
from trainer.demo_api.tests.viewer_record import RecordingPipeline, ask, payload

def _no_arm() -> dict:
    """팔 자세 채점이 오지 않은 기록(운영 2026-11-16 오전 질문들의 모양)."""
    data = payload()
    data["arm_pose_evaluation"] = None
    data["query_context_signatures"].pop("arm_pose", None)
    return data


def _one_line(answer: str) -> None:
    assert "\n" not in answer and len(answer) <= QUICK_LIMIT, (len(answer), answer)


# ---------------------------------------------------------------- R1 공통 출구
def test_lines_and_bullets_become_one_line_of_whole_sentences() -> None:
    assert one_line("첫 문장입니다.\n- 둘째 문장입니다.\n2) 셋째입니다.") == "첫 문장입니다. 둘째 문장입니다. 셋째입니다."


def test_sentences_are_kept_from_the_front_up_to_sixty_characters() -> None:
    first, second = "가" * 39 + ".", "나" * 19 + "."  # 40 + 1 + 20 = 61
    assert one_line(first + " " + "나" * 18 + ".") == first + " " + "나" * 18 + "."  # 정확히 60자
    assert one_line(first + " " + second) == first


def test_a_first_sentence_over_the_limit_is_not_cut() -> None:
    long = "가" * 70 + "."
    assert one_line(long + " 다음 문장입니다.") == long


def test_long_zone_flows_keep_their_head_and_count() -> None:
    answer = one_line("권장 작업 순서는 Zone " + " → ".join(str(zone) for zone in range(1, 40)) + "입니다.")
    _one_line(answer)
    assert answer.startswith("권장 작업 순서는 Zone 1→2→3") and "(39개)" in answer and answer.endswith("입니다.")
    assert one_line("순서는 Zone 7 → 4 → 9입니다.") == "순서는 Zone 7→4→9입니다."


def test_quick_preserves_long_conversation_sentences(caplog: pytest.LogCaptureFixture) -> None:
    reply = "오늘 점심은 " + "아주 " * 25 + "맛있는 것을 드세요.\n둘째 문장도 그대로 보여 드립니다."
    with caplog.at_level(logging.WARNING, logger="trainer.demo_api.copilot_router"):
        found = ask("오늘 점심 뭐 먹을까요?", tab="knowledge", pipeline=RecordingPipeline(converse=reply))
    assert found["answer"] == reply.replace("  ", " ")
    assert not any("quick_answer_over_limit" in record.getMessage() for record in caplog.records)


def test_detailed_answers_pass_through_with_their_lines() -> None:
    detailed = ask("도막이 과하거나 모자란 영역이 있어?", tab="coating", mode="detailed")["answer"]
    assert "\n" in detailed and "저도막: 8개 —" in detailed and len(detailed) > QUICK_LIMIT


# ---------------------------------------------------------------- R2 코드 틀 답(운영에서 답이 잘려 나간 일곱 유형과 점검에서 고친 틀)
@pytest.mark.parametrize("prompt, tab, data, quick_words, detailed_words", [
    # 1. 원인: "거리로는 설명하기 어렵다"가 둘째 문장 뒤에 있었다
    ("왜 이렇게 얇게 나왔어?", "knowledge", None, ("Zone 8", "저도막", "251 mm", "거리로는 설명하기 어렵"),
     ("실측값은 이동 속도", "그래서 얇게 나온 이유를 거리로는 설명하기 어렵습니다.")),
    # 2. 도막 상태: 첫 줄이 기준 설명뿐이었다
    ("도막이 과하거나 모자란 영역이 있어?", "coating", None, ("저도막 8곳", "과도막 0곳", "160µm"), ("저도막: 8개 —", "과도막: 없음")),
    # 3. 속도 처방: 이전 장비 판단이 둘째 문장 뒤에 있었다
    ("나 숙련자보다 느리던데, 속도를 숙련자만큼 올려야 해?", "knowledge", None, ("느린 것은 맞습니다", "이전 장비", "맞추지 않습니다"),
     ("전문가 기록은 이전 장비 기준이라 맞춰 바꾸라고 하지 않습니다.",)),
    # 4. 이 Zone 교정: 교정 문장이 유사도 문장 뒤에 있었다(팔 채점이 없는 기록)
    ("이 Zone에서 어떻게 고쳐?", "pose", "no_arm", ("Zone 8", "85%", "29.9cm 멀리"), ("현재 재생 구간(Map 1 · Zone 8)의 작업 유사도는",)),
    # 5. 다 칠했으니 끝?: 완료 개수가 셋째 문장이었다
    ("다 칠했으니까 이제 끝난 거죠?", "order", None, ("아니요", "9곳 중 0곳", "95%"), ("다 칠한 것과 완료는 다릅니다.",)),
    # 6. 평균 정상이니 끝?: 범위 안 면적이 둘째 문장 뒤였다
    ("Zone 1은 정상이라고 나왔으니까 다 끝난 거지?", "knowledge", None, ("아니요", "Zone 1", "47.9%", "95%", "보완"),
     ("평균 두께가 기준 안이어도 완료는 칠한 면적으로 봅니다.",)),
    # 7. 속도·이격거리 비교: 거리 문장이 둘째 줄이었다
    ("이 Zone 속도랑 이격거리 숙련자랑 비교해줘.", "knowledge", None, ("Zone 8", "251 mm", "550", "2,230", "3,514", "이전 장비"),
     ("Zone 8의 이동 속도는 2,230 mm/s", "Zone 8은 251 mm 거리에서 칠했습니다.")),
    # 점검에서 고친 틀
    ("뭐부터 고쳐야 돼? 딱 하나만 콕 집어줘.", "knowledge", None, ("Zone 3부터 보완", "0%", "95%"), ("범위 밖 면적은 하한",)),
    ("숙련자는 보통 어느 Zone에서 시작해서 어디서 끝내요?", "order", None, ("Zone 7에서 시작", "Zone 9(3회)에서 끝"),
     ("숙련자의 마지막 자리는 Zone 9 3회",)),
    ("거리가 멀어서 얇게 나온 거지?", "knowledge", None, ("아니요", "251 mm", "가까이서"), ("가까이 칠하면 도막은 보통 두꺼워집니다.",)),
    ("제 자세 괜찮아요? 잘하고 있는 거예요?", "pose", None, ("평균 54%", "70%"), ("점수만으로 우열을 판단하지 않습니다",)),
    ("오른손은 어케 고쳐야돼", "pose", None, ("오른팔 교정", "20cm 더 떼어"), ("오른쪽 손목 교정은 없었습니다.",)),
    ("가장 문제가 큰 Zone은?", "pose", None, ("Zone 9(50%)",), ("숙련자 범위를 벗어났습니다",)),
    ("그럼 오른쪽 팔꿈치는 몇 도 더 펴야 돼?", "pose", "no_arm", ("팔 자세 채점 결과가 없어",), ("자세 채점이 끝난 뒤 다시 질문해 주세요.",)),
])
def test_quick_record_answers_keep_the_full_explanation(prompt, tab, data, quick_words, detailed_words) -> None:
    record = _no_arm() if data == "no_arm" else None
    quick = ask(prompt, tab=tab, data=record)["answer"]
    # Missing arm scores keep the explicit quick notice; no work score is substituted.
    expected = quick_words if data == "no_arm" and "몇 도" in prompt else detailed_words
    for word in expected:
        assert word in quick, (word, quick)
    detailed = ask(prompt, tab=tab, data=record, mode="detailed")["answer"]
    for word in detailed_words:
        assert word in detailed, (word, detailed)


def test_a_reference_plan_for_two_processes_fits_one_line() -> None:
    class Engine:
        def reference_plan(self, *, scenario_id: str, process: str) -> dict:
            return {"process": process, "flow": list(range(1, 14)), "reference_session_count": 6}

    quick, _ = reference_order_answer(Engine(), scenario_id="3", process=None)
    detailed, _ = reference_order_answer(Engine(), scenario_id="3", process=None, detailed=True)
    line = one_line(quick)
    _one_line(line)
    assert "등록 기준 순서" in line and "스프레이 Zone 1→2" in line and "블라스팅 Zone 1→2" in line and "(13개)" in line
    assert detailed.startswith("학습 모드 3 현재 등록된 기준 순서 — 스프레이: Zone 1 → 2") and "전문가(스프레이 6회" in detailed


def test_the_quick_high_pressure_answer_keeps_the_complete_safety_answer_and_translation() -> None:
    quick = ask("건 앞에 손 갖다 대도 괜찮아?", tab="knowledge", pipeline=RecordingPipeline(converse="네, 괜찮습니다."))["answer"]
    assert quick == HIGH_PRESSURE_ANSWER and "막지 마세요" in quick and "압력을 뺀 뒤" in quick
    assert ask("건 앞에 손 갖다 대도 괜찮아?", tab="knowledge", mode="detailed")["answer"] == HIGH_PRESSURE_ANSWER
    assert fixed_answer_translation(quick, "en") == fixed_answer_translation(HIGH_PRESSURE_ANSWER, "en") is not None
    assert fixed_answer_translation(one_line(HIGH_PRESSURE_ANSWER), "en") is None and fixed_answer_translation("", "en") is None


def test_a_safety_caution_does_not_replace_the_quick_document_answer() -> None:
    document = ("[답변] 도장할 때는 작업 전에 장비와 주변 환경을 충분히 점검하고 동료와 함께 작업 순서를 확인해야 합니다. 둘째 문장입니다.\n"
                "[근거] DOC-doc_0020-C0071")
    chunks = [{"chunk_id": "DOC-doc_0020-C0071", "text": "보호구", "rerank_score": 9.0}]
    caution = "도장 중에는 호흡보호구·보안경·보호장갑을 착용하세요."
    quick = ask("보호구 없이 칠해도 돼?", tab="knowledge", pipeline=RecordingPipeline(document=document, chunks=chunks))["answer"]
    assert quick.startswith("도장할 때는") and "둘째 문장입니다." in quick and quick.endswith("\n" + caution)
    detailed = ask("보호구 없이 칠해도 돼?", tab="knowledge", mode="detailed", pipeline=RecordingPipeline(document=document, chunks=chunks))["answer"]
    assert detailed.endswith("\n" + caution) and detailed.startswith("도장할 때는")


# ---------------------------------------------------------------- R3 모델 답 지시문
def test_query_model_instructions_preserve_complete_answers_in_both_modes() -> None:
    quick = conversation_messages("q", recent_turns=[], facts={}, response_mode="quick")[0]["content"]
    detailed = conversation_messages("q", recent_turns=[], facts={}, response_mode="detailed")[0]["content"]
    for content in (quick, detailed):
        assert "60자" not in content and "두 번째 문장으로 요약" not in content
        assert "전체 목록·Zone 순서는 유지" in content
    for concise in (True, False):
        instruction = build_query_instruction("q", client_context=None, session_summary="", recent_turns=[], response_mode="quick", concise=concise)
        assert "60자" not in instruction and "2개 이내" not in instruction and "전체 목록·Zone 순서는 유지" in instruction
    assert "전체 목록·Zone 순서는 유지" in build_query_instruction("q", client_context=None, session_summary="", recent_turns=[], response_mode="detailed")
