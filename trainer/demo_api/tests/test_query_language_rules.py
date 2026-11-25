import pytest

from trainer.demo_api.query_language import TranslationUnavailableError, translate_text
from trainer.demo_api.query_language_rules import narrow_language_answer, unrelated_definition


@pytest.mark.parametrize(("prompt", "answer_substring"), [
    ("During the test, what does 'not yet sprayed' mean on the coating chart?", "도료가 도포되지 않은"),
    ("도막 차트에서 ‘미도장’은 무슨 뜻인가요?", "도료가 도포되지 않은"),
    ("Nếu tôi chưa phun, hệ thống có biết độ dày lớp sơn của tôi không?", "실제 도막 두께를 측정할 수 없"),
    ("'Zone 7을 완료했습니다'는 내가 적어본 UI 문구인데, 철자 자연스럽게 바꿔 줘.", "‘7번 구역 작업 완료’로 쓰면 자연스럽습니다."),
    ("‘Zone 16을 작업 완료했습니다.’라는 표시 문구를 자연스럽게 다듬어 줘.", "‘16번 구역 작업 완료’"),
])
def test_literal_copy_and_conditional_state_do_not_turn_into_session_queries(prompt, answer_substring):
    assert answer_substring in (narrow_language_answer(prompt) or "")


@pytest.mark.parametrize("prompt", [
    "지금 Zone 7을 완료했나요?",
    "Is Zone 7 not yet sprayed in my session?",
    "What is the coating thickness now?",
    "Before spraying, can I check the target film thickness?",
    "분사하기 전에 목표 도막 두께를 알 수 있나요?",
    "영어로 'Zone 7을 완료했습니다'라는 UI 문구를 자연스럽게 번역해줘.",
])
def test_actual_session_questions_are_not_covered_by_language_answers(prompt):
    assert narrow_language_answer(prompt) is None


def test_irrelevant_coating_context_does_not_replace_the_main_subject():
    prompt = "스프레이 작업 전에 도시락을 냉장고에 둘까 합니다. 상할까 걱정되는데 냉장 보관이 무슨 뜻인지 한 문장만요."
    assert unrelated_definition(prompt) == "냉장 보관은 음식이 쉽게 상하지 않도록 낮은 온도에서 차갑게 보관하는 것을 말합니다."
    assert unrelated_definition("냉장 보관한 도료의 점도는 어떻게 바뀌나요?") is None
    assert unrelated_definition("냉장 보관 중인 도료에 적용되는 안전 기준은 무엇인가요?") is None


def test_process_context_disambiguates_gun_without_changing_protected_measurements():
    class Translator:
        def translate_copilot_text(self, text, **_languages):
            assert text == "스프레이 건을 300 mm 떨어뜨리세요. 도막을 확인하세요."
            return "Giữ súng phun cách bề mặt 300 mm. Kiểm tra màng sơn."

    text = "건을 300 mm 떨어뜨리세요. 도막을 확인하세요."
    answer = translate_text(Translator(), text, source_language="auto", target_language="vi")
    assert "súng phun" in answer and "300 mm" in answer


def test_disambiguated_tool_must_not_be_returned_as_a_pen():
    class Translator:
        def translate_copilot_text(self, _text, **_languages):
            return "Hãy để bút cách 300 mm. Kiểm tra màng sơn."

    with pytest.raises(TranslationUnavailableError, match="spray gun"):
        translate_text(Translator(), "건을 300 mm 떨어뜨리세요. 도막을 확인하세요.",
                       source_language="auto", target_language="vi")


@pytest.mark.parametrize(("text", "distance", "thickness"), [
    ("건을 300 mm 떨어뜨리고, 도막은 160 µm를 목표로 하세요.", "300 mm", "160 µm"),
    ("스프레이 건을 28.5 cm 떨어뜨리고 도막 두께는 175.5 μm를 목표로 하십시오.", "28.5 cm", "175.5 μm"),
])
def test_vietnamese_standoff_and_thickness_instruction_preserves_both_imperatives_without_generation(text, distance, thickness):
    answer = translate_text(None, text, source_language="auto", target_language="vi")
    assert answer == f"Giữ súng phun cách bề mặt cần sơn {distance} và đặt mục tiêu độ dày màng sơn là {thickness}."
