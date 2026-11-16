"""Live concise decoding must generate a real body before citing evidence."""
import re
import unittest

from trainer.scripts.answer_grammar import answer_regex, quick_answer_regex


class QuickAnswerBodyTests(unittest.TestCase):
    def test_whitespace_cannot_take_the_place_of_a_live_answer(self):
        for has_evidence in (False, True):
            suffix = "\n[근거] DOC-a-C0001" if has_evidence else ""
            for whitespace in ("", " ", "  ", "\t", "\u3000"):
                with self.subTest(evidence=has_evidence, whitespace=repr(whitespace)):
                    self.assertIsNone(re.fullmatch(quick_answer_regex(has_evidence=has_evidence),
                                                  "[답변] " + whitespace + suffix))

    def test_a_short_body_can_include_korean_units_and_punctuation(self):
        for body in ("노즐 상태와 분사 조건을 확인하세요.", "160~200µm 범위입니다.",
                     "A 조건을 먼저 확인하세요.", "두께는 42.3µm입니다."):
            for has_evidence in (False, True):
                suffix = "\n[근거] DOC-a-C0001" if has_evidence else ""
                with self.subTest(body=body, evidence=has_evidence):
                    self.assertIsNotNone(re.fullmatch(quick_answer_regex(has_evidence=has_evidence),
                                                     "[답변] " + body + suffix))

    def test_live_body_still_excludes_transport_and_extra_section_tags(self):
        for body in ("[답변] [원인] 추가 섹션", "[답변] </think>",
                     "[답변] 답변.\n[근거] DOC-a-C0001\n[근거] DOC-b-C0002"):
            self.assertIsNone(re.fullmatch(quick_answer_regex(has_evidence=True), body))

    def test_evaluation_grammar_keeps_its_existing_multiline_contract(self):
        text = "[답변] 답변 첫 줄.\n답변 둘째 줄.\n[근거] DOC-a-C0001, DOC-b-C0002"
        self.assertIsNotNone(re.fullmatch(answer_regex(has_evidence=True), text))


if __name__ == "__main__":
    unittest.main()
