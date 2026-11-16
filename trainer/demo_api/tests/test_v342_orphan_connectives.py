"""앞 문장을 뺀 뒤 "따라서…/이로 인해…"로 시작하는 고아 문장이 남지 않는다(2026-11-14 훈련생 질문 시험).

운영 답: T28 "따라서 건 각도는 괜찮았습니다.", T37 "이로 인해 코팅 두께가 줄어들었을 가능성이 있습니다.",
T43-vi "Do đó, cần thực hiện kiểm tra bổ sung."(따라서 추가 점검이 필요합니다), T16-si "එබැවින්, …"(따라서 …).
대화 답의 출구 검사(drop_ungrounded 등)가 앞 문장을 빼고 접속어로 시작하는 문장만 남겼다. T28은 앞 문장의
"5.2°"가 자료에 있는데도(지표 단위 "deg") 지어낸 값으로 보여 빠졌다.
"""
from __future__ import annotations

import pytest

from trainer.demo_api import dialogue
from trainer.demo_api.tests.viewer_record import AT_3S, RecordingPipeline, ask, payload


def test_logical_connective_is_stripped_after_a_dropped_sentence() -> None:
    kept, dropped = dialogue.drop_sentences("도구 각도는 7.7°입니다. 따라서 건 각도는 괜찮았습니다.", lambda part: "7.7" in part)
    assert dropped and kept == "건 각도는 괜찮았습니다."


@pytest.mark.parametrize("orphan", ["이로 인해 코팅 두께가 줄어들었을 가능성이 있습니다.", "이 때문에 다시 칠해야 합니다.",
                                    "그 결과 도막이 얇아졌습니다."])
def test_sentences_that_point_back_to_a_dropped_sentence_are_dropped(orphan: str) -> None:
    kept, _ = dialogue.drop_sentences(f"Zone 9 이격거리는 999 mm였습니다. {orphan} 겹침을 50%로 맞추세요.", lambda part: "999" in part)
    assert kept == "겹침을 50%로 맞추세요."


def test_connectives_after_a_kept_sentence_are_untouched() -> None:
    text = "도막이 얇습니다. 따라서 한 번 더 칠하세요."
    assert dialogue.drop_sentences(text, lambda part: False) == (text, False)
    assert dialogue.drop_sentences("앞 문장. 둘째 문장. 따라서 셋째.", lambda part: part == "앞 문장.")[0] == "둘째 문장. 따라서 셋째."


def test_drop_ungrounded_no_longer_leaves_an_orphan() -> None:
    answer = dialogue.drop_ungrounded("Zone 9 이격거리는 999 mm였습니다. 이로 인해 코팅 두께가 줄어들었을 가능성이 있습니다.",
                                      facts={}, prompt="왜 그렇게 얇게 나온 거야?")
    assert not answer.startswith("이로 인해") and answer == "세션 기록으로는 확인할 수 없습니다."
    answer = dialogue.drop_ungrounded("도구 각도는 7.7°입니다. 따라서 건 각도는 괜찮았습니다.", facts={}, prompt="건 각도는 괜찮았어?")
    assert answer == "건 각도는 괜찮았습니다."


def test_metric_values_with_a_deg_unit_are_grounded() -> None:
    facts = {"selected_zone": payload()["dialogue_evaluation"]["zones"]["1:8"]}
    tool = next(m for m in facts["selected_zone"]["metrics"] if m["key"] == "tool_angle")
    sentence = f"도구 각도는 {tool['actual']}°로 숙련자 {tool['reference']}°와 거의 같습니다."
    assert dialogue.ungrounded_measurements(sentence, facts=facts, prompt="") == []


def test_t28_pose_tab_answer_keeps_its_measured_first_sentence() -> None:
    zone = payload()["dialogue_evaluation"]["zones"]["1:8"]
    tool = next(m for m in zone["metrics"] if m["key"] == "tool_angle")
    reply = f"Zone 8의 도구 각도는 {tool['actual']}°로 숙련자 {tool['reference']}°와 거의 같습니다. 따라서 건 각도는 괜찮았습니다."
    result = ask("건 각도는 괜찮았어?", tab="pose", playback=AT_3S, pipeline=RecordingPipeline(converse=reply), mode="detailed")  # 상세 분석의 전체 답. 빠른 안내는 한 줄 60자다(2026-11-16)
    # v3.5.0 설계 표 T28: 대화 모델 없이 서명된 도구 각도 행으로 답한다.
    assert result["calls"] == [] and result["source"] == "analysis"
    assert f"{tool['actual']}°" in result["answer"] and f"{tool['reference']}°" in result["answer"]
