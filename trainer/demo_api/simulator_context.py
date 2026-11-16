"""모델이 쓰는 도장·기록 답에 싣는 시뮬레이터 기본 설정(v3.5.1). 모델이 현장 조언을 지어내지 않게 이 훈련이 무엇인지 알려 준다.

지침이 아니라 사실이다. 유출 검사 지문에는 넣지 않고(dialogue._instruction_texts는 이 블록을 싣기 전 지침만 본다),
수치 근거 검사는 이 기록을 가리키지 않는 일반 문장에만 블록 값을 쓴다(`ungrounded_with_simulator`).
"""
from __future__ import annotations

import re
from typing import Any, Final

from trainer.demo_api.coating_query import _RECORD_SENTENCE
from trainer.demo_api.copilot_intent import is_domain_question
from trainer.demo_api.dialogue import _ASKED_ZONE, _MEASUREMENT_EXTENDED, _SPEED_UNIT, ungrounded_measurements
from trainer.demo_api.record_query import has_record_marker

SIMULATOR_CONTEXT: Final = """[시뮬레이터 기본 설정]
- 이 훈련은 실물 스프레이건 모형을 들고 VR 속 가상 피도면에 분사하는 시뮬레이터입니다. 실제 도료는 나가지 않습니다.
- 훈련생은 건, 노즐 코드(단건 519·523·527·531·535, 폴건 323·325), 압력(108~504 bar, 36 bar 단위), 신너 희석률(0~24%, 1.6% 단위), 월(그 달 기온 적용)을 고릅니다.
- 도막 두께는 점도·노즐·압력·이격거리·건 속도·분사 시간 등으로 예측 모델이 계산합니다. 분사 각도는 분사 형상에 영향을 줍니다. 습도는 설정 항목에 없습니다.
- 목표 도막은 1차 160µm, 2차 320µm입니다. Zone은 픽셀의 95%가 목표 두께 이상이고 저도막이 아니면 완료입니다. 90/10 규칙 같은 측정점 규격으로 판정하지 않습니다.
- 이격거리·속도의 기준은 Zone별 전문가 측정 중앙값입니다. 전문가 속도는 이전 장비 기록이라 속도를 맞추라고 하지 않습니다. 거리나 속도를 몇 % 바꾸라는 고정 규칙은 없습니다.
- 단위: 위치 m, 이격거리 mm, 속도 mm/s, 각도 °, 도막 µm, 압력 bar, 온도 ℃."""

_BLASTING: Final = re.compile(r"블라스팅|blasting", re.I)
# 블록의 스프레이건·도막 목표 문장은 블라스팅 세션에 맞지 않는다(query_runtime.retrieval_question과 같은 낱말).
# shortcut: 질문 글자만 본다(앞 대화의 블라스팅 맥락은 보지 않는다).
def _norm(text: str) -> str:
    return re.sub(r"\s+", "", text.replace("μ", "µ"))


# 블록에 쓴 값 → 그 값이 붙은 낱말(단독 값은 낱말과 함께 쓴 문장만 받친다. 범위는 쓴 그대로면 충분하다).
_WITH: Final = {"1.6%": "단위", "36bar": "단위", "95%": "목표|두께|면적|완료|픽셀", "160µm": "1차", "320µm": "2차"}
_ASSERTED: Final = {_norm(m[0]): re.compile(_WITH[_norm(m[0])]) if _norm(m[0]) in _WITH else None
                    for m in _MEASUREMENT_EXTENDED.finditer(SIMULATOR_CONTEXT)}
_ASSERTED_UNITS: Final = set(_SPEED_UNIT.findall(SIMULATOR_CONTEXT))
_BLOCK_WORDS: Final = {"목표", "기준"}  # 블록이 정의하는 낱말


def is_blasting(prompt: str) -> bool:
    return bool(_BLASTING.search(prompt))


def wants_simulator_context(prompt: str) -> bool:
    """대화 경로: 도장 낱말이나 기록 표지가 있는 질문만. 주제 밖 잡담·블라스팅 질문은 지금처럼 둔다."""
    return not is_blasting(prompt) and (is_domain_question(prompt) or has_record_marker(prompt))


# 높임 과거·2인칭("하셨·칠하신")은 훈련생이 한 일을 말한다. 기록 표지가 아니어도 기록 문장으로 본다.
_HONORIFIC: Final = re.compile(r"(?:하|쓰|칠하|뿌리|설정하|사용하|선택하|고르)셨|(?:하|칠하|설정하|사용하|선택하|고르)신")


def refers_to_record(sentence: str) -> bool:
    return bool(_HONORIFIC.search(sentence) or _RECORD_SENTENCE.search(sentence) or _ASKED_ZONE.search(sentence) or has_record_marker(sentence))


def ungrounded_with_simulator(answer: str, *, facts: Any, prompt: str, extended: bool = False) -> list[str]:
    """`ungrounded_measurements`에, 이 기록을 가리키지 않는 문장에 한해 블록이 쓴 그대로의 값(범위는 범위 그대로)을 허용한다."""
    invented: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", answer):
        found = ungrounded_measurements(sentence, facts=facts, prompt=prompt, extended=extended)
        if found and not refers_to_record(sentence):
            found = [m for m in found if not (_norm(m) in _ASSERTED and ((need := _ASSERTED[_norm(m)]) is None or need.search(sentence)))
                     and m not in _ASSERTED_UNITS and m not in _BLOCK_WORDS]
        invented += found
    return invented
