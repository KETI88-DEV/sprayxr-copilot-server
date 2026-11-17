"""모델이 쓰는 도장·기록 답에 싣는 시뮬레이터 기본 설정(v3.5.1). 모델이 현장 조언을 지어내지 않게 이 훈련이 무엇인지 알려 준다.

지침이 아니라 사실이다. 유출 검사 지문에는 넣지 않고(dialogue._instruction_texts는 이 블록을 싣기 전 지침만 본다),
블록의 수치는 모델에게 알려 주기만 하고 근거로 쓰지 않는다(수치 근거 검사는 v3.5.0 그대로).
"""
from __future__ import annotations

import re
from typing import Final

from trainer.demo_api.copilot_intent import is_domain_question
from trainer.demo_api.record_query import has_record_marker

SIMULATOR_CONTEXT: Final = """[시뮬레이터 기본 설정]
- 이 훈련은 실물 스프레이건 모형을 들고 VR 속 가상 피도면에 분사하는 시뮬레이터입니다. 실제 도료는 나가지 않습니다.
- 이 시뮬레이터의 훈련모드(학습모드)는 1~10번, 총 10개입니다.
- 훈련생은 건, 노즐 코드(단건 519·523·527·531·535, 폴건 323·325), 압력(108~504 bar, 36 bar 단위), 신너 희석률(0~24%, 1.6% 단위), 월(그 달 기온 적용)을 고릅니다.
- 도막 두께는 점도·노즐·압력·이격거리·건 속도·분사 시간 등으로 예측 모델이 계산합니다. 분사 각도는 분사 형상에 영향을 줍니다. 습도는 설정 항목에 없습니다.
- 기본 정상 범위는 160~200µm이고 세션마다 다르게 정할 수 있습니다. Zone은 픽셀의 95%가 정상 범위 안이고 평균이 정상 범위일 때 완료입니다. 90/10 규칙 같은 측정점 규격으로 판정하지 않습니다.
- 이격거리·속도의 기준은 Zone별 전문가 측정 중앙값입니다. 전문가 속도는 이전 장비 기록이라 속도를 맞추라고 하지 않습니다. 거리나 속도를 몇 % 바꾸라는 고정 규칙은 없습니다.
- 단위: 위치 m, 이격거리 mm, 속도 mm/s, 각도 °, 도막 µm, 압력 bar, 온도 ℃."""

_BLASTING: Final = re.compile(r"블라스팅|blasting", re.I)
_TRAINING_MODE: Final = re.compile(r"(?:훈련|학습)\s*모드")


def is_blasting(prompt: str) -> bool:
    """블록의 스프레이건·도막 목표 문장은 블라스팅 세션에 맞지 않는다(query_runtime.retrieval_question과 같은 낱말).

    shortcut: 질문 글자만 본다(앞 대화의 블라스팅 맥락은 보지 않는다).
    """
    return bool(_BLASTING.search(prompt))


def wants_simulator_context(prompt: str) -> bool:
    """대화 경로: 도장 낱말이나 기록 표지가 있는 질문만. 주제 밖 잡담·블라스팅 질문은 지금처럼 둔다."""
    return not is_blasting(prompt) and (is_domain_question(prompt) or has_record_marker(prompt)
                                       or bool(_TRAINING_MODE.search(prompt)))
