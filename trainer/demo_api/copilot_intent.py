"""Semantic intent labels for the pose and Zone-order chat tabs."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable

logger = logging.getLogger(__name__)

POSE_INTENTS = {"pose_overall", "pose_worst_zone", "pose_zone", "pose_right", "pose_left", "knowledge"}
ORDER_INTENTS = {"order_reference", "order_actual", "order_missing", "order_adherence", "order_direction", "knowledge"}
_PAINTING_SUBJECT = re.compile(r"도막|두께|이격|속도|노즐|압력|안전|보호구|오버랩|표면\s*처리|전처리|절차"
    r"|도료|도장|스프레이|에어리스|팁|시너|희석|가사\s*시간|경화|건조|블라스팅|마스크|환기|훈련|(?i:vr)|어지럼|어지러|멀미|시험|실기|기능사|자격"
    r"|(?:스프레이|에어리스|도장)\s*건|건\s*(?:끝|각도|거리|속도|이동|선단)")
# 자세 낱말도 도장 질문의 낱말이다(966a3d0: "자세 종류로는 어떤게있어?"). 자세 낱말만 있는 질문은 개념 표지
# (_POSE_CONCEPT)가 있을 때만 일반 지식이다. "자세 유사도는 얼마야?"·"팔꿈치는 왜 지적됐어?"는 기록 질문이다(v3.6.2).
_POSE_SUBJECT = re.compile(r"자세(?!히|하게|한)|동작|도구|도포|팔꿈치|어깨|무릎|허리|posture|pose", re.I)
_POSE_CONCEPT = re.compile(r"종류|유형|뜻|의미|정의")
# 도장이 아닌 활동·분야. 자세·동작 낱말이 겹쳐도 주제 밖 질문이라 대화 경로에 둔다("주제 밖은 일부러 열어 둔다"
# 결정, 2026-11-14). 도장 낱말이 함께 있으면 도장 질문이다. 짧게 유지한다: 문제가 된 질문에서 고른 낱말만.
_OFF_TOPIC = re.compile(r"요리|요가|스쿼트|필라테스|헬스|운동|디스크|골프|축구|야구|등산|달리기|러닝|수영|춤|댄스")


def is_off_topic(prompt: str) -> bool:
    return bool(_OFF_TOPIC.search(prompt)) and not _PAINTING_SUBJECT.search(prompt) and not re.search(_COATING_SUBJECT, prompt, re.I)
_RECORD_REFERENCE = re.compile(r"영역|구역|zone|존|부위|현재|지금|이번|내[ 가]|어디|어느", re.I)


def is_coating_advice_question(prompt: str) -> bool:
    return bool(re.search(r"도막|두께|과도막|저도막|미도장|coating|thickness", prompt, re.I)
                and re.search(r"뭐|무엇|뜻|의미|정의|원인|이유|왜|방법|어떻게|예방|방지|원리|관계|영향|중요|하려면|개선|해야|고치|맞추|조절", prompt))


_COATING_SUBJECT = r"도막|두께|과도막|저도막|coating|thickness"
_COATING_STATE = r"과하|모자|부족|이상|상태|어때|분석|평가|얇|두껍|과다|괜찮|충분|양호|균일|불량|칠했|잘됐"


# 정의·규정을 묻는 말. "구역·부위" 같은 일반 명사만 있고 이 기록을 가리키는 말(지시어·소유·번호)이 없으면 지식 질문이다(2026-11-17 사례 16·17).
_RULE_ASK = re.compile(r"법칙|원칙|규정|규격|말하나요|말하는|말해요|뜻|정의|어떻게\s*판정")
_OWN_RECORD = re.compile(r"(?<![가-힣])(?:이|내|제|우리)\s*(?:zone|존|구역|기록|도막)|(?:zone|존|구역)\s*\d|현재|지금|이번", re.I)


def _asks_coating_state(prompt: str, subject: str, state: str) -> bool:
    if not re.search(subject, prompt, re.I):
        return False
    if _RULE_ASK.search(prompt) and not _OWN_RECORD.search(prompt):
        return False
    specific = _RECORD_REFERENCE.search(prompt)
    definition = re.search(r"뭐|무엇|뜻|의미|정의|원인|예방|방지", prompt)
    return bool(specific or (not definition and re.search(state, prompt)))


def is_coating_status_question(prompt: str) -> bool:
    if is_coating_advice_question(prompt):
        return False
    if _asks_coating_state(prompt, _COATING_SUBJECT, _COATING_STATE):
        return True
    # 넓힌 낱말(미도장·분포·등급)로만 걸리는 질문은 규정·절차를 묻는 지식 질문일 수 있어 문서 검색에 남긴다.
    return (_asks_coating_state(prompt, _COATING_SUBJECT + "|미도장", _COATING_STATE + "|분포|미도장|등급")
            and not re.search(r"규정|규격|절차|종류|기준|ISO|SSPC", prompt, re.I))


def is_domain_question(prompt: str) -> bool:
    # 자세 낱말은 넣지 않는다. 넣으면 서명된 기록의 자세 질문("자세 유사도는 얼마야?")이 대화 대신 문서로 간다(4c5da92와 다름).
    return bool(_PAINTING_SUBJECT.search(prompt) or re.search(_COATING_SUBJECT, prompt, re.I))


def is_explanatory_question(prompt: str, task_type: str | None = None) -> bool:
    """An explanation request must not be replaced by a bare measurement.

    This is an answer-shape guard, not a technical answer or topic classifier.
    Exact UI lookups retain their verified analysis responses.
    """
    if is_off_topic(prompt) or (task_type is not None and preset_intent(task_type, prompt) is not None):
        return False
    if re.search(r"원인|이유|왜|원리|방법|요령|장단점|차이점|영향|관계|부담.*줄|편하게.*(?:유지|하려|하려면)", prompt):
        return True
    # A request for a way of acting differs from asking which correction or
    # measurement the engine returned. Keep those direct analysis lookups.
    return bool("어떻게" in prompt and not re.search(r"어떻게\s*(?:고쳐|고치|교정|돼|되|나와|나왔)", prompt))


def is_general_painting_knowledge(prompt: str) -> bool:
    """기록·Zone·재생 위치를 가리키지 않는 도장 기술 질문(적정값·정의·원인·방법).

    일반 지식 표지가 있어야 하고, 평가형 어미·1인칭·지시어가 있으면 기록 후속 질문으로 본다
    ("속도는 괜찮았어?", "제 이격거리는 적당했나요?", "그럼 속도는?").
    자세 낱말만 있으면 개념 표지가 있어야 하고, 주제 밖 활동 낱말이 있으면 도장 지식이 아니다.
    """
    posture = _POSE_SUBJECT.search(prompt) and _POSE_CONCEPT.search(prompt)
    return bool(not is_off_topic(prompt) and (posture or ((is_coating_advice_question(prompt) or _PAINTING_SUBJECT.search(prompt))
                 and re.search(r"적정|기준|권장|표준|원인|이유|왜|뭐|무엇|뜻|의미|정의|종류|유형|어떤|차이|방법|어떻게|예방|방지|원리|얼마|순서|절차|단계|알려", prompt)))
                and not _RECORD_REFERENCE.search(prompt)
                and not re.search(r"시퀀스|기록|재생|아까|방금|(?<![가-힣])이\s*분사|었|았|했|어땠|괜찮|제\s|저의|나의|그럼|그\s", prompt))


def preset_intent(task_type: str, prompt: str) -> str | None:
    """Skip model classification only for exact, unambiguous UI questions."""
    normalized = re.sub(r"[\s?!.]", "", prompt).lower()
    presets = {
        "pose_recomm": {
            "전체작업유사도": "pose_overall", "전체작업유사도는": "pose_overall",
            "전체자세유사도": "pose_overall", "전체자세유사도는": "pose_overall",
            "가장문제가큰zone": "pose_worst_zone", "가장문제가큰zone은": "pose_worst_zone",
            "오른손을어떻게고쳐": "pose_right", "왼손을어떻게고쳐": "pose_left",
        },
        "trajectory_recomm": {
            "권장zone순서": "order_reference", "권장zone순서는": "order_reference",
            "권장순서는": "order_reference", "실제수행순서는": "order_actual",
            "누락한zone은": "order_missing", "순서준수율은": "order_adherence",
            "도포방향은": "order_direction",
        },
    }
    return presets.get(task_type, {}).get(normalized)


def fallback_intent(task_type: str, prompt: str) -> str:
    if is_coating_advice_question(prompt) or is_general_painting_knowledge(prompt):
        return "knowledge"
    lowered = prompt.lower()
    if task_type == "pose_recomm":
        if "오른" in lowered or "right" in lowered:
            return "pose_right"
        if "왼" in lowered or "left" in lowered:
            return "pose_left"
        if "zone" in lowered or "구역" in lowered:
            return "pose_worst_zone" if any(word in lowered for word in ("문제", "최악", "낮")) else "pose_zone"
        if any(word in lowered for word in ("자세", "유사도", "점수", "평균", "전체")):
            return "pose_overall"
        return "knowledge"
    if any(word in lowered for word in ("권장", "추천", "정석")):
        return "order_reference"
    if any(word in lowered for word in ("실제", "수행 순서", "칠한 순서")):
        return "order_actual"
    if any(word in lowered for word in ("누락", "안 칠", "빠진", "미수행")):
        return "order_missing"
    if "방향" in lowered:
        return "order_direction"
    if any(word in lowered for word in ("준수", "유사", "몇 퍼센트", "점수")):
        return "order_adherence"
    return "knowledge"


def classify_question_intent(
    task_type: str,
    prompt: str,
    llm_classifier: Callable[[str, str], str] | None,
) -> str:
    allowed = POSE_INTENTS if task_type == "pose_recomm" else ORDER_INTENTS
    if llm_classifier is not None:
        try:
            result = str(llm_classifier(task_type, prompt)).strip().lower()
            if result in allowed:
                # A tab is a UI preference. A confidently different industrial
                # subject must not become a Zone-order answer on a bad label.
                if (is_coating_advice_question(prompt) or (result != "knowledge" and fallback_intent(task_type, prompt) == "knowledge"
                        and (_PAINTING_SUBJECT.search(prompt) or _POSE_SUBJECT.search(prompt)))):
                    return "knowledge"
                return result
            logger.warning("copilot intent classifier returned invalid label task=%s label=%s", task_type, result)
        except Exception:  # noqa: BROAD_EXCEPT_OK - deterministic fallback keeps chat available
            logger.exception("copilot intent classification failed task=%s", task_type)
    return fallback_intent(task_type, prompt)
