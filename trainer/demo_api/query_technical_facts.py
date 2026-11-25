"""Bounded factual answers for ambiguities that must survive summarization.

Coating terms and handling distinctions follow configs/term_rules.json and
configs/context_prompts.json. Simulator facts use the same source as assessment.
These answers do not imply that this request retrieved document evidence.
"""
from __future__ import annotations

import re

from .coating_assessment import GRADE_WORDS, assess_readings
from .query_runtime import explicit_work_process
from .term_rules import fix_typos, matched_terms

_SPRAY = re.compile(r"스프레이|도료|도막|도장|페인트|분사\s*건|칠[한하해했]", re.I)
_LOW_FILM = re.compile(r"저\s*도막|(?:도막|두께).{0,10}(?:얇|부족|미달)")
_VISIBLE_SAG = re.compile(r"흘러(?:\s*내리|\s*내려)|내려\s*흐|아래로\s*흘|눈물처럼")
_ROUGH_APPEARANCE = re.compile(r"거칠|가루|모래\s*같|모래처럼")
_DRY_SPRAY = re.compile(r"건조\s*(?:분무|스프레이)|드라이\s*스프레이|dry\s+spray", re.I)
_APPEARANCE_CHECK = re.compile(r"말인|맞(?:아|나|을|는)|일까|인가|있나|이야|\?")
_AT_ONE_SPOT = re.compile(r"(?:한\s*(?:곳|자리|군데|점)|같은\s*(?:곳|자리|점))")
_STAYING = re.compile(r"멈추|멈춰|머물|정지|계속\s*(?:대|누르|뿌리|분사)|오래\s*(?:대|누르|뿌리|분사)")
_SPRAY_GUN = re.compile(r"(?:분사|스프레이)\s*(?:건|총)|(?:도장|도색)\s*건")
_WET_FILM = re.compile(r"습\s*도막|젖은\s*(?:막|도막)|\bWFT\b", re.I)
_DRY_FILM = re.compile(r"건(?:조)?\s*도막|마른\s*(?:막|도막)|\bDFT\b", re.I)
_APPLIED_NOW = re.compile(r"방금\s*칠(?:하고|한)|(?:칠|도장|분사)\s*직후|칠하자마자|마르기\s*전")
_AFTER_DRYING = re.compile(r"(?:마른|건조(?:된|한)?)\s*(?:뒤|후)")
_MEASUREMENT = re.compile(r"잰|재(?:는|요|야|어|보)|측정")
_EQUIVALENT = re.compile(r"같[은아나]|똑같|차이|다르")
_SITE_SPEC = re.compile(
    r"(?:(?:우리|이)\s*(?:조선소|현장|회사|사업장|작업장)|사내).{0,28}(?:표준|시방|기준)"
)
_SPEC_VALUE = re.compile(r"수치|한도|허용값|얼마|몇\s*(?:마이크로|미크론|[µμ]m|mm|퍼센트|%|도|$)", re.I)
_SIMULATION = re.compile(r"시뮬레이터|가상\s*훈련|(?<![A-Za-z])VR(?![A-Za-z])", re.I)
_THICKNESS = r"\s*(?:[µμu]m|㎛|마이크로(?:미터)?|미크론)"
_VALUE = re.compile(rf"(?<![\d.-])(\d{{1,5}}(?:\.\d+)?)(?:{_THICKNESS})?\s*(?:이면|인데|라면|은|는)", re.I)
_NAMED_TARGET = re.compile(rf"목표(?:\s*도막|\s*두께|치|값)?\s*(?:가|는|이|를|을|=|:)?\s*(\d+(?:\.\d+)?)(?:{_THICKNESS})?", re.I)
_THICKNESS_GRADE = re.compile(r"과도막|저도막|(?:도막|두께).{0,12}정상|정상(?:인가|이야|일까)")
_EXCESS_FILM = re.compile(r"과도막|과다\s*도막|두껍게\s*(?:칠|발)|굵게\s*(?:왕창\s*)?칠")
_UNPAINTED = re.compile(r"미도장|안\s*칠한|칠하지\s*않")
_PROGRESS_OFFSET = re.compile(r"점수|진행률|메워|보상|상쇄")
_BLASTING = re.compile(r"블라스트|블라스팅|\bblasting\b", re.I)
_GRINDING = re.compile(r"그라인딩|그라인더|연삭|\bgrind(?:er|ing)\b", re.I)
_PAINT_TARGET_ACTION = re.compile(r"올려야|쌓아야|목표|기준|도료를?\s*더|덧칠|더\s*칠")
_COATING_WARNING = re.compile(r"(?:과도막|저도막|도막).{0,12}(?:경고|알림)")
_ANCHOR = re.compile(r"앵커\s*패턴|anchor\s+pattern", re.I)
_SHOT = re.compile(r"쇼트|숏|\bshot\b", re.I)
_GRIT = re.compile(r"그릿|\bgrit\b", re.I)
_COMPARISON = re.compile(r"뭐|무슨|차이|비교|다른|달라|같[아은나]|동일")
_LOW_FILM_DEFINITION = re.compile(
    r"^저\s*도막(?:\s*(?:이|은|이라면|이라는\s*게))?\s*"
    r"어떤\s*상태(?:인가(?:요)?|야|예요|죠)?[?？.!\s]*$"
)
_THICK_DEFINITION = re.compile(r"두껍게\s*(?:칠|발)|두꺼운\s*(?:도막|상태)")
_DEFINED_TERM = re.compile(r"(?:(?:이|라)는\s*(?:게|건)|뜻|의미|용어|말이(?:야|에요|인가))")


def _film_measurement_answer(prompt: str, *, summarize: bool) -> str | None:
    stages = _APPLIED_NOW.search(prompt) and _AFTER_DRYING.search(prompt) and _MEASUREMENT.search(prompt)
    definition = _WET_FILM.search(prompt) and _DRY_FILM.search(prompt) and _MEASUREMENT.search(prompt)
    if stages and _EQUIVALENT.search(prompt):
        return (
            "같은 숫자라도 판정 기준은 다릅니다. 도장 직후는 젖은 습도막, 건조된 뒤는 건도막입니다. "
            "용제가 휘발하면 건도막 두께는 보통 줄어들며, 체적 고형분 비율로 예상값을 계산합니다. "
            "건도막 목표는 건조 후 실측값으로 판정합니다."
        )
    if summarize and definition and _EQUIVALENT.search(prompt):
        return "습도막(WFT)은 도장 직후 마르기 전, 건도막(DFT)은 도막이 건조된 뒤의 두께를 잽니다."
    return None


def _simulator_grade(prompt: str, *, target_thickness_um: float | None,
                     target_min_thickness_um: float | None, target_max_thickness_um: float | None) -> str | None:
    target = _NAMED_TARGET.search(prompt)
    reading = _VALUE.search(prompt, target.end() if target else 0)
    if reading is None or not _THICKNESS_GRADE.search(prompt):
        return None
    session_band = any(value is not None for value in (
        target_thickness_um, target_min_thickness_um, target_max_thickness_um,
    ))
    try:
        thickness = float(reading[1])
        assessment = assess_readings(
            [thickness],
            target_thickness_um if session_band or target is None else float(target[1]),
            min_thickness_um=target_min_thickness_um,
            max_thickness_um=target_max_thickness_um,
        )
    except ValueError:
        return None
    scope = "이 세션에서는" if session_band else "제시한 목표를 적용하면" if target else "시뮬레이터 기본 설정에서"
    lower, upper = assessment["normal_band_um"]
    grade = GRADE_WORDS.get(assessment["point_grades"][0], "미도장")
    answer = f"{scope} 정상 범위는 {lower:g}~{upper:g}µm입니다. {thickness:g}µm는 {grade}입니다."
    return answer if session_band else answer + " 별도 범위를 설정한 세션에는 그 범위를 적용합니다."


def _process_answer(prompt: str, process: str | None) -> str | None:
    if _GRINDING.search(prompt) and _BLASTING.search(prompt) and _COMPARISON.search(prompt):
        return (
            "블라스팅은 연마재를 분사해 녹 등을 제거하고 표면 조도를 만듭니다. "
            "그라인딩은 회전 디스크를 표면에 접촉시켜 용접 비드처럼 돌출된 부위를 국소적으로 갈아냅니다."
        )
    if process == "grinding" and _COATING_WARNING.search(prompt):
        return "과도막·저도막 경고는 스프레이 도막 분석에서 생성합니다. 그라인딩 작업 기록에는 이 경고를 생성하지 않습니다."
    blasting = process == "blasting" or (process is None and bool(_BLASTING.search(prompt)))
    if blasting and _SPRAY.search(prompt) and _PAINT_TARGET_ACTION.search(prompt):
        return (
            "블라스팅은 연마재로 녹과 오염 등을 제거하는 표면 준비 작업입니다. "
            "도료를 쌓는 스프레이 훈련의 목표 도막 두께(예: 160µm)는 블라스팅에 적용하지 않습니다."
        )
    if _AT_ONE_SPOT.search(prompt) and _STAYING.search(prompt):
        if process == "grinding" or (process is None and _GRINDING.search(prompt)):
            return (
                "디스크가 회전하는 상태로 한 곳에 눌러 대면 국소 과삭이나 열 집중이 생길 수 있습니다. "
                "판단에는 실제 표면 자국과 누른 힘, 접촉 시간, 디스크 종류가 필요합니다."
            )
        if _SPRAY_GUN.search(prompt) and not blasting and process != "grinding":
            return (
                "건의 이동을 멈춰도 분사가 계속되면 같은 자리에 도료가 쌓여 국소 과도막이나 처짐이 생길 수 있습니다. "
                "분사까지 멈추면 그 자리에 도료가 더 쌓이지 않습니다."
            )
    if _ANCHOR.search(prompt) and blasting and re.search(r"순서|순번|외우", prompt):
        return "앵커 패턴은 블라스팅으로 표면에 생긴 미세한 요철(표면 조도)입니다. 도료 접착에 관여하며 작업 순서와는 관계없습니다."
    if _SHOT.search(prompt) and _GRIT.search(prompt) and _COMPARISON.search(prompt):
        # Abrasive manufacturer product specification: Winoa, Sample Card (SAE Spec).
        return "대표적인 강재 연마재에서 쇼트는 둥근 입자, 그릿은 각진 입자입니다. 모양에 따라 표면에 남기는 요철도 달라집니다."
    return None


def _coating_definition_answer(prompt: str) -> str | None:
    if _DRY_SPRAY.search(prompt) and _ROUGH_APPEARANCE.search(prompt) and _APPEARANCE_CHECK.search(prompt):
        return (
            "건조 분무는 도료 입자가 표면에 닿기 전에 용제가 휘발해 마른 상태로 붙는 현상입니다. "
            "가루가 붙은 듯 거칠 수 있지만 외관만으로 확정할 수는 없습니다."
        )
    if _SPRAY.search(prompt) and _VISIBLE_SAG.search(prompt) and _LOW_FILM.search(prompt):
        return (
            "도료가 아래로 흐르는 모습은 처짐에 가깝습니다. 젖은 도료가 한 곳에 많이 쌓이면 생길 수 있으나 "
            "외관만으로 원인을 확정할 수 없습니다. 저도막 여부는 건조 후 두께를 측정해 확인합니다."
        )
    if _LOW_FILM.search(prompt) and re.search(r"희석(?:제|량|을|비)|신너|시너", prompt) and re.search(r"원인|때문|그런|탓|맞", prompt):
        return (
            "과도한 희석은 가능한 원인 중 하나입니다. 건 이동 속도가 빠르거나 이격거리가 멀고, "
            "분사 패스 중첩이 부족해도 저도막이 생길 수 있습니다. 두께만으로 원인을 확정하기 어렵습니다."
        )
    # '어떤 상태' is liable to be mistaken for a personal status lookup. The
    # ordinary '저도막이 뭐야?' already works through the document path.
    if _LOW_FILM_DEFINITION.fullmatch(prompt):
        definition = next((rule["뜻"] for rule in matched_terms(prompt) if rule["용어"] == "저도막"), None)
        return f"{definition}입니다." if definition else None
    if re.search(r"저\s*도막", prompt) and _THICK_DEFINITION.search(prompt) and _DEFINED_TERM.search(prompt):
        definition = next((rule["뜻"] for rule in matched_terms(prompt) if rule["용어"] == "저도막"), None)
        if definition:
            return f"저도막은 {definition}입니다. 요구 범위보다 두꺼운 도막은 과도막입니다."
    return None


def technical_fact_answer(
    prompt: str,
    *,
    target_thickness_um: float | None = None,
    target_min_thickness_um: float | None = None,
    target_max_thickness_um: float | None = None,
    summarize: bool = True,
) -> str | None:
    """Answer a specific ambiguity from domain facts or direct simulator behavior."""
    prompt = fix_typos(prompt).strip()
    process = explicit_work_process(prompt)
    if _SITE_SPEC.search(prompt) and _SPEC_VALUE.search(prompt):
        return "해당 현장 표준서 없이는 허용 수치를 확인할 수 없습니다. 정확한 기준명이나 표준서 번호와 해당 항목을 알려 주세요."
    if (answer := _process_answer(prompt, process)) is not None:
        return answer
    if (answer := _film_measurement_answer(prompt, summarize=summarize)) is not None:
        return answer
    if process in ("blasting", "grinding"):
        return None
    if not _SIMULATION.search(prompt):
        return _coating_definition_answer(prompt)
    if _EXCESS_FILM.search(prompt) and _UNPAINTED.search(prompt) and _PROGRESS_OFFSET.search(prompt):
        return (
            "두껍게 칠한 부분은 옆의 미도장을 보상하지 않습니다. 각 화소의 목표 두께 도달률은 최대 100%이고, "
            "미도장 화소는 0%입니다. 구역 진행률은 이 값의 평균입니다."
        )
    return _simulator_grade(prompt, target_thickness_um=target_thickness_um,
                            target_min_thickness_um=target_min_thickness_um,
                            target_max_thickness_um=target_max_thickness_um)
