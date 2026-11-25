"""Bounded factual answers for ambiguities that must survive summarization.

Coating terms and handling distinctions follow configs/term_rules.json and
configs/context_prompts.json. Simulator facts use the same source as assessment.
These answers do not imply that this request retrieved document evidence.
"""
from __future__ import annotations

import re

from .coating_assessment import GATE_COMPLETION_PERCENT, GRADE_WORDS, assess_readings
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
    r"(?:(?:우리|이)\s*(?:조선소|현장|회사|사업장|작업장)|사내).{0,28}(?:표준|시방|기준|규격)"
)
_SPEC_VALUE = re.compile(r"수치|한도|허용값|얼마|몇\s*(?:마이크로|미크론|[µμ]m|mm|퍼센트|%|도|$)", re.I)
_FILM_ACCEPTANCE_CRITERIA = re.compile(r"(?:판별|판정|합격|검수|수락)\s*기준|허용\s*(?:범위|두께)")
_SIMULATION = re.compile(r"시뮬레이터|가상\s*훈련|(?<![A-Za-z])VR(?![A-Za-z])", re.I)
_THICKNESS = r"\s*(?:[µμu]m|㎛|마이크로(?:미터)?|미크론)"
_VALUE = re.compile(rf"(?<![\d.-])(\d{{1,5}}(?:\.\d+)?)(?:{_THICKNESS})?\s*(?:이면|인데|라면|은|는)", re.I)
_NAMED_TARGET = re.compile(rf"목표(?:\s*도막|\s*두께|치|값)?\s*(?:가|는|이|를|을|=|:)?\s*(\d+(?:\.\d+)?)(?:{_THICKNESS})?", re.I)
_THICKNESS_GRADE = re.compile(r"과도막|저도막|(?:도막|두께).{0,12}정상|정상(?:인가|이야|일까)")
_EXCESS_FILM = re.compile(r"과도막|과다\s*도막|두껍게\s*(?:칠|발)|굵게\s*(?:왕창\s*)?칠")
_UNPAINTED = re.compile(r"미도장|안\s*칠한|칠하지\s*않")
_PROGRESS_OFFSET = re.compile(r"점수|진행률|메워|보상|상쇄")
_MOVE_CRITERION = re.compile(
    r"(?:이동\s*기준|(?:구역|zone|존)(?:을|를|으로)?\s*(?:이동|넘어가)(?:는|기)?\s*기준)"
    r"(?:이|은|는)?\s*\d+(?:\.\d+)?\s*(?:%|퍼센트|프로).{0,24}(?:뜻|의미)", re.I,
)
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
_SUBSTRATE_TEMP = re.compile(r"(?:강판|철판|피도면|(?:바탕|강재)?\s*표면)\s*(?:표면\s*)?온도")
_GAUGE = re.compile(r"도막\s*(?:게이지|측정기)|두께\s*(?:계|게이지|측정기)|DFT\s*(?:게이지|측정기)", re.I)
_BASE_COAT = re.compile(r"하도|밑칠|프라이머|중도", re.I)
_SEPARATE_COAT = re.compile(r"상도\s*(?:만|두께)|(?:상도|마감칠).{0,20}(?:따로|별도|분리)")
_ADHESION = re.compile(r"부착력|접착력")
_TAKE_READING = re.compile(r"측정|재는|잰|찍[었어]|게이지")
_ROUGHEST_SURFACE = re.compile(r"블라스[트팅]|거칠|요철|조도")
_RECOAT = re.compile(r"덧칠|재도장|상도\s*올|한\s*번\s*더\s*칠")
_TOUCH_DRY = re.compile(r"지촉\s*건조|손.{0,15}(?:안\s*묻|묻어나지\s*않)|만져도.{0,10}(?:안\s*묻|묻어나지\s*않)")
_POT_LIFE = re.compile(r"가사\s*시간|가용\s*시간|\bpot\s*life\b", re.I)
_VOLUME_RATIO = re.compile(r"(?:부피|용적|체적)\s*(?:비|기준)")
_MASS_RATIO = re.compile(r"전자\s*저울|저울|(?:질량|중량|무게)\s*(?:비|기준)")
_TOP_PEEL = re.compile(r"(?:상도|마감\s*층|위에\s*칠한\s*층).{0,28}(?:벗겨|들뜨|들떴|들리|박리|떼)")
_BOTTOM_REMAINS = re.compile(r"(?:하도|프라이머|아래\s*(?:칠|층|도막)|밑\s*(?:칠|층|도막)).{0,24}(?:그대로|붙어|남[아았])")
_FAN_TAIL = re.compile(
    r"테일링|핑거링|(?:분사|분무)\s*(?:패턴|자국|모양|폭).{0,22}(?:양\s*(?:끝|쪽)|가장자리).{0,16}(?:줄|띠)"
)
_AIRLESS = re.compile(r"에어리스|airless", re.I)
_AIR_ATOMIZATION = re.compile(r"공기\s*(?:분무|분사|스프레이)|압축\s*공기")
_VISIBLE_INSPECTION = re.compile(r"눈으로|육안|눈에\s*안\s*띄|보이지|안\s*보")
_COATING_THICKNESS_TOPIC = re.compile(r"도막|도료층|\b(?:DFT|WFT)\b", re.I)
_THICKNESS_CHECK = re.compile(r"두께|저도막|과도막|\b(?:DFT|WFT)\b", re.I)
_SUBSTRATE_THICKNESS = re.compile(r"(?:강판|철판|강재|모재|재료|판재)\s*(?:의\s*)?(?:잔존\s*)?두께")
_LOCATE_FILM_DEFECT = re.compile(r"구별|구분|판별|식별|어떻게\s*(?:찾|확인|판정)|찾아내|찾는\s*(?:법|방법)")
_REQUEST_METHOD = re.compile(r"방법|하는\s*법|찾는\s*법|어떻게|찾아내|알려|하려면")
_SALT = re.compile(r"염분|수용성\s*염|소금기")
_OIL = re.compile(r"기름|유분|그리스|기름때")
_ABRASIVE_MEDIA = re.compile(r"연마재|알갱이|그릿|쇼트")
_EDGE = re.compile(r"모서리|절단부|절단면|날\s*선|날카로운")
_COMPARE_INSPECTION = re.compile(
    r"같[아은]|같이|대신|바꿔|환산|변환|차이|다르|다른|별도|각각|하나만|비교|통과|생략|기준.{0,8}(?:이상|이하)"
)
_INSPECTION_DOC_LOOKUP = re.compile(r"(?:NCS|문서|자료|매뉴얼)에서|근거(?:를|가)?\s*(?:찾|인용)", re.I)
_INSPECTION_MEASUREMENTS = (
    (
        re.compile(r"격자\s*(?:절개|검사|시험)|크로스\s*컷|cross[ -]?cut", re.I),
        re.compile(r"풀\s*오프|pull[ -]?off|인장식", re.I),
        # ISO 2409 cross-cut and ISO 4624 pull-off: https://www.iso.org/standard/76041.html
        # https://www.iso.org/standard/83350.html
        "격자 절개 시험은 절개한 도막이 얼마나 떨어지는지 살펴 박리 저항을 등급으로 평가합니다. "
        "풀오프 시험은 도막을 당겨 떼는 데 필요한 인장 응력(MPa)을 측정합니다.",
    ),
    (
        re.compile(
            r"(?:조도|거칠기|표면|[µμ]m|마이크로미터).*(?<![A-Za-z])Ra(?![A-Za-z])|"
            r"(?<![A-Za-z])Ra(?![A-Za-z]).*(?:조도|거칠기|표면|[µμ]m|마이크로미터)",
            re.I,
        ),
        re.compile(r"(?<![A-Za-z])Rz(?![A-Za-z])", re.I),
        # ISO 21920-2 surface profile parameters: https://www.iso.org/standard/72226.html
        "Ra는 표면 단면의 평균적인 높이 편차이고, Rz는 요철의 높낮이 폭을 보는 지표입니다. "
        "단위는 같아도 표면의 높이 분포에 따라 두 지표의 관계가 달라집니다.",
    ),
    (
        re.compile(r"먼지.{0,8}양|분진.{0,8}양|먼지.{0,8}수량"),
        re.compile(r"입자\s*크기|입경"),
        # ISO 8502-3 tape method: https://www.iso.org/obp/ui/#iso:std:iso:8502:-3:ed-2:v1:en
        "먼지 양은 표면에 남아 테이프에 붙은 잔류 먼지가 얼마나 많은지를 봅니다. 입자 크기는 붙은 먼지 입자의 크기를 "
        "따로 분류합니다. 먼지가 적어도 큰 입자가 남을 수 있습니다.",
    ),
    (
        re.compile(r"도막\s*두께|건도막|\bDFT\b", re.I),
        re.compile(r"홀리데이|holiday|도막\s*(?:끊김|불연속|연속성)\s*검사", re.I),
        # AMPP discontinuity checks: https://protect-perform.ampp.org/fbe-application-inspection-common-challenges-and-how-to-avoid-them/
        "도막 두께 검사는 표본 위치의 두께를 잽니다. 홀리데이 검사는 핀홀이나 미도장처럼 "
        "도막이 끊긴 곳을 확인합니다. 두께가 적정해도 다른 위치의 끊김까지 확인된 것은 아닙니다.",
    ),
    (
        re.compile(r"(?:강판|강재|모재|철판)\s*(?:잔존|남은)\s*두께"),
        re.compile(r"도막\s*(?:두께|검사|게이지)|건도막|\bDFT\b", re.I),
        # DeFelsko documents substrate wall and paint measurement separately:
        # https://www.defelsko.com/resources/measurement-of-effects-of-erosion-and-corrosion
        "강판 잔존 두께는 부식·마모 후 남은 강재의 두께입니다. 도막 두께는 강판 위에 형성된 도료층의 두께입니다. "
        "단위를 바꿔도 측정한 대상은 그대로입니다.",
    ),
    (
        re.compile(r"색상\s*(?:코드|번호)|컬러\s*(?:코드|번호)"),
        re.compile(r"광택|유광|반광|무광"),
        # ISO 2813 gloss measurements: https://www.iso.org/standard/56807.html
        "색상 코드는 색을 지정하고 광택은 도장면에 비친 빛의 반사 정도를 봅니다. 같은 색에서도 "
        "유광과 반광 마감은 달라 광택 요구사항을 별도로 확인해야 합니다.",
    ),
)


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


def _film_inspection_method(prompt: str, *, prior_coating: bool) -> str | None:
    """Separate how to check film thickness from a query for session readings."""
    if _INSPECTION_DOC_LOOKUP.search(prompt) or not _THICKNESS_CHECK.search(prompt):
        return None
    coating = _COATING_THICKNESS_TOPIC.search(prompt)
    visual = _VISIBLE_INSPECTION.search(prompt) and (
        coating or (prior_coating and not _SUBSTRATE_THICKNESS.search(prompt))
    )
    how = (coating and re.search(r"저도막|과도막", prompt)
           and _LOCATE_FILM_DEFECT.search(prompt) and _REQUEST_METHOD.search(prompt))
    if not (visual or how):
        return None
    if _SIMULATION.search(prompt):
        return ("구역별 도막 히트맵의 수치나 범례를 이 세션의 정상 범위와 비교하세요. "
                "하한보다 얇은 부분은 저도막, 상한보다 두꺼운 부분은 과도막입니다.")
    # ISO 2808 methods and ISO 19840 sampling and acceptance scope:
    # https://www.iso.org/standard/71583.html and https://www.iso.org/standard/59523.html
    if _WET_FILM.search(prompt):
        return "육안으로 정확한 두께를 판정하기 어렵습니다. 도장 직후 습도막을 측정하고 도료 시방의 습도막 기준과 비교하세요."
    return ("육안으로는 도막 두께 수치를 확인하기 어렵습니다. 현장은 건조 후 두께계로 여러 지점을 재고 시방과 비교하세요. "
            "훈련에서는 구역별 도막 히트맵과 세션 정상 범위를 비교하세요.")


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
    if (
        _SALT.search(prompt) and _VISIBLE_INSPECTION.search(prompt) and re.search(r"검사|확인|측정|넘어", prompt)
        and re.search(r"판|피도면|표면|도막|바닷물|블라스[트팅]", prompt)
    ):
        return (
            "수용성 염분은 눈에 띄지 않아도 남을 수 있습니다. 승인된 방식으로 표면에서 염분을 채취해 확인하세요. "
            "예를 들어 브레슬법으로 채취한 용액의 전도도를 재고, 판정은 현장 시방의 기준을 적용합니다."
        )
    if blasting:
        return _blasting_preparation_answer(prompt)
    if re.search(r"탄소강|연강|철판", prompt) and re.search(r"스테인리스|스텐|STS", prompt, re.I) and re.search(r"디스크|연마석|연삭석", prompt):
        return (
            "탄소강을 갈고 남은 철 입자가 디스크를 통해 스테인리스에 옮아 녹이나 표면 오염을 일으킬 수 있습니다. "
            "스테인리스용으로 지정된 오염되지 않은 디스크를 사용하세요."
        )
    if (
        re.search(r"용접\s*비드", prompt) and _GRINDING.search(prompt)
        and re.search(r"(?:철판|모재|바탕)\s*(?:높이|면)|높이까지|평평하게|평탄하게", prompt)
    ):
        return (
            "용접 비드의 마감 수준은 용접·도장 시방과 현장 승인 조건을 따르세요. 돌출부나 스패터를 정리하되 "
            "주변 철판 높이까지 임의로 갈거나 비드·모재의 단면을 줄이면 안 됩니다."
        )
    return None


def _blasting_preparation_answer(prompt: str) -> str | None:
    if _OIL.search(prompt) and re.search(r"먼저|순서|탈지|그대로|날아갈", prompt):
        return (
            "표면의 기름이나 그리스는 승인된 방법으로 먼저 제거한 뒤 블라스팅하세요. 기름이 남은 채 연마재를 쏘면 "
            "주변 표면이나 회수 연마재가 다시 오염될 수 있습니다."
        )
    if re.search(r"(?:다시|새로|오늘).{0,18}(?:녹슬|녹\s*이|녹\s*이\s*생|발청)", prompt) and re.search(r"하도|도장|칠해", prompt):
        return (
            "어제의 검사 결과는 오늘의 표면 상태를 보증하지 않습니다. 지금 생긴 녹과 오염을 다시 확인하고, "
            "현장 시방이 요구하는 표면 상태로 되돌린 뒤 하도를 칠하세요."
        )
    if _ABRASIVE_MEDIA.search(prompt) and re.search(r"회수|재사용|다시.{0,16}(?:넣|써|사용)", prompt):
        return (
            "먼저 연마재가 재사용 가능한 종류인지 확인하세요. 회수분에서 도막 찌꺼기, 이물질과 파쇄된 미세 입자를 "
            "분리하고 청정도·입도 기준에 맞는 것만 재사용합니다."
        )
    if re.search(r"공기", prompt) and re.search(r"압력계|청정도|오일|수분|물기|검사.{0,15}뭐|검사.{0,15}무엇", prompt):
        return (
            "압력계는 공기 속 오일이나 물을 보여주지 않습니다. 분사 공기에 그런 오염물이 섞이면 표면을 다시 오염시킬 수 "
            "있으므로 승인된 공기 청정도 검사(예: 흰 흡수지 검사)로 따로 확인하세요."
        )
    if _EDGE.search(prompt) and re.search(r"둥글|라운딩|라운드", prompt):
        return (
            "블라스팅은 표면을 세정하고 조도를 만드는 공정입니다. 요구된 모서리 라운딩은 절단부 다듬기나 연삭 등 "
            "승인된 방법으로 따로 처리하고 현장 시방에 맞게 확인하세요."
        )
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


def _coating_inspection_answer(prompt: str) -> str | None:
    if (
        _SPRAY.search(prompt) and _SUBSTRATE_TEMP.search(prompt) and re.search(r"습도|이슬점|결로", prompt)
        and re.search(r"왜|이유|무슨\s*관련", prompt)
    ):
        return (
            "공기의 온도와 습도로 구한 이슬점을 강판의 실제 온도와 비교해 결로 위험을 확인합니다. "
            "강판 온도는 기온과 다를 수 있어 따로 재며, 도장 가능 여부는 현장 시방과 도료 기준으로 확인합니다."
        )
    if _BASE_COAT.search(prompt) and _SEPARATE_COAT.search(prompt) and _GAUGE.search(prompt) and _TAKE_READING.search(prompt):
        return (
            "일반 도막 게이지로 잰 한 값은 여러 층을 합친 전체 두께입니다. 상도 두께를 구하려면 같은 위치의 상도 전후 값을 "
            "비교하거나 그 도막에 맞는 층별 측정법을 써야 합니다."
        )
    if _ROUGHEST_SURFACE.search(prompt) and _GAUGE.search(prompt) and re.search(r"영점|교정|보정", prompt):
        return (
            "블라스팅으로 생긴 표면 요철이 두께계 값에 영향을 줍니다. 매끈한 시편에서 계측기를 검증한 뒤 "
            "규격에 따라 거칠기를 보정하거나, 장비·검사 절차가 지정한 기준면에서 확인하세요."
        )
    if (
        _GAUGE.search(prompt) and re.search(r"모서리|경계|곡면|곡률", prompt)
        and re.search(r"값[^.?!]{0,24}(?:튀|흔들|불안정)|값이\s*달라", prompt)
    ):
        return (
            "모서리나 좁은 곡면에서는 형상과 프로브 위치 때문에 두께계 값이 흔들릴 수 있습니다. "
            "그 값만으로 저도막을 확정할 수 없으니 장비 설명서와 시방에 맞는 프로브와 위치에서 다시 확인하세요."
        )
    if _ADHESION.search(prompt) and re.search(r"도막|두께", prompt) and re.search(r"합격|통과|대신", prompt):
        return (
            "도막 두께의 합격으로 접착력 합격을 판정할 수 없습니다. 접착력은 시방이 지정한 별도 시험과 합격 기준으로 확인합니다."
        )
    if _TOP_PEEL.search(prompt) and _BOTTOM_REMAINS.search(prompt):
        return (
            "하도는 붙어 있고 위층만 들렸다면 층간 박리 가능성이 있습니다. 이 모습만으로 철판과 하도의 접착 불량을 "
            "확정할 수 없습니다. 어느 경계에서 벗겨졌는지와 층간 오염·재도장 조건을 확인하세요."
        )
    return None


def _inspection_comparison_answer(prompt: str) -> str | None:
    """Compare distinct inspection results without treating them as training readings."""
    if not _COMPARE_INSPECTION.search(prompt) or _INSPECTION_DOC_LOOKUP.search(prompt):
        return None
    explanations = [
        explanation for first, second, explanation in _INSPECTION_MEASUREMENTS
        if first.search(prompt) and second.search(prompt)
    ]
    if not explanations:
        return None
    return " ".join([
        *explanations,
        "서로 대체할 수 없으므로 시방이 요구하는 항목과 시험법을 따라 기록하고 판정하세요.",
    ])


def _coating_preparation_answer(prompt: str) -> str | None:
    if _TOUCH_DRY.search(prompt) and _RECOAT.search(prompt):
        return (
            "손에 묻어나지 않는 지촉건조 상태만으로 덧칠 시점을 판단할 수 없습니다. "
            "제품 기술자료(TDS)에 적힌 해당 온도의 최소·최대 재도장 간격을 확인하세요."
        )
    if _POT_LIFE.search(prompt) and re.search(r"지났|넘었|초과|끝났|경과", prompt) and re.search(r"희석|신너|시너|써도|사용", prompt):
        return (
            "가사시간이 지난 배합물은 흐르더라도 사용하지 마세요. 희석제는 이미 진행된 경화 반응이나 가사시간을 되돌리지 못합니다. "
            "제조사 지침에 따라 남은 도료를 처리하고 새로 배합하세요."
        )
    if _VOLUME_RATIO.search(prompt) and _MASS_RATIO.search(prompt) and re.search(r"배합|혼합|경화제", prompt):
        return (
            "부피비를 저울의 질량비에 그대로 적용할 수는 없습니다. 성분의 밀도가 다르면 같은 부피비라도 질량비가 달라집니다. "
            "제품 지침의 부피비로 재거나 제조사가 제공한 질량비를 확인하세요."
        )
    if (
        re.search(r"하도|프라이머|밑칠", prompt) and re.search(r"상도|마감\s*칠", prompt)
        and re.search(r"하는\s*일|무슨\s*일|역할|어떤\s*일|어떤\s*기능", prompt)
        and not re.search(r"(?:문서|자료|매뉴얼|NCS)에서|근거를?\s*(?:찾|인용)", prompt, re.I)
    ):
        return (
            "하도는 바탕면에 잘 붙어 후속 도막의 바탕을 만들고 부식을 막는 데 도움을 줍니다. "
            "상도는 햇빛·비 등 외부 환경에서 도막을 보호하고 색상과 외관을 유지합니다. 구체 역할은 도장 시스템에 따라 달라집니다."
        )
    return None


def _spray_equipment_answer(prompt: str) -> str | None:
    if _FAN_TAIL.search(prompt) and re.search(r"압력|원인|왜", prompt):
        return (
            "분사 폭 양 가장자리에 진한 띠가 생기면 테일링일 수 있습니다. 압력 부족뿐 아니라 도료 점도, 팁 상태나 "
            "공급 상태도 확인하세요. 압력을 최대로 올리기보다 장비와 도료가 허용하는 조건에서 시험 분사 패턴을 확인하세요."
        )
    if _AIRLESS.search(prompt) and _AIR_ATOMIZATION.search(prompt) and re.search(r"무화|잘게|흩트|원리|차이|둘\s*다", prompt):
        return (
            "일반 공기분무 건은 압축공기로 도료를 잘게 흩뜨립니다. 에어리스는 가압된 도료가 작은 팁 구멍을 지나면서 "
            "무화됩니다. 펌프를 구동하는 공기나 일부 장비의 보조 공기는 무화 방식과 구분합니다."
        )
    return None


def technical_fact_answer(
    prompt: str,
    *,
    target_thickness_um: float | None = None,
    target_min_thickness_um: float | None = None,
    target_max_thickness_um: float | None = None,
    summarize: bool = True,
    prior_coating: bool = False,
) -> str | None:
    """Answer a specific ambiguity from domain facts or direct simulator behavior."""
    prompt = fix_typos(prompt).strip()
    process = explicit_work_process(prompt)
    if _SITE_SPEC.search(prompt) and (
        _SPEC_VALUE.search(prompt) or (_COATING_THICKNESS_TOPIC.search(prompt) and _FILM_ACCEPTANCE_CRITERIA.search(prompt))
    ):
        return "해당 현장 표준서 없이는 허용 수치를 확인할 수 없습니다. 정확한 기준명이나 표준서 번호와 해당 항목을 알려 주세요."
    if (answer := _inspection_comparison_answer(prompt)) is not None:
        return answer
    if (answer := _process_answer(prompt, process)) is not None:
        return answer
    if (answer := _film_measurement_answer(prompt, summarize=summarize)) is not None:
        return answer
    if process in ("blasting", "grinding"):
        if _GAUGE.search(prompt):
            return _coating_inspection_answer(prompt)
        return None
    if (answer := _film_inspection_method(prompt, prior_coating=prior_coating)) is not None:
        return answer
    if _MOVE_CRITERION.search(prompt):
        return (
            f"스프레이 훈련에서는 구역 진행률이 {GATE_COMPLETION_PERCENT:g}%에 이르면 다음 구역을 안내합니다. "
            "진행률은 각 화소에서 목표 두께를 얼마나 채웠는지의 평균입니다. "
            "목표에 도달한 화소만 세는 목표 두께 도달 면적은 별도 지표입니다."
        )
    if not _SIMULATION.search(prompt):
        return (
            _coating_definition_answer(prompt)
            or _coating_inspection_answer(prompt)
            or _coating_preparation_answer(prompt)
            or _spray_equipment_answer(prompt)
        )
    if _EXCESS_FILM.search(prompt) and _UNPAINTED.search(prompt) and _PROGRESS_OFFSET.search(prompt):
        return (
            "두껍게 칠한 부분은 옆의 미도장을 보상하지 않습니다. 각 화소의 목표 두께 도달률은 최대 100%이고, "
            "미도장 화소는 0%입니다. 구역 진행률은 이 값의 평균입니다."
        )
    return _simulator_grade(prompt, target_thickness_um=target_thickness_um,
                            target_min_thickness_um=target_min_thickness_um,
                            target_max_thickness_um=target_max_thickness_um)
