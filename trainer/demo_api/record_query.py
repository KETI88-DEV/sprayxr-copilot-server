"""뷰어 기록 질문을 조각·코드·Zone·전제로 가르는 규칙 분류(v3.5.0). 모델을 부르지 않는다.

질문을 조각으로 나누고 조각마다 "기록의 어느 사실을 묻는가"(코드)를 정한다. 답은 만들지 않는다(`record_answer`).
각 정규식 줄 끝 주석은 그 줄이 걸리는 시험 문항(판정_한국어.csv·질문시험_v340.csv 번호)이다.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

from trainer.demo_api.coating_query import _CAUSES, _EFFECTS
from trainer.demo_api.copilot_intent import preset_intent
from trainer.demo_api.viewer_progress import progress_question

CODES: Final = ("progress_status", "progress_next", "coating_extreme", "coating_over", "completion", "distance_compare",
                "distance_filter", "speed", "work_metric", "pose_judgement", "order_shape", "zone_exists",
                "external_standard", "one_priority", "cause")
# zone_exists(R10)는 분류가 내지 않는다. 분류는 기록의 Zone 목록을 모르므로, 물은 Zone이 기록에 없으면
# answer_part가 코드와 상관없이 R10으로 답한다. work_metric의 영역 이름은 part.metric이다(part_domain).
CODE_DOMAIN: Final[Mapping[str, str]] = {
    "coating_extreme": "coating", "coating_over": "coating", "completion": "coating", "external_standard": "coating", "cause": "coating",
    "distance_compare": "distance", "distance_filter": "distance", "speed": "speed", "work_metric": "work_metric",
    "pose_judgement": "pose", "order_shape": "order", "progress_status": "order", "progress_next": "order",
    "zone_exists": "overall", "one_priority": "overall",
}
_JUDGEMENT_CODES: Final = {"completion", "pose_judgement", "external_standard"}
_NEEDS_MARKER: Final = {"completion", "pose_judgement", "external_standard", "coating_extreme", "coating_over", "cause",
                        "speed", "distance_compare", "distance_filter", "work_metric"}
_ZONE_CODES: Final = {"completion", "speed", "distance_compare", "cause", "progress_status", "work_metric"}  # 지시어·"지금"으로 Zone을 받는 코드
_FALLBACK_CODES: Final = {"speed", "distance_compare", "cause"}  # 지시어 없이도 한 Zone이 꼭 필요한 코드

# 라우터와 함께 쓰는 정규식(copilot_router가 여기서 가져간다).
# 대화 계획 모델이 속도·Zone 질문을 도포 방향으로 분류한다(2026-11-11 Spark: "지금 속도는 적당해?"에 방향을 답함).
DIRECTION_WORD: Final = re.compile(r"방향|수평|수직|대각|가로|세로|패턴|direction", re.I)
# 손·손목·팔꿈치·위팔 질문. "오른쪽으로 칠해"처럼 방향만 말한 질문은 걸리지 않게 손·팔을 함께 본다.
BODY_PART: Final = re.compile(r"(?:오른|왼)\s*(?:쪽\s*)?(?:손|팔)|팔꿈치|손목|위팔|(?:right|left)\s*(?:hand|arm)|elbow|wrist", re.I)
WORST_ZONE: Final = re.compile(r"(?:zone|존|구역)\s*별|(?:가장|제일)\s*문제|최악|(?:어디|어느).*(?:문제|나쁘|나빠|안\s*좋|낮|못)", re.I)
_DIRECTION_ONLY: Final = re.compile(DIRECTION_WORD.pattern.replace("|패턴", ""), re.I)  # "패턴"은 작업 지표(paint_pattern)다

# 기록 표지. "저도막"의 저+도처럼 낱말 속 글자는 걸리지 않게 앞뒤를 본다.
_PERSON: Final = re.compile(r"(?<![가-힣])(?:내|제|저|나|우리)(?:가|는|한테|\s)")  # T20 T21 T44 T45
_RECORD_REF: Final = re.compile(r"기록|이번|여기|화면에|(?:zone|존|구역)\s*\d|이\s*(?:zone|존|구역)|지금|현재|방금", re.I)  # T39 T50 v22
_PAST: Final = re.compile(r"칠했|칠한|뿌렸|뿌린|나왔|나온|나와|됐|끝난|끝났|완료|합격|떠\s*있|정상이라고\s*나")  # T34 T36 T42
_EVALUATED: Final = re.compile(r"괜찮았|어땠|적당했|잘한|나쁜\s*거|맞지|거지")  # T27 T28 T43
_SPATIAL: Final = re.compile(r"거기|거긴|그곳|그\s*(?:zone|존|구역)", re.I)  # T38 T43
_ANAPHORA: Final = re.compile(r"그럼|그렇게")  # T37 T41 T46 — 앞 질문이 있을 때만 표지다
_HERE: Final = re.compile(r"지금|이\s*(?:zone|존|구역)|여기|이렇게|현재", re.I)  # T39 v19
_KNOWLEDGE: Final = re.compile(r"적정|기준|권장|표준|정의|뜻|의미|원리|원래|문서|규칙|규격|어떻게\s*칠|왜\s*생")  # T38 T41 T45 T50 v33
_ZONE_NUMBERS: Final = re.compile(r"(?:zone|존|구역)\s*(\d{1,2}(?:\s*(?:,|와|과|이랑|랑|하고|및)\s*\d{1,2})*)", re.I)  # T33

_EFFECT_ANY: Final = "|".join(pattern for _, pattern in _EFFECTS)
# "뭐부터 해야 돼요?"는 작업 절차 질문이라 "고치"가 붙어야 한다.
_ONE_PRIORITY: Final = re.compile(r"(?:뭐|무엇)부터\s*(?:고|개선)|먼저\s*(?:고쳐|신경)|콕\s*집")  # T49 v22 v340-40
_ONLY_ONE: Final = re.compile(r"딱\s*하나|하나만|콕\s*집")  # 질문 전체에 걸린다
_EXTERNAL: Final = re.compile(r"PSPC|90/10|ISO|SSPC|IMO")  # T50
_EXTERNAL_ASK: Final = re.compile(r"합격|판정|통과|충족")
_NOT_COMPLETION: Final = re.compile(r"몇|얼마|어떻게|조건|코트")  # "몇 코트까지 칠해야 끝난 거예요?"는 기준을 묻는다
_ORDER_SUBJECT: Final = re.compile(r"숙련자|전문가|zone|존|구역", re.I)  # 시작·끝 낱말만으로는 작업 절차 질문과 가를 수 없다
_COMPLETION: Final = re.compile(r"끝난|끝났|끝낸|완료|다\s*칠했|다시\s*(?:안\s*)?(?:가|칠)")  # T34 T38 T43
_STATUS_ASKED: Final = re.compile(r"빼먹|빠뜨|빠진|누락|미수행|안\s*칠|못\s*칠|덜\s*칠")  # T33(Zone 번호가 있으면 "왜"가 있어도 상태 질문)
_WHY: Final = re.compile(r"왜|원인|이유|때문")
_EXTREME: Final = re.compile(r"제일|가장|최고|최저")  # T36
_THICK: Final = re.compile(r"두꺼|두껍|과도막|과부착")
_THIN: Final = re.compile(r"얇|저도막")
_WHERE: Final = re.compile(r"있|어디|어느|몇|데")  # T42 T46 v33
_ORDER_WORD: Final = re.compile(r"순서|시작|끝내|마지막")
_ORDER_OPERATORS: Final = (("agreement", r"일치|끼리"), ("adherence", r"준수"), ("allowed", r"정해져|아무\s*데|순서가\s*있"),  # T32 T32 T31
                           ("start", r"시작"), ("end", r"끝내|마지막"))  # T35
_FILTER: Final = re.compile(r"(?:숙련자|전문가)\S*\s*보다\s*(멀|가까|비슷)")  # T46
_FILTER_OPERATOR: Final = {"멀": "higher", "가까": "lower", "비슷": "similar"}
_DISTANCE: Final = re.compile(r"이격|거리")
_SPEED: Final = re.compile(r"속도")
_METRIC_ASK: Final = re.compile(r"괜찮|어땠|어때|적당|몇|얼마")
_METRICS: Final = (("tool_angle", r"각도"), ("posture", r"몸통|기울기|상체"), ("paint_pattern", r"겹침|오버랩|패턴"))  # T28
_POSE_EVALUATION: Final = re.compile(r"괜찮|잘하|잘한|나쁜|좋은|맞지|못한|부족한")  # T20 T21 T27

# 전제: 질문이 사실처럼 깔고 들어오는 말. 기록과 맞는지는 answer_part가 가른다.
_NAMED_NUMBER: Final = re.compile(r"(자세\s*유사도|작업\s*유사도|순서\s*준수율)\D{0,8}?(\d+(?:\.\d+)?)\s*%\s*(?:면|이면|인데|라|이라|인|였|이었)")  # T27
_NAMED_SUBJECT: Final = {"자세 유사도": "pose_similarity", "작업 유사도": "work_similarity", "순서 준수율": "order_adherence"}
_VISITED_ALL: Final = re.compile(r"다\s*칠했")  # T34
_GRADE_NORMAL: Final = re.compile(r"정상\s*이라고\s*나")  # T43
_COMPARISON: Final = re.compile(r"(?:숙련자|전문가)\S*\s*보다\s*(느리|느렸|느려|빠르|빨랐|빨라|멀|가까)\S*?(?:던데|는데|다고)")  # T44
_COMPARISON_SUBJECT: Final = {"느": "speed_slower", "빠": "speed_faster", "빨": "speed_faster", "멀": "distance_farther", "가": "distance_closer"}


@dataclass(frozen=True)
class Premise:
    kind: str            # "named_number" | "cause" | "conclusion" | "comparison"
    subject: str         # named_number: pose_similarity|work_similarity|order_adherence; cause: "far>thin"; conclusion: grade_normal|visited_all; comparison: speed_slower|...
    value: float | None  # named_number의 숫자


@dataclass(frozen=True)
class RecordPart:
    text: str
    code: str | None         # None이면 지식 조각(knowledge=True)이거나 못 채운 조각
    zones: tuple[str, ...]   # 질문이 말한 Zone 번호들, 없으면 1장 순서로 정한 한 Zone, 그것도 없으면 ()
    zone_source: str         # "asked" | "pronoun" | "here" | "subject" | "playback" | "none"
    operator: str | None
    metric: str | None
    cardinality: str         # "one" | "any"
    premises: tuple[Premise, ...]
    knowledge: bool


@dataclass(frozen=True)
class RecordQuery:
    kind: str                # "fact" | "judgement" | "cause" | "mixed" | "none"
    parts: tuple[RecordPart, ...]

    @property
    def knowledge_question(self) -> str:
        """지식 표지가 있는 조각 전부(코드가 있어도 포함). 문서 쪽 질문이 된다."""
        return " ".join(part.text for part in self.parts if part.knowledge)


def part_domain(part: RecordPart) -> str | None:
    return part.metric if part.code == "work_metric" and part.metric else CODE_DOMAIN.get(part.code or "")


def has_record_marker(prompt: str, *, previous_subject: Mapping[str, Any] | None = None) -> bool:
    return bool(_PERSON.search(prompt) or _RECORD_REF.search(prompt) or _PAST.search(prompt) or _EVALUATED.search(prompt)
                or _SPATIAL.search(prompt) or (previous_subject and _ANAPHORA.search(prompt)))


def split_parts(prompt: str) -> list[str]:
    # "고 " 분할은 두 쪽이 모두 기록 코드를 가질 때만이라 classify_record_question 안에서 한다.
    pieces: list[str] = []
    for sentence in re.split(r"(?<=[?？!.])\s+", prompt.strip()):
        pieces += [piece.strip() for piece in re.split(r"(?<=고,)\s*", sentence) if piece.strip()]
    return pieces


def _family(domain: str | None) -> str:
    return domain if domain in {"pose", "order"} else "zone"


def _cause_pair(piece: str) -> str | None:
    causes = sorted((match.start(), kind) for kind, pattern in _CAUSES for match in re.finditer(pattern, piece))
    effects = sorted((match.start(), kind) for kind, pattern in _EFFECTS for match in re.finditer(pattern, piece))
    return next((f"{cause}>{effect}" for start, cause in causes for end, effect in effects if end > start), None)


def _codes(piece: str, task_type: str, has_subject: bool, marker: bool, asked: bool) -> tuple[list[tuple[int, str, str | None, str | None]], str | None]:
    """(위치, 코드, operator, metric) 목록(말한 순서)과 원인 짝."""
    hits: list[tuple[int, str, str | None, str | None]] = []
    pair = _cause_pair(piece)

    def add(position: int, code: str, operator: str | None = None, metric: str | None = None) -> None:
        if marker or code not in _NEEDS_MARKER:
            hits.append((position, code, operator, metric))

    effect = re.search(_EFFECT_ANY, piece)
    if task_type == "knowledge_search" and (match := _ONE_PRIORITY.search(piece)):
        add(match.start(), "one_priority")
    external = _EXTERNAL.search(piece) and _EXTERNAL_ASK.search(piece)
    if external:
        add(external.start(), "external_standard")
    completion = None if external or _NOT_COMPLETION.search(piece) else _COMPLETION.search(piece)
    if completion:
        add(completion.start(), "completion")
    cause = bool(effect and (_WHY.search(piece) or pair)) or bool(has_subject and re.search(r"원인|이유", piece))
    if cause and marker:
        add(0, "cause")
    elif (extreme := _EXTREME.search(piece)) and (_THIN.search(piece) or _THICK.search(piece)):
        add(extreme.start(), "coating_extreme", "argmin" if _THIN.search(piece) else "argmax")
    elif (thick := _THICK.search(piece)) and _WHERE.search(piece) and not _WHY.search(piece):
        add(thick.start(), "coating_over")
    trajectory = task_type == "trajectory_recomm"  # 순서 탭의 질문은 표지 없이도 기록 질문이다
    if not completion and (marker or trajectory or asked):
        status = progress_question(piece)
        if status == "next":
            add(0, "progress_next")
        elif status == "status" or (asked and _STATUS_ASKED.search(piece)):
            add(0, "progress_status")
    if _ORDER_WORD.search(piece) and (marker or trajectory or re.search(r"숙련자|전문가", piece)):
        operator = next((name for name, pattern in _ORDER_OPERATORS if re.search(pattern, piece)), None)
        if operator and (operator not in {"start", "end"} or _ORDER_SUBJECT.search(piece)):
            add(0, "order_shape", operator)
    if not (cause and marker):  # "거리가 멀어서 얇게 나온 거지?"는 원인 질문이지 거리 비교가 아니다(T40)
        if (match := _FILTER.search(piece)) and _WHERE.search(piece) and not asked:
            add(match.start(), "distance_filter", _FILTER_OPERATOR[match.group(1)])
        elif match := _DISTANCE.search(piece):
            add(match.start(), "distance_compare")
        if match := _SPEED.search(piece):
            add(match.start(), "speed")
    metric = next((name for name, pattern in _METRICS if re.search(pattern, piece)), None)
    if metric and _METRIC_ASK.search(piece):
        add(0, "work_metric", None, metric)
    elif _POSE_EVALUATION.search(piece) and (re.search(r"자세|유사도", piece) or (task_type == "pose_recomm" and _PERSON.search(piece))):
        add(0, "pose_judgement")
    return sorted(hits, key=lambda hit: hit[0]), (pair if cause and marker else None)


def _premises(piece: str, codes: set[str], pair: str | None) -> tuple[Premise, ...]:
    out: list[Premise] = []
    if match := _NAMED_NUMBER.search(piece):
        out.append(Premise("named_number", _NAMED_SUBJECT[re.sub(r"\s+", " ", match.group(1))], float(match.group(2))))
    if pair:
        out.append(Premise("cause", pair, None))
    if "completion" in codes:
        out += [Premise("conclusion", subject, None) for subject, pattern in (("visited_all", _VISITED_ALL), ("grade_normal", _GRADE_NORMAL)) if pattern.search(piece)]
    if codes & {"speed", "distance_compare"} and (match := _COMPARISON.search(piece)):
        out.append(Premise("comparison", _COMPARISON_SUBJECT[match.group(1)[0]], None))
    return tuple(out)


def _zones(piece: str, code: str, subject: Mapping[str, Any] | None, playback: str | None) -> tuple[tuple[str, ...], str]:
    asked = tuple(dict.fromkeys(number for match in _ZONE_NUMBERS.finditer(piece) for number in re.findall(r"\d+", match.group(1))))
    if asked:
        return asked, "asked"
    if code not in _ZONE_CODES:
        return (), "none"
    previous = tuple(str(zone) for zone in (subject or {}).get("zone_ids") or ()) if _family((subject or {}).get("domain")) == _family(CODE_DOMAIN[code]) else ()
    if _SPATIAL.search(piece) or (subject and _ANAPHORA.search(piece)):
        if previous:
            return previous, "pronoun"
        if playback:
            return (playback,), "playback"
    if _HERE.search(piece) and playback:
        return (playback,), "here"
    if code in _FALLBACK_CODES:
        if previous:
            return previous, "subject"
        if playback:
            return (playback,), "playback"
    return (), "none"


def classify_record_question(prompt: str, *, task_type: str, previous_subject: Mapping[str, Any] | None,
                             playback_zone: str | None) -> RecordQuery:
    if preset_intent(task_type, prompt):
        return RecordQuery("none", ())
    cardinality = "one" if _ONLY_ONE.search(prompt) else "any"
    parts: list[RecordPart] = []

    def analyse(piece: str, marker: bool | None = None) -> tuple[list[tuple[int, str, str | None, str | None]], str | None]:
        # 손·방향·"뭐부터 고쳐"·구역별 자세 문제는 기존 경로의 몫이다.
        reserved = bool(BODY_PART.search(piece) or _DIRECTION_ONLY.search(piece)
                        or (task_type in {"pose_recomm", "trajectory_recomm"} and (WORST_ZONE.search(piece) or re.search(r"뭐부터|무엇부터", piece))))
        if reserved:
            return [], None
        marker = has_record_marker(piece, previous_subject=previous_subject) if marker is None else marker
        return _codes(piece, task_type, bool(previous_subject), marker, bool(_ZONE_NUMBERS.search(piece)))

    def divide(piece: str, marker: bool) -> list[str]:
        # "~고" 앞뒤가 각각 기록 코드를 가질 때만 가른다("잘하고 있는 거예요?"는 앞 절에 코드가 없어 그대로 둔다).
        for match in re.finditer(r"(?<=고)\s+", piece):
            left, right = piece[:match.start()], piece[match.end():]
            if analyse(left, marker)[0] and analyse(right, marker)[0]:
                return [left, *divide(right, marker)]
        return [piece]

    # 가른 조각은 원래 한 문장의 표지를 물려받는다.
    for piece, marker in ((half, whole_marker) for whole in split_parts(prompt)
                          for whole_marker in [has_record_marker(whole, previous_subject=previous_subject)] for half in divide(whole, whole_marker)):
        knowledge = bool(_KNOWLEDGE.search(piece))
        marker = marker or has_record_marker(piece, previous_subject=previous_subject)
        hits, pair = analyse(piece, marker)
        if not hits:
            if knowledge or marker:  # 표지 없는 맞장구("잘하고 있는 거예요?")는 조각으로 세지 않는다
                parts.append(RecordPart(piece, None, (), "none", None, None, cardinality, (), knowledge))
            continue
        premises = _premises(piece, {code for _, code, _, _ in hits}, pair)
        for _, code, operator, metric in hits:
            zones, source = _zones(piece, code, previous_subject, playback_zone)
            parts.append(RecordPart(piece, code, zones, source, operator, metric, cardinality, premises, knowledge))
    coded = [part for part in parts if part.code]
    if not coded:
        kind = "none"
    elif any(part.knowledge for part in parts):
        kind = "mixed"
    elif all(part.code == "cause" for part in coded):
        kind = "cause"
    elif any(part.code in _JUDGEMENT_CODES or part.premises for part in coded):
        kind = "judgement"
    else:
        kind = "fact"
    return RecordQuery(kind, tuple(parts))
