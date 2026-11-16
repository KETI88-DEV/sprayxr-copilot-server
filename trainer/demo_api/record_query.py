"""뷰어 기록 질문을 조각·코드·Zone·전제로 가르는 규칙 분류(v3.5.0). 모델을 부르지 않는다.

질문을 조각으로 나누고 조각마다 "기록의 어느 사실을 묻는가"(코드)를 정한다. 답은 만들지 않는다(`record_answer`).
각 정규식 줄 끝 주석은 그 줄이 걸리는 시험 문항(판정_한국어.csv·질문시험_v340.csv 번호)이다.
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from trainer.demo_api.coating_query import (_CAUSES, _RECORD_SENTENCE, _DISTANCE_WORDS, _EFFECTS, _INVERTED, _SIMILAR, distance_sentence, measured_relation, measured_sentence,
                                            parse_viewer_coating_context, zone_measures)
from trainer.demo_api.coating_assessment import grade_phrase, mean_text
from trainer.demo_api.copilot_intent import preset_intent
from trainer.demo_api.dialogue import drop_sentences, viewer_claims
from trainer.demo_api.pose_query import josa
from trainer.demo_api.viewer_arm_pose import parse_viewer_arm_pose
from trainer.demo_api.viewer_progress import band_text, gate_lines, normal_band, progress_question, shortfall_side, shortfall_text, viewer_progress_answer

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
_PAST: Final = re.compile(r"칠했|칠한|뿌렸|뿌린|나왔|나온|나와|됐|끝난|끝났|완료|떠\s*있|정상이라고\s*나")  # T34 T36 T42
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
# "몇 코트까지 칠해야 끝난 거예요?"는 기준을 묻는다. "몇 개 Zone 끝났어요?"·"얼마나 끝났어요?"는 완료 개수를 묻는다.
_NOT_COMPLETION: Final = re.compile(r"(?:몇|얼마)(?!\s*(?:개|곳)|나\s*(?:끝|완료|다\s*칠))|어떻게|조건|코트")
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


def has_record_marker(prompt: str, *, previous_subject: Mapping[str, Any] | None = None, person: bool = True) -> bool:
    """`person=False`면 1인칭("제·나")만으로는 표지로 세지 않는다(라우터의 잡담 판정)."""
    return bool((person and _PERSON.search(prompt)) or _RECORD_REF.search(prompt) or _PAST.search(prompt) or _EVALUATED.search(prompt)
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


# ---------------------------------------------------------------- 결정적 답(R1~R12·C)
# 서명된 기록 행만 읽는다. 등급·게이트는 다시 계산하지 않는다(표기만 `coating_assessment.grade_phrase`).
ASK_ZONE: Final = "어느 Zone을 말씀하시는지 알 수 없습니다. Zone 번호를 넣어 다시 물어봐 주세요."
_NO_TARGET: Final = "이 기록에는 목표 대비 판정 자료가 없어 말씀드릴 수 없습니다."
_NO_ZONE: Final = {"", "-1", "0", "none", "null", "nan"}
_GATE_PERCENT: Final = 95
_UNITS: Final = {"deg": "°", "mm": " mm", "um": "µm"}


@dataclass(frozen=True)
class RecordFacts:
    value: str
    payload: Mapping[str, Any]
    claims: Mapping[str, Any]
    coating: Mapping[str, Any] | None
    arm: Mapping[str, Any] | None

    @classmethod
    def from_context(cls, value: str | None) -> "RecordFacts | None":
        found = viewer_claims(value)
        if found is None or value is None:
            return None
        return cls(value, found[0], found[1], parse_viewer_coating_context(value, "knowledge_search"), parse_viewer_arm_pose(value))

    @property
    def zone_ids(self) -> list[str]:
        ids = {str(zone["zone_id"]) for zone in (self.claims.get("zones") or {}).values() if isinstance(zone, Mapping) and zone.get("zone_id") is not None}
        return sorted(ids, key=lambda zone: (not zone.isdigit(), int(zone) if zone.isdigit() else 0, zone))

    @property
    def target_rows(self) -> list[Mapping[str, Any]]:
        target = (self.coating or {}).get("target")
        if not isinstance(target, Mapping) or target.get("analysis_status") != "available":
            return []
        return [row for row in target.get("zones") or [] if isinstance(row, Mapping)]

    @property
    def measures(self) -> dict[str, dict[str, Any]]:
        return zone_measures(self.claims)

    @property
    def playback_zone(self) -> str | None:
        playback = self.payload.get("playback_context")
        zone = str(playback.get("zone_id", "")).strip() if isinstance(playback, Mapping) else ""
        return None if zone.lower() in _NO_ZONE else zone


def _is_num(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _row(facts: RecordFacts, zone: str) -> Mapping[str, Any] | None:
    return next((row for row in facts.target_rows if str(row.get("zone_id")) == zone), None)


def _zone_text(facts: RecordFacts, zone: str) -> str:
    """"Zone 3은 저도막(칠한 곳 평균 95µm, 미도장 0%)" — 등급 표기와 반올림은 `coating_assessment.grade_phrase` 그대로."""
    row = _row(facts, zone)
    if row is None:
        return f"{josa(f'Zone {zone}', '은', '는')} 판정 행이 이 기록에 없습니다"
    return f"{josa(f'Zone {zone}', '은', '는')} {grade_phrase(row)}"


def _target_um(facts: RecordFacts) -> float | None:
    value = (facts.coating or {}).get("target", {}).get("target_thickness_um")
    return float(value) if _is_num(value) else None


def _range(facts: RecordFacts, zone: str | None = None) -> tuple[float, float]:
    """판정 행의 정상 범위. Zone을 주면 그 행, 아니면 모든 행이 같은 범위일 때 그 범위, 그래도 없으면 target 블록.

    한 답에 행 범위(세션 150~250µm)와 target 블록 범위(목표로만 구한 150~187.5µm)가 섞이지 않게 행을 먼저 본다.
    """
    rows = [row for row in ([_row(facts, zone)] if zone else facts.target_rows) if row is not None]
    bands = {normal_band(row) for row in rows}
    return bands.pop() if len(bands) == 1 else normal_band((facts.coating or {}).get("target") or {})


def _band(facts: RecordFacts, zone: str | None = None) -> str:
    low, high = _range(facts, zone)
    return f"{low:g}~{high:g}µm"


def _thickness_ranking(facts: RecordFacts, *, thinnest: bool) -> str:
    rows = sorted((row for row in facts.target_rows if _is_num(row.get("thickness_um"))), key=lambda row: row["thickness_um"], reverse=not thinnest)
    if not rows:
        return _NO_TARGET
    first, rest = rows[0], rows[1:3]
    target = _target_um(facts)
    text = (f"가장 {'얇은' if thinnest else '두꺼운'} 곳은 Zone {first['zone_id']}입니다. 칠한 곳 평균 {first['thickness_um']:.1f}µm"
            + (f"이고 정상 범위는 {_band(facts, str(first['zone_id']))}입니다." if target else "입니다."))
    return text + (" 그다음은 " + ", ".join(f"Zone {row['zone_id']}({row['thickness_um']:.1f}µm)" for row in rest) + "입니다." if rest else "")


def _overpaint(facts: RecordFacts) -> str:
    rows, target = facts.target_rows, _target_um(facts)
    if not rows or target is None:
        return _NO_TARGET
    heavy = sorted((row for row in rows if _is_num(row.get("overpaint_percent")) and row["overpaint_percent"] > 0), key=lambda row: -row["overpaint_percent"])[:3]
    high = _range(facts)[1]
    text = f"칠한 곳 평균이 상한({high:g}µm)을 넘는 Zone은 {sum(row.get('grade') == 'above' for row in rows)}곳입니다."
    return text + (" 상한을 넘는 면적이 있는 Zone은 " + ", ".join(f"Zone {row['zone_id']}({row['overpaint_percent']:g}%)" for row in heavy) + " 순입니다."
                   if heavy else " 상한을 넘는 면적이 있는 Zone은 0곳입니다.")


def _completion(facts: RecordFacts, zones: tuple[str, ...]) -> str:
    rows = [row for row in facts.target_rows if row.get("gate_status")]
    if not rows:
        return _NO_TARGET
    if not zones:
        order = [str(zone) for zone in facts.claims.get("reference_order") or []] or facts.zone_ids
        lines, _ = gate_lines(rows, order=order)
        done = sum(row["gate_status"] == "complete" for row in rows)
        return " ".join([f"정상 범위({_band(facts)}) 안 면적이 {_GATE_PERCENT}% 이상이고 평균이 정상 범위인 완료 Zone은 {len(rows)}곳 중 {done}곳입니다.", *lines])
    out = []
    for zone in zones:
        row = _row(facts, zone)
        if row is None or not row.get("gate_status"):
            out.append(f"{_zone_text(facts, zone)}{'.' if row is None else '입니다.'}")
            continue
        band = band_text(row)
        area = f"정상 범위({band}) 안 면적은 {row['coverage_percent']:g}%" if _is_num(row.get("coverage_percent")) else "칠한 면적 자료가 없고"
        # 완료 전이면 범위 밖 면적이 어느 쪽인지 말한다. 주로 상한 초과면 "보완"을 말하지 않는다(v3.6.2, 기본 160~200µm).
        short = f"{_GATE_PERCENT}% 기준에 못 미칩니다. {shortfall_text(row)}".rstrip()
        verdict = {"complete": f"{_GATE_PERCENT}% 기준을 넘겨 완료입니다.", "overpaint_risk": f"{_GATE_PERCENT}% 기준과 별개로 과도막이라 더 칠하면 안 됩니다."}.get(
            row["gate_status"], short)
        if row.get("grade") == "normal" and row["gate_status"] == "needs_more" and _is_num(row.get("coverage_percent")) and _is_num(row.get("thickness_um")):
            # 등급 "정상"과 "보완 필요"를 나란히 쓰면 모순으로 읽힌다(2026-11-14 사용자 지적). 평균 두께와 면적 기준을 대비로 적는다.
            out.append(f"Zone {zone}의 칠한 곳 평균 두께는 {mean_text(row['thickness_um'], row['target_thickness_um'])}µm로 정상 범위({band}) "
                       f"안이지만, 범위 안 면적이 {row['coverage_percent']:g}%라 {short}")
            continue
        out.append(f"{_zone_text(facts, zone)}입니다. {area}로 {verdict}")
    return " ".join(out)


def _distance_filter(facts: RecordFacts, operator: str | None) -> str:
    wanted = operator or "higher"
    names = {"higher": "멀리서", "lower": "가까이서", "similar": "비슷하게"}
    found: dict[str, list[str]] = {"higher": [], "lower": [], "similar": []}
    for zone, measure in sorted(facts.measures.items(), key=lambda item: (not item[0].isdigit(), int(item[0]) if item[0].isdigit() else 0)):
        if (relation := measured_relation(measure, "distance_mm")) is not None:
            found[relation].append(f"Zone {zone}({measure['distance_mm']:,.0f} mm, 전문가 {measure['reference_distance_mm']:,.0f} mm)")
    if not any(found.values()):
        return "이 기록에는 이격거리를 전문가와 견줄 자료가 없습니다."
    text = f"전문가 중앙값보다 {names[wanted]} 칠한 Zone은 " + (", ".join(found[wanted]) if found[wanted] else "0곳") + "입니다."
    if wanted != "similar":
        text += f" 전문가와 비슷한(±{_SIMILAR * 100:.0f}% 이내) Zone은 " + (", ".join(found["similar"]) if found["similar"] else "0곳") + "입니다."
    return text


def _speed(facts: RecordFacts, zone: str) -> str:
    measure = facts.measures.get(zone) or {}
    if not _is_num(measure.get("speed_mms")):
        return f"Zone {zone}의 이동 속도 자료가 이 기록에 없습니다."
    expert = f"이고 전문가 중앙값은 {measure['reference_speed_mms']:,.0f} mm/s입니다" if _is_num(measure.get("reference_speed_mms")) else "입니다"
    text = (f"Zone {zone}의 이동 속도는 {measure['speed_mms']:,.0f} mm/s{expert}. "
            "전문가 기록은 이전 장비 기준이라 맞춰 바꾸라고 하지 않습니다.")
    return text + (" 이 Zone은 저도막이고, 저도막에서 속도를 올리면 보통 더 얇아집니다." if measure.get("grade") == "below" else "")


def _metric_value(item: Mapping[str, Any], side: str) -> str:
    value = item.get(side)
    if _is_num(value):
        return f"{value:g}{_UNITS.get(str(item.get('unit') or ''), item.get('unit') or '')}"
    return str(item.get(f"{side}_label") or value)


def _work_metric(facts: RecordFacts, part: RecordPart) -> str:
    zones = {str(zone.get("zone_id")): zone for zone in (facts.claims.get("zones") or {}).values() if isinstance(zone, Mapping)}
    items = {zone: next((m for m in row.get("metrics") or [] if isinstance(m, Mapping) and m.get("key") == part.metric), None) for zone, row in zones.items()}
    items = {zone: item for zone, item in items.items() if item is not None and _is_num(item.get("similarity_percent"))}
    if not items:
        return "이 기록에는 그 항목의 작업 유사도 자료가 없습니다."
    label = next(iter(items.values())).get("label") or part.metric
    similarities = {zone: item["similarity_percent"] for zone, item in items.items()}
    low, high = min(similarities.values()), max(similarities.values())
    text = (f"{len(items)}개 Zone 모두 {label} 작업 유사도는 {low:g}%입니다." if low == high else
            f"{len(items)}개 Zone의 {label} 작업 유사도는 {low:g}%~{high:g}%이고 가장 낮은 곳은 Zone {min(similarities, key=similarities.get)}입니다.")  # type: ignore[arg-type]
    for zone in part.zones or ((facts.playback_zone,) if facts.playback_zone else ()):
        if zone in items:
            text += f" Zone {zone}의 {josa(str(label), '은', '는')} {_metric_value(items[zone], 'actual')}이고 전문가는 {_metric_value(items[zone], 'reference')}입니다."
    return text


def _pose_judgement(facts: RecordFacts) -> str:
    arm = facts.arm or {}
    if not _is_num(arm.get("average_similarity_percent")):
        return "이 기록에는 팔 자세를 채점한 결과가 없어 자세 유사도를 말씀드릴 수 없습니다."
    sequences = [seq for seq in arm.get("sequences") or [] if isinstance(seq, Mapping) and _is_num(seq.get("similarity_percent"))]
    text = f"팔 자세 유사도는 평균 {arm['average_similarity_percent']:.0f}%입니다."
    if sequences:
        text += " 분사별로는 " + ", ".join(f"{seq.get('sequence_id')}(Zone {seq.get('zone_id')}) {seq['similarity_percent']:.0f}%" for seq in sequences) + "입니다."
    limited = [str(seq.get("sequence_id")) for seq in sequences if seq.get("confidence") == "limited"]
    if limited:
        text += " 입력 품질이나 비교 범위에 제한이 있어 참고용 수치입니다(" + ", ".join(limited) + ")."
    return text + " 숙련자 중간 수준은 약 70%입니다. 이 수치만으로 잘했다 못했다를 가르지 않습니다."


def _slot_text(slot: object) -> str:
    return ", ".join(f"Zone {item['zone_id']} {item['count']}회" for item in slot or [] if isinstance(item, Mapping))  # type: ignore[attr-defined]


def _order_shape(facts: RecordFacts, operator: str | None) -> str:
    slots = [slot for slot in facts.claims.get("position_distribution") or [] if isinstance(slot, list) and slot]
    count = facts.claims.get("reference_session_count")
    if operator == "adherence":
        rate, best, position = (facts.claims.get(key) for key in ("order_adherence_rate", "best_reference_similarity", "position_similarity"))
        if not all(_is_num(value) for value in (rate, best, position)):
            return "이 기록에는 순서 준수율 자료가 없습니다."
        return (f"순서 준수율은 {rate * 100:.1f}%입니다. 가장 비슷한 숙련자 순서와는 {best * 100:.1f}% 같고, 자리별 일치는 "
                f"{position * 100:.1f}%입니다(숙련자 {count}회 기준).")
    if not slots or not _is_num(count):
        return "이 기록에는 숙련자 순서 분포 자료가 없습니다."
    top = [max(slot, key=lambda item: item["count"]) for slot in slots]
    if operator == "agreement":
        spots = ", ".join(f"{name} 자리는 {count}회 중 {item['count']}회가 Zone {item['zone_id']}" for name, item in zip(("첫", "둘째"), top[:2]))
        return f"숙련자끼리의 일치도 수치는 기록에 없습니다. 다만 {spots}였습니다."
    first = (f"숙련자 {count}회 모두 Zone {slots[0][0]['zone_id']}부터 시작했습니다." if len(slots[0]) == 1 and slots[0][0]["count"] == count
             else f"숙련자의 첫 자리는 {_slot_text(slots[0])}입니다.")
    mine = (facts.claims.get("actual_order") or [None])[0]
    if operator == "allowed":
        rate = facts.claims.get("order_adherence_rate")
        return (f"{first} 둘째 자리는 {_slot_text(slots[1])}입니다." if len(slots) > 1 else first) + (
            f" 이 기록은 Zone {mine}부터 시작했고 순서 준수율은 {rate * 100:.0f}%입니다." if mine and _is_num(rate) else "")
    last = f"숙련자의 마지막 자리는 {_slot_text(slots[-1])}입니다."
    return (f"{last} {first}" if operator == "end" else f"{first} {last}") + (f" 이 기록은 Zone {mine}부터 시작했습니다." if mine else "")


def _external(facts: RecordFacts) -> str:
    text = "PSPC·90/10·ISO·SSPC 같은 외부 규격의 DFT 기준은 이 기록으로는 판정할 수 없습니다."
    rows = facts.target_rows
    target = _target_um(facts)
    if not rows or target is None:
        return f"{text} 이 기록에는 목표 대비 판정 자료가 없어 우리 기준 결과도 말씀드릴 수 없습니다."
    below = sum(row.get("grade") == "below" for row in rows)
    done = sum(row.get("gate_status") == "complete" for row in rows)
    return f"{text} 이 기록의 기준(정상 범위 {_band(facts)})으로는 하한 미만 Zone이 {below}곳이고 완료한 Zone은 {done}곳입니다."


def _missing_zones(facts: RecordFacts, zones: list[str]) -> str:
    known = [int(zone) for zone in facts.zone_ids if zone.isdigit()]
    span = f"{min(known)}~{max(known)}" if known and known == list(range(min(known), max(known) + 1)) else ", ".join(facts.zone_ids)
    return f"{josa('Zone ' + ', '.join(zones), '은', '는')} 이 기록에 없는 Zone입니다. 이 기록의 Zone은 {span}입니다."


def cause_answer(facts: RecordFacts, zone: str | None) -> str:
    """원인 설계(2026-11-14-integrated-cause-analysis-design.md)의 cause_evidence가 붙으면 이 함수 본문만 바꾼다.

    지금은 그 Zone의 등급·칠한 곳 평균·목표와 실측 속도·이격거리(전문가 비교)를 그대로 전한다. 원인 낱말은 새로 만들지 않는다.
    """
    if zone is None:
        return ASK_ZONE
    measured = measured_sentence(zone, facts.measures.get(zone))
    row = _row(facts, zone)
    target = _target_um(facts)
    if row is None or not _is_num(row.get("thickness_um")) or target is None:
        return " ".join(item for item in (_NO_TARGET, measured) if item)
    return " ".join(item for item in (
        f"{_zone_text(facts, zone)}입니다. 정상 범위는 {_band(facts, zone)}입니다.",
        measured or "이 Zone의 이동 속도·이격거리 실측이 기록에 없어 원인은 말씀드릴 수 없습니다.") if item)


def _priorities(facts: RecordFacts) -> list[tuple[str, str]]:
    """(제목, 이유) 순위. 도막 게이트 → 팔 자세 → 작업 유사도, 앞 단계에 후보가 있으면 뒤 단계는 보지 않는다."""
    gate = sorted((row for row in facts.target_rows if row.get("gate_status") == "needs_more" and _is_num(row.get("coverage_percent"))),
                  key=lambda row: row["coverage_percent"])
    if gate:
        # 범위 밖 면적이 주로 상한 초과인 Zone에는 "보완하십시오"를 쓰지 않는다(v3.6.2).
        return [(f"Zone {row['zone_id']}부터 {'보완하십시오' if shortfall_side(row) == 'under' else '보십시오'}.",
                 f"정상 범위({band_text(row)}) 안 면적이 {row['coverage_percent']:g}%로 {_GATE_PERCENT}% 기준에 못 미칩니다. {shortfall_text(row)}".rstrip())
                for row in gate]
    arm = sorted(((zone, item) for zone, item in ((facts.arm or {}).get("zones") or {}).items() if isinstance(item, Mapping) and _is_num(item.get("similarity_percent"))),
                 key=lambda pair: pair[1]["similarity_percent"])
    if arm:
        return [(f"Zone {zone}의 팔 자세부터 보십시오.", f"팔 자세 유사도가 {item['similarity_percent']:.0f}%로 가장 낮습니다." + (f" {item['guidance']}" if item.get("guidance") else ""))
                for zone, item in arm]
    work = sorted((zone for zone in (facts.claims.get("zones") or {}).values() if isinstance(zone, Mapping) and _is_num(zone.get("work_similarity_percent"))),
                  key=lambda zone: zone["work_similarity_percent"])
    return [(f"Zone {zone['zone_id']}의 작업 자세부터 보십시오.", f"작업 유사도가 {zone['work_similarity_percent']:.0f}%로 가장 낮습니다.") for zone in work]


def one_priority(facts: RecordFacts, *, single: bool) -> str:
    ranked = _priorities(facts)
    if not ranked:
        return "이 기록에서 먼저 고칠 곳을 가릴 판정 자료가 없습니다."
    if single or len(ranked) == 1:
        return " ".join(ranked[0])
    return f"1순위 {' '.join(ranked[0])}\n2순위 {' '.join(ranked[1])}"


def answer_part(facts: RecordFacts, part: RecordPart, *, task_type: str) -> tuple[str, dict[str, Any] | None]:
    """조각 하나의 결정적 답과 sequence_context(진행 질문일 때만)."""
    code = part.code or ""
    if part.zone_source == "asked" and (missing := [zone for zone in part.zones if zone not in facts.zone_ids]) and facts.zone_ids:
        return _missing_zones(facts, missing), None
    if code in {"progress_status", "progress_next"}:
        found = viewer_progress_answer(facts.value, part.text, "status" if code == "progress_status" else "next")
        return found if found else ("이 기록에는 방문 순서 자료가 없어 말씀드릴 수 없습니다.", None)
    per_zone = {"completion": lambda zone: _completion(facts, (zone,)), "distance_compare": lambda zone: distance_sentence(zone, facts.measures.get(zone))
                or f"Zone {zone}의 이격거리를 전문가와 견줄 자료가 이 기록에 없습니다.",
                "speed": lambda zone: _speed(facts, zone), "cause": lambda zone: cause_answer(facts, zone)}
    if code in per_zone:
        if not part.zones:
            return (_completion(facts, ()) if code == "completion" else ASK_ZONE), None
        return " ".join(per_zone[code](zone) for zone in part.zones), None
    answers = {"coating_extreme": lambda: _thickness_ranking(facts, thinnest=part.operator != "argmax"), "coating_over": lambda: _overpaint(facts),
               "distance_filter": lambda: _distance_filter(facts, part.operator), "work_metric": lambda: _work_metric(facts, part),
               "pose_judgement": lambda: _pose_judgement(facts), "order_shape": lambda: _order_shape(facts, part.operator),
               "external_standard": lambda: _external(facts), "one_priority": lambda: one_priority(facts, single=part.cardinality == "one"),
               "zone_exists": lambda: _missing_zones(facts, list(part.zones))}
    return answers[code](), None


# ---------------------------------------------------------------- 답 조립(전제 바로잡기 · 조각별 답 · 단서 한 줄 · subject)
_NOT_IN_RECORD: Final = "그 부분은 이 기록에 없습니다."
_QUESTION_SHAPED: Final = re.compile(r"[?？]\s*$|(?:나요|까요|가요|죠|줘|세요)[.]?\s*$")  # 서술형 첫 문장에는 "없습니다"를 붙이지 않는다(P7)
_METRIC_NAMES: Final = {"pose_similarity": "자세 유사도", "work_similarity": "작업 유사도", "order_adherence": "순서 준수율"}
_COATING_CODES: Final = {"coating_extreme", "coating_over", "completion", "external_standard", "cause"}
_SPEED_RELATION: Final = {"speed_slower": "lower", "speed_faster": "higher", "distance_farther": "higher", "distance_closer": "lower"}
_STATED_WHY: Final = re.compile(r"(?:는데|은데|니까).*?(?:왜|어째서|이유)")  # 사실 + 이유 질문
_COMPARISON_PHRASE: Final = {"speed_slower": "속도가 느린", "speed_faster": "속도가 빠른", "distance_farther": "멀리서 칠한", "distance_closer": "가까이서 칠한"}


@dataclass(frozen=True)
class RecordAnswer:
    text: str                                  # 기록 쪽 결정적 문장(섞인 질문이면 기록 조각만)
    kind: str                                  # RecordQuery.kind ("none"은 나오지 않는다)
    subject: dict[str, Any] | None             # {"domain": str, "zone_ids": list[str], "operator": str}
    knowledge_question: str | None             # 섞인 질문의 지식 조각 문장(이어 붙임), 아니면 None
    sequence_context: dict[str, Any] | None    # progress_* 답의 context


def previous_subject(turns: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """바로 앞 행이 결정적 답(analysis)이고 subject가 있을 때만. 모델이 쓴 답의 주제는 이어받지 않는다."""
    last = turns[-1] if turns else None
    subject = last.get("subject") if last else None
    return dict(subject) if last and last.get("answer_source") == "analysis" and isinstance(subject, Mapping) and subject else None


def _metric_values(facts: RecordFacts) -> dict[str, float | None]:
    adherence = facts.claims.get("order_adherence_rate")
    return {"pose_similarity": (facts.arm or {}).get("average_similarity_percent"), "work_similarity": facts.claims.get("work_similarity_percent"),
            "order_adherence": adherence * 100 if _is_num(adherence) else None}


def _premise_lead(facts: RecordFacts, part: RecordPart, premise: Premise) -> str | None:
    """"예."/"아니요." + 서명된 사실. 어느 쪽인지 서명 값으로 가릴 수 없으면 None(동조도 반박도 하지 않는다)."""
    zone = part.zones[0] if part.zones else None
    if premise.kind == "named_number" and premise.value is not None:
        values = _metric_values(facts)
        same = [name for name, value in values.items() if _is_num(value) and round(value) == round(premise.value)]
        asked = values.get(premise.subject)
        if premise.subject in same:
            return None  # 맞는 수는 본문이 말한다. "예."는 "잘한 거 맞지?"에 대한 답으로 읽힌다(최종 검토 I3)
        if same:
            return f"아니요. {premise.value:g}%는 {_METRIC_NAMES[same[0]]}입니다."
        return f"아니요. 이 기록의 {josa(_METRIC_NAMES[premise.subject], '은', '는')} {asked:.0f}%입니다." if _is_num(asked) else None
    if premise.kind == "cause" and zone:
        cause, effect = premise.subject.split(">")
        measure = facts.measures.get(zone)
        relation = measured_relation(measure, "distance_mm")
        if cause in _DISTANCE_WORDS and relation is not None:
            # 실측과 다르거나 방향이 물리적으로 거꾸로면(가까워서 얇다·멀어서 두껍다) "아니요."; 맞아도 "예."로 인과를 인정하지 않고 사실만 말한다
            # "가까웠는데 왜 얇아?"는 사실을 말하고 이유를 묻는다. 전제가 아니므로 예·아니요로 열지 않는다
            wrong = not _STATED_WHY.search(part.text) and (_DISTANCE_WORDS[cause] != relation or (cause, effect) in _INVERTED)
            return ("아니요. " if wrong else "") + (distance_sentence(zone, measure) or "")
        return None  # 속도 원인은 R6(_speed)가 값과 한계만 말한다
    if premise.kind == "conclusion":
        rows = [_row(facts, z) for z in part.zones] if part.zones else list(facts.target_rows)
        if not rows or any(row is None or not row.get("gate_status") for row in rows):
            return None
        done = all(row["gate_status"] == "complete" for row in rows)  # type: ignore[index]
        if done:
            return "예."
        return "아니요. 다 칠한 것과 완료는 다릅니다." if premise.subject == "visited_all" else "아니요. 평균 두께가 기준 안이어도 완료는 칠한 면적으로 봅니다."  # 95% 설명은 본문이 한다
    if premise.kind == "comparison" and zone and premise.subject in _SPEED_RELATION:
        variable = "speed_mm_s" if premise.subject.startswith("speed") else "distance_mm"
        relation = measured_relation(facts.measures.get(zone), variable)
        if relation is None:
            return None
        # "예."로 열면 "올려야 해?"에 대한 처방으로 읽힌다(최종 검토 I3). 비교 사실만 말하고, 값은 본문이 말한다.
        said = f"숙련자 기록보다 {_COMPARISON_PHRASE[premise.subject]} 것은"
        if relation == _SPEED_RELATION[premise.subject]:
            return f"{said} 맞습니다."
        return f"{said} 아닙니다. " + {"lower": "전문가 중앙값보다 낮습니다.", "higher": "전문가 중앙값보다 높습니다.",
                                      "similar": "전문가 중앙값과 비슷합니다(±10% 이내)."}[relation]
    return None


def _subject_zones(facts: RecordFacts, part: RecordPart) -> list[str]:
    if part.code in {"coating_extreme"}:
        rows = sorted((r for r in facts.target_rows if _is_num(r.get("thickness_um"))), key=lambda r: r["thickness_um"], reverse=part.operator == "argmax")
        return [str(rows[0]["zone_id"])] if rows else []
    if part.code == "coating_over":
        heavy = sorted((r for r in facts.target_rows if _is_num(r.get("overpaint_percent")) and r["overpaint_percent"] > 0), key=lambda r: -r["overpaint_percent"])[:3]
        return [str(r["zone_id"]) for r in heavy]
    if part.code == "distance_filter":
        want = {"lower": "lower", "similar": "similar"}.get(part.operator or "", "higher")
        return [z for z, m in facts.measures.items() if measured_relation(m, "distance_mm") == want]
    if part.code == "one_priority":
        ranked = _priorities(facts)
        found = re.search(r"Zone (\d+)", ranked[0][0]) if ranked else None
        return [found.group(1)] if found else []
    return [z for z in part.zones if z in facts.zone_ids]


def answer_record_question(context: str | None, prompt: str, *, task_type: str,
                           recent_turns: Sequence[Mapping[str, Any]]) -> RecordAnswer | None:
    facts = RecordFacts.from_context(context)
    if facts is None:  # 서명 없음·위조 = 기록 분류 끔
        return None
    query = classify_record_question(prompt, task_type=task_type, previous_subject=previous_subject(recent_turns), playback_zone=facts.playback_zone)
    if query.kind == "none":
        return None
    # Dataset selectors can expose several texture faces with the same Zone
    # number. These signed facts explicitly withhold ambiguous measurements;
    # require the face selector instead of treating them as absent or taking
    # another face's last row from the legacy Zone-number fact reader.
    ambiguous = facts.claims.get("ambiguous_zone_ids")
    ambiguous = {str(zone) for zone in ambiguous if str(zone).isdigit()} if isinstance(ambiguous, list) else set()
    unresolved = sorted({zone for part in query.parts if part.code for zone in part.zones if zone in ambiguous}, key=int)
    if unresolved:
        shown = "·".join(unresolved)
        return RecordAnswer(f"Zone {shown} 번호가 여러 면에 있습니다. 훈련 문맥에서 면과 Zone을 선택해 주세요.",
                            "fact", None, None, {"analysis_status": "not_available", "reason_code": "dataset_map_zone_selection_required",
                                                  "zone_ids": unresolved})
    sentences: list[str] = []
    answered: list[RecordPart] = []
    sequence_context: dict[str, Any] | None = None
    seen: set[tuple[Any, ...]] = set()
    any_code = any(part.code for part in query.parts)
    for part in query.parts:
        if part.code is None:
            if any_code and not part.knowledge and _QUESTION_SHAPED.search(part.text):
                sentences.append(_NOT_IN_RECORD)
            continue
        key = (part.code, part.zones, part.operator, part.metric)
        if key in seen:  # 같은 코드·같은 Zone은 답 전체에서 한 번만(T31·T43·T49, 사이에 다른 조각이 있어도: v340 #40)
            continue
        seen.add(key)
        text, context_found = answer_part(facts, part, task_type=task_type)
        sequence_context = sequence_context or context_found
        in_record = all(zone in facts.zone_ids for zone in part.zones) or not facts.zone_ids
        leads = list(dict.fromkeys(lead for premise in part.premises if in_record and premise.kind != "conclusion" and (lead := _premise_lead(facts, part, premise))))
        conclusion = next((lead for premise in part.premises if in_record and premise.kind == "conclusion" and (lead := _premise_lead(facts, part, premise))), None)
        leads += [conclusion] if conclusion else []  # 결론 전제가 둘이어도 바로잡는 문장은 하나
        if part.code == "cause" and part.zones and in_record and any(p.kind == "cause" and p.subject.split(">")[0] in {"slow", "fast"} for p in part.premises):
            text = _speed(facts, part.zones[0])  # 속도 원인은 동조도 반박도 없이 값과 이전 장비 한계만
        if leads and part.code == "cause" and part.zones and _target_um(facts):
            text = f"{_zone_text(facts, part.zones[0])}입니다. 정상 범위는 {_band(facts, part.zones[0])}입니다."  # 실측 거리는 lead가 이미 말했다
        sentences.append(" ".join([*leads, text]))
        if text != ASK_ZONE and text != _NO_TARGET and in_record:
            answered.append(part)
    body = "\n".join(sentences)
    caveats = []
    if any(part.code in _COATING_CODES for part in answered) and "세션이 끝났을 때" not in body:
        caveats.append("도막 수치는 세션이 끝났을 때의 도막 기준입니다.")
    if any(part.code in {"completion", "cause", "external_standard"} for part in answered) and "95%" not in body:
        caveats.append(f"완료는 정상 범위 안 면적이 {_GATE_PERCENT}% 이상이고 평균이 정상 범위일 때입니다.")
    if any(part.code == "pose_judgement" for part in answered):
        if "70%" not in body:
            caveats.append("숙련자 중간 수준은 약 70%입니다.")
        if "참고용" not in body and any(isinstance(s, Mapping) and s.get("confidence") == "limited" for s in (facts.arm or {}).get("sequences") or []):
            caveats.append("입력 품질 제한이 있어 참고용입니다.")
    text = body + ("\n참고: " + " ".join(caveats) if caveats else "")
    first = answered[0] if answered else next((part for part in query.parts if part.code), None)
    subject = ({"domain": part_domain(first) or "overall", "zone_ids": _subject_zones(facts, first), "operator": first.operator or first.code or ""}
               if first and first.code else None)
    return RecordAnswer(text, query.kind, subject, query.knowledge_question or None, sequence_context)


# ---------------------------------------------------------------- 모델이 만든 답의 출구 검사 G3~G6
_NAMED_ZONE: Final = re.compile(r"(?:zone|존|구역)\s*(\d+)|(\d+)\s*번\s*(?:zone|존|구역)", re.I)
_SUPERLATIVE: Final = re.compile(r"(?:가장|제일)\D{0,15}?(얇|두껍|두꺼|낮|높|멀|가깝|가까)")
_EXISTENCE: Final = re.compile(r"(?:zone|존|구역|데|곳|과도막)[은는이가]?\s*(없|있)(?:습니다|어요|다)(?!는)", re.I)  # 유무의 주어가 바로 앞에 있어야 한다("~ㄹ 수 있습니다"는 제외)
_VERDICT_WORD: Final = re.compile(r"불?합격|완료|끝났|다\s*됐|다시\s*안\s*가도|정상\s*범위")
_CONDITIONAL: Final = re.compile(r"(?:완료|끝났)(?:되|된|됐)?(?:면|다면|뒤|후)|다\s*칠하면")  # 판정 낱말이 조건절 안에 있을 때만 절차다
_ZONE_COUNT: Final = re.compile(r"(\d+)\s*(?:개|곳)\s*(?:의\s*)?(?:zone|존|구역)|(?:zone|존|구역)[은는이가]?\s*(?:총\s*)?(\d+)\s*(?:개|곳)", re.I)
_BARE_PASS: Final = re.compile(r"^\W*불?합격(?:입니다|이에요|이다)\W*$")  # "합격입니다."만 있는 문장은 이 기록의 판정이다
_STANDARD_WORD: Final = re.compile(r"권장|기준|표준|적정")
_MM_VALUE: Final = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(mm|cm)")


def ground_generated(answer: str, facts: RecordFacts, *, zone: str | None, drop_only: bool = False, marked: bool = True,
                     zone_asked: bool = True) -> str:
    """모델 답의 문장 가운데 이 기록의 최저·유무·판정·기준값을 말하는 것을 서명 자료의 문장으로 바꾼다(`drop_only`면 뺀다).

    `marked`는 질문이 기록 표지를 가졌는가. 표지도 없고 문장에 Zone 번호·"이 기록" 같은 말도 없으면 일반 지식이라 건드리지 않는다.
    `zone_asked`는 `zone`이 질문의 Zone 번호에서 왔는가. "지금·여기"(재생 위치)에서 온 Zone만으로는 판정 검사(G5)를 켜지 않는다.
    """
    rows = facts.target_rows
    shown_distance = {"멀": "higher", "가깝": "lower", "가까": "lower"}

    def fix(sentence: str) -> str | None:
        named = [a or b for a, b in _NAMED_ZONE.findall(sentence)]
        if not (marked or named or _RECORD_SENTENCE.search(sentence)):
            return None
        thick_zones = [r for r in rows if r.get("grade") == "above"]  # 과도막 Zone = 칠한 곳 평균이 정상 범위 상한 초과(R2의 첫 수)
        if (top := _SUPERLATIVE.search(sentence)) and named:  # G3
            word = top[1]
            if word in shown_distance:
                want = shown_distance[word]
                truth = [z for z, m in facts.measures.items() if measured_relation(m, "distance_mm") == want]
                return _distance_filter(facts, want) if truth and not set(named) <= set(truth) else None
            if word in {"얇", "두껍", "두꺼"} or re.search(r"도막|두께", sentence):
                thin = word in {"얇", "낮"}
                ranked = sorted((r for r in rows if _is_num(r.get("thickness_um"))), key=lambda r: r["thickness_um"], reverse=not thin)
                return _thickness_ranking(facts, thinnest=thin) if ranked and named[0] != str(ranked[0]["zone_id"]) else None
        if (claim := _EXISTENCE.search(sentence)) and not _VERDICT_WORD.search(sentence):  # G4
            exists = claim[1] == "있"
            if re.search(r"숙련자|전문가|전문가보다", sentence) and (relation := re.search(r"멀|가까|비슷", sentence)):
                want = {"멀": "higher", "가": "lower", "비": "similar"}[relation[0]]
                actual = any(measured_relation(m, "distance_mm") == want for m in facts.measures.values())
                return _distance_filter(facts, want) if actual != exists and facts.measures else None
            if re.search(r"과도막|두껍|두꺼", sentence) and rows:
                return _overpaint(facts) if bool(thick_zones) != exists else None
        count = _ZONE_COUNT.search(sentence)
        if _VERDICT_WORD.search(sentence) and not _CONDITIONAL.search(sentence) and (
                named or (zone and zone_asked) or count or _RECORD_SENTENCE.search(sentence) or _BARE_PASS.search(sentence)):  # G5: 질문 표지만으로는 켜지 않는다
            if re.search(r"합격", sentence):
                return _external(facts)
            if count and not named:  # "3개 Zone이 완료되었습니다": 완료 개수가 서명 자료와 다르면 전체 완료 답으로
                gated = [r for r in rows if r.get("gate_status")]
                done = sum(r["gate_status"] == "complete" for r in gated)
                return _completion(facts, ()) if gated and int(count[1] or count[2]) != done else None
            target = tuple(dict.fromkeys(named or ([zone] if zone else [])))
            return _completion(facts, target)
        for value, unit in _MM_VALUE.findall(sentence):  # G6
            if not _STANDARD_WORD.search(sentence):
                break
            mm = float(value.replace(",", "")) * (10 if unit == "cm" else 1)
            decimal = "." in value
            for z in named or ([zone] if zone else list(facts.measures)):
                ref = (facts.measures.get(z) or {}).get("reference_distance_mm")
                if _is_num(ref) and (round(mm, 1) == round(ref, 1) if decimal else int(mm) == int(ref + 0.5)):
                    return (f"Zone {z}의 숙련자 측정 중앙값 이격거리는 {ref:,.1f} mm입니다. 숙련자가 실제로 칠한 값을 잰 것이며 정해진 기준은 아닙니다.")
        return None

    if drop_only:
        return drop_sentences(answer, lambda sentence: fix(sentence) is not None)[0]
    seen: set[str] = set()
    lines: list[str] = []
    for line in answer.split("\n"):
        parts: list[str] = []
        for part in re.split(r"(?<=[.!?])\s+", line):
            fixed = fix(part) if part.strip() else None
            if fixed is None:
                parts.append(part)
            elif fixed not in seen:
                seen.add(fixed)
                parts.append(fixed)
        lines.append(" ".join(parts))
    return "\n".join(lines)
