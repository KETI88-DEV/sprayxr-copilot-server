"""Scoped telemetry and report facts for questions that require exact values."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from statistics import median

from trainer.demo_api import coating_assessment
from trainer.demo_api.copilot_session_context import summarize_session
from trainer.demo_api.playback_direction import LABELS as DIRECTION_LABELS
from trainer.demo_api.sequence_analysis import observed_direction
from trainer.demo_api.session_coating import SOURCE_CROPS

_ZONE = re.compile(r"(?:\bzone\b|구역|존)\s*([1-9]\d*)|([1-9]\d*)\s*번\s*(?:구역|존|면)?", re.I)
_RECENT = re.compile(r"방금|조금\s*전|직전|최근|아까")
_DISTANCE = re.compile(r"이격\s*거리|평균\s*거리|건을?\s*(?:가장\s*)?멀리")
_PART = re.compile(r"(?<![\d.+-])(\d+(?:\.\d+)?)\s*(?:%|퍼센트)\s*(?:는|를|가|에)?\s*"
                   r"(?:보정\s*)?(?:도막\s*)?(?:두께\s*)?(\d+(?:\.\d+)?)\s*(?:마이크론|[µμu]m)", re.I)
_REPLAY = re.compile(r"리포트|보고서|리플레이|\breplay\b|완료\s*(?:기록|화면|표|결과)|종료", re.I)
_SEQUENCE_ID = re.compile(r"\bSEQ-\d{1,12}(?![\d-])", re.I)
_SPEED = re.compile(r"이동\s*속도|건\s*속도|분사\s*속도|속도")
_MOTION_JUDGMENT = re.compile(r"빠르|빨랐|느리|느렸|괜찮|적절|적정|너무\s*(?:가깝|멀)")
_VR_RESULT = re.compile(r"(?:시뮬레이터|\bvr\b).{0,45}(?:점수|합격)|(?:점수|합격).{0,25}(?:시뮬레이터|\bvr\b)", re.I)
_GRADE_MATCHES = {"above": r"과도막|상한\s*초과", "below": r"저도막|하한\s*미달", "normal": r"정상(?:\s*도막)?"}
_PASS_GOAL = re.compile(r"(?:합격|통과)(?:을|를)?\s*(?:하|할)\s*(?:려면|려고|기\s*위해)")
_BARE_PASS_GOAL = re.compile(
    r"\s*(?:합격|통과)(?:을|를)?\s*(?:하|할)\s*려면\s*어떻게\s*"
    r"(?:해야\s*(?:하지|하죠|하나요|해요?|돼요?|되나요)|해요?)\s*[?.!？。]*\s*"
)
_RUN_SCOPE = re.compile(r"(?:이번|전체|내)\s*(?:훈련|작업)|시뮬레이터|\bvr\b|리플레이|리포트|\breplay\b", re.I)
_UNRELATED_PASS = re.compile(r"도막|검사|접착|자격증|시험|인증|채용|면접|몇\s*점")


def _report_reading_intent(prompt: str, task_type: str | None) -> str | None:
    if re.search(r"(?:새|막\s*연)\s*훈련", prompt) and re.search(r"그\s*점수|앞\s*기록", prompt) and re.search(r"합격|결과", prompt):
        return "report_vr_status"
    if (_PASS_GOAL.search(prompt) and not _UNRELATED_PASS.search(prompt)
            and (_RUN_SCOPE.search(prompt) or task_type == "error_analysis" or _BARE_PASS_GOAL.fullmatch(prompt))):
        return "report_pass_improvement"
    if (_RUN_SCOPE.search(prompt) and re.search(r"궤적|경로", prompt)
            and re.search(r"개선|보완|고쳐|수정", prompt)
            and not (_ZONE.search(prompt) or _RECENT.search(prompt) or re.search(r"프레임|분사\s*방향|도포\s*방향", prompt))):
        return "report_trajectory_improvement"
    if not _REPLAY.search(prompt):
        return None
    if re.search(r"빠른\s*안내|상세\s*분석|response_mode", prompt, re.I) and re.search(r"계산|저장|갱신", prompt):
        return "response_mode_record"
    if re.search(r"도막", prompt) and re.search(r"높이맵|히트맵", prompt) and re.search(r"파일명|원본|어떤\s*자료|무엇으로", prompt):
        return "report_coating_source"
    if re.search(r"도막", prompt) and re.search(r"빈|비어|기록\s*없", prompt) and re.search(r"0\s*(?:µm|마이크로)|미도장", prompt):
        return "report_missing_coating"
    if re.search(r"구역", prompt) and re.search(r"면적", prompt) and any(re.search(pattern, prompt) for pattern in _GRADE_MATCHES.values()):
        if re.search(r"비율|분모|퍼센트|%", prompt):
            return "report_coating_ratios"
    if re.search(r"순서\s*유사", prompt) and re.search(r"방향\s*준수", prompt):
        return "report_order_and_direction"
    if re.search(r"재방문", prompt) and re.search(r"방문", prompt) and re.search(r"순서\s*이탈|이탈\s*수", prompt):
        return "report_zone_visits"
    if re.search(r"합격", prompt) and re.search(r"도막", prompt) and re.search(r"전부|모든|전\s*구역|모두", prompt):
        return "report_pass_vs_coating"
    if _VR_RESULT.search(prompt) and re.search(r"완료|끝났|분석\s*상태|생성", prompt):
        return "report_vr_status"
    if re.search(r"분사\s*(?:기록|로그|건수)|분사한\s*로그", prompt) and re.search(r"자세|채점", prompt):
        return "report_pose_coverage"
    if re.search(r"전문가", prompt) and re.search(r"평균|두께값", prompt) and re.search(r"역산|추정|표에\s*적힌", prompt):
        return "report_expert_mean"
    return None


def detail_intent(prompt: str, previous_prompt: str = "", *, task_type: str | None = None) -> str | None:
    if _SEQUENCE_ID.search(prompt) and re.search(r"이격\s*거리|이동\s*속도|분사\s*속도", prompt):
        return "sequence_measurement"
    if _DISTANCE.search(prompt) and re.search(r"중앙값|중앙\s*값", prompt) and re.search(r"저장|분사|시퀀스", prompt):
        return "report_distance_median"
    if reading := _report_reading_intent(prompt, task_type):
        return reading
    if re.search(r"블라스팅|블라스트", prompt) and re.search(r"권장|안내", prompt) and re.search(r"조도|실제\s*작업", prompt):
        return "blasting_start_evidence"
    if re.search(r"정상\s*범위|하한.*상한", prompt) and re.search(r"한가운데|중앙\s*값|중간\s*값", prompt):
        return "target_range_midpoint"
    if re.search(r"점\s*하나|한\s*점|착탄\s*(?:좌표|점)", prompt) and re.search(r"분사|착탄", prompt) and re.search(r"가로|세로|방향", prompt):
        if re.search(r"지금|기록|이번|방금", prompt):
            return "sequence_direction_evidence"
    if (re.search(r"가정|치면|실측\s*말고", prompt) and re.search(r"두께.*면적|도달\s*면적", prompt)
            and re.search(r"진행률|완료율", prompt) and len(_PART.findall(prompt)) > 1):
        return "project_distribution"
    if re.search(r"그라인딩|grinding", prompt, re.I) and re.search(r"전용|순서만|말고", prompt) and re.search(r"순서|경로", prompt):
        return "grinding_reference"
    if re.search(r"블라스팅|블라스트", prompt) and re.search(r"연마재|투입량", prompt) and re.search(r"소비량|사용량|kg", prompt, re.I):
        return "abrasive_usage"
    if re.search(r"학습\s*모드", prompt) and re.search(r"실전", prompt) and re.search(r"등록\s*기준|불러", prompt):
        return "practical_reference"
    if re.search(r"정상\s*범위|목표\s*두께", prompt) and re.search(r"출처|입력.*생략|설정.*생략", prompt):
        return "target_provenance"
    if re.search(r"누적\s*(?:높이맵|히트맵)", prompt) and re.search(r"요약|보낸", prompt) and re.search(r"평균", prompt):
        return "coating_sources"
    if re.search(r"(?:시퀀스|분사).{0,7}(?:식별자|ID|아이디)", prompt, re.I) and _RECENT.search(prompt):
        return "sequence_identity"
    if re.search(r"붙은|접미|뒤에", prompt) and re.search(r"어느\s*(?:구역|존)|[1-9]\d*\s*번\s*(?:구역|존)", prompt) and detail_intent(previous_prompt) == "sequence_identity":
        return "sequence_zone"
    if _RECENT.search(prompt) and re.search(r"배속", prompt) and re.search(r"속도", prompt):
        return "playback_speed"
    if _RECENT.search(prompt) and re.search(r"속도", prompt) and _DISTANCE.search(prompt):
        return "recent_motion"
    if re.search(r"방아쇠|트리거", prompt) and re.search(r"몇\s*초|연속\s*분사", prompt) and re.search(r"기록", prompt):
        return "spray_duration"
    if _RECENT.search(prompt) and re.search(r"도막", prompt) and re.search(r"이전\s*값|전후|증분", prompt) and re.search(r"차이|늘어|늘었", prompt):
        return "coating_change"
    if re.search(r"재방문|다시\s*돌아간", prompt) and re.search(r"순서|흐름", prompt) and re.search(r"최초|처음", prompt):
        return "report_visits"
    if _DISTANCE.search(prompt) and re.search(r"처음|첫\s*분사", prompt) and re.search(r"돌아|재방문|다시\s*(?:와|왔|가|갔|방문)", prompt):
        return "visit_distance_change"
    if _DISTANCE.search(prompt) and re.search(r"분사\s*별|시퀀스\s*별|분사마다", prompt) and re.search(r"가장|최대", prompt):
        return "maximum_sequence_distance"
    if _REPLAY.search(prompt) and re.search(r"누락|빠뜨린|빠진", prompt) and re.search(r"권장|준수", prompt) and re.search(r"순서", prompt):
        return "report_order_coverage"
    if re.search(r"팝업|클라이언트|실시간\s*알림", prompt) and re.search(r"이탈|경고", prompt) and re.search(r"횟수|건수", prompt):
        return "report_notice_count"
    if re.search(r"VR|시뮬레이터", prompt, re.I) and re.search(r"\d+\s*점|수행\s*점수", prompt) and re.search(r"자세", prompt):
        return "report_vr_vs_pose"
    if re.search(r"합격", prompt) and re.search(r"몇\s*점부터|기준\s*점수|커트\s*라인", prompt) and (
        re.search(r"점|VR|시뮬레이터", prompt, re.I) or detail_intent(previous_prompt) == "report_vr_vs_pose"
    ):
        return "report_pass_threshold"
    if re.search(r"분포\s*지수|coating_distribution_index", prompt, re.I):
        return "report_coating_index"
    if _REPLAY.search(prompt) and re.search(r"상한보다|상한과의\s*차이", prompt) and re.search(r"두께|도막", prompt):
        return "report_overage"
    if _REPLAY.search(prompt) and re.search(r"전문가", prompt) and re.search(r"도막", prompt) and re.search(r"판정|정상\s*여부|어떤\s*기준", prompt):
        return "report_coating_basis"
    if re.search(r"속도", prompt) and re.search(r"거리", prompt) and re.search(r"영향|입증|증명", prompt) and re.search(r"리포트|보고서", prompt):
        return "report_coating_cause"
    if re.search(r"리포트|보고서|완료\s*기록", prompt) and re.search(r"구역\s*평균|구역별\s*평균", prompt) and re.search(r"정상|하한|상한", prompt):
        return "report_coating_grades"
    return None


def _mapping(value) -> Mapping:
    return value if isinstance(value, Mapping) else {}


def _number(value) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


def _zone(prompt: str) -> str | None:
    found = _ZONE.search(prompt)
    return (found[1] or found[2]) if found else None


def sequence_id_in_prompt(prompt: str) -> str | None:
    found = _SEQUENCE_ID.search(prompt)
    return found[0].upper() if found else None


def _motion_delta(first: float, second: float, unit: str) -> str:
    difference = second - first
    if difference == 0:
        return "변화 없음"
    direction = "증가" if difference > 0 else "감소"
    return f"{abs(difference):g}{unit} {direction}"


def project_distribution(prompt: str, target: float) -> tuple[str, str | None]:
    portions = [(float(area), float(thickness)) for area, thickness in _PART.findall(prompt)]
    if any(area > 100 for area, _ in portions) or not math.isclose(sum(area for area, _ in portions), 100, abs_tol=0.01):
        return "두께별 면적 비율의 합이 100%인 경우에 구역 전체의 가정 진행률을 계산할 수 있습니다.", "clarification_required"
    coverage = sum(area for area, thickness in portions if thickness >= target)
    completion = sum(area * coating_assessment.uniform_zone_completion_percent(thickness, target) / 100
                     for area, thickness in portions)
    return (f"가정 계산입니다. 목표 {target:g}µm 이상인 면적은 {coverage:g}%, 도막 진행률은 {completion:g}%입니다. "
            "각 픽셀의 진행 기여는 100%까지만 반영합니다."), None


def sequence_answer(intent: str, prompt: str, bodies: Sequence[Mapping]) -> tuple[str, str | None]:
    if intent == "sequence_measurement" and not bodies:
        return f"이 훈련에는 {sequence_id_in_prompt(prompt)} 분사 기록이 없습니다.", "sequence_missing"
    if not bodies:
        return "현재 훈련에서 해당 값을 확인할 저장 분사 기록이 없습니다.", "sequence_missing"
    zone = _zone(prompt)
    if intent in ("visit_distance_change", "maximum_sequence_distance"):
        if intent == "visit_distance_change":
            visits: list[list[Mapping]] = []
            previous = None
            for body in bodies:
                current = str(body.get("zone_id"))
                if current == zone:
                    if current != previous:
                        visits.append([])
                    visits[-1].append(_mapping(body.get("sequence_result")))
                previous = current
            if len(visits) < 2:
                return "지정한 구역의 첫 방문과 재방문에 해당하는 측정 두 건을 확인할 수 없습니다.", "analysis_unavailable"
            compare_speed = bool(_SPEED.search(prompt))
            fields = [("이격거리", "avg_distance_mm", "mm")]
            if compare_speed:
                fields.append(("이동 속도", "avg_speed_mms", "mm/s"))
            measured = []
            for label, field, unit in fields:
                pair = [[value for item in visit if (value := _number(item.get(field))) is not None] for visit in visits[:2]]
                if not all(pair):
                    continue
                first, second = [sum(values) / len(values) for values in pair]
                prefix = label + "는 " if compare_speed else ""
                measured.append(f"{prefix}첫 방문은 {first:g}{unit}, 재방문은 {second:g}{unit}로 {_motion_delta(first, second, unit)}입니다.")
            if not measured:
                return "지정한 구역의 두 방문에서 비교할 평균 측정값이 없습니다.", "analysis_unavailable"
            if len(measured) != len(fields):
                measured.append("다른 요청 항목은 두 방문에서 비교할 측정값이 없습니다.")
            if re.search(r"인과|원인|단정|입증", prompt):
                measured.append("이 변화만으로 최종 도막의 원인을 입증할 수는 없습니다.")
            return f"{zone}번 구역의 " + " ".join(measured), None
        distances = [(_number(_mapping(body.get("sequence_result")).get("avg_distance_mm")), body) for body in bodies]
        values = [(distance, body) for distance, body in distances if distance is not None]
        if not values:
            return "저장된 분사별 평균 이격거리 값이 없습니다.", "analysis_unavailable"
        maximum = max(distance for distance, _ in values)
        zones = dict.fromkeys(str(body.get("zone_id")) for distance, body in values if distance == maximum)
        located = ", ".join(f"{candidate}번" for candidate in zones if candidate not in {"", "None", "-1", "0"})
        place = f"{located} 구역입니다." if located else "구역 번호는 확인되지 않습니다."
        return f"저장된 분사별 평균 이격거리 중 가장 큰 값은 {maximum:g}mm이며 {place} 순간 최댓값은 이 평균으로 확인할 수 없습니다.", None

    latest = bodies[-1]
    result = _mapping(latest.get("sequence_result"))
    if intent == "sequence_measurement":
        fields = []
        if _DISTANCE.search(prompt):
            distance = _number(result.get("avg_distance_mm"))
            fields.append(f"이격거리 {distance:g}mm" if distance is not None else "이격거리 미기록")
        if _SPEED.search(prompt):
            speed = _number(result.get("avg_speed_mms"))
            fields.append(f"이동 속도 {speed:g}mm/s" if speed is not None else "이동 속도 미기록")
        location = latest.get("zone_id")
        scope = f"{location}번 구역의 " if location and str(location) not in {"-1", "0"} else "구역 미판정 분사의 "
        return f"{latest.get('sequence_id')}은 {scope}저장 기록입니다. 평균 측정값은 {', '.join(fields)}입니다.", None
    if intent == "sequence_direction_evidence":
        direction = observed_direction(dict(latest))
        if direction is None:
            coordinates = (_mapping(_mapping(frame).get("gun_metrics")).get("raytexuv") for frame in latest.get("sequence_info") or [])
            points = sum(isinstance(point, (tuple, list)) and len(point) >= 2 and all(_number(x) is not None for x in point[:2])
                         for point in coordinates)
            return f"최근 분사에는 유효한 착탄 좌표가 {points}점뿐입니다. 두 점 이상이 있어야 실제 이동 방향을 판정할 수 있습니다.", "analysis_unavailable"
        label = DIRECTION_LABELS.get(direction, "이동 없음" if direction == "stationary" else direction)
        return f"저장된 착탄 좌표 흐름을 기준으로 최근 분사의 실제 도포 방향은 {label}입니다.", None
    if intent == "sequence_identity":
        sequence_id = latest.get("sequence_id")
        if sequence_id:
            return f"가장 최근에 저장된 분사 시퀀스 ID는 {sequence_id}입니다.", None
        return "최근 분사 기록의 시퀀스 ID를 확인할 수 없습니다.", "sequence_unavailable"
    if intent == "sequence_zone":
        suffix = re.search(r"(?:붙은|끝자리|접미사)\s*(\d{2,})", prompt)
        if suffix:
            recorded = [body for body in bodies if str(body.get("sequence_id")).rsplit("-", 1)[-1] == suffix[1]]
            if len(recorded) != 1:
                return "접미사만으로 어떤 분사 기록인지 확인할 수 없습니다. 시퀀스 ID 전체를 알려 주세요.", "clarification_required"
            latest = recorded[0]
        source_zone = latest.get("zone_id")
        if not source_zone or str(source_zone) in {"-1", "0"}:
            return "시퀀스 ID의 끝자리로 구역을 알 수 없습니다. 이 분사 기록에는 유효한 구역 번호가 없습니다.", "zone_unresolved"
        return (f"시퀀스 ID의 끝자리는 구역 번호를 뜻하지 않습니다. {latest.get('sequence_id')}에 저장된 구역은 {source_zone}번입니다."), None
    if intent == "spray_duration":
        frames = latest.get("sequence_info") or []
        samples = (_number(_mapping(_mapping(frame).get("gun_metrics")).get("spraytime")) for frame in reversed(frames))
        source = next((value for value in samples if value is not None), None)
        if source is None:
            return "분사 시간 항목이 저장되지 않아 방아쇠를 연속으로 눌렀던 시간을 확인할 수 없습니다.", "analysis_unavailable"
        return (f"최근 저장 프레임의 분사 시간 항목은 {source:g}초입니다. "
                "이 프레임 값만으로 방아쇠를 끊김 없이 누른 전체 시간을 확정할 수 없습니다."), None
    speed = _number(result.get("avg_speed_mms"))
    distance = _number(result.get("avg_distance_mm"))
    in_cm = bool(re.search(r"센티미터|\bcm", prompt, re.I))
    if intent == "playback_speed":
        measurement = (f"저장된 평균 건 이동 속도는 {speed:g}mm/s입니다." if speed is not None
                       else "이 분사의 평균 건 이동 속도 입력값은 없습니다.")
        return f"리플레이 배속은 저장된 측정값을 바꾸지 않습니다. {measurement}", None
    scale = 10 if in_cm else 1
    length = "cm" if in_cm else "mm"
    speed_text = f"{speed / scale:g}{length}/s" if speed is not None else "기록 없음"
    distance_text = f"{distance / scale:g}{length}" if distance is not None else "기록 없음"
    answer = f"최근 분사의 저장 평균은 건 이동 속도 {speed_text}, 이격거리 {distance_text}입니다."
    if intent == "recent_motion" and _MOTION_JUDGMENT.search(prompt):
        answer += " 이 측정값만으로는 적정 여부를 판단할 수 없습니다. 적용할 장비·훈련 기준과 비교해야 합니다."
    return answer, None if speed is not None or distance is not None else "analysis_unavailable"


def _recorded_vr_result(payload: Mapping) -> str:
    score = _number(payload.get("vr_score"))
    passed = payload.get("is_pass")
    verdict = "합격" if passed is True else "불합격" if passed is False else "합격 여부 미기록"
    if score is not None:
        return f"시뮬레이터 수행 점수는 {score:g}점({verdict})입니다."
    if isinstance(passed, bool):
        return f"시뮬레이터 결과는 {verdict}이며, 수행 점수는 기록되지 않았습니다."
    return "시뮬레이터 수행 점수와 합격 여부를 확인할 수 없습니다."


def _flow_comparison(sequence: Mapping) -> str | None:
    observed = sequence.get("first_visit_flow")
    recommended = sequence.get("recommended_flow")
    if not observed or not recommended:
        return None
    return f"최초 방문 {'→'.join(map(str, observed))}, 대표 권장 {'→'.join(map(str, recommended))}입니다."


def _trajectory_improvement(sequence: Mapping) -> str | None:
    if sequence.get("analysis_status") != "available":
        return None
    comparison = _flow_comparison(sequence)
    order_rate = _number(sequence.get("adherence_rate"))
    notes = [comparison] if comparison else []
    if order_rate is not None:
        notes.append(f"순서 유사도 {order_rate * 100:g}%입니다.")
        if comparison and order_rate < 1:
            notes.append("대표 권장 흐름으로 구역 이동 순서를 연습하세요.")
    directions = _mapping(sequence.get("direction_summary"))
    direction_rate = _number(sequence.get("direction_adherence_rate"))
    if direction_rate is None:
        direction_rate = _number(directions.get("direction_adherence_rate"))
    if direction_rate is not None:
        notes.append(f"방향 준수율 {direction_rate * 100:g}%입니다.")
        rows = [_mapping(row) for row in directions.get("zone_summary") or []]
        worst = max(rows, key=lambda row: _number(row.get("direction_error_count")) or 0, default={})
        errors = _number(worst.get("direction_error_count"))
        label = DIRECTION_LABELS.get(str(worst.get("expected_direction")))
        if errors and errors > 0 and worst.get("zone_id") and label:
            notes.append(f"방향 불일치가 많았던 {worst['zone_id']}번 구역({errors:g}회)에서 {label} 도포를 연습하세요.")
        elif direction_rate < 1:
            notes.append("리플레이에서 분사 궤적을 기준 도포 방향과 대조하세요.")
    return " ".join(notes) or None


def _priority_coating_action(analysis: Mapping, coating: Mapping) -> str | None:
    band = coating.get("normal_band_um")
    if (coating.get("analysis_status") != "available" or not isinstance(band, (list, tuple)) or len(band) != 2):
        return None
    lower, upper = map(_number, band)
    if lower is None or upper is None or not 0 < lower <= upper:
        return None
    deviations = []
    for item in analysis.get("coating_zones") or []:
        row = _mapping(item)
        average = _number(row.get("thickness_um"))
        if not row.get("zone_id") or average is None:
            continue
        if row.get("grade") == "above" and average > upper:
            deviations.append((average - upper, row, average, "과도막을 줄이세요."))
        if row.get("grade") == "below" and average < lower:
            deviations.append((lower - average, row, average, "목표 미달 부분만 보완하세요."))
    if not deviations:
        return None
    _, row, average, action = max(deviations, key=lambda item: item[0])
    face = f"Map {row['map_num']} " if row.get("map_num") is not None else ""
    return (f"정상 범위 {lower:g}~{upper:g}µm에서 평균 두께 편차가 가장 큰 "
            f"{face}{row['zone_id']}번 구역({average:g}µm)의 {action}")


def report_answer(intent: str, prompt: str, analysis: Mapping, payload: Mapping,
                  bodies: Sequence[Mapping]) -> tuple[str, str | None]:
    sequence = _mapping(analysis.get("sequence_report"))
    coating = _mapping(analysis.get("coating_report"))
    pose_report = _mapping(analysis.get("pose_report"))
    if intent == "report_trajectory_improvement":
        guidance = _trajectory_improvement(sequence)
        if guidance:
            return guidance, None
        return "이 완료 리포트에는 개선할 이동 순서나 도포 방향을 확인할 궤적 평가가 없습니다.", "analysis_unavailable"
    if intent == "report_pass_improvement":
        notes = [_recorded_vr_result(payload), "이 리포트에서 합격 기준 점수는 확인할 수 없습니다."]
        if action := _priority_coating_action(analysis, coating):
            notes.append(action)
        order_rate = _number(sequence.get("adherence_rate"))
        if sequence.get("analysis_status") == "available" and order_rate is not None and order_rate < 1:
            if comparison := _flow_comparison(sequence):
                notes.append(comparison + " 다음 훈련에서는 대표 권장 흐름을 연습하세요.")
        if len(notes) == 2:
            notes.append("이 리포트에서 우선 개선할 도막이나 순서 항목은 확인되지 않습니다.")
        return " ".join(notes), None
    if intent == "report_coating_source":
        source = coating.get("source_artifact_id")
        if source == SOURCE_CROPS:
            return "완료 도막은 저장된 시퀀스의 누적 히트맵 크롭에서 계산했습니다.", None
        if source == "height_map_image":
            return "완료 도막은 최종 제출된 높이맵에서 계산했습니다. 원본 파일명은 저장되지 않았습니다.", None
        if isinstance(source, str) and re.fullmatch(r"[\w.-]+\.png", source, re.I):
            return f"완료 도막의 근거는 마지막에 제출한 높이맵 {source}입니다.", None
        return "완료 도막 리포트에서 높이맵 근거나 원본 파일명을 확인할 수 없습니다.", "analysis_unavailable"
    if intent == "report_missing_coating":
        return ("빈칸이나 미수신 상태는 측정값 0µm나 미도장 100%를 뜻하지 않습니다. "
                "측정 수치는 확인된 완료 도막 평가에서 옮겨 적으세요."), None
    if intent == "report_order_and_direction":
        directions = _mapping(sequence.get("direction_summary"))
        order_rate = _number(sequence.get("adherence_rate"))
        direction_rate = _number(sequence.get("direction_adherence_rate"))
        if direction_rate is None:
            direction_rate = _number(directions.get("direction_adherence_rate"))
        order_text = f"{order_rate * 100:g}%" if order_rate is not None else "미산출"
        direction_text = f"{direction_rate * 100:g}%" if direction_rate is not None else "미산출"
        evaluated, matched = (_number(directions.get(field)) for field in ("evaluated_sequence_count", "matched_sequence_count"))
        counts = f"(방향 평가 {evaluated:g}회 중 일치 {matched:g}회)" if evaluated is not None and matched is not None else ""
        return (f"완료 리포트의 순서 유사도는 {order_text}, 방향 준수율은 {direction_text}{counts}입니다. "
                "구역을 방문한 순서와 각 분사의 방향은 별도로 평가합니다."), None
    if intent == "report_zone_visits":
        selected = _zone(prompt)
        query_count = re.search(r"(\d+|두|세|네)\s*(?:번|회)\s*방문", prompt)
        count = None
        if query_count:
            value = query_count[1]
            count = int(value) if value.isdecimal() else {"두": 2, "세": 3, "네": 4}[value]
        summaries = []
        for row in sequence.get("zone_summary") or []:
            if not isinstance(row, Mapping) or (selected and str(row.get("zone_id")) != selected):
                continue
            visits, revisits, deviations = (_number(row.get(field)) for field in ("visit_count", "revisit_count", "deviation_count"))
            if None in (visits, revisits, deviations) or (count is not None and visits != count):
                continue
            if selected or count is not None or revisits > 0:
                summaries.append(f"{row['zone_id']}번 구역: 방문 {visits:g}회, 재방문 {revisits:g}회, 순서 이탈 {deviations:g}회.")
        if summaries:
            return " ".join(summaries), None
        return "완료 리포트에 요청한 방문 조건과 일치하는 구역별 집계가 없습니다.", "analysis_unavailable"
    if intent == "report_pose_coverage":
        evaluated = _number(pose_report.get("evaluated_sequence_count"))
        if evaluated is None:
            return f"저장된 분사 {len(bodies)}개입니다. 자세 채점 건수는 완료 리포트에서 확인할 수 없습니다.", "analysis_unavailable"
        return (f"저장된 분사 {len(bodies)}개와 자세 채점 {evaluated:g}개는 다른 수치입니다. "
                "채점이 비어 있는 것을 분사 로그 없음이나 자세 점수 0점으로 볼 수 없습니다."), None
    if intent == "report_distance_median":
        samples = sorted(distance for body in bodies
                         if (distance := _number(_mapping(body.get("sequence_result")).get("avg_distance_mm"))) is not None)
        if not samples:
            return "분사별 평균 이격거리가 저장되지 않아 중앙값을 계산할 수 없습니다.", "analysis_unavailable"
        submitted = _number(payload.get("avg_distance_mm"))
        recorded = f"완료 때 제출한 평균은 {submitted:g}mm입니다." if submitted is not None else "완료 제출 평균은 확인되지 않습니다."
        values = f"({', '.join(f'{item:g}' for item in samples)}mm)" if len(samples) <= 10 else ""
        return (f"저장된 분사별 평균 {len(samples)}개{values}의 중앙값은 {median(samples):g}mm입니다. {recorded} "
                "프레임별 거리 표본의 중앙값과는 구분합니다."), None
    if intent in ("report_visits", "report_order_coverage", "report_notice_count"):
        first = sequence.get("first_visit_flow") or []
        recommended = sequence.get("recommended_flow") or []
        deviation = sequence.get("deviation_count")
        if intent == "report_notice_count":
            count = f"완료 리포트의 순서 이탈 판정은 {deviation}회입니다. " if isinstance(deviation, int) else ""
            return (count + "리포트는 분석 결과를 집계합니다. 훈련 화면에 표시된 경고 팝업 횟수는 클라이언트 수신·표시 이력 없이 확인할 수 없습니다."), None
        if not first:
            return "완료 리포트에서 최초 방문 순서를 확인할 수 없습니다.", "analysis_unavailable"
        first_text = "→".join(map(str, first))
        if intent == "report_visits":
            stored = summarize_session(list(bodies))
            visits = json.loads(stored).get("zone_visit_order") or [] if stored else []
            observed = "→".join(map(str, visits))
            replayed = f"재방문을 포함한 저장 흐름은 {observed}입니다." if observed else "재방문을 포함한 분사 기록은 확인할 수 없습니다."
            return f"최초 방문 순서는 {first_text}입니다. {replayed}", None
        skipped = sequence.get("skipped_zone_count")
        count = f"빠뜨린 구역은 {skipped}개입니다. " if isinstance(skipped, int) else ""
        representative = "→".join(map(str, recommended))
        adherence = _number(sequence.get("adherence_rate"))
        evaluation = (f"전문가 복수 기준의 순서 유사도는 {adherence * 100:g}%"
                      if adherence is not None else "순서 유사도는 미산출")
        if isinstance(deviation, int):
            evaluation += f", 순서 이탈은 {deviation}회"
        return (count + f"최초 방문 {first_text}, 대표 권장 {representative or '확인 불가'}입니다. "
                f"누락 수와 별개로 {evaluation}입니다."), None
    if intent in ("report_vr_vs_pose", "report_pass_threshold", "report_vr_status"):
        recorded = _recorded_vr_result(payload)
        if intent == "report_vr_status":
            return f"이 훈련의 리포트 분석이 완료되었습니다. {recorded}", None
        if intent == "report_pass_threshold":
            return (recorded + " 이 리포트에는 합격 기준 점수가 없어 컷오프를 확인할 수 없습니다."), None
        pose = _number(pose_report.get("average_similarity_score"))
        posture = (f"자세 유사도는 별도의 분석 지표로 {pose * 100:g}%입니다." if pose is not None
                   else "자세 유사도는 다른 지표이며, 이 기록에서는 확인할 수 없습니다.")
        return f"{recorded} {posture}", None
    if intent == "report_coating_index":
        score = _number(coating.get("coating_distribution_index"))
        if score is None:
            return "완료 리포트에서 도막 분포 지수는 미산출입니다. 기록이 비어 있는 것을 0점으로 기록하면 안 됩니다.", None
        return f"완료 리포트의 도막 분포 지수는 {score:g}입니다.", None
    all_rows = [row for row in analysis.get("coating_zones") or [] if isinstance(row, Mapping)]
    if intent == "report_coating_ratios":
        distribution = _mapping(coating.get("distribution"))
        fields = {"above": "above", "below": "below", "normal": "in_band"}
        total = [_number(distribution.get(field)) for field in fields.values()]
        if not all_rows or any(value is None or value < 0 for value in total) or sum(total) <= 0:
            return "구역 수 비율과 면적 비율을 각각 계산할 수 있는 완료 도막 분포가 없습니다.", "analysis_unavailable"
        pixels = sum(total)
        statements = []
        for grade, pattern in _GRADE_MATCHES.items():
            if not re.search(pattern, prompt):
                continue
            zones = sum(row.get("grade") == grade for row in all_rows)
            area = _number(distribution[fields[grade]])
            name = coating_assessment.GRADE_WORDS[grade]
            statements.append(f"{name} 구역 비율 {zones}/{len(all_rows)}({zones / len(all_rows) * 100:.1f}%), "
                              f"면적 비율 {area:g}/{pixels:g}픽셀({area / pixels * 100:.1f}%).")
        return " ".join(statements) + " 구역 수와 판정 픽셀 면적의 분모가 다릅니다.", None
    if intent == "report_pass_vs_coating":
        passed = payload.get("is_pass")
        verdict = "합격" if passed is True else "불합격" if passed is False else "판정 미확인"
        measured = [
            f"{row['zone_id']}번 {coating_assessment.GRADE_WORDS[row['grade']]}"
            for row in all_rows if row.get("grade") in coating_assessment.GRADE_WORDS
        ]
        grade_text = f"현재 완료 도막 등급은 {', '.join(measured)}입니다." if measured else "완료 도막 등급은 확인할 수 없습니다."
        return f"시뮬레이터 수행 결과({verdict})와 구역별 도막 품질은 별개입니다. {grade_text}", None
    if intent == "report_expert_mean":
        return ("리포트의 전문가 대비 결과는 비교 등급과 면적 지표입니다. "
                "이 값과 사용자 도막 평균만으로 전문가 평균 두께를 역산할 수 없습니다."), "analysis_unavailable"
    selected = _zone(prompt)
    rows = [row for row in all_rows if str(row.get("zone_id")) == selected]
    band = coating.get("normal_band_um") or []
    lower, upper = (_number(band[0]), _number(band[1])) if isinstance(band, (list, tuple)) and len(band) == 2 else (None, None)
    range_text = f"{lower:g}~{upper:g}µm" if lower is not None and upper is not None else "리포트에 등록된 정상 범위"
    if intent == "report_coating_cause":
        cause = _mapping(coating.get("root_cause"))
        values = [f"{label} 설명력 {value:g}" for label, key in (("작업 조건", "behavior_r2"), ("공정 조건", "process_r2"))
                  if (value := _number(cause.get(key))) is not None]
        if values:
            return ("완료 리포트의 원인 분석 추정값은 " + ", ".join(values)
                    + "입니다. 이 범주별 추정치는 이동 속도와 이격거리 각각의 인과 기여를 입증하지 않습니다."), None
        return ("도막 등급 판정과 원인 입증은 별개입니다. 이번 리포트에는 이동 속도와 이격거리 중 어느 쪽의 영향이 더 컸는지 "
                "수치로 확인할 근거가 없습니다."), "analysis_unavailable"
    if intent == "report_overage":
        if len(rows) != 1 or upper is None or (thickness := _number(rows[0].get("thickness_um"))) is None:
            return "지정한 구역의 완료 도막 평균과 정상 상한을 함께 확인할 수 없습니다. 구역 번호와 면 번호를 확인해 주세요.", "analysis_unavailable"
        if thickness <= upper:
            return f"{selected}번 구역의 완료 평균 {thickness:g}µm는 세션 상한 {upper:g}µm를 넘지 않았습니다.", None
        return f"{selected}번 구역의 완료 평균 {thickness:g}µm는 세션 상한 {upper:g}µm보다 {thickness - upper:g}µm 높습니다.", None
    if intent == "report_coating_basis":
        avg = _number(rows[0].get("thickness_um")) if len(rows) == 1 else None
        grade = coating_assessment.GRADE_WORDS.get(rows[0].get("grade")) if len(rows) == 1 else None
        fact = f" {selected}번 구역의 완료 평균은 {avg:g}µm로 {grade}입니다." if avg is not None and grade else ""
        return (f"이 리포트의 최종 도막 등급은 세션 목표 범위 {range_text}로 판정합니다.{fact} "
                "전문가 도막과의 비교는 별도의 참고 결과입니다."), None
    group = re.search(r"((?:[1-9]\d*\s*(?:·|,|와|과)\s*)+[1-9]\d*)\s*번\s*구역", prompt)
    requested = set(re.findall(r"\d+", group[1])) if group else None
    lines = []
    for row in sorted(all_rows, key=lambda item: (str(item.get("map_num")), str(item.get("zone_id")).zfill(2))):
        if requested is not None and str(row.get("zone_id")) not in requested:
            continue
        avg = _number(row.get("thickness_um"))
        grade = coating_assessment.GRADE_WORDS.get(row.get("grade"))
        if avg is not None and grade:
            where = f"{row['zone_id']}번 구역"
            if sum(str(x.get("zone_id")) == str(row.get("zone_id")) for x in all_rows) > 1:
                where += f"(면 {row['map_num']})"
            lines.append(f"{where} {avg:g}µm, {grade}")
    if not lines:
        return "완료 리포트에 구역별 도막 평균과 등급이 없습니다.", "analysis_unavailable"
    return "\n".join([f"완료 높이맵의 세션 정상 범위는 {range_text}입니다.", *lines]), None
