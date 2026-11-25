"""Scoped telemetry and report facts for questions that require exact values."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence

from trainer.demo_api import coating_assessment
from trainer.demo_api.copilot_session_context import summarize_session

_ZONE = re.compile(r"(?:\bzone\b|구역|존)\s*([1-9]\d*)|([1-9]\d*)\s*번\s*(?:구역|존|면)?", re.I)
_RECENT = re.compile(r"방금|조금\s*전|직전|최근|아까")
_DISTANCE = re.compile(r"이격\s*거리|평균\s*거리|건을?\s*(?:가장\s*)?멀리")
_PART = re.compile(r"(?<![\d.+-])(\d+(?:\.\d+)?)\s*(?:%|퍼센트)\s*(?:는|를|가|에)?\s*"
                   r"(?:보정\s*)?(?:도막\s*)?(?:두께\s*)?(\d+(?:\.\d+)?)\s*(?:마이크론|[µμu]m)", re.I)
_REPLAY = re.compile(r"리포트|보고서|리플레이|\breplay\b|완료\s*기록|종료", re.I)


def detail_intent(prompt: str, previous_prompt: str = "") -> str | None:
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
    if _DISTANCE.search(prompt) and re.search(r"처음|첫\s*분사", prompt) and re.search(r"돌아|재방문|다시\s*(?:와|가|방문)", prompt):
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
    if not bodies:
        return "현재 훈련에서 해당 값을 확인할 저장 분사 기록이 없습니다.", "sequence_missing"
    zone = _zone(prompt)
    if intent in ("visit_distance_change", "maximum_sequence_distance"):
        if intent == "visit_distance_change":
            segment_values: list[list[float]] = []
            previous = None
            for body in bodies:
                current = str(body.get("zone_id"))
                if current == zone:
                    if current != previous:
                        segment_values.append([])
                    value = _number(_mapping(body.get("sequence_result")).get("avg_distance_mm"))
                    if value is not None:
                        segment_values[-1].append(value)
                previous = current
            if len(segment_values) < 2 or not all(segment_values[:2]):
                return "지정한 구역의 첫 방문과 재방문에 해당하는 이격거리 측정 두 건을 확인할 수 없습니다.", "analysis_unavailable"
            first, second = [sum(values) / len(values) for values in segment_values[:2]]
            changed = f"{abs(second - first):g}mm {'증가' if second > first else '감소'}" if second != first else "변화 없음"
            return f"{zone}번 구역의 첫 방문은 {first:g}mm, 재방문은 {second:g}mm로 저장된 분사별 평균을 비교하면 {changed}입니다.", None
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
    return (f"최근 분사의 저장 평균은 건 이동 속도 {speed_text}, 이격거리 {distance_text}입니다.",
            None if speed is not None or distance is not None else "analysis_unavailable")


def report_answer(intent: str, prompt: str, analysis: Mapping, payload: Mapping,
                  bodies: Sequence[Mapping]) -> tuple[str, str | None]:
    sequence = _mapping(analysis.get("sequence_report"))
    coating = _mapping(analysis.get("coating_report"))
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
    if intent in ("report_vr_vs_pose", "report_pass_threshold"):
        score = _number(payload.get("vr_score"))
        passed = payload.get("is_pass")
        result = ("합격" if passed is True else "불합격" if passed is False else "합격 여부 미기록")
        recorded = f"시뮬레이터 수행 점수는 {score:g}점({result})입니다." if score is not None else "시뮬레이터 수행 점수가 기록되지 않았습니다."
        if intent == "report_pass_threshold":
            return (recorded + " 이 리포트에는 합격 기준 점수가 없어 컷오프를 확인할 수 없습니다."), None
        pose = _number(_mapping(analysis.get("pose_report")).get("average_similarity_score"))
        posture = (f"자세 유사도는 별도의 분석 지표로 {pose * 100:g}%입니다." if pose is not None
                   else "자세 유사도는 다른 지표이며, 이 기록에서는 확인할 수 없습니다.")
        return f"{recorded} {posture}", None
    if intent == "report_coating_index":
        score = _number(coating.get("coating_distribution_index"))
        if score is None:
            return "완료 리포트에서 도막 분포 지수는 미산출입니다. 기록이 비어 있는 것을 0점으로 기록하면 안 됩니다.", None
        return f"완료 리포트의 도막 분포 지수는 {score:g}입니다.", None
    all_rows = [row for row in analysis.get("coating_zones") or [] if isinstance(row, Mapping)]
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
