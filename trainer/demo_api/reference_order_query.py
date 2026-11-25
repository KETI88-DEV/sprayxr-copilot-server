"""Registered work plans, separate from measured trainee progress."""
from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Literal

from trainer.demo_api import coating_assessment
from trainer.demo_api.telemetry_contracts import AnalyzeSequenceRequest, learning_mode_from_scenario, process_from_tool_type


_PROCEDURE = r"표면\s*처리|전처리|세척|탈지|건조|혼합|희석|교반|프라이머|절차|단계|공정|도장\s*전에|칠하기\s*전"
_EXCLUDED_ORDER = re.compile(r"(?:순서|경로)(?:는|를|은|을)?\s*(?:말고|빼고|제외하고|대신)")
_REFERENCE_TARGET = re.compile(r"도장|피도면|구역|\bzone\b|존(?!재|중)|블라스팅|스프레이|학습\s*모드|시나리오|\bTR-L\d+\b", re.I)
_NEXT_ZONE = re.compile(r"다음\s*(?:으로\s*)?(?:(?:칠|도장)\s*(?:할|해야\s*할)\s*)?(?:구역|zone|존)", re.I)
_NEXT_ZONE_ASK = re.compile(
    r"^(?:은|는|이|가|을|를)?(?:$|어디|어딘|어느(?:곳|구역|존)|어떤(?:곳|구역|존)|뭐|뭔|무엇|몇번|번호|알려|안내|추천|찍어|가리켜"
    r"|어떻게(?:돼|되)|(?:(?:으)?로)?(?:바로|이제)?(?:이동|넘어가|가도|가야|가면|진행해도|옮겨|칠하면|칠해야))"
)
_AFTER_REFERENCE_ZONE = re.compile(
    r"(?:(?:zone|구역|존)\s*(\d+)|(\d+)\s*번(?:\s*(?:구역|zone|존))?)(?:을|를)?"
    r"\s*(?:끝내면|끝나면|끝내고(?:\s*나면)?|(?:칠|도장|작업)?\s*하고\s*나면|칠하면|칠하고(?:\s*나면)?)"
    r"[\s,]*(?:그\s*)?(?:다음|뒤|어디|어느|몇\s*번)",
    re.I,
)
_FIRST_REFERENCE_ZONE = re.compile(
    r"첫\s*(?:구역|zone|존)(?:은|이|의)?\s*(?:번호|몇|어디|뭐)"
    r"|(?:뭐|무엇|어디|몇\s*번|\d+\s*번|어느\s*(?:구역|zone|존))\s*부터\s*(?:칠|분사|도장)",
    re.I,
)
_MOVE_PERMISSION = re.compile(
    r"(?:이동해|옮겨|넘어가)(?:도\s*(?:될|돼|되|괜찮)|면\s*(?:돼|되|된|안\s*돼))|넘어감\s*(?:되|돼)"
)
_LIVE_ZONE = re.compile(r"지금|현재|방금|이제|여기|더\s*칠|구역|\bzone\b|존(?!재|중)", re.I)
_AREA_METRIC = re.compile(
    r"(?:목표\s*두께\s*(?:도달|이상|충족|넘(?:은|는))|기준\s*두께\s*이상)\s*(?:인\s*)?면적|도달\s*면적"
)
_NUMERIC_AREA_METRIC = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:마이크론|[µμ]m)?\s*(?:을\s*)?(?:이상(?:인)?|넘는|넘은)\s*면적", re.I
)
_MEASURED_PROGRESS = re.compile(r"진행률|완료율|얼마나\s*(?:더\s*해야|(?:더\s*)?칠)|(?:칠했|칠한).*(?:퍼센트|%)")
_PROGRESS_HOLD = re.compile(r"왜\s*(?:아직\s*)?(?:붙잡|못\s*넘어|안\s*넘어|이동.*안)")

ReferencePosition = tuple[Literal["first", "after"], str | None]


# "재도장 간격은?", "다시 칠할 때 주의할 점은?"처럼 방법·조건을 묻는 지식 질문의 표지(2026-11-14 독립 검토).
_KNOWLEDGE_ASK = r"간격|주의|조심|방법|요령|시간|얼마|어떻게|왜|원인|이유|기준|규정|예방|조건"


def is_knowledge_ask(prompt: str) -> bool:
    """진행 낱말(보완·재도장·다음)이 있어도 방법·절차·조건을 묻는 지식 질문인가."""
    return bool(re.search(_KNOWLEDGE_ASK, prompt) or re.search(_PROCEDURE, prompt))


def _asks_to_rewrite_text(prompt: str) -> bool:
    return bool(re.search(r"(?:문구|문장|철자|표기).*(?:다듬|바꿔|고쳐|수정|교정|변경)", prompt))


def reference_position_question(prompt: str) -> ReferencePosition | None:
    """Identify a requested position in the plan without deciding which plan applies."""
    if _asks_to_rewrite_text(prompt) or re.search(r"표면\s*처리|전처리|세척|탈지|건조|희석|교반|방법|요령|안전|원리", prompt):
        return None
    anchor = _AFTER_REFERENCE_ZONE.search(prompt)
    if anchor:
        return "after", anchor[1] or anchor[2]
    return ("first", None) if _FIRST_REFERENCE_ZONE.search(prompt) else None


def is_current_progress_question(prompt: str) -> bool:
    """Separate a measured progress lookup from standards, definitions and hypothetical values."""
    if _asks_to_rewrite_text(prompt) or re.search(r"가정|만약|내\s*값\s*말고|정의|(?:무슨|어떤)\s*뜻|이란|계산(?:법|식|해|하)|바꾸", prompt):
        return False
    scoped = _LIVE_ZONE.search(prompt) or re.search(r"내가|이번|지금까지", prompt)
    asks_area = _AREA_METRIC.search(prompt) or _NUMERIC_AREA_METRIC.search(prompt)
    if asks_area and (scoped or re.search(r"진행률|더\s*칠|못\s*가", prompt)):
        return True
    if _PROGRESS_HOLD.search(prompt) and re.search(r"도막|칠|정상|평균|빨간", prompt):
        return True
    return bool(scoped and _MEASURED_PROGRESS.search(prompt) and not re.search(r"마이크론|[µμ]m", prompt, re.I))


def actual_order_intent(prompt: str) -> str | None:
    if re.search(r"누락|빠진|빼먹|빠뜨|미수행|안\s*칠|못\s*칠|덜\s*칠|다시\s*칠|덧칠|재도장|보완", prompt) and not is_knowledge_ask(prompt):
        return "order_missing"
    # "칠한 순서도 숙련자랑 비슷해?"는 실제 순서가 아니라 순서 유사도다(판정 모델을 건너뛰면서 갈린다).
    if re.search(r"준수|유사도|점수|비슷|닮았|일치", prompt):
        return "order_adherence"
    # "지금까지 내가 칠한 구역 순서 알려줘"처럼 사이에 구역 낱말이 끼어도 실제 순서다(2026-11-24 음성 질의 시험).
    if re.search(r"실제|수행\s*순서|칠한\s*(?:구역|zone|존)?\s*순서|칠했던\s*순서|지금까지.*순서", prompt, re.I):
        return "order_actual"
    return None


def is_reference_order_question(prompt: str) -> bool:
    """Recognize explicit planning requests without borrowing actual-history facts."""
    if actual_order_intent(prompt) is not None or _EXCLUDED_ORDER.search(prompt) or _asks_to_rewrite_text(prompt):
        return False
    if is_current_next_question(prompt):
        return False
    if re.search(r"(?:순서|경로)\s*(?:란|의\s*(?:뜻|의미|정의))|순서.*(?:뭐야|무엇인가|뭔가요|정의|왜\s*중요)", prompt):
        return False
    # 표면처리·혼합 같은 공정 절차의 순서는 Zone 작업 순서가 아니라 문서 질문이다(2026-11-14 운영 q30).
    if re.search(_PROCEDURE, prompt) and not re.search(r"zone|존|구역|피도면|권장|추천|정석", prompt, re.I):
        return False
    # 문서 속 그림·차트의 순서는 등록된 Zone 기준 순서가 아니다(2026-11-17 사례 14). Zone 계획 대상이 있으면 위 절차 줄처럼 남긴다.
    if re.search(r"예시|그림|차트|견본|양식|계획서|마일스톤|이정표", prompt) and not re.search(r"zone|존|구역|피도면|권장|추천|정석", prompt, re.I):
        return False
    # "파이썬 작업 순서"처럼 작업·순서 낱말만 있는 다른 주제를 피도면 순서로 읽지 않는다.
    qualified = re.match(r"^\s*(?:권장|추천|정석|기준)\s*(?:작업\s*)?(?:순서|경로)", prompt)
    return bool(re.search(r"순서|경로|order|route", prompt, re.I) and (_REFERENCE_TARGET.search(prompt) or qualified))


def is_current_next_question(prompt: str) -> bool:
    if _asks_to_rewrite_text(prompt) or re.search(r"다음\s*(?:구역|zone|존)\s*(?:이란|의\s*(?:뜻|의미|정의))", prompt, re.I):
        return False
    # "지금 재생하는 곳 다음에는 어디를 칠해?"처럼 Zone 낱말 없이 묻기도 한다(2026-11-14 운영 q12).
    # "다음에는 어디를 조심해야 해?"는 진행 질문이 아니다. Zone 낱말이 없으면 "칠"이 함께 있어야 한다.
    # TSB 음성은 현재 세션을 따로 보내므로 "그 다음 순서는 뭐지"처럼
    # 구역 낱말을 생략한다. 짧은 조회형 전체 문장만 받아 절차·정의 질문은 남긴다.
    short_order = re.fullmatch(
        r"(?:(?:그럼|그러면|그|지금|이제))*다음(?:작업)?순서(?:는|가|를)?"
        r"(?:뭐(?:지|야|예요|에요|죠|인가요)?|무엇(?:인가요|이야|이죠|이지)?"
        r"|어떻게(?:돼|돼요|되나요|되는지)|알려(?:줘|주세요|줘요)|안내해줘)?",
        re.sub(r"[\s?？!.。]", "", prompt),
    )
    if short_order:
        return True
    zone_move = _MOVE_PERMISSION.search(prompt) and (
        _LIVE_ZONE.search(prompt) or re.search(r"진행률|반올림|건(?:을|은)?\s*(?:멈|내려|손에)", prompt))
    named_move = _LIVE_ZONE.search(prompt) and re.search(
        r"(?:\d+\s*번\s*(?:구역|zone|존)?|다음\s*(?:데|곳))(?:으?로|에)?\s*가(?:면|도)\s*(?:돼|되|됨|된|안\s*돼)",
        prompt, re.I,
    )
    gun_move = re.search(r"(?:지금|이제)\s*가도\s*(?:돼|되).{0,15}건(?:을|은)?\s*(?:손에|내려|쥐)", prompt)
    if (zone_move or named_move or gun_move) and not is_knowledge_ask(prompt):
        return True
    if re.search(r"\d+조는?\s*끝나.*다음\s*[쪼조]\s*.?가", prompt):
        return True
    zone = _NEXT_ZONE.search(prompt)
    if zone is not None:
        # "다음 구역의 두께"·"다음 구역에서 필요한 보호구"는 그 구역의 속성을 묻는다.
        tail = re.sub(r"[\s?？!.。]", "", prompt[zone.end():])
        if is_knowledge_ask(tail) and not re.fullmatch(r"(?:은|는|이|가)?어떻게(?:돼|돼요|되나요|되는지)", tail):
            return False
        return bool(_NEXT_ZONE_ASK.match(tail))
    return bool(re.search(r"다음\s*(?:에는|에|으로는|으로|엔|은)?\s*(?:어디|어느)", prompt)
                and re.search(r"칠|구역|zone|존", prompt, re.I))


_COMPLETION = re.compile(r"다\s*칠했|다\s*칠한\s*거|완료\s*(?:됐|되었|했|된\s*거|야|인가|냐|\?|$)|끝났|끝냈|넘어가도")
_NAMED_ZONE = re.compile(r"(?:zone|존|구역)\s*(\d+)|(\d+)\s*번\s*(?:zone|존|구역)", re.I)


def is_current_completion_question(prompt: str) -> bool:
    """훈련 중 "지금 구역 다 칠했어?"·"2번 구역 완료됐어?": 다음 구역 질문과 같은 완료 판정으로 답한다.

    판정 경로가 없으면 문서 검색이 "완료되었습니다"를 지어냈다(2026-11-24 음성 질의 시험, 시퀀스 0개에서도).
    """
    return bool(not _asks_to_rewrite_text(prompt) and _COMPLETION.search(prompt)
                and re.search(r"구역|zone|존|지금|현재|여기|이제", prompt, re.I)
                and not is_knowledge_ask(prompt))


_PROGRESS_UNAVAILABLE = {
    "session_id_missing": ("session_required", "훈련 세션 ID(session_id)가 없어 현재 구역의 진행 상태를 확인할 수 없습니다. 훈련을 시작한 뒤 세션 ID를 함께 보내 주세요."),
    "playback_live_progress_unavailable": ("analysis_unavailable", "재생 기록에서 현재 도막의 완료 여부를 판정할 자료가 없습니다. 훈련 세션의 분사 기록을 확인해 주세요."),
    "no_sequence_for_session": ("sequence_missing", "아직 받은 분사 기록이 없어 현재 구역의 완료 여부를 확인할 수 없습니다. 분사 시퀀스가 전송된 뒤 다시 질문해 주세요."),
    "current_sequence_unavailable": ("sequence_unavailable", "최근 분사 기록을 읽을 수 없어 현재 구역의 완료 여부를 확인할 수 없습니다. 시퀀스 전송 결과를 확인해 주세요."),
    "current_scenario_mismatch": ("scenario_mismatch", "질문의 콘텐츠와 최근 분사 기록의 콘텐츠가 달라 완료 여부를 확인할 수 없습니다. 현재 훈련의 scenario_id를 확인해 주세요."),
    "zone_unresolved": ("zone_unresolved", "최근 분사 기록에 유효한 구역 번호(Zone ID)가 없어 완료 여부를 확인할 수 없습니다. 구역이 인식된 뒤 다시 질문해 주세요."),
    "sequence_engine_unavailable": ("analysis_unavailable", "구역 진행 분석을 사용할 수 없어 완료 여부를 확인할 수 없습니다. 잠시 후 다시 질문해 주세요."),
    "coating_engine_unavailable": ("analysis_unavailable", "도막 분석을 사용할 수 없어 구역 완료 여부를 확인할 수 없습니다. 잠시 후 다시 질문해 주세요."),
    "progress_metric_unavailable": ("analysis_unavailable", "이 공정의 구역별 진행률 측정값이 없어 현재 진행률을 확인할 수 없습니다."),
    "heatmap_unavailable": ("heatmap_unavailable", "최근 분사 기록에 도막 히트맵이 없어 완료 여부를 확인할 수 없습니다. 히트맵을 포함한 시퀀스가 전송된 뒤 다시 질문해 주세요."),
    "heatmap_map_unavailable": ("heatmap_unavailable", "현재 구역이 속한 면의 도막 히트맵이 없어 완료 여부를 확인할 수 없습니다. 구역의 면 번호와 히트맵을 확인해 주세요."),
    "heatmap_dimensions_invalid": ("heatmap_unavailable", "도막 히트맵의 크기 또는 배치 좌표를 확인할 수 없어 구역 완료를 판정할 수 없습니다. 이미지와 크롭 정보를 확인해 주세요."),
    "heatmap_invalid": ("heatmap_unavailable", "도막 히트맵을 읽을 수 없어 구역 완료를 판정할 수 없습니다. 전송한 이미지 형식을 확인해 주세요."),
    "sequence_reference_unavailable": ("reference_unavailable", "해당 콘텐츠와 공정에 사용할 기준 순서가 없어 구역 순서를 안내할 수 없습니다."),
    "scenario_id_missing": ("scenario_required", "아직 받은 분사 기록이 없어 콘텐츠를 확인할 수 없습니다. 현재 훈련의 scenario_id를 함께 보내 주세요."),
    "coating_reference_unavailable": ("reference_unavailable", "해당 콘텐츠의 도막 분석 기준이 등록되어 있지 않아 완료 여부를 확인할 수 없습니다."),
    "coating_reference_invalid": ("reference_unavailable", "도막 판정에 사용할 공식 구역 기준이 제공되지 않아 완료 여부를 확인할 수 없습니다."),
}
_PROGRESS_UNKNOWN = "현재 구역의 완료 여부를 판단할 도막 분석 결과가 없습니다. 구역과 도막 데이터가 전송됐는지 확인해 주세요."


def progress_unavailability(reason: object) -> tuple[str, str]:
    """Select approved messages from the observed progress result."""
    default = ("analysis_unavailable", _PROGRESS_UNKNOWN)
    return _PROGRESS_UNAVAILABLE.get(reason, default) if isinstance(reason, str) else default


def _progress_unavailable(scope, reason: str):
    scope["reason_code"] = reason
    return progress_unavailability(reason)[1], scope, None


def _first_zone_answer(query, reports, sequence_engine, scope):
    """분사 기록이 아직 없으면 세션 시작 응답과 같은 첫 구역을 안내한다(완료 판정은 하지 않는다)."""
    get_session = getattr(reports, "get_session", None)
    session = get_session(query.session_id) if callable(get_session) else None
    scenario = query.scenario_id or getattr(session, "scenario_id", None)
    process = "blasting" if getattr(session, "tool_type", None) == "blasting_gun" else "spray"
    if not scenario:
        return _progress_unavailable(scope, "scenario_id_missing")
    if not callable(getattr(sequence_engine, "reference_plan", None)):
        return _progress_unavailable(scope, "sequence_engine_unavailable")
    plan = sequence_engine.reference_plan(scenario_id=scenario, process=process)
    if not plan or not plan.get("flow"):
        return _progress_unavailable(scope, "sequence_reference_unavailable")
    first = str(plan["flow"][0])
    scope.update(scope="reference_only", analysis_status="available", reason_code="no_sequence_yet",
                 scenario_id=scenario, process=process, next_zone_id=first)
    flow = " → ".join(str(zone) for zone in plan["flow"])
    return f"아직 분사 기록이 없습니다. {first}구역에서 시작하세요(권장 순서 {flow}).", scope, None


def current_next_answer(query, sequences, reports=None, *, sequence_engine=None, coating_engine=None):
    """Replay saved physical evidence through the telemetry gate, without DB writes."""
    scope = {"scope": "current_progress", "analysis_status": "not_available",
             "current_zone_id": None, "next_zone_id": None, "can_compare": False}
    ask_progress = is_current_progress_question(query.prompt)
    if (query.context or "").startswith("motion_viewer_"):
        return _progress_unavailable(scope, "playback_live_progress_unavailable")
    if not query.session_id:
        return _progress_unavailable(scope, "session_id_missing")
    latest = sequences.latest_sequence_id(query.session_id)
    if not latest:
        if ask_progress or is_current_completion_question(query.prompt):
            return _progress_unavailable(scope, "no_sequence_for_session")
        return _first_zone_answer(query, reports, sequence_engine, scope)
    body = sequences.load_body(session_key=query.session_id, sequence_id=latest)
    try:
        request = AnalyzeSequenceRequest.model_validate(body)
    except (TypeError, ValueError):
        return _progress_unavailable(scope, "current_sequence_unavailable")
    if query.scenario_id and query.scenario_id != request.scenario_id:
        selected_mode = learning_mode_from_scenario(query.scenario_id)
        if selected_mode is None or selected_mode != request.resolved_mode_id:
            return _progress_unavailable(scope, "current_scenario_mismatch")
    scope.update(scenario_id=request.scenario_id, sequence_id=request.sequence_id,
                 process=request.process_type, current_zone_id=request.zone_id)
    if str(request.zone_id).lower() in {"", "-1", "0", "none", "null"}:
        return _progress_unavailable(scope, "zone_unresolved")
    if sequence_engine is None:
        return _progress_unavailable(scope, "sequence_engine_unavailable")
    from trainer.demo_api.sequence_analysis import gate_sequence_decision
    decision = sequence_engine.evaluate(request, sequences)
    if request.process_type == "blasting":
        if ask_progress:
            return _progress_unavailable(scope, "progress_metric_unavailable")
        scope["analysis_status"] = decision.feedback.analysis_status
        if decision.feedback.analysis_status != "available":
            return _progress_unavailable(scope, "sequence_reference_unavailable")
        return decision.text_guide or _PROGRESS_UNKNOWN, scope, None
    if coating_engine is None:
        return _progress_unavailable(scope, "coating_engine_unavailable")
    get_session = getattr(reports, "get_session", None)
    session = get_session(query.session_id) if callable(get_session) else None
    target = coating_assessment.session_target(session)
    def prior_requests():
        for previous in sequences.session_bodies_through(request.session_id, request.sequence_id):
            if previous.get("sequence_id") != request.sequence_id:
                yield AnalyzeSequenceRequest.model_validate(previous)
    progress = coating_engine.evaluate_replay(request, prior_requests(), target_thickness_um=target,
                                             **coating_assessment.session_limits(session))
    scope.update(analysis_status="available" if progress.available else "not_available",
                 reason_code=progress.reason, status=progress.status)
    if not progress.available:
        return progress_unavailability(progress.reason)[1], scope, progress.model_dump(mode="json")
    decision = gate_sequence_decision(request, decision, progress)
    if (progress.zone_id == request.zone_id and progress.status == "complete"
            and progress.recommendation == "move_next_zone"):
        scope["next_zone_id"] = progress.next_zone_id
    # The telemetry guide remains short; Query returns the complete gate explanation.
    guide = decision.detail_guide or decision.text_guide or _PROGRESS_UNKNOWN
    if ask_progress and progress.zone_id == request.zone_id and progress.completion_percent is not None:
        named = _NAMED_ZONE.search(query.prompt)
        asked = (named[1] or named[2]) if named else None
        percentage = progress.completion_percent
        criterion = coating_assessment.GATE_COMPLETION_PERCENT
        guide = f"현재 {request.zone_id}구역 진행률은 {percentage:.2f}%입니다."
        area_threshold = _NUMERIC_AREA_METRIC.search(query.prompt)
        if (_AREA_METRIC.search(query.prompt) or area_threshold) and progress.coverage_percent is not None:
            if area_threshold and float(area_threshold[1]) != progress.target_thickness_um:
                guide += f" {area_threshold[1]}µm를 기준으로 한 면적 비율은 확인할 수 없습니다."
            if progress.target_thickness_um is not None:
                guide += f" 목표 {progress.target_thickness_um:g}µm 이상인 면적은 {progress.coverage_percent:.2f}%입니다."
        if scope["next_zone_id"]:
            guide += f" 이동 기준 {criterion:g}%를 충족해 {scope['next_zone_id']}구역을 안내합니다."
        elif progress.status != "complete":
            guide += f" 이동 기준 {criterion:g}%까지 {max(0.0, criterion - percentage):.2f}%p 남았습니다."
        if asked and asked != str(request.zone_id):
            guide = f"실시간 진행률은 지금 분사 중인 구역만 확인할 수 있습니다. {guide}"
    elif is_current_completion_question(query.prompt) and progress.zone_id == request.zone_id:
        # 완료를 물었으면 판정을 먼저 말한다. 다른 구역 번호를 물어도 훈련 중에는 지금 구역만 판정한다.
        named = _NAMED_ZONE.search(query.prompt)
        asked = (named[1] or named[2]) if named else None
        head = (f"지금 구역은 {request.zone_id}구역입니다. 훈련 중에는 지금 구역의 완료만 판정합니다. "
                if asked and asked != str(request.zone_id) else "")
        verdict = "완료 기준을 넘었습니다." if progress.status == "complete" else "아직 완료되지 않았습니다."
        guide = f"{head}{request.zone_id}구역은 {verdict} {guide}"
    return guide, scope, progress.model_dump(mode="json")


def _processes(text: str) -> set[str]:
    found = set()
    if re.search(r"스프레이|분사\s*도장|spray", text, re.I):
        found.add("spray")
    if re.search(r"블라스팅|블라스트|blasting|blast(?:ing)?_gun", text, re.I):
        found.add("blasting")
    return found


def _selector_context(value: str | None) -> Mapping[str, object]:
    """Client metadata can select a plan; it cannot provide measured results."""
    if not value:
        return {}
    try:
        raw = json.loads(value)
        return raw if isinstance(raw, dict) else {}
    except (TypeError, ValueError):
        return {}


def resolve_reference_target(query, sequences, reports=None, *, facts=None,
                             position: ReferencePosition | None = None) -> tuple[str | None, str | None]:
    """질문의 명시적 공정 선택을 제외하면 저장된 세션·분사 공정을 우선한다."""
    context = _selector_context(query.context)
    prompt_scenario = re.search(r"(?:학습\s*(?:모드)?|시나리오|scenario|mode)\s*[:#-]?\s*(\d+)", query.prompt, re.I)
    named_scenario = prompt_scenario.group(1) if prompt_scenario else None
    scenario = (named_scenario or query.scenario_id) if position else (query.scenario_id or named_scenario)
    scenario = scenario or context.get("scenario_id")
    if scenario is None and query.context:
        match = re.search(r"(?:학습\s*(?:모드)?|시나리오|scenario|mode)\s*[:#-]?\s*(\d+)", query.context, re.I)
        scenario = match.group(1) if match else None
    explicit = _processes(query.prompt)
    selector = set()
    if not explicit:
        process_value = context.get("process") or context.get("tool_type")
        process = process_from_tool_type(process_value)
        selector = {process} if process else _processes(query.context or "")
    process = next(iter(explicit)) if len(explicit) == 1 else None
    session = None
    if query.session_id:
        latest = sequences.latest_sequence_id(query.session_id)
        body = sequences.load_body(session_key=query.session_id, sequence_id=latest) if latest else None
        if scenario is None and body:
            scenario = body.get("scenario_id")
        if not explicit and body:
            setting = body.get("sequence_setting")
            process = process_from_tool_type(setting.get("tool_type")) if isinstance(setting, dict) else None
        get_session = getattr(reports, "get_session", None)
        session = get_session(query.session_id) if callable(get_session) else None
        if scenario is None:
            scenario = getattr(session, "scenario_id", None)
        if not explicit and process is None:
            process = process_from_tool_type(getattr(session, "tool_type", None))
    facts = facts or {}
    scenario = scenario or facts.get("scenario_id")
    if not explicit and process is None:
        process = process_from_tool_type(facts.get("process"))
    if not explicit and process is None and session is not None and not getattr(session, "tool_type", None):
        # 구형 시작 세션도 Start의 생략 기본값과 같다. 실제 도구가 있으면 위에서 이미 골랐다.
        process = "spray"
    if not explicit and process is None and len(selector) == 1:
        process = next(iter(selector))
    return (str(scenario) if scenario is not None else None), process


def reference_order_answer(engine, *, scenario_id: str | None, process: str | None, detailed: bool = False,
                           position: ReferencePosition | None = None):
    """Return only registered plan facts; never infer visits, scores or completion."""
    context = {
        "scope": "reference_only", "can_compare": False, "scenario_id": scenario_id,
        "process": process, "analysis_status": "not_available", "reference_basis": None,
        "recommended_flow": None, "reference_session_count": None, "reference_plans": [],
        "first_visit_flow": None, "adherence_rate": None, "position_similarity": None,
        "best_reference_similarity": None, "evaluated_visit_count": None,
        "on_track_visit_count": None, "direction_adherence_rate": None,
    }
    if not scenario_id:
        context["reason_code"] = "scenario_id_missing"
        return "어느 피도면의 작업 순서인지 확인하려면 학습 모드 또는 scenario_id가 필요합니다.", context
    plans = []
    if callable(getattr(engine, "reference_plan", None)):
        for selected in ([process] if process else ["spray", "blasting"]):
            plan = engine.reference_plan(scenario_id=scenario_id, process=selected)
            if plan is not None and plan.get("flow"):
                plans.append(plan)
    if not plans:
        label = {"spray": "스프레이 ", "blasting": "블라스팅 "}.get(process, "")
        context["reason_code"] = "sequence_reference_unavailable"
        return f"시나리오 {scenario_id}의 {label}작업 순서 기준이 현재 등록되어 있지 않습니다. 학습 모드와 공정에 맞는 기준 등록 여부를 확인해 주세요.", context
    context.update(analysis_status="available", reference_basis="measured_expert_runs", reference_plans=plans)
    if len(plans) == 1:
        context.update(process=plans[0]["process"], recommended_flow=plans[0]["flow"],
                       reference_session_count=plans[0]["reference_session_count"])
    mode = learning_mode_from_scenario(scenario_id)
    label = f"학습 모드 {mode}" if mode is not None else f"시나리오 {scenario_id}"
    if position:
        kind, anchor = position
        lines = []
        for plan in plans:
            process_label = "스프레이" if plan["process"] == "spray" else "블라스팅"
            flow = [str(zone) for zone in plan["flow"]]
            if kind == "first":
                result = f"첫 구역: Zone {flow[0]}"
            elif anchor not in flow:
                result = f"Zone {anchor} 기준 순서 확인 불가"
            elif anchor == flow[-1]:
                result = f"Zone {anchor}이 마지막 구역"
            else:
                result = f"Zone {anchor} 다음: Zone {flow[flow.index(anchor) + 1]}"
            lines.append(f"{process_label} {result}")
        return f"{label} 등록 기준 — {'; '.join(lines)}.", context
    lines = []
    for plan in plans:
        process_label = "스프레이" if plan["process"] == "spray" else "블라스팅"
        lines.append(f"{process_label}: Zone " + " → ".join(str(zone) for zone in plan["flow"]))
    if not detailed:  # 빠른 안내 한 줄(60자): 공정이 둘이면 화살표 목록은 공통 출구(quick_answer)가 앞부분과 개수로 줄인다
        return label + " 등록 기준 순서: " + ", ".join(line.replace(": Zone", " Zone") for line in lines) + ".", context
    answer = label + " 현재 등록된 기준 순서 — " + "; ".join(lines) + "."
    if detailed:
        counts = ", ".join(("스프레이" if plan["process"] == "spray" else "블라스팅")
                           + f" {plan['reference_session_count']}회" for plan in plans)
        answer += f" 전문가({counts}) 기준이며 실제 수행·완료 판정은 아닙니다."
    return answer, context
