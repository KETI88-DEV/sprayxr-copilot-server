"""Registered work plans, separate from measured trainee progress."""
from __future__ import annotations

import json
import re
from collections.abc import Mapping

from trainer.demo_api import coating_assessment
from trainer.demo_api.telemetry_contracts import AnalyzeSequenceRequest, learning_mode_from_scenario, process_from_tool_type


_PROCEDURE = r"표면\s*처리|전처리|세척|탈지|건조|혼합|희석|교반|프라이머|절차|단계|공정|도장\s*전에|칠하기\s*전"


# "재도장 간격은?", "다시 칠할 때 주의할 점은?"처럼 방법·조건을 묻는 지식 질문의 표지(2026-11-14 독립 검토).
_KNOWLEDGE_ASK = r"간격|주의|조심|방법|요령|시간|얼마|어떻게|왜|원인|이유|기준|규정|예방|조건"


def is_knowledge_ask(prompt: str) -> bool:
    """진행 낱말(보완·재도장·다음)이 있어도 방법·절차·조건을 묻는 지식 질문인가."""
    return bool(re.search(_KNOWLEDGE_ASK, prompt) or re.search(_PROCEDURE, prompt))


def actual_order_intent(prompt: str) -> str | None:
    if re.search(r"누락|빠진|빼먹|빠뜨|미수행|안\s*칠|못\s*칠|덜\s*칠|다시\s*칠|덧칠|재도장|보완", prompt) and not is_knowledge_ask(prompt):
        return "order_missing"
    if re.search(r"준수|유사도|점수", prompt):
        return "order_adherence"
    if re.search(r"실제|수행\s*순서|칠한\s*순서", prompt):
        return "order_actual"
    return None


def is_reference_order_question(prompt: str) -> bool:
    """Recognize explicit planning requests without borrowing actual-history facts."""
    if actual_order_intent(prompt) is not None:
        return False
    if is_current_next_question(prompt):
        return False
    if re.search(r"(?:순서|경로)\s*(?:란|의\s*(?:뜻|의미|정의))|순서.*(?:뭐야|무엇인가|뭔가요|정의|왜\s*중요)", prompt):
        return False
    # 표면처리·혼합 같은 공정 절차의 순서는 Zone 작업 순서가 아니라 문서 질문이다(2026-11-14 운영 q30).
    if re.search(_PROCEDURE, prompt) and not re.search(r"zone|존|구역|피도면|권장|추천|정석", prompt, re.I):
        return False
    order = re.search(r"순서|경로|order|route", prompt, re.I)
    target = re.search(r"작업|도장|피도면|구역|zone|존|블라스팅|스프레이|권장|추천|정석|기준|reference|recommend", prompt, re.I)
    return bool(order and target)


def is_current_next_question(prompt: str) -> bool:
    if re.search(r"다음\s*(?:구역|zone|존)\s*(?:이란|의\s*(?:뜻|의미|정의))", prompt, re.I):
        return False
    # "지금 재생하는 곳 다음에는 어디를 칠해?"처럼 Zone 낱말 없이 묻기도 한다(2026-11-14 운영 q12).
    # "다음에는 어디를 조심해야 해?"는 진행 질문이 아니다. Zone 낱말이 없으면 "칠"이 함께 있어야 한다.
    return bool(re.search(r"다음\s*(?:구역|zone|존)", prompt, re.I)
                or (re.search(r"다음\s*(?:에는|에|으로는|으로|엔|은)?\s*(?:어디|어느)", prompt) and re.search(r"칠|구역|zone|존", prompt, re.I)))


def current_next_answer(query, sequences, reports=None, *, sequence_engine=None, coating_engine=None):
    """Replay saved physical evidence through the telemetry gate, without DB writes."""
    scope = {"scope": "current_progress", "analysis_status": "not_available",
             "current_zone_id": None, "next_zone_id": None, "can_compare": False}
    unknown = "현재 구역의 완료 여부를 확인할 수 없습니다. 구역 이동 전 현재 도막과 공식 구역 데이터를 확인해 주세요."
    if not query.session_id or (query.context or "").startswith("motion_viewer_"):
        scope["reason_code"] = "current_progress_unavailable"
        return unknown, scope, None
    latest = sequences.latest_sequence_id(query.session_id)
    body = sequences.load_body(session_key=query.session_id, sequence_id=latest) if latest else None
    try:
        request = AnalyzeSequenceRequest.model_validate(body)
    except (TypeError, ValueError):
        scope["reason_code"] = "current_sequence_unavailable"
        return unknown, scope, None
    if query.scenario_id and query.scenario_id != request.scenario_id:
        selected_mode = learning_mode_from_scenario(query.scenario_id)
        if selected_mode is None or selected_mode != request.resolved_mode_id:
            scope["reason_code"] = "current_scenario_mismatch"
            return unknown, scope, None
    scope.update(scenario_id=request.scenario_id, sequence_id=request.sequence_id,
                 process=request.process_type, current_zone_id=request.zone_id)
    if str(request.zone_id).lower() in {"", "-1", "0", "none", "null"}:
        scope["reason_code"] = "zone_unresolved"
        return unknown, scope, None
    if sequence_engine is None:
        scope["reason_code"] = "sequence_engine_unavailable"
        return unknown, scope, None
    from trainer.demo_api.sequence_analysis import gate_sequence_decision
    decision = sequence_engine.evaluate(request, sequences)
    if request.process_type == "blasting":
        scope["analysis_status"] = decision.feedback.analysis_status
        return decision.text_guide or unknown, scope, None
    if coating_engine is None:
        scope["reason_code"] = "coating_engine_unavailable"
        return unknown, scope, None
    get_session = getattr(reports, "get_session", None)
    session = get_session(query.session_id) if callable(get_session) else None
    target = coating_assessment.session_target(session)
    def prior_requests():
        for previous in sequences.session_bodies_through(request.session_id, request.sequence_id):
            if previous.get("sequence_id") != request.sequence_id:
                yield AnalyzeSequenceRequest.model_validate(previous)
    progress = coating_engine.evaluate_replay(request, prior_requests(), target_thickness_um=target,
                                             **coating_assessment.session_limits(session))
    decision = gate_sequence_decision(request, decision, progress)
    scope.update(analysis_status="available" if progress.available else "not_available",
                 reason_code=progress.reason, status=progress.status)
    if (progress.available and progress.zone_id == request.zone_id and progress.status == "complete"
            and progress.recommendation == "move_next_zone"):
        scope["next_zone_id"] = progress.next_zone_id
    return decision.text_guide or unknown, scope, progress.model_dump(mode="json")


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


def resolve_reference_target(query, sequences, reports=None, *, facts=None) -> tuple[str | None, str | None]:
    """Explicit request metadata wins over the latest saved session telemetry."""
    context = _selector_context(query.context)
    prompt_scenario = re.search(r"(?:학습\s*(?:모드)?|시나리오|scenario|mode)\s*[:#-]?\s*(\d+)", query.prompt, re.I)
    scenario = query.scenario_id or (prompt_scenario.group(1) if prompt_scenario else None)
    scenario = scenario or context.get("scenario_id")
    if scenario is None and query.context:
        match = re.search(r"(?:학습\s*(?:모드)?|시나리오|scenario|mode)\s*[:#-]?\s*(\d+)", query.context, re.I)
        scenario = match.group(1) if match else None
    explicit = _processes(query.prompt)
    if not explicit:
        process_value = context.get("process") or context.get("tool_type")
        process = process_from_tool_type(process_value)
        explicit = {process} if process else _processes(query.context or "")
    process = next(iter(explicit)) if len(explicit) == 1 else None
    body = None
    if query.session_id:
        latest = sequences.latest_sequence_id(query.session_id)
        body = sequences.load_body(session_key=query.session_id, sequence_id=latest) if latest else None
        if scenario is None and body:
            scenario = body.get("scenario_id")
        if not explicit and body:
            setting = body.get("sequence_setting")
            process = process_from_tool_type(setting.get("tool_type")) if isinstance(setting, dict) else None
        if scenario is None:
            get_session = getattr(reports, "get_session", None)
            session = get_session(query.session_id) if callable(get_session) else None
            scenario = getattr(session, "scenario_id", None)
    facts = facts or {}
    scenario = scenario or facts.get("scenario_id")
    if not explicit and process is None:
        process = process_from_tool_type(facts.get("process"))
    return (str(scenario) if scenario is not None else None), process


def reference_order_answer(engine, *, scenario_id: str | None, process: str | None, detailed: bool = False):
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
    lines = []
    for plan in plans:
        label = "스프레이" if plan["process"] == "spray" else "블라스팅"
        lines.append(f"{label}: Zone " + " → ".join(str(zone) for zone in plan["flow"]))
    mode = learning_mode_from_scenario(scenario_id)
    label = f"학습 모드 {mode}" if mode is not None else f"시나리오 {scenario_id}"
    answer = label + " 현재 등록된 기준 순서 — " + "; ".join(lines) + "."
    if detailed:
        counts = ", ".join(("스프레이" if plan["process"] == "spray" else "블라스팅")
                           + f" {plan['reference_session_count']}회" for plan in plans)
        answer += f" 전문가({counts}) 기준이며 실제 수행·완료 판정은 아닙니다."
    return answer, context
