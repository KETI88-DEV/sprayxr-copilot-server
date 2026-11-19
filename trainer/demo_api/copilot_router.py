"""POST /api/v1/copilot/query (v2.5).

``task_type`` selects a preferred training analysis. General explanations and
semantic document follow-ups preserve the question's meaning across these tabs.
Verified measurements keep their engine path; generated dialogue is not evidence.

* ``knowledge_search``: the live pipeline singleton shared with ``/api/chat``
  runs ``run(force_route="fast")`` in a worker thread inside the shared
  generation slot. ``use_rag='N'`` skips retrieval and returns no citations.
  ``answer`` is the parsed prose alone and ``citations`` is what it cites (a
  subset of this request's retrieved chunks), matching the §9.2 example: the
  ``[답변]``/``[근거]`` scaffold the model is trained to emit is stripped at this
  boundary, so no document id reaches ``answer`` at all. ``degraded`` is true when
  no full answer was produced or generation hit its deadline.
* ``pose_recomm`` (명세서 p.9 "전문가 작업 자세 유사도 비교"): 이미 올라간 시퀀스를
  채점한다. 대상은 ``context``(예: ``SEQ-012``)가 가리키는 시퀀스이고, 없으면 그
  ``session_id``의 가장 최근 시퀀스다. 채점은 MCP 자세 엔진에 위임하고 결과는
  ``pose_context``에 싣는다. **자세는 여기서 채점한다** — ``telemetry/sequence``는
  시퀀스를 데이터베이스에 올리는 일만 한다(p.4).
* ``trajectory_recomm``: 등록된 기준 순서와 저장된 수행 순서·진행 결과로 답한다.
  순서의 이유·방법을 묻는 설명 질문은 관련 자료를 제공한 지식 생성 경로로 보낸다.
* ``error_analysis``: 완료 리포트에 저장된 Zone별 도막 평가(분포, 저도막·과도막 Zone의 조절 후보)로
  답한다. 뷰어 문맥에 서명된 목표 대비 판정이 있으면 그것으로, 없으면 리포트로 답한다. 모델을 부르지 않는다. 리포트가 없으면 자료가 없다는 안내와 ``context_status="not_available"``.

Every response uses the v2.5 wrapper. Request validation failures under
``/api/v1/`` are wrapped as ERROR 422; other routes keep FastAPI's format.
"""

from __future__ import annotations

import json
import inspect
import logging
import math
import os
import re
import time
from contextlib import nullcontext
from urllib.parse import quote
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from types import MappingProxyType
from typing import TYPE_CHECKING, Annotated, Any, Final

from anyio import to_thread
from fastapi import APIRouter, Body, FastAPI, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from trainer.demo_api.copilot_contracts import CopilotQueryData, CopilotQueryRequest, CopilotResponse
from trainer.demo_api.coating_query import (
    answer_coating_question, answer_distribution_cause, drop_contradicted_causes, parse_viewer_coating_context,
    report_coating_context, target_advice_facts, zone_measures)
from trainer.demo_api.copilot_intent import RECORD_OBJECT, classify_question_intent, fallback_intent, preset_intent, is_coating_status_question, is_coating_advice_question, is_domain_question, is_general_painting_knowledge, is_explanatory_question, is_off_topic
from trainer.demo_api.dialogue import (GENERAL_KNOWLEDGE_LIMIT, INSTRUCTION_REFUSAL, asks_for_instructions, authenticated_facts, conversation_key,
                                       drop_misattributed_zones, drop_sentences, drop_ungrounded, guard_generated_answer, has_body, parse_plan, plain_answer,
                                       viewer_claims, zone_fact_numbers,
                                       selected_direction_answer, ungrounded_measurements, viewer_findings_answer,
                                       viewer_measurement_answer, viewer_metadata_context, dialogue_turns_for_model, dialogue_resolution_advice, without_hanzi)
from trainer.demo_api.playback_direction import playback_direction_from_context, playback_direction_answer, simple_direction_question, requested_direction_scope, aggregate_direction, LABELS as DIRECTION_LABELS
from trainer.demo_api.copilot_live_analysis import current_zone, live_analysis
from trainer.demo_api.simulator_context import SIMULATOR_CONTEXT, is_blasting, wants_simulator_context, simulator_configuration_facts
from trainer.demo_api.copilot_session_context import build_query_instruction, summarize_session, viewer_definition_context
from trainer.demo_api.context_prompts import context_text, matched_prompt
from trainer.demo_api.term_rules import matched_terms, term_context, with_document_terms
from trainer.demo_api.pose_query import (
    answer_order_question,
    order_request_fields,
    answer_pose_question,
    answer_sequence_question,
    answer_zone_question,
    NOT_JOINT,
    angle_note,
    asked_arm,
    asked_part,
    prefers_hand,
    asks_about_zones,
    asks_whole_session,
    score_pose,
    unsupported_pose_part,
    requested_zone,
    zone_unavailable,
)
from trainer.demo_api.query_language import TranslationUnavailableError, translate_text, translation_references
from trainer.demo_api.glossary_query import lookup_translation_question, parse_translation_request, requested_response_language
from trainer.multilingual.tsb_reference import reference_version
from trainer.demo_api.sequence_store import SequenceStore
from trainer.demo_api.viewer_arm_pose import arm_pose_answer, parse_viewer_arm_pose, zone_pose_summary
from trainer.demo_api.sequence_analysis import SequenceAnalysisEngine
from trainer.demo_api.safety_guard import (
    CLASSES as SAFETY_CLASSES, HIGH_PRESSURE_ANSWER, HIGH_PRESSURE_CITATIONS, classify, fixed_answer_translation, guard_answer, hazard_class,
)
from trainer.demo_api.reference_order_query import actual_order_intent, current_next_answer, is_current_next_question, is_reference_order_question, resolve_reference_target, reference_order_answer
from trainer.demo_api.v1_store import V1Store
from trainer.demo_api.record_query import BODY_PART, DIRECTION_WORD, FIRST_FIX, WORST_ZONE, RecordAnswer, RecordFacts, answer_record_question, split_parts, ground_generated, has_record_marker, previous_subject
from trainer.demo_api.viewer_progress import progress_question, viewer_progress_answer
from trainer.demo_api.query_runtime import (generation_slot_for, query_model_name, retrieval_question,
                                          safe_model_name, select_query_pipeline, translation_pipeline_for)
from trainer.demo_api.quick_answer import summarize_query_answer
from trainer.scripts.answer_semantics import cacheable_candidate
from trainer.scripts.generation_slot import GENERATION_SLOT  # Legacy slot identity for unconfigured clients.
from trainer.scripts.live_response_contract import (
    FAILED_REPAIR_PATHS,
    parse_knowledge_answer,
    retrieved_chunk_ids,
)
from trainer.scripts.openai_backend import OpenAIBackendUnavailableError, generation_deadline_scope

if TYPE_CHECKING:
    from trainer.demo_api.chat_service import _Pipeline

logger = logging.getLogger(__name__)

COPILOT_QUERY_PATH: Final[str] = "/api/v1/copilot/query"
_API_V1_PREFIX: Final[str] = "/api/v1/"
_SUCCESS_MESSAGE: Final[str] = "Copilot query completed"
_MOTION_VIEWER_CONTEXT_PREFIX: Final[str] = "motion_viewer_analysis:"

# Answers for task types whose MCP tools are not finished (plan D2).
PENDING_TASK_ANSWERS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "trajectory_recomm": "이동 궤적·도장 순서 추천 기능은 준비 중입니다. 도장 지식 질문에는 지금 답변할 수 있습니다.",
        "error_analysis": "이 세션에는 완료 분석의 도막 결과가 없어 도막 분포 이상 원인을 분석할 수 없습니다. 세션을 마친 뒤 다시 질문해 주세요.",
    }
)

# 자세 채점이 대상을 찾지 못했을 때의 답변. 왜 못 했는지를 말한다 — 점수를 지어내지 않는다.
_POSE_REASON_ANSWERS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "session_id_missing": "어느 세션의 자세를 비교할지 알 수 없습니다. session_id를 함께 보내주세요.",
        "no_sequence_for_session": "이 세션에 올라온 분사 시퀀스가 없습니다. 먼저 시퀀스를 전송해 주세요.",
        "sequence_not_found": "가리키신 시퀀스를 찾지 못했습니다. context의 시퀀스 식별자를 확인해 주세요.",
        "engine_unavailable": "자세 분석 엔진에 연결할 수 없습니다.",
    }
)
_POSE_FALLBACK_ANSWER: Final[str] = "이 시퀀스로는 자세를 판정할 수 없습니다."

PipelineProvider = Callable[[], "_Pipeline"]


class _InvalidViewerPlayback(ValueError):
    """A signed evaluation was accompanied by an invalid playback selector."""


def _validated_viewer_playback(value: object, analysis: Mapping[str, object], *, task_type: str) -> dict[str, object]:
    # Playback is client state, not a measurement. Only allow it to select a
    # map/Zone already authenticated by the server's evaluation signature.
    allowed = {"playhead_seconds", "source_index", "map_num", "zone_id"}
    if not isinstance(value, dict) or set(value) - allowed:
        raise _InvalidViewerPlayback
    seconds = value.get("playhead_seconds")
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
        raise _InvalidViewerPlayback
    try:
        finite_seconds = math.isfinite(seconds)
    except OverflowError:
        raise _InvalidViewerPlayback from None
    if not finite_seconds or seconds < 0:
        raise _InvalidViewerPlayback
    source_index = value.get("source_index")
    if source_index is not None and (isinstance(source_index, bool) or not isinstance(source_index, int) or source_index < 0):
        raise _InvalidViewerPlayback
    map_num, zone_id = value.get("map_num"), value.get("zone_id")
    result: dict[str, object] = {"playhead_seconds": seconds, "map_num": None, "zone_id": None}
    if source_index is not None:
        result["source_index"] = source_index
    if map_num is None and zone_id is None:
        return result
    if any(isinstance(item, bool) or not isinstance(item, (str, int)) for item in (map_num, zone_id)):
        raise _InvalidViewerPlayback
    map_text, zone_text = str(map_num), str(zone_id)
    if not map_text.isdigit() or not zone_text.isdigit() or int(zone_text) <= 0:
        raise _InvalidViewerPlayback
    key = f"{map_text}:{zone_text}"
    if task_type == "pose_recomm":
        zones = analysis.get("zones")
        row = zones.get(key) if isinstance(zones, Mapping) else None
        if not isinstance(row, Mapping) or str(row.get("zone_id")) != zone_text or str(row.get("map_num", map_text)) != map_text:
            raise _InvalidViewerPlayback
    elif task_type == "trajectory_recomm":
        keys = analysis.get("zone_keys")
        if not isinstance(keys, list) or key not in keys:
            raise _InvalidViewerPlayback
    else:
        raise _InvalidViewerPlayback
    result.update(map_num=map_text, zone_id=zone_text)
    return result


def _parse_motion_viewer_context(value: str | None, *, task_type: str) -> dict[str, object] | None:
    """Accept viewer measurements only when signed by the analysis server."""
    if not value or not value.startswith(_MOTION_VIEWER_CONTEXT_PREFIX):
        return None
    try:
        payload = json.loads(value[len(_MOTION_VIEWER_CONTEXT_PREFIX):])
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    analysis_context = payload.get("analysis_context")
    if not isinstance(analysis_context, dict):
        return None
    signature = payload.get("signature")
    if not isinstance(signature, str):
        return None
    # Transport-only record objects are not part of the signed analysis.
    analysis_context = {key: item for key, item in analysis_context.items() if key != "record"}
    from trainer.demo_api.viewer_context import verify_context
    if not verify_context(task_type, analysis_context, signature):
        return None
    if "playback_context" in payload:
        analysis_context["playback_context"] = _validated_viewer_playback(
            payload["playback_context"], analysis_context, task_type=task_type
        )
    return analysis_context


# analysis(채점·분석 결과를 옮긴 답, 모델 미사용) · documents(문서 검색 + 모델) · conversation(모델 대화)
ANSWER_SOURCE_HEADER: Final = "X-Copilot-Answer-Source"
MODEL_HEADER: Final = "X-Copilot-Model"
TRANSLATION_MODEL_HEADER: Final = "X-Copilot-Translation-Model"
SUBJECT_HEADER: Final = "X-Copilot-Subject"  # 서버 안에서만 쓴다. 바깥 응답에는 싣지 않는다.
# 외국어 질문을 한국어로 옮긴 문장(퍼센트 인코딩). 응답 본문은 바꾸지 않는다.
ANALYSIS_PROMPT_HEADER: Final = "X-Copilot-Analysis-Prompt"
_EVALUATION_QUESTION: Final = re.compile(r"잘한|못한|고쳐야|고칠|개선할|개선점|문제점|부족한 점")
# Preserve main ebb923c3's narrowly scoped v7.1 diagnostic-scaffold repair.
_DIAGNOSTIC_SCAFFOLD: Final = re.compile(r"(?m)^\s*\[(?:분석|판정|원인|조치)\]")
# 자세 탭에서 무엇을 물을 수 있는지 묻거나 낱말을 옮겨 달라는 말은 채점 조회가 아니다(사례 11·12). 자세 탭에만 쓴다.
_TAB_OR_WORD: Final = re.compile(r"탭|(?:물어볼|질문할)\s*수|영어로|한자로|영어\s*(?:단어|표현)")
_WORK_SCORE_ASKED: Final = re.compile(r"작업\s*(?:유사도|점수)|종합\s*유사도")
# 자세를 묻는 말. 팔 채점이 없으면 작업 유사도(다른 값)로 대신 답하지 않는다(2026-11-17 사용자 결정 I7).
_POSE_ASKED: Final = re.compile(r"자세(?!히|하게|한)|동작|관절|posture|pose", re.I)
NO_ARM_POSE: Final = "이 기록에는 팔 자세 채점 결과가 없어 자세 유사도를 알 수 없습니다."
NO_ARM_WORK_NOTE: Final = "참고로 아래는 작업 유사도(거리·각도·방향 비교, 자세 유사도와 다른 값)입니다."
# 이의 제기 말투("아까 답 이상한데", "아닌 거 같은데", "다시 봐줘"). 이것과 군말을 걷어 내고 남는 내용이 없으면
# 이의 제기만 한 말이다. 내용(주장·질문)이나 숫자가 남으면 앞 질문을 다시 답하지 않는다(사용자 결정 I6).
_OBJECTION_WORDS: Final = re.compile(
    r"이상(?:한데|하네|해요?|하다|하지\s*않아)?|틀린|틀렸|틀리|잘못|아닌\s*(?:거|것)\s*같|아니잖아|아닌데|아니야|아냐|아니"
    r"|맞아|맞나|맞는\s*거|확실해|진짜|정말|다시|확인|아까|방금|그거|그건|이거|답|대답|말|좀|봐|줘|해|요|거|같은데|같아|[?？!.~,·\s]")


def _objection_only(prompt: str) -> bool:
    return not re.search(r"\d", prompt) and len(_OBJECTION_WORDS.sub("", prompt)) <= 2


def _no_arm_notice(prompt: str, side: str | None) -> str:
    """팔 채점 없이 자세·신체 부위를 물은 질문의 첫 문장. 작업 유사도 수치는 넣지 않는다."""
    part = BODY_PART.search(prompt)
    if part is None and side is None:
        return NO_ARM_POSE
    label = " ".join(part.group(0).split()) if part else "팔별"
    return f"팔 자세 채점 결과가 없어 {label} 교정 여부는 알 수 없습니다."
# 질문형 문장. 지식 탭에서 계획 모델이 "chat"으로 본 질문 가운데 이것만 먼저 문서로 답해 본다.
# 인사·맞장구("안녕", "고마워", "야")는 걸리지 않아 지금처럼 대화 경로로 간다.
_KNOWLEDGE_QUESTION: Final = re.compile(r"[?？]|뭐|무엇|무슨|몇|어떻|어떤|왜|얼마|언제|인가|인지|나요|까요|되나|돼야|해야|하나요")


# 섞인 질문(기록 조각 + 지식 조각)의 문서 쪽 지시와, 문서 답이 비었을 때 기록 문장 뒤에 붙이는 문장.
MIXED_DOCUMENT_RULE: Final = "이 기록의 수치·Zone·판정은 말하지 말고 문서의 일반 기준만 말하세요."
MIXED_NO_EVIDENCE: Final = "문서에서는 이 질문에 답할 근거를 찾지 못했습니다."


# 기록 없는 세션의 후속 질문("그럼 속도는?")은 대화로 둔다.
_FOLLOWUP: Final = re.compile(r"나의|저의|그럼|그렇게")
# 규정·정의·권장값을 묻는 문형. 이것이면 후속·1인칭 낱말이 있어도 문서로 간다. "몇 점 이상"은 되고 "몇 점이야"는 기록 질문이다.
_KNOWLEDGE_SHAPE: Final = re.compile(r"기준|뜻|의미|무엇|뭐|이란|적정|권장|얼마|무슨|어떻게|몇\s*(?:분|시간)|몇\s*점\s*이상|합격\s*점수")


def _grounded_route(prompt: str, subject: Mapping[str, Any] | None, *, viewer: bool = True) -> bool:
    """기록이나 도장 지식을 묻는 질문인가. 그러면 대화 모델 대신 근거 있는 경로로 답한다.

    서명된 기록이 있으면: 기록 표지, 또는 도메인 낱말 + 질문형.
    기록이 없으면(사용자 결정 4): 도메인 낱말이 있고, 기록 표지면 세션 대화, 지식 문형이면 문서, 후속 형태면 세션 대화, 나머지는 문서.
    1인칭("제·나")은 도메인 낱말과 함께일 때만 표지다("제 이름 뭐예요?"는 잡담).
    """
    domain = is_domain_question(prompt)
    if viewer:
        return has_record_marker(prompt, previous_subject=subject, person=domain) or (domain and bool(_KNOWLEDGE_QUESTION.search(prompt)))
    if not domain:
        return False
    if has_record_marker(prompt, previous_subject=subject):
        return False  # 내 기록을 묻는 말("내 속도는 얼마야?")은 지식 문형이어도 세션 대화다.
    return bool(_KNOWLEDGE_SHAPE.search(prompt)) or not _FOLLOWUP.search(prompt)


def _abstain_below(pipeline: object, response_mode: str) -> float | None:
    """상세 분석은 confidence.abstain_below_rerank_score, 빠른 안내는 quick_abstain_below_rerank_score. 없거나 숫자가 아니면 근거 확인 단계를 끈다."""
    config = getattr(pipeline, "config", None)
    confidence = config.get("confidence") if isinstance(config, dict) else None
    key = "abstain_below_rerank_score" if response_mode == "detailed" else "quick_abstain_below_rerank_score"
    value = confidence.get(key) if isinstance(confidence, dict) else None
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


# 근거 확인 단계(confidence.abstain_below_rerank_score)가 막은 답. 2026-11-13 2-1 측정에서 도입.
NO_EVIDENCE_ANSWER: Final = ("제공된 자료에서 이 질문에 답할 근거를 찾지 못했습니다. "
                             "대상(도장 방식·제품·구역)을 함께 적어 다시 물어 주세요.")


# 근거가 약한 기술 질문 중 정의만 묻는 질문은 일반 설명으로 답한다(2026-11-16 사용자 결정 B안).
# v7 평가에서 문서 없는 대화 답이 "헤그만 눈금 6 = 6µm" 같은 수치를 지어냈다. 검색이 놓친 24건 중 v6.1이 일반 지식으로
# 맞힌 것은 정의 질문뿐이고 원인·절차·수치 질문은 틀렸다. 그래서 정의 질문만 열고 숫자가 든 문장은 뺀다.
HEDGED_PREFIX: Final = "제공된 자료에서는 근거를 찾지 못해 일반적인 설명으로 답합니다. "
QUICK_HEDGED_PREFIX: Final = "문서 밖 일반 설명: "
_TECHNICAL: Final = re.compile(r"µm|μm|㎛|mm|°C|게이지|측정|시험|규격|LD50|독성|위험물|UN\s*번호|용제|크실렌|톨루엔|선체|선박|수선|블록"
                               r"|도면|탱크|부식|방청|녹|강판|철판|용접|그릿|숏|막후계|조도|프로파일|스크러버|집진|배기|SDS|ISO|ASTM|SSPC"
                               r"|NACE|PSPC|예시|그림|양식|견본|계획서|차트", re.I)
_DEFINITION: Final = re.compile(r"뭐야|뭐지|뭔데|뭔가요|뭐예요|뭐에요|무엇인가|무엇이야|무엇인지|뜻|의미|정의|란\s*무엇|어떤\s*(?:장치|현상|도료|것)|차이")
_NOT_DEFINITION: Final = re.compile(r"\d|몇|얼마|%|µm|μm|㎛|mm|°|시간|온도|압력|비율|희석|두께|간격|규격|기준|수치|값|번호|등급|예시|도면|그림"
                                    r"|견본|양식|계획서|표에서|제품|어떻게|방법|절차|순서|원인|이유|왜|조건"
                                    r"|(?<![A-Za-z])[A-Z][A-Z.]{1,7}(?![A-Za-z])")  # 약어는 분야마다 뜻이 달라 문서 없이 풀지 않는다(SVR→회귀 모형)


def is_technical_question(prompt: str) -> bool:
    return is_domain_question(prompt) or bool(_TECHNICAL.search(prompt))


def allows_general_definition(prompt: str) -> bool:
    """근거가 약할 때 일반 설명으로 답해도 되는 질문: 정의·뜻만 묻고 수치·규격·절차·원인·문서 속 예시를 묻지 않는다."""
    return bool(_DEFINITION.search(prompt)) and not _NOT_DEFINITION.search(prompt)


def hedged_answer(answer: str, *, quick: bool = False) -> str | None:
    """일반 설명 답: 숫자가 든 문장을 빼고 근거 없음을 앞에 밝힌다. 남는 문장이 없으면 None(기권)."""
    kept, _ = drop_sentences(plain_answer(answer), lambda part: bool(re.search(r"\d", part)))
    if not has_body(kept) or re.search(r"(?:확인할|답할) 수 없습니다", kept):  # 출구 검사가 남긴 대체 문장은 일반 설명이 아니다
        return None
    return (QUICK_HEDGED_PREFIX if quick else HEDGED_PREFIX) + kept.strip()


# 어지럼을 묻는데 문서에 근거가 없을 때 근거 없는 답 뒤에 붙이는 한 줄(숫자 없음).
DIZZY_NOTE: Final = "몸이 불편하면 멈추고 교관에게 알리세요."
_DIZZY: Final = re.compile(r"어지럼|어지러|어지럽|멀미|메스꺼|울렁")


def _session_findings(found: tuple[str, Mapping[str, Any]] | None) -> str:
    """평가를 묻는 자유 질문의 답: 완료 분석의 판정(자세 최저 구역, 도막 과다·부족 구역)을 그대로 전한다.

    모델에 판정을 넘겨 풀어 쓰게 하면 8B가 없는 판정을 만든다(2026-11-11 Spark: 도막이 정상인 9번 구역을 "도막 부족").
    """
    pose, coating = answer_zone_question(found, context=None), report_coating_context(found)
    parts = [pose.answer] if pose.analysis_status == "available" else []
    if coating["analysis_status"] == "available":
        parts.append(answer_coating_question(coating, "", detailed=True))
    return "\n".join(parts) or "이 세션은 완료 분석 결과가 없어 잘한 점이나 고칠 점을 판정할 수 없습니다."


def _selected_sequence(context: str | None) -> str | None:
    """`context`가 시퀀스 ID(`SEQ-012`)면 그것. 구역 번호나 뷰어 문맥은 시퀀스가 아니다."""
    value = (context or "").strip()
    return value if value and not value.isdigit() and ":" not in value and len(value) <= 64 else None


def _pose_evaluation_lookup(prompt: str) -> bool:
    """Reasons already present in a score/correction are measured lookups.

    An evaluation's computed finding differs from asking how scoring works or
    how to perform a movement. Those conceptual/method requests stay with Qwen.
    """
    if is_off_topic(prompt) or NOT_JOINT.search(prompt):
        return False
    if re.search(r"원리|계산|산출|정의|뜻|의미|종류|유형|방법|요령|예방", prompt):
        return False
    pose_subject = BODY_PART.search(prompt) or re.search(r"자세|동작|관절|posture|pose", prompt, re.I)
    reason = re.search(r"왜|이유|근거", prompt)
    score_finding = re.search(r"(?:점수|유사도).{0,24}(?:낮|높|떨어|올라)", prompt)
    produced_finding = re.search(r"지적(?:을\s*)?(?:됐|되었|된|받)", prompt)
    produced_correction = re.search(r"교정\s*결과|(?:나온|받은|표시된).{0,16}교정|교정.{0,16}(?:나왔|나온|표시됐)", prompt)
    return bool(pose_subject and reason and (score_finding or produced_finding or produced_correction))


def _arm_lookup_question(prompt: str) -> bool:
    """Positive gate for an engine measurement, score or computed correction."""
    if not BODY_PART.search(prompt) or NOT_JOINT.search(prompt) or is_off_topic(prompt):
        return False
    if _pose_evaluation_lookup(prompt):
        return True
    if is_explanatory_question(prompt):
        return False
    normalized = re.sub(r"[\s?？.!。]", "", prompt)
    bare = re.fullmatch(r"(?:(?:오른|왼)(?:쪽)?(?:손|팔)|손목|팔꿈치|위팔)(?:은|는|이|가)?", normalized)
    return bool(bare or re.search(r"고쳐|고치|교정|어때|어땠|유사도|점수|몇\s*도(?=\s*(?:차이|정도|야|인가|였|이야|[?？]|$))|얼마.*차이", prompt)
                or re.search(r"(?:면|도|야)\s*(?:돼|되|해)(?:요|나요|니)?[?？.!。\s]*$", prompt)
                or re.search(r"(?:각도|위치)(?:은|는|가|이)?\s*(?:어떻게\s*(?:돼|되|나와)|얼마|몇|어때)?[?？.!。\s]*$", prompt))


def _pose_engine_question(query: CopilotQueryRequest) -> bool:
    """자세 탭의 자세 질문은 대화 계획·모델 분류를 거치지 않고 채점 엔진으로 간다.

    분류가 빗나가면 자유 대화가 채점 없이 수치와 판정을 지어낸다(2026-11-11 재현).
    """
    return (query.task_type == "pose_recomm" and not is_coating_status_question(query.prompt)
            and not is_coating_advice_question(query.prompt) and (not is_general_painting_knowledge(query.prompt) or _arm_lookup_question(query.prompt))
            and (not BODY_PART.search(query.prompt) or _arm_lookup_question(query.prompt))
            and (not is_explanatory_question(query.prompt) or _pose_evaluation_lookup(query.prompt))
            and not is_off_topic(query.prompt) and not NOT_JOINT.search(query.prompt) and not _TAB_OR_WORD.search(query.prompt)
            and bool(BODY_PART.search(query.prompt)  # "왼손은?"(사례 10)
                     or re.search(r"자세(?!히|하게|한)|동작|교정|고쳐|고치|고칠|개선|pose|posture|팔꿈치|어깨|무릎|허리|elbow|shoulder|knee|(?:zone|존|구역)\s*\d+",
                                  query.prompt, re.I)))


def _asks_current_viewer_pose(prompt: str) -> bool:
    # A current playback selection must not change an explicitly global query.
    if re.search(r"전체|전반|종합|평균|모든|최악|가장.*(?:zone|구역)|(?:zone|구역).*가장", prompt, re.I):
        return False
    return bool(re.search(r"지금|현재\s*(?:자세|동작|zone|구역)|이\s*(?:zone|구역)|current\s+(?:pose|zone)|this\s+zone", prompt, re.I))


def _current_viewer_pose_context(prompt: str, context: dict[str, object]) -> dict[str, object]:
    if "playback_context" not in context or not _asks_current_viewer_pose(prompt):
        return context
    playback = context["playback_context"]
    assert isinstance(playback, dict)  # normalized only after verifying the evaluation
    explicit_zone = requested_zone(prompt, None)
    if explicit_zone is not None and explicit_zone != playback.get("zone_id"):
        return context
    zones = context.get("zones")
    key = f"{playback.get('map_num')}:{playback.get('zone_id')}"
    row = zones.get(key) if isinstance(zones, Mapping) else None
    scored = isinstance(row, Mapping) and isinstance(row.get("similarity_percent"), (int, float))
    return {
        "status": "available" if context.get("status") == "available" and scored else "not_available",
        "scope": "current_zone", "playback_context": playback,
        "map_num": playback.get("map_num"), "zone_id": playback.get("zone_id"),
        "similarity_percent": row.get("similarity_percent") if scored else None,
        "zones": {key: row} if scored else {},
        "corrections": (row.get("corrections") or []) if scored else [],
    }


def _viewer_answer(task_type: str, prompt: str, context: Mapping[str, object], intent: str | None) -> str:
    """Build the factual answer from signed measures, never client prose.

    뷰어의 `similarity_percent`는 도구 각도·몸통 기울기·칠하는 방식·도막 두께·이격거리를 묶은 **작업 유사도**다.
    "자세 유사도"는 전문가 자세(팔 관절 엔진)와의 유사도에만 쓴다 — 같은 말로 다른 숫자를 내지 않는다.
    """
    selected_intent = intent if intent and intent != "knowledge" else fallback_intent(task_type, prompt)
    if task_type == "trajectory_recomm":
        flow = context.get("recommended_flow")
        flow = flow if isinstance(flow, list) else []
        if not flow:
            return "현재 재생 기록에서는 전문가 Zone 순서 기준을 계산할 수 없습니다."
        if len(order_request_fields(prompt)) > 1:
            report = dict(context)
            report.update(analysis_status="available", first_visit_flow=context.get("selected_flow") or [])
            return answer_order_question((None, {"sequence_report":report}), prompt=prompt, intent=selected_intent).answer
        if selected_intent == "order_direction":
            direction = context.get("direction_summary")
            classification = direction.get("classification") if isinstance(direction, Mapping) else None
            return f"전체 기록의 주요 도포 방향은 {DIRECTION_LABELS.get(classification, '비정형')}입니다." if classification and classification != "unavailable" else "현재 기록에서는 주요 도포 방향을 계산할 수 없습니다."
        if selected_intent == "order_adherence":
            adherence = _percent_text(context.get("adherence_rate"))
            if adherence is None:
                return "현재 기록의 전문가 순서 유사도를 계산할 수 없습니다."
            answer = f"현재 기록의 전문가 순서 유사도는 {adherence}입니다."
            closest = _percent_text(context.get("best_reference_similarity"))
            positional = _percent_text(context.get("position_similarity"))
            if closest is not None and positional is not None:
                answer += f" 가장 가까운 전문가 수행과 {closest}, 순번별 통계와 {positional} 유사합니다."
            return answer
        # No visit-count or on-track-count is inferred from a blended score.
        report = dict(context)
        report.update({"analysis_status": "available", "first_visit_flow": context.get("selected_flow") or []})
        return answer_order_question((None, {"sequence_report": report}), prompt=prompt, intent=selected_intent).answer

    zones = context.get("zones")
    rows = list(zones.values()) if isinstance(zones, Mapping) else zones if isinstance(zones, list) else []
    rows = [row for row in rows if isinstance(row, Mapping) and isinstance(row.get("similarity_percent"), (int, float))]
    if context.get("scope") == "current_zone":
        if context.get("zone_id") is None:
            return "현재 재생 위치는 Zone ID가 없는 구간이어서 Zone 기준 작업 비교값을 표시할 수 없습니다."
        if context.get("status") != "available" or not rows:
            return "질문 시점의 Zone 작업 비교값이 없습니다. 재생 위치를 확인한 뒤 다시 질문해 주세요."
        zone = rows[0]
        corrections = zone.get("corrections")
        guidance = " ".join(item for item in corrections if isinstance(item, str)) if isinstance(corrections, list) else ""
        if re.search(r"오른|right|왼|left", prompt, re.I):
            side = "오른" if re.search(r"오른|right", prompt, re.I) else "왼"
            matching = [item for item in corrections if isinstance(item, str) and side in item] if isinstance(corrections, list) else []
            guidance = " ".join(matching) if matching else f"이 Zone에는 {side}손에 특정된 교정 항목이 없습니다."
        return (f"현재 재생 구간(Map {context.get('map_num')} · Zone {zone.get('zone_id')})의 작업 유사도는 "
                f"{_percent_text(zone.get('similarity_percent'), already_percent=True)}입니다. "
                + (guidance or "별도의 작업 개선 항목은 없습니다."))
    if context.get("status") != "available" or not rows:
        return "현재 재생 기록에서는 작업 비교값을 계산할 수 없습니다."
    target = requested_zone(prompt, None)
    if target is not None:
        zone = next((row for row in rows if str(row.get("zone_id")) == target), None)
        if zone is None:
            return f"현재 재생 기록에는 Zone {target}의 작업 비교값이 없습니다."
        corrections = zone.get("corrections")
        guidance = " ".join(str(item) for item in corrections if isinstance(item, str)) if isinstance(corrections, list) else ""
        return f"Zone {target} 작업 유사도는 {_percent_text(zone.get('similarity_percent'), already_percent=True)}입니다." + (f" {guidance}" if guidance else " 별도의 작업 개선 항목은 없습니다.")
    if selected_intent in {"pose_right", "pose_left"}:
        side = "오른손" if selected_intent == "pose_right" else "왼손"
        corrections = context.get("corrections")
        messages = [str(item.get("message") or item.get("label") or "") if isinstance(item, Mapping) else str(item)
                    for item in corrections] if isinstance(corrections, list) else []
        matching = [message for message in messages if side in message or ("오른" if side == "오른손" else "왼") in message]
        return f"{side} 개선 안내: {' '.join(matching)}" if matching else f"현재 기록에서 {side}에 특정된 개선 항목은 없습니다. 전체 작업 유사도는 {_percent_text(context.get('similarity_percent'), already_percent=True)}입니다."
    worst = min(rows, key=lambda row: float(row["similarity_percent"]))
    # 부분 문자열("낮")이면 "팔꿈치를 낮춰야 해?"도 최저 Zone 답이 된다.
    if selected_intent == "pose_worst_zone" or WORST_ZONE.search(prompt) or re.search(r"(?:가장|제일)\s*낮", prompt):
        corrections = worst.get("corrections")
        guidance = " ".join(str(item) for item in corrections if isinstance(item, str)) if isinstance(corrections, list) else ""
        return f"현재 기록에서 작업 유사도가 가장 낮은 구역은 Zone {worst.get('zone_id')}({_percent_text(worst.get('similarity_percent'), already_percent=True)})입니다." + (f" {guidance}" if guidance else "")
    return (f"현재 재생 기록의 전체 작업 유사도는 {_percent_text(context.get('similarity_percent'), already_percent=True)}입니다. "
            f"가장 낮은 구역은 Zone {worst.get('zone_id')}({_percent_text(worst.get('similarity_percent'), already_percent=True)})입니다.")


def _percent_text(value: object, *, already_percent: bool = False) -> str | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if 0 <= number <= 1 and not already_percent:
        number *= 100
    return f"{round(number)}%"


def _flow_text(value: object) -> str | None:
    if not isinstance(value, list) or not value:
        return None
    return " → ".join(str(item) for item in value)


def _stable_order_prefix(
    recommended_value: object, distribution_value: object, *, threshold: float = 0.6
) -> tuple[list[str], float | None]:
    if not isinstance(recommended_value, list) or not isinstance(distribution_value, list):
        return [], None
    stable: list[str] = []
    probabilities: list[float] = []
    for expected, row in zip(recommended_value, distribution_value):
        if not isinstance(row, list) or not row or not isinstance(row[0], dict):
            break
        top = row[0]
        try:
            probability = float(top.get("probability", 0))
        except (TypeError, ValueError):
            break
        if str(top.get("zone_id")) != str(expected) or probability < threshold:
            break
        stable.append(str(expected))
        probabilities.append(probability)
    return stable, min(probabilities) if probabilities else None


def _structured_detail_fallback(
    task_type: str,
    base_answer: str,
    analysis_context: dict[str, object],
    *,
    intent: str | None = None,
) -> str:
    """검증된 수치만 덧붙인 문장으로 답한다 — 제목 틀과 고정 문장은 쓰지 않는다(2026-11-11 지시)."""
    if task_type == "trajectory_recomm" and intent not in (None, "order_reference"):
        evidence = ""
        if intent == "order_missing":
            expected = {str(z) for z in analysis_context.get("recommended_flow", [])}
            visited = {str(z) for z in analysis_context.get("selected_flow", analysis_context.get("first_visit_flow", []))}
            if expected:
                evidence = f"\n전문가 기준 {len(expected)}개 Zone 중 방문 {len(expected & visited)}개, 누락 {len(expected - visited)}개입니다. Zone 방문 여부와 Zone 내부 도막 충족 여부는 별도로 평가합니다."
        elif intent == "order_adherence":
            evidence = "\n전문가 작업 흐름과 순번별 분포를 비교한 순서 유사도입니다. 도막 두께와 자세 평가는 포함하지 않습니다."
        return base_answer + evidence
    if task_type == "trajectory_recomm":
        recommended_value = analysis_context.get("recommended_flow")
        recommended = _flow_text(recommended_value)
        if not recommended:
            return base_answer
        selected = _flow_text(
            analysis_context.get("selected_flow") or analysis_context.get("first_visit_flow")
        )
        stable_prefix, stable_confidence = _stable_order_prefix(
            recommended_value, analysis_context.get("position_distribution")
        )
        adherence_value = analysis_context.get("adherence_rate")
        adherence = _percent_text(adherence_value)
        try:
            adherence_number = float(adherence_value)  # type: ignore[arg-type]
            if adherence_number <= 1:
                adherence_number *= 100
        except (TypeError, ValueError):
            adherence_number = -1
        tendencies: list[str] = []
        if len(stable_prefix) >= 2 and stable_confidence is not None:
            tendencies.append(
                f"전문가 데이터에서는 Zone {' → '.join(stable_prefix)} 구간이 가장 뚜렷했습니다. "
                f"전문가 작업의 약 {stable_confidence * 100:.0f}%가 이 시작 흐름을 따랐습니다."
            )
        elif stable_prefix and stable_confidence is not None:
            tendencies.append(
                f"Zone {stable_prefix[0]}에서 시작하는 경향이 약 {stable_confidence * 100:.0f}%로 가장 높았습니다."
            )
        if isinstance(recommended_value, list) and len(stable_prefix) < len(recommended_value):
            start_at = len(stable_prefix) + 1
            tendencies.append(
                f"{start_at}번째 Zone 이후에는 작업 회차에 따라 순서 차이가 나타납니다. "
                "이 구간은 하나의 순서만 정답으로 판단하지 않고 대표 경로와의 연결성을 함께 평가합니다."
            )
        # The deterministic sections below already contain every verified
        # order statistic.  An LLM paraphrase here often repeats the quick
        # answer verbatim, making detailed mode feel like quick mode with an
        # extra paragraph.  Keep trajectory detail concise and non-duplicated.

        comparison: list[str] = []
        if selected:
            comparison.append(f"실제 수행: {selected}")
        if adherence:
            comparison.append(f"순서 준수율은 {adherence}입니다.")
        metrics = (
            ("가장 가까운 전문가 수행과의 유사도", "는", "best_reference_similarity"),
            ("순번별 통계 유사도", "는", "position_similarity"),
        )
        for label, particle, key in metrics:
            value = _percent_text(analysis_context.get(key))
            if value is not None:
                comparison.append(f"{label}{particle} {value}입니다.")
        direction = analysis_context.get("direction_summary")
        if isinstance(direction, dict) and direction.get("classification"):
            direction_label = DIRECTION_LABELS.get(str(direction["classification"]), str(direction["classification"]))
            comparison.append(f"주요 도포 방향은 {direction_label}입니다.")
        segment_lines: list[str] = []
        if stable_prefix:
            confidence_text = f" ({stable_confidence * 100:.0f}%)" if stable_confidence is not None else ""
            segment_lines.append(f"- 우선 적용 구간: Zone {' → '.join(stable_prefix)}{confidence_text}")
        if isinstance(recommended_value, list) and len(stable_prefix) < len(recommended_value):
            flexible = " → ".join(str(zone) for zone in recommended_value[len(stable_prefix):])
            segment_lines.append(f"- 유동 구간: Zone {flexible}\n  작업 회차별 차이를 허용하는 대표 경로입니다.")
        if comparison:
            segment_lines.append("- 현재 기록 비교\n  " + "\n  ".join(comparison))
        # 자료에서 나온 안내만 쓴다 — 어느 기록에나 붙는 일반 권고 문장은 넣지 않는다.
        action = (
            ["실제 수행 순서가 권장 순서와 같습니다."] if selected and recommended == selected
            else [f"Zone {stable_prefix[0]}에서 시작한 뒤 Zone {stable_prefix[1]}로 이동하세요."] if len(stable_prefix) >= 2
            else []
        )
        return "\n".join([f"전문가 기준 권장 순서는 Zone {recommended}입니다.", *tendencies, *segment_lines, *action])
    elif task_type == "pose_recomm" and analysis_context.get("scope") != "current_zone":
        zones = analysis_context.get("zones")
        zone_rows = list(zones.values()) if isinstance(zones, dict) else zones
        scored = [
            (float(row["similarity_percent"]), row)
            for row in (zone_rows if isinstance(zone_rows, list) else [])
            if isinstance(row, dict) and isinstance(row.get("similarity_percent"), (int, float))
        ]
        if scored:
            _, worst = min(scored, key=lambda item: item[0])
            score = _percent_text(worst.get("similarity_percent"), already_percent=True)
            worst_text = f"가장 낮은 구역은 Zone {worst.get('zone_id')}({score})입니다."
            if worst_text not in base_answer:
                return base_answer + "\n" + worst_text
    return base_answer


def _trim_dangling_detail_label(candidate: str) -> str:
    return re.sub(r"(?:\n|\s)*(?:근거|분석|상세)\s*:\s*$", "", candidate).strip()


def _detail_interpretation(base_answer: str, candidate: str) -> str | None:
    """Keep useful LLM interpretation, but reject a copied quick answer."""
    if candidate.startswith(base_answer.strip()):
        candidate = candidate[len(base_answer.strip()):].strip()
        # A response built by appending statistics to the quick answer is exactly
        # what detailed mode must avoid; the structured formatter handles it better.
        return None
    return candidate if len(candidate) >= 30 else None


def _grounded_detail(candidate: str, base_answer: str, context: Mapping[str, object]) -> bool:
    """Reject novel numbers and reordered Zone paths before prose reaches clients."""
    allowed = set(re.findall(r"\d+(?:\.\d+)?", base_answer))
    flows: list[list[str]] = []
    recommended: list[list[str]] = []

    def collect(value: object, key: str = "") -> None:
        if isinstance(value, Mapping):
            for child_key, child in value.items():
                collect(child, str(child_key))
        elif isinstance(value, list):
            if key.endswith("flow") or key == "recommended_flow":
                flows.append([str(item) for item in value])
                if key == "recommended_flow":
                    recommended.append([str(item) for item in value])
            for child in value:
                collect(child, key)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            allowed.add(str(value))
            allowed.add(f"{value:.0f}")
            if 0 <= value <= 1:
                allowed.add(f"{value * 100:.0f}")
        elif isinstance(value, str):
            allowed.update(re.findall(r"\d+(?:\.\d+)?", value))

    collect(context)
    if any(number not in allowed for number in re.findall(r"\d+(?:\.\d+)?", candidate)):
        return False
    for match in re.finditer(r"\d+(?:\s*(?:→|->)\s*\d+)+", candidate):
        sequence = re.findall(r"\d+", match.group())
        prefix = candidate[max(0, match.start() - 40):match.start()]
        recommendation_at = max(prefix.rfind(word) for word in ("권장", "추천", "대표"))
        actual_at = max(prefix.rfind(word) for word in ("실제", "수행", "현재"))
        allowed_flows = recommended if recommended and recommendation_at > actual_at else flows
        if not any(any(flow[start:start + len(sequence)] == sequence
                       for start in range(len(flow))) for flow in allowed_flows):
            return False
    return True


_HERE: Final = re.compile(r"지금|현재|여기|이렇게|이번|방금|아까|(?<![가-힣])이\s*(?:zone|존|구역|기록|분사|곳)|내\s|제\s|나의|저의", re.I)


_HEADER_LIMIT: Final = 1024
_TRUNCATED: Final = "…"


def analysis_prompt_header(text: str) -> str:
    """퍼센트 인코딩한 분석 질문 헤더 값. 한글은 글자당 9바이트라 nginx 기본 헤더 버퍼를 넘기지 않게 1024바이트로 자른다.

    코드 포인트 경계에서 자르고 끝에 "…"를 붙인다. 전체 문장은 로그에 남는다.
    """
    encoded = quote(text, safe="")
    if len(encoded) <= _HEADER_LIMIT:
        return encoded
    budget = _HEADER_LIMIT - len(quote(_TRUNCATED, safe=""))
    low, high = 0, len(text)
    while low < high:  # quote(text[:n])가 budget 안인 가장 큰 n
        middle = (low + high + 1) // 2
        low, high = (middle, high) if len(quote(text[:middle], safe="")) <= budget else (low, middle - 1)
    return quote(text[:low] + _TRUNCATED, safe="")


def _viewer_measures(context: str | None, prompt: str) -> tuple[dict[str, dict[str, float]] | None, str | None]:
    """서명된 뷰어 Zone 실측(속도·이격거리·전문가 중앙값)과 질문이 가리키는 Zone(번호, 없으면 재생 위치)."""
    claims = viewer_claims(context)
    if claims is None:
        return None, requested_zone(prompt, None)
    playback = claims[0].get("playback_context") if isinstance(claims[0].get("playback_context"), dict) else {}
    # 재생 위치의 Zone은 질문이 "지금·이 구역·이렇게"처럼 그곳을 가리킬 때만 쓴다. 일반 질문은 특정 Zone이 아니다.
    here = playback.get("zone_id") is not None and bool(_HERE.search(prompt))
    return zone_measures(claims[1]), requested_zone(prompt, None) or (str(playback["zone_id"]) if here else None)


_NO_ANSWER: Final = "제공된 자료로는 답할 수 없습니다."


def _checked_generation(answer: str, query: CopilotQueryRequest, *, instructions: tuple[str, ...] = (),
                        evidence: tuple[str, ...] | None = None, drop_only: bool = False, safety: bool = False) -> str:
    """모델이 만든 답의 출구 검사: 지침·내부 키 유출, 근거 없는 수치, 다른 Zone의 수치, 거꾸로 된 인과·조언, 이 기록의 최저·유무·판정·기준값.

    `evidence`(검색 조각 본문)를 주는 문서 경로만 수치 근거 검사(G1)를 한다. 대화 경로는 세션 요약을 따로 대조한다.
    `drop_only`(섞인 질문의 문서 문장)는 바꾸지 않고 빼며, 다 빠지면 빈 문자열이다(기록 문장은 호출부가 붙인다).
    """
    measures, zone = _viewer_measures(query.context, query.prompt)
    answer = guard_generated_answer(answer, instructions=instructions)
    had_body = has_body(answer)
    # 인과·조언 검사는 이 기록(서명된 뷰어 문맥)에 대한 답에만 한다. 일반 지식 답은 그대로 둔다.
    record = RecordFacts.from_context(query.context) if measures is not None else None
    if evidence is not None and (record is not None or parse_viewer_arm_pose(query.context) is not None):
        pool = _grounding(query, evidence)
        if ungrounded_measurements(answer, facts=pool, prompt=query.prompt, extended=True):
            answer = drop_ungrounded(answer, facts=pool, prompt=query.prompt, extended=True)
    answer = drop_misattributed_zones(answer, _zone_facts(query.context))
    if measures is None:
        return answer
    answer = drop_contradicted_causes(answer, measures, default_zone=zone, record_zone=_playback_zone(query.context))
    if record is not None and not safety:
        answer = ground_generated(answer, record, zone=zone, drop_only=drop_only, marked=has_record_marker(query.prompt),
                                  zone_asked=requested_zone(query.prompt, None) is not None)
    if had_body and (not has_body(answer) or answer.strip() in {_NO_ANSWER, "세션 기록으로는 확인할 수 없습니다."}):
        if drop_only or safety:  # 위험 질문은 빈 답을 guard_answer에 넘겨 그 종류의 고정 문장을 받는다(기록 답·"답할 수 없습니다" 금지)
            return ""
        found = answer_record_question(query.context, query.prompt, task_type=query.task_type, recent_turns=())
        return found.text if found is not None and found.kind != "mixed" else _NO_ANSWER
    return answer


def _grounding(query: CopilotQueryRequest, evidence: tuple[str, ...]) -> dict[str, Any]:
    """문서 답의 수치가 기댈 수 있는 값: 서명된 대화 자료, 도막·팔 자세 문맥, 검색 조각 본문. 대화 기록은 넣지 않는다."""
    return {"facts": authenticated_facts(query.context), "coating": parse_viewer_coating_context(query.context, "knowledge_search"),
            "arm": parse_viewer_arm_pose(query.context), "evidence": list(evidence)}


def _playback_zone(context: str | None) -> str | None:
    claims = viewer_claims(context)
    playback = claims[0].get("playback_context") if claims and isinstance(claims[0].get("playback_context"), dict) else {}
    zone = playback.get("zone_id")
    return str(zone) if zone is not None and str(zone) not in {"", "-1", "0"} else None


def _zone_facts(context: str | None) -> dict[str, set[str]]:
    """모델 답의 Zone 번호 검사에 쓰는 서명된 Zone 사실. 뷰어 문맥이 없으면 비어 있다(검사하지 않는다)."""
    claims = viewer_claims(context)
    return zone_fact_numbers(claims[1] if claims else None, parse_viewer_arm_pose(context)) if claims else {}


def _document_rules(pipeline: object) -> tuple[str, ...]:
    """문서 답 지침(fast_system) 가운데 유출 검사에 쓸 줄. 저도막·과도막 정의 줄은 답이 그대로 옮겨도 정상이라 뺀다."""
    config = getattr(pipeline, "config", None)
    prompt = config.get("prompt") if isinstance(config, dict) else None
    text = str(prompt.get("fast_system", "")) if isinstance(prompt, dict) else ""
    return ("\n".join(line for line in text.splitlines() if "저도막" not in line and "과도막" not in line),)


def error_response(code: int, message: str) -> JSONResponse:
    envelope = CopilotResponse(status="ERROR", code=code, message=message, data=None)
    return JSONResponse(status_code=code, content=envelope.model_dump(mode="json"))


def _header_subject(response: Any) -> dict[str, Any] | None:
    try:
        value = json.loads(response.headers.get(SUBJECT_HEADER) or "null")
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def _with_answer_trace(response: JSONResponse, result: Mapping[str, Any]) -> JSONResponse:
    """Carry a bounded diagnostic to persistence, never to the public envelope.

    Only the pipeline's explicit trace schema is accepted. Whole prompts,
    context, model configuration and arbitrary metadata are not copied.
    """
    metadata = result.get("metadata")
    trace = metadata.get("answer_trace") if isinstance(metadata, Mapping) else None
    if not isinstance(trace, Mapping):
        return response
    clean: dict[str, Any] = {}
    for key, limit in (("policy_version", 80), ("question", 512), ("retrieval_query", 512),
                       ("initial_answer", 512), ("final_answer", 512), ("fallback_reason", 80)):
        if isinstance(trace.get(key), str):
            clean[key] = trace[key][:limit]
    count = trace.get("retrieved_candidates")
    if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
        clean["retrieved_candidates"] = min(count, 2147483647)
    evidence = trace.get("selected_evidence")
    if isinstance(evidence, list):
        clean["selected_evidence"] = []
        for item in evidence[:8]:
            if not isinstance(item, Mapping):
                continue
            row: dict[str, Any] = {key: item[key][:limit] for key, limit in (
                ("chunk_id", 160), ("sha256", 64), ("excerpt", 600)) if isinstance(item.get(key), str)}
            for key in ("characters", "rerank_score"):
                number = item.get(key)
                if isinstance(number, (int, float)) and not isinstance(number, bool):
                    try:
                        if math.isfinite(number):
                            row[key] = number
                    except OverflowError:
                        pass
            clean["selected_evidence"].append(row)
    review = trace.get("review")
    if isinstance(review, Mapping):
        clean["review"] = {}
        if isinstance(review.get("verdict"), str):
            clean["review"]["verdict"] = review["verdict"][:80]
        if isinstance(review.get("reason"), str):
            clean["review"]["reason"] = review["reason"][:160]
        for key in ("retries", "calls", "latency_ms"):
            number = review.get(key)
            if isinstance(number, int) and not isinstance(number, bool) and number >= 0:
                clean["review"][key] = min(number, 2147483647)
        if isinstance(review.get("cacheable"), bool):
            clean["review"]["cacheable"] = review["cacheable"]
        events=review.get("events")
        if isinstance(events,list):
            clean["review"]["events"]=[]
            for event in events[:2]:
                if not isinstance(event,Mapping):continue
                entry={key:event[key][:160] for key in ("verdict","reason") if isinstance(event.get(key),str)}
                for key in ("support_valid","within_budget"):
                    if isinstance(event.get(key),bool):entry[key]=event[key]
                if isinstance(event.get("elapsed_ms"),int) and not isinstance(event["elapsed_ms"],bool):
                    entry["elapsed_ms"]=max(0,min(event["elapsed_ms"],2147483647))
                if isinstance(event.get("support"),list):
                    entry["support"]=[value[:160] for value in event["support"][:8] if isinstance(value,str)]
                clean["review"]["events"].append(entry)
        contexts=review.get("contexts")
        if isinstance(contexts,list):
            clean["review"]["contexts"]=[{key:item[key] for key in ("prompt_tokens","max_new_tokens","context_limit")
                if isinstance(item.get(key),int) and not isinstance(item[key],bool) and 0<=item[key]<=2147483647}
                for item in contexts[:2] if isinstance(item,Mapping)]
    if len(json.dumps(clean, ensure_ascii=False).encode("utf-8")) > 16384:
        for item in clean.get("selected_evidence", []):
            item["excerpt"] = item.get("excerpt", "")[:200]
        clean["truncated"] = True
    if len(json.dumps(clean, ensure_ascii=False).encode("utf-8")) > 16384:
        for key in ("question", "retrieval_query", "initial_answer", "final_answer"):
            if key in clean:
                clean[key] = clean[key][:256]
    if len(json.dumps(clean, ensure_ascii=False).encode("utf-8")) > 16384:
        # Eight visible sources and bounded review events can still exceed the
        # byte ceiling with four-byte Unicode. Keep IDs/events, shrink text.
        def byte_prefix(value):
            if isinstance(value,str):return value.encode("utf-8")[:160].decode("utf-8",errors="ignore")
            if isinstance(value,list):return [byte_prefix(item) for item in value]
            if isinstance(value,dict):return {key:byte_prefix(item) for key,item in value.items()}
            return value
        clean=byte_prefix(clean)
        clean["truncated"]=True
    if clean:
        response._copilot_answer_trace = clean
    return response


def _success_response(
    query: CopilotQueryRequest,
    started: float,
    *,
    answer: str,
    retrieval_used: bool,
    citations: list[str],
    degraded: bool = False,
    pose_context: dict[str, object] | None = None,
    sequence_context: dict[str, object] | None = None,
    coating_context: dict[str, object] | None = None,
    context_status: str = "not_available",
    generated: bool = False,
    model_name: str | None = None,
    subject: Mapping[str, Any] | None = None,
) -> JSONResponse:
    """`generated`는 모델이 문장을 만든 답에만 준다. 나머지는 채점·분석 결과를 그대로 옮긴 답이다.

    답을 만든 곳은 응답 헤더로만 알린다 — 본문 키를 엄격히 검사하는 기존 클라이언트가 있어 본문은 바꾸지 않는다.
    화면이 요청한 답변 수준만 보고 모델 이름을 붙이면 채점 결과에도 모델 이름이 붙는다(2026-11-11 뷰어 점검).
    """
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    # Wall clock from handler entry to just before serialization, slot wait included.
    latency_ms = int((time.monotonic() - started) * 1000)
    envelope = CopilotResponse(
        status="SUCCESS",
        code=200,
        message=_SUCCESS_MESSAGE,
        data=CopilotQueryData(
            request_id=query.request_id,
            session_id=query.session_id,
            user_id=query.user_id,
            scenario_id=query.scenario_id,
            generated_at=generated_at,
            latency_ms=latency_ms,
            degraded=degraded,
            answer=answer,
            resolved_intents=[query.task_type],
            context_status=context_status,  # type: ignore[arg-type]
            sequence_context=sequence_context,
            coating_context=coating_context,
            pose_context=pose_context,
            retrieval_used=retrieval_used,
            citations=citations,
        ),
    )
    logger.info(
        "copilot query done request_id=%s task_type=%s latency_ms=%d retrieval=%s citations=%d",
        query.request_id,
        query.task_type,
        latency_ms,
        retrieval_used,
        len(citations),
    )
    source = "documents" if retrieval_used else "conversation" if generated and not degraded else "analysis"
    headers = {ANSWER_SOURCE_HEADER: source}
    if generated and (name := safe_model_name(model_name)) is not None:
        headers[MODEL_HEADER] = name
    if subject:
        headers[SUBJECT_HEADER] = json.dumps(subject, ensure_ascii=True, separators=(",", ":"))
    response = JSONResponse(status_code=200, content=envelope.model_dump(mode="json"), headers=headers)
    # Engine/record builders already select the requested facts and required
    # limits/actions. A generic prose extractor must not reinterpret that core.
    response._copilot_precomposed_answer = not generated or bool(subject)
    return response


async def _execute_copilot_query(
    query: CopilotQueryRequest,
    *,
    pipeline_provider: PipelineProvider | None,
    pending_answers: Mapping[str, str] = PENDING_TASK_ANSWERS,
    sequences: SequenceStore,
    mcp_base_url: str,
    transport: object | None = None,
    started: float | None = None,
    reports: V1Store | None = None,
    sequence_engine: SequenceAnalysisEngine | None = None,
    coating_engine: object | None = None,
    conversation_id: str | None = None,
    original_prompt: str | None = None,
    safety_class: str | None = None,
) -> JSONResponse:
    """Execute one normalized query for both text and voice transports.

    `original_prompt`는 외국어 질문의 원문이다(`query.prompt`는 한국어로 옮긴 질문).
    `safety_class`는 바깥(execute_copilot_query)이 두 질문과 앞 턴의 저장된 종류로 고른 위험 종류다(safety_guard).
    """
    request_started = time.monotonic() if started is None else started
    if asks_for_instructions(query.prompt):
        # 지침 공개·무시 요청은 모델에 보내지 않는다(2026-11-14 운영 q41).
        return _success_response(query, request_started, answer=INSTRUCTION_REFUSAL, retrieval_used=False, citations=[])
    # A tab suggests a subject, not the user's intent. Definitions and types
    # must be explained even when a pose/order tab supplies a signed record.
    arm_lookup = _arm_lookup_question(query.prompt)
    evaluation_lookup = _pose_evaluation_lookup(query.prompt)
    general_knowledge = is_general_painting_knowledge(query.prompt) and not arm_lookup and not evaluation_lookup
    explanatory_request = is_explanatory_question(query.prompt, query.task_type) and not evaluation_lookup
    if query.task_type != "error_analysis" and (general_knowledge or explanatory_request):
        query = query.model_copy(update={"task_type": "knowledge_search"})
    questions = (query.prompt, original_prompt or "")

    dialogue_turns = await to_thread.run_sync(lambda: sequences.recent_query_turns(
        query.session_id or "CHAT:" + query.user_id, limit=6,
        user_id=query.user_id, conversation_key=conversation_id,
    ))
    # 서명된 뷰어 기록이 있을 때만 켠다(사용자 결정 7). 세션 요약으로 답하는 기존 대화 경로(TSB·시퀀스 질문)는 그대로 둔다.
    # 서명된 기록이 없어도 도장·훈련 낱말이 든 지식 질문은 문서로 답한다(사용자 결정 4). 잡담·TSB 요약 대화는 그대로.
    grounded = _grounded_route(query.prompt, previous_subject(dialogue_turns), viewer=viewer_claims(query.context) is not None)
    # 이전 생성 답의 문구는 반박·조건 추가를 이해하는 데 필요하다. 대화로만
    # 표시해 전달하고, 측정 사실·기록 subject·숫자 검증 자료에는 넣지 않는다.
    model_turns = dialogue_turns_for_model(dialogue_turns)
    # 위험 행동을 해도 되는지 묻는 질문은 대화 경로·측정값 짧은 답·뷰어 엔진 답으로 보내지 않고 문서 검색으로 답한 뒤
    # 답을 검사한다(2026-11-14 훈련생 질문 시험 T15). "그럼 해도 돼?"는 같은 대화의 앞 질문으로 본다.
    if safety_class == "hand_tip":
        # 손이 건 끝에 닿는 질문은 경로·탭·답변 수준과 상관없이 모델 없이 고정 답이다(v3.4.1 운영: 베트남어 답이 막으라고 했다).
        return _success_response(query, request_started, answer=HIGH_PRESSURE_ANSWER, retrieval_used=False,
                                 citations=list(HIGH_PRESSURE_CITATIONS), context_status="available")
    # 위험 주제 질문은 대화 경로·측정값 짧은 답·뷰어 엔진 답으로 보내지 않고 문서 검색으로 답한 뒤 답을 검사한다.
    safety = safety_class is not None and query.task_type != "error_analysis"
    if safety and query.task_type != "knowledge_search":
        query = query.model_copy(update={"task_type": "knowledge_search"})
    facts = authenticated_facts(query.context)
    try:
        # Pose/order playback validation applies only when consuming that
        # signed evaluation. Coating has its own independently signed claims.
        viewer_analysis = (_parse_motion_viewer_context(query.context, task_type=query.task_type)
                           if query.task_type in ("pose_recomm", "trajectory_recomm") else None)
    except _InvalidViewerPlayback:
        return error_response(422, "Invalid motion viewer playback context")
    relevant_viewer = viewer_analysis is not None and query.task_type in ("pose_recomm", "trajectory_recomm")
    # 서명된 뷰어 기록을 묻는 질문은 계획 모델 앞에서 기록 행으로만 답한다(record_query). 위험 질문은 위에서 갈렸고 여기 오지 않는다.
    record = (answer_record_question(query.context, query.prompt, task_type=query.task_type, recent_turns=dialogue_turns)
              if safety_class is None else None)
    if record is not None and record.kind != "mixed":
        return _success_response(query, request_started, answer=record.text, retrieval_used=False, citations=[],
                                 sequence_context=record.sequence_context, context_status="available", subject=record.subject)
    # 기록 표지가 없는 누락·보완·다음 Zone 질문("누락한 Zone은?")은 지금처럼 서명된 방문 순서와 최종 도막 게이트로 답한다.
    progress_kind = progress_question(query.prompt) if record is None and safety_class is None else None
    progress = viewer_progress_answer(query.context, query.prompt, progress_kind) if progress_kind else None
    if progress is not None:
        return _success_response(query, request_started, answer=progress[0], retrieval_used=False, citations=[],
                                 sequence_context=progress[1], context_status="available")
    # 섞인 질문(기록 조각 + 지식 조각)은 기록 문장을 먼저 두고, 지식 조각만 문서로 답한다(안전 질문과 같은 문서 경로).
    mixed = record if record is not None and record.kind == "mixed" else None
    if mixed is not None and query.task_type != "knowledge_search":
        query = query.model_copy(update={"task_type": "knowledge_search"})
    documents_only = safety or mixed is not None

    def record_only(found: RecordAnswer) -> JSONResponse:
        # 모델 서버가 없거나 생성이 실패해도 섞인 질문의 기록 쪽 답은 낸다.
        return _success_response(query, request_started, answer=found.text, retrieval_used=False, citations=[],
                                 sequence_context=found.sequence_context, context_status="available", subject=found.subject)

    if is_current_next_question(query.prompt) and not documents_only and not explanatory_request:
        answer, scope, progress = await to_thread.run_sync(lambda: current_next_answer(
            query, sequences, reports, sequence_engine=sequence_engine, coating_engine=coating_engine,
        ))
        return _success_response(query, request_started, answer=answer, retrieval_used=False,
            citations=[], sequence_context=scope, coating_context=progress,
            context_status="available" if scope["analysis_status"] == "available" else "not_available")
    has_reference_plan = callable(getattr(sequence_engine, "reference_plan", None))
    explicit_actual_order = actual_order_intent(query.prompt)
    async def reference_answer():
        scenario, process = await to_thread.run_sync(
            lambda: resolve_reference_target(query, sequences, reports, facts=facts)
        )
        reference_query = query.model_copy(update={"scenario_id": scenario})
        answer, scope = await to_thread.run_sync(lambda: reference_order_answer(
            sequence_engine, scenario_id=scenario, process=process,
            detailed=True,
        ))
        return _success_response(
            reference_query, request_started, answer=answer, retrieval_used=False,
            citations=[], sequence_context=scope,
            context_status="available" if scope["analysis_status"] == "available" else "not_available",
        )
    # A full reference plan is available before any trainee sequence or complete
    # report. It never authorizes realtime movement or supplies actual visits.
    if (has_reference_plan and not relevant_viewer and query.task_type != "error_analysis" and not documents_only
            and is_reference_order_question(query.prompt) and not explanatory_request):
        return await reference_answer()
    frame_direction = playback_direction_from_context(query.context)
    viewer_requested = bool(query.context and query.context.startswith(("motion_viewer_analysis:", "motion_viewer_playback:")))
    def frame_answer():
        if requested_direction_scope(query.prompt) != "frame":
            answer, scope = aggregate_direction(query.context, query.prompt)
            return _success_response(query, request_started, answer=answer, retrieval_used=False,
                                     citations=[], sequence_context=scope,
                                     context_status="available" if scope.get("analysis_status") == "available" else "not_available")
        available = frame_direction is not None and frame_direction.get("analysis_status") == "available"
        return _success_response(query, request_started, answer=playback_direction_answer(frame_direction),
                                 retrieval_used=False, citations=[], sequence_context=frame_direction or {"scope": "current_frame", "analysis_status": "unavailable"},
                                 context_status="available" if available else "not_available")
    if viewer_requested and not documents_only and (simple_direction_question(query.prompt) or preset_intent("trajectory_recomm", query.prompt) == "order_direction"):
        return frame_answer()
    arm_claims = parse_viewer_arm_pose(query.context) if viewer_requested and not documents_only else None
    coating_question = is_coating_status_question(query.prompt) or is_coating_advice_question(query.prompt)
    if arm_claims is not None and arm_lookup and not coating_question:
        # 손·손목·팔꿈치 질문은 어느 탭이든 팔 관절 엔진 결과로만 답한다(운영 q4·q5·q6: 작업 안내·모델의 "정상 범위" 답).
        claims = viewer_claims(query.context)
        playback = (viewer_analysis or {}).get("playback_context") or (claims[0].get("playback_context") if claims else None)
        answer = arm_pose_answer(
            arm_claims, current=_asks_current_viewer_pose(query.prompt), playback=playback if isinstance(playback, dict) else None,
            zone_id=requested_zone(query.prompt, None), side=asked_arm(query.prompt), part=asked_part(query.prompt),
            prefer_hand=prefers_hand(query.prompt), prompt=query.prompt)
        answer = angle_note(query.prompt) + answer
        return _success_response(query, request_started, answer=answer, retrieval_used=False, citations=[],
                                 pose_context={"scope": "arm_pose", "status": arm_claims.get("status")} if query.task_type == "pose_recomm" else None,
                                 context_status="available" if arm_claims.get("status") == "available" else "not_available")
    if (arm_claims is not None and viewer_analysis is not None and query.task_type == "pose_recomm"
            and (preset_intent(query.task_type, query.prompt) == "pose_worst_zone" or WORST_ZONE.search(query.prompt))):
        # "Zone별로 어디가 문제야?"는 자세 유사도가 낮은 Zone과 그 교정으로 답한다(운영 q2, 27B Zone 오기).
        zones = viewer_analysis.get("zones")
        answer = zone_pose_summary(arm_claims, list(zones.values()) if isinstance(zones, Mapping) else None, prompt=query.prompt)
        return _success_response(query, request_started, answer=answer, retrieval_used=False, citations=[],
                                 pose_context={"scope": "arm_pose_zones", "status": arm_claims.get("status")},
                                 context_status="available" if arm_claims.get("status") == "available" else "not_available")
    if viewer_requested and not documents_only and not explanatory_request and preset_intent(query.task_type, query.prompt) is None:
        # 손·팔 질문은 소견 틀이 아니라 팔 자세 틀이 답한다(사례 4). 자세 탭에서 팔 채점이 있으면 "뭐부터 고쳐"는 팔 자세 우선 교정이다(T22).
        # 자세 탭에서 팔 채점 없이 자세를 물으면 아래 채점 없음 안내가 답한다(I7).
        first_fix = bool(FIRST_FIX.search(query.prompt))
        pose_tab = query.task_type == "pose_recomm"
        arm_priority = pose_tab and first_fix and arm_claims is not None
        no_arm_pose = pose_tab and arm_claims is None and bool(_POSE_ASKED.search(query.prompt))
        findings = ((bool(_EVALUATION_QUESTION.search(query.prompt)) or first_fix) and not BODY_PART.search(query.prompt)
                    and not arm_priority and not no_arm_pose)
        signed_answer = viewer_measurement_answer(query.context, query.prompt) or (
            viewer_findings_answer(query.context) if findings else None)
        if signed_answer is not None:
            return _success_response(query, request_started, answer=signed_answer, retrieval_used=False, citations=[],
                                     context_status="available")
    decision = None
    document_followup = False
    resolved_question: str | None = None
    simulator_facts = simulator_configuration_facts(query.prompt) if not documents_only else None
    planned_pipeline = None
    if (query.task_type != "error_analysis" and pipeline_provider is not None and not documents_only
            and preset_intent(query.task_type, query.prompt) is None and not is_coating_status_question(query.prompt) and not _pose_engine_question(query)):
        try:
            planned_pipeline = await to_thread.run_sync(pipeline_provider)
        except Exception:
            # Preserve engine-only responses when the optional generator is offline.
            planned_pipeline = None
        planner = select_query_pipeline(planned_pipeline, "quick") if planned_pipeline else None
        supports_dialogue = getattr(planner, "supports_copilot_dialogue", callable(getattr(planner, "plan_copilot_dialogue", None)))
        if simulator_facts is not None:
            decision = {"kind": "chat", "intent": "knowledge", "scope": "unspecified"}
        elif supports_dialogue and query.task_type == "knowledge_search" and is_general_painting_knowledge(query.prompt):
            # 계획 모델은 직전 자세·순서 대화에 끌려 일반 지식 질문을 기록 후속 대화로 분류한다(2026-11-12 재현). 부르지 않는다.
            decision = {"kind": "coating_knowledge", "intent": "knowledge", "scope": "unspecified"}
        elif supports_dialogue:
            try:
                def plan():
                    with generation_slot_for(planner):
                        return parse_plan(planner.plan_copilot_dialogue(query.prompt, task_type=query.task_type,
                                          recent_turns=model_turns, facts=facts))
                decision = await to_thread.run_sync(plan)
            except Exception:
                logger.exception("copilot dialogue planning failed request_id=%s", query.request_id)
                decision = {"kind": "clarify", "intent": "knowledge", "scope": "unspecified"}

    # Elliptical conditions may omit every hazard keyword. Once the existing
    # planner identifies the same topic, reuse the established hazard path.
    # This re-entry is bounded: an explicit safety_class skips planning below.
    previous_hazard = dialogue_turns[-1].get("safety_class") if dialogue_turns else None
    if (safety_class is None and decision is not None and decision.get("context_relation") == "followup"
            and previous_hazard in SAFETY_CLASSES):
        response = await _execute_copilot_query(
            query, pipeline_provider=pipeline_provider, pending_answers=pending_answers,
            sequences=sequences, mcp_base_url=mcp_base_url, transport=transport,
            started=request_started, reports=reports, sequence_engine=sequence_engine,
            coating_engine=coating_engine, conversation_id=conversation_id,
            original_prompt=original_prompt, safety_class=previous_hazard,
        )
        response._copilot_safety_class = previous_hazard
        return response

    # A semantic continuation of a document explanation needs the same grounded
    # document path even when this utterance omits the original technical noun.
    # The model supplies the relationship; a new-topic decision never inherits it.
    if (decision is not None and decision.get("context_relation") == "followup"
            and decision["kind"] in {"chat", "correction", "coating_knowledge"}
            and decision["intent"] == "knowledge" and dialogue_turns
            and dialogue_turns[-1].get("answer_source") == "documents"):
        document_followup = True
        decision = {**decision, "kind": "coating_knowledge"}
        query = query.model_copy(update={"task_type": "knowledge_search"})
        grounded = True
    if (decision is not None and decision.get("context_relation") == "followup"
            and decision["kind"] == "coating_knowledge" and decision["intent"] == "knowledge"):
        resolved_question = dialogue_resolution_advice(
            decision.get("resolved_question"), prompt=query.prompt, recent_turns=dialogue_turns,
        )

    async def chat_answer(hedged: bool = False, *, conversation_pipeline: object | None = None,
                          knowledge_scope: bool = False) -> JSONResponse:
        """계획 모델이 대화로 분류한 질문에 문서 검색 없이 답한다(세션 요약·측정값 검사 포함)."""
        try:
            # A document miss does not widen the viewer data supplied to a
            # quick general explanation. Use the same scoped facts/history for
            # generation and numeric validation as the knowledge branch.
            conversation_facts = (dict(viewer_facts or {}) if knowledge_scope and viewer_requested else facts)
            if simulator_facts is not None:
                conversation_facts = dict(simulator_facts)
            conversation_turns = [] if knowledge_scope and definition_context is not None else model_turns
            omit_training_values = (simulator_facts is not None
                                    or (knowledge_scope and viewer_requested and query.response_mode == "quick" and not record_pointed))
            if query.session_id and not omit_training_values:
                conversation_facts["training_summary"] = await to_thread.run_sync(
                    lambda: summarize_session(sequences.session_bodies(query.session_id),
                                              selected_sequence_id=_selected_sequence(query.context)))
                if _EVALUATION_QUESTION.search(query.prompt):
                    found_report = (await to_thread.run_sync(lambda: reports.latest_complete_analysis(query.session_id))
                                    if reports is not None else None)
                    return _success_response(query, request_started, answer=_session_findings(found_report),
                                             retrieval_used=False, citations=[],
                                             context_status="available" if found_report else "not_available")
            simulator = {"simulator_context": SIMULATOR_CONTEXT} if wants_simulator_context(query.prompt) else {}
            conversation_engine = conversation_pipeline if conversation_pipeline is not None else planned_pipeline
            generated_model: str | None = None
            def converse():
                nonlocal generated_model
                kwargs = dict(recent_turns=conversation_turns, facts=conversation_facts, response_mode=query.response_mode, **simulator)
                if query.summarize or knowledge_scope:
                    target = getattr(conversation_engine, "_parent", conversation_engine)
                    parameters = inspect.signature(target.converse_copilot).parameters
                    accepts_options = any(item.kind == inspect.Parameter.VAR_KEYWORD for item in parameters.values())
                    if query.summarize and ("summarize" in parameters or accepts_options):
                        kwargs["summarize"] = True
                    if knowledge_scope and ("knowledge_fallback" in parameters or accepts_options):
                        kwargs["knowledge_fallback"] = True
                deadline = generation_deadline_scope(30.0 if query.response_mode == "detailed" else 15.0) if knowledge_scope else nullcontext()
                with generation_slot_for(conversation_engine), deadline:
                    raw = conversation_engine.converse_copilot(query.prompt, **kwargs)
                fallback = select_query_pipeline(conversation_engine, "quick")
                if fallback is not conversation_engine and _DIAGNOSTIC_SCAFFOLD.search(raw):
                    with generation_slot_for(fallback):
                        answer = fallback.converse_copilot(query.prompt, **kwargs)
                    generated_model = query_model_name(fallback)
                    return answer
                generated_model = query_model_name(conversation_engine)
                return raw
            def answer_body(raw):
                # No document was supplied to this fallback generator. Model
                # document IDs must not survive merely because the array is [].
                return (parse_knowledge_answer(raw, []).body if knowledge_scope else plain_answer(raw))
            answer = answer_body(await to_thread.run_sync(converse))
            extended = is_domain_question(query.prompt)
            if ungrounded_measurements(answer, facts=conversation_facts, prompt=query.prompt, extended=extended):
                # 자료에 없는 측정값은 내보내지 않는다. 한 번 다시 생성하고, 그래도 남으면 그 부분을 뺀다.
                answer = drop_ungrounded(answer_body(await to_thread.run_sync(converse)), facts=conversation_facts, prompt=query.prompt, extended=extended)
            answer = _checked_generation(answer, query)
            if answer in {NO_EVIDENCE_ANSWER, _NO_ANSWER, INSTRUCTION_REFUSAL}:
                generated_model = None
            if hedged:
                answer = hedged_answer(answer, quick=query.response_mode == "quick") or NO_EVIDENCE_ANSWER
            if (knowledge_scope and not hedged and has_body(answer) and cacheable_candidate(answer)
                    and answer not in {NO_EVIDENCE_ANSWER, _NO_ANSWER, INSTRUCTION_REFUSAL}
                    and not answer.startswith((GENERAL_KNOWLEDGE_LIMIT, HEDGED_PREFIX, QUICK_HEDGED_PREFIX))):
                answer = GENERAL_KNOWLEDGE_LIMIT + answer
            return _success_response(query, request_started, answer=answer, retrieval_used=False, generated=True,
                                     citations=[], context_status="available" if facts or simulator_facts else "not_available",
                                     model_name=generated_model)
        except Exception:
            logger.exception("copilot conversation unavailable request_id=%s", query.request_id)
            return error_response(503, "Conversation engine temporarily unavailable; retry")

    if simulator_facts is not None:
        return await chat_answer()

    chat_redirected = False
    if decision is not None:
        intent = decision["intent"]
        if intent == "order_reference" and explicit_actual_order:
            intent = explicit_actual_order
            decision = {**decision, "intent": intent}
        if has_reference_plan and not relevant_viewer and intent == "order_reference" and decision["kind"] == "lookup" and not explanatory_request:
            return await reference_answer()
        cross_tab = ((intent.startswith("pose_") and query.task_type != "pose_recomm")
                     or (intent.startswith("order_") and query.task_type != "trajectory_recomm"))
        if viewer_requested and intent == "order_direction" and decision["kind"] in {"lookup", "correction"} and DIRECTION_WORD.search(query.prompt) and not explanatory_request:
            return frame_answer()
        zone_lookup = (intent == "order_direction" or intent.startswith("pose_")) and decision["scope"] in {"zone", "playback"}
        # Preserve positive painting/record grounding, but a generic how/why
        # sentence must not override the planner's semantic chat decision.
        chat_redirected = (decision["kind"] == "chat" and query.task_type == "knowledge_search" and not viewer_requested
                           and not cross_tab and not zone_lookup and bool(_KNOWLEDGE_QUESTION.search(query.prompt))
                           and (grounded or is_technical_question(query.prompt)) and not is_off_topic(query.prompt)
                           and _abstain_below(planned_pipeline, query.response_mode) is not None)
        # 옛 행(v3.4.4 이전)에는 safety_class 열이 없어 앞 질문 문장도 다시 위험 판정한다.
        if (decision["kind"] == "correction" and dialogue_turns and not dialogue_turns[-1].get("safety_class")
                and classify(dialogue_turns[-1]["prompt"]) is None):
            # "아까 답 이상한데"는 앞 질문을 기록으로 다시 답한다. 앞 턴이 위험 질문이면 다시 분류하지 않는다(안전 검사를 건너뛰지 않게).
            again = answer_record_question(query.context, dialogue_turns[-1]["prompt"], task_type=query.task_type,
                                           recent_turns=dialogue_turns[:-1])
            if again is not None and again.kind != "mixed":
                if _objection_only(query.prompt):
                    return _success_response(query, request_started, answer=again.text, retrieval_used=False, citations=[],
                                             sequence_context=again.sequence_context, context_status="available", subject=again.subject)
                # 앞 답이 기록 답이고 새 말에 주장·질문이 있으면 되풀이하지 않고 그 말을 문서로 답한다(사용자 결정 I6).
                # 앞 답이 기록 답이 아니면(잡담·주제 밖 지적) 예전처럼 아래 대화 경로가 받는다.
                decision = {**decision, "kind": "coating_knowledge", "intent": "knowledge"}
                query = query.model_copy(update={"task_type": "knowledge_search"})
                grounded = True
        # 기록·도장 지식 질문은 대화 모델이 아니라 아래 분석·문서 경로가 답한다. 되묻기와 잡담만 대화로 답한다.
        if ((decision["kind"] in {"chat", "clarify", "correction"} or cross_tab or zone_lookup) and not chat_redirected
                and (decision["kind"] == "clarify" or not grounded
                     or (query.task_type == "pose_recomm" and _TAB_OR_WORD.search(query.prompt)))):
            return await chat_answer()

    trusted_coating = parse_viewer_coating_context(query.context, query.task_type)
    # Preserve v3.8.15: quick knowledge answers receive measurements only for a record-directed question.
    record_pointed = bool(RECORD_OBJECT.search(query.prompt))
    coating_advice_context = (trusted_coating if mixed is None and is_coating_advice_question(query.prompt) and trusted_coating and trusted_coating.get("analysis_status") == "available"
                              and (query.response_mode != "quick" or record_pointed) else None)

    async def report_facts() -> dict[str, Any]:
        found_coating = (await to_thread.run_sync(lambda: reports.latest_complete_analysis(query.session_id))
                         if reports is not None and query.session_id else None)
        return report_coating_context(found_coating)

    async def coating_facts() -> dict[str, Any]:
        """서명된 뷰어 자료가 먼저이고, 없으면 완료 리포트의 도막 행."""
        return trusted_coating if trusted_coating is not None else await report_facts()

    if not documents_only and (is_coating_status_question(query.prompt) or (decision and decision["intent"] == "coating_status" and decision["kind"] == "lookup")):
        coating = await coating_facts()
        available = coating.get("analysis_status") == "available"
        return _success_response(query, request_started,
            answer=answer_coating_question(coating, query.prompt, detailed=True),
            retrieval_used=False, citations=[], coating_context=coating,
            context_status="available" if available else "not_available")

    intent: str | None = decision["intent"] if decision else None
    intent_pipeline: object | None = planned_pipeline
    classifier = None
    if query.task_type in ("pose_recomm", "trajectory_recomm"):
        classifier = None
        intent = intent or preset_intent(query.task_type, query.prompt)
        if intent is None and _pose_engine_question(query):
            intent = fallback_intent(query.task_type, query.prompt)
        if pipeline_provider is not None and intent is None:
            try:
                intent_pipeline = await to_thread.run_sync(pipeline_provider)
                method = getattr(intent_pipeline, "classify_copilot_intent", None)
                if callable(method):
                    def classifier(task_type: str, prompt: str) -> str:
                        with generation_slot_for(intent_pipeline):
                            return str(method(task_type, prompt))
            except Exception:  # noqa: BROAD_EXCEPT_OK - keyword fallback keeps structured answers available
                logger.exception("copilot intent pipeline unavailable request_id=%s", query.request_id)
        if intent is None:
            intent = await to_thread.run_sync(
                lambda: classify_question_intent(query.task_type, query.prompt, classifier)
            )
        logger.info("copilot intent request_id=%s task=%s intent=%s", query.request_id, query.task_type, intent)
    if intent == "order_reference" and explicit_actual_order:
        intent = explicit_actual_order
    if has_reference_plan and not relevant_viewer and intent == "order_reference" and not explanatory_request:
        return await reference_answer()
    # If the model is unavailable, keep the selected structured tab operational;
    # only a successful semantic-model classification may cross-route to RAG.
    knowledge_routed = intent == "knowledge" and (classifier is not None or decision is not None)
    if viewer_requested and intent == "order_direction" and not knowledge_routed and not explanatory_request and (
            query.task_type == "trajectory_recomm" or DIRECTION_WORD.search(query.prompt)):
        return frame_answer()
    viewer_context_requested = bool(query.context and query.context.startswith(_MOTION_VIEWER_CONTEXT_PREFIX))
    if viewer_analysis is not None and query.task_type == "pose_recomm" and "playback_context" in viewer_analysis and _asks_current_viewer_pose(query.prompt) and not is_coating_advice_question(query.prompt):
        knowledge_routed = False

    async def _synthesize_structured_detail(
        base_answer: str,
        analysis_context: dict[str, object],
        *,
        available: bool,
    ) -> tuple[str, bool]:
        """Computed answers already contain the requested fields in both modes.

        Selecting detailed must not append unrelated statistics or a second
        score family. Explicit multi-field/list requests are assembled at their
        factual source before this boundary; structured context remains full.
        """
        return base_answer, False

    if viewer_analysis is not None and query.task_type in ("pose_recomm", "trajectory_recomm") and not knowledge_routed:
        declined = unsupported_pose_part(query.prompt) if query.task_type == "pose_recomm" else None
        if declined is not None:
            return _success_response(query, request_started, answer=declined, retrieval_used=False, citations=[],
                                     context_status="not_available")
        analysis_context = (_current_viewer_pose_context(query.prompt, viewer_analysis)
                            if query.task_type == "pose_recomm" else viewer_analysis)
        available = analysis_context.get("scope") != "current_zone" or analysis_context.get("status") == "available"
        verified_answer = _viewer_answer(query.task_type, query.prompt, analysis_context, intent)
        arm_pose = parse_viewer_arm_pose(query.context) if query.task_type == "pose_recomm" else None
        side = asked_arm(query.prompt, intent) if query.task_type == "pose_recomm" else None
        if arm_pose is not None:
            arm_answer = arm_pose_answer(
                arm_pose, current=analysis_context.get("scope") == "current_zone",
                playback=viewer_analysis.get("playback_context"),  # type: ignore[arg-type]
                zone_id=requested_zone(query.prompt, None), side=side, prefer_hand=prefers_hand(query.prompt), prompt=query.prompt,
            )
            # Keep the requested score family; show both only when both were
            # explicitly requested. Signed measures/context remain unchanged.
            work_requested = bool(_WORK_SCORE_ASKED.search(query.prompt))
            pose_requested = bool(re.search(r"자세|관절|팔", query.prompt))
            verified_answer = (arm_answer + "\n" + verified_answer if work_requested and pose_requested
                               else verified_answer if work_requested else arm_answer)
            analysis_context = {**analysis_context, "arm_pose": {key: item for key, item in arm_pose.items() if key != "sequences"}}
            available = available or arm_pose.get("status") == "available"
        elif (query.task_type == "pose_recomm"  # 순서 탭은 팔 채점을 읽지 않는다(검토 I1)
              and (side is not None or BODY_PART.search(query.prompt) or _POSE_ASKED.search(query.prompt))
              and not _WORK_SCORE_ASKED.search(query.prompt)):
            # 자세 유사도(팔 관절 엔진)와 작업 유사도는 다른 값이다. 팔 채점 없이 자세·신체 부위를 물으면 채점이 없다고 답하고,
            # 작업 유사도는 상세 분석에서만 다른 값이라고 밝힌 뒤 덧붙인다(2026-11-17 사용자 결정 I7). 작업 유사도를 직접 물으면 위 조건에서 빠진다.
            notice = _no_arm_notice(query.prompt, side)
            verified_answer = (notice if query.response_mode == "quick" else
                               f"{notice} 자세 채점이 끝난 뒤 다시 질문해 주세요.\n{NO_ARM_WORK_NOTE} {verified_answer}")
        answer, detail_degraded = await _synthesize_structured_detail(
            verified_answer, analysis_context, available=available
        )
        return _success_response(
            query, request_started, answer=answer, retrieval_used=False, citations=[],
            degraded=detail_degraded,
            pose_context=analysis_context if query.task_type == "pose_recomm" else None,
            sequence_context=analysis_context if query.task_type == "trajectory_recomm" else None,
            context_status="available" if available else "not_available",
        )
    # Unsigned/forged viewer claims can still resolve against saved server data.
    if (viewer_context_requested and viewer_claims(query.context) is None
            and parse_viewer_arm_pose(query.context) is None
            and parse_viewer_coating_context(query.context, "knowledge_search") is None):
        query = query.model_copy(update={"context": None})

    context = (query.context or "").strip()
    # 시퀀스(`SEQ-012`)를 가리키지 않은 자세·순서 질문은 complete 가 남긴 세션 결과로 답한다(pose_query 참고).
    # complete 뒤에는 질문에 "구역"이 없어도 그렇다 — 뷰어 채팅은 칩으로 유형만 고른다.
    session_id = query.session_id
    if not query.scenario_id and session_id:
        def _saved_scenario() -> str | None:
            latest = sequences.latest_sequence_id(session_id)
            body = sequences.load_body(session_key=session_id, sequence_id=latest) if latest else None
            if body and body.get("scenario_id"):
                return str(body["scenario_id"])
            get_session = getattr(reports, "get_session", None)
            session = get_session(session_id) if callable(get_session) else None
            return session.scenario_id if session is not None else None
        scenario = await to_thread.run_sync(_saved_scenario)
        if scenario:
            query = query.model_copy(update={"scenario_id": scenario})
    if query.task_type == "pose_recomm" and not knowledge_routed and viewer_analysis is None:
        declined = unsupported_pose_part(query.prompt)
        if declined is not None:
            return _success_response(query, request_started, answer=declined, retrieval_used=False, citations=[],
                                     context_status="not_available")
        if context and not context.isdigit() and asks_whole_session(query.prompt, intent):
            context = ""  # 세션 전체를 묻는 질문은 붙인 시퀀스 대신 세션 결과로 답한다
    found = (
        await to_thread.run_sync(lambda: reports.latest_complete_analysis(session_id))
        if reports is not None and session_id and query.task_type in ("pose_recomm", "trajectory_recomm")
        # 순서 답은 세션 단위라 붙인 시퀀스와 무관하다. 자세는 시퀀스를 붙이면 그 분사를 채점한다.
        and (query.task_type == "trajectory_recomm" or not context or context.isdigit())
        else None
    )
    if found is not None and session_id:
        captured = found[1].get("sequences")
        captured_ids = {str(item["sequence_id"]) for item in captured
                        if isinstance(item, Mapping) and item.get("sequence_id")} if isinstance(captured, list) else set()
        if captured_ids:
            latest = await to_thread.run_sync(lambda: sequences.latest_sequence_id(session_id))
            if latest is not None and latest not in captured_ids:
                found = None
    working_zone = live_zone = None
    if found is None and session_id and not context and query.task_type == "pose_recomm" and not knowledge_routed:
        # 훈련 중에는 지금 작업하는 구역으로 답한다. 세션 전체를 묻는 질문만 올라온 분사 전체를 본다.
        working_zone = await to_thread.run_sync(lambda: current_zone(sequences, session_id))
        live_zone = requested_zone(query.prompt, query.context) or (
            None if asks_whole_session(query.prompt, intent) else working_zone)
    if found is None and session_id and (not context or query.task_type == "trajectory_recomm") and query.task_type in ("pose_recomm", "trajectory_recomm") and not knowledge_routed and (reports is not None or query.task_type == "trajectory_recomm" or asks_about_zones(query.prompt, query.context)):
        # Keep explicit-sequence pose queries on their original scoring path.
        found = await to_thread.run_sync(lambda: live_analysis(
            sequences, session_id=session_id, scenario_id=query.scenario_id,
            task_type=query.task_type, sequence_engine=sequence_engine,
            mcp_base_url=mcp_base_url, transport=transport,
            zone_id=live_zone,
        ))
    elif found is None and session_id and context.isdigit() and query.task_type == "pose_recomm" and not knowledge_routed:
        found = await to_thread.run_sync(lambda: live_analysis(
            sequences, session_id=session_id, scenario_id=query.scenario_id,
            task_type=query.task_type, sequence_engine=sequence_engine,
            mcp_base_url=mcp_base_url, transport=transport, zone_id=context,
        ))
    if query.task_type == "trajectory_recomm" and not knowledge_routed and session_id:
        if found is not None and sequence_engine is not None and query.scenario_id:
            report_id, analysis = found
            refreshed = dict(analysis)
            stored_report = refreshed.get("sequence_report")
            if isinstance(stored_report, dict):
                refreshed["sequence_report"] = sequence_engine.rescore_report(
                    stored_report, scenario_id=query.scenario_id
                )
                found = (report_id, refreshed)
        order = answer_order_question(found, prompt=query.prompt, intent=intent)
        logger.info(
            "copilot trajectory_recomm request_id=%s status=%s reason=%s",
            query.request_id, order.analysis_status, order.reason_code,
        )
        answer, detail_degraded = await _synthesize_structured_detail(
            order.answer, order.context, available=order.analysis_status == "available"
        )
        return _success_response(
            query, request_started, answer=answer, retrieval_used=False, citations=[],
            degraded=detail_degraded,
            sequence_context=order.context,
            context_status="available" if order.analysis_status == "available" else "not_available",

        )

    if query.task_type == "pose_recomm" and not knowledge_routed and (
        found is not None or asks_about_zones(query.prompt, query.context)
    ):
        zone = (
            answer_pose_question(found, prompt=query.prompt, context=live_zone or query.context, intent=intent)
            if session_id
            else zone_unavailable("session_id_missing")
        )
        logger.info(
            "copilot pose_recomm zone request_id=%s zone=%s status=%s reason=%s",
            query.request_id, zone.context.get("zone_id"), zone.analysis_status, zone.reason_code,
        )
        live_context = found[1].get("live_pose_context") if found is not None else None
        report_pose = found[1].get("pose_report") if found is not None else None
        limited_confidence = isinstance(report_pose, Mapping) and report_pose.get("confidence") == "limited"
        if isinstance(report_pose, Mapping):
            for key in ("confidence", "limited_sequence_count", "reason_codes"):
                if key in report_pose:
                    zone.context[key] = report_pose[key]
        base_answer = zone.answer
        if isinstance(live_context, dict):
            zone.context.update(live_context)
            base_answer = ("지금 작업 중인 구역 기준입니다. " if live_zone and live_zone == working_zone
                           else "" if live_zone else "지금까지 올라온 분사 기준입니다. ") + base_answer
        answer, detail_degraded = await _synthesize_structured_detail(
            base_answer, zone.context, available=zone.analysis_status == "available"
        )
        return _success_response(
            query, request_started, answer=answer, retrieval_used=False, citations=[],
            degraded=detail_degraded or limited_confidence or bool(isinstance(live_context, dict) and live_context.get("reason_codes")),
            pose_context=zone.context,
            context_status="available" if zone.analysis_status == "available" else "not_available",

        )

    if query.task_type == "pose_recomm" and not knowledge_routed:
        pose = await to_thread.run_sync(
            lambda: score_pose(
                sequences,
                session_id=query.session_id,
                context=query.context,
                mcp_base_url=mcp_base_url,
                transport=transport,  # type: ignore[arg-type]
            )
        )
        if pose.analysis_status == "available":
            answer = answer_sequence_question(pose, prompt=query.prompt, intent=intent)
        else:
            answer = _POSE_REASON_ANSWERS.get(pose.reason_code or "", _POSE_FALLBACK_ANSWER)
        logger.info(
            "copilot pose_recomm request_id=%s sequence=%s status=%s reason=%s",
            query.request_id, pose.sequence_id, pose.analysis_status, pose.reason_code,
        )
        answer, detail_degraded = await _synthesize_structured_detail(
            answer, pose.as_context(), available=pose.analysis_status == "available"
        )
        return _success_response(
            query, request_started, answer=answer, retrieval_used=False, citations=[],
            degraded=detail_degraded or bool(pose.reason_code and pose.analysis_status == "available"),
            pose_context=pose.as_context(),
            context_status="available" if pose.analysis_status == "available" else "not_available",

        )

    if query.task_type != "knowledge_search" and not knowledge_routed:
        if query.task_type == "error_analysis":
            coating = await coating_facts()
            measures = _viewer_measures(query.context, query.prompt)[0]
            cause = answer_distribution_cause(coating, detailed=True, measures=measures)
            if cause is None and trusted_coating is not None:
                # 뷰어 자료에 목표 대비 판정이 없으면(전문가 대비 행뿐) 완료 리포트로 답한다.
                coating = await report_facts()
                cause = answer_distribution_cause(coating, detailed=True)
            if cause is not None:
                return _success_response(query, request_started, answer=cause, retrieval_used=False, citations=[],
                                         coating_context=coating, context_status="available")
        answer = pending_answers.get(query.task_type) or PENDING_TASK_ANSWERS[query.task_type]
        return _success_response(
            query, request_started, answer=answer, retrieval_used=False, citations=[]
        )

    if pipeline_provider is None:
        return record_only(mixed) if mixed is not None else error_response(503, "Copilot pipeline is not configured")
    try:
        pipeline = intent_pipeline or await to_thread.run_sync(pipeline_provider)
    except Exception:  # noqa: BROAD_EXCEPT_OK - construction failures surface as not ready
        logger.exception("copilot query: pipeline construction failed request_id=%s", query.request_id)
        return record_only(mixed) if mixed is not None else error_response(503, "Copilot pipeline is not ready")

    # Explicit Y/N remains a backwards-compatible override.  When omitted, any
    # question that reaches this knowledge-generation branch needs retrieval;
    # structured pose/order/report questions have already returned above.
    use_rag = query.use_rag != "N" or safety
    session_summary = ""
    # 섞인 질문의 문서 쪽에는 기록 수치가 든 분석 답도 넘기지 않는다(질문만 남겨 지시어를 풀게 한다).
    recent_turns: list[dict[str, Any]] = [{**turn, "answer": ""} for turn in dialogue_turns] if mixed is not None else model_turns
    question = (mixed.knowledge_question or query.prompt) if mixed is not None else query.prompt  # 지시문·검색어의 질문
    if query.session_id and mixed is None:  # 섞인 질문의 문서 쪽에는 세션 수치도 넘기지 않는다
        session_summary = await to_thread.run_sync(
            lambda: summarize_session(sequences.session_bodies(query.session_id),
                                      selected_sequence_id=_selected_sequence(query.context)))
    definition_context = (viewer_definition_context(query.prompt, query.context)
                          if query.response_mode == "quick" and query.task_type == "knowledge_search" and mixed is None else None)
    # 질문 유형(결함 뜻·원인·영향·대책)은 최근 대화에서 대상이 적힌 가장 가까운 질문까지 거슬러 이어받는다(명시적 후속 질문일 때만).
    # 용어 규칙은 이전 질문들의 제외어까지 보므로 직전 질문 하나만 넘긴다.
    followup = bool(dialogue_turns) and decision is not None and decision.get("context_relation") == "followup"
    inherit = [str(dialogue_turns[-1].get("prompt", ""))] if followup else []
    chain = [str(turn.get("prompt", "")) for turn in dialogue_turns] if followup else []
    prompt_type = (matched_prompt(question, chain)
                   if query.task_type == "knowledge_search" and mixed is None and use_rag and coating_advice_context is None else None)
    # 유형 질문은 기록을 가리키지 않으면 일반 지식이다. 상세 분석도 기록 수치를 넘기지 않는다: 넘기면 27B가 "이 기록에서 왜"로 읽고
    # 일반 원인 근거를 받고도 기권했다(2026-11-19 뷰어 '그럼 왜 생겨?'). 내 기록과의 비교는 "내가 한 거랑 비교하면?"이 따로 답한다.
    viewer_facts = facts if facts and (query.response_mode != "quick" or record_pointed) and (prompt_type is None or record_pointed) else None
    knowledge_client_context = MIXED_DOCUMENT_RULE if mixed is not None else (definition_context or (json.dumps(viewer_facts, ensure_ascii=False, separators=(",", ":")) if viewer_facts else viewer_metadata_context(query.context))) if viewer_requested else (definition_context or query.context)
    if coating_advice_context is not None:
        compact = target_advice_facts(coating_advice_context, measures=_viewer_measures(query.context, query.prompt)[0])
        knowledge_client_context = "[검증된 최종 도막 비교 자료]\n" + json.dumps(compact or coating_advice_context, ensure_ascii=False, separators=(",", ":"))
        knowledge_client_context += (
            "\n목표 대비 판정이며 재생 시점의 도막이 아닙니다. 조절 후보는 등급에 따른 일반적인 방향이고 원인으로 확정된 것이 아닙니다."
            if compact else "\n최종 기록의 전문가 대비 대표 두께 비교이며 재생 시점의 현재 도막 판정이 아닙니다."
        ) + " 질문의 원인·방법·관계를 직접 설명하고 확인된 측정 사실과 일반적인 가능한 원인을 구분하세요. 권장 Zone 순서를 대신 답하지 마세요."
    if resolved_question:
        knowledge_client_context = (knowledge_client_context or "") + (
            "\n[후속 질문 해석 — 미검증 대화 참고]\n" + json.dumps(resolved_question, ensure_ascii=False) +
            "\n모델의 임시 해석일 뿐 답변·사실·새 주제의 근거가 아닙니다. "
            "사용자 원문 대화의 대상이나 조건을 추가하거나 바꾸는 부분은 사용하지 마세요."
        )
    # 질문 유형별 맥락 프롬프트(configs/context_prompts.json: 결함 영향·원인, 충전전로 이격거리).
    # 골라 둔 원문 조각을 근거 앞에 붙이고 유형 규칙을 준다. 명시적 후속 질문일 때만 대상을 이어받는다.
    # 유형은 최근 대화에서 대상이 적힌 가장 가까운 질문까지 거슬러 본다("저도막이 뭐야?"→"원인이 뭔데?"→"그러면 어떻게 해야 하는데?").
    # 유형(prompt_type)은 기록 수치를 넘길지 정하기 전에 위에서 골랐다.
    if prompt_type is not None:
        knowledge_client_context = (knowledge_client_context + "\n" if knowledge_client_context else "") + context_text(prompt_type)
    # 교육 용어 ↔ 문서 표현(configs/term_rules.json): 검색어에 문서 표현을 덧붙이고 대응 한 줄을 준다.
    # 내 기록을 묻는 질문(기록 지칭·뷰어 서명 자료·도막 비교)에는 문서 쪽 표현을 섞지 않는다.
    term_rules = (matched_terms(question, inherit) if query.task_type == "knowledge_search" and mixed is None and use_rag
                  and not record_pointed and not facts and coating_advice_context is None else [])
    # 유형 규칙이 맡은 용어는 겹쳐 주지 않는다. 겹치면 8B가 근거에 없는 원인을 지어냈다(2026-11-18 '과도막은 왜 생겨?').
    term_rules = [rule for rule in term_rules if prompt_type is None or rule["용어"] != prompt_type["용어"]]
    if term_rules:
        knowledge_client_context = (knowledge_client_context + "\n" if knowledge_client_context else "") + term_context(term_rules)
    # Add the relation after all fact-specific context builders, which may
    # replace the client context. Retrieval must retain this scope decision.
    if document_followup:
        knowledge_client_context = "[문서 후속 질문]\n직전 기술 설명에 대한 조건 추가 또는 적용 범위의 재검토입니다.\n" + (knowledge_client_context or "")
    elif decision is not None and decision.get("context_relation") == "new_topic":
        knowledge_client_context = "[새 주제 질문]\n이전 검색 주제를 이어받지 않는 독립적인 질문입니다.\n" + (knowledge_client_context or "")
    block = "" if is_blasting(query.prompt) else SIMULATOR_CONTEXT
    instruction = build_query_instruction(
        question,
        client_context=knowledge_client_context,
        session_summary=session_summary,
        recent_turns=[] if definition_context is not None else recent_turns,
        response_mode=query.response_mode,
        concise=query.response_mode == "detailed",
        summarize=query.summarize,
        simulator_context=block,
    )
    generation_instruction = (build_query_instruction(
        question,
        client_context=knowledge_client_context,
        session_summary=session_summary,
        recent_turns=[] if definition_context is not None else recent_turns,
        response_mode=query.response_mode,
        concise=True,
        summarize=query.summarize,
        simulator_context=block,
    ) if query.response_mode == "quick" else instruction)

    # The quick adapter forwards kwargs to its parent. Older SDK/test pipelines
    # keep their exact run signature and their previous post-generation gate.
    run_target = getattr(pipeline, "_parent", pipeline)
    try:
        parameters = inspect.signature(run_target.run).parameters
        supports_deferral = "defer_without_evidence" in parameters
        supports_summary = "answer_summary" in parameters or any(item.kind == inspect.Parameter.VAR_KEYWORD for item in parameters.values())
        supports_pin = "pinned_chunk_ids" in parameters or any(item.kind == inspect.Parameter.VAR_KEYWORD for item in parameters.values())
    except (TypeError, ValueError):
        supports_deferral = False
        supports_summary = False
        supports_pin = False
    may_defer = supports_deferral and use_rag and not documents_only

    def _guarded_run() -> Mapping[str, object]:
        with generation_slot_for(pipeline):
            return pipeline.run(
                instruction=instruction,
                answer_question=question,
                force_route="fast",
                use_pipeline_history=False,
                session_id=query.session_id,
                enable_thinking=False,
                use_rag=use_rag,
                max_tokens=1024,
                **({"answer_summary": True} if query.summarize and supports_summary else {}),
                **({"pinned_chunk_ids": list(prompt_type["조각"])} if prompt_type is not None and supports_pin else {}),
                **({"defer_without_evidence": may_defer,
                    "evidence_score_threshold": _abstain_below(pipeline, query.response_mode)}
                   if supports_deferral else {}),
                **({"compact_analysis": True} if query.response_mode == "detailed" else {}),
                request_id=query.request_id,
                **({
                    # Preserve the established full-body cache namespace for
                    # false, and isolate the new concise-generation namespace.
                    "quick_cache_scope": json.dumps([query.user_id, query.scenario_id, query.task_type] + ([True] if query.summarize else [])),
                    "concise_answer": True,
                    "generation_instruction": generation_instruction,
                    "retrieval_instruction": with_document_terms(question, term_rules),
                } if query.response_mode == "quick" else {
                    # 응답 수준 지시문이 검색어에 섞이면 무관한 문단이 잡힌다(Issue #9).
                    "retrieval_instruction": with_document_terms(retrieval_question(question, context=instruction, expand_terms=False), term_rules),
                }),
            )

    try:
        result = await to_thread.run_sync(_guarded_run)
    except OpenAIBackendUnavailableError:
        logger.exception("copilot query: model server unavailable request_id=%s", query.request_id)
        return record_only(mixed) if mixed is not None else error_response(503, "Model server unavailable")
    except Exception:  # noqa: BROAD_EXCEPT_OK - never leak internals past the boundary
        logger.exception("copilot query: pipeline failed request_id=%s", query.request_id)
        return record_only(mixed) if mixed is not None else error_response(500, "Copilot query failed")

    metadata = result.get("metadata")
    review = metadata.get("answer_review") if isinstance(metadata, Mapping) else None
    review_failed = isinstance(review, Mapping) and review.get("verdict") not in {"ok", "rewritten", "cached", "not_required", "attribution_corrected"}
    retrieval_decision = metadata.get("retrieval_decision") if isinstance(metadata, Mapping) else None
    retrieval_missing = (not review_failed and isinstance(retrieval_decision, Mapping)
            and retrieval_decision.get("status") == "no_relevant_evidence"
            and retrieval_decision.get("attempted") is True
            and retrieval_decision.get("reason") in {"empty", "below_threshold"})
    # The semantic reviewer can find that high-ranked passages do not support
    # this question. Its validated INSUFFICIENT result is distinct from a
    # failed/unavailable judge, a rejected answer or an exhausted time budget.
    review_insufficient = isinstance(review, Mapping) and review.get("verdict") == "insufficient"
    # Failure to produce a verified RAG answer is not a finding that the
    # documents lack evidence. Discard it, then allow a separate no-document
    # answer only when this runtime guarantees that answer its own review.
    unreviewed_body = not cacheable_candidate(str(result.get("response", "")))
    rag_abstained = (isinstance(review, Mapping) and review.get("cacheable") is False and unreviewed_body
                     and not parse_knowledge_answer(str(result.get("response", "")), retrieved_chunk_ids(result.get("rag_chunks") or [])).citations)
    fresh_general = (bool(getattr(pipeline, "supports_reviewed_knowledge_fallback", False))
                     and query.task_type == "knowledge_search" and not record_pointed and not query.context and not query.session_id
                     and (review_failed or rag_abstained))
    if may_defer and (retrieval_missing or review_insufficient or fresh_general):
        response = await chat_answer(conversation_pipeline=pipeline, knowledge_scope=True)
        original_trace = metadata.get("answer_trace") if isinstance(metadata, Mapping) else None
        fallback_trace = dict(original_trace) if isinstance(original_trace, Mapping) else {}
        fallback_trace["fallback_reason"] = ("review_insufficient" if review_insufficient else
            "retrieval_" + str(retrieval_decision["reason"]) if retrieval_missing else "rag_answer_unavailable")
        return _with_answer_trace(response, {**result, "metadata": {**metadata, "answer_trace": fallback_trace}})
    # 근거 확인 단계: 재정렬 1위 점수가 기준값 미만이면 생성한 답을 버린다(설정이 없으면 지금처럼 동작).
    # 기준값은 재정렬 모델의 점수 척도에 묶인다(KURE-Reranker-nano 3.28, 2026-11-13 측정). 상세분석만 잰 값이다.
    abstain_below = _abstain_below(pipeline, query.response_mode)
    # 결함 원인 묶음 조각은 골라 둔 근거라 재정렬 점수로 거르지 않는다(과도막 묶음 최고 2.59 < 3.28, 2026-11-18).
    pinned_used = any(isinstance(c, dict) and c.get("pinned") for c in result.get("rag_chunks") or [])
    if use_rag and abstain_below is not None and not pinned_used:
        scores = [float(c["rerank_score"]) for c in result.get("rag_chunks") or []
                  if isinstance(c, dict) and c.get("rerank_score") is not None]
        if not scores or max(scores) < abstain_below:
            logger.info("copilot answer withheld: weak evidence request_id=%s top=%s threshold=%s redirected=%s",
                        query.request_id, max(scores) if scores else None, abstain_below, chat_redirected)
            technical = is_technical_question(query.prompt)
            if chat_redirected and not technical and not review_failed:
                return await chat_answer(knowledge_scope=True)  # 주제 밖 질문은 일부러 열어 둔다
            if (not safety and mixed is None and not review_failed and allows_general_definition(query.prompt)
                    and callable(getattr(planned_pipeline, "converse_copilot", None))):
                return await chat_answer(hedged=True, knowledge_scope=True)
            # 위험 주제 질문은 근거가 약해도 그 종류의 고정 문장으로 답한다.
            withheld, withheld_citations = guard_answer(safety_class, "", []) if safety else (NO_EVIDENCE_ANSWER, [])
            if not safety and _DIZZY.search(query.prompt):
                # 빠른 안내에서는 몸 상태 안내를 먼저 전한다.
                withheld = DIZZY_NOTE + "\n" + withheld if query.response_mode == "quick" else withheld + "\n" + DIZZY_NOTE
            if mixed is not None:
                withheld = mixed.text + "\n" + MIXED_NO_EVIDENCE
            response = _success_response(query, request_started, answer=withheld, retrieval_used=bool(withheld_citations), citations=withheld_citations,
                                     degraded=review_failed,
                                     context_status="available" if session_summary or coating_advice_context or facts else "not_available",
                                     subject=mixed.subject if mixed is not None else None)
            return _with_answer_trace(response, result)
    retrieved = retrieved_chunk_ids(result.get("rag_chunks") or []) if use_rag else []
    parsed = parse_knowledge_answer(str(result.get("response", "")), retrieved)
    answer = parsed.body
    if query.response_mode == "quick":
        answer = re.sub(r"\n+\s*\[?근거\]?\s*:?\s*$", "", answer).rstrip()
    answer = _checked_generation(
        answer, query, instructions=_document_rules(pipeline), safety=safety, drop_only=mixed is not None,
        evidence=tuple(str(c.get("text", "")) for c in result.get("rag_chunks") or [] if isinstance(c, dict)))
    # 위험한 행동을 해도 되는지 묻는 안전 질문은 한국어 답을 옮기기 전에 검사한다(바꾼 문장도 기존 번역을 지난다).
    generated_answer = answer
    answer, citations = guard_answer(safety_class if safety else None, answer, list(parsed.citations))
    generated_model = (metadata.get("generation_model") if isinstance(metadata, Mapping)
                       and metadata.get("quick_cache_hit") is not True and answer == generated_answer
                       and parsed.repair_path not in FAILED_REPAIR_PATHS and not review_failed else None)
    if review_failed or parsed.repair_path in FAILED_REPAIR_PATHS:
        citations = []
    if mixed is not None:
        # 기록 문장 먼저, 문서 문장은 뒤에. 문서 본문이 비면 근거가 없다고만 덧붙인다(기록 문장을 되풀이하지 않는다).
        body = "" if parsed.repair_path in FAILED_REPAIR_PATHS else answer.strip()  # 빈 본문의 대체 문장("답변을 생성하지 못했습니다")도 빈 답이다
        # 사용자가 지식 조각을 먼저 물었으면 문서 문장을 앞에 둔다(사례 18).
        answer = (body + "\n" + mixed.text if body and mixed.knowledge_first and query.response_mode == "quick"
                  else mixed.text + "\n" + (body or MIXED_NO_EVIDENCE))
    logger.info(
        "copilot answer normalized request_id=%s repair_path=%s citations=%d dropped=%d",
        query.request_id,
        parsed.repair_path,
        len(parsed.citations),
        len(parsed.dropped),
    )
    response = _success_response(
        query,
        request_started,
        answer=answer,
        retrieval_used=bool(use_rag and citations and not review_failed
                            and parsed.repair_path not in FAILED_REPAIR_PATHS),
        citations=citations,
        degraded=review_failed or parsed.repair_path in FAILED_REPAIR_PATHS or bool(result.get("deadline_hit", False)),
        coating_context=coating_advice_context,
        context_status="available" if session_summary or coating_advice_context or facts else "not_available",
        generated=True,
        model_name=generated_model,
        subject=mixed.subject if mixed is not None else None,
    )
    return _with_answer_trace(response, result)


async def execute_copilot_query(
    query: CopilotQueryRequest,
    *,
    pipeline_provider: PipelineProvider | None,
    pending_answers: Mapping[str, str] = PENDING_TASK_ANSWERS,
    sequences: SequenceStore,
    mcp_base_url: str,
    transport: object | None = None,
    started: float | None = None,
    reports: V1Store | None = None,
    sequence_engine: SequenceAnalysisEngine | None = None,
    coating_engine: object | None = None,
    conversation_id: str | None = None,
) -> JSONResponse:
    """Normalize foreign questions, keep Korean analysis, then localize only prose."""
    request_started = time.monotonic() if started is None else started
    response_language = query.response_language or query.language
    original_provider = pipeline_provider
    if original_provider is not None:
        # A per-request binding switches only the HTTP generator, never the shared model.
        pipeline_provider = lambda: select_query_pipeline(original_provider(), query.response_mode)
    pipeline: object | None = None
    last_translation_model: str | None = None

    async def _translate(text: str, source: str, target: str) -> str:
        nonlocal pipeline, last_translation_model
        if source == target:
            return text
        # These language pairs are more reliable from English than from Korean
        # on the existing 27B translator. Validate values and language on BOTH
        # legs; the Korean analysis and public response language stay unchanged.
        if source in {"ko", "auto"} and target in {"ne", "my", "si", "km"}:
            english = await _translate(text, source, "en")
            return await _translate(english, "en", target)
        if pipeline_provider is None:
            raise TranslationUnavailableError("Translation engine unavailable")
        if pipeline is None:
            pipeline = await to_thread.run_sync(pipeline_provider)
        translator = translation_pipeline_for(pipeline, source, target)
        def guarded() -> tuple[str, str | None]:
            generated = False
            def mark_generated() -> None:
                nonlocal generated
                generated = True
            with generation_slot_for(translator):
                translated = translate_text(translator, text, source_language=source, target_language=target,
                                            on_generated=mark_generated)
            return translated, query_model_name(translator) if generated else None
        translated, last_translation_model = await to_thread.run_sync(guarded)
        return translated

    try:
        translation_turns = await to_thread.run_sync(lambda: sequences.recent_query_turns(
            query.session_id or "CHAT:" + query.user_id, limit=6, user_id=query.user_id, conversation_key=conversation_id))
        translation_history = [{"role": "user", "content": turn["prompt"]} for turn in translation_turns]
        translation_request = parse_translation_request(query.prompt, recent_turns=translation_history)
        if translation_request is None:
            response_language = requested_response_language(query.prompt) or response_language
        glossary_context: dict[str, object] | None = None
        if translation_request is not None:
            # A translation request is not a request for a definition or a pose score.
            # Preserve the original source text; translating the question first can
            # erase the foreign term the user actually asked about.
            normalized = query.prompt
        else:
            try:
                normalized = await _translate(query.prompt, query.language, "ko")
            except Exception:
                # 질문 번역 실패 때 원문으로 손-팁 위험임을 알면 기존 고정 안전 답을 유지한다.
                if classify(query.prompt) != "hand_tip":
                    raise
                logger.warning("copilot question translation failed; hand-on-tip original request_id=%s", query.request_id)
                normalized = query.prompt
        if len(normalized) > 8000:
            raise TranslationUnavailableError("Translated question exceeds analysis limit")
        analysis_query = query.model_copy(update={"prompt": normalized})
        if query.language != "ko" and translation_request is None:
            # 질문 번역과 답 번역 가운데 어디가 틀렸는지 가릴 수 있게 분석에 쓴 한국어 질문을 남긴다.
            logger.info("copilot translated question request_id=%s language=%s ko=%r", query.request_id, query.language, normalized)
        # 위험 종류는 한국어 질문·원문으로 고르고, 짧은 후속 질문은 앞 턴에 저장한 종류를 이어받는다(다시 판정하지 않는다).
        previous_turn = translation_turns[-1] if translation_turns else {}
        safety_class = hazard_class((normalized, query.prompt if query.language != "ko" else ""),
                                    previous_class=previous_turn.get("safety_class"), previous_text=previous_turn.get("prompt", ""))
        if translation_request is not None:
            response_language = translation_request.target_language
            dictionary = lookup_translation_question(query.prompt, recent_turns=translation_history)
            source_language = (translation_request.source_language
                               or (dictionary.source_language if dictionary is not None else None)
                               or "auto")
            if dictionary is not None:
                translated_answer = dictionary.answer
                provenance = {"kind": dictionary.source_kind, **dictionary.provenance}
                answer_source = "glossary"
            else:
                pairs, hints = translation_references(translation_request.source_text, source_language, response_language)
                translated_answer = await _translate(translation_request.source_text, source_language, response_language)
                target_key = re.sub(r"\s+", "", translated_answer).casefold()
                applied = [hint for hint in [*pairs, *hints]
                           if re.sub(r"\s+", "", hint["target"]).casefold() in target_key]
                provenance = {"kind": "translation_hints", "reference_version": reference_version(), "applied": applied}
                answer_source = "glossary" if applied else "conversation"
            glossary_context = {"source_text": translation_request.source_text, "source_language": source_language,
                                "target_language": response_language, "provenance": provenance}
            # Explicit translation preserves the whole phrase and its measurements,
            # including Chinese characters. Normal Korean-generation checks remain
            # unchanged for all non-translation answers.
            response = _success_response(analysis_query.model_copy(update={"response_mode": "detailed"}), request_started,
                                         answer=translated_answer, retrieval_used=False, citations=[],
                                         generated=dictionary is None, model_name=last_translation_model)
            response.headers[ANSWER_SOURCE_HEADER] = answer_source
        else:
            response = await _execute_copilot_query(
                analysis_query, pipeline_provider=pipeline_provider, pending_answers=pending_answers,
                sequences=sequences, mcp_base_url=mcp_base_url, transport=transport,
                started=request_started, reports=reports, sequence_engine=sequence_engine,
                coating_engine=coating_engine, conversation_id=conversation_id,
                original_prompt=query.prompt if query.language != "ko" else None, safety_class=safety_class,
            )
        if response.status_code != 200:
            return response
        # A semantic follow-up can recover a hazard that the current utterance
        # did not name. Persist it so a third omitted-subject turn retains it.
        safety_class = getattr(response, "_copilot_safety_class", safety_class)
        body = json.loads(response.body)
        data = body["data"]
        # 분석 답은 항상 한국어다. 모델이 섞어 낸 한자는 번역 전에 뺀다.
        korean_answer = without_hanzi(data["answer"], prompt=normalized) if translation_request is None else None
        if query.summarize and korean_answer is not None:
            # All record/generation/safety guards have run. Present each cached
            # full body independently, before the existing translation checks.
            korean_answer = summarize_query_answer(korean_answer, safety=bool(safety_class),
                precomposed=bool(getattr(response, "_copilot_precomposed_answer", False)))
        try:
            if translation_request is None:
                last_translation_model = None
                data["answer"] = await _translate(korean_answer, "ko", response_language)
        except Exception:
            # 고정 고압 안전 답만 미리 옮겨 둔 문장으로 낸다(v3.4.5, 운영 우즈베크어 답 번역이 늘 실패). 다른 답은 503.
            prepared = fixed_answer_translation(korean_answer, response_language)
            if prepared is None:
                raise
            logger.warning("copilot answer translation failed; prepared safety answer request_id=%s language=%s",
                           query.request_id, response_language)
            data["answer"] = prepared
            last_translation_model = None
    except Exception:  # Translation failures never masquerade as a requested-language answer.
        logger.exception("copilot translation failed request_id=%s", query.request_id)
        return error_response(503, "Query translation unavailable; retry or use Korean")

    if {"language", "response_language"} & query.model_fields_set:
        data["language"] = query.language
        data["response_language"] = response_language
    else:
        # Legacy clients (including the separate rc1 gateway) validate the old
        # response strictly. Only opt-in multilingual calls add language metadata.
        data.pop("language", None)
        data.pop("response_language", None)
    data["latency_ms"] = max(0, int((time.monotonic() - request_started) * 1000))
    data["generated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    query_turn_id = None
    try:
        recent_context = sequences.recent_query_turns(
            query.session_id or "CHAT:" + query.user_id, limit=6,
            user_id=query.user_id, conversation_key=conversation_id,
        )
        query_turn_id = sequences.record_query_turn(
            session_key=query.session_id or "CHAT:" + query.user_id, user_id=query.user_id,
            prompt=query.prompt, answer=data["answer"], task_type=query.task_type,
            created_at=data["generated_at"], conversation_key=conversation_id, safety_class=safety_class,
            answer_source=response.headers.get(ANSWER_SOURCE_HEADER, "analysis"), subject=_header_subject(response),
            query_context={
                "scenario_id": data.get("scenario_id") or query.scenario_id,
                "session_id": query.session_id, "response_mode": query.response_mode,
                "summarize": query.summarize,
                "language": query.language, "response_language": response_language,
                "analysis_prompt": normalized, "context": query.context,
                "sequence_context": data.get("sequence_context"), "pose_context": data.get("pose_context"),
                "coating_context": data.get("coating_context"), "recent_turns": recent_context,
                "context_truncated": False,
                **({"glossary_context": glossary_context} if glossary_context is not None else {}),
                **({"answer_trace": response._copilot_answer_trace}
                   if hasattr(response, "_copilot_answer_trace") else {}),
            },
        )
    except Exception:
        logger.exception("copilot query: failed to store dialogue request_id=%s", query.request_id)
    headers = {ANSWER_SOURCE_HEADER: response.headers.get(ANSWER_SOURCE_HEADER, "analysis")}
    if (model_name := safe_model_name(response.headers.get(MODEL_HEADER))) is not None:
        headers[MODEL_HEADER] = model_name
    if last_translation_model is not None and last_translation_model != headers.get(MODEL_HEADER):
        headers[TRANSLATION_MODEL_HEADER] = last_translation_model
    if isinstance(query_turn_id, int) and not isinstance(query_turn_id, bool):
        headers["X-Copilot-Query-Turn"] = str(query_turn_id)
    if query.language != "ko" and translation_request is None:
        headers[ANALYSIS_PROMPT_HEADER] = analysis_prompt_header(normalized)
    return JSONResponse(status_code=200, content=body, headers=headers)


def create_copilot_router(
    pipeline_provider: PipelineProvider | None,
    *,
    pending_answers: Mapping[str, str] = PENDING_TASK_ANSWERS,
    store: SequenceStore | None = None,
    mcp_base_url: str | None = None,
    transport: object | None = None,
    reports: V1Store | None = None,
    sequence_engine: SequenceAnalysisEngine | None = None,
    coating_engine: object | None = None,
) -> APIRouter:
    """``pipeline_provider`` returns the live pipeline singleton; ``None`` when live mode is off.

    ``reports`` is the complete-report store; without it zone questions fall back to scoring one sequence.
    """
    router = APIRouter()
    sequences = store if store is not None else SequenceStore()
    mcp_base = mcp_base_url or os.environ.get("MCP_BASE_URL", "http://localhost:8100")

    @router.post(
        COPILOT_QUERY_PATH,
        response_model=CopilotResponse,
        tags=["3. AI Co-Pilot 질의"],
        summary="3-1. 텍스트 기반 Co-Pilot 질의",
        description=(
            "VR·Replay 및 일반 UI용 텍스트 질의입니다. response_mode로 모델을, summarize로 답변 분량을 선택합니다.\n\n"
            "- **빠른 안내 (`quick`, 기본값) · Qwen3-8B 모델**: 빠른 안내 모델을 사용합니다.\n"
            "  - 사용자·세션 맥락·검색 근거·요약 설정이 모두 같은 반복 질문은 최대 2분간 생성 결과를 재사용합니다.\n"
            "  - 새 시퀀스·대화·문맥·근거가 반영되면 다시 생성합니다. 새 질문의 처리 시간은 모델 부하에 따라 달라집니다.\n"
            "- **상세 분석 (`detailed`) · Qwen3.6-27B 모델**: 상세 분석 모델을 사용합니다.\n"
            "- **답변 요약 (`summarize`)**: 생략/true는 VR용 핵심 요약, false는 필요한 문단·목록을 유지한 전체 본문입니다. 모델 선택과 독립적이며 안전 경고·한계·수치·단위·Zone 순서는 보존합니다. 명시적인 번역 요청은 전체 문구·문장을 유지합니다.\n"
            "  - 질문에 맞춰 구성된 분석·기록의 사실/조치 안내와 고정 안전 답변은 필수 내용을 그대로 제공합니다.\n"
            "- 두 모드 모두 생성은 최대 1,024토큰으로 제한하며 인용·안전·수치 검사를 유지합니다.\n"
            "- **대화 맥락**: 일반 지식·일상 대화·계산·후속 질문도 처리합니다. 탭 선택이 질문 의도를 강제하지 않습니다. 도장 지식이 필요한 경우에 RAG를 사용합니다.\n"
            "  - `use_rag=Y`는 검색을 허용합니다. 적합한 문서 근거가 없으면 대화 맥락을 반영해 LLM이 답합니다. 안전 기준·훈련 실측값은 추측으로 대체하지 않습니다.\n"
            "  - `retrieval_used=true`와 `citations`는 답변에 실제 사용한 유효 NCS 근거를 뜻합니다. 검색을 시도했다는 의미가 아닙니다.\n"
            "  - 응답 헤더 `X-Copilot-Answer-Source`: `documents`(RAG), `conversation`(LLM), `analysis`(분석 안내), `glossary`(용어집). AI 질문 화면에도 실제 출처를 표시합니다.\n"
            "  - 응답 헤더 `X-Copilot-Model`: 이번 답변을 생성한 실제 모델 식별자입니다. 캐시·규칙·직접 용어집 답변이나 모델을 확인할 수 없는 경우에는 생략합니다. 답변 번역에 다른 모델을 실제 호출하면 `X-Copilot-Translation-Model`로 구분합니다.\n"
            "  - 선택 헤더 `X-Copilot-Conversation`: 채팅 창 식별자(영문·숫자·`._:-`, 최대 128자). 사용자·훈련 세션과 함께 대화 이력을 분리합니다. 생략하면 기본 대화를 사용합니다. 인증 수단은 아닙니다.\n"
            "  - 선택 헤더 `X-Copilot-Source`: `motion-viewer`(3D VIEWER), `assistant-test`(TEST). DB 표시용 출처 힌트이며 훈련 대상 작업자 ID와 별개입니다.\n"
            "- **훈련 전 기준 순서**: `scenario_id`가 확인되고 등록된 기준이 있으면 시퀀스·리포트 없이도 Zone 작업 순서를 안내합니다. 공정이 불명확하면 등록된 스프레이·블라스팅 기준을 구분합니다.\n"
            "  - 기준 순서 조회는 실제 방문·준수율·도막 완료 판정이 아닙니다. 현재 다음 Zone 이동 질문은 저장된 시퀀스와 도막 완료 상태로 판단합니다.\n"
            "- **질문 언어**: `language`로 지정합니다. 생략 시 `ko`입니다.\n"
            "- **답변 언어**: `response_language`로 지정하며, 생략/null이면 질문 언어를 따릅니다.\n"
            "  - 명시적인 용어 번역 질문은 검증된 용어집과 검토를 마친 TSB 용어를 우선 사용합니다. 없는 문구·문장 전체는 LLM으로 번역합니다. 용어집 근거는 NCS `citations`에 섞지 않습니다.\n"
            "- 외국어 질문을 한국어 분석·검색으로 연결하고 최종 `answer`를 요청 언어로 번역합니다.\n"
            "- 점수·단위·Zone 순서·구조화 context·citations는 유지합니다. 번역 처리 시간이 추가됩니다.\n"
            "  - 일부 언어는 번역 품질을 위해 기존 Qwen3.6-27B 모델을 사용합니다.\n"
            "- 번역에 실패하거나 수치가 달라지면 한국어를 외국어 응답으로 표시하지 않고 503을 반환합니다."
        ),
    )
    async def copilot_query(request: Request, query: Annotated[CopilotQueryRequest, Body(openapi_examples={
        "korean": {"summary": "한국어 질문·답변 (기존 호출 호환)", "value": {
            "user_id": "1234", "scenario_id": "3", "session_id": "3_11000346_1",
            "prompt": "스프레이 도장 시 적정 이격거리를 알려주세요.",
            "task_type": "knowledge_search", "response_mode": "quick", "summarize": True, "use_rag": "Y",
            "language": "ko", "response_language": "ko",
        }},
        "vietnamese": {"summary": "베트남어 질문 → 베트남어 답변", "value": {
            "user_id": "1234", "scenario_id": "3", "session_id": "3_11000346_1",
            "prompt": "Khoảng cách phun phù hợp là bao nhiêu?",
            "task_type": "knowledge_search", "response_mode": "quick", "summarize": True, "use_rag": "Y",
            "language": "vi", "response_language": "vi",
        }},
        "english_to_korean": {"summary": "영어 질문 → 한국어 답변", "value": {
            "user_id": "1234", "prompt": "How can I keep a consistent spray distance?",
            "task_type": "knowledge_search", "response_mode": "detailed", "summarize": True, "use_rag": "Y",
            "language": "en", "response_language": "ko",
        }},
    })]) -> JSONResponse:
        return await execute_copilot_query(
            query,
            conversation_id=conversation_key(request.headers.get("x-copilot-conversation")),
            pipeline_provider=pipeline_provider,
            pending_answers=pending_answers,
            sequences=sequences,
            mcp_base_url=mcp_base,
            transport=transport,
            reports=reports,
            sequence_engine=sequence_engine,
            coating_engine=coating_engine,
        )

    return router


def install_copilot_error_handlers(app: FastAPI) -> None:
    """Wraps request validation errors under ``/api/v1/`` in the v2.5 ERROR envelope."""

    async def _on_validation_error(request: Request, exc: Exception) -> JSONResponse:
        assert isinstance(exc, RequestValidationError)
        if request.url.path.startswith(_API_V1_PREFIX):
            logger.info(
                "copilot request rejected path=%s errors=%s",
                request.url.path,
                [(error.get("loc"), error.get("type")) for error in exc.errors()],
            )
            # Only stable metadata crosses the boundary; errors.input/ctx can
            # contain transcripts, credentials, or custom exception contents.
            first = exc.errors()[0] if exc.errors() else {}
            path = ".".join(re.sub(r"[^A-Za-z0-9_\-]", "_", str(item))[:40]
                            for item in first.get("loc", ())[:5])[:120]
            reasons = {
                "missing": "Field required", "extra_forbidden": "Extra field is not permitted",
                "literal_error": "Unsupported value", "enum": "Unsupported value",
                "string_too_short": "Value is too short", "string_too_long": "Value is too long",
                "int_parsing": "Integer required", "float_parsing": "Number required",
                "json_invalid": "Malformed JSON", "model_attributes_type": "Object required",
                "greater_than": "Value is below the allowed range",
                "greater_than_equal": "Value is below the allowed range",
                "less_than": "Value is above the allowed range",
                "less_than_equal": "Value is above the allowed range",
            }
            reason = reasons.get(str(first.get("type")), "Invalid value")
            return error_response(422, f"Invalid request: {path or 'request'} ({reason})")
        return await request_validation_exception_handler(request, exc)

    app.add_exception_handler(RequestValidationError, _on_validation_error)
