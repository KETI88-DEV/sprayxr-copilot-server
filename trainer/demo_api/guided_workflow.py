"""Input-free Swagger walkthrough for people who do not know the simulator."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from trainer.demo_api.expert_guidance import ExpertGuidanceEngine
from trainer.demo_api.telemetry_contracts import AnalyzeSequenceRequest, GuideItem, SequenceGuide
from trainer.demo_api.v1_contracts import ApiResponse, report_example
from trainer.demo_api.v1_store import V1Store

_PREFIX = "/api/v1/test/workflow"
_TAG = "0. Swagger 모의 연동"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ok(data: dict[str, Any]) -> JSONResponse:
    return JSONResponse(content=ApiResponse(data=data).model_dump(mode="json"))


def _test_context(
    input_used: dict[str, Any], input_summary: str, evaluation_focus: str
) -> dict[str, Any]:
    """Explain an input-free demo call without changing the real API contract."""
    return {
        "input_used": input_used,
        "input_summary": input_summary,
        "evaluation_focus": evaluation_focus,
    }


def _swagger_response(data: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """Expose a concrete result in Swagger instead of only the generic envelope."""
    return {
        200: {
            "description": (
                "성공. `data.test_context.input_used`는 서버가 자동으로 사용한 입력값, "
                "`input_summary`는 입력의 의미, `evaluation_focus`는 결과에서 확인할 항목입니다."
            ),
            "content": {
                "application/json": {
                    "example": ApiResponse(data=data).model_dump(mode="json")
                }
            },
        }
    }


_RESET_CONTEXT = _test_context(
    {
        "request_parameters": "없음",
        "server_generated": {
            "user_id": "SWAGGER-DEMO-a1b2c3d4",
            "session_id": "SWAGGER-SESSION-a1b2c3d4",
            "scenario_id": "1",
        },
    },
    "사용자가 입력한 값은 없으며 서버가 모의연동 전용 사용자·세션 식별자를 자동 생성합니다.",
    "workflow_status가 ready인지, 생성된 user_id와 session_id가 무엇인지 확인합니다.",
)

_START_CONTEXT = _test_context(
    {
        "equivalent_real_api": "POST /api/v1/sessions/start",
        "body": {
            "user_id": "SWAGGER-DEMO-a1b2c3d4",
            "session_id": "SWAGGER-SESSION-a1b2c3d4",
            "scenario_id": "1",
            "worker": {"skill_level": "novice", "handedness": "right"},
        },
    },
    "초보자 작업자가 훈련모드 1을 시작하는 것과 같은 입력을 서버가 자동 구성합니다.",
    "session_status가 ACTIVE로 바뀌고 동일한 session_id가 유지되는지 확인합니다.",
)

_SEQUENCE_INPUT = {
    "equivalent_real_api": "POST /api/v1/copilot/analyze-sequence",
    "session_id": "SWAGGER-SESSION-a1b2c3d4",
    "user_id": "SWAGGER-DEMO-a1b2c3d4",
    "scenario_id": "1",
    "zone_id": "1",
    "sequence_id": "SWAGGER-SEQUENCE-a1b2c3d4",
    "sequence_setting": {
        "tool_type": "spray_gun",
        "target_speed_mms": 500,
        "target_distance_mm": 300,
    },
    "sequence_info": [
        {
            "index": 0,
            "tool_pos_rot": {"position": [0.0, 1.2, 0.4]},
            "gun_metrics": {"speed": 300, "distance": 250},
        }
    ],
    "sequence_result": {"avg_speed_mms": 300, "avg_distance_mm": 250},
}
_SEQUENCE_CONTEXT = _test_context(
    _SEQUENCE_INPUT,
    "목표 속도 500 mm/s보다 느린 300 mm/s, 목표 이격거리 300 mm보다 가까운 250 mm의 예제입니다.",
    "guide.speed.status가 slow이고 guide.distance.status가 near인지 확인합니다.",
)

_COMPLETE_CONTEXT = _test_context(
    {
        "equivalent_real_api": "POST /api/v1/sessions/complete",
        "session_id": "SWAGGER-SESSION-a1b2c3d4",
        "completion_payload": {
            "source": "swagger-guided-test",
            "vr_score": 82,
            "is_pass": True,
            "avg_speed_mms": 300,
            "avg_distance_mm": 250,
        },
    },
    "앞 단계의 모의 세션을 종료하고 예제 훈련 결과로 최종 리포트 생성을 요청합니다.",
    "session_status가 COMPLETED인지와 발급된 report_id를 확인합니다.",
)

_REPORT_CONTEXT = _test_context(
    {
        "equivalent_real_api": "GET /api/v1/reports/{report_id}",
        "report_id": "RPT-example",
    },
    "0-3에서 생성한 report_id로 정식 리포트(GET /api/v1/reports/{report_id})와 같은 모양의 완성 리포트를 받습니다.",
    "report 아래 coating_report·sequence_report·pose_report·llm_briefing 네 항목의 구조를 확인합니다. 값은 UI 설계용 예시입니다.",
)


def _mock_report(run: _DemoRun | None = None) -> dict[str, Any]:
    """정식 리포트와 같은 모양. 모의 세션의 식별자만 바꿔 넣는다."""
    detail = report_example()
    if run is not None:
        detail.update(session_id=run.session_id, user_id=run.user_id, scenario_id=run.scenario_id)
        detail["report"]["report_id"] = run.report_id
    return detail


@dataclass
class _DemoRun:
    run_id: str
    user_id: str
    session_id: str
    scenario_id: str = "1"
    started: bool = False
    report_id: str | None = None


class GuidedWorkflow:
    """Small idempotent state machine backing the five Swagger buttons."""

    def __init__(self, store: V1Store, guidance: ExpertGuidanceEngine) -> None:
        self._store = store
        self._guidance = guidance
        self._guard = threading.RLock()
        self._run: _DemoRun | None = None

    def reset(self) -> _DemoRun:
        with self._guard:
            suffix = uuid.uuid4().hex[:8]
            self._run = _DemoRun(
                run_id=suffix,
                user_id=f"SWAGGER-DEMO-{suffix}",
                session_id=f"SWAGGER-SESSION-{suffix}",
            )
            return self._run

    def current(self) -> _DemoRun:
        with self._guard:
            return self._run if self._run is not None else self.reset()

    def start(self) -> _DemoRun:
        with self._guard:
            run = self.current()
            self._store.upsert_user(
                user_id=run.user_id,
                name="Swagger 초보자 모의 훈련자",
                skill_level="novice",
                dominant_hand="right",
                created_at=_now(),
            )
            self._store.start_session(
                session_id=run.session_id,
                user_id=run.user_id,
                scenario_id=run.scenario_id,
                started_at=_now(),
            )
            run.started = True
            return run

    def guide(self) -> tuple[_DemoRun, SequenceGuide]:
        with self._guard:
            run = self.start()
            request = AnalyzeSequenceRequest(
                session_id=run.session_id,
                user_id=run.user_id,
                scenario_id=run.scenario_id,
                zone_id="1:1",
                sequence_id=f"SWAGGER-SEQUENCE-{run.run_id}",
                sequence_setting={
                    "tool_type": "spray_gun",
                    "target_speed_mms": 500,
                    "target_distance_mm": 300,
                },
                sequence_info=[
                    {
                        "index": 0,
                        "time": _now(),
                        "tool_pos_rot": {"position": [0.0, 1.2, 0.4]},
                        "gun_metrics": {"speed": 300, "distance": 250},
                    }
                ],
                sequence_result={"avg_speed_mms": 300, "avg_distance_mm": 250},
            )
            guide = self._guidance.evaluate(request)
            if not guide.speed.available or not guide.distance.available:
                guide = SequenceGuide(
                    speed=GuideItem(
                        status="slow", message="이동 속도를 조금 높이세요.", available=True, reason=None
                    ),
                    distance=GuideItem(
                        status="near",
                        message="스프레이건을 피도면에서 조금 멀리 이동하세요.",
                        available=True,
                        reason=None,
                    ),
                )
            return run, guide

    def complete(self) -> _DemoRun:
        with self._guard:
            run = self.start()
            self._store.end_session(
                session_id=run.session_id,
                user_id=run.user_id,
                scenario_id=run.scenario_id,
                ended_at=_now(),
            )
            if run.report_id is None:
                completion = self._store.create_completion(
                    session_id=run.session_id,
                    requested_at=_now(),
                    payload={
                        "source": "swagger-guided-test",
                        "vr_score": 82,
                        "is_pass": True,
                        "avg_speed_mms": 300,
                        "avg_distance_mm": 250,
                    },
                )
                run.report_id = completion.report_id
            return run


def create_guided_workflow_router(
    store: V1Store, guidance: ExpertGuidanceEngine
) -> tuple[APIRouter, GuidedWorkflow]:
    router = APIRouter(prefix=_PREFIX)
    workflow = GuidedWorkflow(store, guidance)

    @router.post(
        "/0-reset", response_model=ApiResponse, tags=[_TAG],
        summary="0-0. 모의연동 초기화 — 가장 먼저 누르세요",
        description=(
            "입력값 없이 새 모의 훈련자와 세션 식별자를 준비합니다. 실제 훈련 데이터는 삭제하지 않습니다.\n\n"
            "### 자동 사용 입력\n사용자 입력은 없고 `user_id`, `session_id`, `scenario_id=1`을 서버가 생성합니다.\n\n"
            "### 결과 읽는 법\n응답의 `data.test_context.input_used`에서 생성값을 확인하고 `next_endpoint`를 따라가세요."
        ),
        responses=_swagger_response({"test_context": _RESET_CONTEXT, "workflow_status": "ready", "step": 0,
                                     "user_id": "SWAGGER-DEMO-a1b2c3d4", "session_id": "SWAGGER-SESSION-a1b2c3d4",
                                     "what_happened": "모의연동 전용 식별자를 새로 만들었습니다.",
                                     "beginner_tip": "다음으로 0-1을 실행하세요.", "next_endpoint": f"{_PREFIX}/1-start"}),
    )
    def reset() -> JSONResponse:
        run = workflow.reset()
        context = _test_context(
            {"request_parameters": "없음", "server_generated": {
                "user_id": run.user_id, "session_id": run.session_id, "scenario_id": run.scenario_id}},
            _RESET_CONTEXT["input_summary"], _RESET_CONTEXT["evaluation_focus"])
        return _ok({"test_context":context,"workflow_status":"ready","step":0,"user_id":run.user_id,"session_id":run.session_id,
                    "what_happened":"모의연동 전용 식별자를 새로 만들었습니다. 실제 사용자 기록은 지우지 않았습니다.",
                    "beginner_tip":"다음으로 0-1 세션 시작의 Try it out과 Execute를 누르세요.","next_endpoint":f"{_PREFIX}/1-start"})

    @router.post(
        "/1-start", response_model=ApiResponse, tags=[_TAG], summary="0-1. 모의 훈련 세션 시작",
        description=(
            "파라미터 없이 초보자 계정과 ACTIVE 세션을 DB에 생성합니다.\n\n"
            "### 자동 사용 입력\n실제 `POST /api/v1/sessions/start`에 대응하는 사용자·세션·훈련모드·작업자 정보를 구성합니다.\n\n"
            "### 결과 읽는 법\n`input_used.body`와 `session_status=ACTIVE`를 나란히 비교하세요. 초기화를 건너뛰어도 전제값을 자동 준비합니다."
        ),
        responses=_swagger_response({"test_context": _START_CONTEXT, "workflow_status": "running", "step": 1,
                                     "session_status": "ACTIVE", "what_happened": "ACTIVE 세션을 만들었습니다.",
                                     "beginner_tip": "다음으로 0-2를 실행하세요.", "next_endpoint": f"{_PREFIX}/2-sequence"}),
    )
    def start() -> JSONResponse:
        run = workflow.start()
        input_used = dict(_START_CONTEXT["input_used"])
        input_used["body"] = {"user_id": run.user_id, "session_id": run.session_id,
                              "scenario_id": run.scenario_id,
                              "worker": {"skill_level": "novice", "handedness": "right"}}
        return _ok({"test_context":_test_context(input_used, _START_CONTEXT["input_summary"], _START_CONTEXT["evaluation_focus"]),
                    "workflow_status":"running","step":1,"user_id":run.user_id,"session_id":run.session_id,
                    "session_status":"ACTIVE","what_happened":"실제 sessions/start와 같은 상태 전이를 수행해 ACTIVE 세션을 만들었습니다.",
                    "beginner_tip":"다음으로 0-2 시퀀스 분석을 실행해 구조화된 가이드를 확인하세요.","next_endpoint":f"{_PREFIX}/2-sequence"})

    @router.post(
        "/2-sequence", response_model=ApiResponse, tags=[_TAG], summary="0-2. 예제 시퀀스 분석 및 코칭 확인",
        description=(
            "훈련모드 1의 Zone 1:1 시퀀스를 전문가 기준과 비교합니다.\n\n"
            "### 자동 사용 입력\n목표/실제 속도는 `500/300 mm/s`, 목표/실제 이격거리는 `300/250 mm`입니다. "
            "전체 본문은 응답의 `test_context.input_used`에 표시됩니다.\n\n"
            "### 예상 결과\n실제 속도가 목표보다 낮아 `slow`, 실제 거리가 목표보다 짧아 `near`가 되는 과정을 확인하세요."
        ),
        responses=_swagger_response({"test_context": _SEQUENCE_CONTEXT, "workflow_status": "running", "step": 2,
                                     "guide": {"speed": {"status": "slow", "message": "이동 속도를 조금 높이세요."},
                                               "distance": {"status": "near", "message": "피도면에서 조금 멀어지세요."}},
                                     "what_happened": "예제 입력을 비교해 코칭 상태를 만들었습니다.",
                                     "beginner_tip": "입력값과 status를 비교하세요.", "next_endpoint": f"{_PREFIX}/3-complete"}),
    )
    def sequence() -> JSONResponse:
        run, guide = workflow.guide()
        input_used = dict(_SEQUENCE_INPUT)
        input_used.update({"session_id": run.session_id, "user_id": run.user_id,
                           "scenario_id": run.scenario_id,
                           "sequence_id": f"SWAGGER-SEQUENCE-{run.run_id}"})
        return _ok({"test_context":_test_context(input_used, _SEQUENCE_CONTEXT["input_summary"], _SEQUENCE_CONTEXT["evaluation_focus"]),
                    "workflow_status":"running","step":2,"session_id":run.session_id,
                    "sequence_id":f"SWAGGER-SEQUENCE-{run.run_id}","guide":guide.model_dump(mode="json"),
                    "what_happened":"예제 속도와 이격거리를 비교하여 VR에서 바로 사용할 코칭 상태를 만들었습니다.",
                    "beginner_tip":"slow이면 속도를 높이고 near이면 피도면에서 조금 멀어지라는 뜻입니다.","next_endpoint":f"{_PREFIX}/3-complete"})

    @router.post(
        "/3-complete", response_model=ApiResponse, tags=[_TAG], summary="0-3. 모의 훈련 종료 및 리포트 요청",
        description=(
            "세션을 COMPLETED로 종료하고 report_id를 발급합니다.\n\n"
            "### 자동 사용 입력\n예제 VR 점수 82점, 평균 속도 300 mm/s, 평균 이격거리 250 mm를 완료 데이터로 사용합니다.\n\n"
            "### 결과 읽는 법\n`test_context.input_used.completion_payload`와 발급된 `report_id`를 확인하세요. 반복 실행해도 같은 리포트를 유지합니다."
        ),
        responses=_swagger_response({"test_context": _COMPLETE_CONTEXT, "workflow_status": "report_ready", "step": 3,
                                     "session_status": "COMPLETED", "report_id": "RPT-example",
                                     "what_happened": "세션을 종료하고 report_id를 만들었습니다.",
                                     "beginner_tip": "0-4에서 리포트를 확인하세요.", "next_endpoint": f"{_PREFIX}/4-report"}),
    )
    def complete() -> JSONResponse:
        run = workflow.complete()
        input_used = dict(_COMPLETE_CONTEXT["input_used"])
        input_used["session_id"] = run.session_id
        return _ok({"test_context":_test_context(input_used, _COMPLETE_CONTEXT["input_summary"], _COMPLETE_CONTEXT["evaluation_focus"]),
                    "workflow_status":"report_ready","step":3,"session_id":run.session_id,
                    "session_status":"COMPLETED","report_id":run.report_id,
                    "what_happened":"모의 세션을 종료하고 최종 평가를 조회할 report_id를 생성했습니다.",
                    "beginner_tip":"마지막으로 0-4 리포트 확인을 실행하세요.","next_endpoint":f"{_PREFIX}/4-report"})

    @router.post(
        "/4-report", response_model=ApiResponse, tags=[_TAG], summary="0-4. 모의 최종 리포트 확인",
        description=(
            "모의연동의 마지막 단계로 정식 리포트(`GET /api/v1/reports/{report_id}`)와 **같은 모양**의 완성 리포트를 반환합니다.\n\n"
            "### 자동 사용 입력\n0-3에서 생성한 report_id를 사용합니다. 파라미터 입력은 필요 없습니다.\n\n"
            "### 결과 읽는 법\n`report` 아래 `coating_report`(도막)·`sequence_report`(작업 순서·방향)·`pose_report`(자세)·`llm_briefing`(종합 서술)이 최종 리포트 구조입니다. "
            "값은 UI 설계용 예시입니다. `test_context`·`workflow_status` 등은 모의연동 안내용으로만 덧붙습니다. 이전 단계는 필요 시 자동 실행됩니다."
        ),
        responses=_swagger_response({"test_context": _REPORT_CONTEXT, "workflow_status": "complete", "step": 4,
                                     **_mock_report(), "report_id": "RPT-example",
                                     "what_happened": "정식 리포트와 같은 모양의 완성 리포트를 반환했습니다.",
                                     "beginner_tip": "report 아래 네 항목을 확인하세요.", "next_endpoint": f"{_PREFIX}/0-reset"}),
    )
    def report() -> JSONResponse:
        run = workflow.complete()
        input_used = dict(_REPORT_CONTEXT["input_used"])
        input_used["report_id"] = run.report_id
        return _ok({"test_context":_test_context(input_used, _REPORT_CONTEXT["input_summary"], _REPORT_CONTEXT["evaluation_focus"]),
                    "workflow_status":"complete","step":4,
                    **_mock_report(run), "report_id":run.report_id,
                    "what_happened":"모의연동 전 과정을 완료하고 정식 리포트와 같은 모양의 완성 리포트를 반환했습니다.",
                    "beginner_tip":"다시 시험하려면 0-0 초기화부터 반복하세요.","next_endpoint":f"{_PREFIX}/0-reset"})

    return router, workflow
