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
from trainer.demo_api.v1_contracts import ApiResponse
from trainer.demo_api.v1_store import V1Store

_PREFIX = "/api/v1/test/workflow"
_TAG = "0. Swagger 모의 연동"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ok(data: dict[str, Any]) -> JSONResponse:
    return JSONResponse(content=ApiResponse(data=data).model_dump(mode="json"))


@dataclass
class _DemoRun:
    run_id: str
    user_id: str
    session_id: str
    scenario_id: str = "training-mode-1"
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
                mode_id=1,
                sequence_setting={"target_speed_mms": 500, "target_distance_mm": 300},
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
        description="입력값 없이 새 모의 훈련자와 세션 식별자를 준비합니다. 실제 훈련 데이터는 삭제하지 않으며, 아래 Execute 버튼을 누른 뒤 응답의 next_endpoint를 따라가면 됩니다.",
    )
    def reset() -> JSONResponse:
        run = workflow.reset()
        return _ok({"workflow_status":"ready","step":0,"user_id":run.user_id,"session_id":run.session_id,
                    "what_happened":"모의연동 전용 식별자를 새로 만들었습니다. 실제 사용자 기록은 지우지 않았습니다.",
                    "beginner_tip":"다음으로 0-1 세션 시작의 Try it out과 Execute를 누르세요.","next_endpoint":f"{_PREFIX}/1-start"})

    @router.post(
        "/1-start", response_model=ApiResponse, tags=[_TAG], summary="0-1. 모의 훈련 세션 시작",
        description="파라미터를 입력하지 않아도 초보자 계정과 ACTIVE 세션을 DB에 생성합니다. 초기화를 건너뛰어도 필요한 값을 자동 준비하므로 바로 Execute해도 안전합니다.",
    )
    def start() -> JSONResponse:
        run = workflow.start()
        return _ok({"workflow_status":"running","step":1,"user_id":run.user_id,"session_id":run.session_id,
                    "session_status":"ACTIVE","what_happened":"실제 sessions/start와 같은 상태 전이를 수행해 ACTIVE 세션을 만들었습니다.",
                    "beginner_tip":"다음으로 0-2 시퀀스 분석을 실행해 구조화된 가이드를 확인하세요.","next_endpoint":f"{_PREFIX}/2-sequence"})

    @router.post(
        "/2-sequence", response_model=ApiResponse, tags=[_TAG], summary="0-2. 예제 시퀀스 분석 및 코칭 확인",
        description="훈련모드 1의 Zone 1:1을 가정한 현재 시퀀스를 전문가 기준과 비교합니다. 응답의 guide.speed와 guide.distance가 VR 화면에서 사용할 구조화된 판정 예시입니다.",
    )
    def sequence() -> JSONResponse:
        run, guide = workflow.guide()
        return _ok({"workflow_status":"running","step":2,"session_id":run.session_id,
                    "sequence_id":f"SWAGGER-SEQUENCE-{run.run_id}","guide":guide.model_dump(mode="json"),
                    "what_happened":"예제 속도와 이격거리를 비교하여 VR에서 바로 사용할 코칭 상태를 만들었습니다.",
                    "beginner_tip":"slow이면 속도를 높이고 near이면 피도면에서 조금 멀어지라는 뜻입니다.","next_endpoint":f"{_PREFIX}/3-complete"})

    @router.post(
        "/3-complete", response_model=ApiResponse, tags=[_TAG], summary="0-3. 모의 훈련 종료 및 리포트 요청",
        description="필요한 이전 단계를 자동 보완하고 세션을 COMPLETED로 종료한 뒤 report_id를 발급합니다. 같은 모의 실행에서 반복해도 새 리포트를 계속 만들지 않는 멱등 동작입니다.",
    )
    def complete() -> JSONResponse:
        run = workflow.complete()
        return _ok({"workflow_status":"report_ready","step":3,"session_id":run.session_id,
                    "session_status":"COMPLETED","report_id":run.report_id,
                    "what_happened":"모의 세션을 종료하고 최종 평가를 조회할 report_id를 생성했습니다.",
                    "beginner_tip":"마지막으로 0-4 리포트 확인을 실행하세요.","next_endpoint":f"{_PREFIX}/4-report"})

    @router.post(
        "/4-report", response_model=ApiResponse, tags=[_TAG], summary="0-4. 모의 최종 리포트 확인",
        description="모의연동의 마지막 단계입니다. 이전 버튼을 누르지 않았더라도 전제 단계를 자동 실행하고, 화면에 표시할 수 있는 서술형 종합평가와 개선 행동을 반환합니다.",
    )
    def report() -> JSONResponse:
        run = workflow.complete()
        return _ok({"workflow_status":"complete","step":4,"session_id":run.session_id,"report_id":run.report_id,
                    "overall_score":82,
                    "narrative_evaluation":"전체 도장 흐름은 안정적이지만 기준보다 이동 속도가 느리고 피도면에 다소 가깝습니다. 다음 훈련에서는 손목 각도를 유지한 채 일정한 속도로 이동하고, 권장 이격거리를 먼저 확보한 뒤 겹침 폭을 일정하게 유지해 보세요.",
                    "strengths":["세션 절차를 끝까지 완료했습니다.","분사 방향이 예제 경로와 일치합니다."],
                    "improvements":["이동 속도를 조금 높이세요.","피도면에서 조금 멀리 이동하세요."],
                    "what_happened":"모의연동 전 과정을 완료하고 UI에 표시할 서술형 평가를 반환했습니다.",
                    "beginner_tip":"다시 시험하려면 0-0 초기화부터 반복하세요.","next_endpoint":f"{_PREFIX}/0-reset"})

    return router, workflow
