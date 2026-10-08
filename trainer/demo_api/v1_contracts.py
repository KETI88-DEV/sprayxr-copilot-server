"""v2.5 공개 API 계약 — 세션·사용자·리포트·테스트.

정본은 `docs/XR Painting Simulator AI Co-Pilot Service API 정의서 - 대화형 API 명세서.pdf`
(76쪽, 엔드포인트 15개)다. 쪽 번호를 각 모델에 적어 둔다.

명세서 p.25의 `APIResponse`가 모든 응답의 봉투다 — `status`/`code`/`message`/`data`이고
`data`는 `(any | null)`이다. 즉 명세서는 `data` **내용을 규정하지 않는다**. 각 엔드포인트의
`data` 모양은 우리가 정하고 여기에 적는다.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from trainer.demo_api.contracts import StrictModel

SkillLevel = Literal["expert", "novice"]
Handedness = Literal["right", "left"]
ReportStatus = Literal["pending", "complete", "failed"]
AnalysisStatus = Literal["available", "unavailable"]


class ApiResponse(BaseModel):
    """명세서 p.25 `APIResponse`. `data`는 엔드포인트마다 모양이 다르므로 열어 둔다."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["SUCCESS", "ERROR"] = "SUCCESS"
    code: int = 200
    message: str = "Operation completed successfully"
    data: Any | None = None


# --- 1. 세션 관리 (p.1-3) --------------------------------------------------


class WorkerProfile(StrictModel):
    """p.1: `skill_level`·`handedness` 모두 필수. `handedness` 기본값은 `right`다."""

    skill_level: SkillLevel
    handedness: Handedness = "right"


class WorkerProfilePartial(StrictModel):
    """p.3: `/sessions/end`의 `worker`는 선택이고 하위도 선택이다."""

    skill_level: SkillLevel | None = None
    handedness: Handedness | None = None


class SessionStartRequest(StrictModel):
    """p.1. `user_id`가 DB에 없으면 신규 생성, 있으면 숙련도·파지손을 갱신한다."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"example": {
            "session_id": "3_11000346_1", "user_id": "1234",
            "scenario_id": "3", "started_at": "2026-10-08T01:05:44.726Z",
            "worker": {"skill_level": "novice"},
        }},
    )

    session_id: Annotated[
        str,
        Field(
            min_length=1,
            description=(
                "TSB 시뮬레이터가 생성한 훈련 세션 식별자입니다. 완료 시 첨부되는 "
                "CSV 파일명에서 확장자를 뺀 값과 동일하게 사용합니다."
            ),
            examples=["3_11000346_1"],
        ),
    ]
    user_id: Annotated[str, Field(min_length=1, description="훈련자 식별자", examples=["1234"])]
    scenario_id: Annotated[
        str,
        Field(
            min_length=1,
            description=(
                "session_id의 첫 '_' 앞 콘텐츠 식별자입니다. 학습은 1~10, 실전1은 "
                "1-R-A·1-V-AB·1-E-ABC, 실전2는 2-A·2-ABCDE, Unit은 Unit12 형식입니다."
            ),
            examples=["3", "1-R-A", "2-ABCDE", "Unit12"],
        ),
    ]
    started_at: Annotated[str, Field(min_length=1, description="세션 시작 시각(ISO 8601 UTC)", examples=["2026-10-08T01:05:44.726Z"])]
    worker: WorkerProfile


class SessionEndRequest(StrictModel):
    """p.3. 세션을 `COMPLETED`로 전환하고 종료 시각을 기록한다."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"example": {
            "session_id": "3_11000346_1", "user_id": "1234",
            "scenario_id": "3", "ended_at": "2026-10-08T01:08:11.671Z",
        }},
    )

    session_id: Annotated[str, Field(min_length=1, description="sessions/start에서 사용한 동일 세션 식별자", examples=["3_11000346_1"])]
    user_id: Annotated[str, Field(min_length=1, description="sessions/start에서 사용한 동일 훈련자 식별자", examples=["1234"])]
    scenario_id: Annotated[str, Field(min_length=1, description="sessions/start에서 사용한 동일 콘텐츠 식별자", examples=["3"])]
    ended_at: Annotated[str, Field(min_length=1, description="세션 종료 시각(ISO 8601 UTC)", examples=["2026-10-08T01:08:11.671Z"])]
    worker: WorkerProfilePartial | None = None


# --- 6. 사용자 관리 (p.17-20) ---------------------------------------------


class UserCreateRequest(StrictModel):
    """p.18. `user_id`만 필수다."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"example": {
            "user_id": "1234", "name": "테스트 훈련자", "skill_level": "novice",
            "dominant_hand": "right", "age": 35, "gender": "male",
        }},
    )

    user_id: Annotated[str, Field(min_length=1)]
    name: str | None = None
    skill_level: SkillLevel = "novice"
    dominant_hand: Handedness = "right"
    age: int | None = None
    gender: str | None = None


class UserProfile(StrictModel):
    user_id: str
    name: str | None
    skill_level: SkillLevel
    # Migrated legacy records may not have captured handedness.  Keep the
    # stricter right/left contract for writes while faithfully exposing that
    # historical absence on reads.
    dominant_hand: Handedness | Literal["unknown"]
    age: int | None
    gender: str | None


# --- 5. 리포트 (p.11-16) --------------------------------------------------


class ReportArea(StrictModel):
    """p.15-16의 분석 영역 공통 머리.

    명세서: "실제 분석이 아직 없으면 상태는 `pending`, 각 분석 영역은 `unavailable`로
    반환됩니다." 그래서 `analysis_status`가 `unavailable`일 때 수치는 `null`이다 —
    임의 값을 만들지 않는다.
    """

    analysis_status: AnalysisStatus
    algorithm_version: str | None = None
    guideline: str | None = None


class CoatingReport(ReportArea):
    coating_distribution_index: float | None = None


class SequenceReport(ReportArea):
    adherence_rate: float | None = None
    direction_adherence_rate: float | None = None


class PoseReport(ReportArea):
    average_similarity_score: float | None = None


class ReportDetail(StrictModel):
    """p.15-16. `GET /reports/{report_id}`의 `data`."""

    report_id: str
    session_id: str
    report_status: ReportStatus
    generated_at: str
    llm_briefing: str | None
    coating_report: CoatingReport
    sequence_report: SequenceReport
    pose_report: PoseReport


class ReportStatusData(StrictModel):
    """p.14. 폴링 조회의 `data`. 2~5초 간격으로 부른다."""

    report_id: str
    session_id: str
    report_status: ReportStatus
    requested_at: str


class SessionCompleteData(StrictModel):
    """p.11-12. `report_id`를 부여하고 `pending`으로 시작한다.

    명세서: "현재 `pending`은 요청과 heatmap 요약이 저장되었지만 실제 최종 분석 결과는
    아직 생성되지 않았음을 뜻합니다. 실제 분석 작업이 연결되기 전에는 `complete`로
    표시하지 않습니다."
    """

    report_id: str
    session_id: str
    report_status: ReportStatus
    telemetry_csv_bytes: int
    height_map_image_bytes: int | None


class SessionStateData(StrictModel):
    """`/sessions/start`·`/sessions/end`의 `data`."""

    session_id: str
    user_id: str
    scenario_id: str
    session_status: Literal["ACTIVE", "COMPLETED"]
    started_at: str | None
    ended_at: str | None
    user_created: bool
