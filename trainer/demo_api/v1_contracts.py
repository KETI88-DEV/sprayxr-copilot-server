"""v2.5 공개 API 계약 — 세션·사용자·리포트·테스트.

정본은 `docs/XR Painting Simulator AI Co-Pilot Service API 정의서 - 대화형 API 명세서.pdf`
(76쪽, 엔드포인트 15개)다. 쪽 번호를 각 모델에 적어 둔다.

명세서 p.25의 `APIResponse`가 모든 응답의 봉투다 — `status`/`code`/`message`/`data`이고
`data`는 `(any | null)`이다. 즉 명세서는 `data` **내용을 규정하지 않는다**. 각 엔드포인트의
`data` 모양은 우리가 정하고 여기에 적는다.
"""

from __future__ import annotations

import json
from pathlib import Path
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
            "session_id": "TSB-DEMO-001", "user_id": "W-017",
            "scenario_id": "training-mode-3", "started_at": "2026-10-06T10:00:00Z",
            "worker": {"skill_level": "novice", "handedness": "right"},
        }},
    )

    session_id: Annotated[str, Field(min_length=1)]
    user_id: Annotated[str, Field(min_length=1)]
    scenario_id: Annotated[str, Field(min_length=1)]
    started_at: Annotated[str, Field(min_length=1)]
    worker: WorkerProfile


class SessionEndRequest(StrictModel):
    """p.3. 세션을 `COMPLETED`로 전환하고 종료 시각을 기록한다."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"example": {
            "session_id": "TSB-DEMO-001", "user_id": "W-017",
            "scenario_id": "training-mode-3", "ended_at": "2026-10-06T10:15:00Z",
            "worker": {"skill_level": "novice", "handedness": "right"},
        }},
    )

    session_id: Annotated[str, Field(min_length=1)]
    user_id: Annotated[str, Field(min_length=1)]
    scenario_id: Annotated[str, Field(min_length=1)]
    ended_at: Annotated[str, Field(min_length=1)]
    worker: WorkerProfilePartial | None = None


# --- 6. 사용자 관리 (p.17-20) ---------------------------------------------


class UserCreateRequest(StrictModel):
    """p.18. `user_id`만 필수다."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"example": {
            "user_id": "W-017", "name": "테스트 훈련자", "skill_level": "novice",
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
    """세 분석 영역의 공통 머리.

    명세서 p.15: "실제 분석이 아직 없으면 상태는 `pending`, 각 분석 영역은 `unavailable`로
    반환됩니다." 그래서 `analysis_status`가 `unavailable`일 때 수치는 전부 `null`이다 —
    임의 값을 만들지 않는다.

    `source_artifact_id`는 이 영역이 **무엇을 보고** 낸 값인지 가리킨다(예
    `telemetry-SESSION-001.csv`). 같은 세션을 다시 분석했을 때 결과가 다르면 입력이
    달랐던 것인지 알고리즘이 달랐던 것인지 이 둘(`algorithm_version`과 함께)로 가른다.
    """

    analysis_status: AnalysisStatus
    reason_code: str | None = None
    algorithm_version: str | None = None
    source_artifact_id: str | None = None
    guideline: str | None = None


# --- 5.1 도막 ---------------------------------------------------------------


class CoatingZoneRatio(StrictModel):
    """표준 6영역 면적비율. 합이 1에 가깝다."""

    under2: float
    under1: float
    normal: float
    over1: float
    over2: float
    over3: float


class CoatingDistribution(StrictModel):
    """세그먼트 판정 분포(개수)."""

    below: int
    in_band: int
    above: int


class CoatingReachability(StrictModel):
    """행동으로 회복 가능한 구역과 공정 조정이 필요한 구역(개수)."""

    reachable: int
    needs_process: int


class CoatingWorstZone(StrictModel):
    zone_id: str
    map_num: int | None = None
    uv_bin: list[int] | None = None
    thickness_um: float | None = None
    status: str | None = None
    zone_class: str | None = None
    cause: str | None = None
    action: str | None = None
    action_reachable: bool | None = None


class CoatingRootCause(StrictModel):
    dominant: str
    behavior_r2: float | None = None
    process_r2: float | None = None
    note: str | None = None


class CoatingReport(ReportArea):
    """도막 분포 진단.

    `coating_distribution_index`의 **계산식은 아직 받지 못했다.** 예시의 주석이
    "70 - 감산 + σ가산"이라 0~100 눈금인 것까지만 안다. 정의가 오기 전에는 이 값을
    내지 않는다 — `analysis_status="unavailable"`.
    """

    coating_distribution_index: float | None = None
    zone_summary: CoatingZoneRatio | None = None
    distribution: CoatingDistribution | None = None
    reachability: CoatingReachability | None = None
    worst_zones: list[CoatingWorstZone] = Field(default_factory=list)
    root_cause: CoatingRootCause | None = None


# --- 5.2 순서·방향 ----------------------------------------------------------


class SequenceZoneSummary(StrictModel):
    zone_id: str
    visit_count: int
    revisit_count: int
    deviation_count: int


class DirectionZoneSummary(StrictModel):
    zone_id: str
    expected_direction: str
    evaluated_sequence_count: int
    matched_sequence_count: int
    direction_error_count: int
    direction_adherence_rate: float


class DirectionSummary(StrictModel):
    """방향 준수는 순서 점수와 **섞지 않는다**. 두 가지를 한 비율로 합치면
    무엇을 고쳐야 하는지 가려지지 않는다."""

    evaluated_sequence_count: int
    matched_sequence_count: int
    direction_error_count: int
    direction_adherence_rate: float
    affected_zone_ids: list[str] = Field(default_factory=list)
    zone_summary: list[DirectionZoneSummary] = Field(default_factory=list)


class SequenceReport(ReportArea):
    """작업 순서 및 방향 준수.

    `direction_adherence_rate`가 두 곳에 있다. 명세서 p.16은 이 영역의 **최상위**
    필드로 적고, 예시(`trainer/demo_api/report_sample.json`)는 `direction_summary`
    안에 둔다. 둘 다 싣되 **한 번 계산해 같은 값을 쓴다** — 따로 계산하면 갈린다.
    """

    recommended_flow: list[str] = Field(default_factory=list)
    first_visit_flow: list[str] = Field(default_factory=list)
    evaluated_visit_count: int | None = None
    on_track_visit_count: int | None = None
    skipped_zone_count: int | None = None
    adherence_rate: float | None = None
    deviation_count: int | None = None
    direction_adherence_rate: float | None = None
    zone_summary: list[SequenceZoneSummary] = Field(default_factory=list)
    direction_summary: DirectionSummary | None = None


# --- 5.3 자세 ---------------------------------------------------------------


class PoseAngleIssue(StrictModel):
    axis: str
    average_diff_deg: float
    direction: str


class PoseProblemJoint(StrictModel):
    joint: str
    occurrence_count: int
    occurrence_rate: float
    average_distance: float | None = None
    angle_issues: list[PoseAngleIssue] = Field(default_factory=list)


class PoseZoneSummary(StrictModel):
    zone_id: str
    evaluated_sequence_count: int
    average_similarity_score: float | None = None
    problem_joints: list[PoseProblemJoint] = Field(default_factory=list)


class PosePriorityCorrection(StrictModel):
    priority: int
    zone_id: str | None = None
    joint: str
    occurrence_rate: float
    correction_guidance: str


class PoseReport(ReportArea):
    """작업자 자세 유사도 및 교정.

    `average_similarity_score`의 분모는 **채점된 시퀀스만**이다. 판정할 수 없었던
    시퀀스를 0점으로 세면 '자세를 못 잰 것'과 '못한 것'이 섞인다.
    """

    evaluated_sequence_count: int | None = None
    average_similarity_score: float | None = None
    zone_summary: list[PoseZoneSummary] = Field(default_factory=list)
    priority_corrections: list[PosePriorityCorrection] = Field(default_factory=list)


# --- 5.4 리포트 봉투 --------------------------------------------------------


class SessionReport(StrictModel):
    """`data.report`. 리포트 본체 — 세 영역과 종합 총평.

    `llm_briefing`은 `analysis_status="available"`인 영역만 근거로 쓰고, 구조화된
    수치를 바꾸거나 새로 만들지 않는다.
    """

    report_id: str
    report_status: ReportStatus
    coating_report: CoatingReport
    sequence_report: SequenceReport
    pose_report: PoseReport
    llm_briefing: str | None = None


class ReportDetail(StrictModel):
    """`GET /api/v1/reports/{report_id}`의 `data`.

    모양의 정본은 `trainer/demo_api/report_sample.json`이다 — 머리 필드는 이 API의 다른
    v2.5 응답(`copilot/query` 등)과 같은 규약이고 리포트 본체는 `report` 아래 둔다.
    """

    request_id: str | None = None
    session_id: str
    user_id: str | None = None
    scenario_id: str | None = None
    generated_at: str
    latency_ms: int | None = None
    report: SessionReport


def report_example() -> dict[str, Any]:
    """완성 리포트 한 건(`report_sample.json`의 `data`). Swagger 와 모의 연동이 같은 것을 보인다.

    값은 UI 설계용 예시다. 부를 때마다 새 dict 를 돌려준다 — 받는 쪽이 식별자를 바꿔 쓴다.
    """
    raw = json.loads(Path(__file__).with_name("report_sample.json").read_text(encoding="utf-8"))
    return ReportDetail.model_validate(raw["data"]).model_dump(mode="json")


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
