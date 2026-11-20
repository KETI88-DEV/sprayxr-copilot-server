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

from pydantic import BaseModel, ConfigDict, Field, model_validator

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

    skill_level: Annotated[SkillLevel, Field(
        description="훈련 시작 시점의 작업자 숙련도입니다. expert는 전문가, novice는 일반 훈련자를 뜻합니다.",
        examples=["novice"],
    )]
    handedness: Annotated[Handedness, Field(
        description="스프레이건 또는 블라스팅건을 주로 잡는 손입니다. right 또는 left를 사용합니다.",
        examples=["right"],
    )] = "right"


class WorkerProfilePartial(StrictModel):
    """p.3: `/sessions/end`의 `worker`는 선택이고 하위도 선택이다."""

    skill_level: Annotated[SkillLevel | None, Field(
        description="종료 시 갱신할 작업자 숙련도입니다. 변경하지 않을 경우 생략합니다.",
        examples=["novice"],
    )] = None
    handedness: Annotated[Handedness | None, Field(
        description="종료 시 갱신할 주 사용 손입니다. 변경하지 않을 경우 생략합니다.",
        examples=["right"],
    )] = None


class SessionStartRequest(StrictModel):
    """p.1. `user_id`가 DB에 없으면 신규 생성, 있으면 숙련도·파지손을 갱신한다."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"example": {
            "session_id": "3_11000346_1", "user_id": "1234",
            "scenario_id": "3", "started_at": "2026-10-08T01:05:44.726Z",
            "target_thickness_um": 160.0,
            "target_thickness_min_um": 160.0,
            "target_thickness_max_um": 200.0,
            "tool_type": "spray_gun",
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
    user_id: Annotated[str, Field(
        min_length=1,
        description="훈련자를 구분하는 고유 식별자입니다. 같은 사람이 다시 훈련할 때 동일한 값을 사용합니다.",
        examples=["1234"],
    )]
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
    target_thickness_um: Annotated[float | None, Field(
        default=160.0,
        gt=0,
        le=20000,
        description=(
            "기존 연동용 목표 도막 하한(µm)입니다. 단일 값 160을 보내도 정상범위는 160~200입니다. "
            "하한·상한 필드가 있으면 그 범위를 우선합니다. 단일 값만 보내면 상한은 해당 값의 1.25배입니다. "
            "모두 생략하면 정상범위 160~200µm를 적용하며 블라스팅에는 사용하지 않습니다."
        ),
        examples=[160.0],
    )]
    target_thickness_min_um: Annotated[float | None, Field(
        gt=0, le=20000, description="정상 도막 두께의 하한(µm, 경계 포함). max와 함께 입력합니다. 예:160.", examples=[160.0]
    )] = None
    target_thickness_max_um: Annotated[float | None, Field(
        gt=0, le=20000, description="정상 도막 두께의 상한(µm, 경계 포함). min과 함께 입력합니다. 예:200. 상한 초과만 과도막입니다.", examples=[200.0]
    )] = None
    tool_type: Annotated[Literal["spray_gun", "blasting_gun"], Field(
        description="시작 시 첫 Zone 기준을 고를 도구입니다. 생략하면 spray_gun(스프레이). 블라스팅은 blasting_gun을 보냅니다."
    )] = "spray_gun"

    @model_validator(mode="after")
    def validate_thickness_range(self):
        low, high = self.target_thickness_min_um, self.target_thickness_max_um
        if (low is None) != (high is None):
            raise ValueError("target_thickness_min_um and target_thickness_max_um must be provided together")
        if low is not None and low > high:
            raise ValueError("target_thickness_min_um must be <= target_thickness_max_um")
        return self

    worker: Annotated[WorkerProfile, Field(
        description="훈련자의 숙련도와 주 사용 손 정보입니다. 사용자 프로필 생성 또는 갱신에 사용합니다."
    )]


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
    worker: Annotated[WorkerProfilePartial | None, Field(
        description="종료 시 사용자 프로필을 갱신할 때만 보냅니다. 갱신하지 않으면 필드 전체를 생략합니다."
    )] = None


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

    user_id: Annotated[str, Field(
        min_length=1, description="신규 등록할 훈련자의 고유 식별자입니다.", examples=["1234"]
    )]
    name: Annotated[str | None, Field(
        description="화면과 리포트에 표시할 훈련자 이름입니다. 개인정보 정책에 따라 생략할 수 있습니다.",
        examples=["테스트 훈련자"],
    )] = None
    skill_level: Annotated[SkillLevel, Field(
        description="훈련자의 숙련도 분류입니다. expert 또는 novice를 사용합니다.", examples=["novice"]
    )] = "novice"
    dominant_hand: Annotated[Handedness, Field(
        description="훈련자가 도장 도구를 주로 잡는 손입니다. right 또는 left를 사용합니다.", examples=["right"]
    )] = "right"
    age: Annotated[int | None, Field(
        ge=0, le=120, description="훈련자 나이(만 나이)입니다. 수집하지 않으면 생략합니다.", examples=[35]
    )] = None
    gender: Annotated[str | None, Field(
        description="훈련자 성별 또는 시스템에서 사용하는 성별 코드입니다. 수집하지 않으면 생략합니다.",
        examples=["male"],
    )] = None


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
    """표준 6영역 면적비율 — 칠한 면적 대비 비율이다(칠하지 않은 면적은 분모에 없다). 합이 1에 가깝다."""

    under2: float
    under1: float
    normal: float
    over1: float
    over2: float
    over3: float


class CoatingDistribution(StrictModel):
    """목표 미만 / 정상 구간 / 과도막 경계 이상으로 나눈 분포. 단위는 칠한 화소 수다."""

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


class CoatingExpertZone(StrictModel):
    zone_id: str
    map_num: int
    status: Literal["normal", "under_app", "over_app"]
    coverage_percent: float
    completion_percent: float
    overpaint_percent: float


class CoatingExpertComparison(StrictModel):
    """참고 판정: 전문가 최종 도막 칸(q1·q3)과 견준 구역 상태. 두께는 보정 전 값이다.
    주 판정은 상위 `coating_report`의 목표 대비 등급이다. 진행 게이트·등급에는 쓰지 않는다."""

    comparison_basis: Literal["expert_reference"] = "expert_reference"
    analysis_status: AnalysisStatus
    reason_code: str | None = None
    zone_count: int
    normal: int
    under_app: int
    over_app: int
    zones: list[CoatingExpertZone] = Field(default_factory=list)


class CoatingReport(ReportArea):
    """도막 분포 진단. 구역 등급·분포·가장 벗어난 구역은 목표 두께 기준으로 낸다.

    `target_basis`는 세션이 목표를 명시했으면 `session_target`, 생략해 기본 목표로 판정했으면
    `default_target`이다. `expert_comparison`은 참고로 싣는 전문가 대비 판정이다. 주 판정
    (`comparison_basis="session_target"`)과 기준이 다르다.

    `coating_distribution_index`만은 **계산식을 아직 받지 못했다.** 예시의 주석이
    "70 - 감산 + σ가산"이라 0~100 눈금인 것까지만 안다. 정의가 오기 전에는 이 값만 null로 둔다.
    """

    coating_distribution_index: float | None = None
    zone_summary: CoatingZoneRatio | None = None
    distribution: CoatingDistribution | None = None
    reachability: CoatingReachability | None = None
    worst_zones: list[CoatingWorstZone] = Field(default_factory=list)
    root_cause: CoatingRootCause | None = None
    comparison_basis: Literal["expert_reference", "session_target"] = "expert_reference"
    target_thickness_um: float | None = None
    normal_band_um: list[float] | None = None
    target_basis: Literal["session_target", "default_target"] | None = None
    expert_comparison: CoatingExpertComparison | None = None


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


class OrderPositionChoice(StrictModel):
    zone_id: str
    count: int
    probability: float


class SequenceReport(ReportArea):
    """작업 순서 및 방향 준수.

    `direction_adherence_rate`가 두 곳에 있다. 명세서 p.16은 이 영역의 **최상위**
    필드로 적고, 예시(`trainer/demo_api/report_sample.json`)는 `direction_summary`
    안에 둔다. 둘 다 싣되 **한 번 계산해 같은 값을 쓴다** — 따로 계산하면 갈린다.
    """

    recommended_flow: list[str] = Field(default_factory=list)
    first_visit_flow: list[str] = Field(default_factory=list)
    reference_session_count: int | None = None
    reference_flows: list[list[str]] = Field(default_factory=list)
    position_distribution: list[list[OrderPositionChoice]] = Field(default_factory=list)
    best_reference_similarity: float | None = None
    position_similarity: float | None = None
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

    confidence: Literal["limited"] | None = None
    limited_sequence_count: int | None = None
    reason_codes: list[str] | None = None

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


class FirstZoneGuidance(StrictModel):
    """등록된 전문가 순서의 시작 구역. 실전·Unit 기준이 없으면 구역을 추정하지 않는다."""

    analysis_status: Literal["available", "not_available"]
    reason_code: str | None = None
    process: Literal["spray", "blasting"]
    zone_id: Annotated[str | None, Field(
        description="등록된 전문가 작업 순서의 첫 Zone ID입니다. 기준이 없으면 null입니다.",
        examples=["1"],
    )] = None
    recommended_flow: list[str] = Field(
        default_factory=list, description="등록된 실측 전문가 작업 순서입니다. 기준이 없으면 빈 목록입니다.",
        examples=[["1", "2", "3"]],
    )
    reference_basis: Literal["measured_expert_runs"] | None = None
    message: Annotated[str, Field(
        description="시작 구역 또는 기준 미등록을 알리는 짧은 안내입니다.",
        examples=["1구역에서 시작하세요."],
    )]


class SessionStateData(StrictModel):
    """`/sessions/start`·`/sessions/end`의 `data`."""

    session_id: str
    user_id: str
    scenario_id: str
    session_status: Literal["ACTIVE", "COMPLETED"]
    started_at: str | None
    ended_at: str | None
    user_created: bool
    target_thickness_um: Annotated[float | None, Field(
        description="세션 시작 시 저장된 목표 도막 두께(µm)입니다. 입력하지 않은 기존 세션은 null입니다."
    )]
    target_thickness_min_um: float | None = None
    target_thickness_max_um: float | None = None
    first_zone_id: Annotated[str | None, Field(
        description="훈련 시작 시 안내하는 첫 Zone ID입니다. 등록된 기준이 없는 실전·Unit 등은 null입니다.",
        examples=["1"],
    )] = None
    first_zone_guidance: FirstZoneGuidance | None = Field(
        default=None, description="첫 Zone과 전문가 순서 안내입니다. 세션 종료 응답은 null입니다.",
    )


class SessionResponse(ApiResponse):
    """세션 시작·종료의 구체적인 응답 구조. 기존 봉투와 JSON 필드를 유지한다."""

    data: SessionStateData
