"""v2.5 `POST /api/v1/telemetry/sequence` 계약 (PDF p.8 "시퀀스 전송", 계약 §7).

정본 경로는 `telemetry/sequence`다. 벤더 클라이언트가 함께 보내는
`/copilot/analyze-sequence`는 계약에 없고 본문 sha256이 동일한 중복 전송이므로
(`docs/sprayxr_v25_sample_delta_2026-09-28.md` B18) 멱등 별칭으로만 받는다.

엄격성은 계층으로 나눈다(설계 `2026-09-29-sprayxr-runtime-alignment-design.md` §2.2).
최상위는 `forbid` — 오타가 조용히 무시되면 엉뚱한 세션에 쓴다. 프레임 하위는 `allow` —
현행 정책을 그대로 적용하면 실물 샘플의 정상 시퀀스 100%가 422다.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from trainer.demo_api.contracts import StrictModel

AnalysisStatus = Literal["available", "not_available"]


class SequenceFrameIn(BaseModel):
    """프레임 하나. 모르는 키를 받되 세어서 보고한다.

    `skeleton_joints`는 실물이 `{관절: [x,y,z]}`이고 계약 예시는 `{관절: {pos, rot}}`다.
    둘 다 통과시키고 해석은 엔진이 한다 — 실물 샘플이 정본이다(설계 §0-1).
    `time`은 실물이 ISO-8601 문자열이고 숫자 계약도 있어 둘 다 받는다.
    """

    model_config = ConfigDict(extra="allow")

    index: int | None = None
    time: str | float | None = None
    tool_pos_rot: dict[str, Any] | None = None
    skeleton_joints: dict[str, Any] = Field(default_factory=dict)
    gun_metrics: dict[str, Any] | None = None


class TelemetrySequenceRequest(StrictModel):
    """계약 §7.1의 최상위 8키. `level`·`floor_height_m`는 벤더가 보내지 않으므로 받지 않고
    서버가 `scenario_id`에서 유도한다.

    `session_id` 부재를 422로 만들지 않는다(설계 §2.4). 귀속은 `sequence_data_ack`가 보고한다.
    """

    sequence_id: Annotated[str, Field(min_length=1)]
    session_id: str | None = None
    user_id: str | None = None
    scenario_id: str | None = None
    zone_id: str | None = None
    sequence_setting: dict[str, Any] | None = None
    sequence_info: list[SequenceFrameIn] = Field(default_factory=list)
    sequence_result: dict[str, Any] | None = None


class SequenceDataAck(StrictModel):
    stored: bool
    duplicate: bool
    frames_received: Annotated[int, Field(ge=0)]
    frames_accepted: Annotated[int, Field(ge=0)]


class PoseCorrection(StrictModel):
    """계약 §7.3의 교정 항목. MCP 결과의 풍부한 필드는 공개하지 않는다 —
    `evidence`·`target.per_frame`은 내부 인터페이스의 것이다."""

    joint: str
    label: str
    target_validated: bool
    angle_delta_deg: float | None
    direction: str | None


class PoseFeedback(StrictModel):
    analysis_status: AnalysisStatus
    reason_code: str | None
    similarity_score: float | None
    correction_mode: str | None
    corrections: list[PoseCorrection]


class EngineFeedback(StrictModel):
    """`coating_feedback`·`sequence_feedback`의 자리.

    엔진이 없으므로 `analysis_status`와 `reason_code`만 낸다. 계약 예시는 이 두 객체에
    `analysis_status`를 넣지 않았지만, §7.3 말미가 "판정할 데이터가 부족하면 임의 값을
    만들지 않고 `analysis_status="not_available"`과 안정적인 `reason_code`를 반환한다"고
    규정하므로 세 feedback의 확인 방법을 하나로 맞춘다. 계약 개정 대상이다.
    """

    analysis_status: AnalysisStatus
    reason_code: str | None


class TelemetrySequenceData(StrictModel):
    """계약 §7.3의 12키와 정확히 같다. 필드를 더하지 않는다 — SIM이 받는 모양은 계약이 정한다.

    프레임 하위의 미지 키는 응답이 아니라 **요청당 WARNING 로그 1건**으로 보고한다.
    엔진 장애는 `pose_feedback.reason_code`가, 본문 충돌은 `sequence_data_ack`의
    `stored=false, duplicate=false` 조합이 이미 구별해 준다.
    """

    request_id: str
    session_id: str | None
    user_id: str | None
    scenario_id: str | None
    sequence_id: str
    generated_at: str
    latency_ms: Annotated[int, Field(ge=0)]
    sequence_data_ack: SequenceDataAck
    coating_feedback: EngineFeedback
    sequence_feedback: EngineFeedback
    pose_feedback: PoseFeedback
    text_guide: str | None


class TelemetrySequenceResponse(StrictModel):
    status: Literal["SUCCESS", "ERROR"]
    code: int
    message: str
    data: TelemetrySequenceData | None
