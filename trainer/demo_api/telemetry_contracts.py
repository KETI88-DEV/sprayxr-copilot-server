"""v2.5 시퀀스 전송·분석 요청 계약.

**정본은 `docs/XR Painting Simulator AI Co-Pilot Service API 정의서 - 대화형 API 명세서.pdf`**
(76쪽, 엔드포인트 15개)다. `docs/260813 도장 SIM 회의자료(...v2.5).pdf`와 다르면 이쪽을
따른다 — 260813은 엔드포인트 8개만 싣고 있고 필드 이름도 다르다(`pressure_bar` vs
`pressure`).

두 엔드포인트는 **별개**다. 같은 본문을 둘에 보내는 것이 설계다.

* `POST /api/v1/telemetry/sequence` (p.4-5): 90 FPS 센서 배치 전송 + 도막·궤적 평가
* `POST /api/v1/copilot/analyze-sequence`: 1회 패스 3D Ray Point 궤적 비교 +
  원포인트 레슨. 2026-10-08 TSB 실전송 규격처럼 `scenario_id`로 모드를 선택한다.
  과거 클라이언트의 `mode_id`는 선택 호환 필드로만 받는다.

이 문서의 이전 판이 `analyze-sequence`를 "계약에 없는 벤더 중복 전송"이라 적고 멱등
별칭으로 취급한 것은 **틀렸다**. 근거로 삼은 260813에 그 항목이 없었을 뿐이다.

엄격성은 계층으로 나눈다(설계 `2026-09-29-sprayxr-runtime-alignment-design.md` §2.2).
최상위는 `forbid` — 오타가 조용히 무시되면 엉뚱한 세션에 쓴다. 프레임 하위는 `allow` —
현행 정책을 그대로 적용하면 실물 샘플의 정상 시퀀스 100%가 422다.
"""

from __future__ import annotations

import re
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from trainer.demo_api.contracts import StrictModel

AnalysisStatus = Literal["available", "not_available"]
SpeedGuideStatus = Literal["slow", "ok", "fast"]
DistanceGuideStatus = Literal["near", "ok", "far"]

_TOOL_TYPE_ALIASES = {
    "spray": "spray",
    "spray_gun": "spray",
    "spraygun": "spray",
    "blasting": "blasting",
    "blasting_gun": "blasting",
    "blast_gun": "blasting",
    "blastgun": "blasting",
}


def learning_mode_from_scenario(scenario_id: str) -> int | None:
    """Return a 1~10 learning mode without mistaking practical A-1/1-A IDs.

    TSB's current sample sends a bare numeric scenario ID.  Named legacy values
    are retained for Swagger/demo compatibility, but practical scenarios are
    deliberately not coerced to a learning reference.
    """

    value = scenario_id.strip().lower()
    if value.isdigit():
        parsed = int(value)
        return parsed if 1 <= parsed <= 10 else None
    tr_learning = re.fullmatch(r"tr-l(0[1-9]|10)", value)
    if tr_learning is not None:
        return int(tr_learning.group(1))
    match = re.fullmatch(r"(?:training|learning)[-_ ]?mode[-_ ]?(10|[1-9])", value)
    return int(match.group(1)) if match is not None else None


def process_from_tool_type(value: Any) -> Literal["spray", "blasting"] | None:
    if not isinstance(value, str):
        return None
    return _TOOL_TYPE_ALIASES.get(value.strip().lower().replace("-", "_"))  # type: ignore[return-value]


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
    """명세서 p.5의 최상위 8키. **여덟 개 모두 필수다.**

    이 문서의 이전 판은 `sequence_id` 외 전부를 선택으로 두었다. 명세서가 `(필수)`로
    적고 있으므로 맞춘다 — 빠진 필드는 422다. `level`·`floor_height_m`는 명세서에 없고
    벤더도 보내지 않으므로 받지 않고 서버가 `scenario_id`에서 유도한다.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "session_id": "TSB-DEMO-001", "user_id": "W-017",
                "scenario_id": "3", "zone_id": "3", "sequence_id": "SEQ-001",
                "sequence_setting": {
                    "tool_type": "spray_gun",
                    "target_speed_mms": 500,
                    "target_distance_mm": 250,
                },
                "sequence_info": [{
                    "index": 0, "time": "2026-10-06T10:01:00Z",
                    "tool_pos_rot": {"position": [0.0, 1.2, 0.4]},
                    "gun_metrics": {"speed_mms": 450, "distance_mm": 260},
                }],
                "sequence_result": {"avg_speed_mms": 450, "avg_distance_mm": 260},
            }
        },
    )

    session_id: Annotated[str, Field(min_length=1)]
    user_id: Annotated[str, Field(min_length=1)]
    scenario_id: Annotated[str, Field(min_length=1)]
    # "1"~"39", 미지정·미판정은 "-1" (p.5)
    zone_id: Annotated[str, Field(min_length=1)]
    sequence_id: Annotated[str, Field(min_length=1)]
    sequence_setting: dict[str, Any]
    sequence_info: list[SequenceFrameIn]
    sequence_result: dict[str, Any]


class AnalyzeSequenceRequest(TelemetrySequenceRequest):
    """최신 TSB 전송 규격의 8키. `mode_id`는 과거 연동 호환용이다."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "session_id": "TSB-DEMO-001",
                "user_id": "W-017",
                "scenario_id": "3",
                "zone_id": "3",
                "sequence_id": "SEQ-001",
                "sequence_setting": {
                    "tool_type": "spray_gun",
                    "target_speed_mms": 500,
                    "target_distance_mm": 250,
                },
                "sequence_info": [
                    {
                        "index": 0,
                        "time": "2026-10-02T10:00:00Z",
                        "tool_pos_rot": {"position": [0.0, 1.2, 0.4]},
                        "gun_metrics": {"speed_mms": 500, "distance_mm": 250},
                    }
                ],
                "sequence_result": {"avg_speed_mms": 500, "avg_distance_mm": 250},
            }
        },
    )

    # 2026-10-08 실전송에는 없다. 예전 클라이언트만 선택적으로 보낼 수 있다.
    mode_id: Annotated[int | None, Field(ge=1, le=10)] = None

    @model_validator(mode="after")
    def _mode_id_must_match_scenario(self) -> "AnalyzeSequenceRequest":
        scenario_mode = learning_mode_from_scenario(self.scenario_id)
        if self.mode_id is not None and scenario_mode is not None and self.mode_id != scenario_mode:
            raise ValueError("mode_id conflicts with scenario_id")
        return self

    @property
    def resolved_mode_id(self) -> int | None:
        return learning_mode_from_scenario(self.scenario_id) or self.mode_id

    @property
    def process_type(self) -> Literal["spray", "blasting"] | None:
        process = process_from_tool_type(self.sequence_setting.get("tool_type"))
        # Before tool_type was added, analyze-sequence only had spray references.
        # Preserve that behavior solely for clients that still send legacy mode_id.
        return process or ("spray" if self.mode_id is not None else None)


class SequenceDataAck(StrictModel):
    stored: bool
    duplicate: bool
    frames_received: Annotated[int, Field(ge=0)]
    frames_accepted: Annotated[int, Field(ge=0)]


class EngineFeedback(StrictModel):
    """`coating_feedback`·`sequence_feedback`의 자리.

    엔진이 없으므로 `analysis_status`와 `reason_code`만 낸다. 계약 예시는 이 두 객체에
    `analysis_status`를 넣지 않았지만, §7.3 말미가 "판정할 데이터가 부족하면 임의 값을
    만들지 않고 `analysis_status="not_available"`과 안정적인 `reason_code`를 반환한다"고
    규정하므로 세 feedback의 확인 방법을 하나로 맞춘다. 계약 개정 대상이다.
    """

    analysis_status: AnalysisStatus
    reason_code: str | None


class GuideItem(StrictModel):
    status: SpeedGuideStatus | DistanceGuideStatus | None
    message: str | None
    available: bool
    reason: str | None


class SequenceGuide(StrictModel):
    speed: GuideItem
    distance: GuideItem


class TelemetrySequenceData(StrictModel):
    """시퀀스를 데이터베이스에 올린 결과.

    **자세는 여기 없다.** 명세서 p.4는 텔레메트리에 "도막 두께 및 궤적 평가"를 걸고
    자세는 걸지 않는다. 자세 유사도 비교는 `POST /api/v1/copilot/query`의 `pose_recomm`이
    `context`로 시퀀스를 가리켜 수행한다(p.9). 이 모델의 이전 판이 `pose_feedback`을
    실은 것은 틀렸다.

    프레임 하위의 미지 키는 응답이 아니라 **요청당 WARNING 로그 1건**으로 보고한다.
    본문 충돌은 `sequence_data_ack`의 `stored=false, duplicate=false` 조합이 구별해 준다.
    """

    request_id: str
    session_id: str
    user_id: str
    scenario_id: str
    sequence_id: str
    generated_at: str
    latency_ms: Annotated[int, Field(ge=0)]
    sequence_data_ack: SequenceDataAck
    coating_feedback: EngineFeedback
    sequence_feedback: EngineFeedback
    text_guide: str | None


class TelemetrySequenceResponse(StrictModel):
    status: Literal["SUCCESS", "ERROR"]
    code: int
    message: str
    data: TelemetrySequenceData | None


class AnalyzeSequenceData(TelemetrySequenceData):
    guide: SequenceGuide


class AnalyzeSequenceResponse(StrictModel):
    status: Literal["SUCCESS", "ERROR"]
    code: int
    message: str
    data: AnalyzeSequenceData | None
