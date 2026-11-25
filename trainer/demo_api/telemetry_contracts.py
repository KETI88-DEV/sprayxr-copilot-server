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
CoatingProgressStatus = Literal["needs_more", "complete", "overpaint_risk"]
CoatingRecommendation = Literal[
    "continue_current_zone", "move_next_zone", "stop_current_zone"
]

_TOOL_TYPE_ALIASES = {
    "spray": "spray",
    "spray_gun": "spray",
    "spraygun": "spray",
    "blasting": "blasting",
    "blasting_gun": "blasting",
    "blast_gun": "blasting",
    "blastgun": "blasting",
}

TexturePixelCoordinate = Annotated[float, Field(ge=0, le=2048)]


def learning_mode_from_scenario(scenario_id: str) -> int | None:
    """Return a 1~10 learning mode without coercing practical or Unit IDs.

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


class GunMetricsIn(BaseModel):
    """TSB 프레임의 분사 계측값.

    버전별 추가 키는 허용하되, 연동에 합의된 텍스처 좌표는 실제 전송 위치인
    ``sequence_info[].gun_metrics.raytexuv``에서 검증한다.
    """

    model_config = ConfigDict(extra="allow")

    speed: Annotated[float | None, Field(
        description="해당 프레임의 도구 이동 속도(mm/s)입니다. 시뮬레이터에서 산출하지 못하면 null입니다.",
        examples=[450.0],
    )] = None
    distance: Annotated[float | None, Field(
        description="도구 노즐과 피도면 충돌점 사이 이격거리(mm)입니다. 레이 미충돌 시 null입니다.",
        examples=[260.0],
    )] = None
    raypoint: Annotated[list[float] | None, Field(
        description="피도면에 충돌한 Ray의 월드 좌표 [X,Y,Z]입니다. 레이 미충돌 시 null입니다.",
        examples=[[0.0, 1.0, 0.5]],
    )] = None
    raytexmapnum: Annotated[int | None, Field(
        description="Ray가 충돌한 Texture Map 번호입니다. 레이 미충돌 시 null입니다.",
        examples=[1],
    )] = None
    raytexuv: tuple[TexturePixelCoordinate, TexturePixelCoordinate] | None = Field(
        default=None,
        description=(
            "2048x2048 텍스처의 픽셀 좌표 [X,Y]. 원점은 좌하단이고 "
            "X는 오른쪽, Y는 위쪽으로 증가합니다. 레이 미충돌은 null입니다."
        ),
        examples=[[1024, 768]],
    )
    spraytime: Annotated[float | None, Field(
        description="해당 프레임까지의 분사 누적 시간 또는 시뮬레이터가 기록한 분사 시간(초)입니다.",
        examples=[0.011],
    )] = None


class SequenceFrameIn(BaseModel):
    """프레임 하나. 모르는 키를 받되 세어서 보고한다.

    `skeleton_joints`는 실물이 `{관절: [x,y,z]}`이고 계약 예시는 `{관절: {pos, rot}}`다.
    둘 다 통과시키고 해석은 엔진이 한다 — 실물 샘플이 정본이다(설계 §0-1).
    `time`은 실물이 ISO-8601 문자열이고 숫자 계약도 있어 둘 다 받는다.
    """

    model_config = ConfigDict(extra="allow")

    index: Annotated[int | None, Field(
        description="현재 시퀀스 안에서 0부터 증가하는 프레임 번호입니다.", examples=[0]
    )] = None
    time: Annotated[str | float | None, Field(
        description="프레임 기록 시각입니다. ISO 8601 문자열 또는 시퀀스 시작 이후 경과 초를 사용할 수 있습니다.",
        examples=["2026-10-08T01:05:53.000Z"],
    )] = None
    tool_pos_rot: Annotated[dict[str, Any] | None, Field(
        description="HMD·스프레이건·플래시 등 도구의 위치와 회전 Transform 묶음입니다.",
        examples=[{"position": [0.0, 1.2, 0.4], "rotation": [0.0, 0.0, 0.0, 1.0]}],
    )] = None
    skeleton_joints: Annotated[dict[str, Any], Field(
        default_factory=dict,
        description="프레임의 관절별 위치·회전 정보입니다. 관절 이름을 키로 하는 객체이며 없는 경우 빈 객체를 보냅니다.",
    )]
    gun_metrics: Annotated[GunMetricsIn | None, Field(
        description="해당 프레임에서 측정된 속도·이격거리·Ray 충돌 좌표·분사 시간 정보입니다."
    )] = None
    # 2026-10-08 TSB 정본은 gun_metrics 안에 둔다. 이 필드는 구형 전송 호환용이다.
    raytexuv: tuple[TexturePixelCoordinate, TexturePixelCoordinate] | None = Field(
        default=None,
        deprecated=True,
        description=(
            "구형 호환 필드입니다. 신규 연동은 gun_metrics.raytexuv를 사용하세요. "
            "좌표 범위와 원점 규칙은 동일합니다."
        ),
        examples=[[1024, 768]],
    )


class SequenceHeatmapCropIn(BaseModel):
    """TSB 누적 도막 높이맵의 영역별 PNG 크롭."""

    model_config = ConfigDict(extra="allow")

    map_num: Annotated[int, Field(ge=1, description="크롭이 속한 Texture Map 번호입니다.", examples=[1])]
    x0: Annotated[int, Field(ge=0, description="2048×2048 전체 텍스처의 좌하단 원점 기준 시작 X 픽셀입니다.", examples=[612])]
    y0: Annotated[int, Field(ge=0, description="2048×2048 전체 텍스처의 좌하단 원점 기준 시작 Y 픽셀입니다.", examples=[388])]
    width: Annotated[int, Field(gt=0, description="전체 텍스처 좌표계에서 크롭이 차지하는 폭(픽셀)입니다.", examples=[1])]
    height: Annotated[int, Field(gt=0, description="전체 텍스처 좌표계에서 크롭이 차지하는 높이(픽셀)입니다.", examples=[1])]
    img_width: Annotated[int, Field(gt=0, description="data로 전달한 PNG 자체의 폭(픽셀)입니다.", examples=[1])]
    img_height: Annotated[int, Field(gt=0, description="data로 전달한 PNG 자체의 높이(픽셀)입니다.", examples=[1])]
    format: Annotated[str, Field(
        min_length=1,
        description=(
            "픽셀 해석 규격입니다. v2에서는 R+G가 16비트 두께, B가 alpha이며 "
            "두께(µm)=(R/255+G/65025)×20000 입니다."
        ),
        examples=["v2;thickness=RG16;alpha=B8;scale_um=20000"],
    )]
    encoding: Annotated[str, Field(
        min_length=1,
        description="data의 직렬화 형식입니다. 현재 TSB 규격은 png;base64입니다.",
        examples=["png;base64"],
    )]
    data: Annotated[str, Field(
        description=(
            "PNG 바이트를 Base64 문자열로 인코딩한 값입니다. PNG 첫 행은 최대 Y이며 "
            "픽셀 행 변환은 row=img_height-1-y 규칙을 사용합니다."
        ),
        examples=["iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="],
    )]


class SequenceResultIn(BaseModel):
    """시퀀스 요약과 요청 시점 피도면의 누적 도막 입력."""

    model_config = ConfigDict(extra="allow")

    avg_speed_mms: Annotated[float | None, Field(default=None, ge=0, description="해당 시퀀스 평균 이동 속도(mm/s)입니다.", examples=[450.0])]
    avg_distance_mm: Annotated[float | None, Field(default=None, ge=0, description="해당 시퀀스 평균 이격거리(mm)입니다.", examples=[260.0])]
    avg_thickness_um: Annotated[float | None, Field(default=None, ge=0, description="TSB가 산출한 해당 시퀀스 평균 도막 두께(µm)입니다.", examples=[138.91])]
    cur_sequence_heatmap: Annotated[list[SequenceHeatmapCropIn] | None, Field(
        default=None,
        description=(
            "이름과 달리 현재 시퀀스의 증분만이 아니라 요청 시점 피도면의 전체 누적 도막 두께입니다. "
            "이전 회차의 도막도 포함하며, 배열 원소는 현재 시퀀스가 도포한 영역만 자른 Texture Map PNG입니다. "
            "서버는 같은 위치의 직전 절대 두께와 비교해 현재 시퀀스의 증분을 계산합니다. "
            "피도면에 맞지 않은 시퀀스는 생략하거나 빈 배열로 보낼 수 있습니다."
        ),
    )]


_SEQUENCE_REQUEST_EXAMPLE = {
    "session_id": "3_11000346_1",
    "user_id": "1234",
    "scenario_id": "3",
    "zone_id": "3",
    "sequence_id": "SEQ-001",
    "sequence_setting": {
        "tool_type": "spray_gun",
        "pressure": 0.55,
        "nozzle_size": 0.021,
        "thinning_ratio": 10.0,
        "viscosity": 65.0,
        "month": 10,
    },
    "sequence_info": [{
        "index": 0,
        "time": "2026-10-08T01:05:53.000Z",
        "tool_pos_rot": {"position": [0.0, 1.2, 0.4]},
        "gun_metrics": {
            "speed": 450.0,
            "distance": 260.0,
            "raypoint": [0.0, 1.0, 0.5],
            "raytexmapnum": 1,
            "raytexuv": [1024, 768],
            "spraytime": 0.011,
        },
    }],
    "sequence_result": {
        "avg_speed_mms": 450.0,
        "avg_distance_mm": 260.0,
        "avg_thickness_um": 138.91,
        "cur_sequence_heatmap": [{
            "map_num": 1,
            "x0": 612,
            "y0": 388,
            "width": 1,
            "height": 1,
            "img_width": 1,
            "img_height": 1,
            "format": "v2;thickness=RG16;alpha=B8;scale_um=20000",
            "encoding": "png;base64",
            "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=",
        }],
    },
}


class TelemetrySequenceRequest(StrictModel):
    """명세서 p.5의 최상위 8키. **여덟 개 모두 필수다.**

    이 문서의 이전 판은 `sequence_id` 외 전부를 선택으로 두었다. 명세서가 `(필수)`로
    적고 있으므로 맞춘다 — 빠진 필드는 422다. `level`·`floor_height_m`는 명세서에 없고
    벤더도 보내지 않으므로 받지 않고 서버가 `scenario_id`에서 유도한다.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"example": _SEQUENCE_REQUEST_EXAMPLE},
    )

    session_id: Annotated[str, Field(min_length=1, description="sessions/start에서 생성한 동일 세션 식별자입니다.", examples=["3_11000346_1"])]
    user_id: Annotated[str, Field(min_length=1, description="sessions/start에서 사용한 동일 훈련자 식별자입니다.", examples=["1234"])]
    scenario_id: Annotated[
        str,
        Field(
            min_length=1,
            description=(
                "session_id의 첫 `_` 앞 콘텐츠 식별자. 학습은 1~10, "
                "실전은 1-R-A·1-V-AB·2-ABCDE, Unit은 Unit12 형식입니다."
            ),
            examples=["3", "1-R-A", "2-ABCDE", "Unit12"],
        ),
    ]
    # "1"~"39", 미지정·미판정은 "-1" (p.5)
    zone_id: Annotated[str, Field(
        min_length=1,
        description="현재 시퀀스의 Zone 번호 문자열입니다. 피도면에 충돌하지 않았거나 판정할 수 없으면 -1을 사용합니다.",
        examples=["3"],
    )]
    sequence_id: Annotated[str, Field(
        min_length=1,
        description="세션 안에서 분사 도구를 연속으로 켠 한 구간의 식별자입니다. 호출 순서대로 고유하게 부여합니다.",
        examples=["SEQ-001"],
    )]
    sequence_setting: Annotated[
        dict[str, Any],
        Field(description="도구 구분은 tool_type에 spray_gun 또는 blasting_gun을 사용합니다."),
    ]
    sequence_info: Annotated[list[SequenceFrameIn], Field(
        description="현재 시퀀스 동안 수집한 시간순 프레임 배열입니다. 각 프레임에는 도구·관절·Ray 계측값이 포함됩니다."
    )]
    sequence_result: Annotated[SequenceResultIn, Field(
        description="시퀀스 요약값과 TSB 누적 도막 크롭입니다. 아래 필드를 펼쳐 상세 규격을 확인하세요."
    )]


class AnalyzeSequenceRequest(TelemetrySequenceRequest):
    """최신 TSB 전송 규격의 8키. `mode_id`는 과거 연동 호환용이다."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"example": _SEQUENCE_REQUEST_EXAMPLE},
    )

    # 2026-10-08 실전송에는 없다. 예전 클라이언트만 선택적으로 보낼 수 있다.
    mode_id: Annotated[int | None, Field(
        ge=1,
        le=10,
        description="구형 클라이언트 호환용 학습모드 번호입니다. 신규 연동은 scenario_id를 사용하며 보통 생략합니다.",
        examples=[3],
    )] = None

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

    `sequence_feedback`과 저장 전용 경로(`telemetry/sequence`)의 `coating_feedback`은
    `analysis_status`와 `reason_code`만 낸다. analyze-sequence의 `coating_feedback`은 아래
    `CoatingFeedback`이 채운다. 계약 예시는 이 두 객체에
    `analysis_status`를 넣지 않았지만, §7.3 말미가 "판정할 데이터가 부족하면 임의 값을
    만들지 않고 `analysis_status="not_available"`과 안정적인 `reason_code`를 반환한다"고
    규정하므로 세 feedback의 확인 방법을 하나로 맞춘다. 계약 개정 대상이다.
    """

    analysis_status: AnalysisStatus
    reason_code: str | None


class CoatingFeedback(EngineFeedback):
    """analyze-sequence의 `coating_feedback`: 현재 Zone의 도막 등급·분포·조절 후보.

    값은 `coating_assessment.zone_row`의 행을 그대로 옮긴 것이다. 칠한 곳이 없으면
    `not_available`이고 등급·평균만 비며, 도막을 재지 못했으면 나머지가 전부 비어 있다.
    """

    zone_id: str | None = None
    map_num: int | None = None
    status: Literal["under_app", "normal", "over_app"] | None = None
    grade: Literal["below", "normal", "above"] | None = None
    thickness_um: float | None = None
    target_thickness_um: float | None = None
    target_basis: Literal["session_target", "default_target"] | None = None
    normal_band_um: list[float] | None = None
    unpainted_fraction: Annotated[float | None, Field(ge=0, le=1)] = None
    six_band_counts: dict[str, int] | None = None
    adjustments: list[dict[str, Any]] = Field(default_factory=list)
    message: str | None = None
    algorithm_version: str | None = None


class GuideItem(StrictModel):
    """`reason="coating_adjustment_priority"`는 같은 응답의 도막 조절 후보와 엇갈려 비운 항목이다."""

    status: SpeedGuideStatus | DistanceGuideStatus | None
    message: str | None
    available: bool
    reason: str | None


class SequenceGuide(StrictModel):
    speed: GuideItem
    distance: GuideItem


class CoatingProgress(StrictModel):
    """현재 Zone의 세션 누적 도막을 목표 두께와 비교한 진행 판정."""

    zone_id: str
    available: bool
    status: CoatingProgressStatus | None
    recommendation: CoatingRecommendation | None
    completion_percent: Annotated[float | None, Field(ge=0, le=100)]
    coverage_percent: Annotated[float | None, Field(ge=0, le=100)]
    normal_band_percent: Annotated[float | None, Field(ge=0, le=100)] = None
    overpaint_percent: Annotated[float | None, Field(ge=0, le=100)]
    next_zone_id: str | None
    message: str | None
    reason: str | None
    comparison_basis: Literal["expert_reference", "session_target"] = "expert_reference"
    target_thickness_um: float | None = None
    expert_completion_percent: Annotated[float | None, Field(ge=0, le=100)] = None
    target_basis: Literal["session_target", "default_target"] | None = None
    # 엔진이 라우터에 평가 행을 건네는 자리다. 응답·저장에는 직렬화하지 않는다.
    assessment: dict[str, Any] | None = Field(default=None, exclude=True)


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
    text_guide: str | None = Field(
        description="시퀀스 분석의 실시간 안내. 60자 이내 한 줄 명사형으로 상태와 조치를 요약합니다. "
                    "예: 보완 권장·이동 권장·확인 필요·분사 중지. 상세 근거는 coating_progress와 "
                    "피드백 필드를 참고하세요. 분석하지 않는 업로드 경로나 안내 불가 상태는 null입니다.",
        examples=["1구역 진행률 85% 이상 충족. 2구역으로 이동 권장.",
                  "현재 1구역 도막 완료 미확인. 이동 전 도막 자료 확인 필요."],
    )


class TelemetrySequenceResponse(StrictModel):
    status: Literal["SUCCESS", "ERROR"]
    code: int
    message: str
    data: TelemetrySequenceData | None


class AnalyzeSequenceData(TelemetrySequenceData):
    coating_feedback: CoatingFeedback
    guide: SequenceGuide
    coating_progress: CoatingProgress


class AnalyzeSequenceResponse(StrictModel):
    status: Literal["SUCCESS", "ERROR"]
    code: int
    message: str
    data: AnalyzeSequenceData | None
