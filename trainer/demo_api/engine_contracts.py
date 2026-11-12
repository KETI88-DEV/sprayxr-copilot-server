"""Strict contracts for the opt-in corrected Zone v2 integration.

The public v2 draft and the internal engine wire are deliberately separate from
the legacy v1/v2.5 coating contracts.  A session pins ``evaluation_mode`` and
``contract_version`` at start; later requests repeat the version so accidental
cross-mode use fails at the boundary instead of silently changing semantics.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


EVALUATION_MODE = "corrected_zone_v2"
CONTRACT_VERSION = "sprayxr-corrected-zone/v2-draft-20261106"
SOURCE_INSTANCE_ENV = "CORRECTED_SOURCE_INSTANCE_ID"

Identifier = Annotated[str, Field(min_length=1, max_length=160, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:/-]*$")]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
FiniteNonNegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]
PositiveInt = Annotated[int, Field(ge=1, strict=True)]
NonNegativeInt = Annotated[int, Field(ge=0, strict=True)]


def canonical_json(value: object) -> str:
    """Return the one canonical representation used for persistence and hashes."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def validate_aware_iso(value: str, *, field_name: str) -> str:
    """Require an ISO-8601 timestamp with an explicit UTC offset."""

    if not value.endswith(("Z", "+00:00")):
        raise ValueError(f"{field_name} must end in Z or +00:00")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO-8601 timestamp") from exc
    offset = parsed.utcoffset()
    if parsed.tzinfo is None or offset is None:
        raise ValueError(f"{field_name} must include a UTC offset")
    if offset.total_seconds() != 0:  # defensive after the lexical suffix check
        raise ValueError(f"{field_name} must be UTC")
    return value


class StrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelMapping(StrictContract):
    provider_model_key: Identifier
    work_id: Identifier
    session_family_id: Identifier
    round_index: PositiveInt
    round_id: Identifier
    geometry_model_key: Identifier


class ZoneManifest(StrictContract):
    manifest_id: Identifier
    sha256: Sha256
    model_key: Identifier
    map_nums: Annotated[list[PositiveInt], Field(min_length=1)]

    @model_validator(mode="after")
    def unique_maps(self) -> "ZoneManifest":
        if len(set(self.map_nums)) != len(self.map_nums):
            raise ValueError("zone_manifest.map_nums must be unique")
        return self


class ExpectedZones(StrictContract):
    face_id: Identifier
    map_num: PositiveInt
    zone_ids: Annotated[list[Identifier], Field(min_length=1)]

    @model_validator(mode="after")
    def unique_zones(self) -> "ExpectedZones":
        if len(set(self.zone_ids)) != len(self.zone_ids):
            raise ValueError("expected_zones.zone_ids must be unique")
        return self


class CorrectedRequest(StrictContract):
    event_id: Identifier
    session_id: Identifier
    contract_version: Literal[CONTRACT_VERSION]


class SessionStartRequest(CorrectedRequest):
    user_id: Identifier
    scenario_id: Identifier
    started_at: str
    evaluation_mode: Literal[EVALUATION_MODE]
    target_thickness_um: FiniteNonNegative | None = None
    target_source: Annotated[str, Field(min_length=1, max_length=160)] | None = None
    model_mapping: ModelMapping
    zone_manifest: ZoneManifest
    expected_zones: list[ExpectedZones] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_start(self) -> "SessionStartRequest":
        validate_aware_iso(self.started_at, field_name="started_at")
        if self.target_thickness_um == 0:
            raise ValueError("target_thickness_um must be positive when supplied")
        if self.target_thickness_um is not None and self.target_source is None:
            raise ValueError("target_source is required with target_thickness_um")
        if self.target_thickness_um is None and self.target_source is not None:
            raise ValueError("target_source requires target_thickness_um")
        keys = [(item.face_id, item.map_num) for item in self.expected_zones]
        if len(set(keys)) != len(keys):
            raise ValueError("expected_zones face_id/map_num entries must be unique")
        return self


class RaytexUv(StrictContract):
    x: float = Field(allow_inf_nan=False)
    y: float = Field(allow_inf_nan=False)


class SequenceFrame(StrictContract):
    timestamp: str | None = None
    relative_time_s: FiniteNonNegative | None = None
    zone_id: Identifier | None = None
    spray_on: bool
    speed_mm_s: FiniteNonNegative | None = None
    distance_mm: FiniteNonNegative | None = None
    raytexmapnum: PositiveInt | None = None
    raytexuv: RaytexUv | None = None

    @model_validator(mode="after")
    def exactly_one_clock(self) -> "SequenceFrame":
        if (self.timestamp is None) == (self.relative_time_s is None):
            raise ValueError("exactly one of timestamp or relative_time_s is required")
        if self.timestamp is not None:
            validate_aware_iso(self.timestamp, field_name="timestamp")
        if (self.raytexmapnum is None) != (self.raytexuv is None):
            raise ValueError("raytexmapnum and raytexuv must be supplied together")
        return self


class HeightCrop(StrictContract):
    x0: NonNegativeInt
    y0: NonNegativeInt
    width: PositiveInt
    height: PositiveInt
    atlas_width: PositiveInt
    atlas_height: PositiveInt

    @model_validator(mode="after")
    def within_atlas(self) -> "HeightCrop":
        if self.x0 + self.width > self.atlas_width or self.y0 + self.height > self.atlas_height:
            raise ValueError("height crop exceeds atlas bounds")
        return self


class HeightPng(StrictContract):
    data_base64: Annotated[str, Field(min_length=4)]
    sha256: Sha256
    encoding: Literal["rg16_b8", "legacy_rg_positive"]
    basis: Literal["current_round", "cumulative_crop"]
    width: PositiveInt
    height: PositiveInt
    origin: Literal["top_left"]
    crop: HeightCrop

    @model_validator(mode="after")
    def validate_bytes_and_geometry(self) -> "HeightPng":
        try:
            raw = base64.b64decode(self.data_base64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("height_png.data_base64 is not canonical base64") from exc
        if not raw:
            raise ValueError("height_png must not be empty")
        if hashlib.sha256(raw).hexdigest() != self.sha256:
            raise ValueError("height_png.sha256 does not match decoded bytes")
        if (self.width, self.height) != (self.crop.width, self.crop.height):
            raise ValueError("height_png dimensions must match crop dimensions")
        return self


class CumulativeCrop(StrictContract):
    snapshot_id: Identifier
    snapshot_revision: PositiveInt
    height_png: HeightPng

    @model_validator(mode="after")
    def cumulative_basis_only(self) -> "CumulativeCrop":
        if self.height_png.basis != "cumulative_crop":
            raise ValueError("sequence cumulative_crop must use basis=cumulative_crop")
        return self


class SequenceRequest(CorrectedRequest):
    sequence_id: Identifier
    face_id: Identifier
    map_num: PositiveInt
    round_id: Identifier
    captured_at: str
    sequence_started_at: str | None = None
    frames: Annotated[list[SequenceFrame], Field(min_length=1)]
    cumulative_crop: CumulativeCrop | None = None

    @model_validator(mode="after")
    def validate_capture(self) -> "SequenceRequest":
        validate_aware_iso(self.captured_at, field_name="captured_at")
        if self.sequence_started_at is not None:
            validate_aware_iso(self.sequence_started_at, field_name="sequence_started_at")
        if any(frame.relative_time_s is not None for frame in self.frames):
            if self.sequence_started_at is None:
                raise ValueError("relative frames require sequence_started_at")
        return self


class ZoneObservationRequest(CorrectedRequest):
    observation_id: Identifier
    face_id: Identifier
    map_num: PositiveInt
    zone_id: Identifier
    round_id: Identifier
    snapshot_id: Identifier
    snapshot_revision: PositiveInt
    captured_at: str
    height_png: HeightPng
    provider_official_mean_um: FiniteNonNegative | None = None

    @model_validator(mode="after")
    def validate_capture(self) -> "ZoneObservationRequest":
        validate_aware_iso(self.captured_at, field_name="captured_at")
        return self


class SessionEndRequest(CorrectedRequest):
    ended_at: str

    @model_validator(mode="after")
    def validate_end(self) -> "SessionEndRequest":
        validate_aware_iso(self.ended_at, field_name="ended_at")
        return self


class ProviderOfficialSummary(StrictContract):
    avg_thickness_um: FiniteNonNegative | None
    source: Annotated[str, Field(min_length=1, max_length=160)]
    unit: Literal["um"]
    progress_value: FiniteNonNegative | None
    progress_unit: Literal["percent_0_100", "fraction_0_1"] | None

    @model_validator(mode="after")
    def progress_pair_and_range(self) -> "ProviderOfficialSummary":
        if (self.progress_value is None) != (self.progress_unit is None):
            raise ValueError("progress_value and progress_unit must be supplied together")
        if self.progress_value is not None:
            upper = 100.0 if self.progress_unit == "percent_0_100" else 1.0
            if self.progress_value > upper:
                raise ValueError("progress_value exceeds its declared unit range")
        return self


class SessionCompleteRequest(CorrectedRequest):
    completed_at: str
    provider_official_summary: ProviderOfficialSummary

    @model_validator(mode="after")
    def validate_complete(self) -> "SessionCompleteRequest":
        validate_aware_iso(self.completed_at, field_name="completed_at")
        return self


EngineEventType = Literal[
    "session_start", "sequence", "zone_observation", "session_end", "session_complete"
]


class EngineEventEnvelope(StrictContract):
    event_id: Identifier
    event_type: EngineEventType
    session_id: Identifier
    payload: dict[str, JsonValue]


class EngineEventAck(StrictContract):
    status: Literal["accepted", "duplicate"]
    event_id: Identifier
    event_type: EngineEventType
    session_id: Identifier
    payload_hash: Sha256
    assessment_id: Identifier | None
    session_revision: NonNegativeInt
    processing_status: Annotated[str, Field(min_length=1, max_length=80)]
    evidence_ids: list[Identifier]


class SixBandCounts(StrictContract):
    lt_0_9t: NonNegativeInt
    from_0_9t_to_lt_1_0t: NonNegativeInt
    from_1_0t_to_lt_1_5t: NonNegativeInt
    from_1_5t_to_lt_2_0t: NonNegativeInt
    from_2_0t_to_lt_2_5t: NonNegativeInt
    gte_2_5t: NonNegativeInt


class ObservedStatistics(StrictContract):
    painted_mean_um: FiniteNonNegative | None
    painted_sum_um: FiniteNonNegative
    painted_pixel_count: NonNegativeInt
    unpainted_pixel_count: NonNegativeInt
    six_band_counts: SixBandCounts | None


class AssessmentCoverage(StrictContract):
    zone_pixel_count: NonNegativeInt
    observed_zone_pixel_count: NonNegativeInt
    fraction: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    full_zone: bool


class BehaviorSummary(StrictContract):
    count: PositiveInt
    mean: FiniteNonNegative
    minimum: FiniteNonNegative
    maximum: FiniteNonNegative
    median: FiniteNonNegative


class BehaviorStatistics(StrictContract):
    speed_mm_s: BehaviorSummary | None
    distance_mm: BehaviorSummary | None


class AssessmentMetrics(StrictContract):
    observed_statistics: ObservedStatistics | None
    coverage: AssessmentCoverage
    provider_official_summary: ProviderOfficialSummary | None
    behavior_statistics: BehaviorStatistics | None = None


class QualityResult(StrictContract):
    analysis_status: Literal["available", "partial", "unavailable"]
    reason_codes: list[str]
    grade: Literal["below", "normal", "above"] | None
    target_thickness_um: FiniteNonNegative | None
    painted_mean_um: FiniteNonNegative | None
    painted_pixel_count: NonNegativeInt
    zone_pixel_count: NonNegativeInt
    unpainted_fraction: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)] | None
    six_band_counts: SixBandCounts | None
    statistics_scope: Literal["whole_zone", "observed_only", "none"]


class CoachingCause(StrictContract):
    cause_id: Identifier
    statement: Annotated[str, Field(min_length=1)]
    variable: Literal["speed_mm_s", "distance_mm"]
    evidence_ids: list[Identifier]
    evidence_kind: Literal["quality_rule_candidate"]
    causal_status: Literal["candidate_not_established"]


class CoachingAction(StrictContract):
    action_id: Identifier
    statement: Annotated[str, Field(min_length=1)]
    variable: Literal["speed_mm_s", "distance_mm"]
    direction: Literal["increase", "decrease", "maintain"]
    priority: PositiveInt
    evidence_ids: list[Identifier]
    evidence_kind: Literal["quality_rule_candidate"]
    target_range: None
    expected_effect: None


class AssessmentCoaching(StrictContract):
    analysis_status: Literal["available", "withheld"]
    reason_code: str | None
    causes: list[CoachingCause]
    actions: list[CoachingAction]
    primary_action_id: Identifier | None

    @model_validator(mode="after")
    def valid_primary_action(self) -> "AssessmentCoaching":
        action_ids = {action.action_id for action in self.actions}
        if self.primary_action_id is not None and self.primary_action_id not in action_ids:
            raise ValueError("primary_action_id is not present in actions")
        return self


class AssessmentEvidence(StrictContract):
    evidence_id: Identifier
    evidence_type: Literal["quality_measurement", "behavior_observation", "lineage"]
    statement: Annotated[str, Field(min_length=1)]
    value: float | str | None
    unit: str | None
    source_event_ids: list[Identifier]
    contract_version: Literal[CONTRACT_VERSION]
    revision: PositiveInt


class AssessmentZoneKey(StrictContract):
    model_key: Identifier
    face_id: Identifier
    map_num: PositiveInt
    zone_id: Identifier


class AssessmentProvenance(StrictContract):
    engine_version: Annotated[str, Field(min_length=1, max_length=200)]
    criterion_version: Annotated[str, Field(min_length=1, max_length=200)]
    coaching_rule_version: Annotated[str, Field(min_length=1, max_length=200)]
    contract_version: Literal[CONTRACT_VERSION]
    manifest_id: Identifier
    zone_asset_manifest_sha256: Sha256
    label_map_sha256: Sha256 | None
    mask_sha256: Sha256 | None
    source_event_ids: list[Identifier]
    assessment_input_manifest_sha256: Sha256
    snapshot_id: Identifier | None
    snapshot_revision: PositiveInt | None
    captured_at: str | None

    @model_validator(mode="after")
    def validate_capture(self) -> "AssessmentProvenance":
        if self.captured_at is not None:
            validate_aware_iso(self.captured_at, field_name="captured_at")
        return self


class ZoneAssessmentSummary(StrictContract):
    assessment_id: Identifier
    revision: PositiveInt
    zone_key: AssessmentZoneKey
    status: Literal["available", "partial", "unavailable"]
    quality: QualityResult


class MissingZone(StrictContract):
    face_id: Identifier
    map_num: PositiveInt
    zone_id: Identifier


class PendingZone(MissingZone):
    assessment_id: Identifier
    status: Literal["available", "partial", "unavailable"]


class MissingOrPendingZones(StrictContract):
    missing: list[MissingZone]
    pending: list[PendingZone]


class EngineAssessment(StrictContract):
    assessment_id: Identifier
    session_id: Identifier
    revision: PositiveInt
    assessment_type: Literal["zone", "session"]
    status: Literal["available", "partial", "unavailable"]
    evidence_ids: list[Identifier]
    metrics: AssessmentMetrics
    coaching: AssessmentCoaching
    quality: QualityResult
    evidence: list[AssessmentEvidence]
    zone_key: AssessmentZoneKey | None
    provenance: AssessmentProvenance
    zone_assessments: list[ZoneAssessmentSummary] | None = None
    missing_or_pending_zones: MissingOrPendingZones | None = None
    created_at: str

    @model_validator(mode="after")
    def validate_created(self) -> "EngineAssessment":
        validate_aware_iso(self.created_at, field_name="created_at")
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("assessment evidence_ids must be unique")
        embedded = {item.evidence_id for item in self.evidence}
        if set(self.evidence_ids) != embedded:
            raise ValueError("assessment evidence_ids must match embedded evidence")
        coaching_ids = {
            evidence_id
            for cause in self.coaching.causes
            for evidence_id in cause.evidence_ids
        } | {
            evidence_id
            for action in self.coaching.actions
            for evidence_id in action.evidence_ids
        }
        if not coaching_ids.issubset(embedded):
            raise ValueError("coaching references evidence outside the assessment")
        if self.assessment_type == "zone":
            if self.zone_key is None or self.metrics.behavior_statistics is None:
                raise ValueError("zone assessment requires zone_key and behavior_statistics")
            if any(item.revision != self.revision for item in self.evidence):
                raise ValueError("zone assessment evidence revision mismatch")
            if self.zone_assessments is not None or self.missing_or_pending_zones is not None:
                raise ValueError("zone assessment cannot contain session aggregation fields")
        else:
            if self.zone_key is not None or self.metrics.behavior_statistics is not None:
                raise ValueError("session assessment cannot contain zone_key or behavior_statistics")
            if self.zone_assessments is None or self.missing_or_pending_zones is None:
                raise ValueError("session assessment requires aggregation fields")
        return self


class ExplanationModel(StrictContract):
    name: str
    digest: str
    backend: str
    quantization: str | None


class ExplanationValidation(StrictContract):
    passed: bool
    reason_codes: list[str]
    renderer: str
    free_form_prose_generated: bool


class ExplanationResponse(StrictContract):
    status: Literal["generated", "fallback"]
    text: Annotated[str, Field(min_length=1)]
    model: ExplanationModel
    evidence_ids: list[Identifier]
    selected_action_ids: list[Identifier]
    selected_cause_ids: list[Identifier]
    validation: ExplanationValidation
    inference: dict[str, JsonValue]
    audit_id: Identifier
    assessment_id: Identifier
    revision: PositiveInt
    kind: Literal["zone", "session"]

    @model_validator(mode="after")
    def validate_evidence_ids(self) -> "ExplanationResponse":
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("explanation evidence_ids must be unique")
        return self


class ProcessingStatus(StrictContract):
    source_cursor: PositiveInt
    outbox_status: Literal[
        "pending", "processing", "failed_retryable", "delivered", "rejected", "blocked"
    ]
    attempts: NonNegativeInt
    last_error: str | None
    last_confirmed_engine_cursor: NonNegativeInt
    session_revision: NonNegativeInt


class CoatingProgressScope(StrictContract):
    """One engine-confirmed cumulative surface observation.

    This is deliberately an observed state, not an estimate of work-area
    completion.  The corrected engine can maintain absolute cumulative crop
    pixels without knowing the provider's full target mask denominator.
    """

    face_id: Identifier
    map_num: PositiveInt
    known_pixel_count: NonNegativeInt
    raw_sum_um: FiniteNonNegative
    source_event_ids: list[Identifier]
    semantics: Literal["absolute_cumulative_overwrite"]

    @model_validator(mode="after")
    def unique_source_events(self) -> "CoatingProgressScope":
        if len(set(self.source_event_ids)) != len(self.source_event_ids):
            raise ValueError("coating progress source_event_ids must be unique")
        return self


class CoatingProgress(StrictContract):
    """Lightweight sequence progress with an explicit completion limitation."""

    status: Literal["maintained", "pending", "unavailable"]
    basis: Literal["absolute_cumulative_overwrite"] | None
    source_cursor: NonNegativeInt
    confirmed_through_source_cursor: NonNegativeInt
    includes_current_source_cursor: bool
    known_pixel_count: NonNegativeInt
    raw_sum_um: FiniteNonNegative | None
    source_event_ids: list[Identifier]
    scopes: list[CoatingProgressScope]
    completion_percent: None = None
    target_pixel_count: None = None
    completion_unit: None = None
    observation_reason_code: Literal[
        "engine_processing", "cumulative_crop_unavailable", "engine_state_invalid"
    ] | None
    completion_reason_code: Literal["work_target_denominator_unavailable"]
    semantics: Literal["absolute_cumulative_observation_not_area_completion"]

    @model_validator(mode="after")
    def validate_state(self) -> "CoatingProgress":
        if len(set(self.source_event_ids)) != len(self.source_event_ids):
            raise ValueError("coating progress source_event_ids must be unique")
        scope_keys = {(item.face_id, item.map_num) for item in self.scopes}
        if len(scope_keys) != len(self.scopes):
            raise ValueError("coating progress scopes must be unique per face/map")
        if self.status == "maintained":
            if not self.scopes or self.basis is None or self.raw_sum_um is None:
                raise ValueError("maintained coating progress requires cumulative scopes")
            if self.observation_reason_code is not None:
                raise ValueError("maintained coating progress cannot have an observation reason")
        else:
            if self.scopes or self.basis is not None or self.raw_sum_um is not None:
                raise ValueError("unavailable coating progress cannot contain cumulative state")
            if self.known_pixel_count != 0 or self.source_event_ids:
                raise ValueError("unavailable coating progress must have empty observations")
            if self.observation_reason_code is None:
                raise ValueError("unavailable coating progress requires a reason")
        return self


def coating_progress_from_engine_state(
    engine_state: dict[str, JsonValue] | None,
    *,
    source_cursor: int,
    confirmed_through_source_cursor: int,
) -> CoatingProgress:
    """Project persisted engine state without inventing an area denominator."""

    raw_scopes: object = [] if engine_state is None else engine_state.get(
        "cumulative_crop_states", []
    )
    invalid = not isinstance(raw_scopes, list)
    scopes: list[CoatingProgressScope] = []
    if not invalid:
        try:
            scopes = [CoatingProgressScope.model_validate(item) for item in raw_scopes]
        except (TypeError, ValueError):
            invalid = True
            scopes = []

    includes_current = confirmed_through_source_cursor >= source_cursor
    if scopes:
        event_ids = list(
            dict.fromkeys(
                event_id
                for scope in scopes
                for event_id in scope.source_event_ids
            )
        )
        return CoatingProgress(
            status="maintained",
            basis="absolute_cumulative_overwrite",
            source_cursor=source_cursor,
            confirmed_through_source_cursor=confirmed_through_source_cursor,
            includes_current_source_cursor=includes_current,
            known_pixel_count=sum(item.known_pixel_count for item in scopes),
            raw_sum_um=math.fsum(item.raw_sum_um for item in scopes),
            source_event_ids=event_ids,
            scopes=scopes,
            observation_reason_code=None,
            completion_reason_code="work_target_denominator_unavailable",
            semantics="absolute_cumulative_observation_not_area_completion",
        )

    if invalid:
        reason = "engine_state_invalid"
        status = "unavailable"
    elif not includes_current:
        reason = "engine_processing"
        status = "pending"
    else:
        reason = "cumulative_crop_unavailable"
        status = "unavailable"
    return CoatingProgress(
        status=status,
        basis=None,
        source_cursor=source_cursor,
        confirmed_through_source_cursor=confirmed_through_source_cursor,
        includes_current_source_cursor=includes_current,
        known_pixel_count=0,
        raw_sum_um=None,
        source_event_ids=[],
        scopes=[],
        observation_reason_code=reason,
        completion_reason_code="work_target_denominator_unavailable",
        semantics="absolute_cumulative_observation_not_area_completion",
    )


class AcceptedEventResponse(StrictContract):
    status: Literal["accepted", "duplicate"]
    evaluation_mode: Literal[EVALUATION_MODE]
    contract_version: Literal[CONTRACT_VERSION]
    event_id: Identifier
    event_type: EngineEventType
    session_id: Identifier
    processing: ProcessingStatus
    assessment_id: Identifier | None = None
    assessment_revision: PositiveInt | None = None
    coating_progress: CoatingProgress
    engine_state: dict[str, JsonValue] | None = None


class SessionStatusResponse(StrictContract):
    session_id: Identifier
    evaluation_mode: Literal[EVALUATION_MODE]
    contract_version: Literal[CONTRACT_VERSION]
    lifecycle_status: Literal["active", "ended", "complete"]
    source_cursor: NonNegativeInt
    last_source_event_id: Identifier | None
    last_source_event_type: EngineEventType | None
    last_confirmed_engine_cursor: NonNegativeInt
    last_engine_event_id: Identifier | None
    last_engine_event_type: EngineEventType | None
    session_revision: NonNegativeInt
    latest_assessment_id: Identifier | None
    latest_assessment_revision: PositiveInt | None
    engine_outbox: dict[str, NonNegativeInt]
    explanation_outbox: dict[str, NonNegativeInt]
    coating_progress: CoatingProgress
    engine_state: dict[str, JsonValue] | None


class AssessmentResponse(StrictContract):
    assessment: EngineAssessment
    explanation: ExplanationResponse | None
    explanation_status: Literal["pending", "processing", "failed_retryable", "delivered", "rejected"]


class SessionReportResponse(StrictContract):
    session_id: Identifier
    evaluation_mode: Literal[EVALUATION_MODE]
    contract_version: Literal[CONTRACT_VERSION]
    official_summary: ProviderOfficialSummary
    assessment: EngineAssessment | None
    explanation: ExplanationResponse | None
    report_status: Literal["pending", "available", "failed_retryable", "rejected"]


class CopilotQueryRequest(StrictContract):
    session_id: Identifier
    assessment_id: Identifier
    revision: PositiveInt
    question: Annotated[str, Field(min_length=1, max_length=2000)]


class CopilotQueryResponse(StrictContract):
    session_id: Identifier
    assessment_id: Identifier
    revision: PositiveInt
    assessment_type: Literal["zone", "session"]
    answer_status: Literal["generated", "fallback", "pending"]
    answer: str
    evidence_ids: list[Identifier]
    explanation_validation: ExplanationValidation | None


def request_logical_body(body: CorrectedRequest) -> dict[str, JsonValue]:
    """Body used for natural-key deduplication, excluding caller event identity."""

    dumped = body.model_dump(mode="json")
    dumped.pop("event_id", None)
    return dumped  # type: ignore[return-value]


def engine_event_from_request(
    body: CorrectedRequest,
    *,
    source_instance_id: str,
) -> EngineEventEnvelope:
    """Project one public request onto the fixed internal engine contract."""

    data = body.model_dump(mode="json")
    common = {"event_id": data.pop("event_id"), "session_id": data.pop("session_id")}
    if isinstance(body, SessionStartRequest):
        event_type: EngineEventType = "session_start"
        data = {"source_instance_id": source_instance_id, **data}
    elif isinstance(body, SequenceRequest):
        event_type = "sequence"
        data.pop("contract_version", None)
    elif isinstance(body, ZoneObservationRequest):
        event_type = "zone_observation"
        data.pop("contract_version", None)
        data.pop("observation_id", None)
        data.pop("provider_official_mean_um", None)
    elif isinstance(body, SessionEndRequest):
        event_type = "session_end"
        data.pop("contract_version", None)
    elif isinstance(body, SessionCompleteRequest):
        event_type = "session_complete"
        data.pop("contract_version", None)
    else:  # pragma: no cover - the closed union above owns every caller
        raise TypeError(f"unsupported corrected request: {type(body).__name__}")
    return EngineEventEnvelope(
        event_id=common["event_id"],
        event_type=event_type,
        session_id=common["session_id"],
        payload=data,
    )


def natural_key(body: CorrectedRequest) -> str:
    if isinstance(body, SessionStartRequest):
        return "start"
    if isinstance(body, SequenceRequest):
        return f"sequence:{body.sequence_id}"
    if isinstance(body, ZoneObservationRequest):
        return f"observation:{body.observation_id}"
    if isinstance(body, SessionEndRequest):
        return "end"
    if isinstance(body, SessionCompleteRequest):
        return "complete"
    raise TypeError(type(body).__name__)


def captured_at(body: CorrectedRequest) -> str | None:
    if isinstance(body, SequenceRequest | ZoneObservationRequest):
        return body.captured_at
    if isinstance(body, SessionEndRequest):
        return body.ended_at
    if isinstance(body, SessionCompleteRequest):
        return body.completed_at
    if isinstance(body, SessionStartRequest):
        return body.started_at
    return None


def ensure_finite_json(value: JsonValue) -> None:
    """Defence in depth for service responses parsed from untrusted JSON."""

    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite JSON number")
    if isinstance(value, list):
        for item in value:
            ensure_finite_json(item)
    elif isinstance(value, dict):
        for item in value.values():
            ensure_finite_json(item)
