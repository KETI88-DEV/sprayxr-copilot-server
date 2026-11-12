"""Strict wire contracts for the corrected engine HTTP service."""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


EventType = Literal[
    "session_start",
    "sequence",
    "zone_observation",
    "session_end",
    "session_complete",
]
CONTRACT_VERSION = "sprayxr-corrected-zone/v2-draft-20261106"
HeightEncoding = Literal["rg16_b8", "legacy_rg_positive"]
HeightBasis = Literal["current_round", "cumulative_crop"]


def parse_utc(value: str, *, field_name: str) -> datetime:
    """Parse an ISO timestamp whose declared offset is exactly UTC."""
    if not (value.endswith("Z") or value.endswith("+00:00")):
        raise ValueError(f"{field_name} must use UTC (+00:00 or Z)")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field_name} must include a UTC offset")
    if parsed.utcoffset() != timedelta(0):
        raise ValueError(f"{field_name} must use UTC (+00:00 or Z)")
    return parsed


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class EventRequest(StrictModel):
    event_id: str = Field(min_length=1, max_length=200)
    event_type: EventType
    session_id: str = Field(min_length=1, max_length=200)
    payload: dict[str, Any]

    @field_validator("event_id", "session_id")
    @classmethod
    def no_surrounding_whitespace(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("identifier must not contain surrounding whitespace")
        return value


class ModelMapping(StrictModel):
    provider_model_key: str = Field(min_length=1)
    work_id: str = Field(min_length=1)
    session_family_id: str = Field(min_length=1)
    round_index: int = Field(ge=1)
    round_id: str = Field(min_length=1)
    geometry_model_key: str = Field(min_length=1)


class ZoneManifestRef(StrictModel):
    manifest_id: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_key: str = Field(min_length=1)
    map_nums: list[int] = Field(min_length=1)

    @field_validator("map_nums")
    @classmethod
    def maps_are_positive_and_unique(cls, values: list[int]) -> list[int]:
        if any(value < 1 for value in values):
            raise ValueError("map numbers must be positive")
        if len(set(values)) != len(values):
            raise ValueError("map numbers must be unique")
        return values


class ExpectedZoneSet(StrictModel):
    face_id: str = Field(min_length=1)
    map_num: int = Field(ge=1)
    zone_ids: list[str] = Field(min_length=1)

    @field_validator("zone_ids")
    @classmethod
    def zone_ids_are_unique(cls, values: list[str]) -> list[str]:
        if any(not value or value != value.strip() for value in values):
            raise ValueError("zone IDs must be non-empty canonical strings")
        if len(set(values)) != len(values):
            raise ValueError("zone IDs must be unique within a face")
        return values


class SessionStartPayload(StrictModel):
    source_instance_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    started_at: str
    evaluation_mode: Literal["corrected_zone_v2"]
    contract_version: Literal["sprayxr-corrected-zone/v2-draft-20261106"]
    target_thickness_um: float | None
    target_source: str | None
    model_mapping: ModelMapping
    zone_manifest: ZoneManifestRef
    expected_zones: list[ExpectedZoneSet] = Field(default_factory=list)

    @field_validator("started_at")
    @classmethod
    def valid_started_at(cls, value: str) -> str:
        parse_utc(value, field_name="started_at")
        return value

    @field_validator("target_thickness_um")
    @classmethod
    def valid_target(cls, value: float | None) -> float | None:
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError("target_thickness_um must be finite and positive")
        return value

    @model_validator(mode="after")
    def target_source_matches_target(self) -> "SessionStartPayload":
        if self.target_thickness_um is None and self.target_source is not None:
            raise ValueError("target_source requires target_thickness_um")
        if self.target_thickness_um is not None and not self.target_source:
            raise ValueError("target_source is required with target_thickness_um")
        pairs = {(item.face_id, item.map_num) for item in self.expected_zones}
        if len(pairs) != len(self.expected_zones):
            raise ValueError("expected_zones face/map entries must be unique")
        return self


class TextureUV(StrictModel):
    x: float
    y: float

    @field_validator("x", "y")
    @classmethod
    def finite_coordinate(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("UV coordinates must be finite")
        return value


class SequenceFrame(StrictModel):
    timestamp: str | None = None
    relative_time_s: float | None = Field(default=None, ge=0)
    zone_id: str | None = None
    spray_on: bool
    speed_mm_s: float | None = Field(default=None, ge=0)
    distance_mm: float | None = Field(default=None, ge=0)
    raytexmapnum: int | None = Field(default=None, ge=1)
    raytexuv: TextureUV | None = None

    @field_validator("timestamp")
    @classmethod
    def valid_timestamp(cls, value: str | None) -> str | None:
        if value is not None:
            parse_utc(value, field_name="frame.timestamp")
        return value

    @field_validator("relative_time_s", "speed_mm_s", "distance_mm")
    @classmethod
    def finite_number(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("numeric frame values must be finite")
        return value

    @model_validator(mode="after")
    def exactly_one_clock(self) -> "SequenceFrame":
        if (self.timestamp is None) == (self.relative_time_s is None):
            raise ValueError("frame requires exactly one of timestamp or relative_time_s")
        if self.zone_id is not None and (
            not self.zone_id or self.zone_id != self.zone_id.strip()
        ):
            raise ValueError("zone_id must be a non-empty canonical string")
        return self


class CropGeometry(StrictModel):
    x0: int = Field(ge=0)
    y0: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    atlas_width: int = Field(gt=0)
    atlas_height: int = Field(gt=0)

    @model_validator(mode="after")
    def crop_within_atlas(self) -> "CropGeometry":
        if self.x0 + self.width > self.atlas_width:
            raise ValueError("crop extends beyond atlas width")
        if self.y0 + self.height > self.atlas_height:
            raise ValueError("crop extends beyond atlas height")
        return self


class HeightPng(StrictModel):
    data_base64: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    encoding: HeightEncoding
    basis: HeightBasis
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    origin: Literal["top_left"]
    crop: CropGeometry

    @model_validator(mode="after")
    def declared_image_matches_crop(self) -> "HeightPng":
        if (self.width, self.height) != (self.crop.width, self.crop.height):
            raise ValueError("height PNG dimensions must equal the native crop dimensions")
        return self


class CumulativeCrop(StrictModel):
    snapshot_id: str = Field(min_length=1)
    snapshot_revision: int = Field(ge=1)
    height_png: HeightPng

    @model_validator(mode="after")
    def cumulative_basis_only(self) -> "CumulativeCrop":
        if self.height_png.basis != "cumulative_crop":
            raise ValueError("sequence cumulative_crop requires cumulative_crop basis")
        return self


class SequencePayload(StrictModel):
    sequence_id: str = Field(min_length=1)
    face_id: str = Field(min_length=1)
    map_num: int = Field(ge=1)
    round_id: str = Field(min_length=1)
    captured_at: str
    sequence_started_at: str | None = None
    frames: list[SequenceFrame] = Field(min_length=1)
    cumulative_crop: CumulativeCrop | None = None

    @field_validator("captured_at", "sequence_started_at")
    @classmethod
    def valid_clock(cls, value: str | None, info: Any) -> str | None:
        if value is not None:
            parse_utc(value, field_name=info.field_name)
        return value

    @model_validator(mode="after")
    def relative_frames_need_anchor(self) -> "SequencePayload":
        if any(frame.relative_time_s is not None for frame in self.frames):
            if self.sequence_started_at is None:
                raise ValueError("relative frames require sequence_started_at")
        return self


class ZoneObservationPayload(StrictModel):
    face_id: str = Field(min_length=1)
    map_num: int = Field(ge=1)
    zone_id: str = Field(min_length=1)
    round_id: str = Field(min_length=1)
    snapshot_id: str = Field(min_length=1)
    snapshot_revision: int = Field(ge=1)
    captured_at: str
    height_png: HeightPng

    @field_validator("captured_at")
    @classmethod
    def valid_captured_at(cls, value: str) -> str:
        parse_utc(value, field_name="captured_at")
        return value


class SessionEndPayload(StrictModel):
    ended_at: str

    @field_validator("ended_at")
    @classmethod
    def valid_ended_at(cls, value: str) -> str:
        parse_utc(value, field_name="ended_at")
        return value


class ProviderOfficialSummary(StrictModel):
    avg_thickness_um: float | None
    source: str = Field(min_length=1)
    unit: Literal["um"]
    progress_value: float | None
    progress_unit: Literal["percent_0_100", "fraction_0_1"] | None

    @field_validator("avg_thickness_um", "progress_value")
    @classmethod
    def finite_optional(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("provider values must be finite")
        return value

    @model_validator(mode="after")
    def progress_value_and_unit_match(self) -> "ProviderOfficialSummary":
        if (self.progress_value is None) != (self.progress_unit is None):
            raise ValueError("provider progress value and unit must be supplied together")
        if self.progress_value is not None:
            maximum = 1.0 if self.progress_unit == "fraction_0_1" else 100.0
            if not 0.0 <= self.progress_value <= maximum:
                raise ValueError("provider progress is outside the declared unit range")
        return self


class SessionCompletePayload(StrictModel):
    completed_at: str
    provider_official_summary: ProviderOfficialSummary

    @field_validator("completed_at")
    @classmethod
    def valid_completed_at(cls, value: str) -> str:
        parse_utc(value, field_name="completed_at")
        return value


PAYLOAD_MODELS: dict[str, type[StrictModel]] = {
    "session_start": SessionStartPayload,
    "sequence": SequencePayload,
    "zone_observation": ZoneObservationPayload,
    "session_end": SessionEndPayload,
    "session_complete": SessionCompletePayload,
}


def parse_payload(request: EventRequest) -> StrictModel:
    return PAYLOAD_MODELS[request.event_type].model_validate(request.payload)
