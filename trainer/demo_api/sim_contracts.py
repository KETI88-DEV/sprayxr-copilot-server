from __future__ import annotations

from typing import Annotated, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

Phase = Literal["start", "runtime", "session_end"]
Direction = Literal["input", "output"]


class StrictModel(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")


class WorkerProfile(StrictModel):
    trainee_id: str
    skill_level: Literal["beginner", "intermediate", "advanced"]
    handedness: Literal["left", "right"]


class PaintSpec(StrictModel):
    nominal_um: float


class SessionStartInput(StrictModel):
    session_id: Annotated[str, Field(min_length=1)]
    level_id: Annotated[str, Field(min_length=1)]
    worker: WorkerProfile
    spec: PaintSpec


class GunMetrics(StrictModel):
    speed_m_per_s: float
    standoff_mm: float
    angle_deg: float


class EventTelemetry(StrictModel):
    gun_metrics: GunMetrics | None = None
    pose_joints: dict[str, float] | None = None
    coordinates: list[float] | None = None


class SessionEventInput(StrictModel):
    event_id: Annotated[str, Field(min_length=1)]
    user_request: str | None = None
    telemetry: EventTelemetry | None = None
    zone_id: str | None = None


class SessionStartOutput(StrictModel):
    session_id: str
    recommended_flow: list[str]
    flow_source: Literal["rule"]
    guidance: str


class SessionEventOutput(StrictModel):
    session_id: str
    event_id: str
    kind: Literal["rag", "anomaly", "ack"]
    message: str
    anomaly: dict[str, object] | None = None


class SessionReviewOutput(StrictModel):
    session_id: str
    summary: str
    risk_flags: list[dict[str, object]]
    action_items: list[dict[str, str]]
    citations: list[str]


class MonitorEvent(StrictModel):
    seq: int
    session_id: str
    phase: Phase
    direction: Direction
    timestamp: str
    payload: dict[str, object]


class MonitorSnapshot(StrictModel):
    events: list[MonitorEvent]
    next_cursor: int
    dropped_before: int | None = None


class SimSessionView(StrictModel):
    session_id: str
    level_id: str
    phase: Phase
    recommended_flow: list[str]
    current_zone_id: str
    completed_zone_ids: list[str]
    last_event_id: str | None
    anomaly_count: int
    unresolved_zone_id: str | None
    updated_at: str


class SimSessionResponse(StrictModel):
    session: SimSessionView | None
