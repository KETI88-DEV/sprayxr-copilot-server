from __future__ import annotations

from typing import Annotated, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")


ZoneStatus = Literal["current", "next", "completed", "pending"]


class SequenceZone(StrictModel):
    zone_id: Annotated[str, Field(min_length=1)]
    label: Annotated[str, Field(min_length=1)]
    status: ZoneStatus = "pending"


class SequenceSnapshot(StrictModel):
    level_id: Annotated[str, Field(min_length=1)]
    label: Annotated[str, Field(min_length=1)]
    zones: list[SequenceZone]
    current_zone_id: str
    next_zone_ids: list[str]
    completed_zone_ids: list[str]

    @model_validator(mode="after")
    def zones_reference_each_other(self) -> "SequenceSnapshot":
        ids = {zone.zone_id for zone in self.zones}
        if self.current_zone_id not in ids:
            raise ValueError("current_zone_id must exist in zones")
        for group in (self.next_zone_ids, self.completed_zone_ids):
            for zone_id in group:
                if zone_id not in ids:
                    raise ValueError(f"unknown zone_id: {zone_id}")
        return self


class SequenceUpdate(StrictModel):
    type: Literal["update_sequence"] = "update_sequence"
    level_id: str
    current_zone_id: str
    next_zone_ids: list[str]
    completed_zone_ids: list[str]

    @model_validator(mode="after")
    def zones_are_unique(self) -> "SequenceUpdate":
        if len(self.next_zone_ids) != len(set(self.next_zone_ids)):
            raise ValueError("next_zone_ids must be unique")
        return self


class ChatRequest(StrictModel):
    message: Annotated[str, Field(min_length=1, max_length=4000)]
    session_id: Annotated[str, Field(min_length=1)]
    mode: Literal["demo", "live"] = "demo"
    level_id: Annotated[str, Field(min_length=1)] = "TR-L01"


class StreamEvent(StrictModel):
    event: Literal[
        "session", "status", "message_delta", "tool_result",
        "sequence_update", "complete", "error",
    ]
    data: dict[str, object]
