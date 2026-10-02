"""Public response contract for the multipart voice Co-Pilot query."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from trainer.demo_api.contracts import StrictModel
from trainer.demo_api.copilot_contracts import CopilotQueryData
from trainer.demo_api.speech_to_text import VoiceLanguage


class VoiceQueryData(CopilotQueryData):
    transcript: str
    language: VoiceLanguage
    stt_latency_ms: Annotated[int, Field(ge=0)]


class VoiceCopilotResponse(StrictModel):
    status: Literal["SUCCESS", "ERROR"]
    code: int
    message: str
    data: VoiceQueryData | None

    @model_validator(mode="after")
    def data_matches_status(self) -> "VoiceCopilotResponse":
        if (self.status == "SUCCESS") != (self.data is not None):
            raise ValueError("SUCCESS carries data; ERROR carries data=null")
        return self
