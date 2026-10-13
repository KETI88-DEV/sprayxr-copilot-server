"""Public response contract for the multipart voice Co-Pilot query."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import ConfigDict, Field, model_validator

from trainer.demo_api.contracts import StrictModel
from trainer.demo_api.copilot_contracts import CopilotQueryData
from trainer.demo_api.speech_to_text import VoiceLanguage


class VoiceQueryData(CopilotQueryData):
    transcript: str
    language: VoiceLanguage
    stt_latency_ms: Annotated[int, Field(ge=0)]


class VoiceCopilotResponse(StrictModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "status": "SUCCESS",
                "code": 200,
                "message": "Copilot query completed",
                "data": {
                    "request_id": "REQ-VOICE-001",
                    "session_id": "3_11000346_1",
                    "user_id": "1234",
                    "scenario_id": "3",
                    "generated_at": "2026-10-02T10:00:00Z",
                    "latency_ms": 1850,
                    "degraded": False,
                    "answer": "현재 패스를 마친 뒤 오른쪽 인접 Zone으로 이동하세요.",
                    "resolved_intents": ["knowledge_search"],
                    "context_status": "available",
                    "sequence_context": None,
                    "coating_context": None,
                    "pose_context": None,
                    "retrieval_used": True,
                    "citations": ["DOC-doc_0001-C0001"],
                    "transcript": "다음에는 어느 구역을 칠해야 해?",
                    "language": "ko",
                    "stt_latency_ms": 420,
                },
            }
        },
    )

    status: Literal["SUCCESS", "ERROR"]
    code: int
    message: str
    data: VoiceQueryData | None

    @model_validator(mode="after")
    def data_matches_status(self) -> "VoiceCopilotResponse":
        if (self.status == "SUCCESS") != (self.data is not None):
            raise ValueError("SUCCESS carries data; ERROR carries data=null")
        return self
