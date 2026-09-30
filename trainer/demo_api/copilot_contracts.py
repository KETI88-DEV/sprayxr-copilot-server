"""v2.5 copilot query contract (도장 SIM 회의자료 v2.5 p.16, 18-22).

Request fields follow p.16; the response is the common wrapper of p.18 with the
query body of p.19-22. Where ``docs/ai_copilot_response_contract.md`` differs
(``use_rag`` boolean there), v2.5 wins: ``use_rag`` is the string ``'Y'``/``'N'``.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from trainer.demo_api.contracts import StrictModel

TaskType = Literal["trajectory_recomm", "pose_recomm", "error_analysis", "knowledge_search"]
ContextStatus = Literal["available", "not_available"]


class CopilotQueryRequest(StrictModel):
    user_id: Annotated[str, Field(min_length=1)]
    scenario_id: str | None = None
    session_id: str | None = None
    # 명세서 p.9: request_id·use_rag 는 (선택)이고 use_rag 기본값은 Y 다.
    request_id: str | None = None
    prompt: Annotated[str, Field(min_length=1, max_length=4000)]
    task_type: TaskType
    use_rag: Literal["Y", "N"] = "Y"
    context: str | None = None

    @field_validator("prompt")
    @classmethod
    def prompt_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt must not be blank")
        return value


class CopilotQueryData(StrictModel):
    # 요청의 request_id 가 (선택)이므로 에코도 널이 될 수 있다.
    request_id: str | None
    session_id: str | None
    user_id: str
    scenario_id: str | None
    generated_at: str
    latency_ms: Annotated[int, Field(ge=0)]
    # v2.5 §9.2 예시와 §9.3이 요구하는 필드. 정상 완료는 false, 제한 시간 내 생성이
    # 끝나지 않아 축약 응답을 낼 때 true다.
    degraded: bool
    answer: str
    resolved_intents: list[str]
    context_status: ContextStatus
    sequence_context: dict[str, object] | None
    coating_context: dict[str, object] | None
    pose_context: dict[str, object] | None
    retrieval_used: bool
    citations: list[str]


class CopilotResponse(StrictModel):
    status: Literal["SUCCESS", "ERROR"]
    code: int
    message: str
    data: CopilotQueryData | None

    @model_validator(mode="after")
    def data_matches_status(self) -> "CopilotResponse":
        if (self.status == "SUCCESS") != (self.data is not None):
            raise ValueError("SUCCESS carries data; ERROR carries data=null")
        return self
