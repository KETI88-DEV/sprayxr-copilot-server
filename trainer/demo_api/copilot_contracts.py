"""v2.5 copilot query contract (도장 SIM 회의자료 v2.5 p.16, 18-22).

Request fields follow p.16; the response is the common wrapper of p.18 with the
query body of p.19-22. Where ``docs/ai_copilot_response_contract.md`` differs
(``use_rag`` boolean there), v2.5 wins: ``use_rag`` is the string ``'Y'``/``'N'``.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from trainer.demo_api.contracts import StrictModel

TaskType = Literal["trajectory_recomm", "pose_recomm", "error_analysis", "knowledge_search"]
ResponseMode = Literal["quick", "detailed"]
ContextStatus = Literal["available", "not_available"]


class CopilotQueryRequest(StrictModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "user_id": "1234",
                "scenario_id": "3",
                "session_id": "3_11000346_1",
                "request_id": "REQ-TEXT-001",
                "prompt": "다음 Zone을 어떤 순서로 칠해야 하나요?",
                "task_type": "knowledge_search",
                "response_mode": "quick",
                "context": "현재 Zone은 3입니다.",
            }
        },
    )

    user_id: Annotated[str, Field(
        min_length=1,
        description="질문한 훈련자의 고유 식별자입니다. 세션 시작 시 사용한 user_id와 동일하게 보냅니다.",
        examples=["1234"],
    )]
    scenario_id: Annotated[str | None, Field(
        description="질문이 속한 콘텐츠 식별자입니다. 현재 훈련 맥락이 필요할 때 세션의 scenario_id를 보냅니다.",
        examples=["3"],
    )] = None
    session_id: Annotated[str | None, Field(
        description="질문이 속한 훈련 세션 식별자입니다. 세션 데이터·자세·순서 문맥을 사용할 때 입력합니다.",
        examples=["3_11000346_1"],
    )] = None
    # request_id·use_rag 는 선택이다. use_rag 생략 시 서버가 처리 경로에 맞춰 자동 결정한다.
    request_id: Annotated[str | None, Field(
        description="클라이언트가 생성하는 호출 추적용 식별자입니다. 생략하면 서버가 응답 추적값을 별도로 처리합니다.",
        examples=["REQ-TEXT-001"],
    )] = None
    prompt: Annotated[str, Field(
        min_length=1,
        max_length=4000,
        description="사용자가 입력하거나 음성에서 변환된 질문 문장입니다. 공백만 있는 문자열은 허용하지 않습니다.",
        examples=["다음 Zone을 어떤 순서로 칠해야 하나요?"],
    )]
    task_type: Annotated[TaskType, Field(
        description=(
            "질문 처리 유형입니다. trajectory_recomm은 경로, pose_recomm은 자세, "
            "error_analysis는 오류 분석, knowledge_search는 일반 지식 검색에 사용합니다."
        ),
        examples=["knowledge_search"],
    )]
    response_mode: Annotated[ResponseMode, Field(
        description=(
            "답변 수준입니다. quick은 훈련 중 즉시 확인할 핵심 판단·조치를 짧게 답하고, "
            "detailed는 현재 상황·근거·원인·권장 조치를 순서대로 자세히 설명합니다. "
            "생략하면 기존 연동과 같은 quick으로 처리합니다."
        ),
        examples=["quick"],
    )] = "quick"
    use_rag: Annotated[Literal["Y", "N"] | None, Field(
        description=(
            "지식 검색(RAG) 강제 설정입니다. 생략하면 서버가 질문 유형과 처리 경로에 따라 자동 결정합니다. "
            "Y는 검색을 강제로 사용하고 N은 검색 없이 현재 문맥만 사용합니다."
        ),
        examples=["Y"],
    )] = None
    context: Annotated[str | None, Field(
        description="현재 Zone·시퀀스 ID·관찰 상황 등 질문에 추가할 짧은 문맥입니다. 불필요하면 생략합니다.",
        examples=["현재 Zone은 3입니다."],
    )] = None

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
