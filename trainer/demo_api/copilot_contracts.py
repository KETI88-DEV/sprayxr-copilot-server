"""v2.5 copilot query contract (도장 SIM 회의자료 v2.5 p.16, 18-22).

Request fields follow p.16; the response is the common wrapper of p.18 with the
query body of p.19-22. Where ``docs/ai_copilot_response_contract.md`` differs
(``use_rag`` boolean there), the public API uses ``'Y'``/``'N'`` strings.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BeforeValidator, ConfigDict, Field, field_validator, model_validator

from trainer.demo_api.contracts import StrictModel
from trainer.demo_api.query_language import LANGUAGE_DESCRIPTION, QueryLanguage

TaskType = Literal["trajectory_recomm", "pose_recomm", "error_analysis", "knowledge_search"]
ResponseMode = Literal["quick", "detailed"]
ContextStatus = Literal["available", "not_available"]


def _normalize_rag_usage(value: object) -> object:
    return value.upper() if isinstance(value, str) else value


RagUsage = Annotated[Literal["Y", "N"], BeforeValidator(_normalize_rag_usage)]


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
                "language": "ko",
                "response_language": "ko",
                "use_rag": "Y",
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
    # request_id·use_rag 는 선택이다. use_rag 생략 시 Y로 처리한다.
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
            "error_analysis는 오류 분석, knowledge_search는 일반 지식 검색에 사용합니다. 도막의 실제 상태 질문은 서버가 확인할 수 있는 도막 자료로 답하며, 원인·방법 질문은 지식 설명으로 처리합니다."
        ),
        examples=["knowledge_search"],
    )]
    response_mode: Annotated[ResponseMode, Field(
        description=(
            "답변 수준입니다. 생략하면 `quick`(빠른 안내)으로 처리합니다.\n\n"
            "- `quick`: **빠른 안내 · Qwen3-8B 모델**\n"
            "  - 질문의 핵심을 한 문장으로 답하며, 필수 안전 조건이 있을 때만 짧은 두 번째 문장을 덧붙입니다.\n"
            "- `detailed`: **상세 분석 · Qwen3.6-27B 모델**\n"
            "  - 기존 27B 빠른 답변 수준으로 핵심 답변·근거·조치를 간결하게 분석합니다.\n"
            "  - 전체 최대 2문장으로 설명하며 인사말·반복 요약·불필요한 배경은 생략합니다.\n\n"
            "서버에서 확인한 자세·순서 평가 결과는 그대로 사용합니다. "
            "답변 번역은 분석 모델과 별도로 선택합니다. 일반 지원 언어는 Qwen3-8B, "
            "네팔어·미얀마어·싱할라어·크메르어는 영어 중간 번역 후 Qwen3.6-27B 번역 경로를 사용합니다. "
            "언어와 답변 분량에 따라 번역 시간이 추가됩니다."
        ),
        examples=["quick"],
    )] = "quick"
    language: Annotated[QueryLanguage, Field(
        description="질문 언어 코드입니다. 생략 시 ko(한국어)입니다. " + LANGUAGE_DESCRIPTION,
        examples=["ko", "vi"],
    )] = "ko"
    response_language: Annotated[QueryLanguage | None, Field(
        description=("답변 언어 코드입니다. 생략하거나 null이면 language와 같은 언어로 답합니다. "
                     "질문과 다른 언어를 지정할 수도 있습니다. " + LANGUAGE_DESCRIPTION),
        examples=["vi", "en"],
    )] = None
    use_rag: Annotated[RagUsage, Field(
        description=(
            "지식 검색(RAG) 사용 여부입니다. Y는 관련 지식 검색을 사용하고 N은 검색 없이 현재 문맥만 사용합니다. "
            "Y/N만 허용하며 소문자 y/n도 입력할 수 있습니다. 생략하면 Y로 처리합니다."
        ),
        examples=["Y"],
    )] = "Y"
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
    language: QueryLanguage = "ko"
    response_language: QueryLanguage = "ko"
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
