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
SUMMARY_DESCRIPTION = (
    "`response_mode`와 별도로 답변 분량을 지정합니다. 생략하거나 `true`를 보내면 VR용 핵심 요약을, "
    "`false`를 보내면 필요한 설명·문단·목록을 포함한 전체 답변을 반환합니다.\n\n"
    "요약에도 안전 경고, 판정 한계, 확인된 수치·단위·Zone 순서와 필요한 조치가 포함됩니다. "
    "문구나 문장 자체를 번역해 달라는 요청에는 전체 번역을 제공합니다."
)


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
                "summarize": True,
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
            "경로, 자세, 이상 분석, 일반 지식 중 필요한 작업 유형을 지정합니다.\n\n"
            "현재 도막 상태는 확인할 수 있는 훈련 자료를 바탕으로 답합니다. "
            "도막이 생기는 원인이나 작업 방법을 묻는 질문에는 관련 지식을 설명합니다."
        ),
        examples=["knowledge_search"],
    )]
    response_mode: Annotated[ResponseMode, Field(
        description=(
            "답변을 생성할 모델을 선택합니다. 기본값은 빠른 안내(`quick`)입니다.\n\n"
            "- `quick`: **빠른 안내 · Qwen3-8B 모델**\n"
            "- `detailed`: **상세 분석 · Qwen3.6-27B 모델**\n\n"
            "답변 분량은 `summarize`로 지정합니다. 두 모드 모두 최대 1,024토큰을 생성하며 "
            "인용·안전·수치를 검사합니다. 서버가 계산한 자세·순서 평가값은 그대로 사용합니다.\n\n"
            "번역 모델은 별도로 선택됩니다. 일반 지원 언어는 Qwen3-8B를 사용합니다. "
            "네팔어·미얀마어·싱할라어·크메르어는 영어를 거쳐 Qwen3.6-27B로 번역합니다. "
            "번역 시간은 언어와 답변 분량에 따라 달라집니다."
        ),
        examples=["quick"],
    )] = "quick"
    summarize: Annotated[bool, Field(description=SUMMARY_DESCRIPTION, examples=[True, False])] = True
    language: Annotated[QueryLanguage, Field(
        description="질문 언어를 지정합니다. 기본값은 한국어(`ko`)입니다.\n\n지원 언어 — " + LANGUAGE_DESCRIPTION,
        examples=["ko", "vi"],
    )] = "ko"
    response_language: Annotated[QueryLanguage | None, Field(
        description=("답변을 받을 언어입니다. 생략하거나 `null`을 보내면 질문 언어(`language`)로 답합니다.\n\n"
                     "지원 언어 — " + LANGUAGE_DESCRIPTION),
        examples=["vi", "en"],
    )] = None
    use_rag: Annotated[RagUsage, Field(
        description=(
            "관련 문서 검색(RAG)을 허용하려면 `Y`를, 사용하지 않으려면 `N`을 보냅니다. "
            "기본값은 `Y`이며 소문자도 허용합니다.\n\n"
            "답변에 실제로 문서 근거를 사용했는지는 `retrieval_used`에서 확인할 수 있습니다."
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
    degraded: Annotated[bool, Field(description=(
        "답변 생성·형식·의미 검토를 충분히 완료하지 못했거나 근거가 부족해 답변을 보류한 경우 true입니다. "
        "true이면 answer에 표시된 한계를 확인하고, 해당 답변을 확정된 분석 결과로 사용하지 않습니다."
    ))]
    answer: str
    language: QueryLanguage = "ko"
    response_language: QueryLanguage = "ko"
    resolved_intents: list[str]
    context_status: ContextStatus
    sequence_context: dict[str, object] | None
    coating_context: dict[str, object] | None
    pose_context: dict[str, object] | None
    retrieval_used: Annotated[bool, Field(description=(
        "질문에 적합한 NCS 등 등록 문서 근거가 실제 답변을 뒷받침한 경우에만 true입니다. "
        "검색 시도 여부가 아닙니다. 문서 근거를 쓰지 않은 일반 LLM·문맥 답변과 직접 용어집 번역은 false입니다."
    ))]
    citations: Annotated[list[str], Field(description=(
        "실제 답변에 사용한 문서 근거의 식별자 목록입니다. 검색 후보 전체를 뜻하지 않습니다. "
        "문서 근거를 쓰지 않은 일반 LLM·문맥 답변과 직접 용어집 번역은 빈 배열이며, 용어집 출처를 NCS 인용으로 섞지 않습니다."
    ))]


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
