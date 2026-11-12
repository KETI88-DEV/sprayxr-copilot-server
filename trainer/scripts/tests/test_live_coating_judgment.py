"""내부 채팅(/api/chat) 도막 판정 — 2026-11-12 v3.0.0 카나리 결함 B1·P3.

27B 가 dft_judge 를 부르지 않고 fast 재답이 RAG 문서의 옛 90% 규칙으로 "평균 159, 90% 기준(144) 통과"라고
답했다. 질문에 측정값이 있으면 모델 없이 dft_judge(PR #8 기준) 결과로 답해야 한다.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import anyio
import pytest
import yaml

from mcp_servers.dft_judge import DFTEvaluateRequest, evaluate_dft
from trainer.scripts.agent_pipeline import COATING_TERMS_NOTE, AgentPipeline, SemanticRouter
from trainer.scripts.live_coating_judgment import parse_dft_question
from trainer.scripts.tool_calling_adapter import ToolInfrastructureError

PipelineEvent = tuple[str, dict[str, Any]]
_CONFIGS = Path(__file__).resolve().parents[2] / "configs"

JUDGE = "3번 Zone DFT가 150, 158, 171 µm 인데 목표 160 이면 판정은?"
ANOMALY = "3번 Zone DFT 측정값이 100, 110, 120, 170 µm 이고 목표가 160 µm 인데 이상 탐지해서 원인 후보도 알려줘."
DIAGNOSE = "3번 Zone 도막이 불량이야. DFT 측정값 100, 110, 120, 170 µm, 목표 160 µm. 원인을 단계별로 진단해줘."


@pytest.mark.parametrize(
    ("question", "parsed"),
    [
        (JUDGE, ([150.0, 158.0, 171.0], 160.0)),
        (ANOMALY, ([100.0, 110.0, 120.0, 170.0], 160.0)),
        (DIAGNOSE, ([100.0, 110.0, 120.0, 170.0], 160.0)),
        ("DFT 150µm 인데 합격이야?", ([150.0], None)),
        ("도막 두께 210, 230 이면?", None),  # 단위·측정 표지 없는 목록은 두께로 단정하지 않는다
        # 측정값이 없거나 두께가 아닌 수치는 판정하지 않는다
        ("저도막이 뭐야?", None),
        ("도막 상태 판정해줘", None),
        ("분사 거리 300, 250 mm 인데 도막은 괜찮아?", None),
        ("다음 순서가 어떻게 돼?", None),
        ("DFT 150µm, 기준 160µm 이면?", ([150.0], 160.0)),
        ("도막 두께 측정값 150, 158 이면?", ([150.0, 158.0], None)),
        # 리뷰 지적: 두께가 아닌 수·불완전 추출은 모델로 보낸다
        ("왼팔 각도 30, 45도면 자세 괜찮아? 도막 두께에 영향 있어?", None),
        ("분사 거리 300, 250 인데 도막 두께 괜찮아?", None),
        ("WFT 300µm 이면 DFT는 얼마야?", None),
        ("DFT 160µm 목표로 칠하려면 WFT는 몇 µm?", None),
        ("최소 DFT 120µm 이 규정이야?", None),
        ("도막 두께가 150 µm에서 300 µm 사이면 정상이야?", None),
        ("도막 1, 2회 도장 후 DFT 280µm 이면?", None),
        ("도막 두께 측정 3, 5회차 값", None),
        ("DFT 150 158 171 µm 판정", None),
        ("DFT 150과 158µm 판정", None),
        ("건조도막두께 150 ~ 200 µm 권장인데 맞아?", None),
        ("DFT 170µm, 목표 160~200µm", None),
        ("3번 Zone DFT 1,500µm", None),
    ],
)
def test_parse_dft_question(question: str, parsed: tuple[list[float], float | None] | None) -> None:
    assert parse_dft_question(question) == parsed


class _McpAdapter:
    """실제 dft_judge 라우터 함수로 답하는 어댑터 — 판정 규칙을 시험에 다시 쓰지 않는다."""

    def __init__(self, fail: bool = False, raises: bool = False) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.fail = fail
        self.raises = raises

    async def dispatch_tool_call(self, tool_name: str, arguments: dict[str, Any], session_id: str | None = None) -> dict[str, Any]:
        self.calls.append((tool_name, arguments))
        if self.raises:
            raise ToolInfrastructureError("connection refused")
        if self.fail:
            return {"status": "error", "error": {"code": "SERVER_ERROR", "message": "down"}}
        return evaluate_dft(DFTEvaluateRequest(**arguments)).model_dump()


def _stream(instruction: str, adapter: _McpAdapter | None) -> tuple[str, list[PipelineEvent]]:
    pipe = AgentPipeline.__new__(AgentPipeline)
    pipe.router = SemanticRouter()
    pipe.tool_adapter = adapter

    def _model_must_not_run(**_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("측정값이 있는 도막 질문은 모델을 부르지 않는다")

    pipe.run = _model_must_not_run

    async def _drain(stream: AsyncIterator[PipelineEvent]) -> list[PipelineEvent]:
        return [event async for event in stream]

    events = anyio.run(_drain, pipe.run_streaming(instruction=instruction, environment={}))
    return "".join(p["text"] for kind, p in events if kind == "delta"), events


def test_chat_judges_the_canary_readings_with_dft_judge_as_under_applied() -> None:
    adapter = _McpAdapter()
    answer, events = _stream(JUDGE, adapter)

    assert adapter.calls == [("dft_judge", {"measurements": [150.0, 158.0, 171.0], "spec": {"nominal": 160.0}})]
    assert [p["name"] for kind, p in events if kind == "tool_result"] == ["dft_judge"]
    complete = next(p for kind, p in events if kind == "complete")
    assert complete["tool_call_count"] == 1
    assert "저도막" in answer and "159.7µm" in answer and "목표 160µm" in answer
    assert "90%" not in answer.replace("90~100%", "") and "144" not in answer and "통과" not in answer


@pytest.mark.parametrize("question", [ANOMALY, DIAGNOSE])
def test_cause_questions_get_the_grade_and_no_invented_cause(question: str) -> None:
    answer, _events = _stream(question, _McpAdapter())

    assert "저도막" in answer and "평균은 125µm" in answer
    assert "과도" not in answer.replace("과도막)", "")  # 규칙 설명의 "과도막" 외에는 없다
    assert "<think>" not in answer
    assert "원인을 정할 수 없습니다" in answer


def test_six_band_distribution_is_reported() -> None:
    answer, _events = _stream(JUDGE, _McpAdapter())
    assert "90~100% 2개" in answer and "100~150% 1개" in answer


def test_mcp_failure_still_answers_with_the_same_rule_in_process() -> None:
    answer, events = _stream(JUDGE, _McpAdapter(fail=True))

    assert "저도막" in answer and "159.7µm" in answer
    assert not [p for kind, p in events if kind == "tool_result"]


def test_mcp_unreachable_exception_falls_back_in_process() -> None:
    answer, events = _stream(JUDGE, _McpAdapter(raises=True))

    assert "저도막" in answer and "159.7µm" in answer
    assert not [k for k, _ in events if k == "error"]


def test_missing_target_omits_spec_from_the_tool_call() -> None:
    adapter = _McpAdapter()
    _stream("DFT 150µm 인데 합격이야?", adapter)
    assert adapter.calls == [("dft_judge", {"measurements": [150.0]})]


def test_pose_question_with_numbers_is_not_judged() -> None:
    pipe = AgentPipeline.__new__(AgentPipeline)
    pipe.router = SemanticRouter()
    pipe.tool_adapter = None
    seen: list[str] = []
    pipe.run = lambda **kw: seen.append(kw["instruction"]) or {
        "route": "fast", "rag_chunks": [], "tool_calls": [], "response": "[답변] ok", "synthesis_ms": 0, "deadline_hit": False}

    async def _drain() -> list[PipelineEvent]:
        return [e async for e in pipe.run_streaming(instruction="왼팔 각도 30, 45도면 자세 괜찮아? 도막 두께에 영향 있어?", environment={})]

    anyio.run(_drain)
    assert seen


def test_missing_target_is_judged_against_the_default_and_says_so() -> None:
    answer, _events = _stream("DFT 150µm 인데 합격이야?", _McpAdapter())
    assert "기본 목표 160µm" in answer and "저도막" in answer


def test_unclosed_think_tag_never_reaches_the_chat() -> None:
    """진단형 질문의 fast 재답이 "[답변] <think> 단계 1: … dft_judge로 …"를 그대로 보냈다(카나리 chat_diag)."""
    pipe = AgentPipeline.__new__(AgentPipeline)
    pipe.router = SemanticRouter()
    pipe.tool_adapter = None
    pipe.run = lambda **_kwargs: {
        "route": "fast",
        "rag_chunks": [{"chunk_id": "DOC-doc_0033-C0067", "text": ""}],
        "tool_calls": [],
        "response": "[답변] <think> 단계 1: dft_judge로 기본 상태를 확정한다.\n[근거] DOC-doc_0033-C0067",
        "synthesis_ms": 0,
        "deadline_hit": False,
    }

    async def _drain() -> list[PipelineEvent]:
        return [e async for e in pipe.run_streaming(instruction="블라스팅이 뭐야", environment={})]

    answer = "".join(p["text"] for kind, p in anyio.run(_drain) if kind == "delta")
    assert "<think>" not in answer and "단계 1" not in answer
    assert answer.startswith("[답변]")


@pytest.mark.parametrize("config", ["rag_config_spark_vllm.yaml", "rag_config_fewshot_27b.yaml"])
def test_chat_fast_prompt_defines_coating_terms_like_the_quick_answer(config: str) -> None:
    """P3: /api/chat fast 경로는 config fast_system 을 쓴다. 빠른 지식 답과 같은 정의 문장이어야 한다."""
    prompt = yaml.safe_load((_CONFIGS / config).read_text(encoding="utf-8"))["prompt"]["fast_system"]
    assert COATING_TERMS_NOTE in prompt
