"""v3.5.1 시뮬레이터 기본 설정 블록: 모델이 쓰는 도장·기록 답에만 싣는다. 모델에게 알려 줄 뿐 수치 근거로 쓰지 않고, 유출 지문에도 넣지 않는다."""
from __future__ import annotations

import pytest

from trainer.demo_api import dialogue
from trainer.demo_api.copilot_session_context import build_query_instruction
from trainer.demo_api.simulator_context import SIMULATOR_CONTEXT
from trainer.demo_api.tests.viewer_record import AT_35S, RecordingPipeline, ask

MARK = "[시뮬레이터 기본 설정]"
CLARIFY = {"kind": "clarify", "intent": "knowledge", "scope": "unspecified"}
FACT_ANSWER = "목표 도막은 1차 160µm입니다. 압력은 108~504 bar 범위에서 고릅니다. 신너 희석률은 0~24%입니다."


def test_block_has_the_agreed_facts_and_no_unconfirmed_gun_type() -> None:
    assert SIMULATOR_CONTEXT.startswith(MARK)
    for fact in ("160µm", "320µm", "108~504 bar", "36 bar", "0~24%", "1.6%", "519", "323", "90/10", "mm/s"):
        assert fact in SIMULATOR_CONTEXT
    assert "에어리스" not in SIMULATOR_CONTEXT


def test_instruction_carries_the_block_only_when_given() -> None:
    kwargs = dict(client_context=None, session_summary="", recent_turns=[])
    assert MARK not in build_query_instruction("질문", **kwargs)
    for concise in (False, True):
        text = build_query_instruction("질문", simulator_context=SIMULATOR_CONTEXT, concise=concise, **kwargs)
        assert MARK in text
    assert build_query_instruction("질문", simulator_context=SIMULATOR_CONTEXT, concise=True, **kwargs).endswith("질문")


@pytest.mark.parametrize("mode", ["quick", "detailed"])
def test_documents_route_instruction_has_the_block(mode: str) -> None:
    pipeline = RecordingPipeline(document="[답변] 문서 답입니다.")
    ask("저도막의 원인은?", tab="knowledge", pipeline=pipeline, raw_context="", mode=mode)
    run = dict(pipeline.kwargs)["run"]
    assert MARK in run["instruction"]
    if mode == "quick":
        assert MARK in run["generation_instruction"]


def test_conversation_route_gets_the_block_for_domain_questions_only() -> None:
    domain = RecordingPipeline(plan=CLARIFY)
    ask("VR 어지러우면 몇 분 쉬어?", tab="knowledge", pipeline=domain, raw_context="")
    assert dict(domain.kwargs)["converse"].get("simulator_context") == SIMULATOR_CONTEXT
    off = RecordingPipeline(plan=CLARIFY)
    ask("안녕 오늘 점심 뭐 먹지?", tab="knowledge", pipeline=off, raw_context="")
    assert "simulator_context" not in dict(off.kwargs)["converse"]


def test_conversation_messages_append_the_block_after_the_fingerprinted_instruction() -> None:
    plain = dialogue.conversation_messages("q", recent_turns=[], facts={}, response_mode="quick")[0]["content"]
    withblock = dialogue.conversation_messages("q", recent_turns=[], facts={}, response_mode="quick",
                                               simulator_context=SIMULATOR_CONTEXT)[0]["content"]
    assert MARK not in plain and withblock.startswith(plain) and withblock.endswith(SIMULATOR_CONTEXT)


def test_simulator_facts_are_not_instruction_leakage_but_real_instructions_still_are() -> None:
    assert not dialogue.leaks_instructions(FACT_ANSWER)
    assert not dialogue.leaks_instructions(SIMULATOR_CONTEXT.replace(MARK, "").replace("- ", ""))
    assert dialogue.guard_generated_answer(FACT_ANSWER) == FACT_ANSWER
    real = dialogue.conversation_messages("", recent_turns=[], facts={}, response_mode="quick")[0]["content"].split("\n현재 날짜")[0]
    assert dialogue.leaks_instructions(real[:200])


# --- 블록 수치는 어떤 문장도 받치지 않는다(수치 근거 검사는 v3.5.0 그대로) ---------------------------------
GENERAL = "압력은 108~504 bar 범위에서 36 bar 단위로 고릅니다."
RECORD_CLAIMS = ["이 기록은 압력을 504 bar로 설정했습니다.", "이 기록의 신너 희석률은 1.6%입니다.", "이 기록의 속도는 기준보다 24% 느립니다."]


@pytest.mark.parametrize("claim", RECORD_CLAIMS)
def test_record_sentences_are_not_grounded_by_the_block_on_the_conversation_route(claim: str) -> None:
    pipeline = RecordingPipeline(plan=CLARIFY, converse=claim)
    data = ask("VR 어지러우면 몇 분 쉬어?", tab="knowledge", playback=AT_35S, pipeline=pipeline)
    assert pipeline.calls.count("converse") == 2, "한 번 다시 생성한다"
    for number in ("504 bar", "1.6%", "24%"):
        assert number not in data["answer"], data["answer"]


@pytest.mark.parametrize("claim", RECORD_CLAIMS)
def test_record_sentences_are_not_grounded_by_the_block_on_the_documents_route(claim: str) -> None:
    pipeline = RecordingPipeline(document="[답변] " + claim, chunks=[{"chunk_id": "DOC-x", "text": "작업 중 환기를 유지한다.", "rerank_score": 9.0}])
    data = ask("압력은 어떻게 정해?", tab="knowledge", playback=AT_35S, pipeline=pipeline)
    for number in ("504 bar", "1.6%", "24%"):
        assert number not in data["answer"], data["answer"]


def test_blasting_questions_get_no_block_and_are_checked_as_before() -> None:
    pipeline = RecordingPipeline(document="[답변] " + GENERAL)
    ask("블라스팅 노즐 압력은 몇 bar로 해?", tab="knowledge", pipeline=pipeline, raw_context="")
    run = dict(pipeline.kwargs)["run"]
    assert MARK not in run["instruction"]
    chat = RecordingPipeline(plan=CLARIFY, converse=GENERAL)
    data = ask("블라스팅 노즐 압력은 몇 bar로 해?", tab="knowledge", pipeline=chat, raw_context="")
    assert "simulator_context" not in dict(chat.kwargs).get("converse", {})
    assert "108~504" not in data["answer"], data["answer"]


HONORIFIC = ["희석률은 1.6%로 하셨습니다.", "압력을 504 bar로 설정하셨네요."]

EVASIONS = ["희석률은 1.6%로 하셨습니다.", "압력을 504 bar로 설정하셨네요.", "희석률을 1.6% 단위로 맞추셨어요.", "희석률을 1.6% 단위로 올리셨네요.",
            "희석률 1.6% 단위로 넣었어요.", "압력을 36 bar 단위로 두 번 올렸습니다.", "2차 도막은 320µm까지 올라갔습니다.", GENERAL]


@pytest.mark.parametrize("claim", EVASIONS)
def test_block_numbers_ground_nothing_on_either_route(claim: str) -> None:
    pipeline = RecordingPipeline(plan=CLARIFY, converse=claim)
    data = ask("VR 어지러우면 몇 분 쉬어?", tab="knowledge", playback=AT_35S, pipeline=pipeline)
    assert pipeline.calls.count("converse") == 2, "한 번 다시 생성한다"
    for number in ("1.6%", "504 bar", "36 bar", "320µm", "108~504"):
        assert number not in data["answer"], data["answer"]
    docs = RecordingPipeline(document="[답변] " + claim, chunks=[{"chunk_id": "DOC-x", "text": "환기", "rerank_score": 9.0}])
    data = ask("압력은 어떻게 정해?", tab="knowledge", playback=AT_35S, pipeline=docs)
    # 320µm은 서명된 기록의 목표 두께이기도 해서 v3.5.0에서도 문서 경로를 지난다.
    for number in ("1.6%", "504 bar", "36 bar", "108~504"):
        assert number not in data["answer"], data["answer"]
