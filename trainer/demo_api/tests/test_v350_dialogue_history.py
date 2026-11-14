"""v3.5.0 대화 기록: 답 출처·subject 저장, 모델이 만든 이전 답 빼기."""
import sqlite3

from trainer.demo_api.dialogue import conversation_messages, without_generated_answers
from trainer.demo_api.sequence_store import SequenceStore
from trainer.demo_api.tests.viewer_record import ask


def test_recent_turns_return_source_and_subject() -> None:
    store = SequenceStore(":memory:")
    subject = {"domain": "coating", "zone_ids": ["8"], "operator": "argmin"}
    store.record_query_turn(session_key="S", user_id="U", prompt="어디가 제일 얇아?", answer="Zone 8", task_type="knowledge_search",
                            created_at="t", answer_source="analysis", subject=subject)
    turn = store.recent_query_turns("S", user_id="U")[-1]
    assert turn["answer_source"] == "analysis" and turn["subject"] == subject


def test_old_rows_without_a_source_count_as_generated(tmp_path) -> None:
    path = tmp_path / "old.db"
    with sqlite3.connect(path) as conn:  # v3.4.2 모양(conversation_key까지)
        conn.execute("CREATE TABLE query_turns (id INTEGER PRIMARY KEY AUTOINCREMENT, session_key TEXT NOT NULL, user_id TEXT NOT NULL,"
                     " prompt TEXT NOT NULL, answer TEXT NOT NULL, task_type TEXT NOT NULL, created_at TEXT NOT NULL, conversation_key TEXT)")
        conn.execute("INSERT INTO query_turns (session_key,user_id,prompt,answer,task_type,created_at) VALUES ('S','U','q','9번 구역입니다','k','t')")
    turn = SequenceStore(str(path)).recent_query_turns("S", user_id="U")[-1]
    # 값이 없으면 키를 싣지 않는다(계획 모델이 턴 전체를 JSON으로 받는다).
    assert turn.get("answer_source") is None and turn.get("subject") is None
    assert without_generated_answers([turn])[0]["answer"] == ""


def test_generated_answers_are_removed_but_analysis_answers_stay() -> None:
    turns = [{"prompt": "a", "answer": "문서 답", "answer_source": "documents"},
             {"prompt": "b", "answer": "대화 답", "answer_source": "conversation"},
             {"prompt": "c", "answer": "Zone 8 77.2µm", "answer_source": "analysis"}]
    assert [t["answer"] for t in without_generated_answers(turns)] == ["", "", "Zone 8 77.2µm"]
    messages = conversation_messages("d", recent_turns=without_generated_answers(turns), facts={}, response_mode="quick")
    assert [m["role"] for m in messages[1:]] == ["user", "user", "user", "assistant", "user"]


def test_source_is_stored_and_subject_header_is_not_forwarded() -> None:
    store = SequenceStore(":memory:")
    data = ask("다음엔 어디 칠해야 돼?", tab="order", store=store)
    assert data["source"] == "analysis"
    turn = store.recent_query_turns(data["session_id"], user_id="COPILOT-AI-TEST")[-1]
    assert turn["answer_source"] == "analysis"
    assert "x-copilot-subject" not in data["headers"]
