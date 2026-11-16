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


def test_query_context_is_stored_as_complete_json_and_old_rows_remain_unknown(tmp_path) -> None:
    import json
    store = SequenceStore(str(tmp_path / 'contexts.db'))
    context = {'scenario_id': '3', 'context': '현재 Zone 3', 'pose_context': {'similarity_score': .9}, 'context_truncated': False}
    receipt = store.record_query_turn(session_key='S', user_id='U', prompt='지금 자세는?', answer='90%입니다.',
                                     task_type='pose_recomm', created_at='t', query_context=context)
    row = store._conn.execute('SELECT query_context FROM query_turns WHERE id=?', (receipt,)).fetchone()
    assert json.loads(row[0]) == context
    receipt = store.record_query_turn(session_key='S', user_id='U', prompt='q', answer='a', task_type='knowledge_search', created_at='t')
    assert store._conn.execute('SELECT query_context FROM query_turns WHERE id=?', (receipt,)).fetchone()[0] is None


def test_oversized_query_context_is_marked_instead_of_becoming_partial_training_facts() -> None:
    import json
    store = SequenceStore(':memory:')
    receipt = store.record_query_turn(session_key='S', user_id='U', prompt='q', answer='a', task_type='knowledge_search', created_at='t',
                                     query_context={'context': '가' * 131072, 'context_truncated': False})
    raw = store._conn.execute('SELECT query_context FROM query_turns WHERE id=?', (receipt,)).fetchone()[0]
    assert len(raw.encode('utf-8')) <= 131072
    assert json.loads(raw)['context_truncated'] is True
    assert 'context' not in json.loads(raw)


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


def test_subject_header_is_persisted_but_never_reaches_the_client(monkeypatch) -> None:
    from trainer.demo_api import copilot_router
    real = copilot_router._success_response
    monkeypatch.setattr(copilot_router, "_success_response",
                        lambda *a, **k: real(*a, **{**k, "subject": {"domain": "order", "zone_ids": ["8"], "operator": "progress_next"}}))
    store = SequenceStore(":memory:")
    data = ask("다음엔 어디 칠해야 돼?", tab="order", store=store)
    assert copilot_router.SUBJECT_HEADER.lower() not in data["headers"]
    turn = store.recent_query_turns(data["session_id"], user_id="COPILOT-AI-TEST")[-1]
    assert turn["subject"] == {"domain": "order", "zone_ids": ["8"], "operator": "progress_next"}
