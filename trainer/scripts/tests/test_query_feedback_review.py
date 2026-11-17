"""An update review bundle reads existing feedback without migrating or modifying it."""
from __future__ import annotations

import hashlib
import importlib
import json
import sqlite3

import pytest


def _api():
    try:
        return importlib.import_module("trainer.scripts.query_feedback_review")
    except ModuleNotFoundError as error:
        if error.name == "trainer.scripts.query_feedback_review":
            pytest.fail("The read-only update review bundle is not implemented yet")
        raise


def _database(path):
    with sqlite3.connect(path) as conn:
        conn.execute("""CREATE TABLE query_answer_feedback (
            feedback_id TEXT, query_turn_id INTEGER, question TEXT, answer TEXT,
            rating TEXT, comment TEXT, suggested_answer TEXT, corrected_answer TEXT,
            review_status TEXT, reviewed_by TEXT, review_note TEXT, query_context TEXT,
            origin_server TEXT, caller_source TEXT, created_at TEXT, updated_at TEXT,
            session_key TEXT, user_id TEXT)""")
        for ident, status in [("pending-1", "pending"), ("approved-1", "approved"), ("rejected-1", "rejected")]:
            conn.execute("INSERT INTO query_answer_feedback VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                ident, 42, "노즐을 영어로?", "wrong original", "down", "개선 필요", "user suggestion",
                "reviewed correction" if status == "approved" else "", status,
                "operator" if status == "approved" else None, "review note", '{"answer_source":"rag"}',
                "main", "query-chat", "2026-11-17T00:00:00Z", "2026-11-17T01:00:00Z", "CHAT:one", "user-one"))
        conn.execute("""CREATE TABLE user_feedback (
            submission_id TEXT, message TEXT, kind TEXT, page TEXT, status TEXT,
            answer_context TEXT, resolution TEXT, origin_server TEXT, caller_source TEXT,
            created_at TEXT, updated_at TEXT, server_version TEXT)""")
        for ident, status in [("new-1", "new"), ("reviewing-1", "reviewing"), ("done-1", "done")]:
            context = json.dumps({"question": "왜 얇아?", "answer": "original answer",
                                  "correct_answer": "operator suggestion", "answer_source": "session"}, ensure_ascii=False)
            conn.execute("INSERT INTO user_feedback VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", (
                ident, "please review", "bug", "motion", status, context, "release review note",
                "main", "motion-viewer", "2026-11-17T00:00:00Z", "2026-11-17T01:00:00Z", "3.8.12"))


def test_review_bundle_preserves_questions_corrections_ids_status_and_provenance(tmp_path):
    path = tmp_path / "feedback.db"
    _database(path)
    bundle = _api().build_review_bundle(path)
    assert bundle["purpose"] == "update_review_not_training_data"
    assert bundle["automatic_training"] is False
    rows = {row["feedback_id"]: row for row in bundle["items"]}
    assert set(rows) == {"pending-1", "approved-1", "new-1", "reviewing-1"}
    pending = rows["pending-1"]
    assert pending["kind"] == "query_rating" and pending["status"] == "pending"
    assert pending["question"] == "노즐을 영어로?" and pending["answer"] == "wrong original"
    assert pending["suggested_answer"] == "user suggestion" and pending["corrected_answer"] == ""
    assert pending["provenance"]["query_turn_id"] == 42
    assert pending["provenance"]["origin_server"] == "main"
    assert rows["approved-1"]["corrected_answer"] == "reviewed correction"
    assert rows["new-1"]["question"] == "왜 얇아?"
    assert rows["new-1"]["suggested_answer"] == "operator suggestion"
    assert rows["new-1"]["provenance"]["answer_source"] == "session"


def test_review_bundle_never_changes_the_database_or_review_state(tmp_path):
    path = tmp_path / "feedback.db"
    _database(path)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    _api().build_review_bundle(path, include_closed=True)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    with sqlite3.connect(path) as conn:
        assert dict(conn.execute("SELECT feedback_id,review_status FROM query_answer_feedback")) == {
            "pending-1": "pending", "approved-1": "approved", "rejected-1": "rejected"}


def test_closed_entries_are_included_only_when_explicitly_requested(tmp_path):
    path = tmp_path / "feedback.db"
    _database(path)
    bundle = _api().build_review_bundle(path, include_closed=True)
    assert {row["feedback_id"] for row in bundle["items"]} == {
        "pending-1", "approved-1", "rejected-1", "new-1", "reviewing-1", "done-1"}


def test_missing_database_is_not_created_or_reported_as_an_empty_success(tmp_path):
    path = tmp_path / "missing.db"
    with pytest.raises(FileNotFoundError):
        _api().build_review_bundle(path)
    assert not path.exists()


def test_absent_legacy_tables_are_reported_without_creating_schema(tmp_path):
    path = tmp_path / "old.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE unrelated(value TEXT)")
    bundle = _api().build_review_bundle(path)
    assert bundle["items"] == []
    assert set(bundle["missing_tables"]) == {"query_answer_feedback", "user_feedback"}
    with sqlite3.connect(path) as conn:
        assert [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")] == ["unrelated"]


def test_database_uri_characters_do_not_change_the_read_only_target(tmp_path):
    path = tmp_path / "feedback?#.db"
    _database(path)
    bundle = _api().build_review_bundle(path)
    assert len(bundle["items"]) == 4


def test_invalid_saved_context_stays_reviewable_without_becoming_a_training_example(tmp_path):
    path = tmp_path / "feedback.db"
    _database(path)
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE user_feedback SET answer_context='not JSON' WHERE submission_id='new-1'")
    bundle = _api().build_review_bundle(path)
    row = next(row for row in bundle["items"] if row["feedback_id"] == "new-1")
    assert row["context_parse_error"] is True
    assert row["question"] is None and row["answer"] is None
    assert row["message"] == "please review"
