"""Expose saved sequence coaching in the list without changing the stored result."""

import html
import json
from html.parser import HTMLParser
import sqlite3

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from trainer.demo_api.db_viewer import create_db_viewer_router
from trainer.demo_api.sequence_store import SequenceStore


class GuideCells(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.cells = []
        self._current = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "td" and attrs.get("data-column") == "text_guide":
            self._current = {"text": "", "title": None, "legacy": False}
        if self._current is not None:
            if attrs.get("title") is not None:
                self._current["title"] = attrs["title"]
            if attrs.get("data-guide-format") == "legacy":
                self._current["legacy"] = True

    def handle_data(self, value):
        if self._current is not None:
            self._current["text"] += value

    def handle_endtag(self, tag):
        if tag == "td" and self._current is not None:
            self.cells.append(self._current)
            self._current = None


def _client(monkeypatch, tmp_path, result):
    path = tmp_path / "sequence-guides.db"
    store = SequenceStore(str(path))
    store.save_analysis(session_key="3_GUIDE_TEST_1", sequence_id="SEQ-001", analysis_key="range-test",
                        result={"text_guide": "placeholder"})
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE sequence_analysis_results SET result=?", (result,))
    monkeypatch.setenv("SEQUENCE_DB_PATH", str(path))
    app = FastAPI()
    app.include_router(create_db_viewer_router())
    return TestClient(app), path


def _cells(response):
    parser = GuideCells()
    parser.feed(response.text)
    return parser.cells


def test_sequence_analysis_list_shows_top_level_text_guide_and_preserves_raw_result_and_detail(monkeypatch, tmp_path):
    guide = "Zone 3 도포를 계속하세요.\n현재 구역을 먼저 마무리하세요."
    payload = {"text_guide": guide, "guide": {"speed": {"status": "ok"}}, "coating_feedback": {"analysis_status": "available"}}
    raw = json.dumps(payload, ensure_ascii=False)
    client, path = _client(monkeypatch, tmp_path, raw)
    response = client.get("/admin", params={"table": "sequence_analysis_results"})
    assert response.status_code == 200
    assert "<th>text_guide" in response.text and "<th>result</th>" in response.text
    assert _cells(response)[0]["text"] == guide
    assert _cells(response)[0]["title"] == guide
    assert "JSON 보기" in response.text
    detail = client.get("/admin/api/record", params={"table": "sequence_analysis_results", "rowid": 1}).json()
    assert detail["sections"][0]["data"]["result"] == payload
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT result FROM sequence_analysis_results").fetchone()[0] == raw


@pytest.mark.parametrize("result", ["null", "{}", '{"text_guide": null}', '{"text_guide": " "}',
                                  '{"text_guide": 42}', '["unrelated"]', '{"guide":{"text_guide":"unrelated"}}'])
def test_absent_or_non_text_guides_do_not_display_an_invented_answer(monkeypatch, tmp_path, result):
    client, _ = _client(monkeypatch, tmp_path, result)
    response = client.get("/admin?table=sequence_analysis_results")
    assert _cells(response)[0]["text"] == "안내 없음"
    assert _cells(response)[0]["title"] is None


@pytest.mark.parametrize("encoded", [False, True])
def test_legacy_result_strings_are_visible_and_identified_as_legacy(monkeypatch, tmp_path, encoded):
    guide = "구형 안내: Zone 2에서 시작하세요."
    client, _ = _client(monkeypatch, tmp_path, json.dumps(guide, ensure_ascii=False) if encoded else guide)
    result = _cells(client.get("/admin?table=sequence_analysis_results"))[0]
    assert guide in result["text"] and result["legacy"] is True


def test_long_guides_with_markup_quotes_and_newlines_are_safe_and_readable(monkeypatch, tmp_path):
    guide = '<img src=x onerror="alert(1)">\n\'따옴표\' & "강조" ' + "긴 안내 " * 300
    client, _ = _client(monkeypatch, tmp_path, json.dumps({"text_guide": guide}, ensure_ascii=False))
    response = client.get("/admin?table=sequence_analysis_results")
    cell = _cells(response)[0]
    assert cell["text"] == cell["title"] == guide
    assert '<img src=x onerror="alert(1)">' not in response.text
    assert html.escape(guide, quote=True) in response.text
    assert ".sequence-text-guide" in response.text and "white-space:pre-wrap" in response.text


def test_new_guide_column_is_specific_to_sequence_results_and_empty_lists_have_correct_span(monkeypatch, tmp_path):
    client, _ = _client(monkeypatch, tmp_path, '{"text_guide":"specific"}')
    other = client.get("/admin?table=sequences")
    assert 'data-column="text_guide"' not in other.text
    empty = client.get("/admin", params={"table": "sequence_analysis_results", "q": "no-record"})
    assert 'colspan="8"' in empty.text
    assert "조건에 맞는 데이터가 없습니다." in empty.text
