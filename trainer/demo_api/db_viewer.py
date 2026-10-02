"""Small read-only SQLite viewer retained at the legacy /admin link."""

from __future__ import annotations

import html
import os
import sqlite3
from pathlib import Path

from fastapi import APIRouter, Query
from fastapi.responses import HTMLResponse

from trainer.demo_api.sequence_store import DEFAULT_DB_PATH


def _page(body: str) -> HTMLResponse:
    return HTMLResponse(
        "<!doctype html><html lang='ko'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Co-Pilot DB Viewer</title><style>body{font-family:sans-serif;margin:2rem}"
        "table{border-collapse:collapse;width:100%;font-size:13px}th,td{border:1px solid #ccc;"
        "padding:.35rem;text-align:left;vertical-align:top}nav a{margin-right:1rem}"
        "td{max-width:34rem;overflow-wrap:anywhere}</style></head><body>" + body + "</body></html>"
    )


def create_db_viewer_router() -> APIRouter:
    router = APIRouter()

    @router.get("/admin", include_in_schema=False, response_class=HTMLResponse)
    def database_viewer(
        table: str | None = None,
        limit: int = Query(default=100, ge=1, le=200),
    ) -> HTMLResponse:
        path = Path(os.environ.get("SEQUENCE_DB_PATH", DEFAULT_DB_PATH))
        if not path.is_file():
            return _page("<h1>Co-Pilot DB Viewer</h1><p>DB 파일이 아직 생성되지 않았습니다.</p>")
        try:
            conn = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
            conn.row_factory = sqlite3.Row
            tables = [
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
                )
            ]
            selected = table if table in tables else (tables[0] if tables else None)
            links = "".join(
                f"<a href='/admin?table={html.escape(name)}'>{html.escape(name)}</a>" for name in tables
            )
            if selected is None:
                return _page("<h1>Co-Pilot DB Viewer</h1><p>표시할 테이블이 없습니다.</p>")
            quoted = '"' + selected.replace('"', '""') + '"'
            rows = conn.execute(f"SELECT * FROM {quoted} ORDER BY rowid DESC LIMIT ?", (limit,)).fetchall()
            columns = [item[1] for item in conn.execute(f"PRAGMA table_info({quoted})")]
            heading = "".join(f"<th>{html.escape(str(column))}</th>" for column in columns)
            rendered_rows = "".join(
                "<tr>" + "".join(f"<td>{html.escape(str(row[column]))}</td>" for column in columns) + "</tr>"
                for row in rows
            )
            return _page(
                f"<h1>Co-Pilot DB Viewer (읽기 전용)</h1><nav>{links}</nav>"
                f"<h2>{html.escape(selected)} · 최근 {limit}건</h2>"
                f"<table><thead><tr>{heading}</tr></thead><tbody>{rendered_rows}</tbody></table>"
            )
        except sqlite3.Error:
            return _page("<h1>Co-Pilot DB Viewer</h1><p>DB를 읽을 수 없습니다.</p>")
        finally:
            if "conn" in locals():
                conn.close()

    return router
